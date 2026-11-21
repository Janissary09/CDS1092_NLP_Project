"""Offline tests for the Llama evaluator and shared benchmark protocol."""

from contextlib import closing
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_llama", ROOT / "src/29_evaluate_llama.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)
protocol = evaluator.shared


class LlamaEvaluationTests(unittest.TestCase):
    def test_complete_datasets_and_result_paths(self):
        for dataset, size in (("umwp", 5200), ("sum", 10944)):
            with self.subTest(dataset=dataset):
                rows, digest = protocol.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), size)
                self.assertEqual(digest, protocol.DATASETS[dataset][-1])
                self.assertEqual(evaluator.output_path(ROOT, dataset, "run_01"),
                                 ROOT / f"results/llm/llama-3.3-70b-instruct/{dataset}/run_01")

    def test_strict_parsing(self):
        self.assertEqual(protocol.parse_label(" ANSWERABLE\n"), "ANSWERABLE")
        self.assertEqual(protocol.parse_label("UNANSWERABLE"), "UNANSWERABLE")
        for invalid in ("", "answerable", "UNANSWERABLE.", "[INST]ANSWERABLE", "Label: ANSWERABLE"):
            self.assertEqual(protocol.parse_label(invalid), "INVALID")

    def test_resume_and_export(self):
        data = [{"text": "First?", "label": "ANSWERABLE"}, {"text": "Second?", "label": "UNANSWERABLE"}]
        manifest = {"model_id": evaluator.MODEL_ID, "prompt": protocol.PROMPT}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            checkpoint = path / "checkpoint.sqlite3"
            with closing(protocol.open_checkpoint(checkpoint, manifest)) as connection:
                generate = Mock(side_effect=[("ANSWERABLE", [3], False), RuntimeError("interrupted")])
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    protocol.evaluate(connection, data, generate, lambda: False)
                self.assertFalse(protocol.export_results(connection, data, path, manifest))
            with closing(protocol.open_checkpoint(checkpoint, manifest)) as connection:
                generate = Mock(return_value=("UNANSWERABLE.", [4], False))
                protocol.evaluate(connection, data, generate, lambda: False)
                generate.assert_called_once_with(protocol.PROMPT.format(question="Second?"))
                self.assertTrue(protocol.export_results(connection, data, path, manifest))
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["accuracy"], 0.5)
            self.assertEqual(metrics["num_invalid"], 1)
            self.assertEqual(metrics["recall_unanswerable"], 0)
            self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 0, 1]])

    def test_config_requires_hf_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "config.json").write_text(json.dumps({
                "model_type": "llama", "architectures": [evaluator.ARCHITECTURE], "num_hidden_layers": 80,
            }), encoding="utf-8")
            (path / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer":"model-00001.safetensors"}}', encoding="utf-8",
            )
            with patch.object(evaluator, "MODEL_PATH", path):
                with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                    evaluator.load_config()
                (path / "model-00001.safetensors").touch()
                self.assertEqual(evaluator.load_config()["model_type"], "llama")

    def test_device_map_is_contiguous_and_gpu_only(self):
        devices = evaluator.gpu_device_map(80)
        self.assertEqual(set(devices.values()), {0, 1})
        self.assertEqual([devices[f"model.layers.{i}"] for i in range(80)], [0] * 40 + [1] * 40)
        self.assertEqual(devices["model.embed_tokens"], 0)
        self.assertEqual(devices["lm_head"], 1)

    def generator_fixture(self):
        torch = SimpleNamespace(bfloat16="bf16", device=lambda name: name,
                                inference_mode=lambda: lambda function: function)
        transformers = SimpleNamespace(AutoTokenizer=Mock(), LlamaForCausalLM=Mock(),
                                       GenerationConfig=Mock())
        model = Mock()
        loading_info = {}
        transformers.LlamaForCausalLM.from_pretrained.return_value = model, loading_info
        model.is_quantized = False
        model.parameters.return_value = [
            SimpleNamespace(device="cuda:0", dtype="bf16"), SimpleNamespace(device="cuda:1", dtype="bf16"),
        ]
        model.get_input_embeddings.return_value.weight.device = "cuda:0"
        model.get_output_embeddings.return_value.weight = SimpleNamespace(shape=(128256, 8), device="cuda:1")
        model.config = SimpleNamespace(vocab_size=128256, _attn_implementation="sdpa")
        model.generation_config.eos_token_id = [128001, 128009]
        generated = MagicMock()
        generated.__getitem__.return_value.tolist.return_value = [15, 128009]
        model.generate.return_value = generated
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.chat_template = "native template"
        tokenizer.convert_tokens_to_ids.return_value = [128000, 128001, 128006, 128007, 128008, 128009]
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 128009
        tokenizer.is_fast = True
        tokenizer.vocab_size = 128000
        tokenizer.get_vocab.return_value = {"example": 15, "other": 1, "<|eot_id|>": 128009}
        inputs = {"input_ids": SimpleNamespace(shape=(1, 3)), "attention_mask": object()}
        tokenizer.apply_chat_template.return_value.to.return_value = inputs
        tokenizer.decode.return_value = "<|start_header_id|>ANSWERABLE"
        return torch, transformers, model, tokenizer, generated, inputs, loading_info

    def test_native_chat_and_generation(self):
        torch, transformers, model, tokenizer, generated, inputs, loading_info = self.generator_fixture()
        device_map = evaluator.gpu_device_map(80)
        generate = evaluator.load_generator(torch, transformers, device_map)
        prompt = protocol.PROMPT.format(question="How many?")
        raw, ids, hit_limit = generate(prompt)
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        tokenizer.apply_chat_template.return_value.to.assert_called_once_with("cuda:0")
        transformers.LlamaForCausalLM.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True, dtype="bf16", device_map=device_map,
            use_safetensors=True, output_loading_info=True,
        )
        transformers.GenerationConfig.assert_called_once_with(
            do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=[128001, 128009],
            pad_token_id=128009, use_cache=True,
        )
        model.generate.assert_called_once_with(**inputs, generation_config=transformers.GenerationConfig.return_value)
        generated.__getitem__.assert_called_once_with((0, slice(3, None)))
        tokenizer.decode.assert_called_once_with(
            [15], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        self.assertEqual((protocol.parse_label(raw), ids, hit_limit), ("INVALID", [15, 128009], False))

        generated.__getitem__.return_value.tolist.return_value = [15] * 16
        self.assertTrue(generate(prompt)[2])
        tokenizer.decode.assert_called_with(
            [15] * 16, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )

        for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs"):
            with self.subTest(load_problem=key), patch.dict(loading_info, {key: ["test detail"]}):
                with self.assertRaisesRegex(RuntimeError, key):
                    evaluator.load_generator(torch, transformers, device_map)

        with patch.object(model.generation_config, "eos_token_id", 128001):
            with self.assertRaisesRegex(ValueError, "end-of-turn"):
                evaluator.load_generator(torch, transformers, device_map)

        model.parameters.return_value[1].device = "cpu"
        with self.assertRaisesRegex(RuntimeError, "exactly two CUDA GPUs"):
            evaluator.load_generator(torch, transformers, device_map)

        tokenizer.convert_tokens_to_ids.return_value = [0] * 6
        with self.assertRaisesRegex(ValueError, "special-token IDs"):
            evaluator.load_generator(torch, transformers, device_map)

    def test_rejects_out_of_vocabulary_ids_before_decode(self):
        torch, transformers, _, tokenizer, sequences, _, _ = self.generator_fixture()
        generate = evaluator.load_generator(torch, transformers, evaluator.gpu_device_map(80))
        for invalid in (-1, 2**40, 128256, 14, 15.0):
            with self.subTest(token_id=invalid), patch("builtins.print") as output:
                ids = [15, invalid, 128009]
                sequences.__getitem__.return_value.tolist.return_value = ids
                with self.assertRaisesRegex(RuntimeError, "outside the tokenizer vocabulary"):
                    generate("prompt")
                tokenizer.decode.assert_not_called()
                report = json.loads(output.call_args.args[0])
                self.assertEqual(report["event"], "llama_before_decode")
                self.assertEqual(report["generated_token_ids"], ids)
                self.assertEqual(report["invalid_token_positions"], [1])
                self.assertEqual(report["outside_uint32_positions"], [1] if invalid in (-1, 2**40) else [])
                self.assertEqual(report["tokenizer_max_id"], 128009)
                self.assertEqual(report["lm_head_vocab_size"], 128256)
                self.assertTrue(output.call_args.kwargs["flush"])

    def test_plain_generation_logs_valid_ids_on_decoder_overflow(self):
        torch, transformers, _, tokenizer, _, _, _ = self.generator_fixture()
        generate = evaluator.load_generator(torch, transformers, evaluator.gpu_device_map(80))
        error = OverflowError("number too large to fit in target type")
        tokenizer.decode.side_effect = error
        with patch("builtins.print") as output, self.assertRaises(OverflowError) as raised:
            generate("prompt")
        self.assertIs(raised.exception, error)
        report = json.loads(output.call_args.args[0])
        self.assertEqual(report["event"], "llama_decode_overflow")
        self.assertEqual(report["generated_token_ids"], [15, 128009])
        self.assertEqual(report["invalid_token_positions"], [])

    def test_forward_diagnostic_compares_direct_logits_with_plain_generation(self):
        torch, transformers, model, tokenizer, sequences, inputs, _ = self.generator_fixture()
        torch.cuda = Mock()
        logits = MagicMock(shape=(1, 1, 13), dtype="bf16", device="cuda:0")
        final = logits.__getitem__.return_value.float.return_value
        values = [float("nan"), float("inf"), float("-inf"), *range(10)]
        cpu = final.cpu.return_value
        cpu.tolist.return_value = values
        cpu.argmax.return_value.item.return_value = 0
        final.argmax.return_value.item.return_value = 12
        model.return_value = SimpleNamespace(logits=logits)
        tokenizer.decode.return_value = "!!!!!!!!!!!!!!!!"
        sequences.__getitem__.return_value.tolist.return_value = [0] * 16
        tokenizer.get_vocab.return_value["!"] = 0

        def read_scores():
            self.assertEqual([call.args for call in torch.cuda.synchronize.call_args_list],
                             [(0,), (1,), (0,), (1,)])
            model.generate.assert_not_called()
            return values

        cpu.tolist.side_effect = read_scores
        with patch.object(evaluator, "log_generation") as log:
            status = evaluator.forward_diagnostic_test(torch, transformers, evaluator.gpu_device_map(80))
        self.assertEqual(status, 1)
        transformers.LlamaForCausalLM.from_pretrained.assert_called_once()
        self.assertEqual(model.call_args.kwargs, {
            **inputs, "use_cache": True, "logits_to_keep": 1, "return_dict": True,
        })
        self.assertEqual(model.call_count, 1)
        model.generate.assert_called_once_with(
            **inputs, generation_config=transformers.GenerationConfig.return_value,
        )
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": protocol.PROMPT.format(question=evaluator.SMOKE_PROBES[0][0])}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        self.assertEqual([call.args for call in torch.cuda.synchronize.call_args_list], [(0,), (1,)] * 3)
        forward, comparison, prediction = [call.args[1] for call in log.call_args_list]
        self.assertEqual((forward["nan_count"], forward["inf_count"]), (1, 2))
        self.assertEqual((forward["logits_shape"], forward["logits_dtype"]), ([1, 1, 13], "bf16"))
        self.assertEqual((forward["finite_min"], forward["finite_max"]), (0, 9))
        self.assertEqual(forward["top_10_finite"],
                         [{"token_id": i, "score": i - 3} for i in range(12, 2, -1)])
        self.assertEqual(comparison["generated_token_ids"], [0] * 16)
        self.assertTrue(comparison["matches_forward_cpu_argmax"])
        self.assertFalse(comparison["matches_forward_cuda_argmax"])
        self.assertTrue(prediction["hit_token_limit"])
        self.assertEqual(prediction["prediction"], "INVALID")

    def test_smoke_loads_once_and_reports_both_labels_and_gpu_peaks(self):
        torch = SimpleNamespace(cuda=Mock())
        torch.cuda.max_memory_allocated.side_effect = [65 * 1024**3, 66 * 1024**3, 67 * 1024**3, 68 * 1024**3]
        torch.cuda.max_memory_reserved.return_value = 70 * 1024**3
        generate = Mock(side_effect=[("ANSWERABLE", [7, 128009], False),
                                     ("UNANSWERABLE", [8, 128009], False)])
        gpus = [SimpleNamespace(name="A100")] * 2
        transformers = object()
        device_map = evaluator.gpu_device_map(80)
        with patch.object(evaluator, "load_generator", return_value=generate) as load, patch("builtins.print") as output:
            self.assertEqual(evaluator.smoke_test(torch, transformers, gpus, device_map), 0)
        load.assert_called_once_with(torch, transformers, device_map)
        self.assertEqual([call.args for call in generate.call_args_list], [
            (protocol.PROMPT.format(question=question),) for question, _ in evaluator.SMOKE_PROBES
        ])
        self.assertEqual([call.args for call in torch.cuda.reset_peak_memory_stats.call_args_list], [(0,), (1,)])
        report = json.loads(output.call_args.args[0])
        self.assertEqual(report["generation_mode"], "plain_tensor")
        self.assertEqual(report["max_new_tokens"], 16)
        self.assertTrue(report["passed"])
        self.assertEqual([gpu["load_peak_allocated_gib"] for gpu in report["gpus"]], [65, 66])
        self.assertEqual([gpu["combined_peak_allocated_gib"] for gpu in report["gpus"]], [67, 68])
        for result, label, token in zip(report["responses"], ("ANSWERABLE", "UNANSWERABLE"), (7, 8)):
            self.assertEqual((result["expected"], result["prediction"], result["raw_response"]), (label,) * 3)
            self.assertEqual(result["generated_token_ids"], [token, 128009])
            self.assertFalse(result["hit_token_limit"])
            self.assertTrue(result["passed"])

    def test_activation_statistics_report_finite_values_and_zeros(self):
        tensor = Mock(shape=(1, 6), dtype="bf16", device="cuda:1")
        values = tensor.detach.return_value.cpu.return_value.float.return_value.flatten.return_value
        values.tolist.return_value = [0.0, -0.0, -3.0, 7.0, float("nan"), float("-inf")]
        self.assertEqual(evaluator.activation_statistics(tensor), {
            "shape": [1, 6], "dtype": "bf16", "device": "cuda:1",
            "nan_count": 1, "inf_count": 1, "zero_count": 2,
            "finite_min": -3.0, "finite_max": 7.0, "finite_abs_mean": 2.5,
        })
        values.tolist.return_value = [float("nan"), float("inf")]
        report = evaluator.activation_statistics(tensor)
        self.assertIsNone(report["finite_min"])
        self.assertIsNone(report["finite_max"])
        self.assertIsNone(report["finite_abs_mean"])

    def test_activation_diagnostic_samples_requested_modules_and_removes_hooks(self):
        torch, transformers, model, tokenizer, _, inputs, _ = self.generator_fixture()
        torch.cuda = Mock()
        torch.count_nonzero = Mock()
        tensor = object()
        names = ("model.embed_tokens", "model.layers.0", "model.layers.39", "model.layers.40",
                 "model.layers.79", "model.norm", "lm_head")
        modules = [Mock() for _ in names]
        model.model = SimpleNamespace(embed_tokens=modules[0], norm=modules[-2],
                                      layers=dict(zip((0, 39, 40, 79), modules[1:5])))
        model.lm_head = modules[-1]
        weight = Mock(shape=(2, 3), dtype="bf16", device="cuda:1")
        weight.numel.return_value = 6
        torch.count_nonzero.return_value.item.return_value = 0
        for module in modules:
            module.named_parameters.return_value = []
        model.lm_head.named_parameters.return_value = [("weight", weight)]

        def forward(**_kwargs):
            for module in modules:
                hook = module.register_forward_hook.call_args.args[0]
                hook(module, (), (tensor,) if module is modules[1] else tensor)

        model.side_effect = forward
        inspect = evaluator.load_generator(torch, transformers, evaluator.gpu_device_map(80),
                                           activation_diagnostic=True)
        prompt = protocol.PROMPT.format(question=evaluator.SMOKE_PROBES[0][0])
        with patch.object(evaluator, "activation_statistics", return_value={}) as stats, \
                patch.object(evaluator, "log_generation") as log:
            inspect(prompt)
        transformers.LlamaForCausalLM.from_pretrained.assert_called_once()
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        model.assert_called_once_with(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
        model.generate.assert_not_called()
        self.assertEqual([call.args for call in torch.cuda.synchronize.call_args_list], [(0,), (1,)])
        self.assertEqual([call.args for call in stats.call_args_list], [(tensor,)] * 7)
        self.assertEqual([call.args[1]["module"] for call in log.call_args_list[:-1]], list(names))
        self.assertEqual(log.call_args.args, ("llama_layer_parameter", {
            "parameter": "lm_head.weight", "shape": [2, 3], "dtype": "bf16", "device": "cuda:1",
            "zero_count": 6, "all_zero": True,
        }))
        torch.count_nonzero.assert_called_once_with(weight)
        for module in modules:
            module.register_forward_hook.return_value.remove.assert_called_once()
            module.register_forward_hook.return_value.remove.reset_mock()

        model.side_effect = RuntimeError("forward failed")
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            inspect(prompt)
        for module in modules:
            module.register_forward_hook.return_value.remove.assert_called_once()

    def test_boundary_diagnostic_compares_aligned_input_and_checks_residual_when_needed(self):
        for aligned, equal, difference in (([0.0, 0.0], False, 2.0), ([2.0, -1.0], True, 0.0)):
            with self.subTest(aligned=aligned):
                torch, transformers, model, _, _, inputs, _ = self.generator_fixture()
                torch.cuda = Mock()
                layer39 = Mock()
                layer40 = Mock()
                model.model = SimpleNamespace(layers={39: layer39, 40: layer40})
                output_hooks = [
                    (layer39, [2.0, -1.0], "cuda:0"),
                    (layer40.self_attn, [-2.0, 1.0], "cuda:1"),
                    (layer40.mlp, [0.0, 0.0], "cuda:1"),
                    (layer40, [0.0, 0.0], "cuda:1"),
                ]
                input_hooks = [
                    (layer40.input_layernorm, aligned),
                    (layer40.post_attention_layernorm, [0.0, 0.0]),
                ]

                def tensor(values, device):
                    result = Mock(shape=(1, 1, 2), dtype="bf16", device=device)
                    snapshot = result.detach.return_value.cpu.return_value.float.return_value.flatten.return_value

                    def read():
                        self.assertEqual([call.args for call in torch.cuda.synchronize.call_args_list],
                                         [(0,), (1,)])
                        return values

                    snapshot.tolist.side_effect = read
                    return result

                def forward(**_kwargs):
                    for module, values in input_hooks:
                        module.register_forward_pre_hook.call_args.args[0](
                            module, (tensor(values, "cuda:1"),),
                        )
                    for module, values, device in output_hooks:
                        output = tensor(values, device)
                        if module is layer40.self_attn:
                            output = (output, None)
                        module.register_forward_hook.call_args.args[0](module, (), output)

                model.side_effect = forward
                inspect = evaluator.load_generator(torch, transformers, evaluator.gpu_device_map(80),
                                                   boundary_diagnostic=True)
                with patch.object(evaluator, "log_generation") as log:
                    inspect("prompt")
                transformers.LlamaForCausalLM.from_pretrained.assert_called_once()
                model.assert_called_once_with(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
                model.generate.assert_not_called()
                layer40.register_forward_pre_hook.assert_not_called()
                events = [call.args for call in log.call_args_list]
                reports = {report["checkpoint"]: report for event, report in events
                           if event == "llama_boundary_activation"}
                comparison, = [report for event, report in events if event == "llama_boundary_comparison"]
                self.assertEqual(comparison, {
                    "expected_devices": True, "same_shape": True, "exactly_equal": equal,
                    "max_abs_difference_finite_pairs": difference,
                })
                self.assertEqual(reports["layer39_output"]["zero_count"], 0)
                self.assertEqual(reports["layer40_input"]["zero_count"], 0 if equal else 2)
                self.assertEqual(reports["layer40_output"]["zero_count"], 2)
                self.assertEqual(len(reports), 6 if equal else 3)
                if equal:
                    self.assertEqual(reports["layer40_attention_output"]["finite_min"], -2.0)
                    self.assertEqual(reports["layer40_post_attention_residual"]["zero_count"], 2)
                    self.assertEqual(reports["layer40_mlp_output"]["zero_count"], 2)
                for module, _ in input_hooks:
                    module.register_forward_pre_hook.return_value.remove.assert_called_once()
                for module, _, _ in output_hooks:
                    module.register_forward_hook.return_value.remove.assert_called_once()

    def test_smoke_fails_if_either_probe_fails_and_reports_both(self):
        corrupt = 9223231297218904063
        for failed_probe, failure in (
            (0, ("", [], False)),
            (1, ("nonsense", list(range(16)), True)),
            (0, ("UNANSWERABLE", [7, 128009], False)),
            (1, ("UNANSWERABLE", list(range(16)), True)),
            (0, evaluator.InvalidGeneratedTokens([corrupt], False)),
        ):
            with self.subTest(probe=failed_probe, failure=failure):
                torch = SimpleNamespace(cuda=Mock())
                torch.cuda.max_memory_allocated.return_value = 0
                torch.cuda.max_memory_reserved.return_value = 0
                gpus = [SimpleNamespace(name="A100")] * 2
                outcomes = [("ANSWERABLE", [7, 128009], False),
                            ("UNANSWERABLE", [8, 128009], False)]
                outcomes[failed_probe] = failure
                generate = Mock(side_effect=outcomes)
                with patch.object(evaluator, "load_generator", return_value=generate), patch("builtins.print") as output:
                    status = evaluator.smoke_test(torch, object(), gpus, evaluator.gpu_device_map(80))
                self.assertEqual(status, 1)
                self.assertEqual(generate.call_count, 2)
                report = json.loads(output.call_args.args[0])
                self.assertFalse(report["passed"])
                failed = report["responses"][failed_probe]
                self.assertFalse(failed["passed"])
                self.assertTrue(report["responses"][1 - failed_probe]["passed"])
                if isinstance(failure, evaluator.InvalidGeneratedTokens):
                    self.assertEqual(failed["generated_token_ids"], [corrupt])
                    self.assertEqual(failed["prediction"], "INVALID")
                    self.assertIsNone(failed["raw_response"])
                    self.assertIn("outside the tokenizer vocabulary", failed["error"])
                else:
                    self.assertEqual(failed["generated_token_ids"], failure[1])
                    self.assertEqual(failed["hit_token_limit"], failure[2])


if __name__ == "__main__":
    unittest.main()
