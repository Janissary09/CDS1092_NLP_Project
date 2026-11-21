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

    def test_smoke_logs_valid_ids_and_logits_before_decoder_overflow(self):
        torch, transformers, model, tokenizer, sequences, _, _ = self.generator_fixture()
        values = Mock()
        values.argmax.return_value.item.return_value = 15
        logits = MagicMock()
        logits.__getitem__.return_value.float.return_value.cpu.return_value = values
        torch.isfinite = Mock()
        torch.isfinite.return_value.all.return_value.item.return_value = True
        model.generate.return_value = SimpleNamespace(
            sequences=sequences, logits=(logits,), past_key_values=None,
        )
        generate = evaluator.load_generator(torch, transformers, evaluator.gpu_device_map(80), diagnostics=True)
        error = OverflowError("number too large to fit in target type")
        with patch("builtins.print") as output:
            def fail_decode(*args, **kwargs):
                pre_decode, logit_report = [json.loads(call.args[0]) for call in output.call_args_list]
                self.assertEqual(pre_decode["generated_token_ids"], [15, 128009])
                self.assertEqual(pre_decode["invalid_token_positions"], [])
                self.assertEqual(logit_report["raw_logits_cpu_argmax_ids"], [15])
                raise error

            tokenizer.decode.side_effect = fail_decode
            with self.assertRaises(OverflowError) as raised:
                generate("prompt")
            self.assertIs(raised.exception, error)
            self.assertEqual(json.loads(output.call_args.args[0])["event"], "llama_decode_overflow")
        self.assertTrue(model.generate.call_args.kwargs["output_logits"])

        torch.isfinite.return_value.all.return_value.item.return_value = False
        tokenizer.decode.reset_mock()
        with patch("builtins.print") as output:
            with self.assertRaisesRegex(RuntimeError, "non-finite logits"):
                generate("prompt")
            self.assertEqual(json.loads(output.call_args.args[0])["nonfinite_logit_steps"], [0])
        tokenizer.decode.assert_not_called()

    def test_smoke_reports_memory_on_both_gpus(self):
        torch = SimpleNamespace(cuda=Mock())
        torch.cuda.max_memory_allocated.side_effect = [65 * 1024**3, 66 * 1024**3, 67 * 1024**3, 68 * 1024**3]
        torch.cuda.max_memory_reserved.return_value = 70 * 1024**3
        generate = Mock(return_value=("ANSWERABLE", [7], False))
        gpus = [SimpleNamespace(name="A100"), SimpleNamespace(name="A100")]
        with patch.object(evaluator, "load_generator", return_value=generate), patch("builtins.print") as output:
            self.assertEqual(evaluator.smoke_test(torch, object(), gpus, evaluator.gpu_device_map(80)), 0)
        generate.assert_called_once()
        self.assertEqual([call.args for call in torch.cuda.reset_peak_memory_stats.call_args_list], [(0,), (1,)])
        report = json.loads(output.call_args.args[0])
        self.assertEqual([gpu["load_peak_allocated_gib"] for gpu in report["gpus"]], [65, 66])
        self.assertEqual([gpu["combined_peak_allocated_gib"] for gpu in report["gpus"]], [67, 68])

    def test_smoke_rejects_invalid_wrong_or_truncated_output(self):
        for raw, ids, hit_limit in (
            ("", [], False),
            ("nonsense", list(range(16)), True),
            ("UNANSWERABLE", [7, 128009], False),
            ("ANSWERABLE", list(range(16)), True),
        ):
            with self.subTest(raw=raw, hit_limit=hit_limit):
                torch = SimpleNamespace(cuda=Mock())
                torch.cuda.max_memory_allocated.return_value = 0
                torch.cuda.max_memory_reserved.return_value = 0
                gpus = [SimpleNamespace(name="A100")] * 2
                with patch.object(evaluator, "load_generator", return_value=Mock(
                    return_value=(raw, ids, hit_limit),
                )), patch("builtins.print") as output:
                    status = evaluator.smoke_test(torch, object(), gpus, evaluator.gpu_device_map(80))
                self.assertEqual(status, 1)
                self.assertEqual(json.loads(output.call_args.args[0])["generated_token_ids"], ids)


if __name__ == "__main__":
    unittest.main()
