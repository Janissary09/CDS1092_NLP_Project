"""Offline tests for the Phi-4 reasoning evaluator."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_phi4", ROOT / "src/30_evaluate_phi4.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)


class Phi4EvaluationTests(unittest.TestCase):
    def test_shared_benchmark_and_output_paths(self):
        for dataset, size in (("umwp", 5200), ("sum", 10944)):
            with self.subTest(dataset=dataset):
                rows, digest = evaluator.shared.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), size)
                self.assertEqual(digest, evaluator.shared.DATASETS[dataset][-1])
                self.assertEqual(evaluator.output_path(ROOT, dataset, "run_01"),
                                 ROOT / f"results/llm/phi-4-reasoning-plus/{dataset}/run_01")

    def test_config_requires_local_unquantized_phi_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = {"model_type": "phi3", "architectures": ["Phi3ForCausalLM"]}
            config_path = path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            shard = path / "model-00001-of-00006.safetensors"
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"lm_head.weight": shard.name}}), encoding="utf-8",
            )
            with patch.object(evaluator, "MODEL_PATH", path):
                with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                    evaluator.load_config()
                shard.touch()
                self.assertEqual(evaluator.load_config(), config)
                config_path.write_text(json.dumps({**config, "quantization_config": {}}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unquantized"):
                    evaluator.load_config()

    def set_up_generator(self):
        torch = SimpleNamespace(bfloat16="bf16", device=lambda name: name,
                                inference_mode=lambda: lambda function: function)
        transformers = SimpleNamespace(AutoTokenizer=Mock(), Phi3ForCausalLM=Mock(),
                                       GenerationConfig=Mock())
        model = Mock()
        model.is_quantized = False
        model.parameters.return_value = [SimpleNamespace(device="cuda:0", dtype="bf16")]
        model.generation_config.eos_token_id = 100265
        transformers.Phi3ForCausalLM.from_pretrained.return_value = model, {}
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.chat_template = "native ChatML template"
        tokenizer.eos_token_id = 100265
        tokenizer.pad_token_id = 100349
        return torch, transformers, tokenizer, model

    def test_native_chat_and_strict_decode(self):
        torch, transformers, tokenizer, model = self.set_up_generator()
        inputs = {"input_ids": SimpleNamespace(shape=(1, 10)), "attention_mask": object()}
        tokenizer.apply_chat_template.return_value.to.return_value = inputs
        generated = MagicMock()
        tokens = generated.__getitem__.return_value.tolist
        model.generate.return_value = generated
        generate = evaluator.load_generator(torch, transformers)
        prompt = evaluator.shared.PROMPT.format(question="How many?")
        reasoning = "<think>There is enough information.</think> ANSWERABLE"

        tokens.return_value = [100350, 42, 100351, 9, 100265]
        tokenizer.decode.return_value = reasoning
        self.assertEqual(generate(prompt), (reasoning, tokens.return_value, False))
        self.assertEqual(evaluator.shared.parse_label(reasoning), "INVALID")
        tokenizer.decode.assert_called_once_with(
            [100350, 42, 100351, 9], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        tokenizer.apply_chat_template.return_value.to.assert_called_once_with("cuda:0")
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True,
        )
        transformers.Phi3ForCausalLM.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True, dtype="bf16", device_map={"": 0},
            use_safetensors=True, output_loading_info=True,
        )
        transformers.GenerationConfig.assert_called_once_with(
            do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=100265,
            pad_token_id=100349, use_cache=True,
        )
        model.generate.assert_called_once_with(**inputs, generation_config=transformers.GenerationConfig.return_value)
        generated.__getitem__.assert_called_once_with((0, slice(10, None)))

        tokens.return_value = [100350] + [42] * 15
        self.assertTrue(generate(prompt)[2])
        tokenizer.decode.assert_called_with(
            tokens.return_value, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        tokens.return_value = [42] * 15 + [100265]
        self.assertFalse(generate(prompt)[2])

    def test_rejects_missing_template_and_incomplete_or_offloaded_model(self):
        torch, transformers, tokenizer, model = self.set_up_generator()
        with patch.object(tokenizer, "chat_template", None):
            with self.assertRaisesRegex(ValueError, "native chat template"):
                evaluator.load_generator(torch, transformers)
        loader = transformers.Phi3ForCausalLM.from_pretrained
        loader.assert_not_called()
        loading_info = loader.return_value[1]
        with patch.dict(loading_info, {"missing_keys": ["lm_head.weight"]}):
            with self.assertRaisesRegex(RuntimeError, "lm_head.weight"):
                evaluator.load_generator(torch, transformers)
        model.parameters.return_value[0].device = "cpu"
        with self.assertRaisesRegex(RuntimeError, "BF16 inference on cuda:0"):
            evaluator.load_generator(torch, transformers)

    def test_smoke_reports_both_probes_and_fails_for_invalid_output(self):
        for response, hit_limit, status in (
            ("UNANSWERABLE", False, 0),
            ("<think>Missing information</think> UNANSWERABLE", False, 1),
            ("UNANSWERABLE", True, 1),
            ("ANSWERABLE", False, 1),
        ):
            with self.subTest(response=response, hit_limit=hit_limit):
                torch = SimpleNamespace(cuda=Mock())
                torch.cuda.max_memory_allocated.side_effect = [27 * 1024**3, 28 * 1024**3]
                torch.cuda.max_memory_reserved.return_value = 29 * 1024**3
                generate = Mock(side_effect=[("ANSWERABLE", [9, 100265], False),
                                             (response, [42] * 16 if hit_limit else [10, 100265], hit_limit)])
                with patch.object(evaluator, "load_generator", return_value=generate), patch("builtins.print") as output:
                    self.assertEqual(evaluator.smoke_test(torch, object(), SimpleNamespace(name="A100")), status)
                report = json.loads(output.call_args.args[0])
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(report["load_peak_allocated_gib"], 27)
                self.assertEqual(report["combined_peak_allocated_gib"], 28)
                self.assertEqual(report["combined_peak_reserved_gib"], 29)
                self.assertEqual(report["responses"][1]["raw_response"], response)
                self.assertEqual(report["responses"][1]["hit_token_limit"], hit_limit)
                self.assertEqual(report["passed"], status == 0)


if __name__ == "__main__":
    unittest.main()
