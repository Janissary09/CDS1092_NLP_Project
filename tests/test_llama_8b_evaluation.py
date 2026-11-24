"""Offline checks for the single-GPU Llama evaluator."""

from contextlib import redirect_stderr
from io import StringIO
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_llama_8b", ROOT / "src/29_evaluate_llama.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)


class Llama8BEvaluationTests(unittest.TestCase):
    def test_datasets_and_output_paths(self):
        self.assertEqual(evaluator.MODEL_PATH, ROOT / "models/Llama-3.1-8B-Instruct")
        for dataset, size in (("sum", 10944), ("umwp", 5200), ("mate", 9586)):
            with self.subTest(dataset=dataset):
                rows, digest = evaluator.shared.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), size)
                self.assertEqual(digest, evaluator.shared.DATASETS[dataset][-1])
                self.assertEqual(evaluator.output_path(ROOT, dataset, "run_01"),
                                 ROOT / f"results/llm/llama-3.1-8b-instruct/{dataset}/run_01")

    def test_local_config_requires_four_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"model_type": "llama", "architectures": [evaluator.ARCHITECTURE],
                      "num_hidden_layers": 32, "hidden_size": 4096}
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            mapping = {f"layer{i}": f"model-{i}.safetensors" for i in range(4)}
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": mapping}), encoding="utf-8")
            with patch.object(evaluator, "MODEL_PATH", path):
                with self.assertRaisesRegex(FileNotFoundError, "four indexed"):
                    evaluator.load_config()
                for shard in mapping.values():
                    (path / shard).touch()
                self.assertEqual(evaluator.load_config(), config)

    def test_cli_dispatches_smoke_and_dataset_with_valid_arguments(self):
        torch = SimpleNamespace(cuda=Mock())
        transformers = SimpleNamespace(LlamaForCausalLM=object())
        gpu = object()
        with (patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
              patch.object(evaluator, "require_gpu", return_value=gpu),
              patch.object(evaluator, "load_config"),
              patch.object(evaluator, "load_generator") as load_generator,
              patch.object(evaluator, "smoke_test", return_value=0) as smoke):
            with patch.object(sys, "argv", [spec.origin, "--smoke-test"]):
                self.assertEqual(evaluator.main(), 0)
            smoke.assert_called_once_with(
                torch, load_generator.return_value, gpu, torch.cuda.max_memory_allocated.return_value,
            )

            for dataset in ("sum", "umwp", "mate"):
                with (self.subTest(dataset=dataset),
                      patch.object(sys, "argv", [spec.origin, "--dataset", dataset, "--run-id", "run_01"]),
                      patch.object(evaluator.shared, "load_dataset", side_effect=LookupError) as load,
                      self.assertRaises(LookupError)):
                    evaluator.main()
                load.assert_called_once_with(ROOT, dataset)

    def test_native_greedy_generation_and_device_checks_with_mocks(self):
        torch = SimpleNamespace(device=lambda name: name, bfloat16="bf16",
                                inference_mode=lambda: lambda fn: fn)
        transformers = SimpleNamespace(AutoTokenizer=Mock(), LlamaForCausalLM=Mock(),
                                       GenerationConfig=Mock())
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.chat_template = "native template"
        tokenizer.convert_tokens_to_ids.return_value = 128009
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 128009
        tokenizer.get_vocab.return_value = {"word": 7, "<|eot_id|>": 128009}
        tokenizer.decode.return_value = "ANSWERABLE."
        inputs = {"input_ids": SimpleNamespace(shape=(1, 3)), "attention_mask": object()}
        tokenizer.apply_chat_template.return_value.to.return_value = inputs

        model = Mock(is_quantized=False)
        parameter = SimpleNamespace(device="cuda:0", dtype="bf16")
        buffer = SimpleNamespace(device="cuda:0")
        model.parameters.return_value = [parameter]
        model.buffers.return_value = [buffer]
        model.generation_config.eos_token_id = [128001, 128009]
        model.get_output_embeddings.return_value.weight.shape = (128256, 4096)
        transformers.LlamaForCausalLM.from_pretrained.return_value = model, {}
        generated = MagicMock()
        generated.__getitem__.return_value.tolist.return_value = [7, 128009]
        model.generate.return_value = generated

        generate = evaluator.load_generator(torch, transformers)
        transformers.LlamaForCausalLM.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True, dtype="bf16", device_map={"": 0},
            use_safetensors=True, output_loading_info=True,
        )
        transformers.GenerationConfig.assert_called_once_with(
            do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=[128001, 128009],
            pad_token_id=128009, use_cache=True,
        )
        prompt = evaluator.shared.PROMPT.format(question="How many?")
        raw, ids, hit_limit = generate(prompt)
        self.assertEqual((evaluator.shared.parse_label(raw), ids, hit_limit),
                         ("INVALID", [7, 128009], False))
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        tokenizer.apply_chat_template.return_value.to.assert_called_once_with("cuda:0")
        model.generate.assert_called_once_with(
            **inputs, generation_config=transformers.GenerationConfig.return_value,
        )
        tokenizer.decode.assert_called_once_with(
            [7], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        generated.__getitem__.return_value.tolist.return_value = [7] * 16
        self.assertTrue(generate(prompt)[2])
        buffer.device = "cpu"
        with self.assertRaisesRegex(RuntimeError, "cuda:0"):
            evaluator.load_generator(torch, transformers)

    def test_smoke_requires_both_expected_strict_labels(self):
        torch = SimpleNamespace(cuda=Mock())
        torch.cuda.max_memory_allocated.return_value = 16 * 1024**3
        gpu = SimpleNamespace(name="A100")
        for second, status in (("UNANSWERABLE", 0), ("UNANSWERABLE.", 1), ("ANSWERABLE", 1)):
            generate = Mock(side_effect=[
                ("ANSWERABLE", [7, 128009], False), (second, [8, 128009], False),
            ])
            with patch("builtins.print") as output:
                self.assertEqual(evaluator.smoke_test(torch, generate, gpu, 16 * 1024**3), status)
                result = json.loads(output.call_args.args[0])
                self.assertEqual(result["passed"], status == 0)
                self.assertEqual(result["device"], "cuda:0")


    def test_smoke_override_requires_dataset(self):
        stderr = StringIO()
        with (patch.object(sys, "argv", [spec.origin, "--smoke-test", "--allow-failed-smoke"]),
              redirect_stderr(stderr), self.assertRaises(SystemExit) as error):
            evaluator.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--allow-failed-smoke requires --dataset", stderr.getvalue())

    def test_benchmark_smoke_gate_and_exported_metadata(self):
        torch = SimpleNamespace(cuda=Mock(), __version__="offline",
                                version=SimpleNamespace(cuda="offline"))
        torch.cuda.max_memory_allocated.return_value = 0
        transformers = SimpleNamespace(LlamaForCausalLM=object(), __version__="offline")
        gpu = SimpleNamespace(name="A100")
        config = {"model_type": "llama", "architectures": [evaluator.ARCHITECTURE]}
        rows = [{"id": "1", "text": "Benchmark question?", "label": "ANSWERABLE"}]
        argv = [spec.origin, "--dataset", "mate", "--run-id", "run_01"]

        for second, override, status in (
            ("ANSWERABLE", False, 1), ("ANSWERABLE", True, 0),
            ("UNANSWERABLE", False, 0), ("UNANSWERABLE", True, 0),
        ):
            with self.subTest(second=second, override=override), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run_01"
                generate = Mock(side_effect=[
                    ("ANSWERABLE", [7], False), (second, [8], False), ("ANSWERABLE", [9], False),
                ])
                flags = ["--allow-failed-smoke"] if override else []
                with (
                    patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
                    patch.object(sys, "argv", argv + flags),
                    patch.object(evaluator, "require_gpu", return_value=gpu),
                    patch.object(evaluator, "load_config", return_value=config),
                    patch.object(evaluator, "load_generator", return_value=generate) as load_model,
                    patch.object(evaluator, "output_path", return_value=output),
                    patch.object(evaluator, "version", return_value="offline"),
                    patch.object(evaluator.signal, "signal"),
                    patch.object(evaluator.shared, "load_dataset", return_value=(rows, "digest")) as load,
                    patch.object(evaluator.shared, "model_fingerprint", return_value="fingerprint"),
                    patch("builtins.print"),
                ):
                    self.assertEqual(evaluator.main(), status)
                load_model.assert_called_once_with(torch, transformers)
                if status:
                    load.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(generate.call_count, 2)
                else:
                    load.assert_called_once_with(ROOT, "mate")
                    run = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["run"]
                    self.assertEqual(run["smoke_test_passed"], second == "UNANSWERABLE")
                    self.assertEqual(run["allow_failed_smoke"], override)
                    self.assertEqual(run["failed_smoke_override_applied"], second == "ANSWERABLE")
                    self.assertEqual(generate.call_count, 3)

    def test_override_does_not_swallow_technical_errors(self):
        torch = SimpleNamespace(cuda=Mock())
        transformers = SimpleNamespace(LlamaForCausalLM=object())
        torch.cuda.max_memory_allocated.return_value = 0
        argv = [spec.origin, "--dataset", "mate", "--run-id", "run_01", "--allow-failed-smoke"]
        for failure_stage in ("require_gpu", "load_generator", "generate", "synchronize"):
            with self.subTest(failure_stage=failure_stage):
                failure = RuntimeError(f"Technical failure: {failure_stage}")
                torch.cuda.synchronize.side_effect = failure if failure_stage == "synchronize" else None
                generate = Mock(side_effect=failure)
                with (
                    patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
                    patch.object(sys, "argv", argv),
                    patch.object(evaluator, "require_gpu",
                                 side_effect=failure if failure_stage == "require_gpu" else None),
                    patch.object(evaluator, "load_config"),
                    patch.object(evaluator, "load_generator", return_value=generate,
                                 side_effect=failure if failure_stage == "load_generator" else None),
                    patch.object(evaluator.shared, "load_dataset") as load,
                    self.assertRaises(RuntimeError) as error,
                ):
                    evaluator.main()
                self.assertIs(error.exception, failure)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
