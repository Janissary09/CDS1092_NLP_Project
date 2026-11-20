"""Offline tests for the Mistral evaluator and shared benchmark protocol."""

from contextlib import closing
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_mistral", ROOT / "src/27_evaluate_mistral.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)
protocol = evaluator.shared


class MistralEvaluationTests(unittest.TestCase):
    def test_complete_datasets_and_result_paths(self):
        for dataset, size in (("umwp", 5200), ("sum", 10944)):
            with self.subTest(dataset=dataset):
                rows, digest = protocol.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), size)
                self.assertEqual(digest, protocol.DATASETS[dataset][-1])
                self.assertEqual(evaluator.output_path(ROOT, dataset, "run_01"),
                                 ROOT / f"results/llm/mistral-small-3.2-24b/{dataset}/run_01")

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
                "model_type": "mistral3", "architectures": [evaluator.ARCHITECTURE],
            }), encoding="utf-8")
            (path / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer":"model-00001.safetensors"}}', encoding="utf-8",
            )
            with patch.object(evaluator, "MODEL_PATH", path):
                with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                    evaluator.load_config()
                (path / "model-00001.safetensors").touch()
                self.assertEqual(evaluator.load_config()["model_type"], "mistral3")

    def test_native_chat_and_generation(self):
        torch = SimpleNamespace(bfloat16="bf16", long="long", device=lambda name: name,
                                inference_mode=lambda: lambda function: function,
                                tensor=Mock(), ones_like=Mock())
        transformers = SimpleNamespace(Mistral3ForConditionalGeneration=Mock(), GenerationConfig=Mock())
        model = transformers.Mistral3ForConditionalGeneration.from_pretrained.return_value.eval.return_value
        model.is_quantized = False
        model.parameters.return_value = []
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = None
        generated = MagicMock()
        generated.__getitem__.return_value.tolist.return_value = [15, 2]
        model.generate.return_value = generated
        tokenizer, request, decode = Mock(), Mock(), Mock(return_value="[INST]ANSWERABLE")
        tokenizer.encode_chat_completion.return_value.tokens = [1, 9, 10]

        with patch.object(evaluator, "load_tokenizer", return_value=(tokenizer, request, decode, 2)):
            generate = evaluator.load_generator(torch, transformers)
        prompt = protocol.PROMPT.format(question="How many?")
        raw, ids, hit_limit = generate(prompt)
        request.assert_called_once_with(messages=[{"role": "user", "content": prompt}])
        tokenizer.encode_chat_completion.assert_called_once_with(request.return_value)
        torch.tensor.assert_called_once_with([[1, 9, 10]], dtype="long", device="cuda:0")
        transformers.Mistral3ForConditionalGeneration.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, local_files_only=True, dtype="bf16", device_map={"": 0},
            use_safetensors=True,
        )
        transformers.GenerationConfig.assert_called_once_with(
            do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=2, pad_token_id=2, use_cache=True,
        )
        generated.__getitem__.assert_called_once_with((0, slice(3, None)))
        decode.assert_called_once_with([15])
        self.assertEqual((protocol.parse_label(raw), ids, hit_limit), ("INVALID", [15, 2], False))

    def test_smoke_generates_once_and_reports_peak(self):
        torch = SimpleNamespace(cuda=Mock())
        torch.cuda.max_memory_allocated.side_effect = [45 * 1024**3, 46 * 1024**3]
        torch.cuda.max_memory_reserved.return_value = 47 * 1024**3
        generate = Mock(return_value=("ANSWERABLE", [7], False))
        with patch.object(evaluator, "load_generator", return_value=generate), patch("builtins.print") as output:
            self.assertEqual(evaluator.smoke_test(torch, object(), SimpleNamespace(name="A100")), 0)
        generate.assert_called_once()
        torch.cuda.reset_peak_memory_stats.assert_called_once()
        report = json.loads(output.call_args.args[0])
        self.assertEqual(report["load_peak_allocated_gib"], 45)
        self.assertEqual(report["combined_peak_allocated_gib"], 46)
        self.assertEqual(report["combined_peak_reserved_gib"], 47)


if __name__ == "__main__":
    unittest.main()
