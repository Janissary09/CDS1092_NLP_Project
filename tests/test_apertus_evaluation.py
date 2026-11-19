"""Offline coverage for Apertus protocol reuse and model integration."""

from contextlib import closing
import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_apertus", ROOT / "src/26_evaluate_apertus.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)
protocol = evaluator.shared


class ApertusEvaluationTests(unittest.TestCase):
    def test_fixed_benchmarks_and_output_paths(self):
        for dataset, count in (("umwp", 5200), ("sum", 10944)):
            with self.subTest(dataset=dataset):
                rows, digest = protocol.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), count)
                self.assertEqual(sum(row["label"] == "UNANSWERABLE" for row in rows), count // 2)
                self.assertEqual(digest, protocol.DATASETS[dataset][-1])
                self.assertEqual(evaluator.output_path(ROOT, dataset, "run_01"),
                                 ROOT / f"results/llm/apertus-v1.5-8b/{dataset}/run_01")

    def test_modified_dataset_fails_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("data/processed/umwp/test.csv")
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes((ROOT / relative).read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Dataset hash"):
                protocol.load_dataset(root, "umwp")

    def test_strict_parsing_and_invalid_policy(self):
        self.assertEqual(protocol.parse_label(" UNANSWERABLE\n"), "UNANSWERABLE")
        for raw in ("", "unanswerable", "ANSWERABLE.", "<|inner_prefix|>ANSWERABLE"):
            self.assertEqual(protocol.parse_label(raw), "INVALID")
        predictions = [
            {"true_label": truth, "prediction": predicted}
            for truth in protocol.LABELS
            for predicted in (*protocol.LABELS, "INVALID")
        ]
        metrics = protocol.score(predictions)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(metrics["num_invalid"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["precision_unanswerable"], 1 / 2)
        self.assertAlmostEqual(metrics["recall_unanswerable"], 1 / 3)
        self.assertAlmostEqual(metrics["f1_unanswerable"], 2 / 5)

    def test_resume_reuses_committed_responses_and_exports(self):
        rows = [{"id": "1", "text": "First?", "label": "ANSWERABLE"},
                {"id": "2", "text": "Second?", "label": "UNANSWERABLE"}]
        manifest = {"model_id": evaluator.MODEL_ID, "prompt": protocol.PROMPT}
        responses = Mock(side_effect=[("ANSWERABLE", [3], False), RuntimeError("interrupted")])
        with tempfile.TemporaryDirectory() as temporary:
            output = evaluator.output_path(Path(temporary), "umwp", "run_01")
            output.mkdir(parents=True)
            checkpoint = output / "checkpoint.sqlite3"
            with closing(protocol.open_checkpoint(checkpoint, manifest)) as connection:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    protocol.evaluate(connection, rows, responses, lambda: False)
                self.assertFalse(protocol.export_results(connection, rows, output, manifest))
            with self.assertRaisesRegex(ValueError, "settings differ"):
                protocol.open_checkpoint(checkpoint, {**manifest, "prompt": "changed"})
            resumed = Mock(return_value=("Label: UNANSWERABLE", [4], False))
            with closing(protocol.open_checkpoint(checkpoint, manifest)) as connection:
                protocol.evaluate(connection, rows, resumed, lambda: False)
                self.assertTrue(protocol.export_results(connection, rows, output, manifest))
            resumed.assert_called_once_with(protocol.PROMPT.format(question="Second?"))
            with (output / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual([row["id"] for row in predictions], ["1", "2"])
            self.assertEqual(predictions[1]["raw_response"], "Label: UNANSWERABLE")
            self.assertEqual(predictions[1]["prediction"], "INVALID")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["run"], manifest)
            self.assertEqual(metrics["accuracy"], 0.5)
            self.assertTrue((output / "confusion_matrix.csv").is_file())

    def test_config_preflight_reports_unsupported_transformers(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "config.json").write_text('{"model_type":"apertus1p5"}', encoding="utf-8")
            transformers = SimpleNamespace(AutoConfig=Mock())
            transformers.AutoConfig.from_pretrained.side_effect = ValueError("unknown model type")
            with patch.object(evaluator, "MODEL_PATH", model_path):
                with self.assertRaisesRegex(RuntimeError, evaluator.SWISS_AI_TRANSFORMERS_REVISION):
                    evaluator.load_config(transformers)
            transformers.AutoConfig.from_pretrained.assert_called_once_with(model_path, local_files_only=True)

    def test_generator_uses_apertus_template_and_greedy_settings(self):
        torch = SimpleNamespace(
            bfloat16="bf16", float32="fp32", device=lambda name: name,
            inference_mode=lambda: lambda function: function,
        )
        transformers = SimpleNamespace(
            AutoProcessor=Mock(), AutoModelForMultimodalLM=Mock(), GenerationConfig=Mock(),
        )
        processor = transformers.AutoProcessor.from_pretrained.return_value
        processor.tokenizer.pad_token_id = 0
        processor.apply_chat_template.return_value.to.return_value = {
            "input_ids": SimpleNamespace(shape=(1, 7)),
        }
        processor.decode.return_value = "UNANSWERABLE"
        model = transformers.AutoModelForMultimodalLM.from_pretrained.return_value.eval.return_value
        model.is_quantized = False
        model.get_input_embeddings.return_value.weight.dtype = torch.bfloat16
        model.parameters.return_value = []
        model.generation_config.eos_token_id = [1, 2]
        tokens = MagicMock()
        tokens.__getitem__ = Mock(return_value=SimpleNamespace(tolist=lambda: [42, 2]))
        model.generate.return_value = tokens

        config = object()
        generate = evaluator.load_generator(torch, transformers, config)
        prompt = protocol.PROMPT.format(question="How many?")
        self.assertEqual(generate(prompt), ("UNANSWERABLE", [42, 2], False))
        processor.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        transformers.AutoModelForMultimodalLM.from_pretrained.assert_called_once_with(
            evaluator.MODEL_PATH, config=config, local_files_only=True, dtype="bf16",
            device_map={"": 0}, use_safetensors=True,
        )
        transformers.GenerationConfig.assert_called_once_with(
            do_sample=False, num_beams=1, max_new_tokens=16, eos_token_id=[1, 2],
            pad_token_id=0, use_cache=True,
        )
        processor.decode.assert_called_once_with(
            [42], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )

    def test_slurm_configuration(self):
        wrapper = (ROOT / "slurm/evaluate_apertus.sh").read_text(encoding="ascii")
        for expected in (
            "#SBATCH --partition=students", "#SBATCH --gres=gpu:a100:1",
            "conda activate cds1092", 'src/26_evaluate_apertus.py --dataset "$1" --run-id "$2"',
            "--time=03:00:00 slurm/evaluate_apertus.sh umwp",
            "--time=04:00:00 slurm/evaluate_apertus.sh sum",
        ):
            self.assertIn(expected, wrapper)


if __name__ == "__main__":
    unittest.main()
