"""Offline tests for the full-dataset Qwen evaluator."""

from contextlib import closing
import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_qwen", ROOT / "src/25_evaluate_qwen.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class QwenEvaluationTests(unittest.TestCase):
    def test_complete_benchmarks(self):
        for dataset, count, identifier, unique in (
            ("umwp", 5200, "id", 5200), ("sum", 10944, "pair_id", 5472),
        ):
            with self.subTest(dataset=dataset):
                rows, _ = evaluator.load_dataset(ROOT, dataset)
                self.assertEqual(len(rows), count)
                self.assertEqual(len({row[identifier] for row in rows}), unique)
                self.assertEqual(sum(row["label"] == "UNANSWERABLE" for row in rows), count // 2)

    def test_changed_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data/processed/umwp/test.csv"
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / "data/processed/umwp/test.csv").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Dataset hash"):
                evaluator.load_dataset(root, "umwp")

    def test_fingerprint_reuses_unchanged_shards_and_rechecks_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            first = model / "model-00001.safetensors"
            second = model / "model-00002.safetensors"
            config = model / "config.json"
            for path in (first, second, config):
                path.write_bytes(b"original")
            cache = root / "fingerprint.json"
            baseline = evaluator.model_fingerprint(model, cache)

            with patch.object(evaluator, "hash_model_file", wraps=evaluator.hash_model_file) as hashed:
                self.assertEqual(evaluator.model_fingerprint(model, cache), baseline)
                self.assertEqual([call.args[0] for call in hashed.call_args_list], [config])
                hashed.reset_mock()
                replacement = root / "replacement"
                replacement.write_bytes(b"replaced")
                replacement.replace(first)
                changed = evaluator.model_fingerprint(model, cache)
                self.assertNotEqual(changed, baseline)
                self.assertEqual([call.args[0] for call in hashed.call_args_list], [config, first])
                config.write_bytes(b"new metadata")
                self.assertNotEqual(evaluator.model_fingerprint(model, cache), changed)

            cache.write_text("broken JSON", encoding="utf-8")
            with patch.object(evaluator, "hash_model_file", wraps=evaluator.hash_model_file) as hashed:
                evaluator.model_fingerprint(model, cache)
                self.assertEqual([call.args[0] for call in hashed.call_args_list], [config, first, second])

    def test_exact_label_parsing(self):
        self.assertEqual(evaluator.parse_label(" \nANSWERABLE\t"), "ANSWERABLE")
        self.assertEqual(evaluator.parse_label("UNANSWERABLE"), "UNANSWERABLE")
        for raw in ("unanswerable", "UNANSWERABLE.", "", "ANSWERABLE or UNANSWERABLE",
                    "<think></think>ANSWERABLE", "Label: ANSWERABLE"):
            with self.subTest(raw=raw):
                self.assertEqual(evaluator.parse_label(raw), "INVALID")

    def test_invalid_outputs_count_in_primary_metrics(self):
        rows = [
            {"true_label": truth, "prediction": predicted}
            for truth in evaluator.LABELS
            for predicted in (*evaluator.LABELS, "INVALID")
        ]
        metrics = evaluator.score(rows)
        self.assertEqual(metrics["num_examples"], 6)
        self.assertEqual(metrics["num_invalid"], 2)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 1], [1, 1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["precision_unanswerable"], 1 / 2)
        self.assertAlmostEqual(metrics["recall_unanswerable"], 1 / 3)
        self.assertAlmostEqual(metrics["f1_unanswerable"], 2 / 5)

    def test_interruption_resume_and_export(self):
        rows = [
            {"pair_id": "42", "text": "first {x}", "label": "ANSWERABLE"},
            {"pair_id": "42", "text": "second", "label": "UNANSWERABLE"},
            {"pair_id": "43", "text": "third", "label": "UNANSWERABLE"},
        ]
        prompts = []
        manifest = {"run_id": "test"}

        def interrupted(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return "ANSWERABLE", [10], False
            raise RuntimeError("interrupted before commit")

        def resumed(prompt):
            prompts.append(prompt)
            return "\nLabel: UNANSWERABLE\n", [20, 21], True

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint.sqlite3"
            with closing(evaluator.open_checkpoint(checkpoint, manifest)) as connection:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    evaluator.evaluate(connection, rows, interrupted, lambda: False)
                self.assertFalse(evaluator.export_results(connection, rows, output, manifest))
                self.assertFalse((output / "metrics.json").exists())
                with self.assertRaises(sqlite3.OperationalError):
                    evaluator.open_checkpoint(checkpoint, manifest)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                evaluator.open_checkpoint(checkpoint, {"run_id": "other"})
            with closing(evaluator.open_checkpoint(checkpoint, manifest)) as connection:
                evaluator.evaluate(connection, rows, resumed, lambda: False)
                self.assertTrue(evaluator.export_results(connection, rows, output, manifest))
            self.assertEqual(prompts, [
                evaluator.PROMPT.format(question=rows[index]["text"]) for index in [0, 1, 1, 2]
            ])
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual([row["row_index"] for row in predictions], ["0", "1", "2"])
            self.assertEqual([row["pair_id"] for row in predictions], ["42", "42", "43"])
            self.assertEqual(predictions[1]["raw_response"], "\nLabel: UNANSWERABLE\n")
            self.assertEqual(predictions[1]["prediction"], "INVALID")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["num_invalid"], 2)
            self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 0, 2]])


if __name__ == "__main__":
    unittest.main()
