"""Exercise the MathBERT CLI up to the model-runtime import boundary."""

import builtins
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_mathbert", ROOT / "src/17_evaluate_mathbert.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)


class ReachedModelRuntime(Exception):
    """Stop before importing model dependencies."""


class MathBertEvaluationTests(unittest.TestCase):
    def test_explicit_benchmarks_reach_model_runtime_after_dataset_validation(self):
        benchmarks = {
            "sum": (10944, "sum_test", {"pair_id"}),
            "umwp": (5200, "umwp", {"id", "category", "source"}),
            "mate": (9586, "mate", {"id", "pair_id", "source", "category", "label_id"}),
        }

        def import_until_runtime(name, *args, **kwargs):
            if name in {"torch", "transformers"}:
                raise ReachedModelRuntime()
            return original_import(name, *args, **kwargs)

        original_import = builtins.__import__
        original_exists = Path.exists
        for dataset, (count, output_name, metadata) in benchmarks.items():
            with self.subTest(dataset=dataset):
                output = evaluator.CONFIG_PATH.parent / output_name
                checked_outputs = []

                def exists(path):
                    if path == output:
                        checked_outputs.append(path)
                        return False
                    return original_exists(path)

                def load_and_check(name):
                    path, frame = original_loader(name)
                    self.assertEqual(path, ROOT / "data" / "processed" / dataset / "test.csv")
                    self.assertEqual(len(frame), count)
                    self.assertEqual(set(frame.columns), {"text", "label"} | metadata)
                    self.assertEqual(frame["label"].value_counts().to_dict(), {
                        "ANSWERABLE": count // 2, "UNANSWERABLE": count // 2,
                    })
                    return path, frame

                original_loader = evaluator.load_dataset
                with (
                    patch.object(sys, "argv", [spec.origin, "--dataset", dataset]),
                    patch.object(sys, "path", [str(ROOT / "src"), *sys.path]),
                    patch.object(evaluator, "MODEL_PATH", Mock(is_dir=Mock(return_value=True))),
                    patch.object(Path, "exists", exists),
                    patch.object(evaluator, "load_dataset", side_effect=load_and_check) as loader,
                    patch.object(builtins, "__import__", import_until_runtime),
                    self.assertRaises(ReachedModelRuntime),
                ):
                    evaluator.main()
                loader.assert_called_once_with(dataset)
                self.assertEqual(checked_outputs, [output, output])

    def test_default_and_both_select_only_sum_and_umwp(self):
        for arguments in ([], ["--dataset", "both"]):
            with self.subTest(arguments=arguments):
                with (
                    patch.object(sys, "argv", [spec.origin, *arguments]),
                    patch.object(Path, "exists", return_value=False),
                    patch.object(evaluator, "evaluate_dataset") as evaluate,
                ):
                    evaluator.main()
                self.assertEqual([entry.args for entry in evaluate.call_args_list], [
                    ("sum",), ("umwp",),
                ])

    def test_unsupported_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported benchmark"):
            evaluator.evaluate_dataset("unsupported")


if __name__ == "__main__":
    unittest.main()
