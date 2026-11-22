"""Offline tests for the OpenAI benchmark adapter."""

from contextlib import nullcontext
import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("evaluate_openai", ROOT / "src/33_evaluate_openai.py")
evaluator = module_from_spec(spec)
spec.loader.exec_module(evaluator)


class FakeAPIError(Exception):
    def __init__(self, status=None, code=None, headers=None):
        super().__init__("provider error details must stay private")
        self.status_code = status
        self.code = code
        self.response = SimpleNamespace(headers=headers or {})


def response(content, finish_reason="stop", *, with_usage=True):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2, total_tokens=102,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=20))
    return SimpleNamespace(
        id="response-id", model="returned-model-id", usage=usage if with_usage else None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None),
                                 finish_reason=finish_reason)],
    )


class OpenAIEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = evaluator.DEFAULT_MODEL
        self.output = evaluator.output_path(self.root, self.model, "umwp", "run_01", 2)
        self.rows = [{"id": "1", "text": "Known?", "label": "ANSWERABLE"},
                     {"id": "2", "text": "Unknown?", "label": "UNANSWERABLE"}]
        self.manifest = {"parameters": evaluator.request_parameters(self.model),
                         "prices_usd_per_million": evaluator.PRICES[self.model]}
        self.client = Mock()
        self.create = self.client.chat.completions.create
        self.factory = Mock(side_effect=lambda: nullcontext(self.client))
        self.sleep = Mock()

    def evaluate(self, **options):
        evaluator.run_evaluation(self.output, self.rows, self.manifest, self.factory,
                                 FakeAPIError, sleep=self.sleep, **options)

    def metadata(self):
        return json.loads((self.output / "run_metadata.json").read_text(encoding="utf-8"))

    def test_exact_prompt_usage_and_strict_exports(self):
        self.create.side_effect = [response(" ANSWERABLE\n"), response("UNANSWERABLE.", "length")]
        self.evaluate()
        self.assertEqual(self.create.call_count, 2)
        for row, call in zip(self.rows, self.create.call_args_list):
            self.assertEqual(call.kwargs, {
                "model": "gpt-6.1-terra", "reasoning_effort": "none", "temperature": 0,
                "max_completion_tokens": 16, "store": False, "service_tier": "default",
                "messages": [{"role": "user", "content": evaluator.shared.PROMPT.format(question=row["text"])}],
            })
        with (self.output / "predictions.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["prediction"] for row in rows], ["ANSWERABLE", "INVALID"])
        self.assertEqual(rows[1]["raw_response"], "UNANSWERABLE.")
        self.assertEqual(rows[1]["generated_token_ids"], "null")
        self.assertEqual(rows[1]["hit_token_limit"], "True")
        metrics = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 0, 1]])
        self.assertTrue((self.output / "confusion_matrix.csv").is_file())
        usage = self.metadata()["usage"]
        self.assertEqual((usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]), (200, 4, 204))
        self.assertEqual(usage["estimated_cost_usd"], 0.000182)
        self.assertEqual(usage["returned_models"], ["returned-model-id"])
        self.assertEqual(usage["request_attempts"], 2)
        self.evaluate()
        self.factory.assert_called_once()

    def test_resume_does_not_resend_completed_or_uncertain_rows_without_opt_in(self):
        self.create.side_effect = [response("ANSWERABLE"), FakeAPIError()]
        with self.assertRaisesRegex(RuntimeError, "outcome uncertain") as error:
            self.evaluate()
        self.assertTrue(error.exception.__suppress_context__)
        started_at = self.metadata()["started_at_utc"]
        self.assertEqual(self.metadata()["usage"]["attempts_by_state"], {"completed": 1, "pending": 1})
        with self.assertRaisesRegex(RuntimeError, "uncertain API attempt"):
            self.evaluate()
        self.assertEqual(self.create.call_count, 2)
        self.create.side_effect = [response("UNANSWERABLE", with_usage=False)]
        self.evaluate(allow_uncertain=True)
        self.assertEqual(self.create.call_count, 3)
        self.assertIn("Unknown?", self.create.call_args.kwargs["messages"][0]["content"])
        final = self.metadata()
        self.assertEqual(final["started_at_utc"], started_at)
        self.assertEqual(final["usage"]["responses_without_usage"], 1)
        self.assertEqual(final["usage"]["request_attempts"], 3)

    def test_transient_retries_are_bounded_and_quota_is_not_retried(self):
        self.create.side_effect = [FakeAPIError(429, headers={"Retry-After": "4"}), FakeAPIError(503),
                                   FakeAPIError(503)]
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            self.evaluate()
        self.assertEqual(self.create.call_count, 3)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [4, 2])
        self.create.side_effect = [FakeAPIError(429, "insufficient_quota")]
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            self.evaluate()
        self.assertEqual(self.create.call_count, 4)
        self.assertEqual(self.metadata()["usage"]["attempts_by_state"], {"rejected": 4})
        self.assertFalse((self.output / "predictions.csv").exists())

    def test_key_precedence_and_run_path_isolation(self):
        (self.root / ".env").touch()
        read = Mock(return_value={"OPENAI_API_KEY": "placeholder-file"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder-env"}, clear=True):
            self.assertEqual(evaluator.api_key(self.root, read), "placeholder-env")
        read.assert_not_called()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(evaluator.api_key(self.root, read), "placeholder-file")
        read.assert_called_once_with(self.root / ".env", interpolate=False)
        full = evaluator.output_path(self.root, self.model, "umwp", "run_01", None)
        self.assertNotEqual(self.output, full)
        self.assertEqual(self.output.parent.name, "limit_2")


if __name__ == "__main__":
    unittest.main()
