"""Evaluate an OpenAI API model with the frozen CDS1092 zero-shot protocol."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SHARED_PATH = Path(__file__).with_name("25_evaluate_qwen.py")
spec = spec_from_file_location("cds1092_zero_shot_protocol", SHARED_PATH)
shared = module_from_spec(spec)
spec.loader.exec_module(shared)

BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6.1-terra"
# Standard-tier USD rates per million text tokens, verified 2026-11-22.
PRICES = {
    "gpt-6.1-terra": {"input": 1.00, "cached_input": 0.05, "output": 5.00},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
}
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 45
MAX_RETRY_DELAY = 60


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def output_path(root, model, dataset, run_id, limit):
    dataset_dir = root / "results/api/openai" / model / dataset
    return dataset_dir / run_id if limit is None else dataset_dir / f"limit_{limit}" / run_id


def request_parameters(model):
    return {
        "model": model, "reasoning_effort": "none", "temperature": 0,
        "max_completion_tokens": shared.MAX_NEW_TOKENS, "store": False, "service_tier": "default",
    }


def api_key(root, read_dotenv):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and (root / ".env").is_file():
        key = (read_dotenv(root / ".env", interpolate=False).get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the environment or project-root .env.")
    return key


def open_api_tables(connection):
    with connection:
        connection.execute("CREATE TABLE IF NOT EXISTS api_run (started_at TEXT NOT NULL)")
        connection.execute("INSERT INTO api_run SELECT ? WHERE NOT EXISTS (SELECT 1 FROM api_run)",
                           (utc_now(),))
        connection.execute("""
            CREATE TABLE IF NOT EXISTS api_attempts (
                id INTEGER PRIMARY KEY, row_index INTEGER NOT NULL, started_at TEXT NOT NULL,
                state TEXT NOT NULL, http_status INTEGER
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS api_attempt_rows ON api_attempts (row_index, id)")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS api_responses (
                row_index INTEGER PRIMARY KEY, details TEXT NOT NULL
            )
        """)


def retry_delay(error, attempt):
    response = getattr(error, "response", None)
    value = response.headers.get("Retry-After") if response is not None else None
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = None
        if seconds is not None and math.isfinite(seconds) and seconds >= 0:
            return seconds
    return 2**attempt


def response_details(response):
    choice, = response.choices
    usage = response.usage
    details = {
        "response_id": response.id, "model": response.model, "finish_reason": choice.finish_reason,
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "service_tier": getattr(response, "service_tier", None),
        "refused": bool(getattr(choice.message, "refusal", None)),
        "usage": None,
    }
    if usage is not None:
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        details["usage"] = {
            "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens, "cached_input_tokens": cached or 0,
        }
    return choice.message.content or "", details


def request_row(connection, client, api_error, index, prompt, parameters, allow_uncertain, sleep):
    previous = connection.execute(
        "SELECT state FROM api_attempts WHERE row_index = ? ORDER BY id DESC LIMIT 1", (index,),
    ).fetchone()
    if previous == ("pending",) and not allow_uncertain:
        raise RuntimeError(f"Row {index} has an uncertain API attempt. See the resume instructions.")
    for attempt in range(MAX_ATTEMPTS):
        with connection:
            request_id = connection.execute(
                "INSERT INTO api_attempts (row_index, started_at, state) VALUES (?, ?, 'pending')",
                (index, utc_now()),
            ).lastrowid
        try:
            response = client.chat.completions.create(
                **parameters, messages=[{"role": "user", "content": prompt}],
            )
        except api_error as error:
            status = getattr(error, "status_code", None)
            quota = getattr(error, "code", None) in {"insufficient_quota", "billing_hard_limit_reached"}
            # A timeout or dropped connection can hide a successful, billable request.
            if status is None or status == 408 or (status >= 500 and status != 503):
                raise RuntimeError(f"API outcome uncertain at row {index}. See the resume instructions.") from None
            with connection:
                connection.execute("UPDATE api_attempts SET state = 'rejected', http_status = ? WHERE id = ?",
                                   (status, request_id))
            delay = retry_delay(error, attempt)
            if quota or status not in {429, 503} or attempt + 1 == MAX_ATTEMPTS or delay > MAX_RETRY_DELAY:
                raise RuntimeError(f"API request stopped at row {index} (HTTP {status}).") from None
            sleep(delay)
            continue

        raw, details = response_details(response)
        with connection:
            connection.execute("INSERT INTO responses VALUES (?, ?, ?, ?)",
                               (index, raw, "null", int(details["finish_reason"] == "length")))
            connection.execute("INSERT INTO api_responses VALUES (?, ?)", (index, json.dumps(details)))
            connection.execute("UPDATE api_attempts SET state = 'completed' WHERE id = ?", (request_id,))
        return


def summarize_usage(connection, prices):
    counts = dict(connection.execute("SELECT state, COUNT(*) FROM api_attempts GROUP BY state"))
    responses = [json.loads(row[0]) for row in connection.execute("SELECT details FROM api_responses")]
    measured = [row["usage"] for row in responses if row["usage"] is not None]
    totals = {key: sum(usage[key] for usage in measured)
              for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")}
    cached = totals["cached_input_tokens"]
    cost = ((totals["input_tokens"] - cached) * prices["input"] + cached * prices["cached_input"]
            + totals["output_tokens"] * prices["output"]) / 1_000_000
    return {
        **totals, "estimated_cost_usd": round(cost, 10) if measured else None,
        "request_attempts": sum(counts.values()), "attempts_by_state": counts,
        "completed_requests": len(responses), "responses_without_usage": len(responses) - len(measured),
        "returned_models": sorted({row["model"] for row in responses}),
        "finish_reasons": dict(Counter(row["finish_reason"] or "unknown" for row in responses)),
    }


def run_evaluation(output, rows, manifest, client_factory, api_error, *,
                   allow_uncertain=False, sleep=time.sleep):
    output.mkdir(parents=True, exist_ok=True)
    with closing(shared.open_checkpoint(output / "checkpoint.sqlite3", manifest)) as connection:
        open_api_tables(connection)
        started_at, = connection.execute("SELECT started_at FROM api_run").fetchone()
        completed = {row[0] for row in connection.execute("SELECT row_index FROM responses")}
        api_completed = {row[0] for row in connection.execute("SELECT row_index FROM api_responses")}
        if completed != api_completed or any(i < 0 or i >= len(rows) for i in completed):
            raise RuntimeError("Checkpoint rows and API response metadata disagree.")
        try:
            if len(completed) < len(rows):
                with client_factory() as client:
                    for index, row in enumerate(rows):
                        if index in completed:
                            continue
                        prompt = shared.PROMPT.format(question=row["text"])
                        request_row(connection, client, api_error, index, prompt, manifest["parameters"],
                                    allow_uncertain, sleep)
                        if (index + 1) % 100 == 0:
                            print(f"Saved {index + 1}/{len(rows)} responses", flush=True)
            run = {**manifest, "started_at_utc": started_at,
                   "usage": summarize_usage(connection, manifest["prices_usd_per_million"])}
            if not shared.export_results(connection, rows, output, run):
                raise RuntimeError("Run is incomplete. Resume with the same arguments.")
        finally:
            usage = summarize_usage(connection, manifest["prices_usd_per_million"])
            metadata = {
                **manifest, "started_at_utc": started_at, "updated_at_utc": utc_now(), "usage": usage,
                "all_rows_received": usage["completed_requests"] == len(rows),
            }
            shared.atomic_write(output / "run_metadata.json", lambda handle: json.dump(metadata, handle, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=shared.DATASETS)
    parser.add_argument("--model", choices=PRICES, default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="run_01")
    parser.add_argument("--limit", type=int, help="Use the first N fixed dataset rows in an isolated run")
    parser.add_argument("--retry-uncertain", action="store_true",
                        help="Retry a row whose previous request may already have been billed")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with a letter or digit and contain only letters, digits, _ or -")
    rows, dataset_hash = shared.load_dataset(ROOT, args.dataset)
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            parser.error(f"--limit must be between 1 and {len(rows)}")
        rows = rows[:args.limit]

    import openai
    from dotenv import dotenv_values

    # Keep SDK debug logging off even if the parent environment enables it.
    logging.getLogger("openai").setLevel(logging.CRITICAL + 1)
    logging.getLogger("httpx").setLevel(logging.CRITICAL + 1)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL + 1)
    output = output_path(ROOT, args.model, args.dataset, args.run_id, args.limit)
    manifest = {
        "dataset": args.dataset, "dataset_sha256": dataset_hash, "row_count": len(rows),
        "limit": args.limit, "run_id": args.run_id, "model_id": args.model,
        "endpoint": "POST /v1/chat/completions", "openai_sdk_version": openai.__version__,
        "parameters": request_parameters(args.model), "prompt_version": "original_frozen_zero_shot",
        "prompt_sha256": hashlib.sha256(shared.PROMPT.encode("utf-8")).hexdigest(),
        "prompt": shared.PROMPT, "invalid_output_policy": shared.INVALID_POLICY,
        "prices_usd_per_million": PRICES[args.model], "pricing_verified_date": "2026-11-22",
    }

    def make_client():
        key = api_key(ROOT, dotenv_values)
        try:
            return openai.OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=0)
        except Exception:
            raise RuntimeError("OpenAI client initialization failed.") from None

    try:
        run_evaluation(output, rows, manifest, make_client, openai.APIError,
                       allow_uncertain=args.retry_uncertain)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same arguments.", file=sys.stderr)
        return 130
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
