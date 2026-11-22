# OpenAI API evaluation on Windows

Open the CDS1092 NLP project root in VS Code. In its PowerShell terminal, create a separate Python environment and install the API dependencies.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
```

In VS Code, create a project-root file named `.env`. Put your project API key in it as `OPENAI_API_KEY=<your key>`. Edit it in VS Code to keep the key out of shell history. The evaluator prefers an existing `OPENAI_API_KEY` environment variable. Git ignores `.env` and `.env.*`. Use an API project with access and billing enabled for the selected model.

## First 10-question test

This command sends paid requests. Run it from the project root when ready.

```powershell
.\.venv\Scripts\python.exe src/33_evaluate_openai.py --dataset umwp --model gpt-6.1-terra --limit 10 --run-id run_01
```

Use `--dataset sum` for SUM. A limited run evaluates the first N rows in saved order and is a pipeline check. It writes to `results/api/openai/<model>/<dataset>/limit_<N>/<run-id>/`. Full runs omit `--limit` and use `results/api/openai/<model>/<dataset>/<run-id>/`. Keep `--limit 10` for the first test. The output includes `predictions.csv`, `metrics.json`, `confusion_matrix.csv`, `run_metadata.json`, and `checkpoint.sqlite3`.

## Protocol and cost

The evaluator sends the original frozen user message to Chat Completions with reasoning effort `none`, temperature `0`, Standard service tier, and a 16-token completion budget. That API budget includes any reasoning tokens. Temperature zero requests deterministic behavior where supported. The hosted tokenizer, chat serialization, serving backend, and model aliases can affect reproducibility. The API response supplies text and usage counts, so `generated_token_ids` is JSON null. A response terminated with `finish_reason=length` is recorded as hitting the cap. The shared parser trims surrounding whitespace and scores anything other than the two exact labels as INVALID. The checkpoint records the returned model identifier and finish reason for each completed row. [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) · [GPT-6 parameter compatibility](https://developers.openai.com/api/docs/guides/latest-model)

Both options are listed in the official catalog. Standard USD rates per million text tokens on 2026-11-22 are shown below. Access depends on the API project. [GPT-6.1 Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra) · [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)

| Model | Input | Cached input | Output |
|---|---:|---:|---:|
| `gpt-6.1-terra` | $1.00 | $0.05 | $5.00 |
| `gpt-6-luna` | $0.10 | $0.01 | $0.50 |

Requests are sequential, with at most three attempts per row per invocation. The SDK's automatic retries are disabled. The evaluator retries temporary HTTP 429 and 503 responses with bounded backoff and respects a usable `Retry-After` header. Exhausted quota stops the run. The usage report covers recorded responses and estimates text-token cost from their reported usage. Unreported usage, uncertain attempts, or cache-write charges can make actual billing differ. As an illustration, ten Terra requests averaging 200 input tokens each and using all 16 output tokens would cost about $0.0028 at uncached Standard rates. [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)

## Resume

Rerun the same command to resume. SQLite commits each recorded response with its usage, and completed rows are skipped. If a transport failure or interruption leaves a request pending, the evaluator stops at that row on resume. Check the provider usage or logs before deciding whether to retry. Add `--retry-uncertain` only if you accept that the row may already have been billed. The attempt log counts requests scheduled for sending, so an interruption immediately before the send can leave a pending entry even if the request never reached OpenAI.
