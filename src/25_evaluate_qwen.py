"""Run the complete zero-shot Qwen evaluation on one fixed benchmark."""

import argparse
from contextlib import closing
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_PATH = ROOT / "models/Qwen3.6-27B"
LABELS = ("ANSWERABLE", "UNANSWERABLE")
INVALID = "INVALID"
DATASETS = {
    "umwp": (5200, ("id", "text", "label", "category", "source"), "id",
             "92df3bed2d3cee139c6ddbca2673fca0a1687aa9f90b3cac27b2627605d90ecb"),
    "sum": (10944, ("pair_id", "text", "label"), "pair_id",
            "c1e4bac1e1d1f27ab8a6408ac68ae710faf521822bff49a1c5958b027540b1f0"),
}
PROMPT = """Determine whether the following math problem can be answered using the information provided.

Respond with exactly one label:
ANSWERABLE or UNANSWERABLE.

Do not solve the problem or provide an explanation.

Math problem:
{question}

Label:"""
MAX_NEW_TOKENS = 16
INVALID_POLICY = (
    "After trimming surrounding whitespace, only exact uppercase labels are valid. "
    "INVALID is incorrect for accuracy and is a separate confusion-matrix column. "
    "INVALID on a true UNANSWERABLE row counts as a missed positive for recall and F1. "
    "Precision uses all predicted UNANSWERABLE rows. Zero denominators yield zero."
)


def load_dataset(root, dataset):
    expected, fields, identifier, expected_digest = DATASETS[dataset]
    path = root / "data" / "processed" / dataset / "test.csv"
    contents = path.read_bytes()
    # Git checkouts can differ in CSV line endings. Validate against the Git LF form.
    digest = hashlib.sha256(contents.replace(b"\r\n", b"\n")).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"Dataset hash differs from the fixed encoder benchmark: {path}")
    reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    if tuple(reader.fieldnames or ()) != fields or len(rows) != expected:
        raise ValueError(f"Unexpected schema or row count: {path}")
    if any(not row[identifier].strip() or not row["text"].strip() for row in rows):
        raise ValueError(f"Empty identifier or question: {path}")
    if any(sum(row["label"] == label for row in rows) != expected // 2 for label in LABELS):
        raise ValueError(f"Unexpected labels or class counts: {path}")
    return rows, digest


def parse_label(raw):
    label = raw.strip()
    return label if label in LABELS else INVALID


def score(predictions):
    columns = (*LABELS, INVALID)
    matrix = [[0] * len(columns) for _ in LABELS]
    for row in predictions:
        matrix[LABELS.index(row["true_label"])][columns.index(row["prediction"])] += 1
    total = sum(map(sum, matrix))
    tn, fp, invalid_negative = matrix[0]
    fn, tp, invalid_positive = matrix[1]
    missed = fn + invalid_positive

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "num_examples": total,
        "num_invalid": invalid_negative + invalid_positive,
        "invalid_rate": ratio(invalid_negative + invalid_positive, total),
        "accuracy": ratio(tn + tp, total),
        "precision_unanswerable": ratio(tp, tp + fp),
        "recall_unanswerable": ratio(tp, tp + missed),
        "f1_unanswerable": ratio(2 * tp, 2 * tp + fp + missed),
        "positive_class": LABELS[1],
        "invalid_output_policy": INVALID_POLICY,
        "confusion_matrix": matrix,
        "confusion_matrix_rows": LABELS,
        "confusion_matrix_columns": columns,
    }


def file_identity(stat):
    # Linux ctime changes even when a writer restores the original mtime.
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def hash_model_file(path, identity):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if file_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError(f"Model file changed during verification: {path}")
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_path, cache_path):
    """Hash model contents, reusing shard digests only for unchanged local files."""
    files = sorted(path for path in model_path.rglob("*")
                   if path.is_file() and path.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"})
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"No local safetensors weights found in {model_path}")

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        cached = {}
    if not isinstance(cached, dict):
        cached = {}

    shard_cache = {}
    snapshots = {}
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(model_path).as_posix()
        identity = snapshots[path] = file_identity(path.stat())
        is_shard = path.suffix == ".safetensors"
        entry = cached.get(name) if is_shard else None
        if (isinstance(entry, dict) and entry.get("identity") == identity
                and isinstance(entry.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            checksum = entry["sha256"]
        else:
            checksum = hash_model_file(path, identity)
        if is_shard:
            shard_cache[name] = {"identity": identity, "sha256": checksum}
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(checksum))

    for path, identity in snapshots.items():
        if file_identity(path.stat()) != identity:
            raise RuntimeError(f"Model file changed during verification: {path}")
    if shard_cache != cached:
        atomic_write(cache_path, lambda handle: json.dump(shard_cache, handle, indent=2))
    return digest.hexdigest()


def open_checkpoint(path, manifest):
    connection = sqlite3.connect(path, timeout=1)
    try:
        # Rollback journaling and one exclusive writer also work without WAL shared memory.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute("PRAGMA synchronous=FULL")
        with connection:
            connection.execute("BEGIN EXCLUSIVE")
            connection.execute("CREATE TABLE IF NOT EXISTS manifest (config TEXT NOT NULL)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    row_index INTEGER PRIMARY KEY,
                    raw_response TEXT NOT NULL,
                    token_ids TEXT NOT NULL,
                    hit_token_limit INTEGER NOT NULL
                )
            """)
            config = json.dumps(manifest, sort_keys=True)
            existing = connection.execute("SELECT config FROM manifest").fetchone()
            if existing is None:
                connection.execute("INSERT INTO manifest VALUES (?)", (config,))
            elif existing[0] != config:
                raise ValueError(f"Run settings differ from the saved checkpoint: {path}")
    except BaseException:
        connection.close()
        raise
    return connection


def evaluate(connection, rows, generate, stopped):
    completed = {index for (index,) in connection.execute("SELECT row_index FROM responses")}
    if any(index < 0 or index >= len(rows) for index in completed):
        raise ValueError("Checkpoint contains a row outside the dataset")
    for index, row in enumerate(rows):
        if stopped():
            break
        if index in completed:
            continue
        raw, ids, hit_limit = generate(PROMPT.format(question=row["text"]))
        with connection:
            connection.execute("INSERT INTO responses VALUES (?, ?, ?, ?)",
                               (index, raw, json.dumps(ids), int(hit_limit)))
        if (index + 1) % 100 == 0:
            print(f"Saved {index + 1}/{len(rows)} responses", flush=True)


def atomic_write(path, write):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            write(handle)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def export_results(connection, rows, output, manifest):
    records = connection.execute("SELECT * FROM responses ORDER BY row_index").fetchall()
    if [record[0] for record in records] != list(range(len(rows))):
        return False
    predictions = []
    for index, raw, ids, hit_limit in records:
        row = dict(rows[index])
        row["true_label"] = row.pop("label")
        label = parse_label(raw)
        predictions.append({
            "row_index": index, **row, "raw_response": raw, "prediction": label,
            "valid": label != INVALID, "generated_token_ids": ids, "hit_token_limit": bool(hit_limit),
        })

    def write_predictions(handle):
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    def write_matrix(handle):
        writer = csv.writer(handle)
        writer.writerow(("true_label", *metrics["confusion_matrix_columns"]))
        writer.writerows((label, *counts) for label, counts in zip(LABELS, metrics["confusion_matrix"]))

    metrics = score(predictions)
    atomic_write(output / "predictions.csv", write_predictions)
    atomic_write(output / "confusion_matrix.csv", write_matrix)
    atomic_write(output / "metrics.json",
                 lambda handle: json.dump({**metrics, "run": manifest}, handle, indent=2))
    return True


def load_generator(torch, transformers):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    # The Qwen3.6-27B checkpoint declares this architecture in config.json.
    # Transformers 5.22.2 implements it natively and accepts text-only inputs.
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        use_safetensors=True,
    ).eval()
    if (getattr(model, "is_quantized", False)
            or any(p.dtype != torch.bfloat16 or p.device != torch.device("cuda:0")
                   for p in model.parameters() if p.is_floating_point())):
        raise RuntimeError("Expected unquantized BF16 parameters on cuda:0")
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if not eos_ids:
        raise ValueError("Model generation configuration has no EOS token")
    generation = transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )

    @torch.inference_mode()
    def generate(prompt):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to("cuda:0")
        input_length = inputs["input_ids"].shape[1]
        ids = model.generate(**inputs, generation_config=generation)[0, input_length:].tolist()
        terminated = bool(ids and ids[-1] in eos_ids)
        content = ids[:-1] if terminated else ids
        raw = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return raw, ids, len(ids) == MAX_NEW_TOKENS and not terminated

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with an alphanumeric character and use only letters, digits, _ or -")

    rows, dataset_hash = load_dataset(ROOT, args.dataset)
    import torch
    import transformers

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if ("A100" not in gpu.name or "PCIE" not in gpu.name.upper()
            or gpu.total_memory < 75 * 1024**3 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"Expected BF16 on an A100 80 GB PCIe, found {gpu.name}")

    output = ROOT / "results/llm/qwen3.6-27b" / args.dataset / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset, "dataset_sha256": dataset_hash, "row_count": len(rows),
        "run_id": args.run_id, "model_id": MODEL_ID, "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": model_fingerprint(MODEL_PATH, output / "model_fingerprint_cache.json"),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "prompt": PROMPT, "chat_template": "model default, user message only, enable_thinking=False",
        "dtype": "bfloat16", "quantization": None, "gpu": gpu.name,
        "requested_temperature": 0, "do_sample": False, "num_beams": 1,
        "max_new_tokens": MAX_NEW_TOKENS, "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "invalid_output_policy": INVALID_POLICY,
    }
    received_signal = 0

    def stop(signum, _frame):
        nonlocal received_signal
        received_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with closing(open_checkpoint(output / "checkpoint.sqlite3", manifest)) as connection:
        if not export_results(connection, rows, output, manifest):
            if not received_signal:
                generate = load_generator(torch, transformers)
                evaluate(connection, rows, generate, lambda: bool(received_signal))
            if not export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
