"""Inspect or explicitly submit the fixed ten-job Mercury sequence.

Default: read-only plan. --submit is the only submission action; --status and
--cancel operate on this batch's recorded/tagged jobs. No training runs locally.
A permanent, exclusively created batch directory prevents duplicate submission,
including after a partial submission, cancellation or completed sequence.
"""

import argparse
import getpass
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/overnight_runs.json"
TRAINER = ROOT / "src/22_train_encoder.py"
JOB_SCRIPT = ROOT / "slurm/train_encoder.sh"
STATE_ROOT = ROOT / "slurm_batches"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_plan():
    plan = json.loads(MATRIX.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", plan["batch_id"]):
        raise ValueError("Unsafe batch ID.")
    expected = [("deberta", f"run_{i:02d}") for i in range(3, 7)]
    expected += [("modernbert", f"run_{i:02d}") for i in range(1, 7)]
    actual = [(run["model_family"], run["run_name"]) for run in plan["runs"]]
    if actual != expected:
        raise ValueError("Expected the fixed four-DeBERTa/six-ModernBERT sequence.")
    return plan


def paths(plan):
    directory = STATE_ROOT / plan["batch_id"]
    # Stable tag permits recovery of jobs accepted by sbatch just before a crash
    # prevented their ID being recorded. Never resubmit an uncertain batch.
    tag = plan["batch_id"] + "-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:10]
    return directory, tag


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def queued_ids(tag):
    output = command(["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%k"])
    ids = []
    for line in output.splitlines():
        job, separator, comment = line.partition("|")
        if separator and comment.strip() == tag:
            job = job.strip()
            if not job.isdigit():
                raise ValueError(f"Unexpected job ID in tagged queue: {job}")
            ids.append(job)
    return ids


def write_state(directory, state):
    temporary = directory / "submission.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "submission.json")


def submit(plan):
    for tool in ("sbatch", "scontrol", "squeue", "scancel"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Required Mercury command missing: {tool}")
    directory, tag = paths(plan)
    if directory.exists():
        raise FileExistsError(f"Batch already reserved: {directory}. Inspect --status; do not resubmit.")
    if queued_ids(tag):
        raise RuntimeError("Tagged batch jobs already exist in SLURM; refusing duplicate submission.")
    for run in plan["runs"]:
        for parent in ("models", "results"):
            path = ROOT / parent / run["model_family"] / run["run_name"]
            if path.exists():
                raise FileExistsError(f"Experiment path already exists: {path}")
    for file in (MATRIX, TRAINER, JOB_SCRIPT,
                 ROOT / "data/processed/sum/train.csv", ROOT / "data/processed/sum/validation.csv"):
        if not file.is_file():
            raise FileNotFoundError(file)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)  # Permanent reservation, even on failure.
    (directory / "logs").mkdir()
    (directory / "status").mkdir()
    # Freeze exactly the submitted trainer and matrix, rather than allowing a
    # later Git pull to alter queued jobs. Job script is copied by sbatch itself.
    snapshot = directory / "snapshot"
    (snapshot / "src").mkdir(parents=True)
    (snapshot / "configs").mkdir()
    shutil.copyfile(TRAINER, snapshot / "src/22_train_encoder.py")
    shutil.copyfile(MATRIX, snapshot / "configs/overnight_runs.json")
    state = {"batch_id": plan["batch_id"], "tag": tag, "root": str(ROOT),
             "created_utc": now(), "phase": "submitting_held", "jobs": [],
             "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (MATRIX, TRAINER, JOB_SCRIPT)}}
    write_state(directory, state)
    previous = None
    try:
        for run in plan["runs"]:
            label = run["model_family"] + "_" + run["run_name"]
            args = ["sbatch", "--parsable", "--hold", "--comment", tag,
                    "--job-name", label, "--chdir", str(ROOT),
                    "--output", str(directory / "logs" / f"{label}-%j.out"),
                    "--error", str(directory / "logs" / f"{label}-%j.err"),
                    "--open-mode=append"]
            if previous:
                args += ["--dependency", f"afterany:{previous}"]
            args += [str(JOB_SCRIPT), str(ROOT), run["model_family"],
                     run["run_name"], str(directory / "status"), str(snapshot)]
            response = command(args)
            job_id = response.split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unrecognized sbatch response: {response}")
            state["jobs"].append({**run, "job_id": job_id,
                                  "afterany": previous, "submitted_utc": now()})
            write_state(directory, state)
            print(f"Queued held {label}: {job_id}; afterany={previous}", flush=True)
            previous = job_id
        state["phase"] = "releasing"
        write_state(directory, state)
        for job in state["jobs"]:
            command(["scontrol", "release", job["job_id"]])
        state["phase"] = "submitted"
        state["released_utc"] = now()
        write_state(directory, state)
    except BaseException as exc:
        state["phase"] = "submission_or_release_incomplete"
        state["error"] = str(exc)
        write_state(directory, state)
        print("Submission incomplete. Some jobs may be held or released. "
              "Use --status and --cancel; do not delete the reservation or resubmit.", flush=True)
        raise
    print(f"Submitted ten sequential jobs. State/logs: {directory}")


def inspect_or_cancel(plan, cancel=False):
    directory, tag = paths(plan)
    state_file = directory / "submission.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else None
    active = queued_ids(tag)
    if state:
        print(json.dumps(state, indent=2))
    print("Active tagged job IDs:", ",".join(active) or "none")
    if cancel:
        if active:
            command(["scancel", *active])
            print("Cancellation requested for this batch's active jobs only.")
        if state:
            state["cancellation_requested_utc"] = now()
            write_state(directory, state)
        print("Batch reservation retained to prevent accidental duplicate submission.")
    elif state and state["jobs"] and shutil.which("sacct"):
        ids = ",".join(job["job_id"] for job in state["jobs"])
        print(command(["sacct", "-j", ids, "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    plan = read_plan()
    if args.submit:
        submit(plan)
    elif args.status or args.cancel:
        inspect_or_cancel(plan, cancel=args.cancel)
    else:
        for run in plan["runs"]:
            print(run["model_family"], run["run_name"],
                  f"LR={run['learning_rate']}", f"max_epochs={run['max_epochs']}")
        print("Read-only plan. Explicit submission: python src/23_submit_overnight.py --submit")


if __name__ == "__main__":
    main()
