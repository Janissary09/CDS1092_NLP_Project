"""Offline submission tests: all SLURM commands are mocked; no training/imports."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_submit", ROOT / "src/23_submit_overnight.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in ("configs/overnight_runs.json", "src/22_train_encoder.py", "slurm/train_encoder.sh"):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
        (self.root / "data/processed/sum").mkdir(parents=True)
        for name in ("train", "validation"):
            (self.root / f"data/processed/sum/{name}.csv").write_text("test fixture only")
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.multiple(
            workflow, ROOT=self.root, MATRIX=self.root / "configs/overnight_runs.json",
            TRAINER=self.root / "src/22_train_encoder.py", JOB_SCRIPT=self.root / "slurm/train_encoder.sh",
            STATE_ROOT=self.root / "slurm_batches"))
        self.stack.enter_context(patch.object(workflow.shutil, "which", return_value="mock-tool"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.commands = []
        self.submissions = 0

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def mock_command(self, command):
        self.commands.append(command)
        if command[0] == "sbatch":
            self.submissions += 1
            return str(1000 + self.submissions)
        return ""

    def test_chain_hold_snapshot_and_duplicate_guard(self):
        with patch.object(workflow, "command", side_effect=self.mock_command):
            plan = workflow.read_plan()
            workflow.submit(plan)
            submissions = [c for c in self.commands if c[0] == "sbatch"]
            self.assertEqual(len(submissions), 10)
            for index, command in enumerate(submissions):
                self.assertIn("--hold", command)
                self.assertIn("--output", command)
                self.assertIn("--error", command)
                if index:
                    self.assertEqual(command[command.index("--dependency") + 1], f"afterany:{1000 + index}")
                else:
                    self.assertNotIn("--dependency", command)
            releases = [c for c in self.commands if c[0] == "scontrol"]
            self.assertEqual(len(releases), 10)
            self.assertGreater(self.commands.index(releases[0]), self.commands.index(submissions[-1]))
            directory, _ = workflow.paths(plan)
            state = json.loads((directory / "submission.json").read_text())
            self.assertEqual(state["phase"], "submitted")
            self.assertEqual((directory / "snapshot/src/22_train_encoder.py").read_bytes(), workflow.TRAINER.read_bytes())
            with self.assertRaises(FileExistsError):
                workflow.submit(plan)
            self.assertEqual(self.submissions, 10)
            self.assertFalse((self.root / "models").exists())

    def test_existing_experiment_prevents_any_submission(self):
        (self.root / "results/modernbert/run_02").mkdir(parents=True)
        with patch.object(workflow, "command", side_effect=self.mock_command):
            with self.assertRaises(FileExistsError):
                workflow.submit(workflow.read_plan())
        self.assertEqual(self.submissions, 0)

    def test_partial_submission_remains_reserved_without_release(self):
        def fail_third(command):
            if command[0] == "sbatch" and self.submissions == 2:
                raise subprocess.CalledProcessError(1, command)
            return self.mock_command(command)
        with patch.object(workflow, "command", side_effect=fail_third):
            plan = workflow.read_plan()
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.submit(plan)
            directory, _ = workflow.paths(plan)
            state = json.loads((directory / "submission.json").read_text())
            self.assertEqual(state["phase"], "submission_or_release_incomplete")
            self.assertEqual(len(state["jobs"]), 2)
            self.assertFalse(any(c[0] == "scontrol" for c in self.commands))
            with self.assertRaises(FileExistsError):
                workflow.submit(plan)

    def test_cancel_finds_tagged_job_not_yet_recorded(self):
        plan = workflow.read_plan()
        _, tag = workflow.paths(plan)
        def queued(command):
            if command[0] == "squeue":
                return f"9999|{tag}\n8888|unrelated-batch"
            return self.mock_command(command)
        with patch.object(workflow, "command", side_effect=queued):
            workflow.inspect_or_cancel(plan, cancel=True)
        self.assertIn(["scancel", "9999"], self.commands)
        self.assertFalse(any("8888" in command for command in self.commands))


if __name__ == "__main__":
    unittest.main()
