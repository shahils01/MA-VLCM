"""Check two-case LoRA Slurm orchestration with simulated compute steps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_joint_four_launcher import FAKE_PYTHON, FAKE_SRUN

ROOT = Path(__file__).resolve().parents[1]


class LoraLauncherTests(unittest.TestCase):
    def test_two_live_lora_cases_after_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindir = root / "bin"
            bindir.mkdir()
            source = FAKE_PYTHON.replace("<4:", "<2:")
            fake_srun = FAKE_SRUN.replace('case=Path(env["RUN_DIR"]).name', 'case=Path(env["RUN_DIR"]).name\nif case == "gpu_smoke": raise SystemExit(0)')
            for name, code in (("python", source), ("srun", fake_srun)):
                executable = bindir / name
                executable.write_text(f"#!{sys.executable}\n" + code)
                executable.chmod(0o755)
            env = dict(os.environ, PROJECT_ROOT=str(ROOT), PYTHON_BIN=str(bindir / "python"),
                       PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                       SLURM_JOB_ID="test", SLURM_CPUS_PER_TASK="2", RUN_ROOT=str(root / "runs"),
                       MOCK_SRUN_LOG=str(root / "srun.jsonl"), WANDB_ENABLED="0")
            env.pop("RUN_DIR", None)
            result = subprocess.run(["bash", str(ROOT / "train_joint_lora_two_palmetto.sbatch")],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / "srun.jsonl").read_text().splitlines()]
            self.assertEqual({c["case"] for c in calls}, {"agents3456_shared", "agents3456_task"})
            self.assertTrue((root / "runs/preflight/selection_report.json").exists())
            for call in calls:
                self.assertEqual(call["num_gpus"], "1")
                self.assertIn("--gpus-per-task=a100:1", call["flags"])
                record = json.loads((root / "runs" / call["case"] / "invocation.json").read_text())
                args = record["args"]
                self.assertEqual(args[args.index("--backbone-mode") + 1], "lora")
                self.assertEqual(args[-5:], ["--experiment", "joint_without_obstacles", "--require-cuda", "--expected-visible-gpus", "1"])
                self.assertEqual(args[args.index("--batch-size") + 1], "1")
                self.assertNotIn("--cache-only", args)
                self.assertEqual(record["agents"], [3, 4, 5, 6])

if __name__ == "__main__":
    unittest.main()
