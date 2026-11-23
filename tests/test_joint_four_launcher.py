"""Exercise Slurm orchestration locally with a simulated srun allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = ("agents3456_shared", "agents3456_task", "agents35_shared", "agents35_task")

FAKE_PYTHON = r'''
import argparse,json,os,sys,time
from pathlib import Path
if sys.argv[1] == "-c":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
p=argparse.ArgumentParser()
p.add_argument("--output-dir"); p.add_argument("--value-head-mode")
p.add_argument("--num-agents",nargs="+",type=int)
a,_=p.parse_known_args(sys.argv[2:])
root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
record=dict(args=sys.argv[2:],agents=a.num_agents,head=a.value_head_mode,
            num_gpus=os.environ["NUM_GPUS"],visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
            revision=os.environ.get("MODEL_REVISION"),workers=os.environ.get("NUM_WORKERS"))
(root/"invocation.json").write_text(json.dumps(record))
if "--cache-only" in sys.argv:
    (root/"cache_report.json").write_text(json.dumps(dict(model_config=dict(model_revision="pinned-test"))))
    (root/"split_manifest.json").write_text("{}")
elif "--preflight-only" in sys.argv:
    (root/"selection_report.json").write_text(json.dumps(record))
else:
    (root.parent/f"started_{root.name}").touch()
    until=time.monotonic()+10
    while len(list(root.parent.glob("started_*")))<4:
        if time.monotonic()>until: raise SystemExit("The four cases did not start concurrently")
        time.sleep(.02)
    if os.environ.get("FAKE_FAIL_CASE")==root.name: raise SystemExit(23)
'''

FAKE_SRUN = r'''
import json,os,subprocess,sys
from pathlib import Path
args=sys.argv[1:];start=args.index("env");flags=args[:start]
env=dict(os.environ);i=start+1
while "=" in args[i]:
    key,value=args[i].split("=",1);env[key]=value;i+=1
case=Path(env["RUN_DIR"]).name
devices=dict(agents3456_shared="8",agents3456_task="11",agents35_shared="4",agents35_task="6")
env["CUDA_VISIBLE_DEVICES"]=devices.get(case,"8,11,4,6")
with open(os.environ["MOCK_SRUN_LOG"],"a") as stream:
    stream.write(json.dumps(dict(flags=flags,case=case,num_gpus=env["NUM_GPUS"]))+"\n")
output=next(x.split("=",1)[1] for x in flags if x.startswith("--output="))
error=next(x.split("=",1)[1] for x in flags if x.startswith("--error="))
with open(output,"w") as out,open(error,"w") as err:
    raise SystemExit(subprocess.run(args[i:],env=env,stdout=out,stderr=err).returncode)
'''


class FourCaseLauncherTests(unittest.TestCase):
    def run_suite(self, root, fail_case="", check=False):
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name, source in (("python", FAKE_PYTHON), ("srun", FAKE_SRUN)):
            script = bin_dir / name
            script.write_text(f"#!{sys.executable}\n" + source)
            script.chmod(0o755)
        env = dict(os.environ, PROJECT_ROOT=str(ROOT), PYTHON_BIN=str(bin_dir / "python"),
                   PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], GPU_MONITOR="0", NUM_WORKERS="1",
                   SLURM_JOB_ID="test", SLURM_NTASKS="4", SLURM_CPUS_PER_TASK="2", SLURM_MEM_PER_NODE="196608",
                   RUN_ROOT=str(root / "runs"), MOCK_SRUN_LOG=str(root / "srun.jsonl"), FAKE_FAIL_CASE=fail_case)
        env.pop("RUN_DIR", None)
        result = subprocess.run(["bash", str(ROOT / "train_joint_four_palmetto.sbatch"), *(["--check"] if check else [])],
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return result

    def test_four_cases_overlap_and_receive_one_gpu_each_after_shared_caching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.run_suite(root)
            self.assertEqual(result.returncode, 0, result.stdout)
            calls = [json.loads(line) for line in (root / "srun.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 5)
            self.assertEqual(calls[0]["case"], "cache_build")
            self.assertIn("--gpus-per-task=a100:4", calls[0]["flags"])
            self.assertIn("--cpus-per-task=8", calls[0]["flags"])
            visible = set()
            for call in calls[1:]:
                self.assertIn("--gpus-per-task=a100:1", call["flags"])
                self.assertIn("--cpus-per-task=2", call["flags"])
                self.assertIn("--mem=40G", call["flags"])
                self.assertIn("--exact", call["flags"])
                self.assertEqual(call["num_gpus"], "1")
                record = json.loads((root / "runs" / call["case"] / "invocation.json").read_text())
                self.assertEqual(record["agents"], [3, 5] if call["case"].startswith("agents35_") else [3, 4, 5, 6])
                self.assertEqual(record["head"], call["case"].rsplit("_", 1)[1])
                self.assertEqual(record["revision"], "pinned-test")
                self.assertIn("--split-manifest", record["args"])
                self.assertEqual(record["args"][record["args"].index("--batch-size") + 1], "4096")
                self.assertEqual(record["args"][record["args"].index("--clip-len") + 1], "8")
                self.assertEqual(record["workers"], "1")
                visible.add(record["visible"])
            self.assertEqual(len(visible), 4)
            self.assertTrue(all((root / "runs" / f"{case}.exitcode").read_text().strip() == "0" for case in CASES))

    def test_failure_is_reported_while_other_cases_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.run_suite(root, fail_case="agents35_task")
            self.assertEqual(result.returncode, 1, result.stdout)
            for case in CASES:
                expected = "23" if case == "agents35_task" else "0"
                self.assertEqual((root / "runs" / f"{case}.exitcode").read_text().strip(), expected)

    def test_check_needs_no_srun_or_cuda_and_selects_all_four_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.run_suite(root, check=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertFalse((root / "srun.jsonl").exists())
            for case in CASES:
                self.assertTrue((root / "runs/preflight" / case / "selection_report.json").exists())


if __name__ == "__main__":
    unittest.main()
