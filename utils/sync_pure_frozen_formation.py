"""Sync the submitted frozen formation run using transfer and scheduler access only."""

# Support both direct execution and imports from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = a.config.parent
    ssh = ["ssh", "-S", cfg["socket"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    while True:
        status = dict(updated_utc=datetime.now(timezone.utc).isoformat())
        finished = False
        try:
            scheduler = subprocess.check_output(ssh + [cfg["host"],
                f"sacct -n -P -X -j {int(cfg['job_id'])} --format=JobID,State,ExitCode"],
                text=True, timeout=60).strip()
            status["scheduler"] = scheduler
            destination = root / "artifacts"
            destination.mkdir(exist_ok=True)
            subprocess.run(["rsync", "-a", "--exclude=code/", "--exclude=wandb/",
                            "-e", shlex.join(ssh), cfg["host"] + ":" + cfg["run_root"] + "/",
                            str(destination) + "/"], check=True, timeout=300)
            state_path = destination / "policy_seed1/status.json"
            status["training"] = json.loads(state_path.read_text()) if state_path.is_file() else None
            rows = [line.split("|") for line in scheduler.splitlines() if line]
            finished = bool(rows) and all(row[1].split()[0] in terminal for row in rows)
            if finished:
                audit_path = destination / "policy_seed1/frozen_audit.json"
                status["final_audit"] = json.loads(audit_path.read_text()) if audit_path.is_file() else None
            status["last_error"] = None
        except Exception as error:
            status["last_error"] = repr(error)
        temporary = root / "monitor_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "monitor_status.json")
        if a.once or finished:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
