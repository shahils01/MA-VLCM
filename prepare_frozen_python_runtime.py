"""Stage the historical formation environment without altering current sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
ACTOR_COMMIT = "a70d2f5c5d97adf358db823ca7f7c37ff9a11729"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", required=True, type=Path)
    a = p.parse_args()
    if a.destination.exists():
        p.error("Choose a new snapshot destination")
    vlcm, marl = a.destination / "MA-VLCM", a.destination / "MARL_data_collection"
    vlcm.mkdir(parents=True)
    marl.mkdir()
    files = ["train_mappo.py", "python_mappo.py", "model.py", "gat.py", "vlcm_mappo_timing.py",
             "frozen_critic_audit.py", "benchmark_frozen_python.py", "train_frozen_python_value.sh",
             "train_pure_frozen_formation_palmetto.sbatch", "pure_frozen_formation_worker.sh"]
    for name in files:
        shutil.copy2(ROOT / name, vlcm / name)
    environment = ROOT / "outputs/eval_rollouts/assets/formation/source/python_env/RL_bound_overwatch"
    shutil.copytree(environment, marl / "RL_bound_overwatch", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The dataset snapshot contains empty package markers; export only the
    # formation environment without importing unrelated simulator dependencies.
    (marl / "RL_bound_overwatch/envs/__init__.py").write_text(
        "from .FormationControlEnv import FormationControl\n")
    archive = subprocess.run(["git", "-C", str(ROOT.parent / "MARL_data_collection"),
                              "archive", ACTOR_COMMIT, "on-policy/onpolicy/__init__.py",
                              "on-policy/onpolicy/config.py", "on-policy/onpolicy/algorithms",
                              "on-policy/onpolicy/utils"], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isfile():
                path = marl / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.extractfile(member).read())
    # Native actor/config only; the original eager initializer imports runners
    # and simulator packages that this actor-only snapshot does not need.
    (marl / "on-policy/onpolicy/__init__.py").write_text("")
    expected = {
        "FormationControlEnv.py": "d5b0d5c8bdc9493468659c24b7a67e700f4993e73998a3f25352a342f341c671",
        "formation_renderer.py": "fc43ab815f482e519145f9d3ea1d5b3c2cace9a87262151ea295d4b5c54011a6",
        "assets/background_image_real.png": "3715093b4544e36bc6b9f76b6716a24201071f223dca75a1bae979c0b2076347",
    }
    for name, checksum in expected.items():
        if hashlib.sha256((marl / "RL_bound_overwatch/envs" / name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Archived environment checksum mismatch: {name}")
    manifest = {str(path.relative_to(a.destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(a.destination.rglob("*")) if path.is_file()}
    (a.destination / "manifest.json").write_text(json.dumps(dict(actor_source_commit=ACTOR_COMMIT,
                                                                 files=manifest), indent=2) + "\n")
    print(a.destination)


if __name__ == "__main__":
    main()
