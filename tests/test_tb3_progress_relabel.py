import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from relabel_tb3_progress_dataset import relabel_shard


def _add_bytes(tar, name, payload):
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def test_relabel_shard_preserves_rewards_and_adds_progress():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "episode.tar"
        dst = Path(tmp) / "out" / "episode.tar"
        state0 = {
            "episode_meta": {"episode_id": "ep", "step": 0, "done": False},
            "agents": [
                {"id": 0, "dist_to_goal": 1.0, "reached": False},
                {"id": 1, "dist_to_goal": 1.0, "reached": False},
            ],
            "reward": 123.0,
        }
        state1 = {
            "episode_meta": {"episode_id": "ep", "step": 1, "done": True, "success": True},
            "agents": [
                {"id": 0, "dist_to_goal": 0.05, "reached": True},
                {"id": 1, "dist_to_goal": 0.05, "reached": True},
            ],
            "reward": 456.0,
        }
        dist = np.eye(2, dtype=np.float32)

        with tarfile.open(src, "w") as tar:
            for idx, state, reward in ((0, state0, 123.0), (1, state1, 456.0)):
                prefix = f"ep_step{idx:04d}"
                _add_bytes(tar, f"{prefix}.state.json", json.dumps(state).encode("utf-8"))
                _add_bytes(tar, f"{prefix}.reward.json", json.dumps(reward).encode("utf-8"))
                _add_bytes(tar, f"{prefix}.dist.npy", _npy_bytes(dist))

        count = relabel_shard(src, dst, goal_radius_m=0.1, proximity_threshold=0.2)
        assert count == 2

        with tarfile.open(dst, "r") as tar:
            names = tar.getnames()
            assert names.index("ep_step0000.progress.json") > names.index("ep_step0000.state.json")
            reward0 = json.load(tar.extractfile("ep_step0000.reward.json"))
            reward1 = json.load(tar.extractfile("ep_step0001.reward.json"))
            progress0 = json.load(tar.extractfile("ep_step0000.progress.json"))
            progress1 = json.load(tar.extractfile("ep_step0001.progress.json"))

        assert reward0 == 123.0
        assert reward1 == 456.0
        assert progress0["schema"] == "tb3_progress_v2"
        assert progress0["target"] == 0.0
        assert progress0["episode_success"] is True
        assert progress0["initial_distances"] == [1.0, 1.0]
        assert progress1["target"] == 1.0
