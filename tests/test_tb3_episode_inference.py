import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from tools.evaluate_tb3_episode_progress import (
    find_latest_checkpoint,
    inspect_episode_shard,
    write_episode_csv,
)


def _add_bytes(archive, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def test_latest_checkpoint_uses_newest_run_then_highest_epoch(tmp_path):
    older_last = tmp_path / "tb3_isaac_qwen3_vl_2b_20260720_120000_epoch_20.pt"
    newer_first = tmp_path / "tb3_isaac_qwen3_vl_2b_20260721_120000_epoch_1.pt"
    newer_last = tmp_path / "tb3_isaac_qwen3_vl_2b_20260721_120000_epoch_7.pt"
    for path in (older_last, newer_first, newer_last):
        path.touch()

    selected = find_latest_checkpoint(tmp_path, "qwen3_vl")
    assert selected == newer_last.resolve()


def test_latest_checkpoint_accepts_legacy_llava_run_prefix(tmp_path):
    first = tmp_path / "tb3_isaac_0.5B_20260720_120000_epoch_1.pt"
    last = tmp_path / "tb3_isaac_0.5B_20260720_120000_epoch_20.pt"
    first.touch()
    last.touch()

    selected = find_latest_checkpoint(tmp_path, "llava")
    assert selected == last.resolve()


def test_inspect_episode_and_write_aligned_csv(tmp_path):
    shard = tmp_path / "agents_03" / "worker_01" / "episode_0042.tar"
    shard.parent.mkdir(parents=True)
    with tarfile.open(shard, "w") as archive:
        for step, progress in ((0, 0.0), (1, 0.4), (2, 1.0)):
            prefix = f"episode_0042_step{step:04d}"
            _add_bytes(archive, f"{prefix}.overhead.png", _png_bytes())
            _add_bytes(
                archive,
                f"{prefix}.progress.json",
                json.dumps(
                    {"team_progress": progress} if step == 1 else {"target": progress}
                ).encode("utf-8"),
            )
            _add_bytes(
                archive,
                f"{prefix}.state.json",
                json.dumps(
                    {"episode_meta": {"episode_id": "episode_0042", "step": step}}
                ).encode("utf-8"),
            )

    info = inspect_episode_shard(shard, load_frames=True)
    assert info["episode_id"] == "episode_0042"
    assert info["steps"] == [0, 1, 2]
    assert info["frame_count"] == 3
    assert info["targets"] == {0: 0.0, 1: 0.4, 2: 1.0}
    assert sorted(info["frames"]) == [0, 1, 2]

    output = tmp_path / "progress.csv"
    results = {
        "llava": [
            {"step": 1, "prediction": 0.3, "target": 0.4},
            {"step": 2, "prediction": 0.9, "target": 1.0},
        ],
        "vjepa2": [
            {"step": 1, "prediction": 0.5, "target": 0.4},
            {"step": 2, "prediction": 0.8, "target": 1.0},
        ],
    }
    write_episode_csv(output, info, results)
    rows = output.read_text().splitlines()
    assert rows[0] == "step,target,prediction_llava,prediction_vjepa2"
    assert rows[1] == "0,0.0,,"
    assert rows[-1] == "2,1.0,0.9,0.8"


def test_explicit_checkpoint_is_accepted_without_filename_convention(tmp_path):
    checkpoint = tmp_path / "legacy_model.pt"
    checkpoint.touch()
    assert find_latest_checkpoint(tmp_path, "llava", checkpoint) == checkpoint.resolve()
