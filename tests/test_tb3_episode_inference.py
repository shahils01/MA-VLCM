import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

import tools.evaluate_tb3_episode_progress as episode_evaluator

from tools.evaluate_tb3_episode_progress import (
    find_latest_checkpoint,
    inspect_episode_shard,
    select_episode_shards,
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


def test_latest_checkpoint_selects_highest_qwen_epoch(tmp_path):
    first = tmp_path / "tb3_isaac_qwen3_vl_2b_20260720_120000_epoch_1.pt"
    last = tmp_path / "tb3_isaac_qwen3_vl_2b_20260720_120000_epoch_20.pt"
    first.touch()
    last.touch()

    selected = find_latest_checkpoint(tmp_path, "qwen3_vl")
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
                    {
                        "episode_meta": {
                            "episode_id": "episode_0042",
                            "step": step,
                            "done": step == 2,
                            "success": step == 2,
                            "outcome": "success" if step == 2 else "",
                        }
                    }
                ).encode("utf-8"),
            )

    info = inspect_episode_shard(shard, load_frames=True)
    assert info["episode_id"] == "episode_0042"
    assert info["steps"] == [0, 1, 2]
    assert info["frame_count"] == 3
    assert info["episode_success"] is True
    assert info["targets"] == {0: 0.0, 1: 0.4, 2: 1.0}
    assert sorted(info["frames"]) == [0, 1, 2]

    # Simulate a legacy raw target that differs from the checkpoint loader schema.
    info["targets"][1] = 0.75
    output = tmp_path / "progress.csv"
    results = {
        "qwen3_vl": [
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
    assert rows[0] == "step,target,prediction_qwen3_vl,prediction_vjepa2"
    assert rows[1] == "0,0.0,,"
    assert rows[2] == "1,0.4,0.3,0.5"
    assert rows[-1] == "2,1.0,0.9,0.8"


def test_explicit_checkpoint_is_accepted_without_filename_convention(tmp_path):
    checkpoint = tmp_path / "legacy_model.pt"
    checkpoint.touch()
    assert find_latest_checkpoint(tmp_path, "qwen3_vl", checkpoint) == checkpoint.resolve()


def test_episode_inference_disables_training_source_balancing(monkeypatch, tmp_path):
    captured = {}

    def fake_loader(*args, **kwargs):
        captured.update(kwargs)
        captured['inference_args'] = args[0]
        return []

    monkeypatch.setattr(episode_evaluator, "webdataset_loader", fake_loader)
    model_args = SimpleNamespace(
        balance_tb3_sources=True,
        clip_len=16,
        vl_backend="vjepa2",
        temporal_consistency_loss_weight=1.0,
        conditional_lipschitz_loss_weight=1.0,
    )
    episode = {
        "path": str(tmp_path / "episode.tar"),
        "source_id": "episode",
        "steps": [],
    }

    rows = episode_evaluator.infer_episode(
        model=None,
        model_args=model_args,
        model_dtype=torch.float32,
        episode=episode,
        device=torch.device("cpu"),
        clip_stride=1,
    )

    assert rows == []
    assert captured["shards"] == episode["path"]
    assert captured["balance_tb3_sources"] is False
    assert captured['inference_args'].temporal_consistency_loss_weight == 0.0
    assert captured['inference_args'].conditional_lipschitz_loss_weight == 0.0
    assert model_args.temporal_consistency_loss_weight == 1.0
    assert model_args.conditional_lipschitz_loss_weight == 1.0


def test_episode_selection_includes_requested_failures(monkeypatch, tmp_path):
    candidates = [tmp_path / f"episode_{index}.tar" for index in range(7)]

    monkeypatch.setattr(
        episode_evaluator, "_expand_local_shards", lambda dataset: candidates
    )
    monkeypatch.setattr(
        episode_evaluator,
        "split_shards",
        lambda shards, val_split, split_seed: ([], shards),
    )

    def fake_inspect(path, load_frames=False):
        index = int(path.stem.rsplit("_", 1)[1])
        return {
            "path": str(path),
            "source_id": path.stem,
            "frame_count": 20,
            "episode_success": index >= 3,
        }

    monkeypatch.setattr(episode_evaluator, "inspect_episode_shard", fake_inspect)
    args = SimpleNamespace(
        dataset="unused",
        episode_shards=None,
        val_split=0.2,
        split_seed=42,
        minimum_frames=16,
        num_episodes=5,
        num_failed_episodes=2,
        episode_seed=42,
    )

    selected = select_episode_shards(args)

    assert len(selected) == 5
    assert sum(item["episode_success"] is False for item in selected) == 2
    assert sum(item["episode_success"] is True for item in selected) == 3


def test_signed_inference_preserves_target_contract():
    args = episode_evaluator._ensure_inference_args(SimpleNamespace(
        dataset_type="tb3_potential_v1", target_schema="tb3_potential_v1",
        value_output_activation="tanh", potential_config={"obstacle_weight":2.0}))
    assert args.dataset_type == "tb3_potential_v1"
    assert args.value_output_activation == "tanh"
    assert args.potential_config == {"obstacle_weight":2.0}


def test_mixed_forward_and_rewind_shard_is_rejected(tmp_path):
    import pytest
    path=tmp_path/"mixed.tar"
    with tarfile.open(path,"w") as archive:
        _add_bytes(archive,"ep000001_step00000.overhead.png",_png_bytes())
        _add_bytes(archive,"ep000001_rewind_step00000.overhead.png",_png_bytes())
    with pytest.raises(ValueError,match="multiple episodes"):
        inspect_episode_shard(path)


def test_signed_plot_keeps_negative_targets_visible():
    plt,_=episode_evaluator._plot_modules()
    figure,axis=plt.subplots()
    episode={"steps":[0,1],"targets":{0:-.9,1:.8},"target_schema":"tb3_potential_v1"}
    episode_evaluator._plot_curves(axis,episode,{})
    assert axis.get_ylim()[0] <= -1
    assert axis.get_ylabel() == "Signed potential"
    plt.close(figure)
