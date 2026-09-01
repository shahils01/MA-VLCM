import io
import json
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ma_vlcm.train import SequenceWebDataset, webdataset_loader


class _FakeTokenizer:
    def __init__(self):
        self._vocab = {"<obs>": 1}

    def get_vocab(self):
        return self._vocab


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, videos, return_tensors="pt", padding="max_length", truncation=True, max_length=256):
        assert "Traversability information is unavailable" in text
        assert "exactly 3 agents" in text
        assert "exactly 3 static obstacles" in text
        assert "Task domain: static_obstacles" in text
        assert "+X points right in the image, +Y points up in the image" in text
        assert len(videos) >= 1
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "pixel_values_videos": torch.zeros((1, len(videos), 3, 8, 8), dtype=torch.float32),
        }


class _FakeVJEPA2Processor:
    def __call__(self, videos, return_tensors="pt"):
        assert len(videos) >= 1
        return {
            "pixel_values_videos": torch.zeros(
                (1, len(videos), 3, 8, 8), dtype=torch.float32
            )
        }


def _add_bytes(tar, name, payload):
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _make_npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def _make_png_bytes(size=(672, 336)):
    img = Image.new("RGB", size, color=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "tb3_lab_test.tar"
        state = {
            "task_instruction": "Reach assigned goals and avoid static obstacles.",
            "episode_meta": {
                "episode_id": "tb3_lab_episode",
                "step": 1,
                "done": False,
            },
            "agents": [
                {
                    "id": 0,
                    "domain_id": 1,
                    "color": "red",
                    "goal_label": "A",
                    "goal_pos": [-1.2, 0.8],
                    "pos": [-1.0, -0.2],
                    "yaw": 1.57,
                    "vel": [0.10, 0.01],
                    "dist_to_goal": 1.02,
                    "min_neighbor_dist": 0.90,
                    "reached": False,
                    "collision": False,
                    "action": "FORWARD",
                    "reward": 0.12,
                },
                {
                    "id": 1,
                    "domain_id": 2,
                    "color": "blue",
                    "goal_label": "B",
                    "goal_pos": [0.0, 0.8],
                    "pos": [0.0, -0.1],
                    "yaw": 1.54,
                    "vel": [0.08, -0.02],
                    "dist_to_goal": 0.91,
                    "min_neighbor_dist": 0.90,
                    "reached": False,
                    "collision": False,
                    "action": "FORWARD",
                    "reward": 0.11,
                },
                {
                    "id": 2,
                    "domain_id": 3,
                    "color": "green",
                    "goal_label": "C",
                    "goal_pos": [1.2, 0.8],
                    "pos": [1.0, 0.6],
                    "yaw": 1.52,
                    "vel": [0.03, 0.00],
                    "dist_to_goal": 0.28,
                    "min_neighbor_dist": 1.10,
                    "reached": True,
                    "collision": False,
                    "action": "STOP",
                    "reward": 5.0,
                },
            ],
            "reward": 1.74,
            "cumulative_reward": 8.2,
        }
        adj = np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        dist = np.array(
            [
                [0.0, 1.0, 2.2],
                [1.0, 0.0, 1.4],
                [2.2, 1.4, 0.0],
            ],
            dtype=np.float32,
        )

        with tarfile.open(tar_path, "w") as tar:
            prefix = "tb3_lab_episode_step0000"
            _add_bytes(tar, f"{prefix}.overhead.png", _make_png_bytes())
            _add_bytes(tar, f"{prefix}.state.json", json.dumps(state).encode("utf-8"))
            _add_bytes(tar, f"{prefix}.reward.json", json.dumps(1.74).encode("utf-8"))
            _add_bytes(tar, f"{prefix}.episode_reward.json", json.dumps(8.2).encode("utf-8"))
            _add_bytes(
                tar,
                f"{prefix}.progress.json",
                json.dumps({"schema": "tb3_progress_v1", "target": 0.42}).encode("utf-8"),
            )
            _add_bytes(tar, f"{prefix}.adj.npy", _make_npy_bytes(adj))
            _add_bytes(tar, f"{prefix}.dist.npy", _make_npy_bytes(dist))

        dataset = SequenceWebDataset(
            shards=str(tar_path),
            clip_len=1,
            clip_stride=1,
            text_mode="raw",
            robot_source="state",
            reward_reduce="mean",
            done_reduce="any",
            vlm_processor=_FakeProcessor(),
            vl_model_name=None,
            robot_obs_dim=8,
            num_robots=3,
            max_num_robots=3,
            text_prompt_template=None,
            dataset_type="tb3_lab",
            return_mode="nstep",
            target_mode="return",
            n_step=1,
            gamma=0.95,
            keep_raw_video=False,
            include_next=False,
            vlm_max_text_len=256,
            vlm_truncation=True,
            vlm_padding="max_length",
            resize_width=672,
            resize_height=336,
            vl_backend="llava_onevision",
        )

        sample = next(iter(dataset))
        assert sample["robot_obs"].shape == (1, 3, 8)
        assert sample["episode_id"] == "tb3_lab_episode"
        assert sample["adj"].shape == (1, 3, 3)
        torch.testing.assert_close(sample["reward"], torch.tensor([1.74]))
        torch.testing.assert_close(sample["returns"], torch.tensor([1.74]))
        assert sample["done"].item() == 0.0
        last_robot = sample["robot_obs"][0, 2]
        torch.testing.assert_close(last_robot[:2], torch.tensor([1.0, 0.6]))
        assert sample["inputs"]["input_ids"].shape == (3,)

        progress_dataset = SequenceWebDataset(
            shards=str(tar_path),
            clip_len=1,
            clip_stride=1,
            text_mode="raw",
            robot_source="state",
            reward_reduce="mean",
            done_reduce="any",
            vlm_processor=_FakeProcessor(),
            vl_model_name=None,
            robot_obs_dim=8,
            num_robots=3,
            max_num_robots=3,
            text_prompt_template=None,
            dataset_type="tb3_lab",
            return_mode="nstep",
            target_mode="progress",
            n_step=1,
            gamma=0.95,
            keep_raw_video=False,
            include_next=False,
            vlm_max_text_len=256,
            vlm_truncation=True,
            vlm_padding="max_length",
            resize_width=672,
            resize_height=336,
            vl_backend="llava_onevision",
        )
        progress_sample = next(iter(progress_dataset))
        torch.testing.assert_close(
            progress_sample["progress"], torch.tensor([0.42])
        )
        assert "returns" not in progress_sample

        multi_path = Path(tmpdir) / "tb3_lab_multistep.tar"
        state_done = json.loads(json.dumps(state))
        state_done["episode_meta"]["step"] = 2
        state_done["episode_meta"]["done"] = True
        state_done["episode_meta"]["failure"] = True
        state_done["episode_meta"]["termination_reason"] = "manual_failure"
        state_done["reward"] = -25.0
        state_done["cumulative_reward"] = -24.0
        state_done["agents"][0]["dist_to_goal"] = 1.20
        state_done["agents"][0]["failure"] = True
        state_done["agents"][0]["reward"] = -25.0

        with tarfile.open(multi_path, "w") as tar:
            first = "tb3_lab_episode_step0000"
            second = "tb3_lab_episode_step0001"
            _add_bytes(tar, f"{first}.overhead.png", _make_png_bytes())
            _add_bytes(tar, f"{first}.state.json", json.dumps(state).encode("utf-8"))
            _add_bytes(tar, f"{first}.reward.json", json.dumps(1.0).encode("utf-8"))
            _add_bytes(tar, f"{first}.episode_reward.json", json.dumps(1.0).encode("utf-8"))
            _add_bytes(tar, f"{first}.progress.json", json.dumps({"target": 0.2}).encode("utf-8"))
            _add_bytes(tar, f"{first}.adj.npy", _make_npy_bytes(adj))
            _add_bytes(tar, f"{first}.dist.npy", _make_npy_bytes(dist))
            _add_bytes(tar, f"{second}.overhead.png", _make_png_bytes())
            _add_bytes(tar, f"{second}.state.json", json.dumps(state_done).encode("utf-8"))
            _add_bytes(tar, f"{second}.reward.json", json.dumps(-25.0).encode("utf-8"))
            _add_bytes(tar, f"{second}.episode_reward.json", json.dumps(-24.0).encode("utf-8"))
            _add_bytes(tar, f"{second}.progress.json", json.dumps({"target": 0.0}).encode("utf-8"))
            _add_bytes(tar, f"{second}.adj.npy", _make_npy_bytes(adj))
            _add_bytes(tar, f"{second}.dist.npy", _make_npy_bytes(dist))

        temporal_dataset = SequenceWebDataset(
            shards=str(multi_path),
            clip_len=1,
            clip_stride=1,
            text_mode="raw",
            robot_source="state",
            reward_reduce="mean",
            done_reduce="any",
            vlm_processor=_FakeVJEPA2Processor(),
            vl_model_name=None,
            robot_obs_dim=8,
            num_robots=3,
            max_num_robots=3,
            text_prompt_template=None,
            dataset_type="tb3_lab",
            return_mode="nstep",
            target_mode="progress",
            n_step=1,
            gamma=0.95,
            keep_raw_video=False,
            include_next=True,
            vlm_max_text_len=256,
            vlm_truncation=True,
            vlm_padding="max_length",
            resize_width=144,
            resize_height=144,
            vl_backend="vjepa2",
        )
        temporal_sample = next(iter(temporal_dataset))
        assert temporal_sample["inputs"]["task_domain_ids"].item() == 2
        assert temporal_sample["next_inputs"]["task_domain_ids"].item() == 2
        torch.testing.assert_close(
            temporal_sample["progress"], torch.tensor([0.2])
        )
        torch.testing.assert_close(
            temporal_sample["next_progress"], torch.tensor([0.0])
        )

        multi_dataset = SequenceWebDataset(
            shards=str(multi_path),
            clip_len=2,
            clip_stride=1,
            text_mode="raw",
            robot_source="state",
            reward_reduce="mean",
            done_reduce="any",
            vlm_processor=_FakeProcessor(),
            vl_model_name=None,
            robot_obs_dim=8,
            num_robots=3,
            max_num_robots=3,
            text_prompt_template=None,
            dataset_type="tb3_lab",
            return_mode="nstep",
            target_mode="return",
            n_step=2,
            gamma=0.95,
            keep_raw_video=False,
            include_next=False,
            vlm_max_text_len=256,
            vlm_truncation=True,
            vlm_padding="max_length",
            resize_width=672,
            resize_height=336,
            vl_backend="llava_onevision",
        )
        multi_sample = next(iter(multi_dataset))
        assert multi_sample["robot_obs"].shape == (2, 3, 8)
        assert multi_sample["adj"].shape == (2, 3, 3)
        assert multi_sample["done"].item() == 1.0
        torch.testing.assert_close(
            multi_sample["reward"], torch.tensor([-25.0])
        )

        args = SimpleNamespace(
            clip_len=1,
            clip_stride=1,
            text_mode="raw",
            robot_source="state",
            reward_reduce="mean",
            done_reduce="any",
            vl_model_name=None,
            preprocess_in_loader=False,
            robot_obs_dim=8,
            num_robots=3,
            text_prompt_template=None,
            rware_config="mixed-rware",
            return_mode="nstep",
            target_mode="progress",
            n_step=1,
            gamma=0.95,
            value_output_activation="sigmoid",
            vl_max_text_len=256,
            resize_width=672,
            resize_height=336,
            vl_backend="llava_onevision",
            rware_visual_mode="rware_only",
            max_return_horizon=64,
        )
        loader = webdataset_loader(
            args,
            str(tar_path),
            batch_size=1,
            num_workers=0,
            shuffle=False,
            dataset_type="tb3_lab",
        )
        assert loader is not None
        print("tb3_lab loader smoke test passed")


if __name__ == "__main__":
    main()
