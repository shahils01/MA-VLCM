"""Read local or Hugging Face episode archives for the value-model pilot."""

from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image


class EpisodeAdapter:
    def discover_episodes(self, dataset_root, *, revision="main"):
        root = Path(dataset_root)
        if root.is_file():
            paths = [root]
        elif not root.exists():
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(
                repo_id=str(dataset_root), repo_type="dataset", revision=revision
            )
            file_set = set(files)
            return [
                {
                    "repo_id": str(dataset_root),
                    "revision": revision,
                    "tar_path": path,
                    "summary_path": next(
                        candidate for candidate in (
                            path.removesuffix(".tar") + ".summary.json",
                            path.removesuffix(".tar") + ".json",
                        ) if candidate in file_set
                    ),
                }
                for path in sorted(files)
                if path.endswith(".tar") and Path(path).name.startswith("episode_")
            ]

        else:
            paths = sorted(root.glob("**/episode_*.tar"))
        return [
            {
                "tar_path": path,
                "summary_path": next(candidate for candidate in (
                    path.with_suffix(".summary.json"), path.with_suffix(".json")
                ) if candidate.exists()),
            }
            for path in paths
        ]

    def load_episode(self, episode_reference):
        tar_path = episode_reference["tar_path"]
        summary_path = episode_reference["summary_path"]
        if "repo_id" in episode_reference:
            from huggingface_hub import hf_hub_download

            tar_path = hf_hub_download(
                repo_id=episode_reference["repo_id"],
                repo_type="dataset",
                filename=tar_path,
                revision=episode_reference["revision"],
            )
            summary_path = hf_hub_download(
                repo_id=episode_reference["repo_id"],
                repo_type="dataset",
                filename=summary_path,
                revision=episode_reference["revision"],
            )

        frames = {}
        with tarfile.open(tar_path) as archive:
            for member in archive.getmembers():
                name = Path(member.name).name
                if not member.isfile() or "." not in name:
                    continue
                frame_id, field = name.split(".", 1)
                frames.setdefault(frame_id, {})[field] = archive.extractfile(member).read()

        ordered = [frames[key] for key in sorted(frames)]
        states = [json.loads(frame["state.json"]) for frame in ordered]
        agents = [state["agents"] for state in states]
        robot_obs = np.asarray([
            [
                [*agent["pos"], np.cos(agent["yaw"]), np.sin(agent["yaw"]),
                 *agent["vel"], agent["dist_to_goal"], agent["min_neighbor_dist"]]
                for agent in frame_agents
            ]
            for frame_agents in agents
        ], dtype=np.float32)
        progress = [json.loads(frame["progress.json"]) for frame in ordered]
        control_rewards_available = all("control_rewards.json" in frame for frame in ordered)
        control_rewards = [
            json.loads(frame["control_rewards.json"]) if control_rewards_available else []
            for frame in ordered
        ]
        dones = [state["episode_meta"]["done"] for state in states]
        summary = json.loads(Path(summary_path).read_text())

        return {
            "episode_id": summary["episode_id"],
            "images": [
                Image.open(BytesIO(frame["overhead.png"])).convert("RGB")
                for frame in ordered
            ],
            "frame_ids": list(sorted(frames)),
            "timestamps": [state["episode_meta"]["sim_time_s"] for state in states],
            "capture_hz": float(states[0]["capture_hz"]),
            "control_hz": float(states[0]["control_hz"]),
            "prompt": states[0]["task_instruction"],
            "robot_obs": robot_obs,
            "adj": np.asarray([
                np.load(BytesIO(frame["adj.npy"])) for frame in ordered
            ], dtype=np.float32),
            "rewards": [float(json.loads(frame["reward.json"])) for frame in ordered],
            "control_rewards": control_rewards,
            "control_rewards_available": control_rewards_available,
            "progress": [frame["team_progress"] for frame in progress],
            "dones": dones,
            "terminated": bool(dones[-1] and summary["termination_reason"] != "timeout"),
            "truncated": bool(dones[-1] and summary["termination_reason"] == "timeout"),
            "success": summary["success"],
            "num_agents": summary["num_agents"],
        }

    def validate_episode(self, episode):
        count = len(episode["images"])
        control_rewards_ok = (
            not episode["control_rewards_available"]
            or all(len(rewards) == (0 if i == 0 else round(
                (episode["timestamps"][i] - episode["timestamps"][i - 1])
                * episode.get("control_hz", 20)
            )) for i, rewards in enumerate(episode["control_rewards"]))
        )
        return (
            count > 0
            and len(episode["timestamps"]) == count
            and len(episode["rewards"]) == count
            and len(episode["dones"]) == count
            and episode["robot_obs"].shape[:2] == (count, episode["num_agents"])
            and episode["adj"].shape == (count, episode["num_agents"], episode["num_agents"])
            and control_rewards_ok
            and all(a <= b for a, b in zip(episode["timestamps"], episode["timestamps"][1:]))
        )

    def make_clips(self, episode, settings):
        image_indices = self.image_indices(episode, settings.get("image_hz", 5.0))
        length = settings.get("clip_length", 16)
        stride = settings.get("stride", 1)
        horizon = settings.get("horizon", 20)
        gamma = settings.get("gamma", 0.99)
        interval_rewards = [[]]
        if episode["control_rewards_available"]:
            previous = 0
            for index in image_indices[1:]:
                interval_rewards.append([
                    reward
                    for frame_index in range(previous + 1, index + 1)
                    for reward in episode["control_rewards"][frame_index]
                ])
                previous = index
        else:
            interval_rewards.extend([
                [episode["rewards"][index]] for index in image_indices[1:]
            ])
        clips = []

        for anchor in range(0, len(image_indices), stride):
            start = max(0, anchor - length + 1)
            stop = anchor + 1
            target_stop = (
                len(image_indices) if horizon <= 0
                else min(len(image_indices), anchor + horizon + 1)
            )
            target = 0.0
            control_offset = 0
            for index in range(anchor + 1, target_stop):
                if episode["control_rewards_available"]:
                    for reward in interval_rewards[index]:
                        target += gamma ** control_offset * reward
                        control_offset += 1
                else:
                    control_offset += max(1, round(
                        (episode["timestamps"][image_indices[index]]
                         - episode["timestamps"][image_indices[index - 1]])
                        * episode["control_hz"]
                    ))
                    target += gamma ** control_offset * interval_rewards[index][0]
            target_valid = (
                (target_stop == len(image_indices) and episode["terminated"])
                or (horizon > 0 and target_stop - anchor - 1 == horizon)
            )

            clips.append({
                "episode_id": episode["episode_id"],
                "anchor_index": image_indices[anchor],
                "feature_indices": list(range(start, stop)),
                "frame_ids": [episode["frame_ids"][i] for i in image_indices[start:stop]],
                "timestamps": [episode["timestamps"][i] for i in image_indices[start:stop]],
                "images": [episode["images"][i] for i in image_indices[start:stop]],
                "prompt": episode["prompt"],
                "valid_frame_mask": [True] * (stop - start),
                "target": target,
                "target_valid_mask": target_valid,
                "progress": [episode["progress"][i] for i in image_indices[start:stop]],
                "dones": [episode["dones"][i] for i in image_indices[start:stop]],
                "robot_obs": episode["robot_obs"][image_indices[start:stop]],
                "adj": episode["adj"][image_indices[start:stop]],
                "terminated": episode["terminated"],
                "truncated": episode["truncated"],
                "success": episode["success"],
                "robot_mask": np.ones(
                    (stop - start, episode["num_agents"]), dtype=bool
                ),
            })

        return clips

    @staticmethod
    def image_indices(episode, image_hz):
        stride = max(1, round(episode["capture_hz"] / image_hz))
        indices = list(range(0, len(episode["images"]), stride))
        terminal = len(episode["images"]) - 1
        if episode["dones"][terminal] and indices[-1] != terminal:
            indices.append(terminal)
        return indices
