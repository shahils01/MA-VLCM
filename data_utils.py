"""Formation episode archives, causal clips, and return targets at recorded times."""

from collections import defaultdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


OBSERVATION_NAMES = (
    "position_x", "position_y", "sin_yaw", "cos_yaw", "velocity_0", "velocity_1",
    "relative_goal_x", "relative_goal_y", "distance_to_goal", "nearest_robot_distance",
)
INDEX_VERSION = 4


def save_tensor_file(value, path):
    """Publish complete caches/checkpoints without exposing a partially written file."""
    path = Path(path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def discover_episodes(source, revision="main", policy_stage="all"):
    root = Path(source).expanduser()
    if not root.exists() and not root.is_absolute() and len(str(source).split("/")) == 2 and not str(source).startswith("."):
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            repo_id=str(source), repo_type="dataset", revision=revision,
            allow_patterns=["**/episode_*.tar", "**/episode_*.summary.json", "**/episode_*.json"],
        ))
    paths = [root] if root.is_file() else sorted(root.rglob("episode_*.tar"))
    references = []
    for path in paths:
        summary_path = path.with_suffix(".summary.json")
        if not summary_path.exists():
            summary_path = path.with_suffix(".json")
        summary = json.loads(summary_path.read_text())
        if policy_stage == "all" or summary["checkpoint"]["stage"] == policy_stage:
            references.append((path.resolve(), summary_path.resolve()))
    if not references:
        raise ValueError(f"No formation episode archives found at {source}")
    return references


def _fingerprint(path, summary_path):
    stats = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (path, summary_path)]
    return hashlib.sha256(json.dumps([INDEX_VERSION, stats]).encode()).hexdigest()


def read_episode(path, summary_path, cache_dir=None):
    """Index metadata and image byte offsets without loading or decoding PNGs."""
    path, summary_path = Path(path), Path(summary_path)
    identity = _fingerprint(path, summary_path)
    cache = Path(cache_dir) / f"{identity}.pt" if cache_dir is not None else None
    if cache is not None and cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)
    summary = json.loads(summary_path.read_text())
    frames = defaultdict(dict)
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and "." in name:
                frame_id, field = name.split(".", 1)
                frames[frame_id][field] = member
        ordered = [frames[key] for key in sorted(frames)]
        states = [json.load(archive.extractfile(frame["state.json"])) for frame in ordered]
        adjacency = [np.load(BytesIO(archive.extractfile(frame["adj.npy"]).read())) for frame in ordered]
        rewards = [float(json.load(archive.extractfile(frame["reward.json"]))) for frame in ordered]
        control_present = ["control_rewards.json" in frame for frame in ordered]
        if any(control_present) and not all(control_present):
            raise ValueError(f"Partial control reward stream in {path}")
        control_rewards = [json.load(archive.extractfile(frame["control_rewards.json"])) for frame in ordered] if all(control_present) else None
        with Image.open(archive.extractfile(ordered[0]["overhead.png"])) as image:
            image_size = list(image.size)
    observations = []
    for state in states:
        rows = []
        for agent in state["agents"]:
            relative_goal = np.asarray(agent["goal_pos"]) - np.asarray(agent["pos"])
            rows.append([
                *agent["pos"], np.sin(agent["yaw"]), np.cos(agent["yaw"]),
                *agent["vel"], *relative_goal, agent["dist_to_goal"], agent["min_neighbor_dist"],
            ])
        observations.append(rows)
    first, last = states[0], states[-1]
    times = torch.tensor([state["episode_meta"]["sim_time_s"] for state in states], dtype=torch.float64)
    obs = torch.tensor(np.asarray(observations), dtype=torch.float32)
    if not torch.isfinite(obs).all() or not torch.isfinite(torch.tensor(rewards)).all() or not (times[1:] > times[:-1]).all():
        raise ValueError(f"Invalid states, rewards or timestamps in {path}")
    # Group identical initial scenarios across behavior checkpoints before splitting.
    scenario = {
        "robots": [[round(x, 6) for x in [*a["pos"], a["yaw"], *a["goal_pos"]]] for a in first["agents"]],
        "bounds": first["formation"]["robot_center_bounds_m"],
    }
    scenario_id = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
    reason = summary["termination_reason"]
    episode = {
        "identity": identity, "path": str(path), "episode_id": str(summary["episode_id"]),
        "frame_ids": sorted(frames),
        "image_offsets": [(f["overhead.png"].offset_data, f["overhead.png"].size) for f in ordered],
        "prompt": first["task_instruction"], "robot_obs": obs,
        "adj": torch.tensor(np.asarray(adjacency), dtype=torch.float32),
        "timestamps": times, "rewards": torch.tensor(rewards, dtype=torch.float64),
        "control_rewards": control_rewards, "capture_hz": float(first["capture_hz"]),
        "control_hz": float(first["control_hz"]), "num_agents": len(first["agents"]),
        "stage": summary["checkpoint"]["stage"], "policy_id": summary["checkpoint"]["sha256"],
        "success": bool(summary["success"]), "termination_reason": reason,
        "complete": bool(last["episode_meta"]["done"]) and reason != "timeout",
        "seed": summary["seed"], "scenario_id": scenario_id,
        "formation_pattern": first["formation"]["pattern"],
        "formation_spacing": first["formation"]["formation_spacing_m"],
        "bounds": first["formation"]["robot_center_bounds_m"],
        "collision_frames": int(summary.get("collision_frames", 0)), "image_size": image_size,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_tensor_file(episode, cache)
    return episode


def frame_indices(episode, image_hz=5.0):
    ratio = episode["capture_hz"] / image_hz
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1e-5:
        raise ValueError("image_hz must divide the recorded capture rate")
    indices = list(range(0, len(episode["timestamps"]), stride))
    if indices[-1] != len(episode["timestamps"]) - 1:
        indices.append(len(episode["timestamps"]) - 1)
    return indices


def discounted_returns(episode, gamma=0.99, source="sampled", indices=None):
    """V at an image excludes past reward; the terminal state's future return is zero.

    gamma is per control step. A control interval of m steps contributes
    sum(gamma**k * r[k], k=0..m-1). For sampled data only its observed final
    reward is known, so it contributes gamma**(m-1) * r_sample. Missing rewards
    are never interpolated or multiplied by the capture/control-rate ratio.
    """
    indices = list(range(len(episode["timestamps"]))) if indices is None else indices
    times = episode["timestamps"]
    values = torch.zeros(len(indices), dtype=torch.float64)
    if source == "control" and episode["control_rewards"] is None:
        raise ValueError("Full control-step returns require control_rewards.json; use --return-source sampled for the old dataset")
    for i in range(len(indices) - 2, -1, -1):
        left, right = indices[i:i + 2]
        steps = round(float(times[right] - times[left]) * episode["control_hz"])
        if steps < 1:
            raise ValueError("Image interval is shorter than one control step")
        if source == "control":
            interval = [r for frame in range(left + 1, right + 1) for r in episode["control_rewards"][frame]]
            if len(interval) != steps:
                raise ValueError(f"Expected {steps} control rewards, found {len(interval)} in {episode['path']}")
            reward = sum(gamma ** k * r for k, r in enumerate(interval))
        else:
            reward = gamma ** (steps - 1) * float(episode["rewards"][right])
        values[i] = reward + gamma ** steps * values[i + 1]
    return values.float()


def prepare_episode(episode, gamma=0.99, source="sampled", image_hz=5.0):
    indices = frame_indices(episode, image_hz)
    return dict(episode, selected_indices=indices, returns=discounted_returns(episode, gamma, source, indices))


def split_episodes(episodes, seed=42, validation_fraction=0.1, test_fraction=0.1):
    """Stratify whole scenarios by stage, team size and outcome; never split clips."""
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["scenario_id"]].append(episode)
    strata = defaultdict(list)
    for group in grouped.values():
        first = group[0]
        strata[(first["stage"], first["num_agents"], first["termination_reason"])].append(group)
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for groups in strata.values():
        rng.shuffle(groups)
        n = len(groups)
        n_val = max(1, round(n * validation_fraction)) if n >= 3 and validation_fraction else 0
        n_test = max(1, round(n * test_fraction)) if n >= 3 and test_fraction else 0
        n_val = min(n_val, n - 1)
        n_test = min(n_test, n - n_val - 1)
        for name, selected in (
            ("validation", groups[:n_val]), ("test", groups[n_val:n_val + n_test]),
            ("train", groups[n_val + n_test:]),
        ):
            splits[name].extend(episode for group in selected for episode in group)
    return splits


def read_images(episode, indices):
    images = []
    with open(episode["path"], "rb") as archive:
        for index in indices:
            offset, size = episode["image_offsets"][index]
            archive.seek(offset)
            with Image.open(BytesIO(archive.read(size))) as image:
                images.append(image.convert("RGB"))
    return images


def feature_cache_path(episode, cfg, cache_dir):
    identity = [1, episode["identity"], episode["prompt"], cfg.model_id, cfg.model_revision, cfg.vl_dtype, cfg.image_size, episode["selected_indices"]]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return Path(cache_dir) / f"{key}.pt"


def cache_features(episodes, encoder, cfg, cache_dir):
    from tqdm.auto import tqdm

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    encoder.eval()
    for episode in tqdm(episodes, desc="SmolVLM feature cache"):
        path = feature_cache_path(episode, cfg, cache_dir)
        if not path.exists():
            blocks = []
            indices = episode["selected_indices"]
            for start in range(0, len(indices), cfg.vl_batch_size):
                images = read_images(episode, indices[start:start + cfg.vl_batch_size])
                blocks.append(encoder(images, [episode["prompt"]] * len(images)).detach().cpu().half())
            save_tensor_file(torch.cat(blocks), path)


class FormationClipDataset(Dataset):
    def __init__(self, episodes, clip_len=6, clip_stride=1, feature_dir=None, cfg=None, include_terminal=True):
        self.episodes = episodes
        self.clip_len = clip_len
        self.samples = [(i, anchor) for i, ep in enumerate(episodes)
                        for anchor in range(0, len(ep["selected_indices"]) - int(not include_terminal and ep["complete"]), clip_stride)]
        self.features = None if feature_dir is None else [
            torch.load(feature_cache_path(ep, cfg, feature_dir), map_location="cpu", weights_only=True).float()
            for ep in episodes
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ep_index, anchor = self.samples[index]
        ep = self.episodes[ep_index]
        positions = torch.arange(anchor - self.clip_len + 1, anchor + 1)
        frame_mask = positions >= 0
        selected = positions.clamp_min(0)
        frame_indices = torch.tensor(ep["selected_indices"])[selected]
        row = {
            "robot_obs": ep["robot_obs"][frame_indices], "adj": ep["adj"][frame_indices],
            "robot_mask": frame_mask[:, None].expand(-1, ep["num_agents"]),
            "frame_mask": frame_mask,
            "times": (ep["timestamps"][frame_indices] - ep["timestamps"][frame_indices[-1]]).float(),
            "target": ep["returns"][anchor],
            "metadata": {"episode": ep["path"], "anchor": anchor, "stage": ep["stage"],
                         "num_agents": ep["num_agents"], "success": ep["success"]},
        }
        if self.features is None:
            row.update(images=read_images(ep, frame_indices.tolist()), prompt=ep["prompt"])
        else:
            row["vl_features"] = self.features[ep_index][selected]
        return row


def collate_clips(rows):
    batch = {}
    max_robots = max(row["robot_obs"].shape[1] for row in rows)
    for key in ("robot_obs", "adj", "robot_mask"):
        shape = (len(rows), rows[0][key].shape[0], max_robots)
        shape += (max_robots,) if key == "adj" else ((rows[0][key].shape[-1],) if key == "robot_obs" else ())
        batch[key] = torch.zeros(shape, dtype=rows[0][key].dtype)
        for i, row in enumerate(rows):
            n = row["robot_obs"].shape[1]
            if key == "adj":
                batch[key][i, :, :n, :n] = row[key]
            else:
                batch[key][i, :, :n] = row[key]
    for key in ("frame_mask", "times", "target", "vl_features"):
        if key in rows[0]:
            batch[key] = torch.stack([row[key] for row in rows])
    for key in ("images", "prompt", "metadata"):
        if key in rows[0]:
            batch[key] = [row[key] for row in rows]
    return batch


def training_statistics(episodes, clip_stride=1, include_terminal=True):
    observations = torch.cat([
        ep["robot_obs"][ep["selected_indices"][:len(ep["selected_indices"]) - int(not include_terminal and ep["complete"])]].flatten(0, 1)
        for ep in episodes
    ])
    targets = torch.cat([
        ep["returns"][:len(ep["returns"]) - int(not include_terminal and ep["complete"])][::clip_stride]
        for ep in episodes
    ])
    return {
        "obs_mean": observations.mean(0), "obs_std": observations.std(0, unbiased=False).clamp_min(1e-5),
        "target_mean": targets.mean(), "target_std": targets.std(unbiased=False).clamp_min(1e-5),
    }
