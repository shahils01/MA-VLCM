#!/usr/bin/env python3
"""Relabel TurtleBot3 lab WebDataset shards with bounded progress targets."""

import argparse
import glob
import io
import json
import os
import re
import tarfile
from pathlib import Path

import numpy as np


TB3_PROGRESS_SCHEMA = "tb3_progress_v1"
STEP_RE = re.compile(r"_step(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Local shard file/dir/glob or hf://datasets/<owner>/<repo>/<glob>.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tb3_lab_progress_v1",
        help="Directory for relabeled .tar shards.",
    )
    parser.add_argument("--repo-id", default="", help="Optional HF dataset repo to upload.")
    parser.add_argument("--revision", default=None, help="Optional HF branch/revision.")
    parser.add_argument("--private", action="store_true", help="Create/upload private HF dataset.")
    parser.add_argument("--goal-radius-m", type=float, default=0.12)
    parser.add_argument("--proximity-penalty-distance-m", type=float, default=0.20)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect shards and print the output plan without writing/uploading.",
    )
    return parser.parse_args()


def expand_source(source):
    prefix = "hf://datasets/"
    if source.startswith(prefix):
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required for hf:// sources.") from exc
        rest = source[len(prefix):]
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"Invalid hf dataset source: {source}")
        repo_id = f"{parts[0]}/{parts[1]}"
        allow_pattern = parts[2] if len(parts) > 2 else "*.tar"
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_pattern,
        )
        return sorted(glob.glob(os.path.join(local_root, allow_pattern), recursive=True))

    path = Path(source).expanduser()
    if path.is_dir():
        return sorted(str(p) for p in path.rglob("*.tar"))
    if path.is_file():
        return [str(path)]
    return sorted(glob.glob(str(path), recursive=True))


def step_prefix(name):
    for suffix in (
        ".state.json",
        ".reward.json",
        ".episode_reward.json",
        ".progress.json",
        ".overhead.png",
        ".dist.npy",
        ".adj.npy",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def episode_id_from_prefix(prefix):
    if "_step" not in prefix:
        return prefix
    return prefix.rsplit("_step", 1)[0]


def step_index_from_prefix(prefix):
    match = STEP_RE.search(prefix)
    return int(match.group(1)) if match else 0


def load_json(payload):
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    return json.loads(payload)


def npy_from_bytes(payload):
    return np.load(io.BytesIO(payload))


def agent_distances(state):
    return [float(ag.get("dist_to_goal", 0.0)) for ag in state.get("agents", [])]


def reached_flags(state):
    return [bool(ag.get("reached", False)) for ag in state.get("agents", [])]


def has_collision(state, dist_payload, threshold):
    if any(bool(ag.get("collision", False)) for ag in state.get("agents", [])):
        return True
    if dist_payload is None:
        return False
    dist = npy_from_bytes(dist_payload)
    if dist.ndim != 2:
        return False
    eye = np.eye(dist.shape[0], dtype=bool)
    return bool(((dist < float(threshold)) & (~eye)).any())


def compute_progress(
    initial_distances,
    current_distances,
    reached_now,
    done,
    terminal_failure,
    collision_failure,
    goal_radius_m,
):
    agent_progress = []
    for initial, current in zip(initial_distances, current_distances):
        initial_remaining = max(float(initial) - float(goal_radius_m), 1e-6)
        current_remaining = max(float(current) - float(goal_radius_m), 0.0)
        value = (initial_remaining - current_remaining) / initial_remaining
        agent_progress.append(float(np.clip(value, 0.0, 1.0)))

    team_progress = float(np.mean(agent_progress) if agent_progress else 0.0)
    success = bool(done and not terminal_failure and all(reached_now))
    failed = bool(terminal_failure or collision_failure)
    target = 1.0 if success else 0.0 if failed else team_progress
    return {
        "schema": TB3_PROGRESS_SCHEMA,
        "target": float(target),
        "team_progress": team_progress,
        "agent_progress": agent_progress,
        "initial_distances": [float(x) for x in initial_distances],
        "current_distances": [float(x) for x in current_distances],
        "goal_radius_m": float(goal_radius_m),
        "success": success,
        "failure": failed,
        "collision_failure": bool(collision_failure),
        "terminal_failure": bool(terminal_failure),
    }


def terminal_failure_from_state(state):
    meta = state.get("episode_meta", {})
    outcome = str(meta.get("outcome", "")).lower()
    reason = str(meta.get("termination_reason", "")).lower()
    return bool(
        meta.get("failure", False)
        or outcome == "failure"
        or "failure" in reason
        or reason.startswith(("boundary:", "stuck:", "controller_stop:"))
    )


def build_progress_labels(member_payloads, goal_radius_m, proximity_threshold):
    states = {}
    for name, payload in member_payloads.items():
        if name.endswith(".state.json"):
            states[name[: -len(".state.json")]] = load_json(payload)
    dists = {
        prefix[: -len(".dist.npy")]: payload
        for prefix, payload in member_payloads.items()
        if prefix.endswith(".dist.npy")
    }

    by_episode = {}
    for prefix in states:
        by_episode.setdefault(episode_id_from_prefix(prefix), []).append(prefix)

    progress_by_prefix = {}
    for prefixes in by_episode.values():
        prefixes.sort(key=step_index_from_prefix)
        first_state = states[prefixes[0]]
        initial = agent_distances(first_state)
        for prefix in prefixes:
            state = states[prefix]
            meta = state.get("episode_meta", {})
            current = agent_distances(state)
            reached = reached_flags(state)
            done = bool(meta.get("done", False))
            terminal_failure = terminal_failure_from_state(state)
            collision_failure = has_collision(state, dists.get(prefix), proximity_threshold)
            progress_by_prefix[prefix] = compute_progress(
                initial,
                current,
                reached,
                done,
                terminal_failure,
                collision_failure,
                goal_radius_m,
            )
    return progress_by_prefix


def add_bytes(tar, name, payload, template=None):
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = getattr(template, "mode", 0o644) if template is not None else 0o644
    info.mtime = getattr(template, "mtime", 0) if template is not None else 0
    tar.addfile(info, io.BytesIO(payload))


def relabel_shard(src_path, dst_path, goal_radius_m, proximity_threshold):
    with tarfile.open(src_path, "r") as src_tar:
        members = [m for m in src_tar.getmembers() if m.isfile()]
        payloads = {
            member.name: src_tar.extractfile(member).read()
            for member in members
        }

    progress_by_prefix = build_progress_labels(
        payloads,
        goal_radius_m=goal_radius_m,
        proximity_threshold=proximity_threshold,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    with tarfile.open(tmp_path, "w") as dst_tar:
        for member in members:
            name = member.name
            if name.endswith(".progress.json"):
                continue
            add_bytes(dst_tar, name, payloads[name], template=member)
            prefix = step_prefix(name)
            if name.endswith(".state.json") and prefix in progress_by_prefix:
                progress_payload = json.dumps(
                    progress_by_prefix[prefix],
                    sort_keys=True,
                ).encode("utf-8")
                add_bytes(
                    dst_tar,
                    f"{prefix}.progress.json",
                    progress_payload,
                    template=member,
                )
    os.replace(tmp_path, dst_path)
    return len(progress_by_prefix)


def upload_folder(output_dir, repo_id, revision, private):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        folder_path=str(output_dir),
        path_in_repo=".",
        commit_message=f"Add {TB3_PROGRESS_SCHEMA} labels",
    )


def main():
    args = parse_args()
    shards = expand_source(args.source)
    if not shards:
        raise SystemExit(f"No .tar shards found for source: {args.source}")

    output_dir = Path(args.output_dir).expanduser()
    print(f"Found {len(shards)} source shard(s).")
    print(f"Writing relabeled shards to: {output_dir}")
    if args.dry_run:
        for shard in shards:
            print(f"DRY RUN: {shard} -> {output_dir / Path(shard).name}")
        return

    total_steps = 0
    for index, shard in enumerate(shards, start=1):
        dst = output_dir / Path(shard).name
        count = relabel_shard(
            Path(shard),
            dst,
            goal_radius_m=args.goal_radius_m,
            proximity_threshold=args.proximity_penalty_distance_m,
        )
        total_steps += count
        print(f"[{index}/{len(shards)}] {Path(shard).name}: wrote {count} progress label(s)")

    print(f"Done. Wrote {total_steps} progress labels across {len(shards)} shard(s).")
    if args.repo_id:
        print(f"Uploading to Hugging Face dataset repo: {args.repo_id}")
        upload_folder(output_dir, args.repo_id, args.revision, args.private)
        print("Upload complete.")


if __name__ == "__main__":
    main()
