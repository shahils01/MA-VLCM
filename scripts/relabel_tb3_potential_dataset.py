#!/usr/bin/env python3
"""Write signed potential labels to NEW shards; preserve authored rewards."""

import argparse
import copy
from dataclasses import asdict
import glob
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ma_vlcm.potential import PotentialConfig, compute_potential, goal_distance, transition_reward


def relabel_shard(source, destination, config, layouts=None, layout_key=None, include_rewind=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("destination must be a new shard, separate from the source")
    with tarfile.open(source, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        payloads = {m.name: tar.extractfile(m).read() for m in members}
    episodes = {}
    for name, payload in payloads.items():
        if name.endswith(".state.json"):
            prefix = name[:-len(".state.json")]
            match = re.fullmatch(r"(.+)_step(\d+)", prefix)
            if match is None:
                raise ValueError(f"cannot determine episode/step: {prefix}")
            episodes.setdefault(match[1], []).append((int(match[2]), prefix, json.loads(payload)))
    if not episodes:
        raise ValueError("shard contains no state samples")
    rewind_rows = []
    labels, report = {}, {"source": str(source), "destination": str(destination),
                          "frames": 0, "negative_deltas": 0, "positive_deltas": 0}
    for episode, rows in episodes.items():
        rows.sort()
        # Partial episode shards cannot establish a reliable starting distance.
        first_agents = rows[0][2]["agents"]
        if rows[0][0] != 0 and not all("initial_goal_distance" in a for a in first_agents):
            raise ValueError(f"{episode}: requires first frame or saved initial distances")
        initial = {a["id"]: (a["initial_goal_distance"]
                             if config.distance_mode == "euclidean" and "initial_goal_distance" in a
                             else goal_distance(a, config)) for a in first_agents}
        if rows[0][0] != 0 and config.distance_mode == "route_required":
            raise ValueError("partial route episodes require reconstruction before relabeling")
        obstacles = None
        if layouts is not None:
            # Worker directories distinguish reused episode IDs in different shards.
            key = f"{layout_key or source.name}::{episode}"
            if key not in layouts:
                raise ValueError(f"layout manifest has no entry for {key}")
            obstacles = layouts[key]["obstacles"]
            if not isinstance(obstacles, list):
                raise ValueError("layout obstacles must be a list")
        previous = None
        episode_success = bool(rows[-1][2].get("episode_meta", {}).get("success", False))
        for _, prefix, state in rows:
            label = compute_potential(state, initial, config, obstacles)
            label["potential_config"] = asdict(config)
            # Offline outcome metadata is used only for sampling, never in Phi.
            label["episode_success"] = episode_success
            label["transition_reward_gamma_1"] = None if previous is None else transition_reward(previous, label["target"])
            if previous is not None:
                delta = label["target"] - previous
                report["negative_deltas"] += int(delta < -1e-6)
                report["positive_deltas"] += int(delta > 1e-6)
            previous = label["target"]
            labels[prefix] = label
            report["frames"] += 1
        if include_rewind:
            # Keep the demonstration and its rewind in ONE shard to keep splits disjoint.
            for index, (_, prefix, state) in enumerate(reversed(rows)):
                synthetic = copy.deepcopy(state)
                synthetic["synthetic_rewind"] = True
                synthetic["episode_meta"] = {"episode_id": episode+"_rewind", "step": index,
                    "done": index == len(rows)-1, "success": False, "outcome": "synthetic_rewind"}
                synthetic["reward"] = 0.0
                synthetic.pop("cumulative_reward", None)
                for agent in synthetic["agents"]:
                    agent["initial_goal_distance"] = initial[agent["id"]]
                    agent["action"] = "SYNTHETIC_REWIND"
                    agent["reward"] = 0.0
                    for field in ("vel", "action_vector", "linear_velocity_mps", "angular_velocity_rps"):
                        if isinstance(agent.get(field), list):
                            agent[field] = [-float(x) for x in agent[field]]
                        elif isinstance(agent.get(field), (int, float)):
                            agent[field] = -float(agent[field])
                    if isinstance(agent.get("vel"), dict):
                        agent["vel"] = {k: -float(v) for k,v in agent["vel"].items()}
                    if isinstance(agent.get("command"), dict):
                        agent["command"] = {k: -float(v) for k,v in agent["command"].items()}
                label = dict(labels[prefix], episode_success=False, synthetic_rewind=True)
                previous = None if index == 0 else rewind_rows[-1][3]["target"]
                label["transition_reward_gamma_1"] = None if previous is None else transition_reward(previous,label["target"])
                rewind_rows.append((prefix, f"{episode}_rewind_step{index:05d}", synthetic, label))
    # Validate all labels before creating output. A failed geometry check writes no shard.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    try:
        with tarfile.open(temporary, "w") as tar:
            for member in members:
                if member.name.endswith(".progress.json"):
                    continue
                tar.addfile(member, io.BytesIO(payloads[member.name]))
                if member.name.endswith(".state.json"):
                    prefix = member.name[:-len(".state.json")]
                    payload = json.dumps(labels[prefix], sort_keys=True).encode()
                    info = tarfile.TarInfo(prefix + ".progress.json")
                    info.size, info.mode, info.mtime = len(payload), member.mode, member.mtime
                    tar.addfile(info, io.BytesIO(payload))
            for prefix, new_prefix, state, label in rewind_rows:
                # Replay visual/graph evidence. Original actions.npy is deliberately omitted.
                copies = [(n, payload) for n,payload in payloads.items()
                          if n.startswith(prefix+".") and n.endswith((".overhead.png", ".dist.npy", ".adj.npy", ".task.txt"))]
                copies += [(prefix+".state.json", json.dumps(state).encode()),
                           (prefix+".progress.json", json.dumps(label).encode()),
                           (prefix+".reward.json", json.dumps(label["transition_reward_gamma_1"] or 0.0).encode())]
                for name, payload in copies:
                    info = tarfile.TarInfo(new_prefix+name[len(prefix):])
                    info.size = len(payload)
                    tar.addfile(info,io.BytesIO(payload))
        # Hard link prevents overwriting a destination created during processing.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    report["rewind_frames"] = len(rewind_rows)
    report["rewind_negative_deltas"] = sum(
        (row[3]["transition_reward_gamma_1"] or 0.0) < -1e-6 for row in rewind_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local tar, directory, or recursive glob")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="JSON containing PotentialConfig fields")
    parser.add_argument("--layouts", help="JSON mapping relative-shard-path::episode-prefix to {obstacles: [...]}")
    parser.add_argument("--include-rewind", action="store_true", help="Append synthetic reversed episodes within each output shard")
    args = parser.parse_args()
    config = PotentialConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    layouts = json.loads(Path(args.layouts).read_text()) if args.layouts else None
    source = Path(args.source)
    paths = sorted(source.rglob("*.tar")) if source.is_dir() else [source] if source.is_file() else [Path(p) for p in sorted(glob.glob(args.source, recursive=True))]
    if not paths:
        parser.error("no source shards found")
    reports = []
    for path in paths:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        # For multi-worker data, directory mode preserves the worker subdirectories.
        report = relabel_shard(path, Path(args.output_dir)/relative, config, layouts, relative.as_posix(), args.include_rewind)
        reports.append(report)
        print(json.dumps(report))
    Path(args.output_dir, "potential_audit.json").write_text(json.dumps(reports, indent=2)+"\n")


if __name__ == "__main__":
    main()
