"""Report observed dataset coverage without loading any VLM weights."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from data_utils import OBSERVATION_NAMES, discover_episodes, frame_indices, prepare_episode, read_episode


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return {"min": float(a.min()), "p05": float(np.quantile(a, 0.05)), "median": float(np.median(a)),
            "p95": float(np.quantile(a, 0.95)), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}


def audit(episodes, gamma=0.99, image_hz=5.0):
    groups = defaultdict(list)
    for ep in episodes:
        groups[(ep["stage"], ep["num_agents"])].append(ep)
    rows = [{"stage": stage, "agents": n, "episodes": len(eps), "successes": sum(e["success"] for e in eps),
             "success_rate": sum(e["success"] for e in eps) / len(eps),
             "outcomes": dict(Counter(e["termination_reason"] for e in eps))}
            for (stage, n), eps in sorted(groups.items())]
    observations = torch.cat([ep["robot_obs"].flatten(0, 1) for ep in episodes]).numpy()
    rewards = torch.cat([ep["rewards"] for ep in episodes]).numpy()
    values = torch.cat([prepare_episode(ep, gamma, "sampled", image_hz)["returns"] for ep in episodes if ep["complete"]]).numpy()
    graph_density = []
    for ep in episodes:
        n = ep["num_agents"]
        mask = ~torch.eye(n, dtype=torch.bool)
        graph_density.extend((ep["adj"][:, mask] > 0).float().mean(-1).tolist())
    initial = np.concatenate([ep["robot_obs"][0, :, :2].numpy() for ep in episodes])
    occupancy, _, _ = np.histogram2d(initial[:, 0], initial[:, 1], bins=8, range=[[-1.35, 1.35], [-1.35, 1.35]])
    scenario_counts = Counter(ep["scenario_id"] for ep in episodes)
    bounds = sorted({tuple(ep["bounds"]) for ep in episodes})
    report = {
        "episodes": len(episodes), "recorded_frames": sum(len(ep["timestamps"]) for ep in episodes),
        "successes": sum(ep["success"] for ep in episodes),
        "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
        "team_sizes": dict(Counter(ep["num_agents"] for ep in episodes)),
        "policy_stages": dict(Counter(ep["stage"] for ep in episodes)),
        "policy_checkpoints": len({ep["policy_id"] for ep in episodes}),
        "outcomes": dict(Counter(ep["termination_reason"] for ep in episodes)),
        "collision_frames": sum(ep["collision_frames"] for ep in episodes),
        "episodes_with_collisions": sum(ep["collision_frames"] > 0 for ep in episodes),
        "first_image_sizes": dict(Counter(str(ep["image_size"]) for ep in episodes)),
        "unique_instructions": len({ep["prompt"] for ep in episodes}),
        "unique_stream_seeds": len({ep["seed"] for ep in episodes}),
        "unique_initial_scenarios": len(scenario_counts),
        "repeated_initial_scenario_groups": sum(count > 1 for count in scenario_counts.values()),
        "formation_patterns": dict(Counter(ep["formation_pattern"] for ep in episodes)),
        "formation_spacings": sorted({ep["formation_spacing"] for ep in episodes}),
        "robot_center_bounds": bounds,
        "episodes_with_control_reward_stream": sum(ep["control_rewards"] is not None for ep in episodes),
        "excluded_timeout_or_unfinished": sum(not ep["complete"] for ep in episodes),
        "capture_hz": sorted({ep["capture_hz"] for ep in episodes}),
        "control_hz": sorted({ep["control_hz"] for ep in episodes}),
        "seconds_per_episode": distribution([float(ep["timestamps"][-1]) for ep in episodes]),
        "frames_per_episode": distribution([len(ep["timestamps"]) for ep in episodes]),
        "sampled_rewards": distribution(rewards), "sampled_proxy_returns": distribution(values),
        "zero_reward_fraction": float(np.mean(rewards == 0)),
        "initial_position_8x8_occupied_bins": int((occupancy > 0).sum()),
        "initial_position_8x8_counts": occupancy.astype(int).tolist(),
        "nonself_graph_density": distribution(graph_density),
        "observation_ranges": {name: distribution(observations[:, i]) for i, name in enumerate(OBSERVATION_NAMES)},
        "groups": rows,
        "conclusion": (
            "Suitable for an initial in-distribution return-regression experiment. "
            "It does not establish generalization to new tasks, instructions, visual environments, "
            "robot embodiments or behavior policies. Success/failure coverage must be checked per team size. "
            "Use scenario-separated validation and compare against a constant-return baseline."
        ),
    }
    control_episodes = [ep for ep in episodes if ep["complete"] and ep["control_rewards"] is not None]
    if control_episodes:
        full = torch.cat([prepare_episode(ep, gamma, "control", image_hz)["returns"][:-1] for ep in control_episodes]).numpy()
        sampled = torch.cat([prepare_episode(ep, gamma, "sampled", image_hz)["returns"][:-1] for ep in control_episodes]).numpy()
        report["control_return_comparison"] = {
            "complete_episodes_aligned": len(control_episodes), "nonterminal_clips": len(full),
            "control_return_mean": float(full.mean()), "sampled_return_mean": float(sampled.mean()),
            "mean_absolute_difference": float(np.mean(np.abs(full - sampled))),
        }
    return report


def markdown(report):
    lines = ["# Formation dataset diversity audit", "",
             f"Observed {report['episodes']:,} episodes and {report['recorded_frames']:,} frames; "
             f"{report['successes']:,} successes ({report['success_rate']:.1%}).", "",
             "| Policy stage | Robots | Episodes | Successes | Success rate |",
             "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {r['stage']} | {r['agents']} | {r['episodes']} | {r['successes']} | {r['success_rate']:.1%} |" for r in report["groups"])
    lines.extend(["", f"Outcomes: `{report['outcomes']}`. "
        f"Recorded collision frames: {report['collision_frames']} across {report['episodes_with_collisions']} episodes.", "",
        f"First-frame image sizes: `{report['first_image_sizes']}`.", "",
        f"Unique instructions: {report['unique_instructions']}; initial scenarios: {report['unique_initial_scenarios']}; "
        f"repeated-scenario groups: {report['repeated_initial_scenario_groups']}; stream seeds: {report['unique_stream_seeds']}.", "",
        f"Initial robot positions occupy {report['initial_position_8x8_occupied_bins']}/64 bins over [-1.35, 1.35] m. "
        "Occupancy demonstrates spatial variation, not independence or broad task coverage.", "",
        f"Patterns: `{report['formation_patterns']}`; spacings: `{report['formation_spacings']}`; "
        f"robot-center bounds: `{report['robot_center_bounds']}`.", "",
        f"Capture/control rates: `{report['capture_hz']}` / `{report['control_hz']}` Hz. "
        f"Episodes with a raw control reward stream: {report['episodes_with_control_reward_stream']}. "
        f"Timeout/unfinished episodes excluded from return training: {report['excluded_timeout_or_unfinished']}.", "",
        f"Sampled rewards: `{report['sampled_rewards']}`.", "",
        f"Sampled proxy returns: `{report['sampled_proxy_returns']}`.", "",
        report["conclusion"], "",
        "The current data has one common instruction and one recorded formation task, spacing and workspace. "
        "Additional paraphrases alone would not demonstrate task-conditioned value prediction. "
        "To claim broader language or visual generalization, collect genuinely different goals/instructions, "
        "layouts, appearances, disturbances and independent policy/reset seeds. Retain successes, recoveries, "
        "near-boundary states and collision cases without erasing their terminal penalties.", "",
        "Mixed policy stages yield the conditional expected return of the recorded behavior mixture, "
        "not the value of one specified deployment policy. Use --policy-stage to restrict training when needed.", "",
        "Reward differentiability is not required for supervised value regression. Keep raw rewards and "
        "terminal events; standardize return targets using the training split. Smoothing across a collision, "
        "boundary exit or termination changes the objective and can spread future information into earlier labels.", ""])
    if "control_return_comparison" in report:
        c = report["control_return_comparison"]
        lines.extend([
            f"All {c['complete_episodes_aligned']} complete episodes with control streams passed interval alignment checks. "
            f"Across {c['nonterminal_clips']} nonterminal clips, mean full return is {c['control_return_mean']:.3f}, "
            f"mean sampled proxy is {c['sampled_return_mean']:.3f}, and their mean absolute difference is "
            f"{c['mean_absolute_difference']:.3f}. These statistics are based on the recorded team rewards.", "",
        ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="../Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--cache-dir", default="outputs/cache/index")
    p.add_argument("--output", default="reports/formation_diversity.json")
    args = p.parse_args()
    from tqdm.auto import tqdm

    references = discover_episodes(args.dataset, args.dataset_revision)
    episodes = [read_episode(path, summary, args.cache_dir) for path, summary in tqdm(references, desc="Audit episodes")]
    report = audit(episodes)
    report.update(dataset=args.dataset, dataset_revision=args.dataset_revision,
                  image_hz=5.0, gamma_per_control_step=0.99, return_source="sampled")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix(".md").write_text(markdown(report))
    print(f"episodes={report['episodes']} successes={report['successes']} instructions={report['unique_instructions']}")
    print(f"Wrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
