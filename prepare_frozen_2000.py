"""Check the full-reward training collection and independent 60-episode set."""
import argparse
from collections import Counter
import json
from pathlib import Path

from data_utils import discover_episodes, prepare_episode, read_episode, split_episodes


def validate_collections(training, external, expected_training=2000, external_per_team=15):
    if len(training) != expected_training:
        raise ValueError(f"Expected {expected_training} training-collection archives, found {len(training)}")
    if len(external) != 4 * external_per_team:
        raise ValueError(f"Expected {4 * external_per_team} external archives, found {len(external)}")
    counts = Counter(ep["num_agents"] for ep in external)
    if counts != Counter({n: external_per_team for n in (3, 4, 5, 6)}):
        raise ValueError(f"External set must have {external_per_team} episodes per team size: {dict(counts)}")
    if any(ep["control_rewards"] is None for ep in training + external):
        raise ValueError("Every archive must contain the full control reward stream; sampled-proxy data is not interchangeable")
    if any(not ep["complete"] for ep in external):
        raise ValueError("All external episodes must have known returns through true termination")
    train_scenarios = {ep["scenario_id"] for ep in training}
    external_scenarios = {ep["scenario_id"] for ep in external}
    if train_scenarios & external_scenarios:
        raise ValueError("External initial scenarios overlap the training collection")
    if len(external_scenarios) != len(external):
        raise ValueError("External set contains repeated initial scenarios")
    train_seeds = {json.dumps(ep["seed"], sort_keys=True) for ep in training}
    external_seeds = {json.dumps(ep["seed"], sort_keys=True) for ep in external}
    if train_seeds & external_seeds:
        raise ValueError("External reset-stream seeds overlap the training collection")
    train_keys = {tuple(Path(ep["path"]).parts[-3:]) for ep in training}
    if any(tuple(Path(ep["path"]).parts[-3:]) in train_keys for ep in external):
        raise ValueError("External archive path keys overlap the training collection")


def describe(episodes):
    return dict(episodes=len(episodes), successes=sum(ep["success"] for ep in episodes),
                outcomes=dict(Counter(ep["termination_reason"] for ep in episodes)),
                by_team={str(n): dict(episodes=sum(ep["num_agents"] == n for ep in episodes),
                                     successes=sum(ep["num_agents"] == n and ep["success"] for ep in episodes))
                         for n in (3, 4, 5, 6)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--external-dataset", required=True)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--image-hz", type=float, default=5.)
    args = p.parse_args()
    if not 0 < args.gamma <= 1 or args.image_hz <= 0:
        p.error("gamma must be in (0, 1] and image-hz positive")
    collections = [[read_episode(path, sidecar, args.cache_dir / "index")
                    for path, sidecar in discover_episodes(source)]
                   for source in (args.dataset, args.external_dataset)]
    training, external = collections
    validate_collections(training, external)
    complete = [prepare_episode(ep, args.gamma, "control", args.image_hz) for ep in training if ep["complete"]]
    for ep in external:
        prepare_episode(ep, args.gamma, "control", args.image_hz)
    splits = split_episodes(complete, args.seed, .1, .1)
    report = dict(dataset=str(Path(args.dataset).resolve()), external_dataset=str(Path(args.external_dataset).resolve()),
                  training_collection=describe(training), external_collection=describe(external),
                  excluded_timeout_or_unfinished=len(training)-len(complete),
                  complete_episodes=len(complete), seed=args.seed, gamma=args.gamma,
                  image_hz=args.image_hz, return_source="control",
                  splits={name: describe(refs) for name, refs in splits.items()},
                  split_paths={name: [ep["path"] for ep in refs] for name, refs in splits.items()},
                  external_paths=[ep["path"] for ep in external],
                  independence="No training/external initial-scenario, stream-seed or archive-key overlap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete_episodes", "excluded_timeout_or_unfinished", "splits")}))
    print(f"Checked 2000 training-collection archives and 60 independent external episodes; report: {args.output}")


if __name__ == "__main__":
    main()
