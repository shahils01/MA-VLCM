#!/usr/bin/env python3
"""Fit and assess per-formation affine value calibration on complete episodes.

Inputs are CSV/summary artifacts from evaluate_python_vlcm_actor.py. This is
an offline diagnostic; it never changes the actor, critic, or PPO rewards.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np


SHAPES = ("triangle", "square", "pentagon", "hexagon")


def read_episodes(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    groups = {name: [] for name in SHAPES}
    counts = {name: {"completed": 0, "timeouts": 0} for name in SHAPES}
    seen = set()
    for entry in summary["episodes"]:
        shape = entry["formation"]
        if shape not in groups:
            continue
        filename = entry["measurements"]
        if Path(filename).name != filename or filename in seen:
            raise ValueError(f"Invalid or duplicate episode CSV: {filename}")
        seen.add(filename)
        if not entry["return_complete"]:
            counts[shape]["timeouts"] += 1
            continue
        if entry["outcome"] != "success":
            raise ValueError(f"Unexpected complete Python episode outcome: {entry['outcome']}")
        with (root / filename).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 2 or any(row["return_complete"].lower() != "true" for row in rows):
            raise ValueError(f"Invalid complete episode: {filename}")
        # The terminal value is defined as zero by the evaluator, not a prediction.
        rows = rows[:-1]
        predictions = np.array([float(row["predicted_value"]) for row in rows])
        returns = np.array([float(row["observed_discounted_return"]) for row in rows])
        steps = np.array([int(row["step"]) for row in rows])
        if not np.isfinite(predictions).all() or not np.isfinite(returns).all():
            raise ValueError(f"Non-finite values in {filename}")
        groups[shape].append(dict(filename=filename, seed=entry["seed"], steps=steps,
                                  predictions=predictions, returns=returns))
        counts[shape]["completed"] += 1
    return summary, groups, counts


def metrics(episodes, slope, intercept):
    frame_mse, frame_mae, episode_mse = [], [], []
    for episode in episodes:
        error = slope * episode["predictions"] + intercept - episode["returns"]
        frame_mse.extend(error ** 2)
        frame_mae.extend(np.abs(error))
        episode_mse.append(float(np.mean(error ** 2)))
    return dict(episodes=len(episodes), anchors=len(frame_mse),
                mse=float(np.mean(frame_mse)), mae=float(np.mean(frame_mae)),
                mean_episode_mse=float(np.mean(episode_mse)))


def fit_affine(episodes):
    # Each episode has total weight one, regardless of its number of frames.
    xs = np.concatenate([ep["predictions"] for ep in episodes])
    ys = np.concatenate([ep["returns"] for ep in episodes])
    w = np.concatenate([np.full(len(ep["predictions"]), 1. / len(ep["predictions"])) for ep in episodes])
    design = np.column_stack((xs, np.ones_like(xs)))
    weighted = design * np.sqrt(w[:, None])
    if np.linalg.matrix_rank(weighted) < 2:
        raise ValueError("Training predictions have insufficient variation to fit a slope and intercept")
    coeff, _, _, _ = np.linalg.lstsq(weighted, ys * np.sqrt(w), rcond=None)
    return float(coeff[0]), float(coeff[1])


def calibrate(summary, groups, counts, seed, test_fraction):
    results, held_out = {}, []
    for index, shape in enumerate(SHAPES):
        episodes = sorted(groups[shape], key=lambda ep: ep["filename"])
        random.Random(seed + index).shuffle(episodes)
        result = dict(counts=counts[shape])
        # At least two complete episodes for fitting and one independent episode
        # for testing; small fits remain exploratory rather than reliable.
        if len(episodes) < 3:
            result.update(status="insufficient_complete_episodes",
                          reason="Need at least 2 fitting episodes and 1 held-out episode; timeouts are excluded")
            results[shape] = result
            continue
        ntest = max(1, min(len(episodes) - 2, math.ceil(len(episodes) * test_fraction)))
        test, train = episodes[:ntest], episodes[ntest:]
        try:
            slope, intercept = fit_affine(train)
        except ValueError as exc:
            result.update(status="unfit", reason=str(exc))
            results[shape] = result
            continue
        result.update(status="exploratory_fit", slope=slope, intercept=intercept,
                      fit_episodes=[ep["filename"] for ep in train],
                      test_episodes=[ep["filename"] for ep in test],
                      before=metrics(test, 1., 0.), after=metrics(test, slope, intercept))
        results[shape] = result
        for ep in test:
            for step, pred, target in zip(ep["steps"], ep["predictions"], ep["returns"]):
                held_out.append(dict(formation=shape, episode=ep["filename"], seed=ep["seed"], step=int(step),
                                     predicted_value=float(pred), calibrated_value=float(slope * pred + intercept),
                                     realized_return=float(target)))
    return dict(source_checkpoint=summary.get("checkpoint"), critic_sha256=summary.get("critic_sha256"),
                actor_sha256=summary.get("actor_sha256"), gamma=summary.get("gamma"), split_seed=seed,
                test_fraction=test_fraction, formations=results,
                caveat=("Affines are fitted only on completed successful episodes. Timeouts are excluded; "
                        "one observed return is not the conditional expected return. Evaluate on independent "
                        "success and failure cases before interpreting calibration or using it for PPO.")), held_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--test-fraction", type=float, default=.4)
    args = p.parse_args()
    if not 0 < args.test_fraction < 1:
        p.error("--test-fraction must be strictly between 0 and 1")
    summary, groups, counts = read_episodes(args.evaluation_dir)
    report, held_out = calibrate(summary, groups, counts, args.seed, args.test_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "held_out_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["formation", "episode", "seed", "step", "predicted_value",
                                               "calibrated_value", "realized_return"])
        writer.writeheader()
        writer.writerows(held_out)
    for shape, row in report["formations"].items():
        print(f"{shape}: {row['status']}; complete={row['counts']['completed']}, timeouts={row['counts']['timeouts']}")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
