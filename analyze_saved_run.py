"""Inspect local loss logs and checkpoint predictions without loading the VLM.

Optionally rerun every saved checkpoint on validation before creating the report.
The default analyzes existing best-checkpoint predictions; it does not run inference.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics(rows):
    predicted = np.array([float(r["predicted_value"]) for r in rows])
    target = np.array([float(r["target_return"]) for r in rows])
    variance = np.sum((target - target.mean()) ** 2)
    return dict(count=len(rows), mse=float(np.mean((predicted - target) ** 2)),
                mae=float(np.mean(abs(predicted - target))),
                r2=float(1 - np.sum((predicted - target) ** 2) / variance) if variance else None)


def episode_frames(path, image_hz, gamma, return_source):
    with tarfile.open(path, "r:") as archive:
        members = {m.name: m for m in archive.getmembers()}
        names = sorted(n for n in members if n.endswith(".state.json"))
        first = json.load(archive.extractfile(members[names[0]]))
        ratio = first["capture_hz"] / image_hz
        stride = round(ratio)
        if stride < 1 or abs(ratio - stride) > 1e-5:
            raise ValueError("Image rate must divide archive capture rate")
        indices = list(range(0, len(names), stride))
        if indices[-1] != len(names) - 1:
            indices.append(len(names) - 1)
        controls = None
        if return_source == "control":
            controls = [json.load(archive.extractfile(members[name[:-len("state.json")] + "control_rewards.json"]))
                        for name in names]
        result = []
        for index in indices:
            name = names[index]
            state = json.load(archive.extractfile(members[name]))
            stem = name[:-len("state.json")]
            progress = json.load(archive.extractfile(members[stem + "progress.json"]))
            reward = json.load(archive.extractfile(members[stem + "reward.json"]))
            result.append(dict(time_s=state["episode_meta"]["sim_time_s"],
                               recorded_frame_reward=float(reward),
                               team_progress=float(progress["team_progress"]),
                               mean_goal_distance_m=float(np.mean([a["dist_to_goal"] for a in state["agents"]]))))
        target = 0.0
        result[-1]["recomputed_target_return"] = target
        for i in range(len(result) - 2, -1, -1):
            steps = round((result[i + 1]["time_s"] - result[i]["time_s"]) * first["control_hz"])
            if controls is None:
                interval_reward = gamma ** (steps - 1) * result[i + 1]["recorded_frame_reward"]
            else:
                interval = [r for chunk in controls[indices[i] + 1:indices[i + 1] + 1] for r in chunk]
                if len(interval) != steps:
                    raise ValueError(f"Control reward count mismatch: {path}")
                interval_reward = sum(gamma ** k * r for k, r in enumerate(interval))
            target = interval_reward + gamma ** steps * target
            result[i]["recomputed_target_return"] = target
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/zero_margin_900"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/smolvlm-local/validation"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Requires training dependencies; run saved checkpoints on validation")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((args.run_dir / "run.json").read_text())
    history = json.loads((args.run_dir / "metrics.json").read_text())
    best = min(history, key=lambda row: row["validation/mse"])
    checkpoints = sorted(args.run_dir.glob("*.pt"))
    sweep = []
    if args.reevaluate:
        # Use validation for comparisons, preserving the held-out test split.
        for checkpoint in checkpoints:
            folder = output / "checkpoints" / checkpoint.stem
            subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")),
                            "--checkpoint", str(checkpoint), "--split", "validation",
                            "--cache-dir", run["args"]["cache_dir"],
                            "--output-dir", str(folder)], check=True)
            sweep.append(json.loads((folder / "validation_metrics.json").read_text()))
        (output / "checkpoint_metrics.json").write_text(json.dumps(sweep, indent=2))

    prediction_path = args.run_dir / f"{args.split}_predictions.csv"
    if args.reevaluate and args.split == "validation":
        prediction_path = output / "checkpoints/best/validation_predictions.csv"
    with prediction_path.open() as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode"]].append(row)
    aligned = []
    for episode, selected in groups.items():
        frames = episode_frames(episode, run["args"]["image_hz"], run["args"]["gamma"], run["args"]["return_source"])
        for row in sorted(selected, key=lambda r: int(r["anchor"])):
            aligned.append(dict(row, **frames[int(row["anchor"])]))
    target_error = max(abs(float(r["target_return"]) - r["recomputed_target_return"]) for r in aligned)
    if target_error > 1e-5:
        raise ValueError(f"Prediction targets do not match archive returns: max error={target_error}")
    with (output / f"{args.split}_reward_progress.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aligned[0].keys())
        writer.writeheader()
        writer.writerows(aligned)
    summary = dict(run=str(args.run_dir), split=args.split,
                   prediction_source=str(prediction_path), fresh_inference=args.reevaluate and args.split == "validation",
                   return_source=run["args"]["return_source"], best_epoch=best["epoch"],
                   best_logged_validation_mse=best["validation/mse"],
                   final_logged_validation_mse=history[-1]["validation/mse"],
                   metrics=metrics(rows), episodes=len(groups), max_target_recomputation_error=target_error,
                   checkpoints=[str(p) for p in checkpoints])
    summary["by_outcome"] = {value: metrics([r for r in rows if r["success"] == value])
                             for value in sorted({r["success"] for r in rows})}
    for field in ["team_progress", "recorded_frame_reward"]:
        a = np.array([float(r["predicted_value"]) for r in aligned])
        b = np.array([float(r[field]) for r in aligned])
        summary[f"correlation_value_vs_{field}"] = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) else None
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r["epoch"] for r in history]
    for prefix, label in [("train", "Training"), ("validation", "Validation")]:
        axes[0].plot(epochs, [r[f"{prefix}/mse"] for r in history], marker=".", label=label)
    axes[0].axvline(best["epoch"], color="grey", linestyle="--", label=f"Best epoch {best['epoch']}")
    axes[0].set(xlabel="Epoch", ylabel="MSE (return units squared)", title=args.run_dir.name)
    axes[0].legend()
    target = np.array([float(r["target_return"]) for r in rows])
    pred = np.array([float(r["predicted_value"]) for r in rows])
    axes[1].hexbin(target, pred, gridsize=40, mincnt=1, bins="log")
    low, high = min(target.min(), pred.min()), max(target.max(), pred.max())
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set(xlabel="Recorded discounted return", ylabel="Predicted value",
                title=f"Best checkpoint: {args.split}, MSE={summary['metrics']['mse']:.4f}")
    fig.tight_layout()
    fig.savefig(output / "loss_and_return.png", dpi=160)
    plt.close(fig)

    # Deterministic median-error episode per outcome avoids selecting best-looking examples.
    examples = []
    for outcome in sorted({r["success"] for r in rows}):
        candidates = sorted((metrics(group)["mse"], path) for path, group in groups.items()
                            if group[0]["success"] == outcome)
        examples.append((outcome, candidates[len(candidates) // 2][1]))
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 3.5 * len(examples)), squeeze=False)
    for i, (outcome, episode) in enumerate(examples):
        selected = [r for r in aligned if r["episode"] == episode]
        times = [r["time_s"] for r in selected]
        for field, label in [("predicted_value", "Predicted value"), ("target_return", "Recorded return")]:
            axes[i, 0].plot(times, [float(r[field]) for r in selected], label=label)
        axes[i, 0].set(title=f"Success={outcome}: {Path(episode).stem}", ylabel="Discounted return")
        axes[i, 1].plot(times, [r["recorded_frame_reward"] for r in selected])
        axes[i, 1].set(title="Recorded frame reward", ylabel="Reward at image capture")
        axes[i, 2].plot(times, [r["team_progress"] for r in selected])
        axes[i, 2].set(title="Recorded formation progress", ylabel="Team progress")
        for ax in axes[i]:
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.2)
        axes[i, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "reward_progress_trajectories.png", dpi=160)
    plt.close(fig)
    lines = [f"# {args.run_dir.name}: saved-checkpoint analysis", "",
             f"Best logged validation epoch: {best['epoch']}; MSE {best['validation/mse']:.6f}.",
             f"Final logged validation MSE: {history[-1]['validation/mse']:.6f}.", "",
             f"Analyzed {len(rows)} {args.split} clips from {len(groups)} episodes.",
             f"Prediction source: `{prediction_path}`.",
             f"Fresh inference requested: {args.reevaluate}.", "",
             "The critic predicts discounted future reward, not instantaneous reward or progress.",
             "Progress curves are recorded labels; value/progress correlation is descriptive, not progress prediction accuracy.",
             "The recorded frame reward plot shows endpoint rewards, not every 20 Hz control reward.",
             "Example trajectories use the median-MSE episode within each outcome, without appending synthetic terminal predictions.",
             "Clip correlations weight longer episodes more and are not independent-sample significance tests.", "",
             "Use `best.pt` for deployment/evaluation; epoch snapshots are for comparison.",
             "Training metrics include dropout and parameter updates; validation uses evaluation mode.",
             "Full-control and sampled-proxy runs have different labels and cannot be ranked by raw MSE.", "",
             "Reevaluate all saved checkpoints on validation with the training Python environment:", "",
             "```bash", f"python analyze_saved_run.py --run-dir {args.run_dir} --output-dir {output} --reevaluate", "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
