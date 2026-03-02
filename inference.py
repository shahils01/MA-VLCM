#!/usr/bin/env python3
"""
inference.py – Load a MA-VLCM checkpoint and evaluate on a test dataset.

Usage:
    python inference.py \
        --checkpoint /path/to/ckpt_epoch_N.pt \
        --test_shards /path/to/test_data \
        [--batch_size 4] [--num_workers 4] [--output_file results.csv]
"""

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse model and data utilities from the training script
from model import ModelConfig, MultimodalValueModel
from train import (
    build_model,
    _apply_peft,
    webdataset_loader,
)


# ─────────────────────────────── Argument Parsing ───────────────────────────
def parse_inference_args():
    p = argparse.ArgumentParser(
        description="Run inference with a trained MA-VLCM checkpoint."
    )

    # ── Required ──
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file (saved by train.py).",
    )
    p.add_argument(
        "--test_shards",
        type=str,
        required=True,
        help="Path or glob pattern for test WebDataset shards.",
    )

    # ── Optional overrides (defaults pulled from checkpoint) ──
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0, cpu")
    p.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Cap on the number of test samples to evaluate.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output CSV file for per-sample predictions.",
    )
    p.add_argument(
        "--plot_dir",
        type=str,
        default="inference_plots",
        help="Directory to save comparison plots (set to '' to disable).",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode: skip LoRA adapters, keep LLaVA at pretrained weights. "
        "Only loads GNN + value head from checkpoint for apples-to-apples comparison.",
    )
    p.add_argument(
        "--compare_csv",
        type=str,
        default=None,
        help="Path to a CSV from a previous run (e.g. baseline) to overlay on the "
        "current run's plots for side-by-side comparison.",
    )
    p.add_argument(
        "--compare_csvs",
        type=str,
        nargs="+",
        default=None,
        help="Multiple labeled CSVs for multi-way comparison. "
        "Format: 'Label:path.csv' (e.g. "
        "'Baseline:results_baseline.csv' "
        "'LoRA:results_lora.csv'). "
        "If label is omitted, filename is used.",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for the current run in comparison plots "
        "(default: auto-detected from --baseline flag).",
    )
    p.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=["rware", "offroad"],
        help="Override the dataset_type saved in the checkpoint. Necessary if the model was trained on mixed datasets and you want to evaluate on a specific one.",
    )

    return p.parse_args()


def _spearman_corr(x, y):
    """Spearman rank correlation between two 1-D numpy arrays."""
    if len(x) < 2:
        return float("nan")
    from scipy.stats import spearmanr

    corr, _ = spearmanr(x, y)
    return float(corr)


# ────────────────────────── Plotting ────────────────────────────────────────
def _generate_plots(preds, targets, plot_dir, epoch, target_label="Return"):
    """Generate comparison plots of predicted vs true values.

    Args:
        targets: Returns — whatever the model was trained on.
        target_label: Label for the target values in plots.
    """
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in: {plot_dir}")

    true_label = target_label
    true_vals = targets

    # ── 1. Scatter plot: Predicted vs True ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(true_vals, preds, alpha=0.35, s=12, c="#4C72B0", edgecolors="none")
    # Perfect prediction line
    lo = min(true_vals.min(), preds.min())
    hi = max(true_vals.max(), preds.max())
    margin = (hi - lo) * 0.05
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        "--",
        color="#C44E52",
        linewidth=1.5,
        label="y = x (perfect)",
    )
    # Linear fit
    if len(true_vals) > 1:
        coeffs = np.polyfit(true_vals, preds, 1)
        fit_x = np.linspace(lo - margin, hi + margin, 100)
        ax.plot(
            fit_x,
            np.polyval(coeffs, fit_x),
            "-",
            color="#55A868",
            linewidth=1.5,
            label=f"Linear fit (slope={coeffs[0]:.3f})",
        )
    spearman = _spearman_corr(preds, true_vals)
    ax.set_xlabel(f"True {true_label}", fontsize=13)
    ax.set_ylabel("Predicted Value", fontsize=13)
    ax.set_title(
        f"Predicted vs True {true_label}  (epoch {epoch}, \u03c1={spearman:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "scatter_pred_vs_true.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Per-sample comparison (first 500 samples) ─────────────────────
    show_n = min(500, len(preds))
    fig, ax = plt.subplots(figsize=(14, 5))
    idx = np.arange(show_n)
    ax.plot(
        idx,
        true_vals[:show_n],
        "-",
        color="#4C72B0",
        linewidth=1.0,
        alpha=0.8,
        label=f"True {true_label}",
    )
    ax.plot(
        idx,
        preds[:show_n],
        "-",
        color="#C44E52",
        linewidth=1.0,
        alpha=0.8,
        label="Predicted",
    )
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Value", fontsize=13)
    ax.set_title(
        f"Per-Sample Comparison  (first {show_n} samples)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "sample_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _generate_overlay_plots(
    preds, targets, compare_csv, plot_dir, epoch, target_label="Target"
):
    """Overlay current run predictions with a previous run's CSV for comparison."""
    import pandas as pd

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(compare_csv)
    comp_preds = df["prediction"].values
    # Use return if available, else td_target as the ground truth
    if "return" in df.columns:
        comp_targets = df["return"].values
        target_col = "return"
    elif "td_target" in df.columns:
        comp_targets = df["td_target"].values
        target_col = "td_target"
    else:
        # Fallback only
        comp_targets = df["reward"].values
        target_col = "reward"

    # Detect which run is which from filename
    is_current_baseline = "baseline" in plot_dir.lower()
    if is_current_baseline:
        curr_label = "Baseline (no LoRA)"
        comp_label = "Fine-tuned"
    else:
        curr_label = "Fine-tuned"
        comp_label = "Baseline (no LoRA)"

    true_vals = targets if targets is not None else comp_targets
    true_label = target_label
    n = min(len(preds), len(comp_preds))

    print(f"\nGenerating overlay comparison plots in: {plot_dir}")
    print(f"  Current: {curr_label}  ({len(preds)} samples)")
    print(f"  Compare: {comp_label}  ({len(comp_preds)} samples, from {compare_csv})")

    # ── 1. Overlay scatter: both predictions vs true target ──────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        true_vals[:n],
        comp_preds[:n],
        alpha=0.3,
        s=14,
        c="#8DA0CB",
        edgecolors="none",
        label=comp_label,
        zorder=2,
    )
    ax.scatter(
        true_vals[:n],
        preds[:n],
        alpha=0.4,
        s=14,
        c="#E78AC3",
        edgecolors="none",
        label=curr_label,
        zorder=3,
    )
    lo = min(true_vals[:n].min(), preds[:n].min(), comp_preds[:n].min())
    hi = max(true_vals[:n].max(), preds[:n].max(), comp_preds[:n].max())
    m = (hi - lo) * 0.05
    ax.plot(
        [lo - m, hi + m],
        [lo - m, hi + m],
        "--",
        color="#333333",
        linewidth=1.5,
        alpha=0.6,
        label="y = x",
    )
    spearman_curr = _spearman_corr(preds[:n], true_vals[:n])
    spearman_comp = _spearman_corr(comp_preds[:n], true_vals[:n])
    ax.set_xlabel(f"True {true_label}", fontsize=13)
    ax.set_ylabel("Predicted Value", fontsize=13)
    ax.set_title(
        f"Scatter Overlay  (\u03c1: {curr_label}={spearman_curr:.3f}, {comp_label}={spearman_comp:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Overlay per-sample comparison ─────────────────────────────────
    show_n = min(300, n)
    idx = np.arange(show_n)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        idx,
        true_vals[:show_n],
        "-",
        color="#333333",
        linewidth=1.2,
        alpha=0.7,
        label=f"True {true_label}",
    )
    ax.plot(
        idx,
        comp_preds[:show_n],
        "-",
        color="#8DA0CB",
        linewidth=1.0,
        alpha=0.7,
        label=comp_label,
    )
    ax.plot(
        idx,
        preds[:show_n],
        "-",
        color="#E78AC3",
        linewidth=1.0,
        alpha=0.7,
        label=curr_label,
    )
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Value", fontsize=13)
    ax.set_title(
        f"Per-Sample: Baseline vs Fine-tuned  (first {show_n})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_sample_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. MSE bar chart ────────────────────────────────────────────────
    mse_curr = float(np.mean((preds[:n] - true_vals[:n]) ** 2))
    mse_comp = float(np.mean((comp_preds[:n] - true_vals[:n]) ** 2))
    mae_curr = float(np.mean(np.abs(preds[:n] - true_vals[:n])))
    mae_comp = float(np.mean(np.abs(comp_preds[:n] - true_vals[:n])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_i, (metric, vals, title) in enumerate(
        [
            ("MSE", [mse_comp, mse_curr], "MSE"),
            ("MAE", [mae_comp, mae_curr], "MAE"),
        ]
    ):
        ax = axes[ax_i]
        bars = ax.bar(
            [comp_label, curr_label],
            vals,
            color=["#8DA0CB", "#E78AC3"],
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Metrics Comparison (epoch {epoch})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_metrics_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _generate_multi_comparison_plots(
    current_preds,
    current_targets,
    current_label,
    compare_csvs,
    plot_dir,
    epoch,
    target_label="Target",
):
    """Generate plots comparing True Returns vs multiple model runs.

    Args:
        current_preds: numpy array of current run predictions.
        current_targets: numpy array of current run TD targets (ground truth).
        current_label: string label for the current run.
        compare_csvs: list of (label, csv_path) tuples.
        plot_dir: directory to save plots.
        epoch: epoch number for titles.
        target_label: The label for the target axis.
    """
    import pandas as pd

    os.makedirs(plot_dir, exist_ok=True)

    # Collect all runs: list of (label, preds, targets)
    runs = [(current_label, current_preds, current_targets)]
    for label, csv_path in compare_csvs:
        df = pd.read_csv(csv_path)
        p = df["prediction"].values
        if "td_target" in df.columns:
            t = df["td_target"].values
        elif "return" in df.columns:
            t = df["return"].values
        else:
            t = df["reward"].values
        runs.append((label, p, t))

    # Use the shortest common length
    n = min(len(r[1]) for r in runs)
    # Use the first run's targets as ground truth
    true_vals = runs[0][2][:n]
    true_label = target_label

    # Color palette for up to 6 runs
    colors = [
        "#E78AC3",  # pink (current)
        "#8DA0CB",  # blue
        "#66C2A5",  # teal
        "#FC8D62",  # orange
        "#A6D854",  # lime
        "#FFD92F",  # yellow
    ]

    print(f"\nGenerating multi-comparison plots in: {plot_dir}")
    for i, (lbl, _, _) in enumerate(runs):
        print(f"  Run {i}: {lbl}")

    # ── 1. Scatter overlay ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    lo, hi = true_vals.min(), true_vals.max()
    for i, (lbl, preds_i, _) in enumerate(runs):
        pi = preds_i[:n]
        lo = min(lo, pi.min())
        hi = max(hi, pi.max())
        spearman_i = _spearman_corr(pi, true_vals)
        ax.scatter(
            true_vals,
            pi,
            alpha=0.3,
            s=12,
            c=colors[i % len(colors)],
            edgecolors="none",
            label=f"{lbl} (\u03bc={spearman_i:.3f})",
            zorder=2 + i,
        )
    m = (hi - lo) * 0.05
    ax.plot(
        [lo - m, hi + m],
        [lo - m, hi + m],
        "--",
        color="#333333",
        linewidth=1.5,
        alpha=0.6,
        label="y = x",
    )
    ax.set_xlabel(f"True {true_label}", fontsize=13)
    ax.set_ylabel("Predicted Value", fontsize=13)
    ax.set_title(
        f"Multi-Model Comparison (epoch {epoch})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "multi_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Per-sample comparison ────────────────────────────────────────
    show_n = min(300, n)
    idx = np.arange(show_n)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        idx,
        true_vals[:show_n],
        "-",
        color="#333333",
        linewidth=1.4,
        alpha=0.8,
        label=true_label,
    )
    for i, (lbl, preds_i, _) in enumerate(runs):
        ax.plot(
            idx,
            preds_i[:show_n],
            "-",
            color=colors[i % len(colors)],
            linewidth=1.0,
            alpha=0.7,
            label=lbl,
        )
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Value", fontsize=13)
    ax.set_title(
        f"Per-Sample: All Models (first {show_n})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(
        plot_dir,
        "multi_sample_comparison.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. MSE / MAE bar chart ─────────────────────────────────────────
    labels = [r[0] for r in runs]
    mse_vals = [float(np.mean((r[1][:n] - true_vals) ** 2)) for r in runs]
    mae_vals = [float(np.mean(np.abs(r[1][:n] - true_vals))) for r in runs]
    spearman_vals = []
    for r in runs:
        try:
            sp = _spearman_corr(r[1][:n], true_vals)
        except Exception:
            sp = float("nan")
        spearman_vals.append(sp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (metric_vals, title) in enumerate(
        [
            (mse_vals, "MSE \u2193"),
            (mae_vals, "MAE \u2193"),
            (spearman_vals, "Spearman \u03c1 \u2191"),
        ]
    ):
        ax = axes[ax_i]
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(
            labels,
            metric_vals,
            color=bar_colors,
            edgecolor="white",
        )
        for bar, val in zip(bars, metric_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(
        f"Metrics Comparison (epoch {epoch})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(plot_dir, "multi_metrics_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────── Main ──────────────────────────────────────
def main():
    cli_args = parse_inference_args()

    # ── 1. Load checkpoint ──────────────────────────────────────────────────
    print(f"Loading checkpoint: {cli_args.checkpoint}")
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)

    # Restore training args as a namespace (acts as defaults)
    saved_args_dict = ckpt.get("args", {})
    args = SimpleNamespace(**saved_args_dict)

    # Override dataset_type if requested (for mixed dataset checkpoints)
    if cli_args.dataset_type is not None:
        args.dataset_type = cli_args.dataset_type

    # Apply CLI overrides
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    else:
        args.batch_size = getattr(args, "batch_size", 4)

    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    else:
        args.num_workers = getattr(args, "num_workers", 2)

    if cli_args.mixed_precision is not None:
        args.mixed_precision = cli_args.mixed_precision

    # Ensure critical attributes exist
    args.preprocess_in_loader = getattr(args, "preprocess_in_loader", True)
    args.video_preprocessed = getattr(args, "video_preprocessed", True)
    args.compile = getattr(args, "compile", False)
    args.quantization_config = None  # No quantisation for inference
    args.text_prompt_template = getattr(args, "text_prompt_template", None)

    # ── 2. Determine device ─────────────────────────────────────────────────
    if cli_args.device:
        device = torch.device(cli_args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── 3. Rebuild model + load weights ─────────────────────────────────────
    baseline_mode = cli_args.baseline
    if baseline_mode:
        print("Building model in BASELINE mode (no LoRA, pretrained LLaVA backbone)...")
        # Build without LoRA — pretrained LLaVA + random custom heads
        saved_peft = getattr(args, "peft", "none")
        args.peft = "none"
        model = build_model(args, device=device)
        model = _apply_peft(model, args)  # no-op since peft="none"
        args.peft = saved_peft  # restore for logging
    else:
        print("Building model (with LoRA)...")
        model = build_model(args, device=device)
        model = _apply_peft(model, args)

    # Load state dict
    state_dict = ckpt["model"]
    # Handle potential "module." prefix from DDP/FSDP wrapping
    cleaned_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_sd[new_k] = v

    if baseline_mode:
        # Only load custom heads: robot_gnn, value_head, obs_to_lm
        # Skip all LoRA and backbone keys — keep LLaVA at pretrained weights
        custom_prefixes = ("robot_gnn.", "value_head.", "obs_to_lm.")
        baseline_sd = {
            k: v for k, v in cleaned_sd.items() if k.startswith(custom_prefixes)
        }
        print(
            f"  BASELINE: Loading {len(baseline_sd)} custom head keys "
            f"(skipping {len(cleaned_sd) - len(baseline_sd)} backbone/LoRA keys)"
        )
        missing, unexpected = model.load_state_dict(baseline_sd, strict=False)
        # Many "missing" keys expected (entire backbone) — only warn about custom heads
        missing_custom = [k for k in missing if k.startswith(custom_prefixes)]
        if missing_custom:
            print(
                f"  WARNING: {len(missing_custom)} custom head keys missing: {missing_custom}"
            )
    else:
        missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(
                f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}"
            )

    epoch = ckpt.get("epoch", "?")
    mode_str = "BASELINE (no LoRA)" if baseline_mode else "fine-tuned"
    print(f"  Loaded checkpoint from epoch {epoch} [{mode_str}]")

    # Determine dtype for inference
    mp = getattr(args, "mixed_precision", "no")
    if mp == "bf16":
        model_dtype = torch.bfloat16
    elif mp == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    # ── 4. Build test data loader ───────────────────────────────────────────
    print(f"Loading test data from: {cli_args.test_shards}")
    args.train_shards = cli_args.test_shards  # webdataset_loader reads this indirectly

    # Force include_next=True so we get next-state data for TD target computation
    saved_loss_type = getattr(args, "loss_type", "td")
    saved_return_mode = getattr(args, "return_mode", "td")
    args.loss_type = "td"  # Ensures include_next=True in webdataset_loader
    if saved_return_mode in ("nstep", "nsteps"):
        args.return_mode = saved_return_mode
    else:
        args.return_mode = "td"

    test_loader = webdataset_loader(
        args,
        shards=cli_args.test_shards,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Restore original values for logging
    args.loss_type = saved_loss_type
    args.return_mode = saved_return_mode

    gamma = getattr(args, "gamma", 0.95)
    clip_len = getattr(args, "clip_len", 16)
    clip_gamma = gamma**clip_len

    # ── 4b. Optionally load EMA shadow weights for bootstrap ────────────────
    ema_shadow = ckpt.get("ema_shadow", None)
    if ema_shadow is not None:
        print(
            f"  EMA shadow loaded ({len(ema_shadow)} params) — will use for bootstrap V(s')"
        )
    else:
        print("  No EMA shadow in checkpoint — using online weights for bootstrap")

    # ── 5. Inference loop ───────────────────────────────────────────────────
    print("Running inference...")
    print(
        f"  Computing TD targets with gamma={gamma}, clip_len={clip_len}, clip_gamma={clip_gamma:.6f}"
    )
    all_preds = []
    all_td_targets = []
    all_rewards = []
    all_returns = []
    num_processed = 0

    def _move_and_cast(tensor_dict):
        """Move dict of tensors to device and cast floats to model_dtype."""
        out = {}
        for k, v in tensor_dict.items():
            if torch.is_tensor(v):
                v = v.to(device)
                if v.is_floating_point():
                    v = v.to(dtype=model_dtype)
                out[k] = v
            else:
                out[k] = v
        return out

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inference", dynamic_ncols=True):
            # Move inputs to device and cast to model dtype
            inputs = _move_and_cast(batch["inputs"])
            robot_obs = batch["robot_obs"].to(device=device, dtype=model_dtype)
            adj = batch["adj"].to(device=device, dtype=model_dtype)
            reward = batch["reward"]  # keep on CPU for collection
            done = batch["done"].float()

            # Forward pass: V(s)
            pred = model(inputs, robot_obs, adj)
            pred_cpu = pred.detach().cpu().float()

            # Compute bootstrapped TD target matching train.py:
            #   target = nstep_returns + clip_gamma * (1 - done) * V_ema(s')
            if "next_inputs" in batch:
                next_inputs = _move_and_cast(batch["next_inputs"])
                next_robot_obs = batch["next_robot_obs"].to(
                    device=device, dtype=model_dtype
                )
                next_adj = batch["next_adj"].to(device=device, dtype=model_dtype)

                # Use EMA weights for bootstrap if available (matches training)
                if ema_shadow is not None:
                    saved_params = {}
                    for n, p in model.named_parameters():
                        if n in ema_shadow:
                            saved_params[n] = p.data.clone()
                            p.data.copy_(ema_shadow[n].to(p.device))
                    next_pred = model(next_inputs, next_robot_obs, next_adj)
                    for n, p in model.named_parameters():
                        if n in saved_params:
                            p.data.copy_(saved_params[n])
                    del saved_params
                else:
                    next_pred = model(next_inputs, next_robot_obs, next_adj)

                # Match train.py: use nstep returns + clip_gamma bootstrap
                if "returns" in batch:
                    nstep_returns = batch["returns"].float()
                    td_target = (
                        nstep_returns
                        + clip_gamma * (1.0 - done) * next_pred.detach().cpu().float()
                    )
                else:
                    td_target = (
                        reward
                        + clip_gamma * (1.0 - done) * next_pred.detach().cpu().float()
                    )
                all_td_targets.append(td_target.view(-1))

            all_preds.append(pred_cpu)
            all_rewards.append(reward.float())

            if "returns" in batch:
                all_returns.append(batch["returns"].float())

            num_processed += pred_cpu.shape[0]
            if cli_args.max_samples and num_processed >= cli_args.max_samples:
                break

    # ── 6. Compute metrics ──────────────────────────────────────────────────
    preds = torch.cat(all_preds, dim=0).view(-1).numpy()
    rewards = torch.cat(all_rewards, dim=0).view(-1).numpy()
    has_returns = len(all_returns) > 0
    has_td_targets = len(all_td_targets) > 0
    if has_returns:
        returns = torch.cat(all_returns, dim=0).view(-1).numpy()
    if has_td_targets:
        td_targets = torch.cat(all_td_targets, dim=0).view(-1).numpy()

    if cli_args.max_samples:
        preds = preds[: cli_args.max_samples]
        rewards = rewards[: cli_args.max_samples]
        if has_returns:
            returns = returns[: cli_args.max_samples]
        if has_td_targets:
            td_targets = td_targets[: cli_args.max_samples]

    n = len(preds)
    print(f"\n{'=' * 60}")
    print(f"  Inference Results  (N = {n})")
    print(f"{'=' * 60}")

    # Prediction statistics
    print(
        f"  Pred   — mean: {preds.mean():.4f}  std: {preds.std():.4f}  "
        f"min: {preds.min():.4f}  max: {preds.max():.4f}"
    )
    print(
        f"  Reward — mean: {rewards.mean():.4f}  std: {rewards.std():.4f}  "
        f"min: {rewards.min():.4f}  max: {rewards.max():.4f}"
    )

    # Primary comparison: V(s) vs returns (raw sequence values)
    if has_returns:
        mse_ret = float(np.mean((preds - returns) ** 2))
        mae_ret = float(np.mean(np.abs(preds - returns)))
        print(f"\n  vs Cumulative Returns (sum of rewards in clip): [PRIMARY]")
        print(f"    MSE:              {mse_ret:.6f}")
        print(f"    MAE:              {mae_ret:.6f}")
        try:
            spearman_ret = _spearman_corr(preds, returns)
            print(f"    Spearman corr:    {spearman_ret:.4f}")
        except ImportError:
            pass

    if has_td_targets:
        print(
            f"\n  TD Tgt — mean: {td_targets.mean():.4f}  std: {td_targets.std():.4f}  "
            f"min: {td_targets.min():.4f}  max: {td_targets.max():.4f}"
        )
        mse_td = float(np.mean((preds - td_targets) ** 2))
        mae_td = float(np.mean(np.abs(preds - td_targets)))
        print(f"  vs TD Target  (G^H + γ^H*(1-d)*V_ema(s')):  [FOR REFERENCE]")
        print(f"    MSE:              {mse_td:.6f}")
        print(f"    MAE:              {mae_td:.6f}")
        try:
            spearman_td = _spearman_corr(preds, td_targets)
            print(f"    Spearman corr:    {spearman_td:.4f}")
        except ImportError:
            print(f"    Spearman corr:    (scipy not available)")

    print(f"{'=' * 60}")

    # ── Suffix outputs in baseline mode so they don't overwrite fine-tuned results ──
    dataset_name = getattr(args, "dataset_type", "unknown").upper()

    is_lora = getattr(args, "peft", "none") != "none"
    lora_name = "NoLoRA" if (baseline_mode or not is_lora) else "LoRA"

    plot_dir = cli_args.plot_dir
    output_file = cli_args.output_file

    if plot_dir == "inference_plots":
        # Automatically organize into subfolders by dataset and lora type
        plot_dir = os.path.join(plot_dir, dataset_name, lora_name)
    elif baseline_mode and plot_dir:
        plot_dir = plot_dir.rstrip("/") + "_baseline"

    if output_file and baseline_mode:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_baseline{ext}"

    # ── 7. Generate plots ───────────────────────────────────────────────────
    if has_returns:
        primary_target = returns
        primary_label = "Return"
    elif has_td_targets:
        primary_target = td_targets
        primary_label = "TD Target"
    else:
        primary_target = None
        primary_label = "Target"

    if plot_dir:
        _generate_plots(
            preds,
            primary_target,
            plot_dir,
            epoch,
            target_label=primary_label,
        )

    # ── 7b. Overlay comparison plots (if --compare_csv provided) ────────────
    if cli_args.compare_csv and plot_dir:
        _generate_overlay_plots(
            preds,
            primary_target,
            cli_args.compare_csv,
            plot_dir,
            epoch,
            target_label=primary_label,
        )

    # ── 7c. Multi-model comparison (if --compare_csvs provided) ────────────
    if cli_args.compare_csvs and plot_dir:
        # Parse 'Label:path.csv' format
        parsed = []
        for entry in cli_args.compare_csvs:
            if ":" in entry and not entry.startswith("/"):
                parts = entry.split(":", 1)
                lbl, path = parts[0], parts[1]
            else:
                lbl = os.path.splitext(os.path.basename(entry))[0]
                path = entry
            parsed.append((lbl, path))

        # Current run label
        if cli_args.label:
            curr_label = cli_args.label
        elif baseline_mode:
            curr_label = "Baseline (no LoRA)"
        else:
            curr_label = "Fine-tuned (LoRA)"

        _generate_multi_comparison_plots(
            current_preds=preds,
            current_targets=primary_target,
            current_label=curr_label,
            compare_csvs=parsed,
            plot_dir=plot_dir,
            epoch=epoch,
            target_label=primary_label,
        )

    # ── 8. Optional CSV output ──────────────────────────────────────────────
    if output_file:
        print(f"\nWriting per-sample results to: {output_file}")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx", "prediction", "reward"]
            if has_td_targets:
                header.append("td_target")
            if has_returns:
                header.append("return")
            writer.writerow(header)
            for i in range(n):
                row = [i, f"{preds[i]:.6f}", f"{rewards[i]:.6f}"]
                if has_td_targets:
                    row.append(f"{td_targets[i]:.6f}")
                if has_returns:
                    row.append(f"{returns[i]:.6f}")
                writer.writerow(row)
        print(f"  Wrote {n} rows.")

    # ── 9. Summary dict (for programmatic use) ──────────────────────────────
    summary = {
        "checkpoint": cli_args.checkpoint,
        "baseline": baseline_mode,
        "epoch": epoch,
        "num_samples": n,
        "gamma": gamma,
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
    }
    if has_td_targets:
        summary["td_target_mean"] = float(td_targets.mean())
        summary["mse_vs_td_target"] = mse_td
        summary["mae_vs_td_target"] = mae_td
        try:
            summary["spearman_vs_td_target"] = spearman_td
        except NameError:
            pass
    if has_returns:
        summary["mse_vs_return"] = mse_ret
        summary["mae_vs_return"] = mae_ret
        try:
            summary["spearman_vs_return"] = spearman_ret
        except NameError:
            pass

    # Save summary JSON alongside output
    if output_file:
        json_path = os.path.splitext(output_file)[0] + "_summary.json"
    else:
        suffix = "_baseline" if baseline_mode else ""
        json_path = f"inference_summary{suffix}.json"
        if plot_dir:
            json_path = os.path.join(plot_dir, json_path)

    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved to: {json_path}")

    return summary


if __name__ == "__main__":
    main()
