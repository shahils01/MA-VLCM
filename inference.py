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
        default=None,
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

    return p.parse_args()


# ─────────────────────── Metric Helpers ─────────────────────────────────────
def _pearson_corr(x, y):
    """Pearson correlation between two 1-D numpy arrays."""
    if len(x) < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _spearman_corr(x, y):
    """Spearman rank correlation between two 1-D numpy arrays."""
    if len(x) < 2:
        return float("nan")
    from scipy.stats import spearmanr
    corr, _ = spearmanr(x, y)
    return float(corr)


# ────────────────────────── Plotting ────────────────────────────────────────
def _generate_plots(preds, rewards, targets, plot_dir, epoch, target_label=None):
    """Generate comparison plots of predicted vs true values.

    Args:
        targets: TD targets, returns, or None — whatever the model was trained on.
        target_label: Label for the target values in plots (auto-detected if None).
    """
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in: {plot_dir}")

    has_targets = targets is not None
    if target_label is None:
        true_label = "TD Target" if has_targets else "Reward"
    else:
        true_label = target_label
    true_vals = targets if has_targets else rewards

    # ── 1. Scatter plot: Predicted vs True ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(true_vals, preds, alpha=0.35, s=12, c="#4C72B0", edgecolors="none")
    # Perfect prediction line
    lo = min(true_vals.min(), preds.min())
    hi = max(true_vals.max(), preds.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "--", color="#C44E52", linewidth=1.5, label="y = x (perfect)")
    # Linear fit
    if len(true_vals) > 1:
        coeffs = np.polyfit(true_vals, preds, 1)
        fit_x = np.linspace(lo - margin, hi + margin, 100)
        ax.plot(fit_x, np.polyval(coeffs, fit_x), "-",
                color="#55A868", linewidth=1.5,
                label=f"Linear fit (slope={coeffs[0]:.3f})")
    pearson = _pearson_corr(preds, true_vals)
    ax.set_xlabel(f"True {true_label}", fontsize=13)
    ax.set_ylabel("Predicted Value", fontsize=13)
    ax.set_title(f"Predicted vs True {true_label}  (epoch {epoch}, r={pearson:.3f})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "scatter_pred_vs_true.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Residual histogram ────────────────────────────────────────────
    residuals = preds - true_vals
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=min(80, max(20, len(residuals) // 50)),
            color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#C44E52", linestyle="--", linewidth=1.5)
    ax.axvline(residuals.mean(), color="#55A868", linestyle="-", linewidth=1.5,
              label=f"Mean = {residuals.mean():.4f}")
    ax.set_xlabel(f"Residual  (Predicted − True {true_label})", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(plot_dir, "residual_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. Per-sample comparison (first 500 samples) ─────────────────────
    show_n = min(500, len(preds))
    fig, ax = plt.subplots(figsize=(14, 5))
    idx = np.arange(show_n)
    ax.plot(idx, true_vals[:show_n], "-", color="#4C72B0", linewidth=1.0,
            alpha=0.8, label=f"True {true_label}")
    ax.plot(idx, preds[:show_n], "-", color="#C44E52", linewidth=1.0,
            alpha=0.8, label="Predicted")
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Value", fontsize=13)
    ax.set_title(f"Per-Sample Comparison  (first {show_n} samples)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "sample_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 4. If targets AND rewards are different, plot both ─────────────────
    if has_targets:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax_i, (vals, label) in enumerate(
            [(rewards, "Reward"), (targets, true_label)]
        ):
            ax = axes[ax_i]
            ax.scatter(vals, preds, alpha=0.35, s=12, c="#4C72B0", edgecolors="none")
            lo_i = min(vals.min(), preds.min())
            hi_i = max(vals.max(), preds.max())
            m_i = (hi_i - lo_i) * 0.05
            ax.plot([lo_i - m_i, hi_i + m_i], [lo_i - m_i, hi_i + m_i],
                    "--", color="#C44E52", linewidth=1.5)
            r_i = _pearson_corr(preds, vals)
            ax.set_xlabel(f"True {label}", fontsize=12)
            ax.set_ylabel("Predicted", fontsize=12)
            ax.set_title(f"Predicted vs {label}  (r={r_i:.3f})",
                         fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Epoch {epoch}", fontsize=14)
        fig.tight_layout()
        path = os.path.join(plot_dir, "scatter_reward_and_return.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def _generate_overlay_plots(preds, targets, compare_csv, plot_dir, epoch):
    """Overlay current run predictions with a previous run's CSV for comparison."""
    import pandas as pd

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(compare_csv)
    comp_preds = df["prediction"].values
    # Use td_target if available, else reward as the ground truth
    if "td_target" in df.columns:
        comp_targets = df["td_target"].values
        target_col = "td_target"
    else:
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
    true_label = "TD Target"
    n = min(len(preds), len(comp_preds))

    print(f"\nGenerating overlay comparison plots in: {plot_dir}")
    print(f"  Current: {curr_label}  ({len(preds)} samples)")
    print(f"  Compare: {comp_label}  ({len(comp_preds)} samples, from {compare_csv})")

    # ── 1. Overlay scatter: both predictions vs true target ──────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(true_vals[:n], comp_preds[:n], alpha=0.3, s=14,
              c="#8DA0CB", edgecolors="none", label=comp_label, zorder=2)
    ax.scatter(true_vals[:n], preds[:n], alpha=0.4, s=14,
              c="#E78AC3", edgecolors="none", label=curr_label, zorder=3)
    lo = min(true_vals[:n].min(), preds[:n].min(), comp_preds[:n].min())
    hi = max(true_vals[:n].max(), preds[:n].max(), comp_preds[:n].max())
    m = (hi - lo) * 0.05
    ax.plot([lo - m, hi + m], [lo - m, hi + m], "--",
            color="#333333", linewidth=1.5, alpha=0.6, label="y = x")
    r_curr = _pearson_corr(preds[:n], true_vals[:n])
    r_comp = _pearson_corr(comp_preds[:n], true_vals[:n])
    ax.set_xlabel(f"True {true_label}", fontsize=13)
    ax.set_ylabel("Predicted Value", fontsize=13)
    ax.set_title(f"Scatter Overlay  (r: {curr_label}={r_curr:.3f}, {comp_label}={r_comp:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Overlay residual histograms ───────────────────────────────────
    res_curr = preds[:n] - true_vals[:n]
    res_comp = comp_preds[:n] - true_vals[:n]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = min(60, max(20, n // 50))
    ax.hist(res_comp, bins=bins, color="#8DA0CB", alpha=0.6,
            edgecolor="white", label=f"{comp_label} (μ={res_comp.mean():.3f})")
    ax.hist(res_curr, bins=bins, color="#E78AC3", alpha=0.6,
            edgecolor="white", label=f"{curr_label} (μ={res_curr.mean():.3f})")
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel(f"Residual  (Predicted − True {true_label})", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Residual Distribution: Baseline vs Fine-tuned",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. Overlay per-sample comparison ─────────────────────────────────
    show_n = min(300, n)
    idx = np.arange(show_n)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, true_vals[:show_n], "-", color="#333333", linewidth=1.2,
            alpha=0.7, label=f"True {true_label}")
    ax.plot(idx, comp_preds[:show_n], "-", color="#8DA0CB", linewidth=1.0,
            alpha=0.7, label=comp_label)
    ax.plot(idx, preds[:show_n], "-", color="#E78AC3", linewidth=1.0,
            alpha=0.7, label=curr_label)
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Value", fontsize=13)
    ax.set_title(f"Per-Sample: Baseline vs Fine-tuned  (first {show_n})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_sample_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 4. MSE bar chart ────────────────────────────────────────────────
    mse_curr = float(np.mean((preds[:n] - true_vals[:n]) ** 2))
    mse_comp = float(np.mean((comp_preds[:n] - true_vals[:n]) ** 2))
    mae_curr = float(np.mean(np.abs(preds[:n] - true_vals[:n])))
    mae_comp = float(np.mean(np.abs(comp_preds[:n] - true_vals[:n])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_i, (metric, vals, title) in enumerate([
        ("MSE", [mse_comp, mse_curr], "MSE"),
        ("MAE", [mae_comp, mae_curr], "MAE"),
    ]):
        ax = axes[ax_i]
        bars = ax.bar([comp_label, curr_label], vals,
                      color=["#8DA0CB", "#E78AC3"], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Metrics Comparison (epoch {epoch})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plot_dir, "overlay_metrics_bar.png")
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
        baseline_sd = {k: v for k, v in cleaned_sd.items()
                       if k.startswith(custom_prefixes)}
        print(f"  BASELINE: Loading {len(baseline_sd)} custom head keys "
              f"(skipping {len(cleaned_sd) - len(baseline_sd)} backbone/LoRA keys)")
        missing, unexpected = model.load_state_dict(baseline_sd, strict=False)
        # Many "missing" keys expected (entire backbone) — only warn about custom heads
        missing_custom = [k for k in missing if k.startswith(custom_prefixes)]
        if missing_custom:
            print(f"  WARNING: {len(missing_custom)} custom head keys missing: {missing_custom}")
    else:
        missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

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
    args.loss_type = "td"       # Ensures include_next=True in webdataset_loader
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

    gamma = getattr(args, "gamma", 0.99)

    # ── 5. Inference loop ───────────────────────────────────────────────────
    print("Running inference...")
    print(f"  Computing TD targets with gamma={gamma}")
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

            # Compute TD target: r + γ*(1-done)*V(s')
            if "next_inputs" in batch:
                next_inputs = _move_and_cast(batch["next_inputs"])
                next_robot_obs = batch["next_robot_obs"].to(device=device, dtype=model_dtype)
                next_adj = batch["next_adj"].to(device=device, dtype=model_dtype)
                next_pred = model(next_inputs, next_robot_obs, next_adj)
                td_target = reward + gamma * (1.0 - done) * next_pred.detach().cpu().float()
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
    print(f"  Pred   — mean: {preds.mean():.4f}  std: {preds.std():.4f}  "
          f"min: {preds.min():.4f}  max: {preds.max():.4f}")
    print(f"  Reward — mean: {rewards.mean():.4f}  std: {rewards.std():.4f}  "
          f"min: {rewards.min():.4f}  max: {rewards.max():.4f}")

    # Primary comparison: V(s) vs TD target  (this is what the model was trained on)
    if has_td_targets:
        print(f"  TD Tgt — mean: {td_targets.mean():.4f}  std: {td_targets.std():.4f}  "
              f"min: {td_targets.min():.4f}  max: {td_targets.max():.4f}")
        mse_td = float(np.mean((preds - td_targets) ** 2))
        mae_td = float(np.mean(np.abs(preds - td_targets)))
        pearson_td = _pearson_corr(preds, td_targets)
        print(f"\n  vs TD Target  (r + γ*(1-done)*V(s')):  [PRIMARY — training objective]")
        print(f"    MSE:              {mse_td:.6f}")
        print(f"    MAE:              {mae_td:.6f}")
        print(f"    Pearson corr:     {pearson_td:.4f}")
        try:
            spearman_td = _spearman_corr(preds, td_targets)
            print(f"    Spearman corr:    {spearman_td:.4f}")
        except ImportError:
            print(f"    Spearman corr:    (scipy not available)")

    # Secondary: V(s) vs raw reward
    mse_reward = float(np.mean((preds - rewards) ** 2))
    mae_reward = float(np.mean(np.abs(preds - rewards)))
    pearson_reward = _pearson_corr(preds, rewards)
    print(f"\n  vs Raw Reward  (for reference):")
    print(f"    MSE:              {mse_reward:.6f}")
    print(f"    MAE:              {mae_reward:.6f}")
    print(f"    Pearson corr:     {pearson_reward:.4f}")
    try:
        spearman_reward = _spearman_corr(preds, rewards)
        print(f"    Spearman corr:    {spearman_reward:.4f}")
    except ImportError:
        print(f"    Spearman corr:    (scipy not available)")

    if has_returns:
        mse_ret = float(np.mean((preds - returns) ** 2))
        mae_ret = float(np.mean(np.abs(preds - returns)))
        pearson_ret = _pearson_corr(preds, returns)
        print(f"\n  vs Cumulative Returns (sum of rewards in clip):")
        print(f"    MSE:              {mse_ret:.6f}")
        print(f"    MAE:              {mae_ret:.6f}")
        print(f"    Pearson corr:     {pearson_ret:.4f}")
        try:
            spearman_ret = _spearman_corr(preds, returns)
            print(f"    Spearman corr:    {spearman_ret:.4f}")
        except ImportError:
            pass

    print(f"{'=' * 60}")

    # ── Suffix outputs in baseline mode so they don't overwrite fine-tuned results ──
    suffix = "_baseline" if baseline_mode else ""
    plot_dir = cli_args.plot_dir
    output_file = cli_args.output_file
    if suffix:
        if plot_dir:
            plot_dir = plot_dir.rstrip("/") + suffix
        if output_file:
            base, ext = os.path.splitext(output_file)
            output_file = f"{base}{suffix}{ext}"

    # ── 7. Generate plots ───────────────────────────────────────────────────
    td_or_returns = td_targets if has_td_targets else (returns if has_returns else None)

    if plot_dir:
        _generate_plots(
            preds,
            rewards,
            td_or_returns,
            plot_dir,
            epoch,
        )

    # ── 7b. Overlay comparison plots (if --compare_csv provided) ────────────
    if cli_args.compare_csv and plot_dir:
        _generate_overlay_plots(
            preds,
            td_or_returns,
            cli_args.compare_csv,
            plot_dir,
            epoch,
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
        "reward_mean": float(rewards.mean()),
        "mse_vs_reward": mse_reward,
        "mae_vs_reward": mae_reward,
        "pearson_vs_reward": pearson_reward,
    }
    if has_td_targets:
        summary["td_target_mean"] = float(td_targets.mean())
        summary["mse_vs_td_target"] = mse_td
        summary["mae_vs_td_target"] = mae_td
        summary["pearson_vs_td_target"] = pearson_td
    if has_returns:
        summary["mse_vs_return"] = mse_ret
        summary["mae_vs_return"] = mae_ret
        summary["pearson_vs_return"] = pearson_ret

    # Save summary JSON alongside output
    if output_file:
        json_path = os.path.splitext(output_file)[0] + "_summary.json"
    else:
        json_path = f"inference_summary{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved to: {json_path}")

    return summary


if __name__ == "__main__":
    main()


