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
    print("Building model...")
    model = build_model(args, device=device)
    model = _apply_peft(model, args)

    # Load state dict
    state_dict = ckpt["model"]
    # Handle potential "module." prefix from DDP/FSDP wrapping
    cleaned_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_sd[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded checkpoint from epoch {epoch}")

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

    test_loader = webdataset_loader(
        args,
        shards=cli_args.test_shards,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # ── 5. Inference loop ───────────────────────────────────────────────────
    print("Running inference...")
    all_preds = []
    all_rewards = []
    all_returns = []
    num_processed = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inference", dynamic_ncols=True):
            # Move inputs to device
            inputs = {}
            for k, v in batch["inputs"].items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v

            robot_obs = batch["robot_obs"].to(device)
            adj = batch["adj"].to(device)
            reward = batch["reward"]  # keep on CPU for collection
            done = batch["done"]

            # Forward pass
            pred = model(inputs, robot_obs, adj)
            pred_cpu = pred.detach().cpu().float()

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
    if has_returns:
        returns = torch.cat(all_returns, dim=0).view(-1).numpy()

    if cli_args.max_samples:
        preds = preds[: cli_args.max_samples]
        rewards = rewards[: cli_args.max_samples]
        if has_returns:
            returns = returns[: cli_args.max_samples]

    n = len(preds)
    print(f"\n{'=' * 60}")
    print(f"  Inference Results  (N = {n})")
    print(f"{'=' * 60}")

    # Prediction statistics
    print(f"  Pred   — mean: {preds.mean():.4f}  std: {preds.std():.4f}  "
          f"min: {preds.min():.4f}  max: {preds.max():.4f}")
    print(f"  Reward — mean: {rewards.mean():.4f}  std: {rewards.std():.4f}  "
          f"min: {rewards.min():.4f}  max: {rewards.max():.4f}")

    # MSE and MAE vs reward
    mse_reward = float(np.mean((preds - rewards) ** 2))
    mae_reward = float(np.mean(np.abs(preds - rewards)))
    pearson_reward = _pearson_corr(preds, rewards)
    print(f"\n  vs Reward:")
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
        print(f"\n  vs Returns (n-step / TD target):")
        print(f"    MSE:              {mse_ret:.6f}")
        print(f"    MAE:              {mae_ret:.6f}")
        print(f"    Pearson corr:     {pearson_ret:.4f}")
        try:
            spearman_ret = _spearman_corr(preds, returns)
            print(f"    Spearman corr:    {spearman_ret:.4f}")
        except ImportError:
            pass

    print(f"{'=' * 60}")

    # ── 7. Optional CSV output ──────────────────────────────────────────────
    if cli_args.output_file:
        print(f"\nWriting per-sample results to: {cli_args.output_file}")
        with open(cli_args.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx", "prediction", "reward"]
            if has_returns:
                header.append("return")
            writer.writerow(header)
            for i in range(n):
                row = [i, f"{preds[i]:.6f}", f"{rewards[i]:.6f}"]
                if has_returns:
                    row.append(f"{returns[i]:.6f}")
                writer.writerow(row)
        print(f"  Wrote {n} rows.")

    # ── 8. Summary dict (for programmatic use) ──────────────────────────────
    summary = {
        "checkpoint": cli_args.checkpoint,
        "epoch": epoch,
        "num_samples": n,
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "reward_mean": float(rewards.mean()),
        "mse_vs_reward": mse_reward,
        "mae_vs_reward": mae_reward,
        "pearson_vs_reward": pearson_reward,
    }
    if has_returns:
        summary["mse_vs_return"] = mse_ret
        summary["mae_vs_return"] = mae_ret
        summary["pearson_vs_return"] = pearson_ret

    # Save summary JSON alongside output
    summary_path = cli_args.output_file
    if summary_path:
        json_path = summary_path.rsplit(".", 1)[0] + "_summary.json"
    else:
        json_path = "inference_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved to: {json_path}")

    return summary


if __name__ == "__main__":
    main()
