#!/usr/bin/env python3
"""
visualize_shard.py – Run inference on a single trajectory shard and produce
an animated video showing the environment frames alongside a growing plot of
true (TD target) vs predicted returns over time.

Usage:
    PYTHONPATH=src python tools/visualize_shard.py \
        --checkpoint /path/to/ckpt_epoch_N.pt \
        --test_shards /path/to/test_shard \
        --shard_dir  /path/to/raw_trajectory_folder \
        --output_video shard_visualization.mp4 \
        --fps 5 \
        --dataset_type rware
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from tqdm import tqdm

import torch

# Reuse model and data utilities from the training script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ma_vlcm.model import ModelConfig, MultimodalValueModel
from ma_vlcm.train import (
    build_model,
    _apply_peft,
    webdataset_loader,
)


# ─────────────────────────────── Argument Parsing ───────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize single-shard inference with animated true vs predicted returns."
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
        help="Path or glob pattern for WebDataset shard(s) to evaluate.",
    )

    # ── Optional ──
    p.add_argument(
        "--shard_dir",
        type=str,
        default=None,
        help="Path to the raw trajectory folder containing overhead.png / "
        "rware_topdown.png files for visualization.",
    )
    p.add_argument(
        "--output_video",
        type=str,
        default="outputs/videos/shard_visualization.mp4",
        help="Output video filename (default: outputs/videos/shard_visualization.mp4).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Video frames per second (default: 5).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda:0, cpu",
    )
    p.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=["rware", "offroad"],
        help="Override the dataset_type saved in the checkpoint.",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode: skip LoRA adapters, keep LLaVA at pretrained weights.",
    )
    p.add_argument(
        "--disable_lora",
        action="store_true",
        help="Load the full checkpoint but disable LoRA adapter layers.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1 for sequential visualization).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max number of samples to process (default: 500). "
        "The webdataset repeats infinitely, so this caps the loop.",
    )

    return p.parse_args()


# ─────────────────────────── Load raw frame images ──────────────────────────
def _load_shard_images(shard_dir):
    """Load overhead.png, rware_topdown.png, and image.png for each step in the shard.

    Returns:
        overhead_imgs: list of PIL Images (or None per step)
        topdown_imgs:  list of PIL Images (or None per step)
        offroad_imgs:  list of PIL Images (or None per step)
        num_steps:     number of steps found
    """
    if shard_dir is None:
        return [], [], [], 0

    # Discover step files
    overhead_files = sorted(glob.glob(os.path.join(shard_dir, "*overhead.png")))
    topdown_files = sorted(glob.glob(os.path.join(shard_dir, "*rware_topdown.png")))
    offroad_files = sorted(glob.glob(os.path.join(shard_dir, "*image.png")))

    import re

    # Build a mapping step_idx -> files
    def _extract_step_idx(fpath):
        """Extract step index from filenames like trajectory_xxx_stepNNNN.overhead.png
        or traj_009_step_0000.image.png
        """
        base = os.path.basename(fpath)
        match = re.search(r"step_?(\d+)", base)
        if match:
            return int(match.group(1))
        return -1

    overhead_map = {}
    for f in overhead_files:
        idx = _extract_step_idx(f)
        if idx >= 0:
            overhead_map[idx] = f

    topdown_map = {}
    for f in topdown_files:
        idx = _extract_step_idx(f)
        if idx >= 0:
            topdown_map[idx] = f

    offroad_map = {}
    for f in offroad_files:
        idx = _extract_step_idx(f)
        if idx >= 0:
            offroad_map[idx] = f

    all_indices = sorted(
        set(
            list(overhead_map.keys())
            + list(topdown_map.keys())
            + list(offroad_map.keys())
        )
    )
    if not all_indices:
        print(f"  WARNING: No step images found in {shard_dir}")
        return [], [], [], 0

    num_steps = max(all_indices) + 1

    overhead_imgs = []
    topdown_imgs = []
    offroad_imgs = []
    for i in range(num_steps):
        if i in overhead_map:
            overhead_imgs.append(Image.open(overhead_map[i]).convert("RGB"))
        else:
            overhead_imgs.append(None)
        if i in topdown_map:
            topdown_imgs.append(Image.open(topdown_map[i]).convert("RGB"))
        else:
            topdown_imgs.append(None)
        if i in offroad_map:
            offroad_imgs.append(Image.open(offroad_map[i]).convert("RGB"))
        else:
            offroad_imgs.append(None)

    print(
        f"  Loaded {len([x for x in overhead_imgs if x])} overhead, "
        f"{len([x for x in topdown_imgs if x])} topdown, and "
        f"{len([x for x in offroad_imgs if x])} offroad images from {shard_dir}"
    )
    return overhead_imgs, topdown_imgs, offroad_imgs, num_steps


# ──────────────────────────────── Main ──────────────────────────────────────
def main():
    cli_args = parse_args()

    # ── 1. Load checkpoint ──────────────────────────────────────────────────
    print(f"Loading checkpoint: {cli_args.checkpoint}")
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)

    saved_args_dict = ckpt.get("args", {})
    args = SimpleNamespace(**saved_args_dict)

    if cli_args.dataset_type is not None:
        args.dataset_type = cli_args.dataset_type

    # Apply CLI overrides
    args.batch_size = cli_args.batch_size
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    else:
        # Default to 0 workers for single-shard visualization to avoid
        # "fewer shards than workers" errors
        args.num_workers = 0

    # Always bf16
    args.mixed_precision = "bf16"

    # Ensure critical attributes exist
    args.preprocess_in_loader = getattr(args, "preprocess_in_loader", True)
    args.video_preprocessed = getattr(args, "video_preprocessed", True)
    args.compile = getattr(args, "compile", False)
    args.text_prompt_template = getattr(args, "text_prompt_template", None)

    # Re-apply quantization config if trained with qlora
    if getattr(args, "peft", None) == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("QLoRA requested but bitsandbytes not available.") from e
        args.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        args.quantization_config = None

    # ── 2. Determine device ─────────────────────────────────────────────────
    if cli_args.device:
        device = torch.device(cli_args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── 3. Rebuild model + load weights ─────────────────────────────────────
    baseline_mode = cli_args.baseline
    if baseline_mode:
        print("Building model in BASELINE mode (no LoRA)...")
        saved_peft = getattr(args, "peft", "none")
        args.peft = "none"
        model = build_model(args, device=device)
        model = _apply_peft(model, args)
        args.peft = saved_peft
    else:
        print(f"Building model (with {getattr(args, 'peft', 'none')})...")
        model = build_model(args, device=device)
        model = _apply_peft(model, args)

    state_dict = ckpt["model"]
    cleaned_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_sd[new_k] = v

    if baseline_mode:
        custom_prefixes = ("robot_gnn.", "value_head.", "obs_to_lm.")
        baseline_sd = {
            k: v for k, v in cleaned_sd.items() if k.startswith(custom_prefixes)
        }
        print(f"  BASELINE: Loading {len(baseline_sd)} custom head keys")
        model.load_state_dict(baseline_sd, strict=False)
    else:
        missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(
                f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}"
            )

    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded checkpoint from epoch {epoch}")

    # ── 3b. Optionally disable LoRA adapters ────────────────────────────────
    if getattr(cli_args, "disable_lora", False) and not baseline_mode:
        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError("--disable_lora requires `peft`. pip install peft.")
        disabled_count = 0
        for name, module in model.named_modules():
            if isinstance(module, PeftModel):
                module.disable_adapter_layers()
                disabled_count += 1
                print(f"  [DISABLE_LORA] Disabled LoRA on '{name}'")
        if disabled_count == 0:
            print("  [DISABLE_LORA] WARNING: No PeftModel found.")

    model_dtype = torch.bfloat16
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    # ── 4. Compute clip_gamma and load EMA shadow ───────────────────────────
    gamma = getattr(args, "gamma", 0.95)
    clip_len = getattr(args, "clip_len", 16)
    clip_gamma = gamma**clip_len

    ema_shadow = ckpt.get("ema_shadow", None)
    if ema_shadow is not None:
        print(f"  EMA shadow loaded ({len(ema_shadow)} params)")
    else:
        print("  No EMA shadow — using online weights for bootstrap")

    # ── 5. Build test data loader ───────────────────────────────────────────
    print(f"Loading test data from: {cli_args.test_shards}")
    args.train_shards = cli_args.test_shards

    # Force include_next=True for TD target computation
    saved_loss_type = getattr(args, "loss_type", "td")
    saved_return_mode = getattr(args, "return_mode", "td")
    args.loss_type = "td"
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
        dataset_type=getattr(args, "dataset_type", None),
    )

    args.loss_type = saved_loss_type
    args.return_mode = saved_return_mode

    # ── 6. Load raw shard images ────────────────────────────────────────────
    overhead_imgs, topdown_imgs, offroad_imgs, raw_num_steps = _load_shard_images(
        cli_args.shard_dir
    )
    has_images = (
        len(overhead_imgs) > 0 or len(topdown_imgs) > 0 or len(offroad_imgs) > 0
    )

    # ── 7. Inference loop ───────────────────────────────────────────────────
    def _move_and_cast(tensor_dict):
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

    print(f"\nRunning inference (clip_gamma={clip_gamma:.6f})...")
    all_preds = []
    all_td_targets = []
    all_rewards = []
    all_returns = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inference", dynamic_ncols=True):
            inputs = _move_and_cast(batch["inputs"])
            robot_obs = batch["robot_obs"].to(device=device, dtype=model_dtype)
            adj = batch["adj"].to(device=device, dtype=model_dtype)
            reward = batch["reward"]
            done = batch["done"].float()

            # Forward pass: V(s)
            pred = model(inputs, robot_obs, adj)
            pred_cpu = pred.detach().cpu().float()

            # Compute bootstrapped TD target matching train.py
            if "next_inputs" in batch:
                next_inputs = _move_and_cast(batch["next_inputs"])
                next_robot_obs = batch["next_robot_obs"].to(
                    device=device, dtype=model_dtype
                )
                next_adj = batch["next_adj"].to(device=device, dtype=model_dtype)

                # Use EMA weights for bootstrap if available
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

                # Match train.py: nstep returns + clip_gamma bootstrap
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

            all_preds.append(pred_cpu.view(-1))
            all_rewards.append(reward.float().view(-1))
            if "returns" in batch:
                all_returns.append(batch["returns"].float().view(-1))

            num_processed = sum(p.shape[0] for p in all_preds)
            if num_processed >= cli_args.max_samples:
                break

    preds = torch.cat(all_preds, dim=0).numpy()
    has_td_targets = len(all_td_targets) > 0
    has_returns = len(all_returns) > 0

    if has_td_targets:
        td_targets = torch.cat(all_td_targets, dim=0).numpy()
    if has_returns:
        returns = torch.cat(all_returns, dim=0).numpy()

    # Determine the primary target (what the model was trained on)
    if has_td_targets:
        targets = td_targets
        target_label = "TD Target"
    elif has_returns:
        targets = returns
        target_label = "Return"
    else:
        targets = torch.cat(all_rewards, dim=0).numpy()
        target_label = "Reward"

    n_samples = len(preds)
    print(f"\n  Total samples: {n_samples}")
    print(f"  Target type: {target_label}")
    print(f"  Pred range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")

    # ── 8. Generate animated video ──────────────────────────────────────────
    print(f"\nGenerating video ({n_samples} frames at {cli_args.fps} fps)...")

    if has_images:
        # Layout: top row can be [offroad] or [overhead | topdown], bottom = animated plot
        fig = plt.figure(figsize=(14, 14))
        if len(offroad_imgs) > 0:
            gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.25)
            ax_offroad = fig.add_subplot(gs[0, 0])
            ax_plot = fig.add_subplot(gs[1, 0])
            ax_overhead = None
            ax_topdown = None
        else:
            gs = fig.add_gridspec(
                2, 2, height_ratios=[2.5, 1], hspace=0.25, wspace=0.05
            )
            ax_overhead = fig.add_subplot(gs[0, 0])
            ax_topdown = fig.add_subplot(gs[0, 1])
            ax_plot = fig.add_subplot(gs[1, :])
            ax_offroad = None
    else:
        # No images — just the plot
        fig, ax_plot = plt.subplots(figsize=(14, 5))

    # Configure the animated plot
    ax_plot.set_xlim(0, max(n_samples - 1, 1))

    # Compute y limits with some margin
    all_vals = np.concatenate([preds, targets])
    y_min, y_max = all_vals.min(), all_vals.max()
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
    ax_plot.set_ylim(y_min - y_margin, y_max + y_margin)

    ax_plot.set_xlabel("Time Step", fontsize=13)
    ax_plot.set_ylabel("Value", fontsize=13)
    ax_plot.grid(True, alpha=0.3)

    (line_true,) = ax_plot.plot(
        [],
        [],
        "-",
        color="#4C72B0",
        linewidth=2.0,
        alpha=0.9,
        label=f"True {target_label}",
    )
    (line_pred,) = ax_plot.plot(
        [], [], "-", color="#C44E52", linewidth=2.0, alpha=0.9, label="Predicted V(s)"
    )
    ax_plot.legend(fontsize=12, loc="upper left")

    # Title text (will be updated per frame)
    title_text = ax_plot.set_title("", fontsize=14, fontweight="bold")

    # Image axes setup
    if has_images:
        if ax_offroad is not None:
            ax_offroad.set_xticks([])
            ax_offroad.set_yticks([])
            ax_offroad.set_title("Offroad Environment", fontsize=15, fontweight="bold")
            first_o = next((i for i in offroad_imgs if i is not None), None)
            if first_o is not None:
                arr_shape = np.array(first_o).shape
                shape_o = (int(arr_shape[0]), int(arr_shape[1]), 3)
            else:
                shape_o = (168, 168, 3)
            im_offroad = ax_offroad.imshow(np.zeros(shape_o, dtype=np.uint8))
        else:
            ax_overhead.set_xticks([])
            ax_overhead.set_yticks([])
            ax_overhead.set_title(
                "Warehouse Environment (Isaac Sim)", fontsize=15, fontweight="bold"
            )

            ax_topdown.set_xticks([])
            ax_topdown.set_yticks([])
            ax_topdown.set_title(
                "Warehouse Environment (PyGame)", fontsize=15, fontweight="bold"
            )

            first_oh = next((i for i in overhead_imgs if i is not None), None)
            shape_oh = (
                np.array(first_oh).shape if first_oh is not None else (168, 168, 3)
            )
            first_td = next((i for i in topdown_imgs if i is not None), None)
            shape_td = (
                np.array(first_td).shape if first_td is not None else (168, 168, 3)
            )

            # Initialize with placeholder
            im_overhead = ax_overhead.imshow(np.zeros(shape_oh, dtype=np.uint8))
            im_topdown = ax_topdown.imshow(np.zeros(shape_td, dtype=np.uint8))

    def _update(frame_idx):
        """Update function for FuncAnimation."""
        t = frame_idx

        # Update the growing lines
        x_data = np.arange(t + 1)
        line_true.set_data(x_data, targets[: t + 1])
        line_pred.set_data(x_data, preds[: t + 1])

        # Update title
        err = abs(preds[t] - targets[t])
        title_text.set_text(
            f"Step {t}/{n_samples - 1}  |  "
            f"Pred: {preds[t]:.4f}  |  True: {targets[t]:.4f}  |  "
            f"|Error|: {err:.4f}"
        )

        artists = [line_true, line_pred, title_text]

        # Update images if available
        if has_images:
            # Map inference sample index to raw step index
            # The inference samples may not map 1:1 to raw steps (due to clipping),
            # so we use a proportional mapping
            if raw_num_steps > 0 and n_samples > 0:
                raw_idx = min(
                    int(t * raw_num_steps / n_samples),
                    raw_num_steps - 1,
                )
            else:
                raw_idx = t

            if ax_offroad is not None:
                if raw_idx < len(offroad_imgs) and offroad_imgs[raw_idx] is not None:
                    im_offroad.set_data(np.array(offroad_imgs[raw_idx]))
                artists.append(im_offroad)
            else:
                if raw_idx < len(overhead_imgs) and overhead_imgs[raw_idx] is not None:
                    im_overhead.set_data(np.array(overhead_imgs[raw_idx]))
                if raw_idx < len(topdown_imgs) and topdown_imgs[raw_idx] is not None:
                    im_topdown.set_data(np.array(topdown_imgs[raw_idx]))
                artists.extend([im_overhead, im_topdown])

        return artists

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_samples,
        interval=1000 // cli_args.fps,
        blit=True,
    )

    # Save the video
    output_path = cli_args.output_video
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    try:
        writer = animation.FFMpegWriter(fps=cli_args.fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"\n  ✅ Video saved to: {output_path}")
    except Exception as e:
        print(f"\n  FFmpeg not available ({e}), trying Pillow writer...")
        # Fallback: save as GIF
        gif_path = output_path.rsplit(".", 1)[0] + ".gif"
        try:
            writer = animation.PillowWriter(fps=cli_args.fps)
            anim.save(gif_path, writer=writer)
            print(f"\n  ✅ GIF saved to: {gif_path}")
        except Exception as e2:
            print(f"\n  ❌ Could not save animation: {e2}")
            print("    Install ffmpeg: conda install ffmpeg")

    plt.close(fig)

    # ── 9. Print summary ────────────────────────────────────────────────────
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    print(f"\n{'=' * 50}")
    print(f"  Shard Visualization Summary")
    print(f"{'=' * 50}")
    print(f"  Samples:    {n_samples}")
    print(f"  Target:     {target_label}")
    print(f"  MSE:        {mse:.6f}")
    print(f"  MAE:        {mae:.6f}")
    print(f"  Output:     {output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
