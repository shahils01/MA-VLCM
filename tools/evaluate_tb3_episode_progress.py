#!/usr/bin/env python3
"""Compare full-episode TB3 progress predictions from all MA-VLCM backbones.

The evaluator selects complete held-out episode shards, resolves the newest run
and highest saved epoch for each requested backbone, performs sliding-window
inference over every episode, and writes CSVs, static plots, and animated videos.
"""

import argparse
import csv
import gc
import io
import json
import math
import os
import random
import re
import tarfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from ma_vlcm.train import (
    _apply_peft,
    _apply_value_output_activation,
    _expand_hf_dataset_shards,
    build_model,
    split_shards,
    webdataset_loader,
)


MODEL_ORDER = ("llava", "qwen3_vl", "vjepa2")
MODEL_LABELS = {
    "llava": "LLaVA-OneVision",
    "qwen3_vl": "Qwen3-VL",
    "vjepa2": "V-JEPA2",
}
MODEL_COLORS = {
    "llava": "#d62728",
    "qwen3_vl": "#2ca02c",
    "vjepa2": "#9467bd",
}
MODEL_FILENAME_TOKENS = {
    "llava": (
        "llava_onevision",
        "llava-onevision",
        "llava",
        "tb3_isaac_0.5b",
        "turtlebot_0.5b",
    ),
    "qwen3_vl": ("qwen3_vl", "qwen3vl"),
    "vjepa2": ("vjepa2", "v_jepa2"),
}
EXPECTED_BACKENDS = {
    "llava": "llava_onevision",
    "qwen3_vl": "qwen3_vl",
    "vjepa2": "vjepa2",
}
CHECKPOINT_RE = re.compile(
    r"(?P<timestamp>\d{8}_\d{6})_epoch_(?P<epoch>\d+)\.pt$"
)
STEP_RE = re.compile(r"step_?(\d+)", re.IGNORECASE)


def _default_checkpoint_root():
    configured = os.environ.get("MA_VLCM_CHECKPOINT_ROOT")
    if configured:
        return configured
    user_name = os.environ.get("USER", "aparame")
    scratch = Path(f"/scratch/{user_name}/ma_vlcm/checkpoints")
    if scratch.exists():
        return str(scratch)
    return "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the latest LLaVA, Qwen3-VL, and V-JEPA2 checkpoints on "
            "the same complete held-out TurtleBot episodes."
        )
    )
    parser.add_argument("--checkpoint-root", default=_default_checkpoint_root())
    parser.add_argument(
        "--dataset",
        default="hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar",
        help="Episode-shard directory, glob, or Hugging Face dataset pattern.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/plots/tb3_episode_inference"
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument(
        "--num-failed-episodes",
        type=int,
        default=0,
        help=(
            "Require this many unsuccessful episodes among the selected episodes. "
            "Requires episode_success labels in the progress metadata."
        ),
    )
    parser.add_argument("--episode-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--minimum-frames", type=int, default=16)
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
    )
    parser.add_argument(
        "--episode-shards",
        nargs="*",
        default=None,
        help="Explicit episode tar files; bypasses automatic held-out selection.",
    )
    parser.add_argument("--llava-checkpoint", default=None)
    parser.add_argument("--qwen3-vl-checkpoint", default=None)
    parser.add_argument("--vjepa2-checkpoint", default=None)
    parser.add_argument(
        "--no-video", action="store_true", help="Write CSV/PNG only."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory name; defaults to a UTC timestamp.",
    )
    return parser.parse_args()


def _checkpoint_sort_key(path):
    match = CHECKPOINT_RE.search(path.name)
    if match is None:
        return None
    return match.group("timestamp"), int(match.group("epoch")), path.stat().st_mtime


def find_latest_checkpoint(checkpoint_root, model_name, explicit=None):
    """Choose the newest run timestamp, then its highest saved epoch."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{model_name} checkpoint not found: {path}")
        return path

    root = Path(checkpoint_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    tokens = MODEL_FILENAME_TOKENS[model_name]
    candidates = []
    for path in root.rglob("*.pt"):
        lowered = path.name.lower()
        key = _checkpoint_sort_key(path)
        if key is not None and any(token in lowered for token in tokens):
            candidates.append((key, path))
    if not candidates:
        option = model_name.replace("_", "-")
        raise FileNotFoundError(
            f"No timestamped epoch checkpoint for {MODEL_LABELS[model_name]} "
            f"under {root}. Supply --{option}-checkpoint explicitly."
        )

    newest_timestamp = max(key[0] for key, _ in candidates)
    newest_run = [item for item in candidates if item[0][0] == newest_timestamp]
    return max(newest_run, key=lambda item: (item[0][1], item[0][2]))[1]


def _expand_local_shards(source):
    source = str(source)
    if source.startswith("hf://datasets/"):
        expanded = _expand_hf_dataset_shards(source)
        return [Path(path).resolve() for path in expanded]
    path = Path(source).expanduser()
    if path.is_dir():
        return sorted(item.resolve() for item in path.rglob("*.tar"))
    if any(char in source for char in "*?["):
        import glob

        return [Path(item).resolve() for item in sorted(glob.glob(source, recursive=True))]
    if path.is_file():
        return [path.resolve()]
    raise FileNotFoundError(f"No episode shards found for: {source}")


def _step_from_name(name):
    match = STEP_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def inspect_episode_shard(shard_path, load_frames=False):
    """Read raw steps, targets, and optionally overhead frames from one episode."""
    shard_path = Path(shard_path)
    frames = {}
    targets = {}
    steps = set()
    state_episode_id = None
    episode_success = None
    state_episode_success = None
    latest_state_step = -1
    with tarfile.open(shard_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            step = _step_from_name(member.name)
            if step is None:
                continue
            if member.name.endswith(".overhead.png"):
                steps.add(step)
                if load_frames:
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        with Image.open(io.BytesIO(extracted.read())) as image:
                            frames[step] = image.convert("RGB").copy()
            elif member.name.endswith(".progress.json"):
                extracted = archive.extractfile(member)
                if extracted is not None:
                    payload = json.loads(extracted.read().decode("utf-8"))
                    if isinstance(payload, dict):
                        value = payload.get("target", payload.get("team_progress"))
                        if "episode_success" in payload:
                            episode_success = bool(payload["episode_success"])
                    else:
                        value = payload
                    if value is None:
                        continue
                    targets[step] = float(value)
            elif member.name.endswith(".state.json"):
                extracted = archive.extractfile(member)
                if extracted is not None:
                    state = json.loads(extracted.read().decode("utf-8"))
                    episode_meta = state.get("episode_meta", {})
                    if state_episode_id is None:
                        state_episode_id = episode_meta.get("episode_id")
                    if step >= latest_state_step:
                        latest_state_step = step
                        outcome = str(episode_meta.get("outcome", "")).lower()
                        if episode_meta.get("success", False) or outcome == "success":
                            state_episode_success = True
                        elif episode_meta.get("failure", False) or outcome == "failure":
                            state_episode_success = False

    ordered_steps = sorted(steps)
    source_parts = list(shard_path.with_suffix("").parts[-3:])
    source_id = "_".join(source_parts)
    episode_id = str(state_episode_id or source_id)
    return {
        "path": str(shard_path.resolve()),
        "episode_id": episode_id,
        "source_id": source_id,
        "steps": ordered_steps,
        "targets": targets,
        "frames": frames,
        "frame_count": len(ordered_steps),
        "episode_success": (
            episode_success
            if episode_success is not None
            else state_episode_success
        ),
    }


def select_episode_shards(args):
    if args.episode_shards:
        candidates = [Path(path).expanduser().resolve() for path in args.episode_shards]
    else:
        all_shards = _expand_local_shards(args.dataset)
        _, held_out = split_shards(
            [str(path) for path in all_shards], args.val_split, args.split_seed
        )
        candidates = [Path(path) for path in (held_out or all_shards)]

    eligible = []
    for path in tqdm(candidates, desc="Inspecting episode shards", dynamic_ncols=True):
        info = inspect_episode_shard(path, load_frames=False)
        if info["frame_count"] >= args.minimum_frames:
            eligible.append((path, info))
    if len(eligible) < args.num_episodes:
        raise RuntimeError(
            f"Requested {args.num_episodes} episodes with at least "
            f"{args.minimum_frames} frames, but found {len(eligible)}."
        )
    num_failed = int(args.num_failed_episodes)
    if not 0 <= num_failed <= args.num_episodes:
        raise ValueError(
            "--num-failed-episodes must be between zero and --num-episodes"
        )
    rng = random.Random(args.episode_seed)
    if num_failed:
        num_successful = args.num_episodes - num_failed
        failed = [item for item in eligible if item[1]["episode_success"] is False]
        successful = [item for item in eligible if item[1]["episode_success"] is True]
        unknown = [item for item in eligible if item[1]["episode_success"] is None]
        if len(failed) < num_failed or len(successful) < num_successful:
            raise RuntimeError(
                "Requested an outcome-stratified evaluation with "
                f"{num_successful} successful and {num_failed} failed episodes, but "
                f"found {len(successful)} successful, {len(failed)} failed, and "
                f"{len(unknown)} without episode_success labels."
            )
        selected = rng.sample(successful, num_successful) + rng.sample(
            failed, num_failed
        )
        rng.shuffle(selected)
    else:
        selected = rng.sample(eligible, args.num_episodes)
    return [item[1] for item in selected]


def _load_checkpoint_file(path):
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.load(path, **kwargs)


def _ensure_inference_args(args):
    defaults = {
        "dataset_type": "tb3_lab",
        "batch_size": 1,
        "num_workers": 0,
        "preprocess_in_loader": True,
        "video_preprocessed": True,
        "compile": False,
        "text_prompt_template": None,
        "target_mode": "progress",
        "return_mode": "nstep",
        "n_step": 50,
        "gamma": 0.95,
        "clip_len": 16,
        "clip_stride": 1,
        "robot_source": "state",
        "reward_reduce": "mean",
        "done_reduce": "any",
        "rware_config": "mixed-rware",
        "resize_width": 672,
        "resize_height": 336,
        "rware_visual_mode": "rware_only",
        "max_return_horizon": 64,
        "vl_dtype": "bfloat16",
        "mixed_precision": "bf16",
        "peft": "none",
        "lora_scope": "all",
        "lora_target_modules": "",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "freeze_vl": False,
        "freeze_vision_tower": False,
        "contrastive_depth_offsets_list": [0],
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    args.dataset_type = "tb3_lab"
    args.target_mode = "progress"
    args.return_mode = "nstep"
    args.batch_size = 1
    args.num_workers = 0
    args.preprocess_in_loader = True
    args.video_preprocessed = True
    args.quantization_config = None
    if args.peft == "qlora":
        from transformers import BitsAndBytesConfig

        compute_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(args.vl_dtype, torch.bfloat16)
        args.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return args


def load_model_from_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = _load_checkpoint_file(checkpoint_path)
    saved_args = checkpoint.get("args", {})
    model_args = _ensure_inference_args(SimpleNamespace(**saved_args))
    model = build_model(model_args, device=device)
    model = _apply_peft(model, model_args)

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(
            f"  WARNING: {len(unexpected)} unexpected keys "
            f"(first 5): {unexpected[:5]}"
        )
    del state_dict
    del checkpoint
    gc.collect()

    precision = getattr(model_args, "mixed_precision", "no")
    model_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(precision, torch.float32)
    if model_args.peft != "qlora":
        model = model.to(device=device, dtype=model_dtype)
    else:
        model = model.to(device=device)
    if hasattr(model.backbone.model, "gradient_checkpointing_disable"):
        model.backbone.model.gradient_checkpointing_disable()
    model.eval()
    return model, model_args, model_dtype


def _move_inputs(inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def infer_episode(model, model_args, model_dtype, episode, device, clip_stride):
    args = SimpleNamespace(**vars(model_args))
    args.clip_stride = clip_stride
    args.target_mode = "progress"
    args.batch_size = 1
    args.num_workers = 0
    # Checkpoint training may use adjacent clips, but episode inference emits
    # one prediction for every complete clip window.
    args.temporal_consistency_loss_weight = 0.0
    loader = webdataset_loader(
        args,
        shards=episode["path"],
        batch_size=1,
        num_workers=0,
        shuffle=False,
        dataset_type="tb3_lab",
        # The episode was already selected from the held-out shards. Training
        # checkpoints may retain balance_tb3_sources=True, but balancing a
        # single explicit tar path is neither valid nor useful for inference.
        balance_tb3_sources=False,
    )

    clip_len = int(args.clip_len)
    raw_steps = episode["steps"]
    expected = max(0, math.ceil((len(raw_steps) - clip_len + 1) / clip_stride))
    rows = []
    iterator = iter(loader)
    with torch.inference_mode():
        for index in tqdm(
            range(expected),
            desc=f"{MODEL_LABELS.get(args.vl_backend, args.vl_backend)}: {episode['source_id']}",
            dynamic_ncols=True,
        ):
            batch = next(iterator)
            inputs = _move_inputs(batch["inputs"], device, model_dtype)
            robot_obs = batch["robot_obs"].to(device=device, dtype=model_dtype)
            adj = batch["adj"].to(device=device, dtype=model_dtype)
            prediction = model(inputs, robot_obs, adj)
            prediction = _apply_value_output_activation(prediction, args)
            end_position = min(clip_len - 1 + index * clip_stride, len(raw_steps) - 1)
            step = raw_steps[end_position]
            target = float(batch["progress"].view(-1)[0])
            rows.append(
                {
                    "step": int(step),
                    "prediction": float(prediction.detach().float().cpu().view(-1)[0]),
                    "target": target,
                }
            )
    return rows


def _prediction_at_or_before(rows, step):
    available = [row for row in rows if row["step"] <= step]
    return available[-1]["prediction"] if available else None


def _evaluation_targets(episode, results):
    """Prefer loader targets so plots use the checkpoint training schema."""

    targets = dict(episode["targets"])
    aligned = {}
    for rows in results.values():
        for row in rows:
            step = row["step"]
            target = row["target"]
            if step in aligned and not math.isclose(
                aligned[step], target, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise RuntimeError(
                    "Compared checkpoints produced different target schemas at "
                    f"step {step}: {aligned[step]} versus {target}."
                )
            aligned[step] = target
    targets.update(aligned)
    return targets


def write_episode_csv(path, episode, results):
    targets = _evaluation_targets(episode, results)
    maps = {
        name: {row["step"]: row["prediction"] for row in rows}
        for name, rows in results.items()
    }
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "target"] + [f"prediction_{name}" for name in results],
        )
        writer.writeheader()
        for step in episode["steps"]:
            row = {"step": step, "target": targets.get(step, "")}
            for name in results:
                row[f"prediction_{name}"] = maps[name].get(step, "")
            writer.writerow(row)


def _plot_modules():
    """Import plotting dependencies lazily so --help works on login nodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as pyplot

    return pyplot, mpl_animation


def _plot_curves(axis, episode, results, through_step=None):
    targets = _evaluation_targets(episode, results)
    target_steps = [step for step in episode["steps"] if step in targets]
    if through_step is not None:
        target_steps = [step for step in target_steps if step <= through_step]
    axis.plot(
        target_steps,
        [targets[step] for step in target_steps],
        color="#1f77b4",
        linewidth=2.5,
        label="Ground-truth progress",
    )
    for name in MODEL_ORDER:
        if name not in results:
            continue
        rows = results[name]
        if through_step is not None:
            rows = [row for row in rows if row["step"] <= through_step]
        axis.plot(
            [row["step"] for row in rows],
            [row["prediction"] for row in rows],
            color=MODEL_COLORS[name],
            linewidth=2.0,
            label=MODEL_LABELS[name],
        )
    axis.set_xlim(episode["steps"][0], max(episode["steps"][-1], 1))
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Episode step")
    axis.set_ylabel("Normalized progress")
    axis.grid(True, alpha=0.3)


def save_static_plot(path, episode, results):
    plt, _ = _plot_modules()
    fig, axis = plt.subplots(figsize=(12, 6))
    _plot_curves(axis, episode, results)
    axis.legend(loc="best")
    axis.set_title(f"Progress comparison: {episode['episode_id']}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_episode_video(path, episode, results, fps):
    loaded = inspect_episode_shard(episode["path"], load_frames=True)
    plt, animation = _plot_modules()
    frames = loaded["frames"]
    video_steps = [step for step in episode["steps"] if step in frames]
    if not video_steps:
        print(f"  WARNING: no overhead frames for {episode['path']}; skipping video")
        return None

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=(2.3, 1.2), hspace=0.25)
    image_axis = fig.add_subplot(grid[0])
    curve_axis = fig.add_subplot(grid[1])
    image_axis.axis("off")
    image_artist = image_axis.imshow(np.asarray(frames[video_steps[0]]))

    def update(frame_index):
        step = video_steps[frame_index]
        image_artist.set_data(np.asarray(frames[step]))
        image_axis.set_title(
            f"{episode['episode_id']} — step {step}/{video_steps[-1]}", fontsize=14
        )
        curve_axis.clear()
        _plot_curves(curve_axis, episode, results, through_step=step)
        curve_axis.axvline(step, color="black", linestyle="--", alpha=0.5)
        curve_axis.legend(loc="upper left", ncols=2, fontsize=9)
        return [image_artist]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(video_steps),
        interval=max(1, 1000 // max(1, fps)),
        blit=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        movie.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2500))
        saved = path
    except Exception as error:
        saved = path.with_suffix(".gif")
        print(f"  FFmpeg failed ({error}); saving GIF instead: {saved}")
        movie.save(saved, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return saved


def summarize_episode(episode, results):
    summary = {}
    for name, rows in results.items():
        errors = []
        for row in rows:
            errors.append(row["prediction"] - row["target"])
        array = np.asarray(errors, dtype=np.float64)
        summary[name] = {
            "samples": len(rows),
            "mae": float(np.mean(np.abs(array))) if len(array) else None,
            "mse": float(np.mean(array**2)) if len(array) else None,
        }
    return summary


def main():
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.clip_stride <= 0:
        raise ValueError("--clip-stride must be positive")

    explicit = {
        "llava": args.llava_checkpoint or os.environ.get("LLAVA_CHECKPOINT"),
        "qwen3_vl": args.qwen3_vl_checkpoint or os.environ.get("QWEN3_VL_CHECKPOINT"),
        "vjepa2": args.vjepa2_checkpoint or os.environ.get("VJEPA2_CHECKPOINT"),
    }
    checkpoints = {
        name: find_latest_checkpoint(args.checkpoint_root, name, explicit[name])
        for name in args.models
    }
    print("Selected checkpoints:")
    for name, path in checkpoints.items():
        print(f"  {MODEL_LABELS[name]}: {path}")

    episodes = select_episode_shards(args)
    run_name = args.run_name or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "checkpoint_root": str(Path(args.checkpoint_root).expanduser()),
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "episodes": episodes,
        "num_failed_episodes": args.num_failed_episodes,
        "clip_stride": args.clip_stride,
        "fps": args.fps,
    }
    # Frames are empty during selection, but keep the manifest explicitly lean.
    for episode in manifest["episodes"]:
        episode.pop("frames", None)
    with open(run_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    all_results = {episode["source_id"]: {} for episode in episodes}
    model_metadata = {}

    for name in args.models:
        model, model_args, model_dtype = load_model_from_checkpoint(
            checkpoints[name], device
        )
        if model_args.vl_backend != EXPECTED_BACKENDS[name]:
            raise RuntimeError(
                f"Checkpoint selected as {MODEL_LABELS[name]} uses backend "
                f"{model_args.vl_backend!r}, expected {EXPECTED_BACKENDS[name]!r}: "
                f"{checkpoints[name]}"
            )
        model_metadata[name] = {
            "checkpoint": str(checkpoints[name]),
            "backend": model_args.vl_backend,
            "model_name": model_args.vl_model_name,
            "clip_len": int(model_args.clip_len),
            "dtype": str(model_dtype),
        }
        for episode in episodes:
            all_results[episode["source_id"]][name] = infer_episode(
                model,
                model_args,
                model_dtype,
                episode,
                device,
                args.clip_stride,
            )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_summary = {}
    for index, episode in enumerate(episodes, start=1):
        outcome_label = (
            "success"
            if episode["episode_success"] is True
            else "failure"
            if episode["episode_success"] is False
            else "unknown"
        )
        episode_dir = (
            run_dir
            / f"episode_{index:02d}_{outcome_label}_{episode['source_id']}"
        )
        episode_dir.mkdir(parents=True, exist_ok=False)
        results = all_results[episode["source_id"]]
        write_episode_csv(episode_dir / "progress.csv", episode, results)
        save_static_plot(episode_dir / "progress.png", episode, results)
        if not args.no_video:
            save_episode_video(
                episode_dir / "episode_progress.mp4", episode, results, args.fps
            )
        summary = summarize_episode(episode, results)
        aggregate_summary[episode["source_id"]] = summary
        with open(episode_dir / "summary.json", "w") as handle:
            json.dump(
                {
                    "episode": {key: value for key, value in episode.items() if key != "frames"},
                    "models": summary,
                },
                handle,
                indent=2,
            )

    with open(run_dir / "model_metadata.json", "w") as handle:
        json.dump(model_metadata, handle, indent=2)
    with open(run_dir / "summary.json", "w") as handle:
        json.dump(aggregate_summary, handle, indent=2)
    print(f"Done. Episode inference outputs: {run_dir}")


if __name__ == "__main__":
    main()
