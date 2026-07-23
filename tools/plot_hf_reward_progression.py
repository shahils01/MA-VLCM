#!/usr/bin/env python3
"""Plot average reward progression across WebDataset episodes."""

import argparse
import contextlib
import io
import csv
import glob
import json
import os
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_HF_REPO = "adi2440/tb3-lab-vlcm-progress-v1"
DEFAULT_SOURCE = f"hf://datasets/{os.environ.get('HF_DATASET_REPO', DEFAULT_HF_REPO)}/*.tar"
DEFAULT_PLOT = "outputs/plots/reward_analysis/hf_tb3_lab_average_reward_progression.png"
DEFAULT_CSV = "outputs/results/hf_tb3_lab_reward_progression.csv"

STEP_RE = re.compile(r"_step(\d+)$")
JSON_SUFFIXES = {
    ".episode_reward.json": "episode_reward",
    ".reward.json": "reward",
    ".progress.json": "progress",
    ".state.json": "state",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help="Local .tar, shard directory/glob, or hf://datasets/<owner>/<repo>/<glob>.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for snapshot_download.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Read only the first N shards after sorting.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Stop after collecting this many unique episodes.",
    )
    parser.add_argument("--output", default=DEFAULT_PLOT, help="Output PNG path.")
    parser.add_argument("--csv-output", default=DEFAULT_CSV, help="Output CSV path.")
    parser.add_argument(
        "--include-raw-rewards",
        action="store_true",
        help="Also plot unbounded dense step reward and cumulative reward diagnostics.",
    )
    return parser.parse_args()


def expand_source(source, revision=None, cache_dir=None):
    prefix = "hf://datasets/"
    if source.startswith(prefix):
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError("huggingface_hub is required for hf:// sources.") from exc

        rest = source[len(prefix) :]
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"Invalid Hugging Face dataset source: {source}")
        repo_id = f"{parts[0]}/{parts[1]}"
        allow_pattern = parts[2] if len(parts) > 2 else "*.tar"
        print(f"Resolving Hugging Face dataset: repo={repo_id}, pattern={allow_pattern}")
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_pattern,
            revision=revision,
            cache_dir=cache_dir,
        )
        shards = sorted(glob.glob(os.path.join(local_root, allow_pattern), recursive=True))
        if not shards:
            shards = sorted(glob.glob(os.path.join(local_root, "**", "*.tar"), recursive=True))
        return shards

    path = Path(source).expanduser()
    if path.is_dir():
        return sorted(str(p) for p in path.rglob("*.tar"))
    if path.is_file():
        return [str(path)]
    return sorted(glob.glob(str(path), recursive=True))


def split_member_name(name):
    for suffix, kind in JSON_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], kind
    return None, None


def load_json(payload):
    return json.loads(payload.decode("utf-8"))


def numeric_value(data, keys):
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if value is not None:
                return float(value)
    return None


def step_from_prefix(prefix, state=None):
    match = STEP_RE.search(Path(prefix).name)
    if match:
        return int(match.group(1))
    if isinstance(state, dict):
        meta = state.get("episode_meta", {})
        for key in ("step", "timestep", "frame", "sample_idx"):
            if key in meta:
                return int(meta[key])
            if key in state:
                return int(state[key])
    return 0


def episode_from_prefix(prefix, state=None):
    if isinstance(state, dict):
        meta = state.get("episode_meta", {})
        for key in ("episode_id", "episode", "trajectory_id", "traj_id"):
            value = meta.get(key, state.get(key))
            if value is not None:
                return str(value)

    stem = Path(prefix).name
    if "_step" in stem:
        return stem.rsplit("_step", 1)[0]
    return stem


def read_shard(path):
    records = defaultdict(dict)
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            prefix, kind = split_member_name(member.name)
            if kind is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                records[prefix][kind] = load_json(handle.read())
            except Exception as exc:
                print(f"Warning: skipped malformed JSON member {member.name}: {exc}")
    return records


def collect_episode_series(shards, max_episodes=None):
    rewards = defaultdict(dict)
    cumulative = defaultdict(dict)
    progress = defaultdict(dict)
    steps_seen = 0

    for shard_idx, shard in enumerate(shards, start=1):
        print(f"[{shard_idx}/{len(shards)}] Reading {shard}")
        for prefix, record in read_shard(shard).items():
            state = record.get("state")
            episode_id = episode_from_prefix(prefix, state)
            step = step_from_prefix(prefix, state)

            reward = numeric_value(
                record.get("reward"),
                ("reward", "value", "target", "step_reward"),
            )
            if reward is None and isinstance(state, dict):
                reward = numeric_value(state, ("reward",))
            if reward is not None:
                rewards[episode_id][step] = reward

            episode_reward = numeric_value(
                record.get("episode_reward"),
                ("episode_reward", "cumulative_reward", "reward", "value"),
            )
            if episode_reward is None and isinstance(state, dict):
                episode_reward = numeric_value(state, ("cumulative_reward", "episode_reward"))
            if episode_reward is not None:
                cumulative[episode_id][step] = episode_reward

            progress_value = numeric_value(
                record.get("progress"),
                ("target", "team_progress", "progress", "value"),
            )
            if progress_value is not None:
                progress[episode_id][step] = progress_value

            steps_seen += 1

        if max_episodes is not None and len(rewards) >= max_episodes:
            keep = set(sorted(rewards)[:max_episodes])
            rewards = defaultdict(dict, {k: rewards[k] for k in keep})
            cumulative = defaultdict(dict, {k: cumulative.get(k, {}) for k in keep})
            progress = defaultdict(dict, {k: progress.get(k, {}) for k in keep})
            break

    if not rewards:
        raise RuntimeError("No reward values were found in the selected shards.")

    fill_cumulative_from_rewards(rewards, cumulative)
    print(f"Collected {steps_seen} step records from {len(rewards)} episodes.")
    return rewards, cumulative, progress


def fill_cumulative_from_rewards(rewards, cumulative):
    for episode_id, series in rewards.items():
        if cumulative.get(episode_id):
            continue
        running = 0.0
        for step in sorted(series):
            running += series[step]
            cumulative[episode_id][step] = running


def curve_from_series(series_by_episode):
    all_steps = sorted({step for series in series_by_episode.values() for step in series})
    rows = []
    for step in all_steps:
        values = np.array(
            [series[step] for series in series_by_episode.values() if step in series],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(
            {
                "step": int(step),
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": std,
                "sem": float(std / np.sqrt(values.size)) if values.size > 1 else 0.0,
            }
        )
    return rows


def row_by_step(rows):
    return {row["step"]: row for row in rows}


def write_csv(path, reward_rows, cumulative_rows, progress_rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    reward_by_step = row_by_step(reward_rows)
    cumulative_by_step = row_by_step(cumulative_rows)
    progress_by_step = row_by_step(progress_rows)
    steps = sorted(set(reward_by_step) | set(cumulative_by_step) | set(progress_by_step))
    fieldnames = [
        "step",
        "reward_count",
        "reward_mean",
        "reward_std",
        "reward_sem",
        "cumulative_count",
        "cumulative_mean",
        "cumulative_std",
        "cumulative_sem",
        "progress_count",
        "progress_mean",
        "progress_std",
        "progress_sem",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            reward = reward_by_step.get(step, {})
            cumulative = cumulative_by_step.get(step, {})
            progress = progress_by_step.get(step, {})
            writer.writerow(
                {
                    "step": step,
                    "reward_count": reward.get("count", ""),
                    "reward_mean": reward.get("mean", ""),
                    "reward_std": reward.get("std", ""),
                    "reward_sem": reward.get("sem", ""),
                    "cumulative_count": cumulative.get("count", ""),
                    "cumulative_mean": cumulative.get("mean", ""),
                    "cumulative_std": cumulative.get("std", ""),
                    "cumulative_sem": cumulative.get("sem", ""),
                    "progress_count": progress.get("count", ""),
                    "progress_mean": progress.get("mean", ""),
                    "progress_std": progress.get("std", ""),
                    "progress_sem": progress.get("sem", ""),
                }
            )


def plot_curve_matplotlib(ax, rows, label, color, ylabel):
    steps = np.array([row["step"] for row in rows], dtype=np.int64)
    means = np.array([row["mean"] for row in rows], dtype=np.float64)
    sems = np.array([row["sem"] for row in rows], dtype=np.float64)
    ax.plot(steps, means, color=color, linewidth=2.0, label=label)
    if np.any(sems > 0):
        ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.18, linewidth=0)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def panel_specs(reward_rows, cumulative_rows, progress_rows, include_raw_rewards=False):
    panels = []
    if progress_rows:
        panels.append(
            (progress_rows, "Mean normalized reward", "Normalized reward", "#2ca02c")
        )
    else:
        panels.append((reward_rows, "Mean step reward", "Step reward", "#d62728"))

    if include_raw_rewards:
        panels.extend(
            [
                (reward_rows, "Mean raw step reward", "Raw step reward", "#d62728"),
                (
                    cumulative_rows,
                    "Mean raw cumulative reward",
                    "Raw cumulative reward",
                    "#1f77b4",
                ),
            ]
        )
    return panels


def plot_title(progress_rows, include_raw_rewards, episode_count, shard_count):
    if progress_rows and not include_raw_rewards:
        prefix = "Average normalized reward progression"
    elif progress_rows:
        prefix = "Average reward progression with raw diagnostics"
    else:
        prefix = "Average raw reward progression"
    return f"{prefix} across {episode_count} episodes ({shard_count} shards)"


def plot_progression_matplotlib(
    path,
    reward_rows,
    cumulative_rows,
    progress_rows,
    episode_count,
    shard_count,
    include_raw_rewards=False,
):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panels = panel_specs(
        reward_rows,
        cumulative_rows,
        progress_rows,
        include_raw_rewards=include_raw_rewards,
    )
    nrows = len(panels)
    height = max(4, 3 * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, height), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (rows, label, ylabel, color) in zip(axes, panels):
        plot_curve_matplotlib(
            ax,
            rows,
            label,
            color,
            ylabel,
        )

    axes[-1].set_xlabel("Episode step")
    fig.suptitle(
        plot_title(progress_rows, include_raw_rewards, episode_count, shard_count),
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def pillow_text_size(draw, text):
    box = draw.textbbox((0, 0), str(text))
    return box[2] - box[0], box[3] - box[1]


def pillow_points(rows, x_to_px, y_to_px, upper=False, lower=False):
    points = []
    for row in rows:
        y = row["mean"]
        if upper:
            y += row["sem"]
        if lower:
            y -= row["sem"]
        points.append((int(round(x_to_px(row["step"]))), int(round(y_to_px(y)))))
    return points


def pillow_panel(draw, rows, x_min, x_max, bounds, title, ylabel, color):
    left, top, width, height = bounds
    y_values = []
    for row in rows:
        y_values.extend([row["mean"] - row["sem"], row["mean"] + row["sem"]])
    y_min, y_max = padded_range(y_values)
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-9)
    rgb = hex_rgb(color)

    def x_to_px(x):
        return left + ((x - x_min) / x_span) * width

    def y_to_px(y):
        return top + height - ((y - y_min) / y_span) * height

    draw.rectangle((left, top, left + width, top + height), fill="white")
    draw.text((left, top - 22), title, fill=(17, 17, 17))
    draw.text((left + 220, top - 22), ylabel, fill=(68, 68, 68))

    for i in range(6):
        frac = i / 5
        y = top + height - frac * height
        value = y_min + frac * y_span
        draw.line((left, y, left + width, y), fill=(229, 231, 235), width=1)
        label = format_tick(value)
        tw, th = pillow_text_size(draw, label)
        draw.text((left - tw - 10, y - th / 2), label, fill=(68, 68, 68))

    for i in range(7):
        frac = i / 6
        x = left + frac * width
        value = x_min + frac * x_span
        draw.line((x, top, x, top + height), fill=(243, 244, 246), width=1)
        label = f"{value:.0f}"
        tw, _ = pillow_text_size(draw, label)
        draw.text((x - tw / 2, top + height + 10), label, fill=(68, 68, 68))

    draw.rectangle((left, top, left + width, top + height), outline=(55, 65, 81), width=1)

    if any(row["sem"] > 0 for row in rows):
        upper = pillow_points(rows, x_to_px, y_to_px, upper=True)
        lower = pillow_points(list(reversed(rows)), x_to_px, y_to_px, lower=True)
        band_rgb = tuple(int(channel * 0.18 + 255 * 0.82) for channel in rgb)
        draw.polygon(upper + lower, fill=band_rgb)

    points = pillow_points(rows, x_to_px, y_to_px)
    if len(points) > 1:
        draw.line(points, fill=rgb, width=3, joint="curve")
    elif points:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=rgb)


def plot_progression_pillow(
    path,
    reward_rows,
    cumulative_rows,
    progress_rows,
    episode_count,
    shard_count,
    include_raw_rewards=False,
):
    from PIL import Image, ImageDraw

    output_path = Path(path)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = panel_specs(
        reward_rows,
        cumulative_rows,
        progress_rows,
        include_raw_rewards=include_raw_rewards,
    )

    all_rows = reward_rows + cumulative_rows + progress_rows
    x_values = [row["step"] for row in all_rows]
    x_min, x_max = min(x_values), max(x_values)
    width = 1200
    panel_height = 220
    top_margin = 80
    panel_gap = 74
    bottom_margin = 70
    left = 120
    plot_width = width - left - 50
    height = top_margin + len(panels) * panel_height + (len(panels) - 1) * panel_gap + bottom_margin

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = plot_title(progress_rows, include_raw_rewards, episode_count, shard_count)
    tw, _ = pillow_text_size(draw, title)
    draw.text(((width - tw) / 2, 28), title, fill=(17, 17, 17))

    for idx, (rows, panel_title, ylabel, color) in enumerate(panels):
        top = top_margin + idx * (panel_height + panel_gap)
        pillow_panel(draw, rows, x_min, x_max, (left, top, plot_width, panel_height), panel_title, ylabel, color)

    xlabel = "Episode step"
    tw, _ = pillow_text_size(draw, xlabel)
    draw.text(((width - tw) / 2, height - 34), xlabel, fill=(34, 34, 34))
    image.save(output_path)
    return str(output_path)


def svg_text(x, y, text, size=16, anchor="start", color="#222222", extra=""):
    text = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{color}" {extra}>{text}</text>'
    )


def format_tick(value):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def padded_range(values):
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        pad = max(abs(vmin) * 0.1, 1.0)
    else:
        pad = (vmax - vmin) * 0.08
    return vmin - pad, vmax + pad


def svg_points(rows, x_to_px, y_to_px, upper=False, lower=False):
    points = []
    for row in rows:
        y = row["mean"]
        if upper:
            y += row["sem"]
        if lower:
            y -= row["sem"]
        points.append(f"{x_to_px(row['step']):.1f},{y_to_px(y):.1f}")
    return " ".join(points)


def svg_panel(rows, x_min, x_max, bounds, title, ylabel, color):
    left, top, width, height = bounds
    y_values = []
    for row in rows:
        y_values.extend([row["mean"] - row["sem"], row["mean"] + row["sem"]])
    y_min, y_max = padded_range(y_values)
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-9)

    def x_to_px(x):
        return left + ((x - x_min) / x_span) * width

    def y_to_px(y):
        return top + height - ((y - y_min) / y_span) * height

    items = [
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#ffffff" />',
        svg_text(left, top - 14, title, size=18, color="#111111"),
        svg_text(left - 64, top + height / 2, ylabel, size=14, anchor="middle", extra=f'transform="rotate(-90 {left - 64:.1f} {top + height / 2:.1f})"'),
    ]

    for i in range(6):
        frac = i / 5
        y = top + height - frac * height
        value = y_min + frac * y_span
        items.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        items.append(svg_text(left - 10, y + 5, format_tick(value), size=12, anchor="end", color="#444444"))

    for i in range(7):
        frac = i / 6
        x = left + frac * width
        value = x_min + frac * x_span
        items.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + height:.1f}" stroke="#f3f4f6" stroke-width="1" />'
        )
        items.append(svg_text(x, top + height + 22, f"{value:.0f}", size=12, anchor="middle", color="#444444"))

    items.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" fill="none" stroke="#374151" stroke-width="1.2" />'
    )

    if any(row["sem"] > 0 for row in rows):
        upper = svg_points(rows, x_to_px, y_to_px, upper=True)
        lower = svg_points(list(reversed(rows)), x_to_px, y_to_px, lower=True)
        items.append(
            f'<polygon points="{upper} {lower}" fill="{color}" opacity="0.16" stroke="none" />'
        )

    items.append(
        f'<polyline points="{svg_points(rows, x_to_px, y_to_px)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />'
    )
    return "\n".join(items)


def plot_progression_svg(
    path,
    reward_rows,
    cumulative_rows,
    progress_rows,
    episode_count,
    shard_count,
    include_raw_rewards=False,
):
    output_path = Path(path)
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = panel_specs(
        reward_rows,
        cumulative_rows,
        progress_rows,
        include_raw_rewards=include_raw_rewards,
    )

    all_rows = reward_rows + cumulative_rows + progress_rows
    x_values = [row["step"] for row in all_rows]
    x_min, x_max = min(x_values), max(x_values)
    width = 1200
    panel_height = 220
    top_margin = 80
    panel_gap = 74
    bottom_margin = 70
    left = 120
    plot_width = width - left - 50
    height = top_margin + len(panels) * panel_height + (len(panels) - 1) * panel_gap + bottom_margin

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        svg_text(
            width / 2,
            38,
            plot_title(progress_rows, include_raw_rewards, episode_count, shard_count),
            size=24,
            anchor="middle",
            color="#111111",
        ),
    ]

    for idx, (rows, title, ylabel, color) in enumerate(panels):
        top = top_margin + idx * (panel_height + panel_gap)
        items.append(
            svg_panel(rows, x_min, x_max, (left, top, plot_width, panel_height), title, ylabel, color)
        )

    items.append(svg_text(width / 2, height - 24, "Episode step", size=16, anchor="middle"))
    items.append("</svg>")
    output_path.write_text("\n".join(items), encoding="utf-8")
    return str(output_path)


def plot_progression(
    path,
    reward_rows,
    cumulative_rows,
    progress_rows,
    episode_count,
    shard_count,
    include_raw_rewards=False,
):
    try:
        return plot_progression_matplotlib(
            path,
            reward_rows,
            cumulative_rows,
            progress_rows,
            episode_count,
            shard_count,
            include_raw_rewards=include_raw_rewards,
        )
    except Exception as exc:
        try:
            fallback_path = plot_progression_pillow(
                path,
                reward_rows,
                cumulative_rows,
                progress_rows,
                episode_count,
                shard_count,
                include_raw_rewards=include_raw_rewards,
            )
            print(f"Matplotlib plot failed ({exc}); saved Pillow PNG fallback to {fallback_path}")
            return fallback_path
        except Exception as pillow_exc:
            fallback_path = plot_progression_svg(
                path,
                reward_rows,
                cumulative_rows,
                progress_rows,
                episode_count,
                shard_count,
                include_raw_rewards=include_raw_rewards,
            )
            print(
                f"Matplotlib plot failed ({exc}); Pillow fallback failed ({pillow_exc}); "
                f"saved SVG fallback to {fallback_path}"
            )
            return fallback_path


def main():
    args = parse_args()
    shards = expand_source(args.source, revision=args.revision, cache_dir=args.cache_dir)
    if args.limit_shards is not None:
        shards = shards[: args.limit_shards]
    if not shards:
        raise RuntimeError(f"No .tar shards matched source: {args.source}")

    rewards, cumulative, progress = collect_episode_series(
        shards,
        max_episodes=args.max_episodes,
    )
    reward_rows = curve_from_series(rewards)
    cumulative_rows = curve_from_series(cumulative)
    progress_rows = curve_from_series(progress)

    write_csv(args.csv_output, reward_rows, cumulative_rows, progress_rows)
    plot_path = plot_progression(
        args.output,
        reward_rows,
        cumulative_rows,
        progress_rows,
        episode_count=len(rewards),
        shard_count=len(shards),
        include_raw_rewards=args.include_raw_rewards,
    )
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
