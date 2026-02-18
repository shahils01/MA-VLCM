import os
import json
import io
import numpy as np
import tarfile
import argparse
import glob
from pathlib import Path
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# From MA-VLCM/train.py
SHELF_MAP = {
    1: (1, 1),
    2: (2, 1),
    3: (7, 1),
    4: (8, 1),
    5: (1, 2),
    6: (2, 2),
    7: (7, 2),
    8: (8, 2),
    9: (1, 3),
    10: (2, 3),
    11: (7, 3),
    12: (8, 3),
    13: (1, 4),
    14: (2, 4),
    15: (7, 4),
    16: (8, 4),
    17: (1, 5),
    18: (2, 5),
    19: (7, 5),
    20: (8, 5),
    21: (1, 6),
    22: (2, 6),
    23: (7, 6),
    24: (8, 6),
    25: (1, 7),
    26: (2, 7),
    27: (7, 7),
    28: (8, 7),
    29: (1, 8),
    30: (2, 8),
    31: (7, 8),
    32: (8, 8),
}


def calculate_step_reward(state_data, dist_matrix, times_col):
    """
    Replicates the reward logic from MA-VLCM/train.py
    """
    # 1. Base Reward
    base_reward = 0.0
    if times_col is not None:
        # >0 -> +1, <0 -> -5, else 0
        rs = [1.0 if t > 0 else -5.0 if t < 0 else 0.0 for t in times_col]
        if len(rs) > 0:
            base_reward = sum(rs) / max(len(rs), 1)
        else:
            base_reward = float(state_data.get("reward", 0.0))
    else:
        base_reward = float(state_data.get("reward", 0.0))

    # 2. Distance Penalty
    dist_penalty = 0.0
    if dist_matrix is not None:
        d = dist_matrix
        if d.ndim == 2 and d.shape[0] > 1:
            eye = np.eye(d.shape[0], dtype=bool)
            # Collision check: distance < 3.0 between different agents
            if ((d < 3.0) & (~eye)).any():
                dist_penalty = -1.0

    # 3. r_dist (Distance to requests)
    r_dist = 0.0
    try:
        requests = state_data.get("requests", [])
        # Filter valid requests present in our map
        valid_request_positions = [SHELF_MAP[r] for r in requests if r in SHELF_MAP]

        if not valid_request_positions:
            r_dist = 0.0
        else:
            agents = state_data.get("agents", [])
            num_agents = len(agents)
            dist_sum = 0.0

            for ag in agents:
                # Agent Pos
                pos = ag.get("pos", [0, 0])
                carrying_id = ag.get("carrying")

                # Check if carrying a requested box
                is_carrying_request = (carrying_id is not None) and (
                    carrying_id in requests
                )

                if is_carrying_request:
                    # Distance cost is 0
                    dist = 0.0
                else:
                    # Calculate min distance to any valid request
                    min_d = 1000.0
                    ax, ay = pos[0], pos[1]
                    for sx, sy in valid_request_positions:
                        d_val = np.sqrt((ax - sx) ** 2 + (ay - sy) ** 2)
                        if d_val < min_d:
                            min_d = d_val
                    dist = min_d

                dist_sum += dist

            avg_min_dist = dist_sum / max(num_agents, 1)
            r_dist = -avg_min_dist
    except Exception:
        pass

    return {
        "total": base_reward + dist_penalty + r_dist,
        "base": base_reward,
        "penalty": dist_penalty,
        "r_dist": r_dist,
    }


def load_trajectory_from_folder(folder_path):
    """
    Loads step data from a folder of raw files.
    """
    rewards = []
    files = sorted(os.listdir(folder_path))

    # Group by step
    steps = {}
    for f in files:
        if "step" not in f:
            continue
        parts = f.split("_")
        step_part = [p for p in parts if p.startswith("step")]
        if not step_part:
            continue

        # Extract "stepXXXX" from "stepXXXX.state.json" or "stepXXXX.dist.npy"
        step_id = step_part[0].split(".")[0]

        if step_id not in steps:
            steps[step_id] = {}

        full_path = os.path.join(folder_path, f)
        if f.endswith("state.json"):
            with open(full_path, "r") as fp:
                steps[step_id]["state"] = json.load(fp)
        elif f.endswith("dist.npy"):
            steps[step_id]["dist"] = np.load(full_path)

    # Process steps in order
    sorted_step_ids = sorted(steps.keys())
    for sid in sorted_step_ids:
        data = steps[sid]
        if "state" not in data:
            continue

        state = data["state"]
        dist = data.get("dist", None)

        # Check for times.npy
        times_col = None

        r = calculate_step_reward(state, dist, times_col)
        rewards.append(r)

    return rewards


def load_trajectory_from_tar(tar_path):
    """
    Loads step data from a .tar shard.
    """
    rewards = []
    step_data = {}

    # Check for times.npy sibling
    times_path = str(tar_path).replace(".tar", "_times.npy")
    times_matrix = None
    if os.path.exists(times_path):
        try:
            times_matrix = np.load(times_path)
        except Exception as e:
            print(f"Error loading {times_path}: {e}")

    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name

                # Assume format like trajectory_ID_stepXXXX.ext
                match = re.search(r"step(\d+)", name)
                if not match:
                    continue
                step_idx = int(match.group(1))  # 0-indexed integer

                if step_idx not in step_data:
                    step_data[step_idx] = {}

                if name.endswith("state.json"):
                    f = tar.extractfile(member)
                    step_data[step_idx]["state"] = json.load(f)
                elif name.endswith("dist.npy"):
                    f = tar.extractfile(member)
                    bytes_io = io.BytesIO(f.read())
                    step_data[step_idx]["dist"] = np.load(bytes_io)

    except Exception:
        # print(f"Error reading tar {tar_path}: {e}")
        return []

    # Process sorted steps
    sorted_indices = sorted(step_data.keys())
    for idx in sorted_indices:
        data = step_data[idx]
        if "state" not in data:
            continue

        state = data["state"]
        dist = data.get("dist", None)

        # Get times column if valid
        curr_step_field = int(state.get("step", 0))
        if curr_step_field == 0:
            eff_idx = idx
        else:
            eff_idx = curr_step_field - 1

        times_col = None
        if times_matrix is not None:
            if 0 <= eff_idx < times_matrix.shape[1]:
                times_col = times_matrix[:, eff_idx]

        r = calculate_step_reward(state, dist, times_col)
        rewards.append(r)

    return rewards


def calculate_returns(rewards, clip_lens):
    results = {}
    rewards_np = np.array(rewards)
    T = len(rewards_np)

    for L in clip_lens:
        if T < L:
            results[L] = np.nan
            continue

        # Sliding window sum using convolution
        window = np.ones(L)
        sums = np.convolve(rewards_np, window, mode="valid")
        avg_return = np.mean(sums)
        results[L] = avg_return

    return results


def process_config(config_name, shards, args):
    print(f"\nProcessing configuration: {config_name}")
    print(f"Found {len(shards)} shards.")

    if args.limit:
        shards = shards[: args.limit]
        print(f"Limiting to {len(shards)} shards.")

    all_rewards_seq = []

    for t in tqdm(shards, desc=f"Analyzing {config_name}"):
        r = load_trajectory_from_tar(t)
        if r:
            all_rewards_seq.append(r)

    if not all_rewards_seq:
        print("No rewards found.")
        return

    # Flatten for step reward distribution
    flat_rewards = [r for sublist in all_rewards_seq for r in sublist]

    # Calculate returns stats
    agg_stats = {L: [] for L in args.clip_lens}
    all_returns_for_plot = {
        L: [] for L in args.clip_lens
    }  # For plotting distribution of returns

    for r_seq_dicts in all_rewards_seq:
        r_seq = [d["total"] for d in r_seq_dicts]
        # We want the distribution of returns for *each window*.
        # But `calculate_returns` above returned the MEAN return of windows in a trajectory.
        # The user likely wants to see the distribution of "returns" (i.e., sum of rewards over clip len).
        # Let's collect ALL window sums to plot their distribution.

        rewards_np = np.array(r_seq)
        T = len(rewards_np)
        for L in args.clip_lens:
            if T >= L:
                window = np.ones(L)
                # sums of each window in this trajectory
                sums = np.convolve(rewards_np, window, mode="valid")
                # Add all these sums to the plot data
                all_returns_for_plot[L].extend(sums)
                # Also track the mean for the textual report
                agg_stats[L].append(np.mean(sums))

    # Text Report
    print(f"--- Results for {config_name} ---")
    for L in args.clip_lens:
        vals = agg_stats[L]
        if vals:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(
                f"Clip Length {L}: Mean Return = {mean_val:.4f} \u00b1 {std_val:.4f} (over {len(vals)} trajectories)"
            )
        else:
            print(f"Clip Length {L}: No valid data")

    # Plotting
    # 2 Plots:
    # 1. Step Reward Distribution
    # 2. Returns Distribution (for each Clip Length)

    num_clip_lens = len(args.clip_lens)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Step Rewards
    axes[0].hist(flat_rewards, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Step Reward Distribution\n{config_name}")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Returns Distribution (overlapping histograms or separate?)
    # Overlapping might be messy if scales differ widely. Let's do overlapping step/density or independent?
    # Let's do overlapping with alpha.
    colors = ["r", "g", "b", "orange", "purple"]
    for i, L in enumerate(args.clip_lens):
        data = all_returns_for_plot[L]
        if data:
            c = colors[i % len(colors)]
            axes[1].hist(
                data, bins=50, alpha=0.5, label=f"Clip-{L}", density=True, color=c
            )

    axes[1].set_title(f"Return Distribution (Density)\n{config_name}")
    axes[1].set_xlabel("Return (Sum of Rewards)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Sanitize filename
    safe_name = config_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    out_file = f"analysis_{safe_name}.png"
    plt.savefig(out_file)
    print(f"Saved plots to {out_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze rewards in VLCM data")
    parser.add_argument(
        "input_path",
        help="Path to a folder (raw trajectory) or directory containing tar shards",
    )
    parser.add_argument(
        "--clip_lens",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="List of clip lengths to calculate returns for",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of shards per config"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Path not found: {input_path}")
        return

    # Determine if input_path is a config itself or a container of configs
    # Heuristic: Check for .tar files directly.
    tars_direct = sorted(list(input_path.glob("*.tar")))

    if tars_direct:
        # It's a single configuration
        config_name = input_path.name
        process_config(config_name, tars_direct, args)
    else:
        # Check subdirectories
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        configs_found = []
        for d in subdirs:
            tars = sorted(list(d.rglob("*.tar")))
            if tars:
                configs_found.append((d.name, tars))

        if configs_found:
            print(f"Found {len(configs_found)} configurations in {input_path}")
            for name, shards in configs_found:
                process_config(name, shards, args)
        else:
            # Fallback: maybe raw trajectory folder?
            # If it has stepXXXX files
            if any("step" in f.name for f in input_path.iterdir()):
                print("Detected raw trajectory folder.")
                # Treat as single config, load using folder loader (special case, or just adapt)
                # The loop above uses load_trajectory_from_tar.
                # Let's just handle raw folder analysis:
                rewards = load_trajectory_from_folder(str(input_path))
                # For raw folder, we only have 1 trajectory, so histograms might be sparse but still valid.
                # Re-wrap to reuse logic?
                # It's different structure.

                print("Processing single raw folder.")
                # Just quick manual process
                print(f"Trajectory Length: {len(rewards)}")
                print(
                    f"First 10 Rewards: {[round(x['total'], 2) for x in rewards[:10]]}"
                )
                # Calculate returns using TOTALS
                totals = [d["total"] for d in rewards]
                returns_map = calculate_returns(totals, args.clip_lens)
                print(f"Returns: {returns_map}")

                # Plotting for single trajectory
                print("Generating plots for single trajectory...")

                # Reuse plotting logic
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Plot 1: Step Rewards
                axes[0].hist(
                    totals, bins=50, color="skyblue", edgecolor="black", alpha=0.7
                )
                axes[0].set_title(f"Step Reward Distribution\nSingle Trajectory")
                axes[0].set_xlabel("Reward")
                axes[0].set_ylabel("Frequency")
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Returns Distribution
                colors = ["r", "g", "b", "orange", "purple"]
                for i, L in enumerate(args.clip_lens):
                    rewards_np = np.array(totals)
                    T = len(rewards_np)
                    if T >= L:
                        window = np.ones(L)
                        sums = np.convolve(rewards_np, window, mode="valid")
                        c = colors[i % len(colors)]
                        axes[1].hist(
                            sums,
                            bins=50,
                            alpha=0.5,
                            label=f"Clip-{L}",
                            density=True,
                            color=c,
                        )

                axes[1].set_title(f"Return Distribution\nSingle Trajectory")
                axes[1].set_xlabel("Return")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                # Plot 3: Component Values Over Time
                bases = [d["base"] for d in rewards]
                penalties = [d["penalty"] for d in rewards]
                rdists = [d["r_dist"] for d in rewards]
                steps = range(len(rewards))

                axes[2].plot(steps, bases, label="Base", color="green", alpha=0.5)
                axes[2].plot(steps, penalties, label="Penalty", color="red", alpha=0.5)
                axes[2].plot(steps, rdists, label="R_Dist", color="purple", alpha=0.8)
                axes[2].set_title("Reward Components Over Time")
                axes[2].set_xlabel("Step")
                axes[2].set_ylabel("Value")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                out_file = f"analysis_single_trajectory.png"
                plt.savefig(out_file)
                print(f"Saved plots to {out_file}")
                plt.close()
            else:
                print("No valid data found (no .tar files or recognizable raw data).")


if __name__ == "__main__":
    main()
