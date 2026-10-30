#!/usr/bin/env python3
"""Evaluate a saved scratch MAPPO actor and its matching Python MA-VLCM critic."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch

from model import ModelConfig, SmolVLMBackbone
from python_mappo import (EpisodeHistory, PythonValueCritic, R_Actor, actor_config,
                          critic_values, make_env)
from argparse import Namespace

NAMES = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon"}


def load_pair(directory, device):
    # Snapshot the small files in memory and reject a checkpoint changing as read.
    directory = Path(directory)
    paths = [directory / name for name in ("actor.pt", "critic.pt", "training_metadata.json")]
    before = [(p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    raw = [p.read_bytes() for p in paths]
    after = [(p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    if before != after or before[2][2] < max(before[0][2], before[1][2]):
        raise RuntimeError("Checkpoint changed or is being written; retry with a stable checkpoint directory")
    actor_state = torch.load(io.BytesIO(raw[0]), map_location="cpu", weights_only=True)
    saved = torch.load(io.BytesIO(raw[1]), map_location="cpu", weights_only=True)
    metadata = json.loads(raw[2])
    if saved.get("format") != "python_mappo_vlcm_v1" or saved.get("critic_kind") != "vlcm":
        raise ValueError("Expected a Python MAPPO VLCM critic and matching actor")
    if saved["steps"] != metadata["total_environment_steps"]:
        raise RuntimeError("Checkpoint is incomplete or being written; retry with a stable checkpoint")
    args = Namespace(**saved["args"])
    cfg = ModelConfig(**saved["model_config"])
    if cfg.backbone_mode != "frozen":
        raise ValueError("This evaluator supports the frozen-backbone Python critic")
    env = make_env(args, args.agent_counts[0])
    actor = R_Actor(actor_config(args.hidden_size, args.layer_n), env.observation_spaces[0],
                    env.action_spaces[0], device)
    actor.load_state_dict(actor_state, strict=True)
    critic = PythonValueCritic(cfg, args, env, device)
    critic.load_state_dict(saved["model_state"], strict=True)
    return actor.eval(), critic.eval(), args, metadata, [hashlib.sha256(x).hexdigest() for x in raw[:2]]


def discounted_returns(rewards, gamma):
    """Return observed reward from each state through the end of this rollout."""
    result = np.zeros(len(rewards) + 1, dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        result[t] = rewards[t] + gamma * result[t + 1]
    return result


def collect_episode(actor, args, device, count, seed):
    env = make_env(args, count)
    obs, _, _ = env.reset(seed=seed)
    queue = []
    history = EpisodeHistory(env, args, queue)
    contexts, frames, steps, max_errors, mse_errors, hold, rewards = [], [], [], [], [], [], []

    def record():
        contexts.append(history.context())
        frames.append(np.asarray(env.render(size=args.image_size, view_size_m=args.view_size_m,
                                            tile_floor=args.tile_floor)).copy())
        steps.append(env.steps)
        max_errors.append(float(env._slot_errors().max()))
        mse_errors.append(float(env._formation_error()))
        hold.append(int(env.stable_team_steps))

    record()
    with torch.no_grad():
        while True:
            action, _, _ = actor(obs, np.zeros((6, 1, args.hidden_size), np.float32),
                                 np.ones((6, 1), np.float32), deterministic=True)
            old_pos = env.pos.copy()
            obs, _, reward, dones, infos, _ = env.step(action.cpu().numpy())
            history.append(queue, old_pos)
            rewards.append(float(reward[:count].mean()))
            if env.steps % args.camera_stride == 0 or dones[0]:
                record()
            if dones[0]:
                break
    return dict(env=env, contexts=contexts, queue=queue, frames=frames, steps=steps,
                max_errors=max_errors, mse_errors=mse_errors, hold=hold, rewards=rewards,
                terminated=bool(infos[0]["terminated"]), truncated=bool(infos[0]["truncated"]))


def evaluate_values(episode, critic, backbone, args, device):
    queue = episode["queue"]
    with torch.no_grad():
        for start in range(0, len(queue), backbone.cfg.vl_batch_size):
            batch = queue[start:start + backbone.cfg.vl_batch_size]
            features = backbone([f.image for f in batch], [f.prompt for f in batch]).detach().cpu()
            for frame, feature in zip(batch, features):
                frame.features, frame.image = feature, None
    predicted = critic_values(critic, episode["contexts"], args, device).astype(float)
    # At a true terminal, the continuation value is defined as zero, regardless
    # of the network's untrained prediction on that terminal observation.
    if episode["terminated"]:
        predicted[-1] = 0.0
    observed = discounted_returns(episode["rewards"], args.gamma)[episode["steps"]]
    return predicted, observed


def save_episode(episode, predicted, observed, args, root, count, seed):
    outcome = "success" if episode["terminated"] else "timeout"
    name = f"{NAMES[count]}_seed{seed}_{outcome}"
    steps = episode["steps"]
    seconds = np.asarray(steps) * args.control_dt
    intervals = [0.0] + [sum(episode["rewards"][steps[i-1]:steps[i]]) for i in range(1, len(steps))]
    label = "Realized discounted return" if episode["terminated"] else "Observed return to timeout (partial)"
    with (root / f"{name}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "seconds", "predicted_value", "observed_discounted_return",
                         "return_complete", "reward_since_previous_frame", "formation_error_m2",
                         "max_slot_error_m", "stable_hold_steps"])
        for i, step in enumerate(steps):
            writer.writerow([step, seconds[i], predicted[i], observed[i], episode["terminated"], intervals[i],
                             episode["mse_errors"][i], episode["max_errors"][i], episode["hold"][i]])
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.25])
    image_ax = fig.add_subplot(grid[:, 0])
    value_ax, error_ax, reward_ax = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    fps = 1 / (args.control_dt * args.camera_stride)
    writer = FFMpegWriter(fps=fps, metadata={"title": name})
    with writer.saving(fig, str(root / f"{name}.mp4"), dpi=110):
        for i in range(len(steps)):
            for ax in (image_ax, value_ax, error_ax, reward_ax):
                ax.clear()
            image_ax.imshow(episode["frames"][i])
            image_ax.set_title(f"{NAMES[count].title()} | t={seconds[i]:.2f}s | {outcome if i == len(steps)-1 else 'running'}")
            image_ax.axis("off")
            value_ax.plot(seconds[:i+1], predicted[:i+1], label="VLCM estimated value")
            value_ax.plot(seconds[:i+1], observed[:i+1], label=label)
            value_ax.set_ylabel("Discounted reward")
            value_ax.legend(fontsize=7, loc="best")
            error_ax.plot(seconds[:i+1], episode["max_errors"][:i+1], label="Worst slot error (m)")
            error_ax.axhline(args.formation_tolerance, color="red", linestyle="--", label="Tolerance")
            error_ax.set_ylabel("Distance (m)")
            error_ax.legend(fontsize=7, loc="best")
            reward_ax.plot(seconds[:i+1], intervals[:i+1], color="tab:green")
            reward_ax.set_ylabel("Reward / interval")
            reward_ax.set_xlabel("Time (s)")
            for ax in (value_ax, error_ax, reward_ax):
                ax.grid(alpha=.25)
                ax.set_xlim(0, max(seconds[-1], args.control_dt))
            fig.suptitle(f"Actor evaluation: {count} robots; seed {seed}; critic observes history")
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    errors = predicted[:-1] - observed[:-1]
    return dict(formation=NAMES[count], agents=count, seed=seed, outcome=outcome,
                steps=int(steps[-1]), total_reward=float(sum(episode["rewards"])),
                initial_max_slot_error_m=episode["max_errors"][0],
                final_max_slot_error_m=episode["max_errors"][-1],
                final_formation_error_m2=episode["mse_errors"][-1],
                return_complete=episode["terminated"],
                realized_return_mse=float(np.mean(errors ** 2)) if episode["terminated"] else None,
                video=f"{name}.mp4", measurements=f"{name}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory with matching actor.pt and critic.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formations", nargs="+", choices=list(NAMES.values()),
                        default=["triangle", "square", "pentagon"])
    parser.add_argument("--episodes-per-formation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9700001)
    parser.add_argument("--max-steps", type=int, help="Override the saved episode time limit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    if opt.episodes_per_formation < 1:
        parser.error("--episodes-per-formation must be positive")
    if opt.max_steps is not None and opt.max_steps < 1:
        parser.error("--max-steps must be positive")
    if len(set(opt.formations)) != len(opt.formations):
        parser.error("duplicate formations would overwrite outputs")
    device = torch.device(opt.device)
    actor, critic, args, metadata, hashes = load_pair(opt.checkpoint, device)
    if opt.max_steps is not None:
        args.env_max_steps = opt.max_steps
    root = Path(opt.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    backbone = SmolVLMBackbone(critic.cfg, device).eval()
    by_name = {name: count for count, name in NAMES.items()}
    report = dict(checkpoint=str(Path(opt.checkpoint).resolve()), checkpoint_metadata=metadata,
                  actor_sha256=hashes[0], critic_sha256=hashes[1], gamma=args.gamma,
                  formation_tolerance_m=args.formation_tolerance,
                  formation_hold_steps=args.formation_hold_steps, episode_max_steps=args.env_max_steps,
                  note="Actor alone chooses actions. Values are post-rollout VLCM estimates; timeout returns are partial.",
                  episodes=[])
    for formation in opt.formations:
        count = by_name[formation]
        for index in range(opt.episodes_per_formation):
            seed = opt.seed + 1000 * count + index
            print(f"Evaluating {formation}, seed {seed}", flush=True)
            episode = collect_episode(actor, args, device, count, seed)
            predicted, observed = evaluate_values(episode, critic, backbone, args, device)
            result = save_episode(episode, predicted, observed, args, root, count, seed)
            report["episodes"].append(result)
            (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{formation}: {result['outcome']} after {result['steps']} steps", flush=True)
    print(f"Results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
