#!/usr/bin/env python3
"""Evaluate an existing shared MAPPO actor and VLCM on 7/8-robot polygons.

This is a standalone evaluation path: it does not modify the environment or
training code and it does not update actor or critic weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_python_vlcm_actor as base
from model import SmolVLMBackbone
from python_mappo import EpisodeHistory, FormationControl

NAMES = {7: "heptagon", 8: "octagon"}
# The existing video writer can render any formation once names are supplied.
base.NAMES.update(NAMES)


def make_ood_env(args, count, max_steps):
    return FormationControl(
        num_agents=count, max_num_agents=count, pattern="polygon",
        formation_spacing=args.formation_spacing,
        formation_tolerance=args.formation_tolerance,
        initial_position_range=args.initial_position_range,
        min_start_separation=args.min_start_separation,
        observation_scale=args.observation_scale, dt=args.control_dt,
        max_steps=max_steps, formation_hold_steps=args.formation_hold_steps,
    )


def collect_episode(actor, args, device, count, seed, max_steps):
    env = make_ood_env(args, count, max_steps)
    obs, _, _ = env.reset(seed=seed)
    if obs.shape != (count, 25):
        raise ValueError(f"Unexpected actor observations: {obs.shape}; expected ({count}, 25)")
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
            action, _, _ = actor(obs, np.zeros((count, 1, args.hidden_size), np.float32),
                                 np.ones((count, 1), np.float32), deterministic=True)
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


def context_batch_ood(contexts, device, feature_dim, graph_radius, pad_to=None):
    """Same graph/context contract as training, with a variable robot dimension."""
    b, t = len(contexts), len(contexts[0]["frame_mask"])
    nmax = max(row["robot_obs"].shape[1] for row in contexts)
    if pad_to is not None:
        if pad_to < nmax:
            raise ValueError("Padding cannot be smaller than the active team")
        nmax = pad_to
    obs = np.zeros((b, t, nmax, 10), np.float32)
    mask = np.zeros((b, t, nmax), bool)
    features = torch.zeros(b, t, feature_dim)
    for i, row in enumerate(contexts):
        n = row["robot_obs"].shape[1]
        obs[i, :, :n] = row["robot_obs"]
        mask[i, :, :n] = True
        for j, frame in enumerate(row["cameras"]):
            if frame.features is None:
                raise RuntimeError("All camera features must be encoded before VLCM inference")
            features[i, j] = frame.features
    distance = np.linalg.norm(obs[:, :, :, None, :2] - obs[:, :, None, :, :2], axis=-1)
    adj = (distance <= graph_radius) & mask[:, :, :, None] & mask[:, :, None, :]
    batch = dict(robot_obs=obs, robot_mask=mask, adj=adj, vl_features=features,
                 frame_mask=np.stack([row["frame_mask"] for row in contexts]),
                 times=np.stack([row["times"] for row in contexts]),
                 extras=np.stack([np.column_stack((row["image_age"], row["extras"])) for row in contexts]),
                 shared_state=np.stack([row["shared_state"] for row in contexts]))
    return {key: torch.as_tensor(value, device=device) for key, value in batch.items()}


def evaluate_values(episode, critic, backbone, args, device):
    with torch.no_grad():
        queue = episode["queue"]
        for start in range(0, len(queue), backbone.cfg.vl_batch_size):
            frames = queue[start:start + backbone.cfg.vl_batch_size]
            features = backbone([frame.image for frame in frames], [frame.prompt for frame in frames]).detach().cpu()
            for frame, feature in zip(frames, features):
                frame.features, frame.image = feature, None
        values = []
        for start in range(0, len(episode["contexts"]), args.critic_batch_size):
            contexts = episode["contexts"][start:start + args.critic_batch_size]
            batch = context_batch_ood(contexts, device, critic.cfg.vl_hidden_dim, args.graph_radius)
            values.append(critic(batch).cpu().numpy())
    predicted = np.concatenate(values).astype(float)
    if episode["terminated"]:
        predicted[-1] = 0.0  # terminal continuation, not the network output
    observed = base.discounted_returns(episode["rewards"], args.gamma)[episode["steps"]]
    return predicted, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory containing matching actor.pt and critic.pt")
    parser.add_argument("--output-dir", required=True, help="A new output directory")
    parser.add_argument("--formations", nargs="+", type=int, choices=(7, 8), default=[7, 8])
    parser.add_argument("--episodes-per-formation", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9700001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    if opt.episodes_per_formation < 1 or opt.max_steps < 1 or opt.seed < 0:
        parser.error("Episode count and maximum steps must be positive; seed must be non-negative")
    if len(set(opt.formations)) != len(opt.formations):
        parser.error("Duplicate formations would overwrite output")
    device = torch.device(opt.device)
    actor, critic, args, metadata, hashes = base.load_pair(opt.checkpoint, device)
    root = Path(opt.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    backbone = SmolVLMBackbone(critic.cfg, device).eval()
    report = dict(checkpoint=str(Path(opt.checkpoint).resolve()), checkpoint_metadata=metadata,
                  actor_sha256=hashes[0], critic_sha256=hashes[1], gamma=args.gamma,
                  formation_tolerance_m=args.formation_tolerance,
                  formation_hold_steps=args.formation_hold_steps, episode_max_steps=opt.max_steps,
                  note=("Actor and VLCM are evaluated without training on unseen team sizes. "
                        "Actor controls actions; critic values are computed afterward. "
                        "Timeout returns are partial; terminal value is fixed to zero."), episodes=[])
    for count in opt.formations:
        for index in range(opt.episodes_per_formation):
            seed = opt.seed + count * 1000 + index
            print(f"Evaluating {NAMES[count]} with {count} robots, seed {seed}", flush=True)
            episode = collect_episode(actor, args, device, count, seed, opt.max_steps)
            predicted, observed = evaluate_values(episode, critic, backbone, args, device)
            result = base.save_episode(episode, predicted, observed, args, root, count, seed)
            report["episodes"].append(result)
            (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{NAMES[count]}: {result['outcome']} after {result['steps']} steps", flush=True)
    print(f"Results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
