#!/usr/bin/env python3
"""Collect Python rollout returns and unadapted frozen Isaac VLCM predictions.

The actor is a saved Python MAPPO actor, used only to generate behavior. The
critic is exclusively the original supervised Isaac checkpoint. This script
performs no training or calibration fit; feed its output to the separate
calibrate_critic_by_formation.py program.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_python_vlcm_actor as base
from model import ModelConfig, SmolVLMBackbone
from python_mappo import PythonValueCritic, make_env


def make_isaac_critic(saved, actor_args, device):
    if saved.get("format") == "python_mappo_vlcm_v1":
        raise ValueError("--isaac-critic must point to the supervised Isaac best.pt")
    cfg = ModelConfig(**saved["model_config"])
    if cfg.backbone_mode != "frozen":
        raise ValueError("Expected a supervised frozen-backbone checkpoint")
    args = deepcopy(actor_args)
    source_args = saved["run"]["args"]
    camera_hz = 1 / (args.control_dt * args.camera_stride)
    if not np.isclose(camera_hz, source_args["image_hz"]) or not np.isclose(args.gamma, source_args["gamma"]):
        raise ValueError("Actor environment camera rate and gamma must match the supervised critic")
    args.critic, args.freeze_critic, args.frozen_isaac_critic = "vlcm", True, True
    args.freeze_critic_representation = False
    args.image_size, args.clip_len = cfg.image_size, cfg.clip_len
    cfg.vl_batch_size, cfg.dropout = args.vl_batch_size, 0.
    critic = PythonValueCritic(cfg, args, make_env(args, 3), device, saved).eval()
    return critic, args


def save_csv(root, shape, seed, episode, predictions, returns, dt):
    outcome = "success" if episode["terminated"] else "timeout"
    name = f"{shape}_seed{seed}_{outcome}.csv"
    with (root / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "seconds", "predicted_value", "observed_discounted_return", "return_complete"])
        for step, prediction, target in zip(episode["steps"], predictions, returns):
            writer.writerow([step, step * dt, float(prediction), float(target), episode["terminated"]])
    return dict(formation=shape, seed=seed, outcome=outcome,
                steps=int(episode["steps"][-1]), return_complete=episode["terminated"], measurements=name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor-checkpoint", required=True, type=Path,
                   help="Existing Python MAPPO checkpoint directory; only its actor selects actions")
    p.add_argument("--isaac-critic", required=True, type=Path, help="Original supervised Isaac best.pt")
    p.add_argument("--output-dir", required=True, type=Path, help="New directory for calibration rollout CSVs")
    p.add_argument("--episodes-per-formation", type=int, default=30)
    p.add_argument("--seed", type=int, default=9800001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt = p.parse_args()
    if opt.episodes_per_formation < 1 or opt.seed < 0:
        p.error("Episode count must be positive and seed nonnegative")
    device = torch.device(opt.device)
    actor, _, actor_args, metadata, actor_hashes = base.load_pair(opt.actor_checkpoint, device)
    raw = opt.isaac_critic.read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    critic, args = make_isaac_critic(saved, actor_args, device)
    opt.output_dir.mkdir(parents=True, exist_ok=False)
    backbone = SmolVLMBackbone(critic.cfg, device).eval()
    report = dict(checkpoint=str(opt.isaac_critic.resolve()), critic_sha256=hashlib.sha256(raw).hexdigest(),
                  actor_checkpoint=str(opt.actor_checkpoint.resolve()), actor_sha256=actor_hashes[0],
                  actor_checkpoint_metadata=metadata, gamma=args.gamma, value_units="Isaac discounted clipped reward",
                  target_units="Python raw discounted reward", episode_max_steps=args.env_max_steps,
                  note=("Original Isaac critic and actor are frozen; no gradients or model updates. "
                        "Timeout rewards are partial; the calibration fit excludes them. "
                        "Fits describe this behavior policy and selected successes."), episodes=[])
    for count, shape in base.NAMES.items():
        for index in range(opt.episodes_per_formation):
            seed = opt.seed + 1000 * count + index
            print(f"Collecting {shape}, seed {seed}", flush=True)
            ep = base.collect_episode(actor, args, device, count, seed)
            predicted, observed = base.evaluate_values(ep, critic, backbone, args, device)
            entry = save_csv(opt.output_dir, shape, seed, ep, predicted, observed, args.control_dt)
            report["episodes"].append(entry)
            (opt.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{shape}: {entry['outcome']} after {entry['steps']} controls", flush=True)
    print(f"Calibration rollout data: {opt.output_dir.resolve()}")


if __name__ == "__main__":
    main()
