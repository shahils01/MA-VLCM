"""Sampled temporal/conditional smoothness; no global Lipschitz guarantee.

The input metric is fixed, never built from trainable embeddings or targets.
The condition (task, goals, layout, team identity) is held fixed within pairs.
"""
import json
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def tb3_condition_key(state, episode_id):
    agents = sorted(state.get("agents", []), key=lambda agent: agent.get("id", 0))
    context = {
        "episode": str(episode_id),
        "domain": state.get("task_domain", ""),
        "task": state.get("task", ""),
        "instruction": state.get("task_instruction", ""),
        "team_goals": [(agent.get("id", index), agent.get("goal")) for index, agent in enumerate(agents)],
        "obstacles": state.get("obstacles", state.get("obstacle_layout")),
    }
    return json.dumps(context, sort_keys=True, separators=(",", ":"))


def fixed_tb3_clip_distance(clip, next_clip, position_scale_m=0.1, pixel_scale=0.1):
    """L2 joint metric on fixed, normalized input measurements.

    Video uses RMS across frames. Graphs use the last frame, as does the model,
    and average over nodes so team size alone does not enlarge the distance.
    TB3 observation scales: x/y/goal distances 0.1m, cos/sin 1, linear
    velocity 0.22m/s, angular velocity 2.84rad/s, traversability 1.
    Video uses RGB [0,1] at a fixed 32x32 resolution, divided by 0.1.
    Adjacency uses RMS over entries. Only one scalar leaves the CPU loader.
    """
    if not all(math.isfinite(s) and s > 0 for s in (position_scale_m, pixel_scale)):
        raise ValueError("Lipschitz metric scales must be finite and positive")
    if len(clip) != len(next_clip) or not clip:
        raise ValueError("conditional Lipschitz pairs require equal nonempty clips")
    scales = torch.tensor([position_scale_m, position_scale_m, 1., 1.,
                           .22, 2.84, position_scale_m, 1., position_scale_m])
    obs, next_obs = clip[-1]["robot_obs"].float(), next_clip[-1]["robot_obs"].float()
    if obs.shape != next_obs.shape or obs.ndim != 2 or obs.shape[1] != 9 or not obs.shape[0]:
        raise ValueError("conditional Lipschitz metric requires matching TB3 nine-feature graphs")
    state_sq = float(((next_obs - obs) / scales).square().sum(dim=-1).mean())
    adjacency_sq = float((next_clip[-1]["adj"].float() - clip[-1]["adj"].float()).square().mean())
    video_sq = 0.
    for current, following in zip(clip, next_clip):
        pixels = []
        for frame in (current, following):
            image = frame["image"].convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
            pixels.append(np.asarray(image, dtype=np.float32) / 255.)
        video_sq += float(np.mean(((pixels[1] - pixels[0]) / pixel_scale) ** 2))
    distance = math.sqrt(state_sq + video_sq / len(clip) + adjacency_sq)
    if not math.isfinite(distance):
        raise ValueError("nonfinite conditional Lipschitz input distance")
    return distance


def temporal_delta_loss(pred, next_pred, target, next_target, beta=0.02):
    """Supervise both signs of change; do not force monotonically rising scores."""
    predicted_delta = next_pred.float().view(-1) - pred.float().view(-1)
    target_delta = next_target.float().view(-1) - target.float().view(-1)
    return F.smooth_l1_loss(predicted_delta, target_delta, beta=beta)


def conditional_lipschitz_loss(pred, next_pred, distance, valid):
    """One-sided sampled penalty: mean(max(0, |delta score| - d)^2).

    A zero input distance requires equal outputs. Finite pairs excluded because
    of changed conditions or collision status still receive target supervision.
    This is a soft empirical constraint, not a certified global bound.
    """
    delta = (next_pred.float().view(-1) - pred.float().view(-1)).abs()
    distance = distance.detach().float().view(-1).to(delta.device)
    valid = valid.detach().bool().view(-1).to(delta.device)
    if delta.shape != distance.shape or delta.shape != valid.shape:
        raise ValueError("Lipschitz predictions, distances and validity must align")
    if not torch.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Lipschitz distances must be finite and nonnegative")
    excess = F.relu(delta - distance)
    if not valid.any():
        return delta.sum() * 0.
    return excess[valid].square().mean()


def conditional_lipschitz_diagnostics(pred, next_pred, target, next_target, distance, valid):
    distance, valid = distance.detach().float().view(-1), valid.bool().view(-1)
    count = int(valid.sum())
    delta = (next_pred.detach().float().view(-1) - pred.detach().float().view(-1)).abs()
    target_delta = (next_target.float().view(-1) - target.float().view(-1)).abs()
    return {
        "lipschitz_valid_pairs": count,
        "lipschitz_excluded_pairs": int(valid.numel()) - count,
        "lipschitz_prediction_violations": int(((delta > distance + 1e-6) & valid).sum()),
        "lipschitz_target_violations": int(((target_delta > distance + 1e-6) & valid).sum()),
    }
