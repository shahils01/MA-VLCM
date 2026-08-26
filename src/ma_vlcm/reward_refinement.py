"""Losses and ordering utilities for RoboMeter/TOPReward-style refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrajectoryOutcome:
    success: bool
    final_progress: float
    completion_time: float


def trajectory_rank_key(outcome: TrajectoryOutcome) -> tuple[float, float, float]:
    """Rank success, then final progress, then shorter completion time."""

    return (
        float(outcome.success),
        float(outcome.final_progress),
        -float(outcome.completion_time),
    )


def preferred_trajectory(left: TrajectoryOutcome, right: TrajectoryOutcome) -> int:
    """Return 1 when left is stronger, 0 when right is stronger, -1 for ties."""

    left_key, right_key = trajectory_rank_key(left), trajectory_rank_key(right)
    return 1 if left_key > right_key else 0 if right_key > left_key else -1


def aggregate_trajectory_scores(frame_scores, mask=None):
    """Aggregate prefix/frame scores without rewarding trajectory length."""

    scores = torch.as_tensor(frame_scores, dtype=torch.float32)
    if mask is None:
        return scores.mean(dim=-1)
    weights = torch.as_tensor(mask, dtype=scores.dtype, device=scores.device)
    return (scores * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def bradley_terry_loss(preferred_scores, weaker_scores):
    """Pairwise preference loss ``-log sigmoid(S_strong-S_weak)``."""

    preferred = torch.as_tensor(preferred_scores, dtype=torch.float32)
    weaker = torch.as_tensor(weaker_scores, dtype=torch.float32, device=preferred.device)
    return -F.logsigmoid(preferred - weaker).mean()


def reward_refinement_loss(
    predicted_progress,
    target_progress,
    preferred_scores,
    weaker_scores,
    preference_weight: float = 1.0,
):
    """Dual absolute-progress and Bradley–Terry trajectory objective."""

    absolute = F.mse_loss(
        torch.as_tensor(predicted_progress, dtype=torch.float32),
        torch.as_tensor(target_progress, dtype=torch.float32),
    )
    preference = bradley_terry_loss(preferred_scores, weaker_scores)
    return absolute + float(preference_weight) * preference, {
        "absolute_progress_loss": absolute.detach(),
        "trajectory_preference_loss": preference.detach(),
    }


def temporal_order_accuracy(prefix_scores, mask=None):
    """Fraction of successful adjacent prefixes whose scores do not decrease."""

    scores = torch.as_tensor(prefix_scores, dtype=torch.float32)
    ordered = scores[..., 1:] >= scores[..., :-1]
    if mask is not None:
        valid = torch.as_tensor(mask, dtype=torch.bool, device=scores.device)
        valid = valid[..., 1:] & valid[..., :-1]
        return ordered[valid].float().mean() if valid.any() else scores.new_tensor(float("nan"))
    return ordered.float().mean()


def value_order_correlation(prefix_scores, mask=None):
    """Spearman correlation between score and prefix time for evaluation."""

    scores = torch.as_tensor(prefix_scores, dtype=torch.float32).reshape(-1)
    if mask is not None:
        valid = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
        scores = scores[valid]
    if scores.numel() < 2:
        return scores.new_tensor(float("nan"))
    score_rank = scores.argsort().argsort().float()
    time_rank = torch.arange(scores.numel(), device=scores.device).float()
    score_rank = score_rank - score_rank.mean()
    time_rank = time_rank - time_rank.mean()
    return (score_rank * time_rank).sum() / (
        score_rank.square().sum().sqrt() * time_rank.square().sum().sqrt()
    ).clamp_min(1.0e-8)
