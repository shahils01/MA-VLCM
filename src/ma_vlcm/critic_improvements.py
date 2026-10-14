"""Optional event emphasis and supervised APF components, without label inputs."""
import math
import torch
import torch.nn.functional as F


def potential_component_target(metadata):
    names = ("goal_cost", "agent_risk", "obstacle_risk", "collision")
    if not all(name in metadata for name in names):
        raise ValueError("auxiliary potential supervision requires all four APF components")
    values = [float(metadata[name]) for name in names]
    if not all(math.isfinite(value) for value in values) or values[0] < 0:
        raise ValueError("invalid potential components")
    if not all(0 <= value <= 1 for value in values[1:]):
        raise ValueError("risk/collision targets must be in [0,1]")
    values[0] = math.log1p(values[0])
    return torch.tensor(values, dtype=torch.float32)


def event_weights(target, next_target=None, *, boost=1.0, threshold=0.03):
    # Absolute collision plateaus also need emphasis: zero delta does not
    # imply a safe state. Other events receive the same weight in either sign.
    target = target.detach().float().view(-1)
    event = target.le(-1 + 1e-6)
    if next_target is not None:
        following = next_target.detach().float().view(-1)
        event = event | following.le(-1 + 1e-6) | (following - target).abs().ge(threshold)
    return torch.where(event, target.new_full(target.shape, boost), torch.ones_like(target))


def weighted_mean(losses, weights):
    if losses.shape != weights.shape:
        raise ValueError("event weights must align with per-example losses")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def potential_auxiliary_loss(outputs, targets):
    targets = targets.float()
    predicted = outputs["potential_components"].float()
    logits = outputs["collision_logit"].float().view(-1)
    if targets.shape != (logits.numel(), 4) or predicted.shape != targets[:, :3].shape:
        raise ValueError("potential auxiliary output/target shapes do not match")
    losses = {
        "aux_goal_loss": F.smooth_l1_loss(predicted[:, 0], targets[:, 0], beta=0.05),
        "aux_agent_risk_loss": F.smooth_l1_loss(predicted[:, 1], targets[:, 1], beta=0.05),
        "aux_obstacle_risk_loss": F.smooth_l1_loss(predicted[:, 2], targets[:, 2], beta=0.05),
    }
    collision = targets[:, 3]
    # Inverse-frequency weights; handle single-class batches without NaNs.
    positives, negatives = collision.sum(), (1 - collision).sum()
    pos_weight = (collision.numel() / (2 * positives.clamp_min(1))).clamp(max=10)
    neg_weight = (collision.numel() / (2 * negatives.clamp_min(1))).clamp(max=10)
    weights = torch.where(collision.bool(), pos_weight, neg_weight)
    losses["aux_collision_loss"] = weighted_mean(
        F.binary_cross_entropy_with_logits(logits, collision, reduction="none"), weights)
    return sum(losses.values()), losses


def change_metrics(pred, following, target, next_target, threshold=0.03):
    delta = following.detach().float().view(-1) - pred.detach().float().view(-1)
    truth = next_target.detach().float().view(-1) - target.detach().float().view(-1)
    drops, rises = truth.le(-threshold), truth.ge(threshold)
    quiet = truth.abs().lt(threshold)
    return {"drop_pairs": int(drops.sum()), "drop_direction_hits": int((drops & delta.lt(0)).sum()),
            "recovery_pairs": int(rises.sum()), "recovery_direction_hits": int((rises & delta.gt(0)).sum()),
            "quiet_pairs": int(quiet.sum()), "false_drop_pairs": int((quiet & delta.le(-threshold)).sum()),
            "change_abs_error_sum": float((delta - truth).abs().sum()), "change_pairs": delta.numel()}
