"""Categorical signed-potential prediction with interpolated scalar supervision."""
import torch
import torch.nn.functional as F


def value_support(num_bins, minimum=-1.0, maximum=1.0, *, device=None):
    if num_bins < 2 or not minimum < maximum:
        raise ValueError("categorical support requires >=2 bins and minimum < maximum")
    return torch.linspace(minimum, maximum, num_bins, device=device, dtype=torch.float32)


def project_scalar(target, support):
    """Two-bin linear interpolation, including exact centers and endpoints."""
    target = target.float().reshape(-1)
    if not torch.isfinite(target).all():
        raise ValueError("categorical targets must be finite")
    if ((target < support[0] - 1e-6) | (target > support[-1] + 1e-6)).any():
        raise ValueError("categorical target outside support")
    position = ((target.clamp(support[0], support[-1]) - support[0]) /
                (support[-1] - support[0]) * (support.numel() - 1))
    lower = position.floor().long().clamp(0, support.numel() - 1)
    upper = (lower + 1).clamp(max=support.numel() - 1)
    fraction = position - lower.float()
    result = target.new_zeros((target.numel(), support.numel()))
    result.scatter_add_(1, lower[:, None], (1.0 - fraction)[:, None])
    result.scatter_add_(1, upper[:, None], fraction[:, None])
    return result


def categorical_value(logits, minimum=-1.0, maximum=1.0):
    support = value_support(logits.shape[-1], minimum, maximum, device=logits.device)
    return (logits.float().softmax(dim=-1) * support).sum(dim=-1)


def categorical_value_loss(logits, target, minimum=-1.0, maximum=1.0, reduction="mean"):
    support = value_support(logits.shape[-1], minimum, maximum, device=logits.device)
    projected = project_scalar(target.to(logits.device), support)
    if logits.ndim != 2 or logits.shape != projected.shape:
        raise ValueError("categorical logits and targets must have matching batch sizes")
    losses = -(projected * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    if reduction == "none":
        return losses
    if reduction != "mean":
        raise ValueError("unsupported categorical loss reduction")
    return losses.mean()
