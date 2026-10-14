"""Task-conditioned whole-trajectory preference auxiliary supervision.

Four uniformly spaced, internally contiguous clips represent each trajectory.
Their learned scores are averaged; BCE on their difference trains the shared
encoder and an independent preference head. Labels never enter model inputs.
This is a pooled-clip adaptation of RoboMeter, not its pref-token architecture.
"""
import copy
import torch
import torch.nn.functional as F


def preference_loader_args(args):
    result = copy.copy(args)
    result.clip_stride = result.clip_len
    result.val_clip_stride = result.clip_len
    result.temporal_consistency_loss_weight = 0
    result.conditional_lipschitz_loss_weight = 0
    result.potential_aux_loss_weight = 0
    result.event_sampling_boost = 1
    result.num_workers = 2
    result.disable_persistent_workers = False
    return result


def trajectory_preference_loss(scores, batch):
    scores = scores.float().view(-1)
    pair = batch['preference_pair_id'].to(scores.device).view(-1)
    side = batch['preference_side'].to(scores.device).view(-1)
    labels = batch['preference_label'].to(scores.device).float().view(-1)
    if scores.numel() != 8 or pair.unique().numel() != 1 or labels.unique().numel() != 1:
        raise ValueError('each preference batch must contain exactly one paired trajectory')
    if int((side == 0).sum()) != 4 or int((side == 1).sum()) != 4 or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('each trajectory requires four clips and a binary preference label')
    logit = scores[side == 0].mean() - scores[side == 1].mean()
    loss = F.binary_cross_entropy_with_logits(logit, labels[0])
    hit = float((logit.detach().gt(0) == labels[0].bool()).item())
    return loss, hit
