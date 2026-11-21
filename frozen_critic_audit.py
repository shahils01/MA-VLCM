"""Fail closed if an actor-only experiment changes its frozen value model."""
import hashlib
import json

import torch


def state_digest(module):
    """Hash tensor values, including buffers and bfloat16, without pickle metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenCriticAudit:
    def __init__(self, critic, backbone, optimizer, source_state=None):
        self.critic, self.backbone = critic, backbone
        self._check_frozen(optimizer)
        if source_state is not None:
            actual = critic.core.state_dict()
            if actual.keys() != source_state.keys() or any(
                not torch.equal(actual[name].detach().cpu(), value.detach().cpu())
                for name, value in source_state.items()
            ):
                raise RuntimeError("Frozen critic differs from the supervised source checkpoint")
        self.critic_digest = state_digest(critic)
        self.backbone_digest = state_digest(backbone) if backbone is not None else None
        self.source_verified = source_state is not None

    def _check_frozen(self, optimizer):
        if optimizer is not None:
            raise RuntimeError("A fully frozen critic must have no optimizer")
        for label, module in (("critic", self.critic), ("backbone", self.backbone)):
            if module is None:
                continue
            if any(child.training for child in module.modules()):
                raise RuntimeError(f"Frozen {label} must remain in evaluation mode")
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError(f"Frozen {label} has a trainable parameter or gradient")

    def check(self, optimizer, full_backbone=False):
        self._check_frozen(optimizer)
        if state_digest(self.critic) != self.critic_digest:
            raise RuntimeError("Frozen critic parameters or buffers changed")
        if full_backbone and self.backbone is not None:
            if state_digest(self.backbone) != self.backbone_digest:
                raise RuntimeError("Frozen visual backbone parameters or buffers changed")
        return dict(critic_state_sha256=self.critic_digest,
                    backbone_state_sha256=self.backbone_digest,
                    source_core_verified=self.source_verified,
                    critic_unchanged=True, backbone_unchanged=True if full_backbone else None,
                    critic_trainable_parameters=0, backbone_trainable_parameters=0,
                    critic_optimizer=None)
