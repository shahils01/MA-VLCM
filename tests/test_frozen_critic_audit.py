import unittest

import torch
from torch import nn

from frozen_critic_audit import FrozenCriticAudit, state_digest


class FrozenAuditTests(unittest.TestCase):
    def setUp(self):
        self.critic = nn.Module()
        self.critic.core = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
        self.critic.requires_grad_(False).eval()
        self.backbone = nn.Linear(2, 3).to(torch.bfloat16).requires_grad_(False).eval()
        self.source = {k: v.clone() for k, v in self.critic.core.state_dict().items()}
        self.audit = FrozenCriticAudit(self.critic, self.backbone, None, self.source)

    def test_actor_training_keeps_frozen_model_and_buffers_identical(self):
        actor = nn.Linear(2, 3)
        before = state_digest(actor)
        optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
        actor(torch.ones(2, 2)).square().mean().backward()
        optimizer.step()
        self.assertNotEqual(before, state_digest(actor))
        self.assertTrue(self.audit.check(None, full_backbone=True)["backbone_unchanged"])

    def test_catches_buffer_mutation(self):
        self.critic.core[1].running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, "buffers changed"):
            self.audit.check(None)

    def test_catches_encoder_mutation(self):
        self.backbone.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "backbone parameters or buffers changed"):
            self.audit.check(None, full_backbone=True)

    def test_rejects_trainable_critic_optimizer_or_training_mode(self):
        self.critic.core[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "trainable"):
            self.audit.check(None)
        self.critic.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            self.audit.check(object())
        self.critic.train()
        with self.assertRaisesRegex(RuntimeError, "evaluation mode"):
            self.audit.check(None)

    def test_rejects_different_supervised_source(self):
        self.source["0.bias"].add_(1)
        with self.assertRaisesRegex(RuntimeError, "source checkpoint"):
            FrozenCriticAudit(self.critic, self.backbone, None, self.source)


if __name__ == "__main__":
    unittest.main()
