"""Exercise live-feature gradients and strict adapter-state restoration."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model import ModelConfig, MultimodalValueModel
from train_multitask_lora import load_training_state

torch.set_num_threads(1)


class TinyBackbone(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.adapter = nn.Parameter(torch.arange(8, dtype=torch.float32))
        self.frozen_base = nn.Parameter(torch.ones(8), requires_grad=False)

    def forward(self, images, prompts):
        return torch.stack([self.adapter + self.frozen_base for _ in images])


class JointLoraTests(unittest.TestCase):
    def test_live_backbone_gradients_and_checkpoint_restore(self):
        for head in ("shared", "task"):
            cfg = ModelConfig(backbone_mode="lora", vl_hidden_dim=8, d_model=16,
                              dropout=0, task_names=("formation_control", "goal_to_goal"),
                              value_head_mode=head, formation_context=True)
            with patch("model.SmolVLMBackbone", TinyBackbone):
                model = MultimodalValueModel(cfg)
            batch = dict(images=[[None] * 6], prompt=["task"],
                         robot_obs=torch.randn(1, 6, 3, 10), adj=torch.ones(1, 6, 3, 3),
                         robot_mask=torch.ones(1, 6, 3, dtype=torch.bool),
                         frame_mask=torch.ones(1, 6, dtype=torch.bool), times=torch.zeros(1, 6),
                         task_domain=["formation_control"], formation_context=torch.zeros(1, 6, 2))
            model(batch).sum().backward()
            self.assertGreater(model.backbone.adapter.grad.abs().sum().item(), 0)
            self.assertIsNone(model.backbone.frozen_base.grad)
            state = {k: v.clone() for k, v in model.checkpoint_state().items()}
            self.assertIn("backbone.adapter", state)
            self.assertNotIn("backbone.frozen_base", state)
            with torch.no_grad():
                model.backbone.adapter.add_(1)
            load_training_state(model, state)
            torch.testing.assert_close(model.backbone.adapter, state["backbone.adapter"])
            del state["backbone.adapter"]
            with self.assertRaisesRegex(ValueError, "Incomplete adapter checkpoint"):
                load_training_state(model, state)


if __name__ == "__main__":
    unittest.main()
