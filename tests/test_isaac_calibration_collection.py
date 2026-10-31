import json
import tempfile
from pathlib import Path
import unittest

import torch

import collect_isaac_critic_calibration as collect
import evaluate_python_vlcm_actor as base
from calibrate_critic_by_formation import read_episodes
from model import ModelConfig, MultimodalValueModel
from train_mappo import parser


class IsaacCalibrationTests(unittest.TestCase):
    def test_original_critic_is_preserved_and_rollout_labels_are_compatible(self):
        torch.set_num_threads(1)
        args = parser().parse_args(['--env-max-steps', '5', '--image-size', '32'])
        cfg = ModelConfig(vl_hidden_dim=16, d_model=16, attention_heads=4,
                          image_size=32, clip_len=6, vl_batch_size=2, dropout=0.)
        source = MultimodalValueModel(cfg, torch.device('cpu'), load_backbone=False).eval()
        with torch.no_grad():
            source.target_mean.fill_(-.4)
            source.target_std.fill_(.6)
        from dataclasses import asdict
        saved = dict(model_config=asdict(cfg), model_state=source.state_dict(),
                     run=dict(args=dict(image_hz=5., gamma=.99)))
        critic, inferred_args = collect.make_isaac_critic(saved, args, torch.device('cpu'))
        self.assertTrue(all(not p.requires_grad for p in critic.parameters()))
        self.assertTrue(all(torch.equal(v, critic.core.state_dict()[k]) for k, v in source.state_dict().items()))
        class FakeActor:
            def __call__(self, obs, rnn, masks, deterministic):
                return torch.zeros((6, 2)), None, None
        class FakeBackbone:
            cfg = ModelConfig(vl_batch_size=2)
            def __call__(self, images, prompts):
                return torch.ones((len(images), 16))
        episode = base.collect_episode(FakeActor(), inferred_args, torch.device('cpu'), 3, 1)
        predicted, returns = base.evaluate_values(episode, critic, FakeBackbone(), inferred_args, torch.device('cpu'))
        self.assertTrue(episode['truncated'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = collect.save_csv(root, 'triangle', 1, episode, predicted, returns, args.control_dt)
            (root/'summary.json').write_text(json.dumps(dict(episodes=[entry])))
            _, groups, counts = read_episodes(root)
            self.assertEqual(counts['triangle']['timeouts'], 1)
            self.assertFalse(groups['triangle'])


if __name__ == '__main__':
    unittest.main()
