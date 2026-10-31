"""Checks for the standalone 7/8-agent evaluation adapter."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

import evaluate_polygon_ood as ood
from model import ModelConfig
from python_mappo import EpisodeHistory, PythonValueCritic, context_batch
from train_mappo import parser


class PolygonOODTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = parser().parse_args([])
        self.args.image_size = 32
        self.args.vl_batch_size = 2

    def test_variable_batch_matches_existing_six_robot_contract(self):
        env = ood.make_ood_env(self.args, 3, 4)
        env.reset(seed=4)
        queue = []
        history = EpisodeHistory(env, self.args, queue)
        for frame in queue:
            frame.features, frame.image = torch.ones(16), None
        context = history.context()
        expected = context_batch([context], torch.device('cpu'), 16, 1.)
        actual = ood.context_batch_ood([context], torch.device('cpu'), 16, 1., pad_to=6)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_seven_eight_robots_actor_and_graph_shapes(self):
        class Actor:
            def __call__(self, obs, recurrent, mask, deterministic):
                self.shape = obs.shape
                return torch.zeros((len(obs), 2)), None, None
        class Backbone:
            cfg = ModelConfig(vl_batch_size=2)
            def __call__(self, images, prompts):
                return torch.ones((len(images), 16))
        class Critic:
            cfg = ModelConfig(vl_hidden_dim=16)
            def __call__(self, batch):
                self.nrobots = batch['robot_obs'].shape[2]
                return torch.ones(batch['robot_obs'].shape[0])
        for count in (7, 8):
            with self.subTest(count=count):
                actor, critic = Actor(), Critic()
                episode = ood.collect_episode(actor, self.args, torch.device('cpu'), count, 123, 5)
                self.assertEqual(actor.shape, (count, 25))
                self.assertTrue(episode['truncated'])
                self.assertEqual(episode['steps'], [0, 4, 5])
                slots = episode['env']._formation_config()
                spacing = np.linalg.norm(slots - np.roll(slots, -1, axis=0), axis=1)
                np.testing.assert_allclose(spacing, self.args.formation_spacing)
                predicted, observed = ood.evaluate_values(episode, critic, Backbone(), self.args, torch.device('cpu'))
                self.assertEqual(critic.nrobots, count)
                self.assertEqual(predicted.shape, observed.shape)
                self.assertTrue(np.isfinite(predicted).all())
                # Exercise the actual graph/transformer architecture on the OOD
                # graph using randomly initialized test weights and fake features.
                cfg = ModelConfig(vl_hidden_dim=16, d_model=16, attention_heads=4, dropout=0.)
                real_critic = PythonValueCritic(cfg, self.args, episode['env'], torch.device('cpu')).eval()
                batch = ood.context_batch_ood(episode['contexts'], torch.device('cpu'), 16, 1.)
                with torch.no_grad():
                    self.assertEqual(tuple(real_critic(batch).shape), (len(episode['contexts']),))
                with tempfile.TemporaryDirectory() as temp:
                    result = ood.base.save_episode(episode, predicted, observed, self.args, Path(temp), count, 123)
                    self.assertEqual(result['outcome'], 'timeout')
                    self.assertIsNone(result['realized_return_mse'])
                    self.assertTrue((Path(temp) / result['video']).stat().st_size > 0)


if __name__ == '__main__':
    unittest.main()
