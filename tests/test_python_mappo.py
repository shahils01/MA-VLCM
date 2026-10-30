"""Behavioral checks for PPO, asynchronous inputs and deployment compatibility."""
import tempfile
from dataclasses import asdict
from pathlib import Path
import unittest

import numpy as np
import torch

from model import ModelConfig
from python_mappo import (R_Actor, PythonValueCritic, RolloutCollector, actor_config,
                          compute_gae, context_batch, ppo_update, load_python_critic)
from train_mappo import parser, save_checkpoint, prune_checkpoints


class PythonMAPPOTests(unittest.TestCase):
    def test_python_warm_start_freezes_representation_but_updates_head_and_actor(self):
        self.args.critic = "vlcm"
        self.args.image_size = 32
        collector = RolloutCollector(self.args)
        env = collector.envs[0]
        source = PythonValueCritic(self.cfg, self.args, env, torch.device("cpu"))
        with torch.no_grad():
            source.context_embedding[0].weight.fill_(0.1)
        saved = dict(format="python_mappo_vlcm_v1", critic_kind="vlcm",
                     model_config=asdict(self.cfg), model_state=source.state_dict())
        self.args.freeze_critic_representation = True
        critic = PythonValueCritic(self.cfg, self.args, env, torch.device("cpu"), saved).eval()
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, critic.state_dict()[name]), name)
        self.assertTrue(all(p.requires_grad == name.startswith("core.value_head.")
                            for name, p in critic.named_parameters()))
        before = {name: value.clone() for name, value in critic.state_dict().items()}
        actor = R_Actor(actor_config(), env.observation_spaces[0], env.action_spaces[0])
        actor_before = {name: value.clone() for name, value in actor.state_dict().items()}
        rollout = collector.collect(actor, torch.device("cpu"), 5)
        # Synthetic features are test fixtures; no real VLM or user policy is run.
        for frame in collector.camera_queue:
            frame.features, frame.image = torch.ones(16), None
        ao = torch.optim.Adam(actor.parameters(), lr=.0005)
        co = torch.optim.Adam([p for p in critic.parameters() if p.requires_grad], lr=.0005)
        ppo_update(actor, critic, ao, co, rollout, self.args, torch.device("cpu"), 1)
        changed = {name for name, value in critic.state_dict().items() if not torch.equal(value, before[name])}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("core.value_head.") for name in changed), changed)
        self.assertTrue(any(not torch.equal(value, actor_before[name]) for name, value in actor.state_dict().items()))
        with tempfile.TemporaryDirectory() as root:
            save_checkpoint(Path(root), actor, critic, ao, co, self.args, 10, 1)
            restored, _ = load_python_critic(Path(root) / "critic.pt")
            self.assertTrue(all(p.requires_grad == name.startswith("core.value_head.")
                                for name, p in restored.named_parameters()))
            for name, value in critic.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_retention_preserves_best_and_unrelated_entries(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            for name in ("best", "step_2", "step_10", "step_100", "step_notes"):
                (path/name).mkdir()
                (path/name/"sentinel").write_text("checkpoint")
            (path/"metadata.json").write_text("{}")
            prune_checkpoints(path, 2)
            self.assertEqual({p.name for p in path.iterdir()},
                             {"best", "step_10", "step_100", "step_notes", "metadata.json"})
            prune_checkpoints(path, 0)
            self.assertTrue((path/"step_10/sentinel").is_file())

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        np.random.seed(9)
        self.args = parser().parse_args(["--critic", "state-temporal", "--num-envs", "2",
                                         "--env-max-steps", "3", "--critic-warmup-updates", "0",
                                         "--ppo-epochs", "1", "--critic-epochs", "1",
                                         "--critic-batch-size", "4", "--eval-interval", "0"])
        self.cfg = ModelConfig(vl_hidden_dim=16, d_model=16, attention_heads=4, dropout=0.)

    def test_gae_success_timeout_reset_and_rollout_cutoff(self):
        # Terminal success keeps the +50 and ignores a spurious next value;
        # timeout keeps +gamma*V(actual final state), but neither crosses reset.
        rewards = np.array([[50., 1.], [2., 3.]], np.float32)
        values = np.zeros_like(rewards)
        next_values = np.array([[1000., 10.], [4., 5.]], np.float32)
        terminal = np.array([[True, False], [False, False]])
        done = np.array([[True, True], [False, False]])
        adv, targets = compute_gae(rewards, values, next_values, terminal, done)
        np.testing.assert_allclose(targets, [[50., 10.9], [5.96, 7.95]], atol=1e-5)
        np.testing.assert_array_equal(adv, targets)

    def test_histories_use_current_state_with_causal_stale_images(self):
        self.args.env_max_steps = 100
        collector = RolloutCollector(self.args)
        env = collector.envs[0]
        for step in range(1, 24):
            previous = env.pos.copy()
            env.step(np.zeros((6, 2)))
            collector.histories[0].append(collector.camera_queue, previous)
            row = collector.histories[0].context()
            self.assertAlmostEqual(float(row["image_age"][-1]), (step % 4)*.05, places=6)
            np.testing.assert_allclose(row["robot_obs"][-1, :, :2], env.pos)
            self.assertEqual(row["frame_mask"][-1], True)
        self.assertTrue(row["frame_mask"].all())
        self.assertAlmostEqual(float(row["times"][0]), -1.)

    def test_actual_actor_and_graph_temporal_weights_update_and_export(self):
        collector = RolloutCollector(self.args)
        env = collector.envs[0]
        actor = R_Actor(actor_config(), env.observation_spaces[0], env.action_spaces[0])
        critic = PythonValueCritic(self.cfg, self.args, env, torch.device("cpu"))
        critic.eval()
        actor_before = {k: v.clone() for k, v in actor.state_dict().items()}
        critic_before = {k: v.clone() for k, v in critic.state_dict().items()}
        rollout = collector.collect(actor, torch.device("cpu"), 5)
        self.assertEqual(int(rollout["done"].sum()), 2)
        self.assertEqual(int(rollout["terminated"].sum()), 0)
        self.assertEqual(rollout["contexts"][0]["frame_mask"].sum(), 1)
        # Reset histories contain no previous-episode state.
        self.assertEqual(rollout["contexts"][6]["frame_mask"].sum(), 1)
        ao = torch.optim.Adam(actor.parameters(), lr=.0005)
        co = torch.optim.Adam(critic.parameters(), lr=.0005)
        metrics = ppo_update(actor, critic, ao, co, rollout, self.args, torch.device("cpu"), 1)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertEqual(metrics["train/actor_updated"], 1)
        self.assertTrue(any(not torch.equal(v, actor_before[k]) for k, v in actor.state_dict().items()))
        for prefix in ("core.graph.", "core.temporal."):
            self.assertTrue(any(not torch.equal(v, critic_before[k]) for k, v in critic.state_dict().items()
                                if k.startswith(prefix)), prefix)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            save_checkpoint(path, actor, critic, ao, co, self.args, 10, 1)
            restored = R_Actor(actor_config(), env.observation_spaces[0], env.action_spaces[0])
            restored.load_state_dict(torch.load(path / "actor.pt", weights_only=True))
            obs = collector.obs[0]
            inputs = (obs, np.zeros((6, 1, 64), np.float32), np.ones((6, 1), np.float32))
            self.assertTrue(torch.equal(actor(*inputs, deterministic=True)[0], restored(*inputs, deterministic=True)[0]))
            restored_critic, _ = load_python_critic(path / "critic.pt")
            batch = context_batch(rollout["contexts"][:2], torch.device("cpu"), 16, 1.)
            self.assertTrue(torch.equal(critic(batch), restored_critic(batch)))

    def test_padding_cannot_change_graph_value(self):
        collector = RolloutCollector(self.args)
        critic = PythonValueCritic(self.cfg, self.args, collector.envs[0], torch.device("cpu"))
        critic.eval()
        batch = context_batch([collector.histories[0].context()], torch.device("cpu"), 16, 1.)
        first = critic(batch)
        batch["robot_obs"][~batch["robot_mask"]] = 100000.
        self.assertTrue(torch.allclose(first, critic(batch), atol=1e-5))

    def test_camera_cache_encodes_once_across_rollouts(self):
        self.args.critic = "vlcm"
        self.args.image_size = 32
        self.args.env_max_steps = 100
        self.args.num_envs = 1
        collector = RolloutCollector(self.args)
        class TestEncoder:
            # Test fixture only; training has no synthetic/fallback image path.
            cfg = ModelConfig(vl_batch_size=2)
            count = 0
            def __call__(self, images, prompts):
                self.count += len(images)
                return torch.ones(len(images), 16)
        encoder = TestEncoder()
        self.assertEqual(collector.encode_cameras(encoder), 1)
        for _ in range(5):
            old = collector.envs[0].pos.copy()
            collector.envs[0].step(np.zeros((6, 2)))
            collector.histories[0].append(collector.camera_queue, old)
        self.assertEqual(collector.encode_cameras(encoder), 1)
        self.assertEqual(collector.encode_cameras(encoder), 0)
        self.assertEqual(encoder.count, 2)
        row = collector.histories[0].context()
        self.assertTrue(all(frame.image is None for frame in row["cameras"]))
        context_batch([row], torch.device("cpu"), 16, 1.)


if __name__ == "__main__":
    unittest.main()
