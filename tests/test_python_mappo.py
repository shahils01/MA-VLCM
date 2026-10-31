"""Behavioral checks for PPO, asynchronous inputs and deployment compatibility."""
import tempfile
from dataclasses import asdict
from pathlib import Path
import unittest

import numpy as np
import torch

from model import ModelConfig, MultimodalValueModel
from python_mappo import (R_Actor, PythonValueCritic, RolloutCollector, actor_config,
                          compute_gae, context_batch, ppo_update, load_python_critic)
from train_mappo import parser, save_checkpoint, prune_checkpoints, validate, load_value_calibration


class PythonMAPPOTests(unittest.TestCase):
    def test_calibrated_isaac_values_remain_frozen_and_round_trip(self):
        self.args.critic = "vlcm"
        self.args.image_size = 32
        self.args.freeze_critic = True
        self.args.frozen_isaac_critic = True
        self.args.agent_counts = [3, 4]
        self.args.value_calibration_coefficients = {"3": [2., 3.], "4": [5., -1.]}
        source = MultimodalValueModel(self.cfg, torch.device("cpu"), load_backbone=False).eval()
        saved = {"model_state": {k: v.clone() for k, v in source.state_dict().items()}}
        collector = RolloutCollector(self.args)
        critic = PythonValueCritic(self.cfg, self.args, collector.envs[0], torch.device("cpu"), saved).eval()
        actor = R_Actor(actor_config(), collector.envs[0].observation_spaces[0], collector.envs[0].action_spaces[0])
        rollout = collector.collect(actor, torch.device("cpu"), 5)
        for frame in collector.camera_queue:
            frame.features, frame.image = torch.ones(16), None
        contexts = rollout["contexts"][:2]
        batch = context_batch(contexts, torch.device("cpu"), 16, 1.)
        with torch.no_grad():
            raw = source(batch)
            expected = raw * torch.tensor([2., 5.]) + torch.tensor([3., -1.])
            self.assertTrue(torch.allclose(critic(batch), expected, atol=1e-5))
        before = {k: v.clone() for k, v in critic.state_dict().items()}
        ao = torch.optim.Adam(actor.parameters(), lr=.0005)
        metrics = ppo_update(actor, critic, ao, None, rollout, self.args, torch.device("cpu"), 1)
        self.assertEqual(metrics["train/actor_updated"], 1)
        self.assertEqual(metrics["train/critic_updated"], 0)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in critic.state_dict().items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in critic.parameters()))
        with tempfile.TemporaryDirectory() as root:
            save_checkpoint(Path(root), actor, critic, ao, None, self.args, 10, 1)
            restored, _ = load_python_critic(Path(root) / "critic.pt")
            with torch.no_grad():
                self.assertTrue(torch.equal(critic(batch), restored(batch)))

    def test_calibration_rejects_wrong_source_or_missing_formation(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "source.pt"
            checkpoint.write_bytes(b"source")
            calibration = Path(root) / "calibration.json"
            report = {"critic_sha256": hashlib.sha256(b"source").hexdigest(), "gamma": .99,
                      "formations": {"triangle": {"status": "exploratory_fit", "slope": 2., "intercept": 3.}}}
            calibration.write_text(json.dumps(report))
            self.assertEqual(load_value_calibration(calibration, checkpoint, [3], .99), {"3": [2., 3.]})
            with self.assertRaises(ValueError):
                load_value_calibration(calibration, checkpoint, [3, 4], .99)
            with self.assertRaises(ValueError):
                load_value_calibration(calibration, checkpoint, [3], .9)
            checkpoint.write_bytes(b"different")
            with self.assertRaises(ValueError):
                load_value_calibration(calibration, checkpoint, [3], .99)

    def test_original_isaac_critic_is_unchanged_and_restores_its_output(self):
        self.args.critic = "vlcm"
        self.args.image_size = 32
        self.args.freeze_critic = True
        self.args.frozen_isaac_critic = True
        source = MultimodalValueModel(self.cfg, torch.device("cpu"), load_backbone=False).eval()
        with torch.no_grad():
            source.target_mean.fill_(-0.4)
            source.target_std.fill_(0.6)
            source.obs_mean.fill_(0.1)
            source.obs_std.fill_(1.3)
        pretrained = {"model_state": {k: v.clone() for k, v in source.state_dict().items()}}
        collector = RolloutCollector(self.args)
        env = collector.envs[0]
        critic = PythonValueCritic(self.cfg, self.args, env, torch.device("cpu"), pretrained).eval()
        self.assertTrue(all(not p.requires_grad for p in critic.parameters()))
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, critic.core.state_dict()[name]), name)
        for frame in collector.camera_queue:
            frame.features, frame.image = torch.ones(16), None
        batch = context_batch([collector.histories[0].context()], torch.device("cpu"), 16, 1.)
        with torch.no_grad():
            self.assertTrue(torch.allclose(critic(batch), source(batch), atol=1e-6))
        actor = R_Actor(actor_config(), env.observation_spaces[0], env.action_spaces[0])
        ao = torch.optim.Adam(actor.parameters())
        with tempfile.TemporaryDirectory() as root:
            save_checkpoint(Path(root), actor, critic, ao, None, self.args, 0, 0)
            restored, restored_args = load_python_critic(Path(root) / "critic.pt")
            self.assertTrue(restored_args.frozen_isaac_critic)
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            with torch.no_grad():
                self.assertTrue(torch.equal(critic(batch), restored(batch)))

    def test_fully_frozen_critic_still_updates_actor_and_saves_without_optimizer(self):
        self.args.critic = "vlcm"
        self.args.image_size = 32
        self.args.freeze_critic = True
        self.args.critic_warmup_updates = 5
        collector = RolloutCollector(self.args)
        env = collector.envs[0]
        critic = PythonValueCritic(self.cfg, self.args, env, torch.device("cpu")).eval()
        self.assertTrue(all(not p.requires_grad for p in critic.parameters()))
        before = {name: value.clone() for name, value in critic.state_dict().items()}
        actor = R_Actor(actor_config(), env.observation_spaces[0], env.action_spaces[0])
        actor_before = {name: value.clone() for name, value in actor.state_dict().items()}
        rollout = collector.collect(actor, torch.device("cpu"), 5)
        for frame in collector.camera_queue:
            frame.features, frame.image = torch.ones(16), None
        ao = torch.optim.Adam(actor.parameters(), lr=.0005)
        metrics = ppo_update(actor, critic, ao, None, rollout, self.args, torch.device("cpu"), 1)
        self.assertEqual(metrics["train/actor_updated"], 1)
        self.assertEqual(metrics["train/critic_updated"], 0)
        self.assertNotIn("critic/loss", metrics)
        self.assertTrue(any(not torch.equal(value, actor_before[name]) for name, value in actor.state_dict().items()))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in critic.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        with tempfile.TemporaryDirectory() as root:
            save_checkpoint(Path(root), actor, critic, ao, None, self.args, 10, 1)
            saved = torch.load(Path(root) / "optimizers.pt", weights_only=True)
            self.assertIsNone(saved["critic_optimizer"])
            restored, _ = load_python_critic(Path(root) / "critic.pt")
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            self.assertTrue(all(torch.equal(value, restored.state_dict()[name]) for name, value in before.items()))

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

    def test_frozen_representation_updates_only_value_head_and_round_trips(self):
        device = torch.device("cpu")
        for kind in ("vlcm", "state-temporal"):
            with self.subTest(critic=kind):
                self.args.critic = kind
                self.args.image_size = 32
                self.args.freeze_critic_representation = True
                collector = RolloutCollector(self.args)
                env = collector.envs[0]
                source = MultimodalValueModel(self.cfg, device, load_backbone=False)
                critic = PythonValueCritic(self.cfg, self.args, env, device,
                                          {"model_state": source.state_dict()}).eval()
                for name, param in critic.named_parameters():
                    self.assertEqual(param.requires_grad, name.startswith("core.value_head."), name)
                    if name.startswith("core.") and not name.startswith("core.value_head."):
                        self.assertTrue(torch.equal(param, source.state_dict()[name[5:]]), name)
                actor = R_Actor(actor_config(), env.observation_spaces[0], env.action_spaces[0])
                actor_before = {k: v.clone() for k, v in actor.state_dict().items()}
                before = {k: v.clone() for k, v in critic.state_dict().items()}
                rollout = collector.collect(actor, device, 5)
                if kind == "vlcm":
                    class TestEncoder:
                        cfg = ModelConfig(vl_batch_size=2)
                        def __call__(self, images, prompts):
                            return torch.ones(len(images), 16)
                    collector.encode_cameras(TestEncoder())
                ao = torch.optim.Adam(actor.parameters(), lr=.0005)
                co = torch.optim.Adam([p for p in critic.parameters() if p.requires_grad], lr=.0005)
                metrics = ppo_update(actor, critic, ao, co, rollout, self.args, device, 1)
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                changed = {k for k, v in critic.state_dict().items() if not torch.equal(v, before[k])}
                self.assertTrue(changed)
                self.assertTrue(all(k.startswith("core.value_head.") for k in changed), changed)
                self.assertTrue(any(not torch.equal(v, actor_before[k]) for k, v in actor.state_dict().items()))
                self.assertTrue(all(p.grad is None for p in critic.parameters() if not p.requires_grad))
                with tempfile.TemporaryDirectory() as root:
                    save_checkpoint(Path(root), actor, critic, ao, co, self.args, 10, 1)
                    restored, _ = load_python_critic(Path(root) / "critic.pt")
                    batch = context_batch(rollout["contexts"][:2], device, 16, 1.)
                    self.assertTrue(torch.equal(critic(batch), restored(batch)))
                    self.assertEqual([k for k, p in critic.named_parameters() if p.requires_grad],
                                     [k for k, p in restored.named_parameters() if p.requires_grad])
                # The scalar regressor is linear, not tanh-bounded.
                with torch.no_grad():
                    critic.core.value_head[-1].weight.zero_()
                    critic.core.value_head[-1].bias.fill_(2.)
                self.assertTrue(torch.allclose(critic(batch), torch.full((2,), 100.)))

    def test_freezing_requires_pretrained_graph_critic(self):
        import contextlib
        import io
        for extra in ([], ["--critic", "mlp"]):
            p = parser()
            args = p.parse_args(["--freeze-critic-representation", *extra])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate(p, args)

    def test_frozen_head_optimization_without_environment(self):
        from unittest.mock import patch
        self.args.freeze_critic_representation = True
        device = torch.device("cpu")
        batch = dict(robot_obs=torch.randn(2, 3, 6, 10),
                     adj=torch.ones(2, 3, 6, 6, dtype=torch.bool),
                     robot_mask=torch.ones(2, 3, 6, dtype=torch.bool),
                     frame_mask=torch.ones(2, 3, dtype=torch.bool),
                     vl_features=torch.randn(2, 3, 16), times=torch.zeros(2, 3),
                     extras=torch.randn(2, 3, 3))
        for kind in ("vlcm", "state-temporal"):
            with self.subTest(critic=kind):
                self.args.critic = kind
                source = MultimodalValueModel(self.cfg, device, load_backbone=False)
                critic = PythonValueCritic(self.cfg, self.args, None, device,
                                          {"model_state": source.state_dict()}).eval()
                before = {k: v.clone() for k, v in critic.state_dict().items()}
                for name, param in critic.named_parameters():
                    self.assertEqual(param.requires_grad, name.startswith("core.value_head."))
                optimizer = torch.optim.Adam([p for p in critic.parameters() if p.requires_grad], lr=.001)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (critic(batch) / self.args.value_scale - torch.tensor([2., -2.])).square().mean()
                    loss.backward()
                    optimizer.step()
                changed = {k for k, v in critic.state_dict().items() if not torch.equal(v, before[k])}
                self.assertTrue(changed)
                self.assertTrue(all(k.startswith("core.value_head.") for k in changed), changed)
                self.assertTrue(all(p.grad is None for p in critic.parameters() if not p.requires_grad))
                with tempfile.TemporaryDirectory() as root:
                    actor = torch.nn.Linear(1, 1)
                    ao = torch.optim.Adam(actor.parameters())
                    save_checkpoint(Path(root), actor, critic, ao, optimizer, self.args, 0, 0)
                    # Graph critics do not use env at construction; isolate this
                    # checkpoint contract test from the external environment.
                    with patch("python_mappo.make_env", return_value=None):
                        restored, _ = load_python_critic(Path(root) / "critic.pt")
                    self.assertTrue(torch.equal(critic(batch), restored(batch)))
                    self.assertEqual([k for k, p in critic.named_parameters() if p.requires_grad],
                                     [k for k, p in restored.named_parameters() if p.requires_grad])
                with torch.no_grad():
                    critic.core.value_head[-1].weight.zero_()
                    critic.core.value_head[-1].bias.fill_(2.)
                self.assertTrue(torch.allclose(critic(batch), torch.full((2,), 100.)))

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
