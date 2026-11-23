"""Check the math, routing and saved-run behavior of the two joint ablations."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from data_utils import (FormationClipDataset, cache_features, collate_clips,
                        feature_cache_path, prepare_episode, read_episode)
from model import ModelConfig, MultimodalValueModel
from train_multitask import JointClipDataset
from test_multitask_value import fixture_episode

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
TASKS = ("formation_control", "goal_to_goal")


class PixelEncoder(torch.nn.Module):
    def forward(self, images, prompts):
        return torch.tensor([image.getpixel((0, 0))[0] for image in images]).float()[:, None].repeat(1, 8)


class JointHeadTests(unittest.TestCase):
    def test_head_routing_normalization_and_selected_head_gradients(self):
        batch = dict(vl_features=torch.randn(2, 6, 8), robot_obs=torch.randn(2, 6, 3, 10),
                     adj=torch.ones(2, 6, 3, 3), robot_mask=torch.ones(2, 6, 3, dtype=torch.bool),
                     frame_mask=torch.ones(2, 6, dtype=torch.bool), times=torch.zeros(2, 6),
                     task_domain=list(TASKS), formation_context=torch.zeros(2, 6, 2))
        for mode in ("shared", "task"):
            cfg = ModelConfig(vl_hidden_dim=8, d_model=16, dropout=0, task_names=TASKS,
                              value_head_mode=mode, formation_context=True)
            model = MultimodalValueModel(cfg, load_backbone=False).eval()
            with torch.no_grad():
                model.task_target_mean.copy_(torch.tensor([10., -100.]))
                model.task_target_std.copy_(torch.tensor([2., 20.]))
                model.value_head[-1].weight.zero_()
                model.value_head[-1].bias.copy_(torch.tensor([2., -3.] if mode == "task" else [2.]))
            z = model(batch, return_normalized=True)
            torch.testing.assert_close(z, torch.tensor([2., -3.] if mode == "task" else [2., 2.]))
            native = model(batch)
            torch.testing.assert_close(native, torch.tensor([14., -160.] if mode == "task" else [14., -60.]))
            torch.testing.assert_close(model.normalize_targets(native, batch), z)
            if mode == "task":
                model(batch, return_normalized=True)[0].backward()
                self.assertEqual(model.value_head[-1].bias.grad.tolist(), [1., 0.])
                self.assertIsNotNone(model.value_head[-1].weight.grad)
            clone = MultimodalValueModel(cfg, load_backbone=False).eval()
            clone.load_state_dict(model.checkpoint_state(), strict=True)
            torch.testing.assert_close(clone(batch), native)

    def test_physical_discount_context_and_cpu_feature_prefetch(self):
        with tempfile.TemporaryDirectory() as temp:
            episodes = [read_episode(*fixture_episode(temp, 1, task)) for task in TASKS]
            prepared = [prepare_episode(ep, .99, "control", 10., 20.) for ep in episodes]
            self.assertAlmostEqual(prepared[0]["control_gamma"] ** 2, prepared[1]["control_gamma"])
            for ep in prepared:
                rewards = sum(ep["control_rewards"][1:], [])
                expected = sum(ep["control_gamma"] ** i * r for i, r in enumerate(rewards))
                self.assertAlmostEqual(ep["returns"][0].item(), expected, places=6)
                self.assertEqual(ep["returns"][-1].item(), 0.)
            torch.testing.assert_close(prepared[0]["formation_context"][:, 0], torch.tensor([0., .1, .2, .3]))
            self.assertEqual(prepared[1]["formation_context"].abs().sum().item(), 0.)
            cfg = ModelConfig(vl_hidden_dim=8, vl_batch_size=3)
            cache_features(prepared, PixelEncoder(), cfg, temp, num_workers=2)
            for ep in prepared:
                features = torch.load(feature_cache_path(ep, cfg, temp), weights_only=True)
                torch.testing.assert_close(features[:, 0].float(), torch.tensor([0., 40., 80., 120.]))
            batch = collate_clips([FormationClipDataset(prepared, clip_len=6)[2]])
            self.assertEqual(batch["task_domain"], ["formation_control"])
            self.assertEqual(batch["formation_context"].shape, (1, 6, 2))
            new_nav = read_episode(*fixture_episode(temp, 2, TASKS[1], control_hz=20.))
            self.assertEqual(prepare_episode(new_nav, source="control", image_hz=10., discount_reference_hz=20.)["control_gamma"], .99)

    def test_equal_task_and_team_size_probability_despite_unequal_clip_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            episodes = []
            for task, counts in zip(TASKS, ((1, 2, 3, 4), (5, 3, 2, 1))):
                for n, count in zip(range(3, 7), counts):
                    episodes += [prepare_episode(read_episode(*fixture_episode(temp, i, task, num_agents=n)),
                                                 source="control", image_hz=10.) for i in range(count)]
            stats = {task: dict(target_std=1.) for task in TASKS}
            dataset = JointClipDataset(episodes, task_statistics=stats, balance_agent_counts=True)
            mass = Counter()
            for w, (i, _) in zip(dataset.sample_weights, dataset.samples):
                mass[(episodes[i]["task_domain"], episodes[i]["num_agents"])] += float(w)
            self.assertEqual(len(mass), 8)
            for value in mass.values():
                self.assertAlmostEqual(value, .25)

    def test_both_ablations_resume_standalone_evaluation_and_rate_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backbone = root / "backbone"
            backbone.mkdir()
            (backbone / "config.json").write_text(json.dumps(dict(model_type="idefics3", text_config=dict(
                model_type="llama", hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
                num_hidden_layers=1, intermediate_size=16, vocab_size=16))))
            cfg = ModelConfig(model_id=str(backbone), vl_hidden_dim=8, d_model=16, image_size=32,
                              dropout=0, vl_dtype="float32")
            features = root / "cache/features"
            features.mkdir(parents=True)
            for task in TASKS:
                for n in range(3, 7):
                    for i in range(9):
                        ref = fixture_episode(root / "data", i, task, num_agents=n)
                        ep = prepare_episode(read_episode(*ref), .99, "control", 10., 20.)
                        torch.save(torch.randn(4, 8), feature_cache_path(ep, cfg, features))
            base = [str(ROOT / "train_multitask.py"), "--experiment", "joint",
                    "--formation-dataset", str(root / "data/formation_control"),
                    "--gotogoal-dataset", str(root / "data/goal_to_goal"), "--model-id", str(backbone),
                    "--cache-dir", str(root / "cache"), "--epochs", "1", "--batch-size", "16",
                    "--mixed-precision", "no", "--image-size", "32", "--d-model", "16", "--dropout", "0",
                    "--task-normalization", "--formation-context", "--balance-agent-counts",
                    "--discount-reference-hz", "20", "--expected-capture-hz", "10", "--num-workers", "2"]

            def run(command, env=None):
                result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout)

            for mode in ("shared", "task"):
                run([sys.executable, *base, "--value-head-mode", mode, "--output-dir", str(root / mode)])
                saved = torch.load(root / mode / "latest.pt", weights_only=True)
                self.assertEqual(saved["model_state"]["value_head.3.weight"].shape[0], 1 if mode == "shared" else 2)
                self.assertEqual(saved["run"]["control_gamma_by_task"]["goal_to_goal"], [.99 ** 2])
                self.assertEqual(saved["run"]["args"]["num_agents"], [3, 4, 5, 6])
            shared = json.loads((root / "shared/run.json").read_text())
            task = json.loads((root / "task/run.json").read_text())
            self.assertEqual(shared["splits"], task["splits"])
            self.assertEqual(shared["task_statistics"], task["task_statistics"])
            resume = [sys.executable, *base, "--value-head-mode", "task", "--output-dir", str(root / "task"),
                      "--epochs", "2", "--resume", str(root / "task/latest.pt")]
            run(resume)
            self.assertEqual(torch.load(root / "task/latest.pt", weights_only=True)["epoch"], 2)
            run([sys.executable, str(ROOT / "utils/evaluate.py"), "--checkpoint", str(root / "task/best.pt"),
                 "--cache-dir", str(root / "cache"), "--output-dir", str(root / "evaluation"), "--split", "test"])
            with (root / "evaluation/test_predictions.csv").open() as stream:
                for row in csv.DictReader(stream):
                    path = Path(row["episode"])
                    ep = prepare_episode(read_episode(path, path.with_suffix(".summary.json")), .99, "control", 10., 20.)
                    self.assertAlmostEqual(float(row["target_return"]), ep["returns"][int(row["anchor"])].item())
            rejected = subprocess.run([sys.executable, *base, "--output-dir", str(root / "bad_rate"),
                                      "--expected-control-hz", "20"], text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("recollect rather than relabel", rejected.stdout)

    def test_two_process_cpu_training_and_unpadded_distributed_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backbone = root / "backbone"
            backbone.mkdir()
            (backbone / "config.json").write_text(json.dumps(dict(model_type="idefics3", text_config=dict(
                model_type="llama", hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
                num_hidden_layers=1, intermediate_size=16, vocab_size=16))))
            cfg = ModelConfig(model_id=str(backbone), vl_hidden_dim=8, d_model=16,
                              image_size=32, vl_dtype="float32")
            features = root / "cache/features"
            features.mkdir(parents=True)
            for task in TASKS:
                for i in range(9):
                    ep = prepare_episode(read_episode(*fixture_episode(root / "data", i, task)),
                                         source="control", image_hz=10., discount_reference_hz=20.)
                    torch.save(torch.randn(4, 8), feature_cache_path(ep, cfg, features))
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       "--nproc_per_node=2", str(ROOT / "train_multitask.py"), "--experiment", "joint",
                       "--num-agents", "3", "--formation-dataset", str(root / "data/formation_control"),
                       "--gotogoal-dataset", str(root / "data/goal_to_goal"), "--model-id", str(backbone),
                       "--cache-dir", str(root / "cache"), "--output-dir", str(root / "run"),
                       "--epochs", "1", "--batch-size", "4", "--mixed-precision", "no",
                       "--image-size", "32", "--d-model", "16", "--value-head-mode", "task",
                       "--task-normalization", "--formation-context", "--discount-reference-hz", "20"]
            result = subprocess.run(command, env=dict(os.environ, ACCELERATE_USE_CPU="true", CUDA_VISIBLE_DEVICES=""),
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout)
            saved = torch.load(root / "run/latest.pt", weights_only=True)
            self.assertEqual(saved["run"]["throughput"]["world_size"], 2)
            metrics = json.loads((root / "run/best_validation_metrics.json").read_text())
            expected = len(saved["run"]["splits"]["validation"]) * 3
            self.assertEqual(metrics["count"], expected)
            with (root / "run/best_validation_predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({(row["episode"], row["anchor"]) for row in rows}), expected)


if __name__ == "__main__":
    unittest.main()
