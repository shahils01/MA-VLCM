import contextlib
from dataclasses import asdict, replace
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from data_utils import (
    FormationClipDataset, collate_clips, discounted_returns, discover_episodes,
    feature_cache_path, prepare_episode, read_episode, split_episodes, training_statistics,
)
from gat import GraphAttentionEncoder
from model import ModelConfig, MultimodalValueModel, SmolVLMBackbone


torch.set_num_threads(1)


def write_episode(root, index, robots=3, control=False):
    path = Path(root) / f"episode_{index:06d}.tar"
    success = index % 2 == 0
    with tarfile.open(path, "w") as archive:
        for step in range(4):
            agents = [{
                "pos": [i * 0.1 + index * 0.01 + step * 0.02, i * 0.1],
                "yaw": 0.1 * i, "vel": [0.1, 0.01], "goal_pos": [0.5, 0.5],
                "dist_to_goal": 0.5 - 0.1 * step, "min_neighbor_dist": 0.2,
            } for i in range(robots)]
            state = {
                "agents": agents, "capture_hz": 5.0, "control_hz": 20.0,
                "task_instruction": "Form the assigned polygon.",
                "episode_meta": {"sim_time_s": step * 0.2, "done": step == 3},
                "formation": {"robot_center_bounds_m": [-1.0, 1.0, -1.0, 1.0],
                              "pattern": "polygon", "formation_spacing_m": 0.5},
            }
            pixels = BytesIO()
            Image.new("RGB", (32, 32), (step * 40, index % 255, 100)).save(pixels, format="PNG")
            adjacency = BytesIO()
            np.save(adjacency, np.ones((robots, robots), dtype=np.float32) - np.eye(robots, dtype=np.float32))
            fields = {
                "state.json": json.dumps(state).encode(), "overhead.png": pixels.getvalue(),
                "adj.npy": adjacency.getvalue(),
                "reward.json": json.dumps(0.0 if step == 0 else ((1.0 if success else -1.0) if step == 3 else -0.01)).encode(),
                # This leaks the future outcome and must never become an input feature.
                "progress.json": json.dumps({"episode_success": success}).encode(),
            }
            if control:
                fields["control_rewards.json"] = json.dumps([] if step == 0 else [0.1] * 4).encode()
            for suffix, contents in fields.items():
                info = tarfile.TarInfo(f"ep{index:06d}_step{step:05d}.{suffix}")
                info.size = len(contents)
                archive.addfile(info, BytesIO(contents))
    summary = path.with_suffix(".summary.json")
    summary.write_text(json.dumps({
        "episode_id": index, "num_agents": robots, "success": success,
        "termination_reason": "success" if success else "out_of_bounds",
        "checkpoint": {"stage": "finetuned" if success else "initial", "sha256": "synthetic-policy"},
        "seed": index, "collision_frames": 0,
    }))
    return path, summary


class ReturnTests(unittest.TestCase):
    def episode(self):
        return {
            "timestamps": torch.tensor([0.0, 0.2, 0.3], dtype=torch.float64),
            "control_hz": 20.0, "rewards": torch.tensor([999.0, 2.0, 6.0]),
            "control_rewards": [[], [1., 2., 3., 4.], [5., 6.]], "path": "test.tar",
        }

    def test_control_return_includes_all_steps_and_irregular_terminal_interval(self):
        values = discounted_returns(self.episode(), 0.5, "control")
        torch.testing.assert_close(values, torch.tensor([3.75, 8.0, 0.0]))

    def test_sampled_proxy_weights_observed_endpoint_at_its_control_index(self):
        values = discounted_returns(self.episode(), 0.5, "sampled")
        torch.testing.assert_close(values, torch.tensor([0.4375, 3.0, 0.0]))
        # Subsampling images must not discard raw control rewards.
        torch.testing.assert_close(discounted_returns(self.episode(), 0.5, "control", [0, 2]), torch.tensor([3.75, 0.0]))

    def test_missing_or_incomplete_control_stream_is_rejected(self):
        ep = self.episode()
        ep["control_rewards"] = None
        with self.assertRaises(ValueError):
            discounted_returns(ep, source="control")
        ep["control_rewards"] = [[], [1.0], [1.0]]
        with self.assertRaises(ValueError):
            discounted_returns(ep, source="control")


class ArchiveAndModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.episodes = [prepare_episode(read_episode(*write_episode(self.root, i, 3 + i % 4))) for i in range(4)]

    def tearDown(self):
        self.directory.cleanup()

    def test_archive_index_and_causal_padding(self):
        dataset = FormationClipDataset(self.episodes[:1], clip_len=6)
        row = dataset[1]
        self.assertEqual(row["frame_mask"].tolist(), [False, False, False, False, True, True])
        self.assertEqual(row["robot_obs"].shape, (6, 3, 10))
        self.assertAlmostEqual(row["times"][-1].item(), 0.0)
        self.assertTrue((row["times"] <= 0).all())
        self.assertAlmostEqual(row["robot_obs"][-1, 0, 0].item(), 0.02)
        self.assertNotIn("progress", row)
        self.assertEqual(len(row["images"]), 6)
        torch.testing.assert_close(torch.tensor(row["images"][-1].getpixel((0, 0))), torch.tensor([40, 0, 100]))

    def test_terminal_anchors_are_excluded_and_known_terminal_value_is_zero(self):
        dataset = FormationClipDataset(self.episodes[:1], clip_len=6, include_terminal=False)
        self.assertEqual(len(dataset), 3)
        self.assertEqual([anchor for _, anchor in dataset.samples], [0, 1, 2])
        stats = training_statistics(self.episodes[:1], include_terminal=False)
        torch.testing.assert_close(stats["target_mean"], self.episodes[0]["returns"][:-1].mean())
        import predict

        checkpoint = self.root / "terminal.pt"
        torch.save({"model_config": asdict(ModelConfig()), "run": {
            "args": {"gamma": 0.99, "return_source": "sampled", "image_hz": 5.0},
            "return_definition": "sampled proxy",
        }}, checkpoint)
        from io import StringIO
        stream = StringIO()
        with patch.object(sys, "argv", ["predict.py", "--checkpoint", str(checkpoint),
                                       "--episode", self.episodes[0]["path"], "--anchor", "-1"]), contextlib.redirect_stdout(stream):
            predict.main()
        self.assertEqual(json.loads(stream.getvalue())["predicted_value"], 0.0)

    def test_variable_robot_collation_and_finite_gradients(self):
        dataset = FormationClipDataset(self.episodes, 6)
        batch = collate_clips([dataset[1], dataset[13]])
        batch["vl_features"] = torch.randn(2, 6, 8)
        cfg = ModelConfig(vl_hidden_dim=8, d_model=16, temporal_layers=1, gnn_layers=1, dropout=0)
        model = MultimodalValueModel(cfg, load_backbone=False)
        prediction = model(batch)
        prediction.square().sum().backward()
        self.assertTrue(torch.isfinite(prediction).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(batch["robot_obs"].shape[2], 6)

    def test_padded_robots_and_graph_permutations_do_not_change_team_feature(self):
        torch.manual_seed(4)
        graph = GraphAttentionEncoder(10, 16, 2, 4, 0).eval()
        observations = torch.randn(1, 3, 6, 10)
        mask = torch.tensor([[[False] * 6, [True] * 3 + [False] * 3, [True] * 3 + [False] * 3]])
        adjacency = torch.ones(1, 3, 6, 6)
        original = graph(observations, adjacency, mask)
        changed = observations.clone()
        changed[~mask] = 10000
        torch.testing.assert_close(original, graph(changed, adjacency, mask))
        order = torch.tensor([5, 3, 0, 4, 2, 1])
        permuted = graph(observations[:, :, order], adjacency[:, :, order][:, :, :, order], mask[:, :, order])
        torch.testing.assert_close(original, permuted, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(original).all())

    def test_temporal_model_uses_past_graph_and_masks_padding(self):
        torch.manual_seed(7)
        dataset = FormationClipDataset(self.episodes[:1], 6)
        batch = collate_clips([dataset[2]])
        batch["vl_features"] = torch.randn(1, 6, 8)
        batch["robot_obs"].requires_grad_()
        model = MultimodalValueModel(ModelConfig(vl_hidden_dim=8, d_model=16, temporal_layers=1, gnn_layers=1, dropout=0), load_backbone=False).eval()
        prediction = model(batch)
        prediction.sum().backward()
        self.assertGreater(batch["robot_obs"].grad[0, -2].abs().sum().item(), 0)
        changed = dict(batch)
        changed["vl_features"] = batch["vl_features"].clone()
        changed["vl_features"][~batch["frame_mask"]] = 10000
        torch.testing.assert_close(prediction, model(changed))

    def test_scenario_splits_do_not_leak_across_policies(self):
        episodes = [dict(self.episodes[0], scenario_id=str(i), stage=stage) for i in range(12) for stage in ("initial", "finetuned")]
        splits = split_episodes(episodes)
        groups = {name: {ep["scenario_id"] for ep in eps} for name, eps in splits.items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(sum(map(len, splits.values())), len(episodes))

    def test_train_only_statistics_and_feature_cache_identity(self):
        stats = training_statistics(self.episodes[:1])
        torch.testing.assert_close(stats["target_mean"], self.episodes[0]["returns"].mean())
        cfg = ModelConfig()
        original = feature_cache_path(self.episodes[0], cfg, self.root)
        changed = feature_cache_path(self.episodes[0], replace(cfg, model_revision="different"), self.root)
        self.assertNotEqual(original, changed)


class TinySmolVLMTests(unittest.TestCase):
    def make_backbone(self, mode):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            Idefics3Config, Idefics3ImageProcessor, Idefics3Model,
            Idefics3Processor, PreTrainedTokenizerFast,
        )

        special = ["<unk>", "<pad>", "<s>", "</s>", "<image>",
                   "<fake_token_around_image>", "<end_of_utterance>", "<global-img>"]
        vocabulary = {token: i for i, token in enumerate(special + [f"word{i}" for i in range(92)])}
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>",
            bos_token="<s>", eos_token="</s>", additional_special_tokens=special[4:],
        )
        template = "{% for message in messages %}{% for item in message['content'] %}{% if item['type'] == 'image' %}<image>{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %} Assistant:"
        processor = Idefics3Processor(
            image_processor=Idefics3ImageProcessor(size={"longest_edge": 32}, max_image_size={"longest_edge": 32}),
            tokenizer=tokenizer, image_seq_len=16, chat_template=template,
        )
        config = Idefics3Config(
            text_config={"model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 1,
                         "vocab_size": len(tokenizer), "pad_token_id": 1, "bos_token_id": 2, "eos_token_id": 3},
            vision_config={"hidden_size": 32, "intermediate_size": 64, "num_attention_heads": 4,
                           "num_hidden_layers": 1, "image_size": 32, "patch_size": 4},
            image_token_id=tokenizer.convert_tokens_to_ids("<image>"), scale_factor=2,
        )
        with patch("transformers.Idefics3Model.from_pretrained", return_value=Idefics3Model(config)), patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
            return SmolVLMBackbone(ModelConfig(backbone_mode=mode, vl_hidden_dim=32, image_size=32), torch.device("cpu"))

    def features(self, encoder):
        return encoder([Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32), "blue")], ["goal", "goal"])

    def test_actual_processor_and_idefics3_frozen_forward(self):
        encoder = self.make_backbone("frozen")
        features = self.features(encoder)
        self.assertEqual(features.shape, (2, 32))
        self.assertTrue(torch.isfinite(features).all())
        self.assertFalse(features.requires_grad)
        self.assertGreater((features[0] - features[1]).abs().sum().item(), 0)

    def test_actual_processor_and_full_backbone_gradients(self):
        encoder = self.make_backbone("full")
        features = self.features(encoder)
        features[:, 0].sum().backward()
        gradients = [p.grad for name, p in encoder.named_parameters() if "vision_model" in name]
        self.assertTrue(any(g is not None and g.abs().sum().item() > 0 for g in gradients))

    @unittest.skipUnless(importlib.util.find_spec("peft"), "Optional PEFT package unavailable in test environment")
    def test_lora_gradients(self):
        encoder = self.make_backbone("lora")
        features = self.features(encoder)
        features[:, 0].sum().backward()
        gradients = [p.grad for name, p in encoder.named_parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum().item() > 0 for g in gradients))
        self.assertTrue(all("lora_" in name for name, p in encoder.named_parameters() if p.requires_grad))


class TrainingSmokeTest(unittest.TestCase):
    def test_two_epoch_training_checkpoint_evaluation_and_resume(self):
        import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, cache, output = root / "data", root / "cache", root / "output"
            data.mkdir()
            for i in range(24):
                write_episode(data, i, 3 + i % 4)
            cfg = ModelConfig(model_revision="unit-test", vl_dtype="float32", vl_hidden_dim=8, d_model=16, gnn_layers=1, temporal_layers=1, dropout=0)
            for path, summary in discover_episodes(data):
                ep = prepare_episode(read_episode(path, summary, cache / "index"))
                feature = feature_cache_path(ep, cfg, cache / "features")
                feature.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.randn(4, 8).half(), feature)
            args = ["train.py", "--dataset", str(data), "--cache-dir", str(cache), "--output-dir", str(output),
                    "--epochs", "2", "--mixed-precision", "no", "--batch-size", "4", "--d-model", "16",
                    "--gnn-layers", "1", "--temporal-layers", "1", "--dropout", "0", "--checkpoint-every", "1"]
            config = SimpleNamespace(_commit_hash="unit-test", text_config=SimpleNamespace(hidden_size=8))
            with patch.object(sys, "argv", args), patch("transformers.AutoConfig.from_pretrained", return_value=config):
                train.main()
            self.assertTrue((output / "best.pt").exists())
            self.assertTrue((output / "epoch_002.pt").exists())
            self.assertTrue((output / "test_predictions.csv").exists())
            checkpoint = torch.load(output / "epoch_002.pt", weights_only=True)
            self.assertFalse(any(name.startswith("backbone.") for name in checkpoint["model_state"]))
            self.assertIn("optimizer", checkpoint)
            self.assertIn("constant_baseline_mse", json.loads((output / "test_metrics.json").read_text()))
            # Resume past the snapshot into epoch three, using the identical data and model configuration.
            resumed = args.copy()
            resumed[resumed.index("--epochs") + 1] = "3"
            resumed += ["--resume", str(output / "epoch_002.pt")]
            with patch.object(sys, "argv", resumed), patch("transformers.AutoConfig.from_pretrained", return_value=config):
                train.main()
            self.assertEqual(torch.load(output / "epoch_003.pt", weights_only=True)["epoch"], 3)
            self.assertEqual(len(json.loads((output / "metrics.json").read_text())), 3)
            self.assertIn("scheduler", torch.load(output / "epoch_003.pt", weights_only=True))

            stopped = args.copy()
            stopped[stopped.index("--epochs") + 1] = "5"
            stopped[stopped.index("--output-dir") + 1] = str(root / "stopped")
            stopped += ["--lr", "0", "--patience", "1", "--skip-test"]
            with patch.object(sys, "argv", stopped), patch("transformers.AutoConfig.from_pretrained", return_value=config):
                train.main()
            self.assertEqual(len(json.loads((root / "stopped/metrics.json").read_text())), 2)
            self.assertTrue((root / "stopped/epoch_002.pt").exists())
            self.assertFalse((root / "stopped/test_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
