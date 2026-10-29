"""Evaluate an existing checkpoint on its saved episode split."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from data_utils import FormationClipDataset, cache_features, discover_episodes, feature_cache_path, prepare_episode, read_episode
from model import ModelConfig, MultimodalValueModel, SmolVLMBackbone
from train import evaluate_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="", help="Optional relocated dataset root")
    parser.add_argument("--split", choices=("train", "validation", "test", "external"), default="validation")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-episodes", type=int, default=0,
                        help="Require this many complete episodes before inference; 0 disables the check")
    parser.add_argument("--include-terminal", action="store_true", help="Evaluate raw network output on terminal frames too")
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be nonnegative")
    if args.split == "external" and not args.dataset:
        parser.error("--split external requires --dataset pointing to the new episode archives")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["model_config"])
    run_args = checkpoint["run"]["args"]
    # The last three path components identify a formation episode across machines.
    references = discover_episodes(args.dataset or run_args["dataset"], run_args["dataset_revision"])
    if args.split != "external":
        saved = {tuple(Path(path).parts[-3:]) for path in checkpoint["run"]["splits"][args.split]}
        references = [(path, summary) for path, summary in references if tuple(path.parts[-3:]) in saved]
        if len(references) != len(saved):
            raise ValueError("Could not locate all episodes in the checkpoint's saved split")
    else:
        saved = {tuple(Path(path).parts[-3:]) for paths in checkpoint["run"]["splits"].values() for path in paths}
        if any(tuple(path.parts[-3:]) in saved for path, _ in references):
            raise ValueError("External archives overlap saved split path keys; use saved-split evaluation or uniquely named new episodes")
    episodes = [prepare_episode(read_episode(path, summary, Path(args.cache_dir) / "index"),
                                run_args["gamma"], run_args["return_source"], run_args["image_hz"])
                for path, summary in references]
    excluded = sum(not ep["complete"] for ep in episodes)
    episodes = [ep for ep in episodes if ep["complete"]]
    if not episodes:
        raise ValueError("No complete episodes with known future-return targets")
    if args.expected_episodes and len(episodes) != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} complete evaluation episodes, found {len(episodes)} (excluded {excluded})")
    feature_dir = Path(args.cache_dir) / "features"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.split == "external" and cfg.backbone_mode == "frozen":
        missing = [ep for ep in episodes if not feature_cache_path(ep, cfg, feature_dir).exists()]
        if missing:
            encoder = SmolVLMBackbone(cfg, device)
            cache_features(missing, encoder, cfg, feature_dir)
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
    cached = cfg.backbone_mode == "frozen" and all(feature_cache_path(ep, cfg, feature_dir).exists() for ep in episodes)
    dataset = FormationClipDataset(episodes, cfg.clip_len, run_args["clip_stride"],
                                   feature_dir if cached else None, cfg, include_terminal=args.include_terminal)
    model = MultimodalValueModel(cfg, device, load_backbone=not cached).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=cached)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    autocast = torch.autocast("cuda", dtype=getattr(torch, cfg.vl_dtype)) if device.type == "cuda" and cfg.vl_dtype != "float32" else nullcontext()
    with autocast:
        metrics = evaluate_test(model, dataset, args.batch_size, device, output, args.split)
    metrics.update(checkpoint=str(args.checkpoint), epoch=checkpoint["epoch"], include_terminal=args.include_terminal,
                   return_definition=checkpoint["run"]["return_definition"],
                   excluded_timeout_or_unfinished_episodes=excluded,
                   evaluation_split=args.split)
    (output / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{args.split}: MSE={metrics['mse']:.5f}, MAE={metrics['mae']:.5f}, "
          f"constant baseline MSE={metrics['constant_baseline_mse']:.5f}, episodes={metrics['episodes']}")


if __name__ == "__main__":
    main()
