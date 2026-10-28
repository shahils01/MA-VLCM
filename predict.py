"""Predict the value at an episode image from its causal observation history."""

import argparse
from contextlib import nullcontext
import json

import torch

from data_utils import FormationClipDataset, collate_clips, discover_episodes, prepare_episode, read_episode
from model import ModelConfig, MultimodalValueModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True, help="One episode_*.tar path with its summary sidecar")
    parser.add_argument("--anchor", type=int, default=0, help="Image index after selection at 5 Hz; -1 selects the terminal image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["model_config"])
    device = torch.device(args.device)
    run_args = checkpoint["run"]["args"]
    path, summary = discover_episodes(args.episode)[0]
    episode = prepare_episode(read_episode(path, summary), run_args["gamma"], run_args["return_source"], run_args["image_hz"])
    dataset = FormationClipDataset([episode], cfg.clip_len)
    row = dataset[args.anchor]
    batch = collate_clips([row])
    if episode["complete"] and row["metadata"]["anchor"] == len(episode["selected_indices"]) - 1:
        prediction = 0.0
    else:
        model = MultimodalValueModel(cfg, device).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model.eval()
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        autocast = torch.autocast("cuda", dtype=getattr(torch, cfg.vl_dtype)) if device.type == "cuda" and cfg.vl_dtype != "float32" else nullcontext()
        with torch.no_grad(), autocast:
            prediction = model(batch).item()
    print(json.dumps({"predicted_value": prediction, "recorded_target": batch["target"].item(),
                      "return_source": run_args["return_source"], "definition": checkpoint["run"]["return_definition"]}, indent=2))


if __name__ == "__main__":
    main()
