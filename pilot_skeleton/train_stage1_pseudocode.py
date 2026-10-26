"""Train the Iteration 1 image-language value head with Qwen frozen."""

import hashlib
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config_pseudocode import PilotConfig
from data.episode_adapter_pseudocode import EpisodeAdapter
from evaluate_pseudocode import evaluate
from models.image_language_value_pseudocode import ImageLanguageValueModel


class ClipDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_clips(rows):
    lengths = [row["features"].shape[0] for row in rows]
    features = torch.nn.utils.rnn.pad_sequence(
        [row["features"] for row in rows], batch_first=True
    )
    mask = torch.arange(features.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    return features, mask, torch.tensor([row["target"] for row in rows])


def split_episodes(episodes, config):
    episodes = list(episodes)
    random.Random(config.seed).shuffle(episodes)
    count = len(episodes)
    test_count = max(1, round(count * config.test_fraction)) if count >= 3 else 0
    val_count = max(1, round(count * config.validation_fraction)) if count >= 2 else 0
    while test_count + val_count >= count:
        if test_count:
            test_count -= 1
        elif val_count:
            val_count -= 1
    train_count = count - test_count - val_count
    return {
        "train": episodes[:train_count],
        "validation": episodes[train_count:train_count + val_count],
        "test": episodes[train_count + val_count:],
    }


def _features_for_episode(model, episode, reference, config, cache_dir):
    key = "|".join((
        config.dataset,
        config.dataset_revision,
        config.model_id,
        config.model_revision,
        str(config.image_hz),
        str(reference["tar_path"]),
    ))
    cache_path = cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    image_indices = EpisodeAdapter.image_indices(episode, config.image_hz)
    images = [episode["images"][index] for index in image_indices]
    features = model.encode_frames(images, episode["prompt"]).half()
    torch.save(features, cache_path)
    return features


def make_feature_rows(adapter, model, references, config, cache_dir):
    rows = []
    for reference in references:
        episode = adapter.load_episode(reference)
        if not adapter.validate_episode(episode):
            print("Skipping invalid episode", episode["episode_id"])
            continue
        clips = adapter.make_clips(episode, {
            "clip_length": config.clip_length,
            "stride": config.clip_stride,
            "horizon": config.horizon,
            "gamma": config.gamma,
            "image_hz": config.image_hz,
        })
        clips = [clip for clip in clips if clip["target_valid_mask"]]
        if not clips:
            print("Skipping episode without complete 20 Hz reward targets", episode["episode_id"])
            continue
        features = _features_for_episode(model, episode, reference, config, cache_dir)
        for clip in clips:
            rows.append({
                "features": features[clip["feature_indices"]],
                "target": clip["target"],
                "episode_id": clip["episode_id"],
                "frame_id": clip["frame_ids"][-1],
                "timestamp": clip["timestamps"][-1],
            })
    return rows


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    model.backbone.eval()
    total_loss = 0.0
    batches = 0

    for features, mask, targets in loader:
        features = features.to(device=device, dtype=torch.float32)
        mask = mask.to(device)
        targets = targets.to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            predictions = model(features, mask)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def train_stage1(config=None):
    config = config or PilotConfig()
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    cache_dir = Path(config.feature_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = EpisodeAdapter()
    episodes = adapter.discover_episodes(
        config.dataset, revision=config.dataset_revision
    )
    splits = split_episodes(episodes, config)
    if not splits["train"]:
        raise ValueError("The dataset has no training episodes")
    first_episode = adapter.load_episode(splits["train"][0])
    if not first_episode["control_rewards_available"]:
        raise ValueError(
            "This dataset has only sampled 5 Hz rewards. Expected-return training needs "
            "the 20 Hz control_rewards.json stream, which is absent from the selected episode."
        )
    model = ImageLanguageValueModel(config.to_dict(), device).to(device)

    datasets = {
        name: ClipDataset(make_feature_rows(adapter, model, refs, config, cache_dir))
        for name, refs in splits.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            collate_fn=collate_clips,
        )
        for name, dataset in datasets.items()
        if len(dataset)
    }

    if "train" not in loaders:
        raise ValueError(
            "No training clips have per-control-step rewards. Existing 5 Hz archives "
            "cannot reconstruct the missing 20 Hz rewards; collect control_rewards.json first."
        )

    run_info = {
        "config": config.to_dict(),
        "episode_splits": {
            name: [str(ref["tar_path"]) for ref in refs]
            for name, refs in splits.items()
        },
        "target": "discounted 20 Hz control-step return from each 5 Hz image through true termination",
    }
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    best_validation = float("inf")
    history = []
    for epoch in range(config.epochs):
        train_loss = run_epoch(model, loaders["train"], optimizer, device)
        validation_loss = (
            run_epoch(model, loaders["validation"], None, device)
            if "validation" in loaders else train_loss
        )
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_loss,
            "validation_mse": validation_loss,
        })
        print(f"epoch={epoch + 1} train_mse={train_loss:.5f} validation_mse={validation_loss:.5f}")

        if validation_loss < best_validation:
            best_validation = validation_loss
            head = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if not key.startswith("backbone.")
            }
            torch.save({"head": head, "run": run_info}, output_dir / "best.pt")

    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    if "test" in loaders:
        checkpoint = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["head"], strict=False)
        test_metrics = evaluate(
            model, datasets["test"].rows, device, output_dir / "test_predictions.csv"
        )
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    return output_dir / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = PilotConfig()
    for argument, field in (
        (args.dataset, "dataset"),
        (args.dataset_revision, "dataset_revision"),
        (args.model_id, "model_id"),
        (args.model_revision, "model_revision"),
        (args.output_dir, "output_dir"),
        (args.epochs, "epochs"),
    ):
        if argument is not None:
            setattr(config, field, argument)
    train_stage1(config)
