"""Train a multimodal return estimator locally or with Accelerate/DDP on Slurm."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

from data_utils import (
    FormationClipDataset, cache_features, collate_clips, discover_episodes,
    feature_cache_path, prepare_episode, read_episode, split_episodes, training_statistics,
    save_tensor_file,
)
from model import ModelConfig, MultimodalValueModel, SmolVLMBackbone


RETURN_DEFINITIONS = {
    "sampled": "Discounted sum of observed future frame rewards, weighted at their control-step timestamps. Missing control rewards are unknown; this is a sampled-reward proxy, not the full 20 Hz policy return.",
    "control": "Full discounted sum of recorded future control-step rewards through true termination, with gamma applied once per control step.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="../Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--policy-stage", default="all", choices=("all", "initial", "intermediate", "trained", "finetuned"))
    p.add_argument("--return-source", choices=("sampled", "control"), default="sampled")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount per 20 Hz control step, not per image")
    p.add_argument("--image-hz", type=float, default=5.0)
    p.add_argument("--clip-len", "--clip_len", type=int, choices=(6, 8), default=6)
    p.add_argument("--clip-stride", type=int, default=1)
    p.add_argument("--model-id", default=ModelConfig.model_id)
    p.add_argument("--model-revision", default="main")
    p.add_argument("--backbone-mode", choices=("frozen", "lora", "full"), default="frozen")
    p.add_argument("--vl-batch-size", type=int, default=4, help="Frames per SmolVLM microbatch")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=1)
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--batch-size", "--batch_size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-episodes", type=int, default=0, help="Random episode subset for a smoke run; 0 uses all")
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--output-dir", default="outputs/smolvlm-value")
    p.add_argument("--checkpoint-every", type=int, default=2)
    p.add_argument("--checkpoint-every-steps", type=int, default=0, help="Save rolling latest.pt every N optimizer steps; 0 disables")
    p.add_argument("--log-every-steps", type=int, default=0, help="Log training loss every N optimizer steps; 0 disables")
    p.add_argument("--patience", type=int, default=6, help="Stop after this many epochs without validation improvement; 0 disables")
    p.add_argument("--skip-test", action="store_true", help="Keep the test split untouched during experiments")
    p.add_argument("--resume", default="")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="ma-vlcm")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-run-name", default="")
    args = p.parse_args()
    if not 0 < args.gamma <= 1 or args.clip_stride < 1 or args.checkpoint_every < 1 or args.patience < 0:
        p.error("gamma must be in (0, 1]; clip-stride/checkpoint-every must be positive and patience nonnegative")
    if min(args.validation_fraction, args.test_fraction) < 0 or args.validation_fraction + args.test_fraction >= 1:
        p.error("Split fractions must be nonnegative and leave a training partition")
    if args.checkpoint_every_steps < 0 or args.log_every_steps < 0:
        p.error("step intervals must be nonnegative")
    return args


def run_epoch(model, loader, optimizer, accelerator, start_batch=0, totals=None, on_step=None):
    training = optimizer is not None
    model.train(training)
    totals = dict(totals or {"squared": 0, "absolute": 0, "baseline_squared": 0, "target": 0, "target_squared": 0, "count": 0})
    remaining = accelerator.skip_first_batches(loader, start_batch) if start_batch else loader
    for batch_index, batch in enumerate(remaining, start=start_batch):
        with accelerator.accumulate(model) if training else nullcontext(), torch.set_grad_enabled(training):
            predictions = model(batch)
            scale = accelerator.unwrap_model(model).target_std
            loss = F.mse_loss(predictions / scale, batch["target"] / scale)
            if training:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        pred, target = accelerator.gather_for_metrics((predictions.detach().float(), batch["target"].float()))
        error = pred - target
        totals["squared"] += error.square().sum().item()
        totals["absolute"] += error.abs().sum().item()
        baseline = accelerator.unwrap_model(model).target_mean
        totals["baseline_squared"] += (target - baseline).square().sum().item()
        totals["target"] += target.sum().item()
        totals["target_squared"] += target.square().sum().item()
        totals["count"] += len(error)
        if training and accelerator.sync_gradients and on_step:
            on_step(batch_index + 1, totals, error.square().mean().item())
    variance = totals["target_squared"] - totals["target"] ** 2 / totals["count"]
    return {
        "mse": totals["squared"] / totals["count"], "mae": totals["absolute"] / totals["count"],
        "constant_baseline_mse": totals["baseline_squared"] / totals["count"],
        "r2": 1 - totals["squared"] / variance if variance > 1e-8 else 0.0,
        "count": totals["count"],
    }


def regression_metrics(predictions, targets, baseline):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    errors = predictions - targets
    variance = np.sum((targets - targets.mean()) ** 2)
    return {
        "count": len(targets), "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(np.abs(errors))),
        "pearson": float(np.corrcoef(predictions, targets)[0, 1]) if len(targets) > 1 and min(np.std(predictions), np.std(targets)) > 1e-8 else None,
        "r2": float(1 - np.sum(errors ** 2) / variance) if variance > 1e-8 else None,
        "constant_baseline_mse": float(np.mean((targets - baseline) ** 2)),
    }


@torch.no_grad()
def evaluate_test(model, dataset, batch_size, device, output_dir, split_name="test"):
    import csv
    from accelerate.utils import send_to_device

    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_clips):
        metadata = batch.pop("metadata")
        batch = send_to_device(batch, device)
        # Evaluation on the main rank uses the same precision as training.
        predictions = model(batch).float().cpu().tolist()
        for info, pred, target in zip(metadata, predictions, batch["target"].cpu().tolist()):
            rows.append(dict(info, predicted_value=pred, target_return=target))
    with (output_dir / f"{split_name}_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    baseline = model.target_mean.item()
    metrics = regression_metrics([r["predicted_value"] for r in rows], [r["target_return"] for r in rows], baseline)
    episode_errors = {}
    for row in rows:
        episode_errors.setdefault(row["episode"], []).append((row["predicted_value"] - row["target_return"]) ** 2)
    metrics["episode_mean_mse"] = float(np.mean([np.mean(errors) for errors in episode_errors.values()]))
    metrics["episodes"] = len(episode_errors)
    for key in ("num_agents", "stage", "success"):
        metrics[f"by_{key}"] = {}
        for value in sorted({r[key] for r in rows}):
            selected = [r for r in rows if r[key] == value]
            metrics[f"by_{key}"][str(value)] = regression_metrics(
                [r["predicted_value"] for r in selected], [r["target_return"] for r in selected], baseline
            )
    # A pooled team-size score can conceal large errors on successful episodes.
    metrics["by_formation_outcome"] = {}
    names = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon"}
    for n, success in sorted({(r["num_agents"], r["success"]) for r in rows}):
        selected = [r for r in rows if r["num_agents"] == n and r["success"] == success]
        result = regression_metrics([r["predicted_value"] for r in selected],
                                    [r["target_return"] for r in selected], baseline)
        errors = {}
        for row in selected:
            errors.setdefault(row["episode"], []).append((row["predicted_value"] - row["target_return"]) ** 2)
        result.update(episodes=len(errors), episode_mean_mse=float(np.mean([np.mean(v) for v in errors.values()])))
        metrics["by_formation_outcome"][f"{names.get(n, str(n))}_{'success' if success else 'failure'}"] = result
    return metrics


def main():
    args = parse_args()
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum_steps,
        log_with="wandb" if args.wandb else None,
    )
    set_seed(args.seed, device_specific=True)
    output_dir, cache_dir = Path(args.output_dir), Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run.json").exists() and not args.resume:
        raise FileExistsError(f"{output_dir} already contains a run; choose a new --output-dir or resume its checkpoint")
    references = discover_episodes(args.dataset, args.dataset_revision, args.policy_stage)
    if args.max_episodes:
        references = random.Random(args.seed).sample(references, min(len(references), args.max_episodes))
    with accelerator.main_process_first():
        episodes = [read_episode(path, summary, cache_dir / "index") for path, summary in references]
    excluded = sum(not ep["complete"] for ep in episodes)
    episodes = [prepare_episode(ep, args.gamma, args.return_source, args.image_hz) for ep in episodes if ep["complete"]]
    splits = split_episodes(episodes, args.seed, args.validation_fraction, args.test_fraction)
    if not splits["train"]:
        raise ValueError("No complete training episodes remain")
    accelerator.print(f"Return source: {args.return_source}. {RETURN_DEFINITIONS[args.return_source]}")
    accelerator.print(f"Complete episodes: {len(episodes)}; excluded timeout/unfinished: {excluded}")
    signature = hashlib.sha256(json.dumps({
        "source": args.return_source, "gamma": args.gamma, "image_hz": args.image_hz,
        "clip_stride": args.clip_stride, "include_terminal": False,
        "splits": {name: [ep["identity"] for ep in refs] for name, refs in splits.items()},
    }, sort_keys=True).encode()).hexdigest()

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
    cfg = ModelConfig(
        model_id=args.model_id, model_revision=hf_config._commit_hash or args.model_revision,
        vl_hidden_dim=hf_config.text_config.hidden_size, backbone_mode=args.backbone_mode,
        vl_dtype={"no": "float32", "fp16": "float16", "bf16": "bfloat16"}[args.mixed_precision],
        clip_len=args.clip_len, vl_batch_size=args.vl_batch_size, image_size=args.image_size,
        d_model=args.d_model, gnn_layers=args.gnn_layers, temporal_layers=args.temporal_layers,
        attention_heads=args.attention_heads, dropout=args.dropout, lora_rank=args.lora_rank,
    )
    if args.wandb:
        accelerator.init_trackers(args.wandb_project, config=vars(args), init_kwargs={"wandb": {
            "name": args.wandb_run_name or None, "entity": args.wandb_entity or None,
        }})
    feature_dir = cache_dir / "features" if args.backbone_mode == "frozen" else None
    used_splits = {name: refs for name, refs in splits.items() if name != "test" or not args.skip_test}
    if feature_dir is not None:
        # Only rank zero creates caches; all ranks then share the immutable feature files.
        if accelerator.is_main_process:
            missing = [ep for refs in used_splits.values() for ep in refs
                       if not feature_cache_path(ep, cfg, feature_dir).exists()]
            if missing:
                encoder = SmolVLMBackbone(cfg, accelerator.device)
                cache_features(missing, encoder, cfg, feature_dir)
                del encoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    datasets = {name: FormationClipDataset(refs, args.clip_len, args.clip_stride, feature_dir, cfg, include_terminal=False) for name, refs in used_splits.items()}
    shuffle_generator = torch.Generator()
    loaders = {name: DataLoader(
        dataset, batch_size=args.batch_size, shuffle=name == "train", collate_fn=collate_clips,
        generator=shuffle_generator if name == "train" else None,
        num_workers=args.num_workers, pin_memory=accelerator.device.type == "cuda",
    ) for name, dataset in datasets.items() if len(dataset)}
    model = MultimodalValueModel(cfg, accelerator.device, load_backbone=feature_dir is None)
    stats = training_statistics(splits["train"], args.clip_stride, include_terminal=False)
    for name, value in stats.items():
        getattr(model, name).copy_(value)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_validation, epochs_without_improvement = 0, float("inf"), 0
    start_batch, saved_totals, global_step = 0, None, 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != asdict(cfg) or checkpoint["run"]["signature"] != signature:
            raise ValueError("Resume requires the same model, dataset, splits and return settings")
        model.load_state_dict(checkpoint["model_state"], strict=feature_dir is not None)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_validation = checkpoint["epoch"], checkpoint["best_validation"]
        start_batch = checkpoint.get("next_batch", 0)
        saved_totals = checkpoint.get("train_totals")
        global_step = checkpoint.get("global_step", 0)
        if start_batch and (checkpoint.get("world_size") != accelerator.num_processes or any(
                checkpoint["run"]["args"][name] != getattr(args, name)
                for name in ("seed", "batch_size", "grad_accum_steps"))):
            raise ValueError("Mid-epoch resume requires the same GPU count, seed, batch size and gradient accumulation")
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        if args.epochs <= start_epoch:
            raise ValueError("--epochs must exceed the resumed checkpoint's epoch")
        if not (output_dir / "best.pt").exists():
            best_validation = float("inf")
            epochs_without_improvement = 0
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, loaders["train"])
    val_loader = accelerator.prepare(loaders["validation"]) if "validation" in loaders else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if checkpoint and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    run = {
        "args": vars(args), "model_config": asdict(cfg), "return_definition": RETURN_DEFINITIONS[args.return_source],
        "signature": signature,
        "policy_value": "Conditional expected recorded return under the dataset behavior-policy mixture" if args.policy_stage == "all" else f"Recorded behavior value for policy stage {args.policy_stage}",
        "excluded_timeout_or_unfinished_episodes": excluded,
        "terminal_anchors_excluded": True,
        "splits": {name: [ep["path"] for ep in refs] for name, refs in splits.items()},
        "clips": {name: len(dataset) for name, dataset in datasets.items()},
    }
    if accelerator.is_main_process:
        (output_dir / "run.json").write_text(json.dumps(run, indent=2))
    metrics_path = output_dir / "metrics.json"
    history = json.loads(metrics_path.read_text()) if args.resume and metrics_path.exists() else []
    history = [row for row in history if row["epoch"] <= start_epoch]
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        epoch_lr = optimizer.param_groups[0]["lr"]
        shuffle_generator.manual_seed(args.seed + epoch)
        def on_step(next_batch, totals, batch_mse):
            nonlocal global_step
            global_step += 1
            if args.log_every_steps and global_step % args.log_every_steps == 0:
                accelerator.print(f"epoch={epoch + 1} batch={next_batch}/{len(train_loader)} step={global_step} train_mse={batch_mse:.5f}")
                if args.wandb:
                    accelerator.log({"train/step_mse": batch_mse, "train/epoch": epoch + 1,
                                     "train/batch": next_batch}, step=global_step)
            if args.checkpoint_every_steps and global_step % args.checkpoint_every_steps == 0 and accelerator.is_main_process:
                save_tensor_file({"epoch": epoch, "next_batch": next_batch, "train_totals": totals,
                                  "global_step": global_step, "world_size": accelerator.num_processes, "best_validation": best_validation,
                                  "epochs_without_improvement": epochs_without_improvement,
                                  "model_config": asdict(cfg), "model_state": accelerator.unwrap_model(model).checkpoint_state(),
                                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "run": run},
                                 output_dir / "latest.pt")
        train_metrics = run_epoch(model, train_loader, optimizer, accelerator, start_batch, saved_totals, on_step)
        start_batch, saved_totals = 0, None
        validation = run_epoch(model, val_loader, None, accelerator) if val_loader is not None else train_metrics
        improved = validation["mse"] < best_validation
        best_validation = min(best_validation, validation["mse"])
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        stop_early = val_loader is not None and args.patience and epochs_without_improvement >= args.patience
        scheduler.step(validation["mse"])
        metrics = {f"{split}/{key}": value for split, result in (("train", train_metrics), ("validation", validation)) for key, value in result.items()}
        metrics.update(epoch=epoch + 1, seconds=time.monotonic() - started, **{"train/lr": epoch_lr})
        history.append(metrics)
        accelerator.print(f"epoch={epoch + 1} train_mse={train_metrics['mse']:.5f} validation_mse={validation['mse']:.5f}")
        if args.wandb:
            accelerator.log(metrics, step=global_step)
        if accelerator.is_main_process:
            (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))
            if improved or (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs or stop_early:
                checkpoint = {
                    "epoch": epoch + 1, "global_step": global_step, "best_validation": best_validation,
                    "epochs_without_improvement": epochs_without_improvement,
                    "model_config": asdict(cfg), "model_state": accelerator.unwrap_model(model).checkpoint_state(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "run": run,
                }
                if improved:
                    save_tensor_file(checkpoint, output_dir / "best.pt")
                if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs or stop_early:
                    save_tensor_file(checkpoint, output_dir / f"epoch_{epoch + 1:03d}.pt")
        if stop_early:
            accelerator.print(f"Early stopping after {epoch + 1} epochs; best validation MSE={best_validation:.5f}")
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model_state"], strict=feature_dir is not None)
        for split_name in ("validation", "test"):
            if split_name in datasets and len(datasets[split_name]):
                with accelerator.autocast():
                    metrics = evaluate_test(unwrapped, datasets[split_name], args.batch_size, accelerator.device, output_dir, split_name)
                (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
                accelerator.print(f"Best-checkpoint {split_name} MSE: {metrics['mse']:.5f}; baseline MSE: {metrics['constant_baseline_mse']:.5f}")
                if args.wandb:
                    accelerator.log({f"best_{split_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=global_step + 1)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
