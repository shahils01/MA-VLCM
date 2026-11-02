"""Warm-start one frozen-backbone VLCM on formation and go-to-goal returns."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, send_to_device, set_seed

from data_utils import (FormationClipDataset, cache_features, collate_clips,
                        discover_episodes, feature_cache_path, prepare_episode,
                        read_episode, save_tensor_file, split_episodes, training_statistics,
                        episode_path_key)
from model import ModelConfig, MultimodalValueModel, SmolVLMBackbone
from train import evaluate_test

TASKS = ("formation_control", "goal_to_goal")


def archive_key(path):
    return episode_path_key(path)


def joint_splits(episodes, initial_run, seed=42, validation_fraction=.1, test_fraction=.1):
    """Reuse old formation partitions; never train on an old held-out episode."""
    original = {}
    for name, paths in initial_run["splits"].items():
        for path in paths:
            key = archive_key(path)
            if key in original and original[key] != name:
                raise ValueError("Initial checkpoint split contains overlapping archives")
            original[key] = name
    result = {name: [] for name in ("train", "validation", "test")}
    new = []
    for episode in episodes:
        if episode["task_domain"] == "formation_control":
            key = archive_key(episode["path"])
            if key not in original:
                raise ValueError("Formation dataset does not match the initial checkpoint's recorded split; supply its original collection")
            result[original[key]].append(episode)
        else:
            new.append(episode)
    fresh = split_episodes(new, seed, validation_fraction, test_fraction)
    for name in result:
        result[name].extend(fresh[name])
    seen = {}
    for name, refs in result.items():
        for ep in refs:
            scenario = (ep["task_domain"], ep["scenario_id"])
            if scenario in seen and seen[scenario] != name:
                raise ValueError("Initial scenario crosses train/validation/test partitions")
            seen[scenario] = name
    return result


def training_partitions(partitions, training_tasks):
    """Select gradient-bearing episodes without changing either task's holdouts."""
    selected = {name:list(refs) for name,refs in partitions.items()}
    selected["train"] = [ep for ep in selected["train"] if ep["task_domain"] in training_tasks]
    if {ep["task_domain"] for ep in selected["train"]} != set(training_tasks):
        raise ValueError("Every selected training task must have complete training episodes")
    return selected


@torch.no_grad()
def adapt_normalizers(model, statistics):
    """Adopt joint train-only statistics without changing initial predictions."""
    old_mean, old_std = model.obs_mean.clone(), model.obs_std.clone()
    old_target_mean, old_target_std = model.target_mean.clone(), model.target_std.clone()
    new = {k: value.to(model.obs_mean.device) for k, value in statistics.items()}
    first = model.graph.input[0]
    old_weights = first.weight.clone()
    first.weight.mul_(new["obs_std"] / old_std)
    first.bias.add_(old_weights @ ((new["obs_mean"] - old_mean) / old_std))
    last = model.value_head[-1]
    last.weight.mul_(old_target_std / new["target_std"])
    last.bias.copy_((last.bias * old_target_std + old_target_mean - new["target_mean"]) / new["target_std"])
    for key, value in new.items():
        getattr(model, key).copy_(value)


class JointClipDataset(FormationClipDataset):
    def __init__(self, *args, task_statistics, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_statistics = task_statistics
        self.task_counts = Counter(self.episodes[i]["task_domain"] for i, _ in self.samples)
        self.sample_weights = torch.tensor([1. / self.task_counts[self.episodes[i]["task_domain"]]
                                            for i, _ in self.samples], dtype=torch.double)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        task = row["metadata"]["task_domain"]
        row["loss_scale"] = self.task_statistics[task]["target_std"]
        return row


def collate_joint(rows):
    batch = collate_clips(rows)
    batch["loss_scale"] = torch.tensor([row["loss_scale"] for row in rows], dtype=torch.float32)
    return batch


def selection_score(metrics, task_statistics):
    if set(metrics["by_task"]) != set(TASKS):
        raise ValueError("Validation must contain both tasks")
    return float(np.mean([metrics["by_task"][task]["mse"] / task_statistics[task]["target_std"]**2 for task in TASKS]))


def train_epoch(model, loader, optimizer, accelerator):
    model.train()
    squared, raw_squared, count = 0., 0., 0
    for batch in loader:
        with accelerator.accumulate(model):
            pred = model(batch)
            error = pred - batch["target"]
            loss = (error / batch["loss_scale"]).square().mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        raw, normalized = accelerator.gather_for_metrics((error.detach().float(), (error / batch["loss_scale"]).detach().float()))
        raw_squared += raw.square().sum().item()
        squared += normalized.square().sum().item()
        count += len(raw)
    return dict(normalized_mse=squared/count, mse=raw_squared/count, count=count)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formation-dataset", required=True)
    p.add_argument("--gotogoal-dataset", required=True)
    p.add_argument("--init-checkpoint", required=True, help="Formation-trained frozen-backbone best.pt; initializes weights, not its optimizer")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--expected-formation-episodes", type=int, default=3000)
    p.add_argument("--expected-gotogoal-episodes", type=int, default=2000)
    p.add_argument("--training-tasks", nargs="+", choices=TASKS, default=list(TASKS),
                   help="Tasks supplying gradients; goal_to_goal performs new-task-only fine-tuning while both tasks are evaluated")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--clip-stride", type=int, default=1)
    p.add_argument("--image-hz", type=float, default=10.)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--validation-fraction", type=float, default=.1, help="For the new task only; old formation partitions are preserved")
    p.add_argument("--test-fraction", type=float, default=.1)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--checkpoint-every", type=int, default=2)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--resume", default="", help="Resume an epoch checkpoint from this same joint run")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="ma-vlcm-multitask")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-run-name", default="")
    a = p.parse_args()
    a.training_tasks = [task for task in TASKS if task in a.training_tasks]
    if min(a.epochs, a.batch_size, a.grad_accum_steps, a.clip_stride, a.checkpoint_every) < 1 or a.num_workers < 0 or a.patience < 0:
        p.error("Invalid epoch, batch, worker, patience or checkpoint settings")
    if not 0 < a.gamma <= 1 or min(a.validation_fraction, a.test_fraction) <= 0 or a.validation_fraction+a.test_fraction >= 1:
        p.error("Use gamma in (0,1] and positive held-out fractions summing below one")
    return a


def main():
    a = parse_args()
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    accelerator = Accelerator(mixed_precision=a.mixed_precision, gradient_accumulation_steps=a.grad_accum_steps,
                              log_with="wandb" if a.wandb else None)
    set_seed(a.seed)
    root, cache = Path(a.output_dir), Path(a.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"run.json").exists() and not a.resume:
        raise FileExistsError("Choose a new output directory; the initial checkpoint will not be overwritten")
    initial = torch.load(a.init_checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**initial["model_config"])
    if cfg.backbone_mode != "frozen":
        raise ValueError("This experiment requires a frozen-backbone supervised checkpoint")
    # Keep standard prediction/evaluation entrypoints compatible with the joint
    # checkpoint while recording both dataset roots explicitly.
    a.dataset = a.formation_dataset
    a.additional_datasets = [a.gotogoal_dataset]
    a.dataset_revision = "main"
    a.return_source = "control"
    a.clip_len = cfg.clip_len
    a.backbone_mode = "frozen"
    a.policy_stage = "all"
    if initial["run"]["args"]["return_source"] != "control" or a.gamma != initial["run"]["args"]["gamma"]:
        raise ValueError("Keep the original full-control return definition and discount")
    references = []
    for task, source, expected in ((TASKS[0], a.formation_dataset, a.expected_formation_episodes),
                                  (TASKS[1], a.gotogoal_dataset, a.expected_gotogoal_episodes)):
        refs = discover_episodes(source)
        if expected and len(refs) != expected:
            raise ValueError(f"Expected {expected} raw {task} episode archives, found {len(refs)}")
        references.extend((task,path,summary) for path,summary in refs)
    with accelerator.main_process_first():
        episodes = []
        excluded = Counter()
        for task,path,summary in references:
            ep = read_episode(path,summary,cache/"index")
            if ep["task_domain"] != task:
                raise ValueError(f"Wrong task domain in {path}")
            if ep["complete"]:
                episodes.append(prepare_episode(ep,a.gamma,"control",a.image_hz))
            else:
                excluded[task] += 1
    partitions = joint_splits(episodes,initial["run"],a.seed,a.validation_fraction,a.test_fraction)
    splits = training_partitions(partitions,a.training_tasks)
    for name,refs in splits.items():
        expected_tasks = set(a.training_tasks) if name=="train" else set(TASKS)
        if {ep["task_domain"] for ep in refs} != expected_tasks:
            raise ValueError(f"{name} must contain complete episodes from {expected_tasks}")
    task_stats = {}
    for task in TASKS:
        task_training = [ep for ep in splits["train"] if ep["task_domain"] == task]
        if not task_training:
            # For formation evaluation in navigation-only mode, reuse the
            # original training baseline; no formation frames affect this fit.
            task_stats[task] = {key:initial["model_state"][key].item() for key in ("target_mean","target_std")}
            continue
        stats = training_statistics(task_training,a.clip_stride,include_terminal=False)
        task_stats[task] = dict(target_mean=stats["target_mean"].item(),target_std=stats["target_std"].item())
    signature = hashlib.sha256(json.dumps(dict(
        init_sha256=hashlib.sha256(Path(a.init_checkpoint).read_bytes()).hexdigest(), gamma=a.gamma,
        image_hz=a.image_hz,clip_stride=a.clip_stride,seed=a.seed,training_tasks=a.training_tasks,
        splits={name:[ep["identity"] for ep in refs] for name,refs in splits.items()}),sort_keys=True).encode()).hexdigest()
    used = {name:refs for name,refs in splits.items() if name!="test" or not a.skip_test}
    feature_dir = cache/"features"
    if accelerator.is_main_process:
        missing = [ep for refs in used.values() for ep in refs if not feature_cache_path(ep,cfg,feature_dir).exists()]
        if missing:
            encoder = SmolVLMBackbone(cfg,accelerator.device)
            cache_features(missing,encoder,cfg,feature_dir)
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    datasets = {name:JointClipDataset(refs,cfg.clip_len,a.clip_stride,feature_dir,cfg,
                    include_terminal=False,task_statistics=task_stats) for name,refs in used.items()}
    generator = torch.Generator()
    sampler = (WeightedRandomSampler(datasets["train"].sample_weights,len(datasets["train"]),replacement=True,generator=generator)
               if len(a.training_tasks)>1 else None)
    train_loader = DataLoader(datasets["train"],batch_size=a.batch_size,sampler=sampler,
                             shuffle=sampler is None,generator=generator,
                             num_workers=a.num_workers,collate_fn=collate_joint)
    model = MultimodalValueModel(cfg,accelerator.device,load_backbone=False)
    model.load_state_dict(initial["model_state"],strict=True)
    joint_stats = training_statistics(splits["train"],a.clip_stride,include_terminal=False)
    adapt_normalizers(model,joint_stats)
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=2)
    start, best, stale = 0, float("inf"), 0
    if a.resume:
        checkpoint = torch.load(a.resume,map_location="cpu",weights_only=True)
        if checkpoint["run"]["signature"] != signature or checkpoint["model_config"] != asdict(cfg):
            raise ValueError("Resume requires the same joint datasets, splits, initialization and configuration")
        if Path(checkpoint["run"]["args"]["output_dir"]).resolve() != root.resolve():
            raise ValueError("Resume must write to its original output directory")
        model.load_state_dict(checkpoint["model_state"],strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start,best,stale = checkpoint["epoch"],checkpoint["best_validation"],checkpoint["epochs_without_improvement"]
        if a.epochs <= start:
            raise ValueError("epochs must exceed the resumed epoch")
    run = dict(args=vars(a),model_config=asdict(cfg),signature=signature,
        initial_checkpoint=str(Path(a.init_checkpoint).resolve()), task_statistics=task_stats,
        return_definition="Full future native control-step rewards, gamma once per control step; incoming/past reward excluded",
        control_hz_by_task={task:sorted({ep['control_hz'] for ep in episodes if ep['task_domain']==task}) for task in TASKS},
        policy_value="Conditional expected recorded return under each task's dataset behavior-policy mixture",
        splits={name:[ep["path"] for ep in refs] for name,refs in splits.items()},
        pretrained_splits=initial["run"]["splits"],
        excluded_timeout_or_unfinished_by_task=dict(excluded),
        training_tasks=a.training_tasks,
        training_episode_counts=dict(Counter(ep["task_domain"] for ep in splits["train"])),
        sampling=("Equal probability per task at clip level" if len(a.training_tasks)>1 else "Only selected task; shuffled without replacement")+"; per-task train-only return standard deviation scales the loss",
        normalizer_transfer="Selected training tasks' statistics with affine weight conversion preserving initial predictions",
        formation_split="Original formation holdouts preserved; formation training episodes used only if selected")
    accelerator.print(f"Training tasks: {a.training_tasks}; gradient-bearing episodes: {run['training_episode_counts']}; both tasks evaluated")
    if accelerator.is_main_process:
        (root/"run.json").write_text(json.dumps(run,indent=2)+"\n")
    if a.wandb:
        accelerator.init_trackers(a.wandb_project,config=vars(a),init_kwargs={"wandb":{
            "name":a.wandb_run_name or None,"entity":a.wandb_entity or None}})
    model,optimizer,train_loader = accelerator.prepare(model,optimizer,train_loader)
    baselines = {task:stats["target_mean"] for task,stats in task_stats.items()}

    def evaluate(name,label):
        accelerator.wait_for_everyone()
        payload=[None]
        if accelerator.is_main_process:
            with accelerator.autocast():
                metrics=evaluate_test(accelerator.unwrap_model(model),datasets[name],a.batch_size,
                    accelerator.device,root,label,task_baselines=baselines)
            metrics["macro_task_normalized_mse"]=selection_score(metrics,task_stats)
            (root/f"{label}_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
            payload[0]=metrics
        return broadcast_object_list(payload)[0]

    if not a.resume:
        before = evaluate("validation","before_validation")
        accelerator.print(f"Initial validation macro task normalized MSE: {before['macro_task_normalized_mse']:.5f}")
        best = before["macro_task_normalized_mse"]
        if accelerator.is_main_process:
            save_tensor_file(dict(epoch=0,model_config=asdict(cfg),
                model_state=accelerator.unwrap_model(model).checkpoint_state(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                best_validation=best,epochs_without_improvement=0,run=run),root/"best.pt")
    for epoch in range(start,a.epochs):
        started=time.monotonic()
        generator.manual_seed(a.seed+epoch)
        train_metrics=train_epoch(model,train_loader,optimizer,accelerator)
        validation=evaluate("validation","validation")
        score=validation["macro_task_normalized_mse"]
        improved=score<best
        best=min(best,score)
        stale=0 if improved else stale+1
        scheduler.step(score)
        record=dict(epoch=epoch+1,train=train_metrics,validation=validation,seconds=time.monotonic()-started)
        if accelerator.is_main_process:
            with (root/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            accelerator.print(f"Epoch {epoch+1}: validation macro NMSE {score:.5f}; " +
                "; ".join(f"{task} MAE={validation['by_task'][task]['mae']:.4f}" for task in TASKS))
            if a.wandb:
                log={"train/normalized_mse":train_metrics["normalized_mse"],"validation/macro_task_normalized_mse":score}
                log.update({f"validation/{task}/{key}":validation["by_task"][task][key]
                            for task in TASKS for key in ("mse","mae","r2")})
                accelerator.log(log,step=epoch+1)
            checkpoint=dict(epoch=epoch+1,model_config=asdict(cfg),
                model_state=accelerator.unwrap_model(model).checkpoint_state(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                best_validation=best,epochs_without_improvement=stale,run=run)
            save_tensor_file(checkpoint,root/"latest.pt")
            if improved:
                save_tensor_file(checkpoint,root/"best.pt")
            if (epoch+1)%a.checkpoint_every==0 or epoch+1==a.epochs:
                save_tensor_file(checkpoint,root/f"epoch_{epoch+1:03d}.pt")
        if a.patience and stale>=a.patience:
            break
    accelerator.wait_for_everyone()
    unwrapped=accelerator.unwrap_model(model)
    best_checkpoint=torch.load(root/"best.pt",map_location="cpu",weights_only=True)
    unwrapped.load_state_dict(best_checkpoint["model_state"],strict=True)
    final_validation=evaluate("validation","best_validation")
    if not a.skip_test:
        after_test=evaluate("test","after_test")
        unwrapped.load_state_dict(initial["model_state"],strict=True)
        before_test=evaluate("test","before_test")
        if accelerator.is_main_process:
            comparison={task:dict(before=before_test["by_task"][task],after=after_test["by_task"][task]) for task in TASKS}
            (root/"task_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    accelerator.end_training()


if __name__=="__main__":
    main()
