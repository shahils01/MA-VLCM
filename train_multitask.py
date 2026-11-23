"""Train one frozen-backbone VLCM on selected formation/navigation returns."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list, set_seed

from data_utils import (FormationClipDataset, cache_features, collate_clips,
                        discover_episodes, feature_cache_path, prepare_episode,
                        read_episode, save_tensor_file, split_episodes, training_statistics,
                        episode_path_key)
from model import ModelConfig, MultimodalValueModel, SmolVLMBackbone
from train import evaluate_test

TASKS = ("formation_control", "goal_to_goal")
EXPERIMENTS = {
    "joint": TASKS,
    "formation_only": ("formation_control",),
    "joint_without_obstacles": TASKS,
    "joint_with_obstacles": TASKS,
}


def select_references(references, task, num_agents=None, navigation_condition="all"):
    """Filter summary metadata before opening archives or caching image features."""
    selected = []
    for path, summary_path in references:
        summary = json.loads(Path(summary_path).read_text())
        if num_agents and int(summary["num_agents"]) not in num_agents:
            continue
        if task == "goal_to_goal" and navigation_condition != "all":
            declared = summary.get("condition")
            locations = set(Path(path).parts) & {"with_obstacles", "without_obstacles"}
            if declared is None and len(locations) == 1:
                declared = next(iter(locations))
            if declared not in {"with_obstacles", "without_obstacles"}:
                raise ValueError(f"Cannot determine obstacle condition for {path}; use the combined dataset with condition metadata")
            if locations and locations != {declared}:
                raise ValueError(f"Obstacle condition disagrees with directory: {path}")
            if declared != navigation_condition:
                continue
        selected.append((path, summary_path))
    if not selected:
        raise ValueError(f"No {task} episodes match agents={num_agents}, navigation_condition={navigation_condition}")
    return selected


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
    def __init__(self, *args, task_statistics, balance_agent_counts=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_statistics = task_statistics
        self.task_counts = Counter(self.episodes[i]["task_domain"] for i, _ in self.samples)
        self.sample_weights = torch.tensor([1. / self.task_counts[self.episodes[i]["task_domain"]]
                                            for i, _ in self.samples], dtype=torch.double)
        if balance_agent_counts:
            groups = Counter((self.episodes[i]["task_domain"], self.episodes[i]["num_agents"]) for i, _ in self.samples)
            sizes = Counter(task for task, _ in groups)
            # P(task)=1/K; P(team size | task)=1/number_of_sizes.
            self.sample_weights = torch.tensor([
                1. / (sizes[self.episodes[i]["task_domain"]] * groups[(self.episodes[i]["task_domain"], self.episodes[i]["num_agents"])])
                for i, _ in self.samples], dtype=torch.double)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        task = row["metadata"]["task_domain"]
        row["loss_scale"] = self.task_statistics[task]["target_std"]
        row["task_id"] = torch.tensor(list(self.task_statistics).index(task), dtype=torch.long)
        return row


def collate_joint(rows):
    batch = collate_clips(rows)
    batch["loss_scale"] = torch.tensor([row["loss_scale"] for row in rows], dtype=torch.float32)
    return batch


def selection_score(metrics, task_statistics):
    if set(metrics["by_task"]) != set(task_statistics):
        raise ValueError("Validation must contain every evaluated task")
    return float(np.mean([metrics["by_task"][task]["mse"] / task_statistics[task]["target_std"]**2 for task in task_statistics]))


def train_epoch(model, loader, optimizer, accelerator):
    model.train()
    totals = torch.zeros(3, device=accelerator.device)
    core = accelerator.unwrap_model(model)
    for batch in loader:
        with accelerator.accumulate(model):
            if core.cfg.task_names:
                # z_k = (G - mu_k)/sigma_k; train f_k directly in normalized units.
                pred = model(batch, return_normalized=True)
                normalized_error = pred.float() - core.normalize_targets(batch["target"], batch)
                ids = core.task_indices(batch, pred.device)
                error = normalized_error * core.task_target_std[ids]
            else:
                error = model(batch) - batch["target"]
                normalized_error = error / batch["loss_scale"]
            loss = normalized_error.square().mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # One synchronization per epoch instead of copying CUDA metrics every batch.
        totals += torch.stack((normalized_error.detach().square().sum(), error.detach().square().sum(),
                               error.new_tensor(error.numel())))
    squared, raw_squared, count = accelerator.reduce(totals, reduction="sum").tolist()
    return dict(normalized_mse=squared/count, mse=raw_squared/count, count=int(count))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formation-dataset", required=True)
    p.add_argument("--gotogoal-dataset")
    p.add_argument("--experiment", choices=["legacy"] + list(EXPERIMENTS), default="legacy")
    p.add_argument("--num-agents", nargs="+", type=int)
    p.add_argument("--init-checkpoint", default="", help="Optional frozen-backbone best.pt; required for legacy mode; starts a new optimizer")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--expected-formation-episodes", type=int, default=None, help="Raw source count before filters; 0 disables")
    p.add_argument("--expected-gotogoal-episodes", type=int, default=None, help="Raw source count before filters; 0 disables")
    p.add_argument("--training-tasks", nargs="+", choices=TASKS,
                   help="Tasks supplying gradients; goal_to_goal performs new-task-only fine-tuning while both tasks are evaluated")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--value-head-mode", choices=("shared", "task"), default="shared")
    p.add_argument("--task-normalization", action="store_true", help="Train normalized outputs using each task's train-only mean/std")
    p.add_argument("--balance-agent-counts", action="store_true", help="Sample team sizes equally within each task")
    p.add_argument("--formation-context", action="store_true", help="Add recorded hold fraction and settled fraction to both ablations")
    p.add_argument("--discount-reference-hz", type=float, default=0., help="Gamma's reference rate; 0 keeps legacy native-step discounting")
    p.add_argument("--expected-control-hz", type=float, default=0., help="Require this native control rate in every archive; 0 accepts recorded rates")
    p.add_argument("--expected-capture-hz", type=float, default=0., help="Require this recorded image rate; 0 disables")
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
    p.add_argument("--model-id", default=ModelConfig.model_id)
    p.add_argument("--model-revision", default="main")
    p.add_argument("--clip-len", type=int, choices=(6,8), default=6)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--vl-batch-size", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=1)
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.2)
    p.add_argument("--preflight-only", action="store_true", help="Write selection counts/paths without opening tars, downloading the model, or training")
    a = p.parse_args(argv)
    a.evaluation_tasks = list(TASKS if a.experiment == "legacy" else EXPERIMENTS[a.experiment])
    if a.experiment == "legacy" and not a.init_checkpoint:
        p.error("Legacy mode requires --init-checkpoint; select --experiment for a fresh 0.5B experiment")
    if "goal_to_goal" in a.evaluation_tasks and not a.gotogoal_dataset:
        p.error("Joint experiments require --gotogoal-dataset")
    if a.num_agents is None and a.experiment != "legacy":
        a.num_agents = [3,4,5,6] if a.experiment == "joint" else [3,5]
    if a.num_agents and any(n < 1 for n in a.num_agents):
        p.error("Agent counts must be positive")
    a.navigation_condition = {"joint_without_obstacles":"without_obstacles", "joint_with_obstacles":"with_obstacles"}.get(a.experiment,"all")
    if a.training_tasks and not set(a.training_tasks) <= set(a.evaluation_tasks):
        p.error("Training tasks must belong to the selected experiment")
    a.training_tasks = [task for task in a.evaluation_tasks if task in (a.training_tasks or a.evaluation_tasks)]
    if a.value_head_mode == "task" and not a.task_normalization:
        p.error("Task heads require --task-normalization")
    if a.init_checkpoint and (a.task_normalization or a.formation_context or a.value_head_mode != "shared"):
        p.error("The new ablations start fresh; omit --init-checkpoint to keep their initialization comparable")
    if min(a.discount_reference_hz, a.expected_control_hz, a.expected_capture_hz) < 0 or a.prefetch_factor < 1:
        p.error("Rates must be nonnegative and prefetch-factor positive")
    for key, legacy in (("expected_formation_episodes",3000),("expected_gotogoal_episodes",2000)):
        if getattr(a,key) is None:
            setattr(a,key,legacy if a.experiment == "legacy" else 0)
        if getattr(a,key)<0:
            p.error("Expected episode counts must be nonnegative")
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
                              log_with="wandb" if a.wandb else None,
                              dataloader_config=DataLoaderConfiguration(non_blocking=True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_seed(a.seed)
    root, cache = Path(a.output_dir), Path(a.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"run.json").exists() and not a.resume:
        raise FileExistsError("Choose a new output directory; the initial checkpoint will not be overwritten")
    tasks = a.evaluation_tasks
    references = []
    counts = {}
    for task, source, expected in ((TASKS[0], a.formation_dataset, a.expected_formation_episodes),
                                  (TASKS[1], a.gotogoal_dataset, a.expected_gotogoal_episodes)):
        if task not in tasks:
            continue
        raw = discover_episodes(source)
        if expected and len(raw) != expected:
            raise ValueError(f"Expected {expected} raw {task} episode archives, found {len(raw)}")
        refs = select_references(raw, task, a.num_agents, a.navigation_condition)
        summaries = [json.loads(Path(summary).read_text()) for _,summary in refs]
        counts[task] = dict(raw_episodes=len(raw), selected_episodes=len(refs),
            by_num_agents=dict(Counter(row["num_agents"] for row in summaries)),
            by_noise=dict(Counter(row.get("noise","unspecified") for row in summaries)),
            timeout_or_unfinished=sum(row["termination_reason"] in {"timeout","max_steps"} for row in summaries))
        references.extend((task,path,summary) for path,summary in refs)
    selection = dict(experiment=a.experiment, num_agents=a.num_agents,
        navigation_condition=a.navigation_condition, tasks=counts,
        selected_archives=[str(path) for _,path,_ in references])
    if accelerator.is_main_process:
        (root/"selection_report.json").write_text(json.dumps(selection,indent=2)+"\n")
    accelerator.print(json.dumps({key:value for key,value in selection.items() if key!="selected_archives"},indent=2))
    if a.preflight_only:
        return
    initial = torch.load(a.init_checkpoint, map_location="cpu", weights_only=True) if a.init_checkpoint else None
    if initial:
        cfg = ModelConfig(**initial["model_config"])
        if cfg.backbone_mode != "frozen":
            raise ValueError("This experiment requires a frozen-backbone supervised checkpoint")
        if initial["run"]["args"]["return_source"] != "control" or a.gamma != initial["run"]["args"]["gamma"]:
            raise ValueError("Keep the original full-control return definition and discount")
    elif a.resume:
        cfg = ModelConfig(**torch.load(a.resume,map_location="cpu",weights_only=True)["model_config"])
    else:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(a.model_id, revision=a.model_revision)
        cfg = ModelConfig(model_id=a.model_id, model_revision=hf_config._commit_hash or a.model_revision,
            vl_hidden_dim=hf_config.text_config.hidden_size, backbone_mode="frozen",
            vl_dtype={"no":"float32","fp16":"float16","bf16":"bfloat16"}[a.mixed_precision],
            clip_len=a.clip_len, vl_batch_size=a.vl_batch_size, image_size=a.image_size,
            d_model=a.d_model, gnn_layers=a.gnn_layers, temporal_layers=a.temporal_layers,
            attention_heads=a.attention_heads, dropout=a.dropout,
            task_names=tuple(tasks) if a.task_normalization else (),
            value_head_mode=a.value_head_mode, formation_context=a.formation_context)
    if a.resume and (cfg.value_head_mode != a.value_head_mode or bool(cfg.task_names) != a.task_normalization
                     or cfg.formation_context != a.formation_context):
        raise ValueError("Resume requires the original head, normalization and context flags")
    a.dataset = a.formation_dataset
    a.additional_datasets = [a.gotogoal_dataset] if "goal_to_goal" in tasks else []
    a.dataset_revision = "main"
    a.return_source = "control"
    a.clip_len = cfg.clip_len
    a.backbone_mode = "frozen"
    a.policy_stage = "all"
    with accelerator.main_process_first():
        episodes = []
        excluded = Counter()
        with ThreadPoolExecutor(max_workers=max(1, a.num_workers)) as pool:
            indexed = list(pool.map(lambda ref: read_episode(ref[1],ref[2],cache/"index"), references))
        for (task,path,summary),ep in zip(references,indexed):
            if ep["task_domain"] != task:
                raise ValueError(f"Wrong task domain in {path}")
            if a.num_agents and ep["num_agents"] not in a.num_agents:
                raise ValueError(f"Archive/summary agent count mismatch in {path}")
            if task == "goal_to_goal" and a.navigation_condition != "all" and ep["navigation_condition"] != a.navigation_condition:
                raise ValueError(f"Archive/summary obstacle condition mismatch in {path}")
            for key, expected in (("control_hz", a.expected_control_hz), ("capture_hz", a.expected_capture_hz)):
                if expected and not np.isclose(ep[key], expected):
                    raise ValueError(f"{path}: recorded {key}={ep[key]}, expected {expected}; recollect rather than relabel timestamps")
            if cfg.formation_context and not ep["has_formation_context"]:
                raise ValueError(f"Missing recorded formation hold/settling state in {path}")
            if ep["complete"]:
                episodes.append(prepare_episode(ep,a.gamma,"control",a.image_hz,a.discount_reference_hz))
            else:
                excluded[task] += 1
    if initial:
        partitions = joint_splits(episodes,initial["run"],a.seed,a.validation_fraction,a.test_fraction)
    else:
        # Separate task RNGs keep formation holdouts identical in all three experiments.
        partitions = {name:[] for name in ("train","validation","test")}
        for task in tasks:
            task_splits = split_episodes([ep for ep in episodes if ep["task_domain"]==task],
                                        a.seed,a.validation_fraction,a.test_fraction)
            for name in partitions:
                partitions[name].extend(task_splits[name])
    splits = training_partitions(partitions,a.training_tasks)
    for name,refs in splits.items():
        expected_tasks = set(a.training_tasks) if name=="train" else set(tasks)
        if {ep["task_domain"] for ep in refs} != expected_tasks:
            raise ValueError(f"{name} must contain complete episodes from {expected_tasks}")
    if a.num_agents:
        for task in a.training_tasks:
            present = {ep["num_agents"] for ep in splits["train"] if ep["task_domain"] == task}
            if set(a.num_agents) - present:
                raise ValueError(f"{task} has no complete training episodes for some requested agent counts")
    task_stats = {}
    for task in tasks:
        task_training = [ep for ep in splits["train"] if ep["task_domain"] == task]
        if not task_training:
            if initial is None:
                raise ValueError("Evaluation-only tasks require their pretrained training statistics")
            # For formation evaluation in navigation-only mode, reuse the
            # original training baseline; no formation frames affect this fit.
            task_stats[task] = {key:initial["model_state"][key].item() for key in ("target_mean","target_std")}
            continue
        stats = training_statistics(task_training,a.clip_stride,include_terminal=False)
        task_stats[task] = dict(target_mean=stats["target_mean"].item(),target_std=stats["target_std"].item())
    signature = hashlib.sha256(json.dumps(dict(
        init_sha256=hashlib.sha256(Path(a.init_checkpoint).read_bytes()).hexdigest() if initial else None,
        model_config=asdict(cfg), experiment=a.experiment, num_agents=a.num_agents,
        navigation_condition=a.navigation_condition, gamma=a.gamma,
        image_hz=a.image_hz,clip_stride=a.clip_stride,seed=a.seed,training_tasks=a.training_tasks,
        discount_reference_hz=a.discount_reference_hz,
        balance_agent_counts=a.balance_agent_counts,
        splits={name:[ep["identity"] for ep in refs] for name,refs in splits.items()}),sort_keys=True).encode()).hexdigest()
    used = {name:refs for name,refs in splits.items() if name!="test" or not a.skip_test}
    feature_dir = cache/"features"
    missing = [ep for refs in used.values() for ep in refs if not feature_cache_path(ep,cfg,feature_dir).exists()]
    accelerator.wait_for_everyone()
    # Every rank computes its own disjoint episode shard, including during cache creation.
    assigned = missing[accelerator.process_index::accelerator.num_processes]
    if assigned:
        encoder = SmolVLMBackbone(cfg,accelerator.device)
        cache_features(assigned,encoder,cfg,feature_dir,num_workers=a.num_workers)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    datasets = {name:JointClipDataset(refs,cfg.clip_len,a.clip_stride,feature_dir,cfg,
                    include_terminal=False,task_statistics=task_stats,
                    balance_agent_counts=a.balance_agent_counts) for name,refs in used.items()}
    generator = torch.Generator()
    sampler = (WeightedRandomSampler(datasets["train"].sample_weights,len(datasets["train"]),replacement=True,generator=generator)
               if len(a.training_tasks)>1 or a.balance_agent_counts else None)
    train_loader = DataLoader(datasets["train"],batch_size=a.batch_size,sampler=sampler,
                             shuffle=sampler is None,generator=generator,
                             num_workers=a.num_workers,collate_fn=collate_joint,
                             pin_memory=accelerator.device.type=="cuda",
                             **(dict(persistent_workers=True,prefetch_factor=a.prefetch_factor) if a.num_workers else {}))
    # Cache creation must not change the initial critic weights across experiments.
    set_seed(a.seed)
    model = MultimodalValueModel(cfg,accelerator.device,load_backbone=False)
    joint_stats = training_statistics(splits["train"],a.clip_stride,include_terminal=False)
    if initial:
        model.load_state_dict(initial["model_state"],strict=True)
        adapt_normalizers(model,joint_stats)
    else:
        for key,value in joint_stats.items():
            getattr(model,key).copy_(value)
        if cfg.task_names:
            for i,task in enumerate(cfg.task_names):
                model.task_target_mean[i] = task_stats[task]["target_mean"]
                model.task_target_std[i] = task_stats[task]["target_std"]
    initial_model_state = {key:value.clone() for key,value in model.checkpoint_state().items()}
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=2)
    start, best, stale = 0, float("inf"), 0
    if a.resume:
        checkpoint = torch.load(a.resume,map_location="cpu",weights_only=True)
        accepted_signatures = {signature}
        if a.experiment == "legacy" and a.num_agents is None and initial:
            # Preserve continuation of checkpoints created before experiment filters existed.
            legacy_payload = dict(init_sha256=hashlib.sha256(Path(a.init_checkpoint).read_bytes()).hexdigest(),
                gamma=a.gamma, image_hz=a.image_hz, clip_stride=a.clip_stride, seed=a.seed,
                training_tasks=a.training_tasks,
                splits={name:[ep["identity"] for ep in refs] for name,refs in splits.items()})
            accepted_signatures.add(hashlib.sha256(json.dumps(legacy_payload,sort_keys=True).encode()).hexdigest())
        saved_cfg = asdict(ModelConfig(**checkpoint["model_config"]))
        if checkpoint["run"]["signature"] not in accepted_signatures or saved_cfg != asdict(cfg):
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
        initial_checkpoint=str(Path(a.init_checkpoint).resolve()) if initial else None, task_statistics=task_stats,
        initialization="warm_start" if initial else "pretrained SmolVLM-500M with fresh critic layers",
        selection=selection,
        return_definition="Full future native control-step rewards, gamma once per control step; incoming/past reward excluded",
        control_hz_by_task={task:sorted({ep['control_hz'] for ep in episodes if ep['task_domain']==task}) for task in tasks},
        capture_hz_by_task={task:sorted({ep['capture_hz'] for ep in episodes if ep['task_domain']==task}) for task in tasks},
        control_gamma_by_task={task:sorted({ep['control_gamma'] for ep in episodes if ep['task_domain']==task}) for task in tasks},
        normalization="Per-task z=(G-mu_k)/sigma_k; forward returns sigma_k*z+mu_k" if cfg.task_names else "Legacy shared affine output; task-scaled loss",
        throughput=dict(world_size=accelerator.num_processes,per_gpu_batch_size=a.batch_size,
                        effective_batch_size=a.batch_size*a.grad_accum_steps*accelerator.num_processes,
                        workers_per_rank=a.num_workers),
        policy_value="Conditional expected recorded return under each task's dataset behavior-policy mixture",
        splits={name:[ep["path"] for ep in refs] for name,refs in splits.items()},
        pretrained_splits=initial["run"]["splits"] if initial else {},
        excluded_timeout_or_unfinished_by_task=dict(excluded),
        training_tasks=a.training_tasks,
        training_episode_counts=dict(Counter(ep["task_domain"] for ep in splits["train"])),
        sampling=("Equal probability per task at clip level" if len(a.training_tasks)>1 else "Only selected task; shuffled without replacement")+"; per-task train-only return standard deviation scales the loss",
        agent_sampling="Equal team-size probability within each task" if a.balance_agent_counts else "Natural within-task clip frequency",
        normalizer_transfer="Affine conversion preserving warm-start predictions" if initial else "Training-only statistics; fresh critic layers",
        formation_split="Original checkpoint holdouts preserved" if initial else "Task-local seeded scenario splits; consistent formation holdouts across experiments")
    accelerator.print(f"Training tasks: {a.training_tasks}; gradient-bearing episodes: {run['training_episode_counts']}; evaluated tasks: {tasks}")
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
        with accelerator.autocast():
            metrics=evaluate_test(accelerator.unwrap_model(model),datasets[name],a.batch_size,
                accelerator.device,root,label,task_baselines=baselines,
                accelerator=accelerator,num_workers=a.num_workers)
        if accelerator.is_main_process:
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
        train_seconds=time.monotonic()-started
        train_metrics.update(seconds=train_seconds, clips_per_second=train_metrics["count"]/train_seconds)
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
                "; ".join(f"{task} MAE={validation['by_task'][task]['mae']:.4f}" for task in tasks))
            if a.wandb:
                log={"train/normalized_mse":train_metrics["normalized_mse"],"validation/macro_task_normalized_mse":score,
                     "train/clips_per_second":train_metrics["clips_per_second"],"train/seconds":train_seconds}
                log.update({f"validation/{task}/{key}":validation["by_task"][task][key]
                            for task in tasks for key in ("mse","mae","r2")})
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
        unwrapped.load_state_dict(initial_model_state,strict=True)
        before_test=evaluate("test","before_test")
        if accelerator.is_main_process:
            comparison={task:dict(before=before_test["by_task"][task],after=after_test["by_task"][task]) for task in tasks}
            (root/"task_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    accelerator.end_training()


if __name__=="__main__":
    main()
