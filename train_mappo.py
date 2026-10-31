"""Train the native Python MAPPO actor with a centralized MA-VLCM critic."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch

from model import ModelConfig, SmolVLMBackbone
from python_mappo import (MARL_ROOT, R_Actor, PythonValueCritic, RolloutCollector,
                          actor_config, evaluate_actor, ppo_update)

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--critic", choices=("vlcm", "state-temporal", "mlp"), default="vlcm")
    p.add_argument("--critic-checkpoint", type=Path,
                   help="Supervised best.pt (resets scalar) or Python critic.pt/directory (preserves full critic)")
    freeze = p.add_mutually_exclusive_group()
    freeze.add_argument("--freeze-critic-representation", action="store_true",
                   help="Freeze visual/GAT/fusion/time/temporal/context layers; train only the entire value_head")
    freeze.add_argument("--freeze-critic", action="store_true",
                   help="Freeze the complete pretrained VLCM critic; train the actor only")
    p.add_argument("--frozen-isaac-critic", action="store_true",
                   help="Preserve the original supervised Isaac critic and its return units; requires --freeze-critic")
    p.add_argument("--value-calibration", type=Path,
                   help="Fixed per-formation calibration JSON for the frozen supervised Isaac critic")
    actor_init = p.add_mutually_exclusive_group()
    actor_init.add_argument("--actor-checkpoint", type=Path, help="Native actor.pt or checkpoint directory; omit to train from scratch")
    actor_init.add_argument("--actor-from-scratch", action="store_true",
                            help="Explicitly require a fresh actor; incompatible with --actor-checkpoint")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/python_mappo_vlcm")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-env-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=300)
    p.add_argument("--agent-counts", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--formation-spacing", type=float, default=1.)
    p.add_argument("--formation-tolerance", type=float, default=.05)
    p.add_argument("--initial-position-range", type=float, default=1.5)
    p.add_argument("--min-start-separation", type=float, default=.55)
    p.add_argument("--observation-scale", type=float, default=3.)
    p.add_argument("--formation-hold-steps", type=int, default=20)
    p.add_argument("--env-max-steps", type=int, default=300)
    p.add_argument("--control-dt", type=float, default=.05)
    p.add_argument("--camera-stride", type=int, default=4)
    p.add_argument("--view-size-m", type=float, default=6.096)
    p.add_argument("--tile-floor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--vl-batch-size", type=int, default=4)
    p.add_argument("--clip-len", type=int, default=6)
    p.add_argument("--graph-radius", type=float, default=1.)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--gae-lambda", type=float, default=.95)
    p.add_argument("--actor-lr", type=float, default=5e-4)
    p.add_argument("--critic-lr", type=float, default=5e-4)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--critic-epochs", type=int, default=5)
    p.add_argument("--num-minibatches", type=int, default=1)
    p.add_argument("--critic-batch-size", type=int, default=64)
    p.add_argument("--critic-warmup-updates", type=int, default=5)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--entropy-coef", type=float, default=.0001)
    p.add_argument("--value-scale", type=float, default=50.)
    p.add_argument("--max-grad-norm", type=float, default=10.)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layer-n", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=2,
                   help="Keep this many recent periodic checkpoint directories, plus best; 0 keeps all")
    p.add_argument("--eval-interval", type=int, default=10, help="0 disables; evaluates actor without critic")
    p.add_argument("--eval-at-start", action=argparse.BooleanOptionalAction, default=True,
                   help="Record actor-only validation and preserve its checkpoint before training")
    p.add_argument("--eval-episodes", type=int, default=4, help="per formation, on fixed held-out seeds")
    p.add_argument("--eval-seed", type=int, default=9300001)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-project", default="formation-mappo-vlcm")
    p.add_argument("--wandb-entity")
    p.add_argument("--run-name")
    return p


def validate(p, a):
    for name in ("num_envs", "num_env_steps", "rollout_steps", "env_max_steps", "camera_stride",
                 "clip_len", "image_size", "vl_batch_size", "ppo_epochs", "critic_epochs",
                 "num_minibatches", "critic_batch_size", "torch_threads", "save_interval",
                 "eval_episodes", "hidden_size", "formation_hold_steps"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.num_env_steps % a.num_envs:
        p.error("--num-env-steps must be divisible by --num-envs")
    if not a.agent_counts or any(n not in (3, 4, 5, 6) for n in a.agent_counts):
        p.error("--agent-counts must contain polygon sizes 3, 4, 5, 6")
    if len(set(a.agent_counts)) != len(a.agent_counts):
        p.error("--agent-counts must not repeat")
    if not 0 < a.gamma <= 1 or not 0 <= a.gae_lambda <= 1:
        p.error("Invalid gamma or GAE lambda")
    for name in ("control_dt", "view_size_m", "graph_radius", "value_scale", "actor_lr", "critic_lr",
                 "clip", "max_grad_norm", "formation_spacing", "formation_tolerance", "observation_scale",
                 "initial_position_range"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if a.eval_interval < 0 or a.keep_checkpoints < 0 or a.critic_warmup_updates < 0 or a.entropy_coef < 0 or a.min_start_separation < 0:
        p.error("Intervals, warmup, entropy coefficient and minimum separation cannot be negative")
    if a.critic == "mlp" and a.critic_checkpoint:
        p.error("The native MLP baseline cannot load a VLCM checkpoint")
    if a.freeze_critic_representation and (a.critic == "mlp" or not a.critic_checkpoint):
        p.error("--freeze-critic-representation requires a pretrained graph/temporal critic checkpoint")
    if a.freeze_critic and (a.critic != "vlcm" or not a.critic_checkpoint):
        p.error("--freeze-critic requires a pretrained VLCM critic checkpoint")
    if a.frozen_isaac_critic and not a.freeze_critic:
        p.error("--frozen-isaac-critic requires --freeze-critic")
    if a.value_calibration and not a.frozen_isaac_critic:
        p.error("--value-calibration requires --frozen-isaac-critic and --freeze-critic")
    if a.value_calibration and not a.value_calibration.is_file():
        p.error(f"Calibration file does not exist: {a.value_calibration}")
    if a.critic_checkpoint:
        if a.critic_checkpoint.is_dir():
            a.critic_checkpoint = a.critic_checkpoint / "critic.pt"
        if not a.critic_checkpoint.is_file():
            p.error(f"Critic checkpoint does not exist: {a.critic_checkpoint}")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("Output directory is nonempty; choose a new run directory (warm-start is not an exact resume)")


def jsonable(args):
    return {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_value_calibration(path, critic_path, counts, gamma):
    calibration = json.loads(path.read_text())
    if calibration.get("critic_sha256") != file_hash(critic_path):
        raise ValueError("Calibration was produced for a different Isaac critic checkpoint")
    if not math.isclose(calibration.get("gamma", float("nan")), gamma, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Calibration gamma differs from training gamma")
    names = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon"}
    coefficients = {}
    for count in counts:
        row = calibration.get("formations", {}).get(names[count], {})
        if row.get("status") != "exploratory_fit":
            raise ValueError(f"Missing completed calibration fit for {names[count]}")
        slope, intercept = float(row["slope"]), float(row["intercept"])
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise ValueError(f"Non-finite calibration coefficients for {names[count]}")
        coefficients[str(count)] = [slope, intercept]
    return coefficients


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, args, steps, update):
    path.mkdir(parents=True, exist_ok=True)
    atomic_save({k: v.detach().cpu() for k, v in actor.state_dict().items()}, path / "actor.pt")
    atomic_save(dict(format="python_mappo_vlcm_v1", critic_kind=args.critic,
                     model_config=asdict(critic.cfg), model_state={k: v.detach().cpu() for k, v in critic.state_dict().items()},
                     args=jsonable(args), steps=steps), path / "critic.pt")
    atomic_save(dict(actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict() if critic_optimizer is not None else None,
                     torch_rng=torch.get_rng_state(), steps=steps, update=update), path / "optimizers.pt")
    metadata = dict(algorithm="mappo", pattern="polygon", agent_counts=args.agent_counts,
                    max_num_agents=6, observation_dim=25, state_dim=52,
                    initialized_from=str(args.actor_checkpoint) if args.actor_checkpoint else None,
                    formation_spacing=args.formation_spacing, formation_tolerance=args.formation_tolerance,
                    observation_scale=args.observation_scale, total_environment_steps=steps, update=update,
                    critic_kind=args.critic, hidden_size=args.hidden_size, layer_N=args.layer_n,
                    freeze_critic_representation=args.freeze_critic_representation,
                    freeze_critic=args.freeze_critic,
                    deployment="Use actor.pt with native local observations; critic/images are training only",
                    checkpoint_semantics="Warm-start artifacts; environment/NumPy/CUDA RNG are not restored for exact resume")
    (path / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def prune_checkpoints(root, keep):
    """Bound periodic snapshots inside this new run; preserve best and other files."""
    if not keep:
        return
    snapshots = sorted((path for path in root.iterdir()
                        if path.is_dir() and not path.is_symlink()
                        and re.fullmatch(r"step_[0-9]+", path.name)),
                       key=lambda path: int(path.name.split("_")[1]))
    for path in snapshots[:-keep]:
        shutil.rmtree(path)


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    validate(p, a)
    torch.set_num_threads(a.torch_threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          "cpu" if a.device == "auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    source = None
    cfg = ModelConfig(image_size=a.image_size, clip_len=a.clip_len, vl_batch_size=a.vl_batch_size, dropout=0.)
    if a.critic_checkpoint:
        source = torch.load(a.critic_checkpoint, map_location="cpu", weights_only=True)
        python_source = source.get("format") == "python_mappo_vlcm_v1"
        if a.value_calibration and python_source:
            raise ValueError("Calibrated Isaac mode requires the original supervised checkpoint")
        if a.freeze_critic and not python_source and not a.frozen_isaac_critic:
            raise ValueError("--freeze-critic requires a trained Python critic.pt, not an Isaac supervised checkpoint")
        if a.frozen_isaac_critic and python_source and not source.get("args", {}).get("frozen_isaac_critic", False):
            raise ValueError("The supplied Python critic has Python value units, not original Isaac value units")
        if python_source and source.get("args", {}).get("frozen_isaac_critic", False) and not a.frozen_isaac_critic:
            raise ValueError("This checkpoint uses Isaac value units; specify --frozen-isaac-critic")
        cfg = ModelConfig(**source["model_config"])
        if cfg.backbone_mode != "frozen":
            raise ValueError("Cached PPO features require a frozen backbone checkpoint; LoRA/full are not supported here")
        if cfg.robot_obs_dim != 10:
            raise ValueError("Expected the MA-VLCM ten-feature robot schema")
        cfg.image_size, cfg.clip_len, cfg.vl_batch_size, cfg.dropout = a.image_size, a.clip_len, a.vl_batch_size, 0.
    if a.value_calibration:
        a.value_calibration_coefficients = load_value_calibration(a.value_calibration, a.critic_checkpoint,
                                                                   a.agent_counts, a.gamma)
    collector = RolloutCollector(a)
    env = collector.envs[0]
    actor = R_Actor(actor_config(a.hidden_size, a.layer_n), env.observation_spaces[0], env.action_spaces[0], device)
    if a.actor_checkpoint:
        actor_path = a.actor_checkpoint / "actor.pt" if a.actor_checkpoint.is_dir() else a.actor_checkpoint
        actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True), strict=True)
    critic = PythonValueCritic(cfg, a, env, device, source)
    backbone = SmolVLMBackbone(cfg, device) if a.critic == "vlcm" else None
    # Dropout is disabled while gradients still update the head: rollout values
    # and PPO's old/new value comparison must use the same deterministic map.
    actor.eval()
    critic.eval()
    if backbone is not None:
        backbone.eval()
    ao = torch.optim.Adam(actor.parameters(), lr=a.actor_lr, eps=1e-5)
    critic_parameters = [v for v in critic.parameters() if v.requires_grad]
    co = torch.optim.Adam(critic_parameters, lr=a.critic_lr, eps=1e-5) if critic_parameters else None
    critic_trainable = [name for name, parameter in critic.named_parameters() if parameter.requires_grad]
    python_source = source is not None and source.get("format") == "python_mappo_vlcm_v1"
    print(f"Critic checkpoint: {a.critic_checkpoint}; actor: {'scratch' if a.actor_checkpoint is None else a.actor_checkpoint}", flush=True)
    print(f"Trainable critic parameters: {', '.join(critic_trainable)}", flush=True)
    if a.frozen_isaac_critic:
        if a.value_calibration:
            print("Frozen Isaac values use fixed per-formation calibration fitted on earlier Python rollouts; "
                  "calibration may not generalize to the new actor.", flush=True)
        else:
            print("WARNING: frozen critic values are in Isaac reward units; PPO uses unmodified Python rewards. "
                  "GAE mixes different units and may be biased; no reward calibration is applied.", flush=True)
    contract = dict(args=jsonable(a), model_config=asdict(cfg), device=str(device),
                    environment_sha256=file_hash(MARL_ROOT / "RL_bound_overwatch/envs/FormationControlEnv.py"),
                    texture_sha256=file_hash(MARL_ROOT / "RL_bound_overwatch/envs/assets/background_image_real.png"),
                    actor_sha256=file_hash(actor_path) if a.actor_checkpoint else None,
                    actor_initialization="random_from_scratch" if a.actor_checkpoint is None else "pretrained_actor_warm_start",
                    source_critic_sha256=file_hash(a.critic_checkpoint) if a.critic_checkpoint else None,
                    reward_units="Unmodified Python team rewards, mean over active robots only",
                    critic_value_units=("calibrated estimate of Python discounted reward" if a.value_calibration else
                                        "Isaac discounted clipped reward" if a.frozen_isaac_critic else "Python discounted reward"),
                    critic_reward_units_mismatch=bool(a.frozen_isaac_critic and not a.value_calibration),
                    value_calibration_sha256=file_hash(a.value_calibration) if a.value_calibration else None,
                    target=f"GAE: dt={a.control_dt}, gamma={a.gamma}, lambda={a.gae_lambda}; true-terminal zero, timeout final-state bootstrap",
                    image_schedule="Latest available causal image every camera_stride controls; image age explicit",
                    vision="Real SmolVLM frozen features" if backbone is not None else "No image encoder in this baseline",
                    critic_trainable_parameters=critic_trainable,
                    critic_trainable_parameter_count=sum(parameter.numel() for parameter in critic.parameters() if parameter.requires_grad),
                    critic_adaptation=("fully_frozen" if a.freeze_critic else "value_head_only" if a.freeze_critic_representation else "representation_and_value_head"),
                    source_critic_steps=source.get("steps") if python_source else None,
                    transfer=("Original supervised Isaac critic frozen without Python adaptation" if a.frozen_isaac_critic else
                              "Full Python critic warm-start; head, context and normalization preserved" if python_source else
                              "Representation initialization only; Python scalar regressor reset; not zero-shot value inference"))
    (a.output_dir / "run.json").write_text(json.dumps(contract, indent=2) + "\n")
    preview = env.render(size=a.image_size, view_size_m=a.view_size_m, tile_floor=a.tile_floor)
    from PIL import Image
    Image.fromarray(preview).save(a.output_dir / "training_view.png")
    run = None
    if a.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, mode=a.wandb_mode,
                         name=a.run_name or f"{a.critic}-seed{a.seed}", config=contract,
                         dir=str(a.output_dir.resolve()))
        run.define_metric("environment_steps")
        run.define_metric("*", step_metric="environment_steps")
        run.log({"render/training_view": wandb.Image(preview), "environment_steps": 0})
        (a.output_dir / "wandb_run.json").write_text(json.dumps(dict(id=run.id, url=run.url,
                                                                      project=run.project, entity=run.entity), indent=2) + "\n")
    start = time.monotonic()
    steps, update, best_score = 0, 0, (-math.inf, -math.inf)
    metrics_path = a.output_dir / "metrics.jsonl"
    try:
        if a.eval_interval and a.eval_at_start:
            baseline = evaluate_actor(actor, a, device)
            baseline.update(environment_steps=0, update=0)
            best_score = (baseline["eval/success"], float(np.mean([baseline[f"eval/{n}_agents/reward"] for n in a.agent_counts])))
            save_checkpoint(a.output_dir / "best", actor, critic, ao, co, a, 0, 0)
            (a.output_dir / "initial_evaluation.json").write_text(json.dumps(baseline, indent=2) + "\n")
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(baseline) + "\n")
            print(json.dumps(baseline), flush=True)
            if run is not None:
                run.log(baseline)
        while steps < a.num_env_steps:
            update += 1
            rollout_steps = min(a.rollout_steps, (a.num_env_steps - steps) // a.num_envs)
            collection_start = time.monotonic()
            rollout = collector.collect(actor, device, rollout_steps)
            collection_seconds = time.monotonic() - collection_start
            encode_start = time.monotonic()
            images = collector.encode_cameras(backbone)
            encode_seconds = time.monotonic() - encode_start
            train_start = time.monotonic()
            metrics = ppo_update(actor, critic, ao, co, rollout, a, device, update)
            steps += rollout_steps * a.num_envs
            metrics.update(environment_steps=steps, update=update, **{
                "timing/collection_seconds": collection_seconds,
                "timing/vision_seconds": encode_seconds,
                "timing/ppo_seconds": time.monotonic() - train_start,
                "timing/environment_steps_per_second": steps / (time.monotonic()-start),
                "vision/encoded_images": images, "train/actor_lr": ao.param_groups[0]["lr"],
                "train/critic_lr": co.param_groups[0]["lr"] if co is not None else 0.0})
            if a.eval_interval and (update % a.eval_interval == 0 or steps == a.num_env_steps):
                evaluation = evaluate_actor(actor, a, device)
                metrics.update(evaluation)
                score = (evaluation["eval/success"], np.mean([evaluation[f"eval/{n}_agents/reward"] for n in a.agent_counts]))
                if score > best_score:
                    best_score = score
                    save_checkpoint(a.output_dir / "best", actor, critic, ao, co, a, steps, update)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Non-finite training metrics")
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if run is not None:
                run.log(metrics)
            if update % a.save_interval == 0 or steps == a.num_env_steps:
                save_checkpoint(a.output_dir / "checkpoints" / f"step_{steps:09d}", actor, critic, ao, co, a, steps, update)
                prune_checkpoints(a.output_dir / "checkpoints", a.keep_checkpoints)
        if run is not None:
            run.summary["finished_environment_steps"] = steps
            run.summary["actor_deploys_without_critic"] = True
    finally:
        if run is not None:
            run.finish()
    print(f"Saved native actor.pt and Python VLCM critic under {a.output_dir / 'checkpoints'}", flush=True)


if __name__ == "__main__":
    main()
