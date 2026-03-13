import argparse
import os
import functools
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

try:
    from accelerate.utils import DistributedDataParallelKwargs
except Exception:
    DistributedDataParallelKwargs = None

try:
    from accelerate import DataLoaderConfiguration
except Exception:
    DataLoaderConfiguration = None

try:
    from accelerate.utils import FullyShardedDataParallelPlugin
except Exception:
    try:
        from accelerate.utils import FSDPPlugin as FullyShardedDataParallelPlugin
    except Exception:
        FullyShardedDataParallelPlugin = None

try:
    from torch.distributed.fsdp import CPUOffload
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
except Exception:
    CPUOffload = None
    size_based_auto_wrap_policy = None

from model import ModelConfig, MultimodalValueModel
from data_loading import webdataset_loader


def parse_args():
    p = argparse.ArgumentParser()

    # Data / webdataset
    p.add_argument("--train_shards", type=str, required=True, help="WebDataset shard pattern for training")
    p.add_argument("--val_shards", type=str, default="", help="Optional WebDataset shard pattern for validation")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--samples_per_epoch", type=int, default=10000)
    p.add_argument("--text_mode", type=str, default="raw", choices=["raw", "emb"])
    p.add_argument(
        "--text_prompt_template",
        type=str,
        default="You are a critic model. You are given video frames, robot state sequences, and a graph adjacency per timestep for a robot team. Assess how good or bad the current policy is at the task and respond with a single scalar judgment.",
    )

    # Sequence building
    p.add_argument("--clip_len", type=int, default=20)
    p.add_argument("--clip_stride", type=int, default=1)
    p.add_argument("--robot_source", type=str, default="obs", choices=["obs", "state"])
    p.add_argument("--reward_reduce", type=str, default="mean", choices=["mean", "sum", "first"])
    p.add_argument("--done_reduce", type=str, default="all", choices=["any", "all", "mean", "sum", "first"])
    p.add_argument("--preprocess_in_loader", default=True, action="store_true", help="Use VLM image processor in dataloader")
    p.add_argument("--debug_save_video", action="store_true", help="Save one video sample for debugging")
    p.add_argument("--debug_out_dir", type=str, default="debug_samples")
    p.add_argument("--debug_decode_text", action="store_true", help="Decode VLM token predictions for debugging")
    p.add_argument("--debug_decode_every", type=int, default=200, help="Decode/print debug text every N steps")
    p.add_argument("--debug_decode_max_tokens", type=int, default=32, help="Max decoded tokens for debug text")

    # Value targets
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--return_mode", type=str, default="td", choices=["td", "nstep"])
    p.add_argument(
        "--return_horizon",
        type=str,
        default="trajectory",
        choices=["clip", "trajectory"],
        help="For nstep-style targets: discounted return over clip or from clip start to terminal state in episode.",
    )
    p.add_argument("--n_step", type=int, default=50)
    p.add_argument("--loss_type", type=str, default="td_contrastive", choices=["td", "contrastive", "td_contrastive"])
    p.add_argument(
        "--contrastive_objective",
        type=str,
        default="infonce",
        choices=["point_to_set", "infonce"],
        help="Contrastive objective used when loss_type includes contrastive.",
    )
    p.add_argument("--contrastive_margin", type=float, default=0.0)
    p.add_argument("--infonce_temperature", type=float, default=0.1, help="Temperature for InfoNCE contrastive loss.")
    p.add_argument(
        "--infonce_topk_pos",
        type=int,
        default=0,
        help="Top-K high-reward samples used as positive pool for InfoNCE (0 => B//2).",
    )
    p.add_argument(
        "--contrastive_multidepth",
        action="store_true",
        help="Apply contrastive supervision on multiple hidden depths using attention-max pooled features.",
    )
    p.add_argument(
        "--contrastive_depth_offsets",
        type=str,
        default="0",
        help="Comma-separated hidden-layer offsets from final layer for multidepth contrastive (e.g., '0,4,8').",
    )
    p.add_argument(
        "--contrastive_depth_weights",
        type=str,
        default="",
        help="Optional comma-separated weights for multidepth losses. Empty => uniform.",
    )
    p.add_argument("--lambda_td", type=float, default=1.0, help="Weight for TD loss when using td_contrastive")
    p.add_argument("--lambda_c", type=float, default=1.0, help="Weight for contrastive loss when using td_contrastive")
    p.add_argument(
        "--lambda_value_c",
        type=float,
        default=1.0,
        help="Weight for value-head pairwise contrastive loss when loss_type=contrastive.",
    )
    p.add_argument(
        "--shard_aware_batching",
        action="store_true",
        help="Build batches with at most one mini-trajectory per shard on each process.",
    )
    p.add_argument(
        "--shard_batch_max_queue_per_shard",
        type=int,
        default=16,
        help="Max queued samples per shard while waiting to form unique-shard batches.",
    )

    # Accelerate
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--fsdp", action="store_true", help="Use FSDP to shard model parameters across GPUs")
    p.add_argument("--fsdp_min_num_params", type=int, default=1_000_000, help="Auto-wrap threshold for FSDP")
    p.add_argument("--fsdp_cpu_offload", action="store_true", help="Offload FSDP parameters to CPU when not in use")
    p.add_argument("--fsdp_use_orig_params", action="store_true", help="Use FSDP use_orig_params to allow mixed requires_grad")
    p.add_argument("--ddp_find_unused_parameters", action="store_true", help="Set DDP find_unused_parameters=True")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing on VLM backbone")
    p.add_argument("--disable_vl_cache", action="store_true", help="Disable VLM KV cache during training for lower memory")
    p.add_argument("--allow_tf32", action="store_true", help="Enable TF32 matmul/cuDNN kernels on Ampere+ GPUs")
    p.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection for debugging in-place/backward errors.",
    )

    # VLM backbone
    p.add_argument("--vl_backend", type=str, default="deepseek_vl", choices=["deepseek_vl", "deepseek_vl2", "llava_video"])
    p.add_argument("--vl_model_name", type=str, default="deepseek-community/deepseek-vl-1.3b-base")
    p.add_argument(
        "--vl_model_preset",
        type=str,
        default="custom",
        choices=["custom", "llava_next_video_7b", "llava_onevision_0p5b"],
        help="Convenience preset for known LLaVA video-capable checkpoints.",
    )
    p.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--vl_max_text_len", type=int, default=256)
    p.add_argument("--freeze_vl", action="store_true")
    p.add_argument(
        "--value_pooling",
        type=str,
        default="last_token_logits",
        choices=["last_token_logits", "hidden_mean"],
        help="Feature pooling strategy for value head; last_token_logits is more memory efficient.",
    )
    p.add_argument("--vl_logits_to_keep", type=int, default=0, help="If supported, keep logits for only last K tokens")
    p.add_argument("--obs_summary_tokens", type=int, default=2, help="Number of temporal graph summary tokens injected as <obs>.")

    # PEFT / LoRA
    p.add_argument("--peft", type=str, default="none", choices=["none", "lora", "qlora"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    # Video
    p.add_argument("--video_channels", type=int, default=3)
    p.add_argument("--video_height", type=int, default=224)
    p.add_argument("--video_width", type=int, default=224)
    p.add_argument("--video_frames", type=int, default=100)
    p.add_argument("--video_preprocessed", action="store_true")
    p.add_argument("--video_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--video_std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    # Robots / graph
    p.add_argument("--num_robots", type=int, default=5)
    p.add_argument("--robot_obs_dim", type=int, default=40)

    # Text
    p.add_argument("--text_dim", type=int, default=512)

    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--temporal_layers", type=int, default=2)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--temporal_dropout", type=float, default=0.1)
    p.add_argument("--gnn_layers", type=int, default=2)
    p.add_argument("--fusion_hidden", type=int, default=512)
    p.add_argument("--use_moe", action="store_true")
    p.add_argument("--moe_experts", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=2)

    # Train
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="checkpoints_new")
    p.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint .pt to resume training")
    p.add_argument("--load_model_only", action="store_true", help="Load only model weights from checkpoint")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging via Accelerate trackers")
    p.add_argument("--wandb_project", type=str, default="ma-vlcm", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team (optional)")
    p.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional)")
    p.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags (optional)")

    return p.parse_args()


def _parse_int_csv(value: str):
    if value is None:
        return []
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    out = []
    for x in items:
        out.append(int(x))
    return out


def _parse_float_csv(value: str):
    if value is None or str(value).strip() == "":
        return []
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    out = []
    for x in items:
        out.append(float(x))
    return out


def _resolve_contrastive_depth_args(args):
    offsets = _parse_int_csv(args.contrastive_depth_offsets)
    if not offsets:
        offsets = [0]
    offsets = [max(0, int(o)) for o in offsets]
    args.contrastive_depth_offsets_list = offsets

    weights = _parse_float_csv(args.contrastive_depth_weights)
    if weights and len(weights) != len(offsets):
        raise ValueError(
            "contrastive_depth_weights length must match contrastive_depth_offsets. "
            f"Got {len(weights)} vs {len(offsets)}."
        )
    args.contrastive_depth_weights_list = weights


def build_model(args, device):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        vl_max_text_len=args.vl_max_text_len,
        freeze_vl=args.freeze_vl,
        quantization_config=getattr(args, "quantization_config", None),
        video_channels=args.video_channels,
        video_height=args.video_height,
        video_width=args.video_width,
        video_frames=args.video_frames,
        video_preprocessed=args.video_preprocessed,
        video_mean=tuple(args.video_mean),
        video_std=tuple(args.video_std),
        num_robots=args.num_robots,
        robot_obs_dim=args.robot_obs_dim,
        text_dim=args.text_dim,
        d_model=args.d_model,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        temporal_dropout=args.temporal_dropout,
        gnn_layers=args.gnn_layers,
        fusion_hidden=args.fusion_hidden,
        use_moe=args.use_moe,
        moe_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        debug_save_video=args.debug_save_video,
        value_pooling=args.value_pooling,
        logits_to_keep=args.vl_logits_to_keep,
        obs_summary_tokens=args.obs_summary_tokens,
        contrastive_multidepth=args.contrastive_multidepth,
        contrastive_depth_offsets=tuple(getattr(args, "contrastive_depth_offsets_list", [0])),
    )
    return MultimodalValueModel(cfg, device=device)


def _configure_memory_optimizations(model, args):
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.disable_vl_cache and hasattr(model.backbone.model, "config") and hasattr(model.backbone.model.config, "use_cache"):
        model.backbone.model.config.use_cache = False

    if args.gradient_checkpointing:
        fn = getattr(model.backbone.model, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                fn()
        if hasattr(model.backbone.model, "enable_input_require_grads"):
            try:
                model.backbone.model.enable_input_require_grads()
            except Exception:
                pass


def _resolve_vl_model_preset(args):
    if args.vl_model_preset == "llava_next_video_7b":
        args.vl_backend = "llava_video"
        args.vl_model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    elif args.vl_model_preset == "llava_onevision_0p5b":
        args.vl_backend = "llava_video"
        args.vl_model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


def _parse_lora_targets(args):
    if args.lora_target_modules:
        return [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_peft(model, args):
    if args.peft == "none":
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError("PEFT requested but 'peft' is not installed. `pip install peft`.") from e

    for p in model.backbone.model.parameters():
        p.requires_grad = False

    if args.peft == "qlora":
        model.backbone.model = prepare_model_for_kbit_training(model.backbone.model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_targets(args),
        task_type="CAUSAL_LM",
    )
    model.backbone.model = get_peft_model(model.backbone.model, lora_cfg)
    return model


def _count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _save_debug_video(batch, args, accelerator, tag="train"):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.debug_out_dir, exist_ok=True)
    if "video" not in batch:
        return
    out_dir = os.path.join(args.debug_out_dir, f"{tag}_sample")
    os.makedirs(out_dir, exist_ok=True)


def _load_checkpoint_state(model, ckpt_state, args, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    try:
        unwrapped.load_state_dict(ckpt_state)
        return
    except RuntimeError as e:
        if args.peft not in {"lora", "qlora"}:
            raise
        accelerator.print(f"Strict checkpoint load failed under PEFT ({e}). Retrying with filtered non-strict load.")

    # bitsandbytes quantized modules can expose version-dependent metadata keys.
    quant_meta_tokens = ("absmax", "quant_map", "quant_state", "bitsandbytes__")
    filtered_state = {k: v for k, v in ckpt_state.items() if not any(tok in k for tok in quant_meta_tokens)}

    incompatible = unwrapped.load_state_dict(filtered_state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    accelerator.print(
        "Non-strict PEFT load complete: "
        f"skipped_quant_meta={len(ckpt_state) - len(filtered_state)} "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )
    if missing:
        accelerator.print(f"First missing keys: {missing[:10]}")
    if unexpected:
        accelerator.print(f"First unexpected keys: {unexpected[:10]}")


def _contrastive_pairwise_loss(scores, rewards, margin=0.0):
    diff_r = rewards[:, None] - rewards[None, :]
    diff_s = scores[:, None] - scores[None, :]
    sign = diff_r.sign()
    mask = sign.ne(0)
    if mask.sum() == 0:
        return scores.sum() * 0.0
    signed = diff_s * sign
    if margin != 0.0:
        signed = signed - margin
    loss = F.softplus(-signed)
    return loss[mask].mean()


def _contrastive_point_to_set_loss(embeddings, rewards, margin=0.0):
    if embeddings.ndim != 2:
        embeddings = embeddings.view(embeddings.shape[0], -1)
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    bsz = embeddings.shape[0]
    k = max(1, bsz // 2)
    topk_idx = torch.topk(rewards.view(-1), k=k, largest=True).indices
    desirable_set = embeddings[topk_idx]

    dists = torch.cdist(embeddings, desirable_set, p=2)
    scores = -dists.min(dim=1).values
    return _contrastive_pairwise_loss(scores, rewards.view(-1), margin=margin)


def _contrastive_infonce_loss(embeddings, rewards, temperature=0.1, topk_pos=0):
    if embeddings.ndim != 2:
        embeddings = embeddings.view(embeddings.shape[0], -1)
    bsz = int(embeddings.shape[0])
    if bsz < 3:
        return embeddings.sum() * 0.0

    if temperature <= 0:
        raise ValueError("infonce_temperature must be > 0")

    rewards = rewards.view(-1)
    k = int(topk_pos) if int(topk_pos) > 0 else max(1, bsz // 2)
    k = min(max(1, k), bsz)

    # Reward-defined desirable pool.
    topk_idx = torch.topk(rewards, k=k, largest=True).indices
    desirable_mask = torch.zeros(bsz, dtype=torch.bool, device=embeddings.device)
    desirable_mask[topk_idx] = True

    z = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(z, z.t()) / float(temperature)  # [B, B]

    losses = []
    for i in range(bsz):
        valid_neg_mask = torch.ones(bsz, dtype=torch.bool, device=embeddings.device)
        valid_neg_mask[i] = False  # remove self from denominator
        if valid_neg_mask.sum() == 0:
            continue

        pos_mask = desirable_mask.clone()
        pos_mask[i] = False  # avoid trivial self-positive
        if pos_mask.sum() == 0:
            continue

        # Select one positive from desirable pool: most similar desirable sample.
        pos_candidates = pos_mask.nonzero(as_tuple=False).view(-1)
        pos_sim = sim[i, pos_candidates]
        pos_idx = pos_candidates[pos_sim.argmax()]

        denom_idx = valid_neg_mask.nonzero(as_tuple=False).view(-1)
        logits = sim[i, denom_idx]
        target = (denom_idx == pos_idx).nonzero(as_tuple=False)
        if target.numel() == 0:
            continue
        target = target.view(-1)[0]
        losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))

    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def _compute_contrastive_loss(embeddings, rewards, args):
    if args.contrastive_objective == "infonce":
        return _contrastive_infonce_loss(
            embeddings,
            rewards,
            temperature=args.infonce_temperature,
            topk_pos=args.infonce_topk_pos,
        )
    return _contrastive_point_to_set_loss(embeddings, rewards, margin=args.contrastive_margin)


def _compute_multidepth_contrastive_loss(main_embeddings, depth_embeddings, rewards, args):
    feats = []
    if depth_embeddings:
        feats.extend(depth_embeddings)
    else:
        feats.append(main_embeddings)

    if args.contrastive_depth_weights_list:
        weights = args.contrastive_depth_weights_list
    else:
        weights = [1.0 for _ in feats]

    total_w = 0.0
    total_loss = None
    for w, feat in zip(weights, feats):
        w = float(w)
        if w <= 0.0:
            continue
        l = _compute_contrastive_loss(feat, rewards, args)
        total_w += w
        total_loss = (w * l) if total_loss is None else (total_loss + w * l)

    if total_loss is None or total_w <= 0:
        return main_embeddings.sum() * 0.0
    return total_loss / total_w


def run_epoch(model, loader, optimizer, accelerator, log_every, gamma, args, train=True, global_step=0):
    model.train() if train else model.eval()
    raw_model = accelerator.unwrap_model(model)

    loss_fn = nn.MSELoss()
    total_loss = 0.0
    step = 0

    for batch in loader:
        step += 1

        def _move_inputs(inputs):
            moved = {}
            for k, v in inputs.items():
                moved[k] = v.to(accelerator.device) if torch.is_tensor(v) else v
            return moved

        inputs = _move_inputs(batch["inputs"])
        robot_obs = batch["robot_obs"].to(accelerator.device)
        adj = batch["adj"].to(accelerator.device)
        reward = batch["reward"].to(accelerator.device)
        done = batch["done"].to(accelerator.device).float()
        use_td = args.loss_type in {"td", "td_contrastive"} and args.return_mode == "td"
        if use_td:
            next_inputs = _move_inputs(batch["next_inputs"])
            next_robot_obs = batch["next_robot_obs"].to(accelerator.device)
            next_adj = batch["next_adj"].to(accelerator.device)
        use_nstep_td = args.loss_type in {"td", "td_contrastive"} and args.return_mode == "nstep"
        if use_nstep_td:
            nstep_inputs = _move_inputs(batch["td_nstep_inputs"])
            nstep_robot_obs = batch["td_nstep_robot_obs"].to(accelerator.device)
            nstep_adj = batch["td_nstep_adj"].to(accelerator.device)
            nstep_returns = batch["td_nstep_return"].to(accelerator.device)
            nstep_done = batch["td_nstep_done"].to(accelerator.device).float()

        if train and args.debug_save_video and not getattr(run_epoch, "_debug_saved", False):
            _save_debug_video(batch, args, accelerator, tag="train")
            run_epoch._debug_saved = True

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                next_pred = None
                nstep_bootstrap_pred = None
                if args.loss_type in {"td", "td_contrastive"} and args.return_mode == "td":
                    with torch.no_grad():
                        next_pred = raw_model(next_inputs, next_robot_obs, next_adj)
                elif args.loss_type in {"td", "td_contrastive"} and args.return_mode == "nstep":
                    with torch.no_grad():
                        nstep_bootstrap_pred = raw_model(nstep_inputs, nstep_robot_obs, nstep_adj)

                should_decode = (
                    args.debug_decode_text
                    and accelerator.is_main_process
                    and (args.debug_decode_every > 0)
                    and (step % args.debug_decode_every == 0)
                )
                model_out = model(
                    inputs,
                    robot_obs,
                    adj,
                    return_debug=should_decode,
                    return_features=args.loss_type in {"contrastive", "td_contrastive"},
                    debug_max_tokens=args.debug_decode_max_tokens,
                )
                if isinstance(model_out, dict):
                    pred = model_out["value"]
                    vlm_feature = model_out.get("vlm_feature")
                    vlm_multidepth_features = model_out.get("vlm_multidepth_features")
                    debug_text = model_out.get("debug_text")
                else:
                    pred = model_out
                    vlm_feature = None
                    vlm_multidepth_features = None
                    debug_text = None
                contrastive_targets = (
                    batch["returns"].to(accelerator.device)
                    if args.return_mode == "nstep" and "returns" in batch
                    else reward
                )

                if args.loss_type == "contrastive":
                    if vlm_feature is None:
                        raise RuntimeError("Contrastive loss requires model features; got None from model forward.")
                    vlm_contrastive_loss = _compute_multidepth_contrastive_loss(
                        vlm_feature, vlm_multidepth_features, contrastive_targets.view(-1), args
                    )
                    value_contrastive_loss = _contrastive_pairwise_loss(
                        pred.view(-1), contrastive_targets.view(-1), margin=args.contrastive_margin
                    )
                    loss = args.lambda_c * vlm_contrastive_loss + args.lambda_value_c * value_contrastive_loss
                else:
                    if args.return_mode == "td":
                        target = reward + gamma * (1.0 - done) * next_pred
                    elif args.return_mode == "nstep":
                        gamma_n = gamma ** int(args.n_step)
                        target = nstep_returns + gamma_n * (1.0 - nstep_done) * nstep_bootstrap_pred
                    else:
                        target = batch["returns"].to(accelerator.device)
                    td_loss = loss_fn(pred, target)
                    if args.loss_type == "td_contrastive":
                        if vlm_feature is None:
                            raise RuntimeError("td_contrastive loss requires model features; got None from model forward.")
                        contrastive_loss = _compute_multidepth_contrastive_loss(
                            vlm_feature, vlm_multidepth_features, contrastive_targets.view(-1), args
                        )
                        loss = args.lambda_td * td_loss + args.lambda_c * contrastive_loss
                    else:
                        loss = td_loss

                if train:
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        if debug_text:
            text0 = debug_text[0].replace("\n", " ").strip()
            phase = "train" if train else "val"
            accelerator.print(f"{phase} step={step} debug_decode[0]: {text0}")

        total_loss += loss.item()
        if log_every > 0 and step % log_every == 0:
            avg = total_loss / step
            phase = "train" if train else "val"
            accelerator.print(f"{phase} step={step} loss={avg:.4f}")
            if args.wandb:
                metrics = {f"{phase}/loss": avg}
                if train:
                    metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(metrics, step=global_step + step)
                else:
                    accelerator.log(metrics, step=global_step)

    avg_loss = total_loss / max(step, 1)
    if args.wandb:
        phase = "train" if train else "val"
        metrics = {f"{phase}/epoch_loss": avg_loss}
        accelerator.log(metrics, step=(global_step + step if train else global_step))

    return avg_loss, (global_step + step if train else global_step)


def main():
    args = parse_args()
    _resolve_contrastive_depth_args(args)
    _resolve_vl_model_preset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.preprocess_in_loader:
        args.video_preprocessed = True

    if args.peft == "qlora" and args.fsdp:
        raise RuntimeError("FSDP + QLoRA is not supported. Use DDP (no --fsdp) or LoRA.")

    if args.peft == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("QLoRA requested but bitsandbytes/transformers are not available.") from e
        if args.vl_dtype == "float16":
            compute_dtype = torch.float16
        elif args.vl_dtype == "float32":
            compute_dtype = torch.float32
        else:
            compute_dtype = torch.bfloat16
        args.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        args.quantization_config = None

    fsdp_plugin = None
    if args.fsdp:
        if FullyShardedDataParallelPlugin is None:
            raise RuntimeError("FSDP requested but accelerate FSDP plugin is unavailable.")
        fsdp_kwargs = {}
        use_orig_params = args.fsdp_use_orig_params or (args.peft != "none")
        if size_based_auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = functools.partial(size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params)
        else:
            try:
                params = inspect.signature(FullyShardedDataParallelPlugin).parameters
                if "min_num_params" in params:
                    fsdp_kwargs["min_num_params"] = args.fsdp_min_num_params
            except Exception:
                pass
        try:
            params = inspect.signature(FullyShardedDataParallelPlugin).parameters
            if "use_orig_params" in params:
                fsdp_kwargs["use_orig_params"] = use_orig_params
        except Exception:
            pass
        if args.fsdp_cpu_offload:
            if CPUOffload is None:
                raise RuntimeError("FSDP CPU offload requested but torch.distributed.fsdp is unavailable.")
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

    ddp_kwargs = None
    if not args.fsdp and DistributedDataParallelKwargs is not None:
        # Full fine-tuning can leave some backbone params outside the loss graph
        # (for example modality-specific heads or untied LM/output weights), which
        # causes standard DDP reduction errors. LoRA/QLoRA keep those frozen.
        find_unused = bool(args.ddp_find_unused_parameters or args.peft == "none")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)

    accelerator_kwargs = dict(
        mixed_precision=args.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else [],
    )
    # IterableDataset + variable-shaped samples across ranks can break with dispatch_batches=True.
    if DataLoaderConfiguration is not None:
        accelerator_kwargs["dataloader_config"] = DataLoaderConfiguration(dispatch_batches=False, split_batches=False)
    else:
        accel_params = inspect.signature(Accelerator).parameters
        if "dispatch_batches" in accel_params:
            accelerator_kwargs["dispatch_batches"] = False
        if "split_batches" in accel_params:
            accelerator_kwargs["split_batches"] = False
    if args.wandb:
        accelerator_kwargs["log_with"] = ["wandb"]
    accelerator = Accelerator(**accelerator_kwargs)

    if args.detect_anomaly:
        accelerator.print("Autograd anomaly detection enabled. Expect slower training and a more specific stack trace.")

    if args.wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_kwargs = {}
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        if tags:
            wandb_kwargs["tags"] = tags
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": wandb_kwargs})

    model = build_model(args, device=accelerator.device)
    model = _apply_peft(model, args)
    _configure_memory_optimizations(model, args)
    total_params, trainable_params = _count_parameters(model)
    accelerator.print(
        f"parameters total={total_params:,} trainable={trainable_params:,} "
        f"peft={args.peft}"
    )
    if not args.fsdp and DistributedDataParallelKwargs is not None:
        effective_find_unused = bool(args.ddp_find_unused_parameters or args.peft == "none")
        accelerator.print(
            "DDP find_unused_parameters="
            f"{effective_find_unused} (auto-enabled for full fine-tuning when --peft none)"
        )

    if args.fsdp:
        if args.mixed_precision == "bf16":
            target_dtype = torch.bfloat16
        elif args.mixed_precision == "fp16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
        model = model.to(dtype=target_dtype)
        for p in model.parameters():
            if p.dtype != target_dtype:
                p.data = p.data.to(dtype=target_dtype)
        for b in model.buffers():
            if torch.is_floating_point(b) and b.dtype != target_dtype:
                b.data = b.data.to(dtype=target_dtype)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = webdataset_loader(args, args.train_shards, args.batch_size, args.num_workers)
    val_loader = webdataset_loader(args, args.val_shards, args.batch_size, args.num_workers) if args.val_shards else None

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    start_epoch = 1
    global_step = 0

    if args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        _load_checkpoint_state(model, ckpt["model"], args, accelerator)
        if not args.load_model_only:
            if "optimizer" not in ckpt:
                raise KeyError("Checkpoint does not contain optimizer state. Use --load_model_only to ignore this.")
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))

        accelerator.print(
            f"Resumed from {ckpt_path} (start_epoch={start_epoch}, global_step={global_step}, "
            f"load_model_only={args.load_model_only})"
        )
        accelerator.wait_for_everyone()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, accelerator, args.log_every, args.gamma, args, train=True, global_step=global_step
        )
        val_loss = None
        if val_loader is not None:
            val_loss, global_step = run_epoch(
                model, val_loader, optimizer, accelerator, args.log_every, args.gamma, args, train=False, global_step=global_step
            )

        accelerator.print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss if val_loss is not None else 'n/a'}")
        if args.wandb:
            epoch_metrics = {"epoch": epoch, "train/epoch_loss": train_loss}
            if val_loss is not None:
                epoch_metrics["val/epoch_loss"] = val_loss
            accelerator.log(epoch_metrics, step=global_step)

        if accelerator.is_main_process:
            ckpt = {
                "model": accelerator.get_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt"))
        accelerator.wait_for_everyone()

    if args.wandb and hasattr(accelerator, "end_training"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
