import argparse
import os
import io
import json
import pathlib
import glob
import functools
import inspect
from collections import deque

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
import numpy as np

# Extracted Shelf Map (Partial - to be updated with full map)
SHELF_MAP = {
    1: (1, 1),
    2: (2, 1),
    3: (7, 1),
    4: (8, 1),
    5: (1, 2),
    6: (2, 2),
    7: (7, 2),
    8: (8, 2),
    9: (1, 3),
    10: (2, 3),
    11: (7, 3),
    12: (8, 3),
    13: (1, 4),
    14: (2, 4),
    15: (7, 4),
    16: (8, 4),
    17: (1, 5),
    18: (2, 5),
    19: (7, 5),
    20: (8, 5),
    21: (1, 6),
    22: (2, 6),
    23: (7, 6),
    24: (8, 6),
    25: (1, 7),
    26: (2, 7),
    27: (7, 7),
    28: (8, 7),
    29: (1, 8),
    30: (2, 8),
    31: (7, 8),
    32: (8, 8),
}

# Maximum possible distance in the warehouse grid for normalization.
# Grid spans (1,1) to (8,8), so diagonal = sqrt(7^2 + 7^2) ≈ 9.9
_WAREHOUSE_MAX_DIST = float(np.sqrt(7**2 + 7**2))  # ~9.899


try:
    from accelerate.utils import DistributedDataParallelKwargs
except Exception:
    DistributedDataParallelKwargs = None
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

import webdataset as wds
from model import ModelConfig, MultimodalValueModel


def parse_args():
    p = argparse.ArgumentParser()

    # Data / webdataset
    p.add_argument(
        "--train_shards",
        type=str,
        default="/scratch/aparame/Research/VLCM_Data_Collection/data_scratch",
        help="WebDataset shard pattern for training",
    )
    p.add_argument(
        "--val_shards",
        type=str,
        default="",
        help="Optional WebDataset shard pattern for validation",
    )
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--samples_per_epoch",
        type=int,
        default=50000,
        help="Approximate samples per epoch for LR schedule (970 shards × ~50 clips)",
    )
    p.add_argument("--text_mode", type=str, default="raw", choices=["raw", "emb"])
    p.add_argument("--text_prompt_template", type=str, default=None)
    p.add_argument(
        "--dataset_type",
        type=str,
        default="rware",
        choices=["default", "rware", "offroad"],
    )
    p.add_argument("--rware_config", type=str, default="tiny-2ag-hard")
    p.add_argument(
        "--offroad_shards",
        type=str,
        default="",
        help="WebDataset shard directory for OFFROAD data (enables multi-dataset training)",
    )
    p.add_argument(
        "--offroad_num_robots",
        type=int,
        default=5,
        help="Number of agents in the OFFROAD environment",
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of shards to hold out for validation",
    )

    # Sequence building
    p.add_argument("--clip_len", type=int, default=10)
    p.add_argument("--clip_stride", type=int, default=1)
    p.add_argument("--robot_source", type=str, default="obs", choices=["obs", "state"])
    p.add_argument(
        "--reward_reduce", type=str, default="mean", choices=["mean", "sum", "first"]
    )
    p.add_argument(
        "--done_reduce",
        type=str,
        default="any",
        choices=["any", "all", "mean", "sum", "first"],
    )
    p.add_argument(
        "--preprocess_in_loader",
        default=True,
        action="store_true",
        help="Use VLM image processor in dataloader",
    )
    p.add_argument(
        "--debug_save_video",
        action="store_true",
        help="Save one video sample for debugging",
    )
    p.add_argument("--debug_out_dir", type=str, default="debug_samples")

    # Value targets
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--return_mode", type=str, default="nstep", choices=["nstep"])
    p.add_argument("--n_step", type=int, default=50)
    p.add_argument(
        "--loss_type",
        type=str,
        default="contrastive_mse",
        choices=["mse", "contrastive", "contrastive_mse"],
    )
    p.add_argument("--contrastive_margin", type=float, default=0.0)
    p.add_argument(
        "--mse_loss_weight",
        type=float,
        default=1.0,
        help="Weight for MSE loss component in contrastive_mse mode",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable)",
    )

    # Accelerate
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "int8", "fp16", "bf16"],
    )
    p.add_argument(
        "--fsdp",
        action="store_true",
        help="Use FSDP to shard model parameters across GPUs",
    )
    p.add_argument(
        "--fsdp_min_num_params",
        type=int,
        default=1_000_000,
        help="Auto-wrap threshold for FSDP",
    )
    p.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        help="Offload FSDP parameters to CPU when not in use",
    )
    p.add_argument(
        "--fsdp_use_orig_params",
        action="store_true",
        help="Use FSDP use_orig_params to allow mixed requires_grad",
    )
    p.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Set DDP find_unused_parameters=True",
    )
    p.add_argument(
        "--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps"
    )

    # DeepSeek VLM backbone
    p.add_argument(
        "--vl_backend",
        type=str,
        default="llava_video",
        choices=["deepseek_vl", "deepseek_vl2", "llava_video", "llava_onevision"],
    )
    p.add_argument(
        "--vl_model_name", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    )
    p.add_argument(
        "--vl_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--vl_max_text_len", type=int, default=8192)
    p.add_argument("--freeze_vl", action="store_true")
    p.add_argument(
        "--freeze_vision_tower",
        action="store_true",
        help="Also freeze the vision tower when freeze_vl is set",
    )
    p.add_argument(
        "--vision_lr",
        type=float,
        default=1e-5,
        help="Separate learning rate for the vision tower parameters",
    )

    # PEFT / LoRA
    p.add_argument(
        "--peft", type=str, default="none", choices=["none", "lora", "qlora"]
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")
    p.add_argument(
        "--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"]
    )

    # Video
    p.add_argument("--video_channels", type=int, default=3)
    p.add_argument("--video_frames", type=int, default=8)
    p.add_argument("--video_preprocessed", action="store_true")
    p.add_argument("--video_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--video_std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    # Robots / graph
    p.add_argument("--num_robots", type=int, default=2)
    p.add_argument("--robot_obs_dim", type=int, default=6)

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
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="checkpoints_rware")

    # Optimization
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--resize_width", type=int, default=672)
    p.add_argument("--resize_height", type=int, default=336)

    return p.parse_args()


def build_model(args, device):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        vl_max_text_len=args.vl_max_text_len,
        freeze_vl=args.freeze_vl,
        freeze_vision_tower=getattr(args, "freeze_vision_tower", True),
        quantization_config=getattr(args, "quantization_config", None),
        video_channels=args.video_channels,
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
    )
    return MultimodalValueModel(cfg, device=device)


def _parse_lora_targets(args):
    if args.lora_target_modules:
        return [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    # Default targets for LLaMA-style blocks used by LLaVA
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_peft(model, args):
    if args.peft == "none":
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise RuntimeError(
            "PEFT requested but 'peft' is not installed. `pip install peft`."
        ) from e

    # Freeze backbone weights; keep custom heads trainable.
    for p in model.backbone.model.parameters():
        p.requires_grad = False

    # Enable gradient checkpointing for ALL PEFT modes (saves ~30-40% activation memory)
    if hasattr(model.backbone.model, "gradient_checkpointing_enable"):
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[PEFT] Gradient checkpointing ENABLED")

    if args.peft == "qlora":
        model.backbone.model = prepare_model_for_kbit_training(
            model.backbone.model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=_parse_lora_targets(args),
        task_type="CAUSAL_LM",
    )
    model.backbone.model = get_peft_model(model.backbone.model, lora_cfg)

    # Apply LoRA to vision tower if user wants it trainable (instead of full fine-tuning).
    freeze_vision_tower = getattr(args, "freeze_vision_tower", True)
    if not freeze_vision_tower:
        # Locate vision tower after PEFT wrapping
        base = model.backbone.model
        vt = None
        for attr_path in [
            ("model", "model", "vision_tower"),
            ("base_model", "model", "vision_tower"),
            ("model", "vision_tower"),
            ("vision_tower",),
        ]:
            candidate = base
            for attr in attr_path:
                candidate = getattr(candidate, attr, None)
                if candidate is None:
                    break
            if candidate is not None:
                vt = candidate
                break

        if vt is not None:
            # Apply a separate LoRA config to the vision tower
            # Target the attention projection layers in the vision encoder
            vision_lora_targets = []
            vt_param_names = {n for n, _ in vt.named_parameters()}
            # Detect common attention projection names in vision encoders
            for candidate_name in [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "qkv",
                "proj",
            ]:
                if any(candidate_name in n for n in vt_param_names):
                    vision_lora_targets.append(candidate_name)

            if vision_lora_targets:
                from peft import LoraConfig as VisionLoraConfig

                vision_lora_cfg = VisionLoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    target_modules=vision_lora_targets,
                )
                try:
                    vt = get_peft_model(vt, vision_lora_cfg)
                    # Replace the vision tower in the model
                    for attr_path in [
                        ("model", "model"),
                        ("base_model", "model"),
                        ("model",),
                    ]:
                        parent = base
                        for attr in attr_path:
                            parent = getattr(parent, attr, None)
                            if parent is None:
                                break
                        if parent is not None and hasattr(parent, "vision_tower"):
                            parent.vision_tower = vt
                            break

                    trainable = sum(
                        p.numel() for p in vt.parameters() if p.requires_grad
                    )
                    total = sum(p.numel() for p in vt.parameters())
                    print(
                        f"[PEFT] Vision tower LoRA applied: "
                        f"{trainable:,} trainable / {total:,} total params, "
                        f"targets={vision_lora_targets}"
                    )
                except Exception as e:
                    print(
                        f"[PEFT] WARNING: Vision tower LoRA failed ({e}); "
                        f"falling back to full fine-tuning."
                    )
                    for p in vt.parameters():
                        p.requires_grad = True
                    print(
                        f"[PEFT] Vision tower UNFROZEN (full fine-tune fallback, "
                        f"{sum(p.numel() for p in vt.parameters()):,} params)"
                    )
            else:
                # No recognized attention layers; fall back to full unfreeze
                for p in vt.parameters():
                    p.requires_grad = True
                print(
                    f"[PEFT] Vision tower UNFROZEN (no LoRA targets found, "
                    f"full fine-tune, "
                    f"{sum(p.numel() for p in vt.parameters()):,} params)"
                )
        else:
            print(
                "[PEFT] WARNING: Could not locate vision_tower "
                "after LoRA wrapping; it remains frozen."
            )

    return model


def _as_numpy(x):
    if isinstance(x, bytes):
        import numpy as np

        try:
            return np.load(io.BytesIO(x), allow_pickle=True)
        except Exception:
            import torch

            return torch.load(io.BytesIO(x), map_location="cpu")
    return x


def _edge_index_to_adj(edge_index, num_nodes):
    edge_index = _as_numpy(edge_index)
    if (
        hasattr(edge_index, "shape")
        and len(edge_index.shape) == 2
        and edge_index.shape[0] == num_nodes
        and edge_index.shape[1] == num_nodes
    ):
        return torch.from_numpy(edge_index).float()
    if hasattr(edge_index, "shape") and edge_index.shape[0] == 2:
        src = edge_index[0]
        dst = edge_index[1]
        mask = (src >= 0) & (dst >= 0)
        src = src[mask]
        dst = dst[mask]
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if len(src) > 0:
            adj[src, dst] = 1.0
        return adj
    raise ValueError("edge_index has unexpected shape")


def _reduce_value(x, reduce_mode):
    if reduce_mode == "sum":
        return x.sum()
    if reduce_mode == "first":
        return x.flatten()[0]
    return x.mean()


def _reduce_done(x, reduce_mode):
    if reduce_mode == "all":
        return (x > 0).all()
    if reduce_mode == "sum":
        return x.sum() > 0
    if reduce_mode == "first":
        return x.flatten()[0] > 0
    if reduce_mode == "mean":
        return x.float().mean() > 0.5
    return (x > 0).any()


def _save_debug_video(batch, args, accelerator, tag="train"):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.debug_out_dir, exist_ok=True)
    if "video" not in batch:
        return
    video = batch["video"]
    out_dir = os.path.join(args.debug_out_dir, f"{tag}_sample")
    os.makedirs(out_dir, exist_ok=True)


def _reward_from_frame(sample, reduce_mode):
    arr = _as_numpy(sample["rewards.npy"])
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    t = torch.tensor(arr, dtype=torch.float32)
    return _reduce_value(t, reduce_mode)


def _done_from_frame(sample, reduce_mode):
    arr = _as_numpy(sample["dones.npy"])
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    t = torch.tensor(arr, dtype=torch.float32)
    return _reduce_done(t, reduce_mode)


def _parse_rware_state(state_json, num_robots=None, robot_obs_dim=8):
    # state_json: dict from state.json
    # returns: torch.Tensor [num_robots, robot_obs_dim]
    # Encoding: [x, y, dx, dy, carrying, 0, 0, 0]

    agents = state_json.get("agents", [])
    # Sort by ID just in case
    agents = sorted(agents, key=lambda x: x.get("id", 0))

    # If num_robots is specified, ensure we have that many rows
    if num_robots is None:
        num_robots = len(agents)

    obs = torch.zeros((num_robots, robot_obs_dim), dtype=torch.float32)

    # Direction mapping (matching standard trig unit vectors)
    # NORTH (+Y), SOUTH (-Y), EAST (+X), WEST (-X)
    dir_map = {
        "NORTH": (0.0, 1.0),
        "SOUTH": (0.0, -1.0),
        "EAST": (1.0, 0.0),
        "WEST": (-1.0, 0.0),
    }

    # Action mapping
    action_map = {
        "NOOP": 0,
        "FORWARD": 1,
        "LEFT": 2,
        "RIGHT": 3,
        "TOGGLE_LOAD": 4,
    }

    for i, ag in enumerate(agents):
        if i >= num_robots:
            break

        # Pos
        pos = ag.get("pos", [0, 0])
        obs[i, 0] = float(pos[0])
        obs[i, 1] = float(pos[1])

        # Dir
        d_str = ag.get("dir", "EAST")
        dx, dy = dir_map.get(d_str, (1.0, 0.0))
        obs[i, 2] = dx
        obs[i, 3] = dy

        # Carrying
        carrying = ag.get("carrying")
        obs[i, 4] = 1.0 if carrying is not None else 0.0

        # Action
        a_str = ag.get("action", "NOOP")
        obs[i, 5] = float(action_map.get(a_str, 0))

    return obs


def _parse_offroad_state(state_json, num_robots=None, robot_obs_dim=8):
    """Parse OFFROAD state.json into a robot observation tensor.

    Encoding per agent: [x, y, cos(yaw), sin(yaw), v_cmd, w_cmd, dist_to_goal, traversability]

    Args:
        state_json: dict from state.json
        num_robots: number of robots to pad/truncate to
        robot_obs_dim: observation dimension per robot (default 8)

    Returns:
        torch.Tensor [num_robots, robot_obs_dim]
    """
    agents = state_json.get("agents", [])
    agents = sorted(agents, key=lambda x: x.get("id", 0))

    if num_robots is None:
        num_robots = len(agents)

    obs = torch.zeros((num_robots, robot_obs_dim), dtype=torch.float32)

    for i, ag in enumerate(agents):
        if i >= num_robots:
            break

        # Position
        pos = ag.get("pos", [0.0, 0.0])
        obs[i, 0] = float(pos[0])
        obs[i, 1] = float(pos[1])

        # Heading (yaw) as cos/sin for continuity
        yaw = ag.get("yaw", 0.0)
        obs[i, 2] = float(np.cos(yaw))
        obs[i, 3] = float(np.sin(yaw))

        # Velocity commands [v_cmd, w_cmd]
        vel = ag.get("vel", [0.0, 0.0])
        obs[i, 4] = float(vel[0])  # v_cmd (linear)
        obs[i, 5] = float(vel[1])  # w_cmd (angular)

        # Distance to goal
        obs[i, 6] = float(ag.get("dist_to_goal", 0.0))

        # Traversability at current position
        obs[i, 7] = float(ag.get("traversability", 0.0))

    return obs


class SequenceWebDataset(IterableDataset):
    def __init__(
        self,
        shards,
        clip_len,
        clip_stride,
        text_mode,
        robot_source,
        reward_reduce,
        done_reduce,
        vlm_processor=None,
        vl_model_name=None,
        robot_obs_dim=8,
        num_robots=1,
        max_num_robots=None,
        shuffle_shards=False,
        text_prompt_template=None,
        return_mode="nstep",
        n_step=50,
        gamma=0.99,
        keep_raw_video=False,
        include_next=False,
        vlm_max_text_len=256,
        vlm_truncation=False,
        vlm_padding="longest",
        dataset_type="default",
        rware_config="tiny-2ag-hard",
        resize_width=672,
        resize_height=336,
        vl_backend="llava_video",
    ):
        if isinstance(shards, str):
            if shards.startswith(("hf://", "http://", "https://", "pipe:")):
                print(f"Using remote dataset URL: {shards}")
            elif os.path.isdir(shards):
                print(f"Expanding shard directory: {shards}")
                # Auto-expand directory to all .tar files recursively
                expanded = sorted(
                    glob.glob(os.path.join(shards, "**", "*.tar"), recursive=True)
                )
                if not expanded:
                    print(f"Warning: No .tar files found in {shards}")
                shards = expanded
            elif "*" in shards or "?" in shards or "[" in shards:
                print(f"Expanding shard pattern: {shards}")
                expanded = sorted(glob.glob(shards, recursive=True))
                if not expanded:
                    print(f"Warning: No files matched pattern {shards}")
                shards = expanded

        if isinstance(shards, list):
            print(f"Found {len(shards)} shards.")

        self.shards = shards
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.text_mode = text_mode
        self.robot_source = robot_source
        self.reward_reduce = reward_reduce
        self.done_reduce = done_reduce
        self.vlm_processor = vlm_processor
        self.vl_model_name = vl_model_name
        self.robot_obs_dim = robot_obs_dim
        self.text_prompt_template = text_prompt_template
        self.dataset_type = dataset_type
        self.rware_config = rware_config
        self.return_mode = return_mode
        self.n_step = n_step
        self.gamma = gamma
        self.keep_raw_video = keep_raw_video
        self.include_next = include_next
        self.vlm_max_text_len = vlm_max_text_len
        self.vlm_truncation = vlm_truncation
        self.vlm_padding = vlm_padding
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.num_robots = num_robots
        self.max_num_robots = max_num_robots
        self.shuffle_shards = shuffle_shards
        self.times_cache = {}
        self.vl_backend = vl_backend

    def _custom_decoder(self, key, data):
        # We only care about images here.
        extension = key.split(".")[-1].lower()
        if extension in ["png", "jpg", "jpeg"]:
            # Check for specific keywords to identify "main" camera (overhead/image)
            # This skips "front" camera or other images not used in training.
            if "overhead" in key or "image" in key:
                # Decode and resize
                try:
                    with io.BytesIO(data) as stream:
                        img = Image.open(stream)
                        img.load()
                        img = img.convert("RGB")
                        if self.resize_width > 0 and self.resize_height > 0:
                            img = img.resize((self.resize_width, self.resize_height))
                        return img
                except Exception as e:
                    print(f"Warning: Broken image at {key}, skipping: {e}")
                    return None
            # For other images (e.g. front camera), return raw bytes.
            return data

        # Fallback for non-image data (npy, json, etc.)
        return data

    def __iter__(self):
        if wds is None:
            raise RuntimeError("webdataset is not installed.")

        if self.vlm_processor is None and self.vl_model_name is not None:
            if (
                self.vl_backend == "llava_onevision"
                or "llava-onevision" in self.vl_model_name.lower()
            ):
                from transformers import LlavaOnevisionProcessor

                try:
                    self.vlm_processor = LlavaOnevisionProcessor.from_pretrained(
                        self.vl_model_name
                    )
                    tokenizer = getattr(self.vlm_processor, "tokenizer", None)
                    if tokenizer is not None and "<obs>" not in tokenizer.get_vocab():
                        tokenizer.add_special_tokens(
                            {"additional_special_tokens": ["<obs>"]}
                        )
                except Exception as e:
                    print(f"Warning: Failed to load LLaVA-OneVision processor: {e}")
            else:
                from transformers import LlavaNextVideoProcessor

                try:
                    self.vlm_processor = LlavaNextVideoProcessor.from_pretrained(
                        self.vl_model_name
                    )
                    # Ensure special tokens
                    tokenizer = getattr(self.vlm_processor, "tokenizer", None)
                    if tokenizer is not None and "<obs>" not in tokenizer.get_vocab():
                        tokenizer.add_special_tokens(
                            {"additional_special_tokens": ["<obs>"]}
                        )
                except Exception as e:
                    print(f"Warning: Failed to load VLM processor: {e}")

        # Error handler: skip broken samples instead of crashing
        _handler = getattr(wds, "warn_and_continue", None)

        # Prefer explicit node/worker splitters for multi-GPU setups
        try:
            wds_kwargs = dict(
                shardshuffle=(1000 if self.shuffle_shards else False),
                nodesplitter=getattr(wds, "split_by_node", None),
                workersplitter=getattr(wds, "split_by_worker", None),
            )
            if _handler is not None:
                wds_kwargs["handler"] = _handler
            dataset = (
                wds.WebDataset(self.shards, **wds_kwargs)
                .repeat()
                .decode(self._custom_decoder, handler=_handler)
            )
        except TypeError:
            dataset = (
                wds.WebDataset(self.shards, shardshuffle=False)
                .repeat()
                .decode(self._custom_decoder)
            )
            if hasattr(dataset, "split_by_node"):
                dataset = dataset.split_by_node()
            if hasattr(dataset, "split_by_worker"):
                dataset = dataset.split_by_worker()

        current_ep = None
        buffer = deque()
        episode_frame_count = 0

        def _process_clip_data(clip, next_clip=None):
            raw_video = [f["image"] for f in clip]
            # Safety check for empty or invalid video
            if not raw_video:
                return None

            raw_next_video = None
            if next_clip is not None:
                raw_next_video = [f["image"] for f in next_clip]
                if not raw_next_video:  # Should not happen if logic holds, but safe
                    return None

            if self.vlm_processor is not None:

                def _proc(frames, text):
                    if not isinstance(text, str):
                        text = self.text_prompt_template
                    tokenizer = getattr(self.vlm_processor, "tokenizer", None)
                    if tokenizer is not None:
                        vocab = tokenizer.get_vocab()
                        if (
                            "<video>" in vocab
                            and "<video>" not in text
                            and "<image>" not in text
                        ):
                            text = f"<video>\n{text}"
                        if "<obs>" in vocab and "<obs>" not in text:
                            if "<video>" in text:
                                text = text.replace("<video>\n", "<video><obs>\n", 1)
                            else:
                                text = f"<obs>\n{text}"
                    try:
                        max_len = self.vlm_max_text_len
                        inputs = self.vlm_processor(
                            text=text,
                            videos=frames,
                            return_tensors="pt",
                            padding=self.vlm_padding,
                            truncation=self.vlm_truncation,
                            max_length=max_len,
                        )
                    except TypeError:
                        print("using image processor instead of video processor")
                        inputs = self.vlm_processor(
                            images=frames,
                            return_tensors="pt",
                        )
                    packed = {}
                    for k, v in dict(inputs).items():
                        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                            v = v.squeeze(0)
                        packed[k] = v
                    return packed

                text = clip[0]["text"]
                inputs = _proc(raw_video, text)
                next_inputs = None
                if next_clip is not None:
                    next_inputs = _proc(raw_next_video, text)
            else:
                raise RuntimeError("Dataloader proprocessor not set.")

            robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
            adj = torch.stack([f["adj"] for f in clip], dim=0)
            if next_clip is not None:
                next_robot_obs = torch.stack([f["robot_obs"] for f in next_clip], dim=0)
                next_adj = torch.stack([f["adj"] for f in next_clip], dim=0)

            reward = clip[-1]["reward"]
            done = clip[-1]["done"]

            # Calculate discounted return (n-step TD style)
            returns = 0.0
            for frame in reversed(clip):
                returns = (
                    float(frame["reward"])
                    + self.gamma * (1.0 - float(frame["done"])) * returns
                )
            returns = torch.tensor(returns, dtype=torch.float32)

            out = {
                "robot_obs": robot_obs,
                "adj": adj,
                "reward": reward.view(1),
                "returns": returns.view(1),
                "done": done.view(1),
            }
            out["inputs"] = inputs
            if next_clip is not None:
                out["next_inputs"] = next_inputs
                out["next_robot_obs"] = next_robot_obs
                out["next_adj"] = next_adj

            return out

        for sample in dataset:
            key = sample.get("__key__", "")
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")

            if "_" in key:
                # Extract trajectory ID by removing the last part (e.g., _step0000)
                parts = key.split("_")
                ep_id = "_".join(parts[:-1]) if "step" in parts[-1] else parts[0]
            else:
                ep_id = key

            if current_ep is None:
                current_ep = ep_id

            # Reset buffer on new episode
            if ep_id != current_ep:
                buffer = deque()
                current_ep = ep_id
                episode_frame_count = 0

            # Image loading
            if "image.png" in sample:
                image = sample["image.png"]
            elif "overhead.png" in sample:
                image = sample["overhead.png"]
            else:
                # print(f"Warning: No image found for key {key}. Skipping.")
                continue

            # Skip broken images (decoded as None)
            if image is None:
                continue

            # Robot Obs and Adj
            if self.dataset_type == "rware":
                # RWARE specific parsing NEW LOGIC
                state_data = {}
                state_json_present = "state.json" in sample

                if state_json_present:
                    state_data = sample["state.json"]
                    if isinstance(state_data, bytes):
                        state_data = json.loads(state_data)
                    robot_obs = _parse_rware_state(
                        state_data,
                        num_robots=self.max_num_robots,
                        robot_obs_dim=self.robot_obs_dim,
                    )
                else:
                    robot_obs = torch.zeros(
                        (
                            (
                                self.max_num_robots
                                if self.max_num_robots is not None
                                else (
                                    self.num_robots
                                    if hasattr(self, "num_robots")
                                    else 1
                                )
                            ),
                            self.robot_obs_dim,
                        ),
                        dtype=torch.float32,
                    )

                if "adj.npy" in sample:
                    adj_np = _as_numpy(sample["adj.npy"])
                    if hasattr(adj_np, "numpy"):
                        adj_np = adj_np.numpy()
                    adj = torch.from_numpy(adj_np).float()
                    if (
                        self.max_num_robots is not None
                        and adj.shape[0] < self.max_num_robots
                    ):
                        new_adj = torch.zeros(
                            (self.max_num_robots, self.max_num_robots),
                            dtype=torch.float32,
                        )
                        k = adj.shape[0]
                        new_adj[:k, :k] = adj
                        adj = new_adj
                else:
                    n = (
                        self.max_num_robots
                        if self.max_num_robots is not None
                        else robot_obs.shape[0]
                    )
                    adj = torch.eye(n, dtype=torch.float32)

                # Reward
                # --- NEW REWARD LOGIC: Time-based ---
                base_reward = 0.0
                try:
                    traj_id = ep_id
                    step_val = state_data.get("step", 0)
                    # npy index is 0-based, step is 1-based
                    step_idx = int(step_val) - 1

                    if traj_id not in self.times_cache:
                        # Base directory for times files as provided by user
                        times_dir = "/home/adi2440/Desktop/MARL_Shahil_Aditya/VLCM_Data_Collection/RWARE/data_test"
                        times_path = os.path.join(times_dir, f"{traj_id}_times.npy")
                        if os.path.exists(times_path):
                            self.times_cache[traj_id] = np.load(times_path)
                        else:
                            self.times_cache[traj_id] = None

                    times_data = self.times_cache[traj_id]
                    if times_data is not None and 0 <= step_idx < times_data.shape[1]:
                        step_times = times_data[:, step_idx]
                        # Reward rules: >0 -> +5, <0 -> -5, else 0
                        # Scaled positive reward to balance the asymmetry
                        rs = [
                            5.0 if t > 0 else -5.0 if t < 0 else 0.0 for t in step_times
                        ]
                        base_reward = sum(rs) / max(len(rs), 1)
                    else:
                        base_reward = float(state_data.get("reward", 0.0))
                except Exception:
                    base_reward = float(state_data.get("reward", 0.0))

                dist_penalty = 0.0
                if "dist.npy" in sample:
                    d = _as_numpy(sample["dist.npy"])
                    if hasattr(d, "numpy"):
                        d = d.numpy()
                    if d.ndim == 2 and d.shape[0] > 1:
                        eye = np.eye(d.shape[0], dtype=bool)
                        if ((d < 3.0) & (~eye)).any():
                            dist_penalty = -1.0

                total_reward = base_reward + dist_penalty

                # --- NEW REWARD LOGIC ---
                # r_dist: Negative average minimum distance to requested boxes
                # If an agent is carrying a REQUESTED box, its distance cost is 0.

                try:
                    requests = state_data.get("requests", [])
                    # Filter valid requests present in our map
                    valid_request_positions = [
                        SHELF_MAP[r] for r in requests if r in SHELF_MAP
                    ]

                    if not valid_request_positions:
                        # Fallback if no known requested shelves:
                        # Maybe 0 reward or keep existing structure?
                        # If requests exist but not in map, we can't calculate distance.
                        # Proceed with 0 r_dist contribution if empty.
                        r_dist = 0.0
                    else:
                        num_agents = len(state_data.get("agents", []))
                        dist_sum = 0.0

                        agent_positions = []
                        for ag in state_data.get("agents", []):
                            # Agent Pos
                            pos = ag.get("pos", [0, 0])
                            # RWARE pos is [y, x] or [x, y]?
                            # Looking at extraction script: pos tuple(ag['pos']) matched (x,y) or (row, col)
                            # In map extraction, we just took raw pos values.
                            # So we should use raw pos values here too.

                            carrying_id = ag.get("carrying")

                            # Check if carrying a requested box
                            is_carrying_request = (carrying_id is not None) and (
                                carrying_id in requests
                            )

                            if is_carrying_request:
                                # Distance cost is 0
                                dist = 0.0
                            else:
                                # Calculate min distance to any valid request
                                # Use Euclidean distance
                                # pos and shelf_pos are lists/tuples
                                min_d = 1000.0
                                ax, ay = pos[0], pos[1]
                                for sx, sy in valid_request_positions:
                                    d = np.sqrt((ax - sx) ** 2 + (ay - sy) ** 2)
                                    if d < min_d:
                                        min_d = d

                                # Cap or just take min? Logic says "avg_min_dist"
                                # If min_d is still 1000.0 (no requests), handled by outer check
                                dist = min_d

                            dist_sum += dist

                        avg_min_dist = dist_sum / max(num_agents, 1)
                        # Normalize r_dist to [-1, 0] using max warehouse diagonal
                        r_dist = -min(avg_min_dist / _WAREHOUSE_MAX_DIST, 1.0)

                    total_reward += r_dist

                except Exception as e:
                    # Fallback to avoid crashing training if data is weird
                    # print(f"Reward Calc Error: {e}")
                    pass

                reward = torch.tensor(total_reward, dtype=torch.float32)
                done = torch.tensor(0.0, dtype=torch.float32)

                # Text
                step_val = state_data.get("step", 0)
                requests = state_data.get("requests", [])
                agents = state_data.get("agents", [])

                # Parse config for environment description
                cfg = self.rware_config
                n_ag = len(agents) if agents else "unknown"
                # Extract difficulty from config name
                # e.g. "tiny-2ag-hard", "tiny-4ag-easy-v2",
                #      "mixed-rware"
                cfg_lower = cfg.lower() if cfg else ""
                if "hard" in cfg_lower:
                    difficulty = "hard"
                elif "easy" in cfg_lower:
                    difficulty = "easy"
                else:
                    difficulty = "default"

                # Build structured observation lines
                obs_lines = []
                obs_lines.append(f"Timestep: {step_val}.")
                obs_lines.append(f"Requested boxes: {requests}.")
                for ag in agents:
                    aid = ag.get("id", "?")
                    pos = ag.get("pos", "?")
                    d = ag.get("dir", "?")
                    act = ag.get("action", "?")
                    carry = ag.get("carrying")
                    carry_str = "yes" if carry else "no"
                    obs_lines.append(
                        f"Agent {aid}: position {pos},"
                        f" facing {d},"
                        f" action {act},"
                        f" carrying {carry_str}."
                    )

                if self.text_prompt_template is None:
                    header = (
                        "You are an expert vision language critic model for multi-agent teams able to critize given trajectories of data for their n-step returns, thus critizing the policy. "
                        f"This is a robotic warehouse environment with {n_ag} agents ({difficulty} difficulty, config: {cfg}). "
                        "The reward is the sum of the minimum euclidean distance between an agent and its closest box, plus if an agent successfully places a box in the goal location (+5), and a penalty if the agents come within 3m of one another (-1). "
                        "Assess the quality of the current policy based on these observations: "
                    )
                    text = header + " ".join(obs_lines)
                else:
                    text = self.text_prompt_template + " " + " ".join(obs_lines)

            elif self.dataset_type == "offroad":
                # ---------------- OFFROAD Handling ----------------
                try:
                    state_json = json.loads(sample["state.json"])
                except Exception as e:
                    print(f"Error parsing state.json: {e}")
                    continue

                # Parse OFFROAD observation tensor [max_num_robots, 8]
                robot_obs = _parse_offroad_state(
                    state_json,
                    num_robots=self.num_robots,
                    robot_obs_dim=self.robot_obs_dim,
                )

                # Adj Matrix directly from saved numpy array
                adj = _as_numpy(sample["adj.npy"])
                if hasattr(adj, "numpy"):
                    adj = adj.numpy()
                adj = torch.tensor(adj, dtype=torch.float32)
                if (
                    self.max_num_robots is not None
                    and adj.shape[0] < self.max_num_robots
                ):
                    new_adj = torch.zeros(
                        (self.max_num_robots, self.max_num_robots),
                        dtype=torch.float32,
                    )
                    k = adj.shape[0]
                    new_adj[:k, :k] = adj
                    adj = new_adj

                # Reward: mean of per-agent rewards
                agents = state_json.get("agents", [])
                if len(agents) > 0:
                    agent_rewards = [ag.get("reward", 0.0) for ag in agents]
                    reward = float(np.mean(agent_rewards))
                else:
                    reward = 0.0

                # Done: True if ANY agent reached goal (or use dones.npy)
                # Currently using individual reached signals
                done = any([ag.get("reached", False) for ag in agents])

                # Construct text prompt specific to OFFROAD
                obs_lines = []
                for i, ag in enumerate(agents[: self.num_robots]):
                    ag_id = ag.get("id", i)
                    color = ag.get("color", "unknown")
                    pos = ag.get("pos", [0.0, 0.0])
                    yaw = ag.get("yaw", 0.0)
                    vel = ag.get("vel", [0.0, 0.0])
                    v = np.linalg.norm(vel)
                    dist_to_goal = ag.get("dist_to_goal", 0.0)
                    traversability = ag.get("traversability", 0.0)
                    reached = "yes" if ag.get("reached", False) else "no"
                    collision = "yes" if ag.get("collision", False) else "no"

                    obs_lines.append(
                        f"Agent {ag_id} ({color}): position ({pos[0]:.2f}, {pos[1]:.2f}), "
                        f"heading {yaw:.2f} rad, speed {v:.2f} m/s, dist_to_goal {dist_to_goal:.2f}m, "
                        f"traversability {traversability:.2f}, reached: {reached}, collision: {collision}."
                    )

                n_ag = len(agents)
                step_idx = state_json.get("episode_meta", {}).get(
                    "step", episode_frame_count
                )

                if self.text_prompt_template is None:
                    header = (
                        "You are an expert vision language critic model for multi-agent teams able to critize given trajectories of data for their n-step returns, thus critizing the policy. "
                        f"This is an offroad navigation environment with {n_ag} agents traversing rough terrain. "
                        "The reward is based on the progress towards the goal, a heading alignment reward, minus penalties for the distance to goal, each step taken (which increases over time), idling, control effort, and a terrain traversability penalty equal to 2.0 * (1 - traversability)^2. "
                        "Assess the quality of the current policy based on these observations: "
                        f"Timestep: {step_idx}. "
                    )
                    text = header + " ".join(obs_lines)
                else:
                    text = self.text_prompt_template + " " + " ".join(obs_lines)

            else:
                # Default MA-VLCM behavior
                robot_src = _as_numpy(sample[f"{self.robot_source}.npy"])
                if hasattr(robot_src, "numpy"):
                    robot_src = robot_src.numpy()
                robot_obs = torch.tensor(robot_src, dtype=torch.float32)

                num_nodes = robot_obs.shape[0]
                edge_index = _as_numpy(sample["edge_index.npy"])
                adj = _edge_index_to_adj(edge_index, num_nodes)

                if self.text_mode == "raw":
                    text = self.text_prompt_template
                else:
                    text = _as_numpy(sample["text_emb.npy"])
                    if hasattr(text, "numpy"):
                        text = text.numpy()
                    text = torch.tensor(text, dtype=torch.float32)

                reward = _reward_from_frame(sample, self.reward_reduce)
                done = _done_from_frame(sample, self.done_reduce)

            buffer.append(
                {
                    "image": image,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "text": text,
                    "reward": reward,
                    "done": done,
                }
            )
            episode_frame_count += 1

            # Sliding window yield logic
            min_len = self.clip_len + (1 if self.include_next else 0)

            # While we have enough data to yield at least one clip
            while len(buffer) >= min_len:
                # Check if the clip starting at buffer[0] is valid according to stride
                # The start index of the clip at buffer[0] is (episode_frame_count - len(buffer))
                current_start_index = episode_frame_count - len(buffer)

                if current_start_index % self.clip_stride == 0:
                    buf_list = list(buffer)
                    clip = buf_list[: self.clip_len]
                    next_clip = None
                    if self.include_next:
                        next_clip = buf_list[1 : 1 + self.clip_len]

                    out_sample = _process_clip_data(clip, next_clip)
                    if out_sample is not None:
                        yield out_sample

                # Pop the oldest frame to slide the window
                buffer.popleft()


def webdataset_loader(
    args, shards, batch_size, num_workers, shuffle=False, dataset_type=None
):
    # Support glob patterns if passed as shards string
    if isinstance(shards, str) and "*" in shards:
        import glob

        expanded = sorted(glob.glob(shards))
        if expanded:
            shards = expanded
            print(f"Expanded shard pattern specific to {len(shards)} files.")
        else:
            print(f"Warning: No files matched glob pattern {shards}")

    if dataset_type is None:
        dataset_type = args.dataset_type

    vlm_processor = None
    # vlm_processor is now lazily loaded in the dataset worker to avoid pickling issues
    # and ensure robustness.

    dataset = SequenceWebDataset(
        shards=shards,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        text_mode=args.text_mode,
        robot_source=args.robot_source,
        reward_reduce=args.reward_reduce,
        done_reduce=args.done_reduce,
        vlm_processor=None,
        vl_model_name=args.vl_model_name if args.preprocess_in_loader else None,
        robot_obs_dim=args.robot_obs_dim,
        num_robots=args.num_robots,
        max_num_robots=args.num_robots,
        shuffle_shards=shuffle,
        text_prompt_template=args.text_prompt_template,
        dataset_type=args.dataset_type,
        rware_config=args.rware_config,
        return_mode=args.return_mode,
        n_step=args.n_step,
        gamma=args.gamma,
        keep_raw_video=False,
        include_next=True,  # Always bootstrap with nstep returns
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=True,
        vlm_padding="max_length",
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        vl_backend=args.vl_backend,
    )

    def _collate(batch):
        def _stack_inputs(items):
            out = {}
            if not items:
                return out
            keys = items[0].keys()
            for k in keys:
                vals = [d[k] for d in items]
                if torch.is_tensor(vals[0]):
                    if k in ["input_ids", "attention_mask"]:
                        out[k] = torch.nn.utils.rnn.pad_sequence(
                            vals, batch_first=True, padding_value=0
                        )
                    elif k == "labels":
                        out[k] = torch.nn.utils.rnn.pad_sequence(
                            vals, batch_first=True, padding_value=-100
                        )
                    else:
                        out[k] = torch.stack(vals, dim=0)
                else:
                    out[k] = vals
            return out

        out = {
            "inputs": _stack_inputs([b["inputs"] for b in batch]),
            "robot_obs": torch.stack([b["robot_obs"] for b in batch], dim=0),
            "adj": torch.stack([b["adj"] for b in batch], dim=0),
            "reward": torch.stack([b["reward"] for b in batch], dim=0).view(-1),
            "done": torch.stack([b["done"] for b in batch], dim=0).view(-1),
        }
        if "next_inputs" in batch[0]:
            out["next_inputs"] = _stack_inputs([b["next_inputs"] for b in batch])
            out["next_robot_obs"] = torch.stack(
                [b["next_robot_obs"] for b in batch], dim=0
            )
            out["next_adj"] = torch.stack([b["next_adj"] for b in batch], dim=0)
        if "returns" in batch[0]:
            out["returns"] = torch.stack([b["returns"] for b in batch], dim=0).view(-1)
        return out

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _contrastive_pairwise_loss(scores, rewards, margin=0.0):
    # scores: [B], rewards: [B]
    # For each pair with different reward, enforce higher reward -> higher score.
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


def run_epoch(
    model,
    loader,
    optimizer,
    accelerator,
    log_every,
    gamma,
    args,
    train=True,
    scheduler=None,
    max_steps=None,
):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0.0
    step = 0

    phase = "train" if train else "val"
    show_pbar = accelerator.is_main_process
    pbar = tqdm(
        loader,
        desc=f"{phase}",
        disable=not show_pbar,
        dynamic_ncols=True,
    )

    for batch in pbar:
        if max_steps is not None and step >= max_steps:
            break
        step += 1

        def _move_inputs(inputs):
            moved = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    moved[k] = v.to(accelerator.device)
                else:
                    moved[k] = v
            return moved

        inputs = _move_inputs(batch["inputs"])
        robot_obs = batch["robot_obs"].to(accelerator.device)
        adj = batch["adj"].to(accelerator.device)
        reward = batch["reward"].to(accelerator.device)
        done = batch["done"].to(accelerator.device).float()
        use_next = True  # Always bootstrap with nstep returns
        if use_next:
            next_inputs = _move_inputs(batch["next_inputs"])
            next_robot_obs = batch["next_robot_obs"].to(accelerator.device)
            next_adj = batch["next_adj"].to(accelerator.device)
        else:
            next_inputs = None
            next_robot_obs = None
            next_adj = None

        if (
            train
            and args.debug_save_video
            and not getattr(run_epoch, "_debug_saved", False)
        ):
            _save_debug_video(batch, args, accelerator, tag="train")
            run_epoch._debug_saved = True

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                pred = model(inputs, robot_obs, adj)
                if args.loss_type in ("contrastive", "contrastive_mse", "mse"):
                    # Calculate bootstrapped target
                    if use_next:
                        with torch.no_grad():
                            next_pred = model(
                                next_inputs,
                                next_robot_obs,
                                next_adj,
                            )
                        clip_gamma = gamma**args.clip_len
                        if "returns" in batch:
                            nstep_returns = batch["returns"].to(accelerator.device)
                            target = (
                                nstep_returns + clip_gamma * (1.0 - done) * next_pred
                            )
                        else:
                            # Fallback if somehow returns are missing, but SequenceWebDataset provides it
                            target = reward + clip_gamma * (1.0 - done) * next_pred
                    else:
                        if "returns" in batch:
                            target = batch["returns"].to(accelerator.device)
                        else:
                            target = reward

                    if args.loss_type in ("contrastive", "contrastive_mse"):
                        contrastive_loss = _contrastive_pairwise_loss(
                            pred.view(-1),
                            target.view(-1),
                            margin=args.contrastive_margin,
                        )
                        if args.loss_type == "contrastive_mse":
                            mse_loss = F.mse_loss(pred.view(-1), target.view(-1))
                            loss = contrastive_loss + args.mse_loss_weight * mse_loss
                        else:
                            mse_loss = None
                            loss = contrastive_loss
                    elif args.loss_type == "mse":
                        loss = loss_fn(pred, target)
                if train:
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0 and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), max_norm=args.max_grad_norm
                        )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        avg_loss = total_loss / step

        # Update progress bar
        postfix = {"loss": f"{avg_loss:.4f}"}
        if train and scheduler is not None:
            try:
                lr_val = scheduler.get_last_lr()[0]
                postfix["lr"] = f"{lr_val:.2e}"
            except Exception:
                pass
        pbar.set_postfix(postfix)

        # Log step-level metrics to wandb (main process only)
        if train and accelerator.is_main_process:
            try:
                import wandb

                if wandb.run is not None:
                    log_dict = {
                        "train/step_loss": loss.item(),
                        "train/step": step,
                        "train/pred_mean": pred.detach().mean().item(),
                        "train/reward_mean": reward.detach().mean().item(),
                    }
                    if "returns" in batch:
                        log_dict["train/true_returns_mean"] = (
                            batch["returns"].to(accelerator.device).mean().item()
                        )

                    # Always log the bootstrapped target mean
                    log_dict["train/target_mean"] = target.detach().mean().item()
                    if args.loss_type in ("contrastive", "contrastive_mse"):
                        log_dict["train/contrastive_loss"] = contrastive_loss.item()
                        if mse_loss is not None:
                            log_dict["train/mse_loss"] = mse_loss.item()
                    wandb.log(log_dict)
            except ImportError:
                pass

    pbar.close()
    return total_loss / max(step, 1)


def split_shards(shards_pattern, val_split=0.2, seed=42):
    import glob, random

    if not isinstance(shards_pattern, str):
        return shards_pattern, None

    # Handle remote URLs seamlessly
    if shards_pattern.startswith(("hf://", "http://", "https://", "pipe:")):
        # Cannot glob remote URLs easily. Assume validation was handled earlier or use them entirely
        return shards_pattern, None

    if "*" not in shards_pattern:
        return shards_pattern, None

    files = sorted(glob.glob(shards_pattern))
    if not files:
        return shards_pattern, None

    random.Random(seed).shuffle(files)
    split_idx = int(len(files) * (1.0 - val_split))
    train_shards = files[:split_idx]
    val_shards = files[split_idx:]

    if len(val_shards) == 0:
        val_shards = None

    return train_shards, val_shards


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_str = "ContrastiveMSE"
    if args.loss_type == "mse":
        loss_str = "MSE"
    elif args.loss_type == "contrastive":
        loss_str = "Contrastive"

    ret_str = f"{args.n_step}StepReturn"

    args.run_name = f"{ret_str}_{loss_str}_{timestamp}"

    # ── Performance: enable TF32 for H100 / Ampere+ GPUs ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.preprocess_in_loader:
        args.video_preprocessed = True

    if args.peft == "qlora" and args.fsdp:
        raise RuntimeError(
            "FSDP + QLoRA is not supported. Use DDP (no --fsdp) or LoRA."
        )

    if args.peft == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "QLoRA requested but bitsandbytes/transformers are not available."
            ) from e
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
            raise RuntimeError(
                "FSDP requested but accelerate FSDP plugin is unavailable."
            )
        fsdp_kwargs = {}
        use_orig_params = args.fsdp_use_orig_params or (args.peft != "none")
        if size_based_auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = functools.partial(
                size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params
            )
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
                raise RuntimeError(
                    "FSDP CPU offload requested but torch.distributed.fsdp is unavailable."
                )
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

    ddp_kwargs = None
    if not args.fsdp and DistributedDataParallelKwargs is not None:
        find_unused = (
            args.ddp_find_unused_parameters or (args.peft != "none") or args.freeze_vl
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else [],
    )

    # --- WandB init (main process only) ---
    if accelerator.is_main_process:
        try:
            import wandb

            wandb.init(
                entity="i2rLAB",
                project="VLCM_Training_RWARE",
                name=args.run_name,
                config=vars(args),
            )
        except ImportError:
            accelerator.print("wandb not installed, skipping logging.")
    accelerator.wait_for_everyone()

    # Serialize model loading to avoid System RAM OOM (SIGKILL)
    # Each process loads the model sequentially instead of simultaneously.
    with accelerator.main_process_first():
        model = build_model(args, device=accelerator.device)
    model = _apply_peft(model, args)
    if args.fsdp:
        if args.mixed_precision == "bf16":
            target_dtype = torch.bfloat16
        elif args.mixed_precision == "fp16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
        model = model.to(dtype=target_dtype)
        # Enforce a uniform dtype across all params/buffers for FSDP flattening.
        for p in model.parameters():
            if p.dtype != target_dtype:
                p.data = p.data.to(dtype=target_dtype)
        for b in model.buffers():
            if torch.is_floating_point(b) and b.dtype != target_dtype:
                b.data = b.data.to(dtype=target_dtype)

    if args.compile:
        model = torch.compile(model)

    # ── Build param groups: separate LR for vision tower ──
    vision_tower = getattr(model, "backbone", None)
    if vision_tower is not None:
        vision_tower = getattr(vision_tower, "model", None)
    if vision_tower is not None:
        vision_tower = getattr(vision_tower, "vision_tower", None)

    if vision_tower is not None:
        vision_tower_ids = {id(p) for p in vision_tower.parameters() if p.requires_grad}
        vision_params = [
            p
            for p in model.parameters()
            if p.requires_grad and id(p) in vision_tower_ids
        ]
        other_params = [
            p
            for p in model.parameters()
            if p.requires_grad and id(p) not in vision_tower_ids
        ]
        accelerator.print(
            f"Param groups: vision_tower={len(vision_params)} params (lr={args.vision_lr}), "
            f"other={len(other_params)} params (lr={args.lr})"
        )
        param_groups = [
            {"params": other_params, "lr": args.lr},
            {"params": vision_params, "lr": args.vision_lr},
        ]
    else:
        accelerator.print("No vision tower found; using single param group.")
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.lr,
            }
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ── Cosine LR scheduler with linear warmup ──
    # Estimate total steps: (samples_per_epoch / batch_size / grad_accum) * epochs
    warmup_fraction = 0.05
    estimated_steps_per_epoch = max(
        args.samples_per_epoch
        // max(args.batch_size * max(1, args.grad_accum_steps), 1),
        100,
    )
    total_steps = estimated_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * warmup_fraction)
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    # --- Shard Splitting & Loader Creation ---
    train_main_shards, val_main_shards = split_shards(args.train_shards, args.val_split)

    if args.offroad_shards:
        accelerator.print(
            f"Multi-dataset training enabled. Interleaving {args.dataset_type} and offroad datasets."
        )
        train_offroad_shards, val_offroad_shards = split_shards(
            args.offroad_shards, args.val_split
        )

        class InterleavedDataLoader:
            def __init__(self, loader1, loader2, main_shards, offroad_shards):
                self.loader1 = loader1
                self.loader2 = loader2
                self.main_shards_count = len(main_shards) if main_shards else 0
                self.offroad_shards_count = len(offroad_shards) if offroad_shards else 0

            def __iter__(self):
                iter1 = iter(self.loader1)
                iter2 = iter(self.loader2)
                exhausted1 = False
                exhausted2 = False
                while not (exhausted1 and exhausted2):
                    if not exhausted1:
                        try:
                            yield next(iter1)
                        except StopIteration:
                            exhausted1 = True

                    if not exhausted2:
                        try:
                            yield next(iter2)
                        except StopIteration:
                            exhausted2 = True

            def __len__(self):
                # Approximation of dataset length for progress bar and steps
                # Assumes ~50 clips per shard (as stated in defaults)
                total_samples = (
                    self.main_shards_count + self.offroad_shards_count
                ) * 100
                # Scale correctly down based on batch size & world size
                return max(
                    1,
                    total_samples
                    // max(1, args.batch_size)
                    // max(1, accelerator.num_processes),
                )

        # Train Loader
        main_train_loader = webdataset_loader(
            args,
            train_main_shards,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            dataset_type=args.dataset_type,
        )
        offroad_train_loader = webdataset_loader(
            args,
            train_offroad_shards,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            dataset_type="offroad",
        )
        train_loader = InterleavedDataLoader(
            main_train_loader,
            offroad_train_loader,
            train_main_shards,
            train_offroad_shards,
        )

        # Val Loader
        val_loader = None
        if val_main_shards and val_offroad_shards:
            main_val_loader = webdataset_loader(
                args,
                val_main_shards,
                args.batch_size,
                args.num_workers,
                shuffle=False,
                dataset_type=args.dataset_type,
            )
            offroad_val_loader = webdataset_loader(
                args,
                val_offroad_shards,
                args.batch_size,
                args.num_workers,
                shuffle=False,
                dataset_type="offroad",
            )
            val_loader = InterleavedDataLoader(main_val_loader, offroad_val_loader)

    else:
        # Standard single dataset loading
        train_loader = webdataset_loader(
            args, train_main_shards, args.batch_size, args.num_workers, shuffle=True
        )
        val_loader = None
        if val_main_shards:
            val_loader = webdataset_loader(
                args, val_main_shards, args.batch_size, args.num_workers, shuffle=False
            )

    if val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            accelerator,
            args.log_every,
            args.gamma,
            args,
            train=True,
            scheduler=scheduler,
            max_steps=estimated_steps_per_epoch,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                accelerator,
                args.log_every,
                args.gamma,
                args,
                train=False,
                max_steps=max(int(estimated_steps_per_epoch * 0.25), 10),
            )

        accelerator.print(
            f"epoch={epoch} train_loss={train_loss:.4f}"
            f" val_loss={val_loss if val_loss is not None else 'n/a'}"
        )

        # Log epoch-level metrics to wandb
        if accelerator.is_main_process:
            try:
                import wandb

                if wandb.run is not None:
                    log_dict = {
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                    }
                    if val_loss is not None:
                        log_dict["val/epoch_loss"] = val_loss
                    wandb.log(log_dict)
            except ImportError:
                pass

        if accelerator.is_main_process:
            ckpt_name = f"{args.run_name}_epoch_{epoch}"

            ckpt = {
                "model": accelerator.get_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"{ckpt_name}.pt"))

            # Save accompanying JSON spec file
            info_dict = {
                "dataset_type": args.dataset_type,
                "config": vars(args),
                "sample_text_prompt": (
                    args.text_prompt_template
                    if args.text_prompt_template is not None
                    else "(auto-generated depending on dataset_type)"
                ),
            }
            with open(os.path.join(args.save_dir, f"{ckpt_name}_info.json"), "w") as f:
                json.dump(info_dict, f, indent=4)

        accelerator.wait_for_everyone()

    # Finish wandb run
    if accelerator.is_main_process:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
