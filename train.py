import argparse
import os
import io
import glob
import functools
import inspect

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator

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
        default="osdf:///ospool/ap40/data/aditya.parameshwaran",
        help="WebDataset shard pattern for training",
    )
    p.add_argument(
        "--val_shards",
        type=str,
        default="",
        help="Optional WebDataset shard pattern for validation",
    )
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--samples_per_epoch", type=int, default=1000)
    p.add_argument("--text_mode", type=str, default="raw", choices=["raw", "emb"])
    p.add_argument("--text_prompt_template", type=str, default=None)
    p.add_argument(
        "--dataset_type", type=str, default="rware", choices=["default", "rware"]
    )
    p.add_argument("--rware_config", type=str, default="tiny-2ag-hard")

    # Sequence building
    p.add_argument("--clip_len", type=int, default=20)
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
    p.add_argument("--return_mode", type=str, default="td", choices=["td", "nstep"])
    p.add_argument("--n_step", type=int, default=50)
    p.add_argument(
        "--loss_type", type=str, default="contrastive", choices=["td", "contrastive"]
    )
    p.add_argument("--contrastive_margin", type=float, default=0.0)

    # Accelerate
    p.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"]
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
        choices=["deepseek_vl", "deepseek_vl2", "llava_video"],
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
    p.add_argument("--vl_max_text_len", type=int, default=256)
    p.add_argument("--freeze_vl", action="store_true")

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
    p.add_argument("--video_height", type=int, default=224)
    p.add_argument("--video_width", type=int, default=224)
    p.add_argument("--video_frames", type=int, default=8)
    p.add_argument("--video_preprocessed", action="store_true")
    p.add_argument("--video_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--video_std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    # Robots / graph
    p.add_argument("--num_robots", type=int, default=2)
    p.add_argument("--robot_obs_dim", type=int, default=8)

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

    return p.parse_args()


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
        text_prompt_template=None,
        return_mode="td",
        n_step=50,
        gamma=0.99,
        keep_raw_video=False,
        include_next=False,
        vlm_max_text_len=256,
        vlm_truncation=False,
        vlm_padding="longest",
        dataset_type="default",
        rware_config="tiny-2ag-hard",
    ):
        if isinstance(shards, str):
            if os.path.isdir(shards):
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

    def __iter__(self):
        if wds is None:
            raise RuntimeError("webdataset is not installed.")

        if self.vlm_processor is None and self.vl_model_name is not None:
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

        # Prefer explicit node/worker splitters for multi-GPU setups
        try:
            dataset = wds.WebDataset(
                self.shards,
                shardshuffle=False,
                nodesplitter=getattr(wds, "split_by_node", None),
                workersplitter=getattr(wds, "split_by_worker", None),
            ).decode("pil")
        except TypeError:
            dataset = wds.WebDataset(self.shards, shardshuffle=False).decode("pil")
            if hasattr(dataset, "split_by_node"):
                dataset = dataset.split_by_node()
            if hasattr(dataset, "split_by_worker"):
                dataset = dataset.split_by_worker()

        current_ep = None
        buffer = []

        def flush_buffer():
            min_len = self.clip_len + (1 if self.include_next else 0)
            if len(buffer) < min_len:
                return
            max_i = len(buffer) - self.clip_len - (1 if self.include_next else 0)
            for i in range(0, max_i + 1, self.clip_stride):
                clip = buffer[i : i + self.clip_len]

                raw_video = [f["image"] for f in clip]
                raw_next_video = None
                if self.include_next:
                    next_clip = buffer[i + 1 : i + 1 + self.clip_len]
                    raw_next_video = [f["image"] for f in next_clip]
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
                                    text = text.replace(
                                        "<video>\n", "<video><obs>\n", 1
                                    )
                                else:
                                    text = f"<obs>\n{text}"
                        try:
                            max_len = (
                                self.vlm_max_text_len if self.vlm_truncation else None
                            )
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
                    if self.include_next:
                        next_inputs = _proc(raw_next_video, text)
                else:
                    raise RuntimeError("Dataloader proprocessor not set.")

                robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
                adj = torch.stack([f["adj"] for f in clip], dim=0)
                if self.include_next:
                    next_robot_obs = torch.stack(
                        [f["robot_obs"] for f in next_clip], dim=0
                    )
                    next_adj = torch.stack([f["adj"] for f in next_clip], dim=0)

                reward = clip[-1]["reward"]
                done = clip[-1]["done"]

                returns = torch.stack([f["reward"] for f in clip], dim=0).sum(dim=0)

                out = {
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "reward": reward.view(1),
                    "returns": returns.view(1),
                    "done": done.view(1),
                }
                out["inputs"] = inputs
                if self.include_next:
                    out["next_inputs"] = next_inputs
                    out["next_robot_obs"] = next_robot_obs
                    out["next_adj"] = next_adj

                yield out

        for sample in dataset:
            key = sample.get("__key__", "")
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")

            if "_" in key:
                ep_id = key.split("_")[0]
            else:
                ep_id = key

            if current_ep is None:
                current_ep = ep_id

            if ep_id != current_ep:
                yield from flush_buffer()
                buffer = []
                current_ep = ep_id

            # Image loading
            if "image.png" in sample:
                image = sample["image.png"]
            elif "overhead.png" in sample:
                image = sample["overhead.png"]
            else:
                # print(f"Warning: No image found for key {key}. Skipping.")
                continue

            # Robot Obs and Adj
            if self.dataset_type == "rware":
                # RWARE specific parsing
                if "state.json" in sample:
                    # Parse JSON state
                    # We don't have easy access to num_robots here unless we hardcode or infer
                    # Let's infer from list length for now, or use safe default if empty
                    robot_obs = _parse_rware_state(
                        sample["state.json"], robot_obs_dim=self.robot_obs_dim
                    )
                else:
                    # Fallback or error?
                    robot_obs = torch.zeros((1, 8), dtype=torch.float32)

                if "adj.npy" in sample:
                    adj_np = _as_numpy(sample["adj.npy"])
                    if hasattr(adj_np, "numpy"):
                        adj_np = adj_np.numpy()
                    adj = torch.from_numpy(adj_np).float()
                else:
                    # Default fully connected or diagonal?
                    n = robot_obs.shape[0]
                    adj = torch.eye(n, dtype=torch.float32)

                # Reward (scalar json)
                step_reward = 0.0
                if "reward.json" in sample:
                    step_reward = float(sample["reward.json"])
                reward = torch.tensor(step_reward, dtype=torch.float32)

                # Done (inference)
                # WebDataset doesn't explicitly mark done for RWARE steps usually until end of file?
                # or is it in state? state.json doesn't seem to have "done".
                # We will assume 0 until episode change logic handles it (which is implicit in buffer flush?)
                # Actually buffer flushes on new episode ID.
                # So we can set done=0 always here?
                done = torch.tensor(0.0, dtype=torch.float32)

                # Text Prompt Generation
                # Text Prompt Generation
                if self.text_prompt_template is None:
                     # Auto-generate based on config
                     base_prompt = (
                        "You are a highly skilled vision language critic model. "
                        "Your goal is to criticise trajectories of data on the task at hand provided to you. "
                        "You are trained to identify the differences between good policies and bad policies, and return a critic value."
                     )
                     
                     # Check if we can extract config from the sample url/path
                     # sample['__url__'] typically contains the tar path
                     # e.g. "path/to/rware-tiny-2ag-easy-v2/xxxx.tar"
                     current_config = self.rware_config # fallback to default/arg
                     
                     if "__url__" in sample:
                         url = sample["__url__"]
                         # Heuristic: look for a segment that looks like rware config
                         # or simplistic: assume structure data_scratch/{config}/...
                         # We can try to split by / and find the one with "rware"
                         parts = url.split("/")
                         for p in parts:
                             if "rware" in p.lower():
                                 current_config = p
                                 break
                     
                     # Parse config for details
                     # e.g. "rware:rware-tiny-2ag-hard-v2" or "tiny-2ag-hard"
                     # Cleanup prefix if needed
                     clean_config = current_config.replace("rware:", "")
                     
                     parts = clean_config.split("-")
                     size_map = {"tiny": "11x10", "small": "22x20"} # Approximation
                     size = "standard"
                     n_agents = "N"
                     difficulty = "unknown"
                     
                     # Parse heuristic: rware-tiny-2ag-hard-v2
                     # parts: [rware, tiny, 2ag, hard, v2]
                     
                     if "tiny" in parts: size = size_map["tiny"]
                     if "small" in parts: size = size_map["small"]
                     
                     for p in parts:
                         if "ag" in p and p != "ag":
                             n_agents = p.replace("ag", "")
                             
                     if "hard" in parts: difficulty = "hard"
                     if "easy" in parts: difficulty = "easy"
                     
                     specifics = (
                         f" The environment is the robotic warehouse (RWARE). "
                         f"Agents must pick up requested boxes, drop them at goal locations, then return boxes to empty spots. "
                         f"Config: {clean_config}. This means {n_agents} agents in a {size} grid with {difficulty} load."
                     )
                     text = base_prompt + specifics
                else:
                    text = self.text_prompt_template

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

        yield from flush_buffer()


def webdataset_loader(args, shards, batch_size, num_workers):
    # Support glob patterns if passed as shards string
    if isinstance(shards, str) and "*" in shards:
        import glob
        expanded = sorted(glob.glob(shards))
        if expanded:
            shards = expanded
            print(f"Expanded shard pattern specific to {len(shards)} files.")
        else:
            print(f"Warning: No files matched glob pattern {shards}")

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
        text_prompt_template=args.text_prompt_template,
        dataset_type=args.dataset_type,
        rware_config=args.rware_config,
        return_mode=args.return_mode,
        n_step=args.n_step,
        gamma=args.gamma,
        keep_raw_video=False,
        include_next=(args.loss_type != "contrastive" and args.return_mode == "td"),
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=True,
        vlm_padding="max_length",
    )

    def _collate(batch):
        def _stack_inputs(items):
            out = {}
            keys = items[0].keys()
            for k in keys:
                vals = [d[k] for d in items]
                if torch.is_tensor(vals[0]):
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

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate
    )


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
    model, loader, optimizer, accelerator, log_every, gamma, args, train=True
):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0.0
    step = 0

    for batch in loader:
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
        use_td = args.loss_type != "contrastive" and args.return_mode == "td"
        if use_td:
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
                if args.loss_type == "contrastive":
                    # Use returns if available, otherwise use per-clip reward.
                    if args.return_mode == "nstep" and "returns" in batch:
                        returns = batch["returns"].to(accelerator.device)
                    else:
                        returns = reward
                    loss = _contrastive_pairwise_loss(
                        pred.view(-1), returns.view(-1), margin=args.contrastive_margin
                    )
                else:
                    if args.return_mode == "td":
                        with torch.no_grad():
                            next_pred = model(
                                next_inputs,
                                next_robot_obs,
                                next_adj,
                            )
                        target = reward + gamma * (1.0 - done) * next_pred
                    else:
                        target = batch["return"].to(accelerator.device)
                    loss = loss_fn(pred, target)
                if train:
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        if log_every > 0 and step % log_every == 0:
            avg = total_loss / step
            phase = "train" if train else "val"
            accelerator.print(f"{phase} step={step} loss={avg:.4f}")

    return total_loss / max(step, 1)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loader = webdataset_loader(
        args, args.train_shards, args.batch_size, args.num_workers
    )
    val_loader = None
    if args.val_shards:
        val_loader = webdataset_loader(
            args, args.val_shards, args.batch_size, args.num_workers
        )

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
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
            )

        accelerator.print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss if val_loss is not None else 'n/a'}"
        )

        if accelerator.is_main_process:
            ckpt = {
                "model": accelerator.get_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt"))
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
