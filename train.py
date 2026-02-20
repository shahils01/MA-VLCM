import argparse
import os
import io
import functools
import inspect
import importlib
import sys

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
def _resolve_hf_load_dataset():
    # Prefer Hugging Face `datasets` even if a local `datasets.py` exists in the repo.
    try:
        mod = importlib.import_module("datasets")
        fn = getattr(mod, "load_dataset", None)
        if fn is not None:
            return fn, None
    except Exception as e:
        first_err = e
    else:
        first_err = RuntimeError("Imported module `datasets` has no attribute `load_dataset`.")

    original_path = list(sys.path)
    try:
        cwd = os.path.abspath(os.getcwd())
        cleaned = []
        for p in original_path:
            pp = os.path.abspath(p) if p else cwd
            if pp != cwd and p != "":
                cleaned.append(p)
        sys.path = cleaned
        mod = importlib.import_module("datasets")
        fn = getattr(mod, "load_dataset", None)
        if fn is None:
            raise RuntimeError("Imported module `datasets` has no attribute `load_dataset`.")
        return fn, None
    except Exception as e:
        return None, (first_err, e)
    finally:
        sys.path = original_path


load_dataset, _load_dataset_import_err = _resolve_hf_load_dataset()
from model import ModelConfig, MultimodalValueModel

KEY_SUFFIX_ALIASES = {
    ".image.png": "image.png",
    ".obs.npy": "obs.npy",
    ".state.npy": "state.npy",
    ".edge_index.npy": "edge_index.npy",
    ".text_emb.npy": "text_emb.npy",
    ".caption.txt": "caption.txt",
    ".rewards.npy": "rewards.npy",
    ".dones.npy": "dones.npy",
}


def parse_args():
    p = argparse.ArgumentParser()

    # Data / webdataset
    p.add_argument("--data_backend", type=str, default="webdataset", choices=["webdataset", "huggingface"])
    p.add_argument("--train_shards", type=str, default="", help="WebDataset shard pattern for training")
    p.add_argument("--val_shards", type=str, default="", help="Optional WebDataset shard pattern for validation")
    p.add_argument("--hf_dataset", type=str, default="", help="Hugging Face dataset name or local dataset path")
    p.add_argument("--hf_config", type=str, default="", help="Optional Hugging Face dataset config")
    p.add_argument("--hf_train_split", type=str, default="train", help="Hugging Face dataset split for training")
    p.add_argument("--hf_val_split", type=str, default="", help="Optional Hugging Face dataset split for validation")
    p.add_argument("--hf_streaming", action="store_true", help="Use streaming mode for Hugging Face datasets")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--samples_per_epoch", type=int, default=10000)
    p.add_argument("--text_mode", type=str, default="raw", choices=["raw", "emb"])
    p.add_argument("--text_prompt_template", type=str, default="You are a critic model. You are given video frames, robot state sequences, and a graph adjacency per timestep for a robot team. Assess how good or bad the current policy is at the task and respond with a single scalar judgment.")

    # Sequence building
    p.add_argument("--clip_len", type=int, default=20)
    p.add_argument("--clip_stride", type=int, default=1)
    p.add_argument("--robot_source", type=str, default="obs", choices=["obs", "state"])
    p.add_argument("--reward_reduce", type=str, default="mean", choices=["mean", "sum", "first"])
    p.add_argument("--done_reduce", type=str, default="any", choices=["any", "all", "mean", "sum", "first"])
    p.add_argument("--preprocess_in_loader", default=True, action="store_true", help="Use VLM image processor in dataloader")
    p.add_argument("--debug_save_video", action="store_true", help="Save one video sample for debugging")
    p.add_argument("--debug_out_dir", type=str, default="debug_samples")

    # Value targets
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--return_mode", type=str, default="td", choices=["td", "nstep"])
    p.add_argument("--n_step", type=int, default=50)
    p.add_argument("--loss_type", type=str, default="contrastive", choices=["td", "contrastive"])
    p.add_argument("--contrastive_margin", type=float, default=0.0)

    # Accelerate
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--fsdp", action="store_true", help="Use FSDP to shard model parameters across GPUs")
    p.add_argument("--fsdp_min_num_params", type=int, default=1_000_000, help="Auto-wrap threshold for FSDP")
    p.add_argument("--fsdp_cpu_offload", action="store_true", help="Offload FSDP parameters to CPU when not in use")
    p.add_argument("--fsdp_use_orig_params", action="store_true", help="Use FSDP use_orig_params to allow mixed requires_grad")
    p.add_argument("--ddp_find_unused_parameters", action="store_true", help="Set DDP find_unused_parameters=True")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")

    # DeepSeek VLM backbone
    p.add_argument(
        "--vl_backend", type=str, default="deepseek_vl", choices=["deepseek_vl", "deepseek_vl2", "llava_video"]
    )
    p.add_argument("--vl_model_name", type=str, default="deepseek-community/deepseek-vl-1.3b-base")
    p.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--vl_max_text_len", type=int, default=256)
    p.add_argument("--freeze_vl", action="store_true")

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
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging via Accelerate trackers")
    p.add_argument("--wandb_project", type=str, default="ma-vlcm", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team (optional)")
    p.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional)")
    p.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags (optional)")

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
        raise RuntimeError("PEFT requested but 'peft' is not installed. `pip install peft`.") from e

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

    print('edge_index shape = ', edge_index.shape)
    if hasattr(edge_index, "shape") and len(edge_index.shape) == 2 and edge_index.shape[0] == num_nodes and edge_index.shape[1] == num_nodes:
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
        text_prompt_template=None,
        return_mode="td",
        n_step=50,
        gamma=0.99,
        keep_raw_video=False,
        include_next=False,
        vlm_max_text_len=256,
        vlm_truncation=False,
        vlm_padding="longest",
    ):
        self.shards = shards
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.text_mode = text_mode
        self.robot_source = robot_source
        self.reward_reduce = reward_reduce
        self.done_reduce = done_reduce
        self.vlm_processor = vlm_processor
        self.text_prompt_template = text_prompt_template
        self.return_mode = return_mode
        self.n_step = n_step
        self.gamma = gamma
        self.keep_raw_video = keep_raw_video
        self.include_next = include_next
        self.vlm_max_text_len = vlm_max_text_len
        self.vlm_truncation = vlm_truncation
        self.vlm_padding = vlm_padding

    @staticmethod
    def _as_bool(x):
        if torch.is_tensor(x):
            if x.numel() == 0:
                return False
            return bool(x.detach().float().max().item() > 0.0)
        return bool(x)

    @staticmethod
    def _terminal_pad_from(frame):
        padded = dict(frame)
        reward = frame["reward"]
        done = frame["done"]
        if torch.is_tensor(reward):
            padded["reward"] = reward.new_zeros(())
        else:
            padded["reward"] = 0.0
        if torch.is_tensor(done):
            padded["done"] = torch.ones_like(done, dtype=done.dtype)
        else:
            padded["done"] = True
        return padded

    def _apply_done_termination(self, clip):
        """Terminate clip at first done=True and pad the tail with terminal-frame copies."""
        done_idx = None
        for idx, frame in enumerate(clip):
            if self._as_bool(frame["done"]):
                done_idx = idx
                break
        if done_idx is None or done_idx == len(clip) - 1:
            return clip

        terminal = clip[done_idx]
        out = list(clip[: done_idx + 1])
        for _ in range(done_idx + 1, len(clip)):
            out.append(self._terminal_pad_from(terminal))
        return out

    def __iter__(self):
        if wds is None:
            raise RuntimeError("webdataset is not installed.")

        # Prefer explicit node/worker splitters for multi-GPU setups
        try:
            dataset = wds.WebDataset(
                self.shards,
                shardshuffle=100,
                nodesplitter=getattr(wds, "split_by_node", None),
                workersplitter=getattr(wds, "split_by_worker", None),
            ).decode("pil")
        except TypeError:
            dataset = wds.WebDataset(self.shards, shardshuffle=100).decode("pil")
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
            start_idxs = list(range(0, max_i + 1, self.clip_stride))
            if len(start_idxs) > 1:
                perm = torch.randperm(len(start_idxs)).tolist()
                start_idxs = [start_idxs[p] for p in perm]

            for i in start_idxs:
                clip = self._apply_done_termination(buffer[i : i + self.clip_len])

                raw_video = [f["image"] for f in clip]
                raw_next_video = None
                if self.include_next:
                    next_clip = self._apply_done_termination(buffer[i + 1 : i + 1 + self.clip_len])
                    raw_next_video = [f["image"] for f in next_clip]
                if self.vlm_processor is not None:
                    def _proc(frames, text):
                        if not isinstance(text, str):
                            text = self.text_prompt_template
                        tokenizer = getattr(self.vlm_processor, "tokenizer", None)
                        if tokenizer is not None:
                            vocab = tokenizer.get_vocab()
                            if "<video>" in vocab and "<video>" not in text and "<image>" not in text:
                                text = f"<video>\n{text}"
                            if "<obs>" in vocab and "<obs>" not in text:
                                if "<video>" in text:
                                    text = text.replace("<video>\n", "<video><obs>\n", 1)
                                else:
                                    text = f"<obs>\n{text}"
                        try:
                            max_len = self.vlm_max_text_len if self.vlm_truncation else None
                            inputs = self.vlm_processor(
                                text=text,
                                videos=frames,
                                return_tensors="pt",
                                padding=self.vlm_padding,
                                truncation=self.vlm_truncation,
                                max_length=max_len,
                            )
                        except TypeError:
                            print('using image processor instead of video processor')
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
                    next_robot_obs = torch.stack([f["robot_obs"] for f in next_clip], dim=0)
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
            ep_id = _extract_episode_id(key)

            if current_ep is None:
                current_ep = ep_id

            if ep_id != current_ep:
                yield from flush_buffer()
                buffer = []
                current_ep = ep_id

            image = sample["image.png"]
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


def _collate_sequence_batch(batch):
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
        out["next_robot_obs"] = torch.stack([b["next_robot_obs"] for b in batch], dim=0)
        out["next_adj"] = torch.stack([b["next_adj"] for b in batch], dim=0)
    if "returns" in batch[0]:
        out["returns"] = torch.stack([b["returns"] for b in batch], dim=0).view(-1)
    return out


def _get_first_present(sample, keys, default=None):
    for k in keys:
        if k in sample:
            return sample[k]
    return default


def _normalize_hf_sample(sample):
    out = dict(sample)
    aliases = {
        "image": "image.png",
        "obs": "obs.npy",
        "state": "state.npy",
        "edge_index": "edge_index.npy",
        "text_emb": "text_emb.npy",
        "caption": "caption.txt",
        "rewards": "rewards.npy",
        "dones": "dones.npy",
    }
    for src, dst in aliases.items():
        if src in out and dst not in out:
            out[dst] = out[src]

    # Handle WebDataset-like flattened keys, e.g. "traj_011_step_0085.image.png".
    for k in list(out.keys()):
        if not isinstance(k, str):
            continue
        for suffix, dst in KEY_SUFFIX_ALIASES.items():
            if k.endswith(suffix):
                if dst not in out:
                    out[dst] = out[k]
                if "__key__" not in out:
                    out["__key__"] = k[: -len(suffix)]
                break
    return out


def _path_to_canonical_key(path):
    base = os.path.basename(str(path))
    for suffix, canonical in KEY_SUFFIX_ALIASES.items():
        if base.endswith(suffix):
            return base[: -len(suffix)], canonical
    return None, None


def _extract_episode_id(key):
    key = str(key)
    # New dataset style: traj_011_step_0085 -> episode is traj_011
    if "_step_" in key:
        return key.split("_step_", 1)[0]
    # Legacy style: 000011_000044 -> episode is 000011
    if "_" in key:
        return key.rsplit("_", 1)[0]
    return key


def _ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        b = image.get("bytes", None)
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = image.get("path", None)
        if p:
            return Image.open(p).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    try:
        import numpy as np
        arr = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception as e:
        raise ValueError(f"Unable to convert image sample to PIL format. type={type(image)}") from e


def webdataset_loader(args, shards, batch_size, num_workers):
    vlm_processor = None
    if args.preprocess_in_loader:
        from transformers import LlavaNextVideoProcessor
        vlm_processor = LlavaNextVideoProcessor.from_pretrained(args.vl_model_name)
        tokenizer = getattr(vlm_processor, "tokenizer", None)
        if tokenizer is not None and "<obs>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})

    dataset = SequenceWebDataset(
        shards=shards,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        text_mode=args.text_mode,
        robot_source=args.robot_source,
        reward_reduce=args.reward_reduce,
        done_reduce=args.done_reduce,
        vlm_processor=vlm_processor,
        text_prompt_template=args.text_prompt_template,
        return_mode=args.return_mode,
        n_step=args.n_step,
        gamma=args.gamma,
        keep_raw_video=False,
        include_next=(args.loss_type != "contrastive" and args.return_mode == "td"),
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=(args.vl_backend != "llava_video"),
        vlm_padding=("longest" if args.vl_backend == "llava_video" else "max_length"),
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_sequence_batch)


class SequenceHFDataset(SequenceWebDataset):
    def __init__(self, hf_dataset, hf_config, hf_split, hf_streaming, **kwargs):
        # Base class expects `shards`; HF backend does not use it.
        kwargs = dict(kwargs)
        kwargs.setdefault("shards", "")
        super().__init__(**kwargs)
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self.hf_split = hf_split
        self.hf_streaming = hf_streaming

    def __iter__(self):
        if load_dataset is None:
            raise RuntimeError(
                "Hugging Face dataset loading requested but `datasets.load_dataset` could not be imported. "
                "This is commonly caused by a local `datasets.py` shadowing the package. "
                f"Import errors: {_load_dataset_import_err}"
            )

        ds_kwargs = {}
        if self.hf_config:
            ds_kwargs["name"] = self.hf_config
        dataset = load_dataset(self.hf_dataset, split=self.hf_split, streaming=self.hf_streaming, **ds_kwargs)
        if not self.hf_streaming and hasattr(dataset, "to_iterable_dataset"):
            dataset = dataset.to_iterable_dataset()

        current_ep = None
        buffer = []
        partial_frames = {}

        def _meta_episode_id(meta):
            if not isinstance(meta, dict):
                return "episode"
            ep = _get_first_present(meta, ["episode_id", "traj_id", "trajectory_id", "episode", "id"], default="")
            return str(ep) if ep != "" else "episode"

        def _extract_indexed_value(v, i, total):
            if isinstance(v, (list, tuple)):
                return v[i] if i < len(v) else None
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == total:
                return v[i]
            if hasattr(v, "shape") and len(getattr(v, "shape", [])) > 0 and v.shape[0] == total:
                try:
                    return v[i]
                except Exception:
                    return v
            return v

        def _expand_episode_row(raw):
            # Some HF exports store one row per episode: {"trajectory": ..., "metadata": ...}
            if not isinstance(raw, dict):
                yield raw
                return
            traj = raw.get("trajectory", None)
            meta = raw.get("metadata", None)
            if traj is None:
                yield raw
                return

            ep = _meta_episode_id(meta)
            if isinstance(traj, list):
                for i, step in enumerate(traj):
                    if not isinstance(step, dict):
                        continue
                    frame = dict(step)
                    if "__key__" not in frame:
                        frm = _get_first_present(frame, ["frame_id", "step", "timestep"], default=i)
                        frame["__key__"] = f"{ep}_{frm}"
                    yield frame
                return

            if isinstance(traj, dict):
                lengths = []
                for v in traj.values():
                    if isinstance(v, (list, tuple)):
                        lengths.append(len(v))
                    elif torch.is_tensor(v) and v.dim() > 0:
                        lengths.append(v.shape[0])
                    elif hasattr(v, "shape") and len(getattr(v, "shape", [])) > 0:
                        lengths.append(v.shape[0])
                if not lengths:
                    yield raw
                    return
                t = min(lengths)
                for i in range(t):
                    frame = {}
                    for k, v in traj.items():
                        frame[k] = _extract_indexed_value(v, i, t)
                    if "__key__" not in frame:
                        frame["__key__"] = f"{ep}_{i}"
                    yield frame
                return

            yield raw

        def _is_complete_frame(sample_dict):
            required = {"image.png", f"{self.robot_source}.npy", "edge_index.npy", "rewards.npy", "dones.npy"}
            if self.text_mode == "emb":
                required.add("text_emb.npy")
            return required.issubset(set(sample_dict.keys()))

        def flush_buffer():
            min_len = self.clip_len + (1 if self.include_next else 0)
            if len(buffer) < min_len:
                return
            max_i = len(buffer) - self.clip_len - (1 if self.include_next else 0)
            start_idxs = list(range(0, max_i + 1, self.clip_stride))
            if len(start_idxs) > 1:
                perm = torch.randperm(len(start_idxs)).tolist()
                start_idxs = [start_idxs[p] for p in perm]

            for i in start_idxs:
                clip = self._apply_done_termination(buffer[i : i + self.clip_len])
                raw_video = [f["image"] for f in clip]
                raw_next_video = None
                if self.include_next:
                    next_clip = self._apply_done_termination(buffer[i + 1 : i + 1 + self.clip_len])
                    raw_next_video = [f["image"] for f in next_clip]
                if self.vlm_processor is not None:
                    def _proc(frames, text):
                        if not isinstance(text, str):
                            text = self.text_prompt_template
                        tokenizer = getattr(self.vlm_processor, "tokenizer", None)
                        if tokenizer is not None:
                            vocab = tokenizer.get_vocab()
                            if "<video>" in vocab and "<video>" not in text and "<image>" not in text:
                                text = f"<video>\n{text}"
                            if "<obs>" in vocab and "<obs>" not in text:
                                if "<video>" in text:
                                    text = text.replace("<video>\n", "<video><obs>\n", 1)
                                else:
                                    text = f"<obs>\n{text}"
                        try:
                            max_len = self.vlm_max_text_len if self.vlm_truncation else None
                            inputs = self.vlm_processor(
                                text=text,
                                videos=frames,
                                return_tensors="pt",
                                padding=self.vlm_padding,
                                truncation=self.vlm_truncation,
                                max_length=max_len,
                            )
                        except TypeError:
                            inputs = self.vlm_processor(images=frames, return_tensors="pt")
                        packed = {}
                        for k, v in dict(inputs).items():
                            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                                v = v.squeeze(0)
                            packed[k] = v
                        return packed

                    text = clip[0]["text"]
                    inputs = _proc(raw_video, text)
                    next_inputs = _proc(raw_next_video, text) if self.include_next else None
                else:
                    raise RuntimeError("Dataloader proprocessor not set.")

                robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
                adj = torch.stack([f["adj"] for f in clip], dim=0)
                if self.include_next:
                    next_robot_obs = torch.stack([f["robot_obs"] for f in next_clip], dim=0)
                    next_adj = torch.stack([f["adj"] for f in next_clip], dim=0)

                reward = clip[-1]["reward"]
                done = clip[-1]["done"]
                returns = torch.stack([f["reward"] for f in clip], dim=0).sum(dim=0)
                out = {
                    "inputs": inputs,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "reward": reward.view(1),
                    "returns": returns.view(1),
                    "done": done.view(1),
                }
                if self.include_next:
                    out["next_inputs"] = next_inputs
                    out["next_robot_obs"] = next_robot_obs
                    out["next_adj"] = next_adj
                yield out

        for raw_sample in dataset:
            for sample in _expand_episode_row(raw_sample):
                sample = _normalize_hf_sample(sample)
                if _get_first_present(sample, ["image.png", "image"]) is None:
                    p = _get_first_present(sample, ["path", "filepath", "file_path", "filename", "file_name"])
                    b = _get_first_present(sample, ["bytes", "content", "data"])
                    if p is not None and b is not None:
                        frame_key, canonical_key = _path_to_canonical_key(p)
                        if frame_key is not None and canonical_key is not None:
                            assembled = partial_frames.setdefault(frame_key, {"__key__": frame_key})
                            assembled[canonical_key] = b
                            if _is_complete_frame(assembled):
                                sample = assembled
                                del partial_frames[frame_key]
                            else:
                                continue

                key = _get_first_present(sample, ["__key__", "key", "id"], default="")
                if isinstance(key, bytes):
                    key = key.decode("utf-8", errors="ignore")
                if not key:
                    ep = _get_first_present(sample, ["episode_id", "episode", "traj_id"], default="")
                    frm = _get_first_present(sample, ["frame_id", "timestep", "step"], default="")
                    if ep != "":
                        key = f"{ep}_{frm}" if frm != "" else str(ep)
                ep_id = _extract_episode_id(key)

                if current_ep is None:
                    current_ep = ep_id
                if ep_id != current_ep:
                    yield from flush_buffer()
                    buffer = []
                    current_ep = ep_id

                image = _get_first_present(sample, ["image.png", "image"])
                if image is None:
                    raise KeyError(
                        f"Hugging Face sample is missing image field. keys={list(sample.keys())[:20]}"
                    )
                image = _ensure_pil_image(image)

                robot_raw = _get_first_present(sample, [f"{self.robot_source}.npy", self.robot_source])
                if robot_raw is None:
                    raise KeyError(f"Hugging Face sample missing robot field for --robot_source={self.robot_source}")
                robot_src = _as_numpy(robot_raw)
                if hasattr(robot_src, "numpy"):
                    robot_src = robot_src.numpy()
                robot_obs = torch.tensor(robot_src, dtype=torch.float32)

                num_nodes = robot_obs.shape[0]
                edge_raw = _get_first_present(sample, ["edge_index.npy", "edge_index"])
                if edge_raw is None:
                    raise KeyError("Hugging Face sample missing edge_index field")
                adj = _edge_index_to_adj(_as_numpy(edge_raw), num_nodes)

                if self.text_mode == "raw":
                    text = self.text_prompt_template
                else:
                    text_raw = _get_first_present(sample, ["text_emb.npy", "text_emb"])
                    if text_raw is None:
                        raise KeyError("text_mode=emb requires `text_emb` in Hugging Face dataset samples.")
                    text = _as_numpy(text_raw)
                    if hasattr(text, "numpy"):
                        text = text.numpy()
                    text = torch.tensor(text, dtype=torch.float32)

                rewards_raw = _get_first_present(sample, ["rewards.npy", "rewards"])
                dones_raw = _get_first_present(sample, ["dones.npy", "dones"])
                if rewards_raw is None or dones_raw is None:
                    raise KeyError("Hugging Face sample missing rewards/dones fields")
                reward = _reduce_value(torch.tensor(_as_numpy(rewards_raw), dtype=torch.float32), self.reward_reduce)
                done = _reduce_done(torch.tensor(_as_numpy(dones_raw), dtype=torch.float32), self.done_reduce)

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


def huggingface_loader(args, batch_size, num_workers, split):
    vlm_processor = None
    if args.preprocess_in_loader:
        from transformers import LlavaNextVideoProcessor
        vlm_processor = LlavaNextVideoProcessor.from_pretrained(args.vl_model_name)
        tokenizer = getattr(vlm_processor, "tokenizer", None)
        if tokenizer is not None and "<obs>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})

    dataset = SequenceHFDataset(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=split,
        hf_streaming=args.hf_streaming,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        text_mode=args.text_mode,
        robot_source=args.robot_source,
        reward_reduce=args.reward_reduce,
        done_reduce=args.done_reduce,
        vlm_processor=vlm_processor,
        text_prompt_template=args.text_prompt_template,
        return_mode=args.return_mode,
        n_step=args.n_step,
        gamma=args.gamma,
        keep_raw_video=False,
        include_next=(args.loss_type != "contrastive" and args.return_mode == "td"),
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=(args.vl_backend != "llava_video"),
        vlm_padding=("longest" if args.vl_backend == "llava_video" else "max_length"),
    )
    worker_count = num_workers if args.hf_streaming else 0
    return DataLoader(dataset, batch_size=batch_size, num_workers=worker_count, collate_fn=_collate_sequence_batch)


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


def run_epoch(model, loader, optimizer, accelerator, log_every, gamma, args, train=True, global_step=0):
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

        if train and args.debug_save_video and not getattr(run_epoch, "_debug_saved", False):
            _save_debug_video(batch, args, accelerator, tag="train")
            run_epoch._debug_saved = True

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                pred = model(
                    inputs,
                    robot_obs,
                    adj)
                if args.loss_type == "contrastive":
                    # Use returns if available, otherwise use per-clip reward.
                    if args.return_mode == "nstep" and "returns" in batch:
                        returns = batch["returns"].to(accelerator.device)
                    else:
                        returns = reward
                    loss = _contrastive_pairwise_loss(pred.view(-1), returns.view(-1), margin=args.contrastive_margin)
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
        if train:
            accelerator.log(metrics, step=global_step + step)
        else:
            accelerator.log(metrics, step=global_step)

    return avg_loss, (global_step + step if train else global_step)


def main():
    args = parse_args()
    if args.data_backend == "webdataset" and not args.train_shards:
        raise ValueError("data_backend=webdataset requires --train_shards.")
    if args.data_backend == "huggingface" and not args.hf_dataset:
        raise ValueError("data_backend=huggingface requires --hf_dataset.")
    os.makedirs(args.save_dir, exist_ok=True)

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
                raise RuntimeError("FSDP CPU offload requested but torch.distributed.fsdp is unavailable.")
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

    ddp_kwargs = None
    if not args.fsdp and DistributedDataParallelKwargs is not None:
        find_unused = args.ddp_find_unused_parameters or (args.peft != "none") or args.freeze_vl
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)

    accelerator_kwargs = dict(
        mixed_precision=args.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else [],
    )
    if args.wandb:
        accelerator_kwargs["log_with"] = ["wandb"]
    accelerator = Accelerator(**accelerator_kwargs)
    if args.wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_kwargs = {}
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        if tags:
            wandb_kwargs["tags"] = tags
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": wandb_kwargs},
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.data_backend == "webdataset":
        train_loader = webdataset_loader(args, args.train_shards, args.batch_size, args.num_workers)
        val_loader = None
        if args.val_shards:
            val_loader = webdataset_loader(args, args.val_shards, args.batch_size, args.num_workers)
    else:
        train_loader = huggingface_loader(args, args.batch_size, args.num_workers, args.hf_train_split)
        val_loader = None
        if args.hf_val_split:
            val_loader = huggingface_loader(args, args.batch_size, args.num_workers, args.hf_val_split)

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
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
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt"))
        accelerator.wait_for_everyone()

    if args.wandb and hasattr(accelerator, "end_training"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
