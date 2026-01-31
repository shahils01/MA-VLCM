import argparse
import os
import io
import functools
import inspect

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
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
    p.add_argument("--train_shards", type=str, required=True, help="WebDataset shard pattern for training")
    p.add_argument("--val_shards", type=str, default="", help="Optional WebDataset shard pattern for validation")
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
    p.add_argument("--loss_type", type=str, default="td", choices=["td", "contrastive"])
    p.add_argument("--contrastive_margin", type=float, default=0.0)

    # Accelerate
    p.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--fsdp", action="store_true", help="Use FSDP to shard model parameters across GPUs")
    p.add_argument("--fsdp_min_num_params", type=int, default=1_000_000, help="Auto-wrap threshold for FSDP")
    p.add_argument("--fsdp_cpu_offload", action="store_true", help="Offload FSDP parameters to CPU when not in use")
    p.add_argument("--fsdp_use_orig_params", action="store_true", help="Use FSDP use_orig_params to allow mixed requires_grad")
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
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="checkpoints")

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


def _compute_n_step_return(buffer, start_idx, n_step, gamma):
    r0 = buffer[start_idx]["reward"]
    ret = torch.zeros_like(r0)
    discount = 1.0
    for k in range(n_step):
        idx = start_idx + k
        if idx >= len(buffer):
            break
        r = buffer[idx]["reward"]
        ret = ret + discount * r
        done = buffer[idx]["done"]
        done_flag = bool(done.item() if torch.is_tensor(done) else done)
        if done_flag:
            break
        discount *= gamma
    return ret


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
        image_processor=None,
        text_prompt_template=None,
        return_mode="td",
        n_step=50,
        gamma=0.99,
        keep_raw_video=False,
    ):
        self.shards = shards
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.text_mode = text_mode
        self.robot_source = robot_source
        self.reward_reduce = reward_reduce
        self.done_reduce = done_reduce
        self.image_processor = image_processor
        self.text_prompt_template = text_prompt_template
        self.return_mode = return_mode
        self.n_step = n_step
        self.gamma = gamma
        self.keep_raw_video = keep_raw_video

    def __iter__(self):
        if wds is None:
            raise RuntimeError("webdataset is not installed.")

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
            if len(buffer) < self.clip_len + 1:
                return
            max_i = len(buffer) - self.clip_len - 1
            for i in range(0, max_i + 1, self.clip_stride):
                clip = buffer[i : i + self.clip_len]
                next_clip = buffer[i + 1 : i + 1 + self.clip_len]

                raw_video = [f["image"] for f in clip]
                raw_next_video = [f["image"] for f in next_clip]
                if self.image_processor is not None:
                    def _proc(frames):
                        try:
                            proc = self.image_processor(videos=frames, return_tensors="pt")
                        except TypeError:
                            proc = self.image_processor(images=frames, return_tensors="pt")
                        if "pixel_values" in proc:
                            pixel_values = proc["pixel_values"]
                        elif "video_values" in proc:
                            pixel_values = proc["video_values"]
                        elif "pixel_values_videos" in proc:
                            pixel_values = proc["pixel_values_videos"]
                        elif hasattr(proc, "pixel_values"):
                            pixel_values = proc.pixel_values
                        elif hasattr(proc, "video_values"):
                            pixel_values = proc.video_values
                        else:
                            raise KeyError("image processor output has no pixel_values/video_values")
                        image_sizes = proc.get("image_sizes", None)
                        return pixel_values, image_sizes

                    video, image_sizes = _proc(raw_video)
                    next_video, next_image_sizes = _proc(raw_next_video)
                else:
                    video = raw_video
                    next_video = raw_next_video

                robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
                next_robot_obs = torch.stack([f["robot_obs"] for f in next_clip], dim=0)

                adj = torch.stack([f["adj"] for f in clip], dim=0)
                next_adj = torch.stack([f["adj"] for f in next_clip], dim=0)

                text = clip[0]["text"]

                reward = clip[-1]["reward"]
                done = clip[-1]["done"]
                n_step_return = None
                if self.return_mode == "nstep":
                    start_idx = i + self.clip_len - 1
                    n_step_return = _compute_n_step_return(buffer, start_idx, self.n_step, self.gamma)

                out = {
                    "video": video,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "next_video": next_video,
                    "next_robot_obs": next_robot_obs,
                    "next_adj": next_adj,
                    "reward": reward.view(1),
                    "done": done.view(1),
                }
                if self.image_processor is not None and image_sizes is not None:
                    out["image_sizes"] = image_sizes
                if self.image_processor is not None and next_image_sizes is not None:
                    out["next_image_sizes"] = next_image_sizes
                if self.keep_raw_video:
                    out["raw_video"] = raw_video
                    out["raw_next_video"] = raw_next_video
                if n_step_return is not None:
                    out["return"] = n_step_return.view(1)
                if self.text_mode == "raw":
                    out["text_raw"] = text
                else:
                    out["text_emb"] = text
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


def webdataset_loader(args, shards, batch_size, num_workers):
    image_processor = None
    text_tokenizer = None
    if args.preprocess_in_loader:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.vl_model_name)
        image_processor = getattr(proc, "video_processor", None) or getattr(
            proc, "image_processor", None
        ) or getattr(proc, "vision_processor", None)
        text_tokenizer = getattr(proc, "tokenizer", None)
        
        if image_processor is None:
            raise RuntimeError("Could not find image processor for the LLaVA-NeXT-Video FM.")
        if text_tokenizer is None:
            raise RuntimeError("Could not find text tokenizerr for the sLLaVA-NeXT-Video FM.")

    # if args.text_mode == "raw" and text_tokenizer is None:
    #     from transformers import AutoTokenizer
    #     text_tokenizer = AutoTokenizer.from_pretrained(args.vl_model_name)

    dataset = SequenceWebDataset(
        shards=shards,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        text_mode=args.text_mode,
        robot_source=args.robot_source,
        reward_reduce=args.reward_reduce,
        done_reduce=args.done_reduce,
        image_processor=image_processor,
        text_prompt_template=args.text_prompt_template,
        return_mode=args.return_mode,
        n_step=args.n_step,
        gamma=args.gamma,
        keep_raw_video=False,
    )

    def _collate(batch):
        if not torch.is_tensor(batch[0]["video"]):
            raise RuntimeError(
                "video is not a tensor. Use --preprocess_in_loader to convert frames to tensors "
                "or precompute video tensors in the dataset."
            )
        if not torch.is_tensor(batch[0]["next_video"]):
            raise RuntimeError(
                "next_video is not a tensor. Use --preprocess_in_loader to convert frames to tensors "
                "or precompute video tensors in the dataset."
            )
        out = {
            "video": torch.stack([b["video"] for b in batch], dim=0),
            "robot_obs": torch.stack([b["robot_obs"] for b in batch], dim=0),
            "adj": torch.stack([b["adj"] for b in batch], dim=0),
            "next_video": torch.stack([b["next_video"] for b in batch], dim=0),
            "next_robot_obs": torch.stack([b["next_robot_obs"] for b in batch], dim=0),
            "next_adj": torch.stack([b["next_adj"] for b in batch], dim=0),
            "reward": torch.stack([b["reward"] for b in batch], dim=0).view(-1),
            "done": torch.stack([b["done"] for b in batch], dim=0).view(-1),
        }
        if "image_sizes" in batch[0]:
            out["image_sizes"] = [b["image_sizes"] for b in batch]
        if "next_image_sizes" in batch[0]:
            out["next_image_sizes"] = [b["next_image_sizes"] for b in batch]
        if args.return_mode == "nstep":
            out["return"] = torch.stack([b["return"] for b in batch], dim=0).view(-1)
        if args.text_mode == "raw":
            texts = [b["text_raw"] for b in batch]
            if args.vl_backend == "llava_video":
                token = None
                vocab = text_tokenizer.get_vocab()
                if "<video>" in vocab:
                    token = "<video>"
                elif "<image>" in vocab:
                    token = "<image>"
                else:
                    for t in getattr(text_tokenizer, "additional_special_tokens", []) or []:
                        if "image" in t or "video" in t:
                            token = t
                            break
                if token is not None:
                    texts = [f"{token}\n{t}" for t in texts]
            tokens = text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.vl_max_text_len,
                return_tensors="pt",
            )
            out["text_ids"] = tokens["input_ids"]
            out["text_mask"] = tokens["attention_mask"]
        else:
            out["text_emb"] = torch.stack([b["text_emb"] for b in batch], dim=0)
        return out

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)


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


def run_epoch(model, loader, optimizer, accelerator, log_every, gamma, args, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0.0
    step = 0

    for batch in loader:
        step += 1
        video = batch["video"]
        robot_obs = batch["robot_obs"].to(accelerator.device)
        adj = batch["adj"].to(accelerator.device)
        reward = batch["reward"].to(accelerator.device)
        done = batch["done"].to(accelerator.device).float()
        image_sizes = batch.get("image_sizes", None)
        next_image_sizes = batch.get("next_image_sizes", None)
        if args.return_mode == "td":
            next_video = batch["next_video"]
            next_robot_obs = batch["next_robot_obs"].to(accelerator.device)
            next_adj = batch["next_adj"].to(accelerator.device)
        else:
            next_video = None
            next_robot_obs = None
            next_adj = None

        text_emb = batch.get("text_emb", None)
        text_raw = batch.get("text_raw", None)
        text_ids = batch.get("text_ids", None)
        text_mask = batch.get("text_mask", None)
        if text_emb is not None:
            text_emb = text_emb.to(accelerator.device)
        if text_ids is not None:
            text_ids = text_ids.to(accelerator.device)
        if text_mask is not None:
            text_mask = text_mask.to(accelerator.device)

        if train and args.debug_save_video and not getattr(run_epoch, "_debug_saved", False):
            _save_debug_video(batch, args, accelerator, tag="train")
            run_epoch._debug_saved = True

        with accelerator.accumulate(model):
            with torch.set_grad_enabled(train):
                pred = model(
                    video,
                    robot_obs,
                    adj,
                    text_raw=text_raw,
                    text_emb=text_emb,
                    text_ids=text_ids.clone() if text_ids is not None else None,
                    text_mask=text_mask.clone() if text_mask is not None else None,
                    image_sizes=image_sizes,
                )
                if args.loss_type == "contrastive":
                    # Use returns if available, otherwise use per-clip reward.
                    if args.return_mode == "nstep" and "return" in batch:
                        returns = batch["return"].to(accelerator.device)
                    else:
                        returns = reward
                    loss = _contrastive_pairwise_loss(pred.view(-1), returns.view(-1), margin=args.contrastive_margin)
                else:
                    if args.return_mode == "td":
                        with torch.no_grad():
                            next_pred = model(
                                next_video,
                                next_robot_obs,
                                next_adj,
                                text_emb=text_emb,
                                text_ids=text_ids.clone() if text_ids is not None else None,
                                text_mask=text_mask.clone() if text_mask is not None else None,
                                image_sizes=next_image_sizes,
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

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
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

    train_loader = webdataset_loader(args, args.train_shards, args.batch_size, args.num_workers)
    val_loader = None
    if args.val_shards:
        val_loader = webdataset_loader(args, args.val_shards, args.batch_size, args.num_workers)

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, accelerator, args.log_every, args.gamma, args, train=True)
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, accelerator, args.log_every, args.gamma, args, train=False)

        accelerator.print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss if val_loss is not None else 'n/a'}")

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
