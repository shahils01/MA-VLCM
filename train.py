import argparse
import os
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

try:
    import webdataset as wds
except Exception:
    wds = None

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

    # Sequence building
    p.add_argument("--clip_len", type=int, default=8)
    p.add_argument("--clip_stride", type=int, default=1)
    p.add_argument("--robot_source", type=str, default="obs", choices=["obs", "state"])
    p.add_argument("--value_source", type=str, default="rewards", choices=["rewards", "state"])
    p.add_argument("--value_reduce", type=str, default="mean", choices=["mean", "sum", "first"])
    p.add_argument("--value_time", type=str, default="last", choices=["last", "mean"])

    # DeepSeek VLM backbone
    p.add_argument("--vl_backend", type=str, default="deepseek_vl", choices=["deepseek_vl", "deepseek_vl2"])
    p.add_argument("--vl_model_name", type=str, default="deepseek-community/deepseek-vl-1.3b-base")
    p.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--vl_max_text_len", type=int, default=256)
    p.add_argument("--freeze_vl", action="store_true")

    # Video
    p.add_argument("--video_channels", type=int, default=3)
    p.add_argument("--video_height", type=int, default=224)
    p.add_argument("--video_width", type=int, default=224)
    p.add_argument("--video_frames", type=int, default=8)
    p.add_argument("--video_preprocessed", action="store_true")
    p.add_argument("--video_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--video_std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    # Robots / graph
    p.add_argument("--num_robots", type=int, default=8)
    p.add_argument("--robot_obs_dim", type=int, default=16)

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
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="checkpoints")

    return p.parse_args()


def build_model(args):
    cfg = ModelConfig(
        vl_backend=args.vl_backend,
        vl_model_name=args.vl_model_name,
        vl_dtype=args.vl_dtype,
        vl_max_text_len=args.vl_max_text_len,
        freeze_vl=args.freeze_vl,
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
    )
    return MultimodalValueModel(cfg, device=torch.device(args.device))


def _as_numpy(x):
    if isinstance(x, bytes):
        # try numpy load from bytes
        import numpy as np
        try:
            return np.load(io.BytesIO(x), allow_pickle=True)
        except Exception:
            # try torch load fallback
            import torch
            return torch.load(io.BytesIO(x), map_location="cpu")
    return x


def _pil_to_tensor(img):
    # img: PIL.Image
    import numpy as np
    arr = np.array(img, copy=False)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).float() / 255.0


def _edge_index_to_adj(edge_index, num_nodes):
    # edge_index: [2, E] possibly with -1 paddings
    edge_index = _as_numpy(edge_index)
    if hasattr(edge_index, "shape") and len(edge_index.shape) == 2 and edge_index.shape[0] == num_nodes and edge_index.shape[1] == num_nodes:
        adj = torch.from_numpy(edge_index).float()
        return adj
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


def _value_from_frame(sample, value_source, reduce_mode):
    arr = _as_numpy(sample[f"{value_source}.npy"])
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    t = torch.tensor(arr, dtype=torch.float32)
    return _reduce_value(t, reduce_mode)


class SequenceWebDataset(IterableDataset):
    def __init__(
        self,
        shards,
        clip_len,
        clip_stride,
        text_mode,
        robot_source,
        value_source,
        value_reduce,
        value_time,
    ):
        self.shards = shards
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.text_mode = text_mode
        self.robot_source = robot_source
        self.value_source = value_source
        self.value_reduce = value_reduce
        self.value_time = value_time

    def __iter__(self):
        if wds is None:
            raise RuntimeError("webdataset is not installed.")

        dataset = wds.WebDataset(self.shards, shardshuffle=False).decode("pil")

        current_ep = None
        buffer = []

        def flush_buffer():
            if len(buffer) < self.clip_len:
                return
            for i in range(0, len(buffer) - self.clip_len + 1, self.clip_stride):
                clip = buffer[i : i + self.clip_len]

                video = torch.stack([f["image"] for f in clip], dim=0)
                robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
                adj = torch.stack([f["adj"] for f in clip], dim=0)
                text = clip[0]["text"]

                if self.value_time == "mean":
                    value = torch.stack([f["value"] for f in clip]).mean()
                else:
                    value = clip[-1]["value"]

                out = {
                    "video": video,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "value": value.view(1),
                }
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

            image = _pil_to_tensor(sample["image.png"])
            robot_src = _as_numpy(sample[f"{self.robot_source}.npy"])
            if hasattr(robot_src, "numpy"):
                robot_src = robot_src.numpy()
            robot_obs = torch.tensor(robot_src, dtype=torch.float32)

            num_nodes = robot_obs.shape[0]
            edge_index = _as_numpy(sample["edge_index.npy"])
            adj = _edge_index_to_adj(edge_index, num_nodes)

            if self.text_mode == "raw":
                text = sample["caption.txt"]
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="ignore")
            else:
                text = _as_numpy(sample["text_emb.npy"])
                if hasattr(text, "numpy"):
                    text = text.numpy()
                text = torch.tensor(text, dtype=torch.float32)

            value = _value_from_frame(sample, self.value_source, self.value_reduce)

            buffer.append(
                {
                    "image": image,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "text": text,
                    "value": value,
                }
            )

        # flush last episode
        yield from flush_buffer()


def webdataset_loader(args, shards, batch_size, num_workers):
    dataset = SequenceWebDataset(
        shards=shards,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        text_mode=args.text_mode,
        robot_source=args.robot_source,
        value_source=args.value_source,
        value_reduce=args.value_reduce,
        value_time=args.value_time,
    )

    def _collate(batch):
        video = torch.stack([b["video"] for b in batch], dim=0)
        robot_obs = torch.stack([b["robot_obs"] for b in batch], dim=0)
        adj = torch.stack([b["adj"] for b in batch], dim=0)
        value = torch.stack([b["value"] for b in batch], dim=0).view(-1)
        out = {"video": video, "robot_obs": robot_obs, "adj": adj, "value": value}
        if args.text_mode == "raw":
            out["text_raw"] = [b["text_raw"] for b in batch]
        else:
            out["text_emb"] = torch.stack([b["text_emb"] for b in batch], dim=0)
        return out

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)


def run_epoch(model, loader, optimizer, device, log_every, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.MSELoss()
    total_loss = 0.0
    step = 0

    for batch in loader:
        step += 1
        video = batch["video"].to(device)
        robot_obs = batch["robot_obs"].to(device)
        adj = batch["adj"].to(device)
        text_emb = batch.get("text_emb", None)
        text_raw = batch.get("text_raw", None)
        if text_emb is not None:
            text_emb = text_emb.to(device)
        value = batch["value"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(video, robot_obs, adj, text_emb=text_emb, text_raw=text_raw)
            loss = loss_fn(pred, value)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        if log_every > 0 and step % log_every == 0:
            avg = total_loss / step
            phase = "train" if train else "val"
            print(f"{phase} step={step} loss={avg:.4f}")

    return total_loss / max(step, 1)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model = build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = webdataset_loader(args, args.train_shards, args.batch_size, args.num_workers)
    val_loader = None
    if args.val_shards:
        val_loader = webdataset_loader(args, args.val_shards, args.batch_size, args.num_workers)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device, args.log_every, train=True)
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, optimizer, args.device, args.log_every, train=False)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss if val_loss is not None else 'n/a'}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch_{epoch}.pt"))


if __name__ == "__main__":
    main()
