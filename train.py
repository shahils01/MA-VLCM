import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import webdataset as wds
except Exception as e:
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
    p.add_argument("--text_mode", type=str, default="emb", choices=["emb", "raw"])

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


def webdataset_loader(shards, batch_size, num_workers, text_mode):
    if wds is None:
        raise RuntimeError("webdataset is not installed. Please install it or use a different loader.")

    if text_mode == "raw":
        dataset = (
            wds.WebDataset(shards)
            .decode("torch", "utf-8")
            .to_tuple("video.pth", "robot_obs.pth", "adj.pth", "text.txt", "value.pth")
        )
    else:
        dataset = (
            wds.WebDataset(shards)
            .decode("torch")
            .to_tuple("video.pth", "robot_obs.pth", "adj.pth", "text_emb.pth", "value.pth")
        )

    def _collate(batch):
        video, robot_obs, adj, text_field, value = zip(*batch)
        out = {
            "video": torch.stack(video, dim=0),
            "robot_obs": torch.stack(robot_obs, dim=0),
            "adj": torch.stack(adj, dim=0),
            "value": torch.stack(value, dim=0).view(-1),
        }
        if text_mode == "raw":
            out["text_raw"] = [t if isinstance(t, str) else t.decode("utf-8") for t in text_field]
        else:
            out["text_emb"] = torch.stack(text_field, dim=0)
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

    train_loader = webdataset_loader(args.train_shards, args.batch_size, args.num_workers, args.text_mode)
    val_loader = None
    if args.val_shards:
        val_loader = webdataset_loader(args.val_shards, args.batch_size, args.num_workers, args.text_mode)

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
