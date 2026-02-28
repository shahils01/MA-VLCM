import argparse
import math
from types import SimpleNamespace

import torch

from data_loading import webdataset_loader
from train import _apply_peft, _resolve_vl_model_preset, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MA-VLCM critic predictions against return targets.")
    p.add_argument("--checkpoint", type=str, default="", help="Optional path to checkpoint (.pt) saved by train.py")
    p.add_argument(
        "--skip_checkpoint_weights",
        action="store_true",
        help="If set, do NOT load model weights from checkpoint (uses random init with same checkpoint config).",
    )
    p.add_argument("--eval_shards", type=str, default="", help="Shard pattern for general eval set")
    p.add_argument("--good_shards", type=str, default="", help="Optional shard pattern containing good trajectories")
    p.add_argument("--bad_shards", type=str, default="", help="Optional shard pattern containing bad trajectories")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=512, help="Max samples to evaluate per dataset stream")
    p.add_argument("--print_samples", type=int, default=40, help="How many random pred-vs-return rows to print")
    p.add_argument("--seed", type=int, default=0)
    # Fallback config (used when --checkpoint is not provided).
    p.add_argument("--clip_len", type=int, default=20)
    p.add_argument("--clip_stride", type=int, default=20)
    p.add_argument("--text_mode", type=str, default="raw", choices=["raw", "emb"])
    p.add_argument("--robot_source", type=str, default="obs", choices=["obs", "state"])
    p.add_argument("--reward_reduce", type=str, default="mean", choices=["mean", "sum", "first"])
    p.add_argument("--done_reduce", type=str, default="any", choices=["any", "all", "mean", "sum", "first"])
    p.add_argument("--preprocess_in_loader", action="store_true", default=True)
    p.add_argument("--vl_max_text_len", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--return_mode", type=str, default="nstep", choices=["td", "nstep"])
    p.add_argument("--return_horizon", type=str, default="clip", choices=["clip", "trajectory"])
    p.add_argument("--loss_type", type=str, default="contrastive", choices=["td", "contrastive"])
    p.add_argument("--n_step", type=int, default=50)
    p.add_argument("--vl_backend", type=str, default="llava_video", choices=["deepseek_vl", "deepseek_vl2", "llava_video"])
    p.add_argument("--vl_model_name", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    p.add_argument(
        "--vl_model_preset",
        type=str,
        default="llava_onevision_0p5b",
        choices=["custom", "llava_next_video_7b", "llava_onevision_0p5b"],
    )
    p.add_argument("--vl_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--freeze_vl", action="store_true")
    p.add_argument("--value_pooling", type=str, default="last_token_logits", choices=["last_token_logits", "hidden_mean"])
    p.add_argument("--vl_logits_to_keep", type=int, default=1)
    p.add_argument("--video_channels", type=int, default=3)
    p.add_argument("--video_height", type=int, default=224)
    p.add_argument("--video_width", type=int, default=224)
    p.add_argument("--video_frames", type=int, default=100)
    p.add_argument("--video_preprocessed", action="store_true", default=True)
    p.add_argument("--video_mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--video_std", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--num_robots", type=int, default=5)
    p.add_argument("--robot_obs_dim", type=int, default=40)
    p.add_argument("--text_dim", type=int, default=512)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--temporal_layers", type=int, default=2)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--temporal_dropout", type=float, default=0.1)
    p.add_argument("--gnn_layers", type=int, default=2)
    p.add_argument("--fusion_hidden", type=int, default=512)
    p.add_argument("--use_moe", action="store_true")
    p.add_argument("--moe_experts", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--debug_save_video", action="store_true", default=False)
    p.add_argument("--peft", type=str, default="none", choices=["none", "lora", "qlora"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="")
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    return p.parse_args()


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    # Rank transform for Spearman computation; ties are unlikely for continuous predictions.
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(0, x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = torch.sqrt((x_center.pow(2).sum() * y_center.pow(2).sum()).clamp(min=1e-12))
    return float((x_center * y_center).sum() / denom)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _pairwise_ranking_accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    n = int(scores.numel())
    if n < 2:
        return float("nan")
    sdiff = scores[:, None] - scores[None, :]
    tdiff = targets[:, None] - targets[None, :]
    mask = tdiff.ne(0)
    if mask.sum().item() == 0:
        return float("nan")
    correct = (sdiff[mask] * tdiff[mask]).gt(0).float().mean()
    return float(correct)


def _load_train_args(ckpt):
    if "args" not in ckpt:
        raise KeyError("Checkpoint does not contain saved training args under key 'args'.")
    return SimpleNamespace(**ckpt["args"])


def _args_from_cli(cli_args):
    keys = [
        "batch_size",
        "num_workers",
        "clip_len",
        "clip_stride",
        "text_mode",
        "robot_source",
        "reward_reduce",
        "done_reduce",
        "preprocess_in_loader",
        "vl_max_text_len",
        "gamma",
        "return_mode",
        "return_horizon",
        "loss_type",
        "n_step",
        "vl_backend",
        "vl_model_name",
        "vl_model_preset",
        "vl_dtype",
        "freeze_vl",
        "value_pooling",
        "vl_logits_to_keep",
        "video_channels",
        "video_height",
        "video_width",
        "video_frames",
        "video_preprocessed",
        "video_mean",
        "video_std",
        "num_robots",
        "robot_obs_dim",
        "text_dim",
        "d_model",
        "temporal_layers",
        "temporal_heads",
        "temporal_dropout",
        "gnn_layers",
        "fusion_hidden",
        "use_moe",
        "moe_experts",
        "moe_top_k",
        "debug_save_video",
        "peft",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_bias",
    ]
    return SimpleNamespace(**{k: getattr(cli_args, k) for k in keys})


def _init_quant_config_if_needed(args):
    if getattr(args, "peft", "none") != "qlora":
        args.quantization_config = None
        return
    from transformers import BitsAndBytesConfig

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


def _load_checkpoint_state(model, ckpt_state, peft_mode):
    try:
        model.load_state_dict(ckpt_state, strict=True)
        return
    except RuntimeError:
        if peft_mode not in {"lora", "qlora"}:
            raise
    quant_meta_tokens = ("absmax", "quant_map", "quant_state", "bitsandbytes__")
    filtered = {k: v for k, v in ckpt_state.items() if not any(tok in k for tok in quant_meta_tokens)}
    model.load_state_dict(filtered, strict=False)


def _move_inputs_to_device(inputs, device):
    moved = {}
    for k, v in inputs.items():
        moved[k] = v.to(device) if torch.is_tensor(v) else v
    return moved


@torch.no_grad()
def run_stream(model, loader, device, max_samples, train_args):
    preds = []
    rets = []
    aligned_targets = []
    for batch in loader:
        inputs = _move_inputs_to_device(batch["inputs"], device)
        robot_obs = batch["robot_obs"].to(device)
        adj = batch["adj"].to(device)
        out = model(inputs, robot_obs, adj)

        ret = batch["returns"].to(device) if "returns" in batch else None
        target = None
        if getattr(train_args, "loss_type", "contrastive") != "contrastive":
            gamma = float(getattr(train_args, "gamma", 0.99))
            return_mode = getattr(train_args, "return_mode", "td")
            if return_mode == "td" and all(k in batch for k in ("next_inputs", "next_robot_obs", "next_adj", "reward", "done")):
                next_inputs = _move_inputs_to_device(batch["next_inputs"], device)
                next_robot_obs = batch["next_robot_obs"].to(device)
                next_adj = batch["next_adj"].to(device)
                next_pred = model(next_inputs, next_robot_obs, next_adj)
                reward = batch["reward"].to(device)
                done = batch["done"].to(device).float()
                target = reward + gamma * (1.0 - done) * next_pred
            elif return_mode == "nstep" and all(
                k in batch for k in ("td_nstep_inputs", "td_nstep_robot_obs", "td_nstep_adj", "td_nstep_return", "td_nstep_done")
            ):
                nstep_inputs = _move_inputs_to_device(batch["td_nstep_inputs"], device)
                nstep_robot_obs = batch["td_nstep_robot_obs"].to(device)
                nstep_adj = batch["td_nstep_adj"].to(device)
                nstep_bootstrap_pred = model(nstep_inputs, nstep_robot_obs, nstep_adj)
                nstep_returns = batch["td_nstep_return"].to(device)
                nstep_done = batch["td_nstep_done"].to(device).float()
                gamma_n = gamma ** int(getattr(train_args, "n_step", 1))
                target = nstep_returns + gamma_n * (1.0 - nstep_done) * nstep_bootstrap_pred

        preds.append(out.detach().float().cpu())
        if ret is not None:
            rets.append(ret.detach().float().cpu())
        if target is not None:
            aligned_targets.append(target.detach().float().cpu())
        if sum(x.numel() for x in preds) >= max_samples:
            break

    if not preds:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    pred = torch.cat(preds, dim=0)[:max_samples]
    ret = torch.cat(rets, dim=0)[:max_samples] if rets else torch.empty(0)
    aligned = torch.cat(aligned_targets, dim=0)[:max_samples] if aligned_targets else torch.empty(0)
    return pred, ret, aligned


def _print_sample_table(pred: torch.Tensor, target: torch.Tensor, k: int, seed: int, target_name: str):
    if pred.numel() == 0 or target.numel() == 0:
        print("No samples to print.")
        return
    n = min(pred.numel(), target.numel())
    k = min(k, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]
    print(f"\nRandom sample comparison (predicted_value vs {target_name}):")
    print(f"idx\tpred\t{target_name}\terror")
    for i in idx.tolist():
        p = float(pred[i])
        t = float(target[i])
        print(f"{i}\t{p:.6f}\t{t:.6f}\t{(p - t):+.6f}")


def _print_core_metrics(name: str, pred: torch.Tensor, ret: torch.Tensor):
    if pred.numel() == 0:
        print(f"\n[{name}] no samples.")
        return
    err = pred - ret
    mae = float(err.abs().mean())
    rmse = float(torch.sqrt((err.pow(2)).mean()))
    pearson = _pearson(pred, ret)
    spearman = _spearman(pred, ret)
    pair_acc = _pairwise_ranking_accuracy(pred, ret)
    print(f"\n[{name}] N={pred.numel()}")
    print(f"  MAE={mae:.6f}")
    print(f"  RMSE={rmse:.6f}")
    print(f"  Pearson={pearson:.6f}")
    print(f"  Spearman={spearman:.6f}")
    print(f"  PairwiseRankingAcc={pair_acc:.6f}")

    q_low = torch.quantile(ret, 0.30)
    q_high = torch.quantile(ret, 0.70)
    low_mask = ret <= q_low
    high_mask = ret >= q_high
    if low_mask.any() and high_mask.any():
        low_mean = float(pred[low_mask].mean())
        high_mean = float(pred[high_mask].mean())
        sep = high_mean - low_mean
        print(f"  MeanPred(top30% return)={high_mean:.6f}")
        print(f"  MeanPred(bottom30% return)={low_mean:.6f}")
        print(f"  Separation(top-bottom)={sep:+.6f}")


def _good_bad_pair_accuracy(good_scores: torch.Tensor, bad_scores: torch.Tensor) -> float:
    if good_scores.numel() == 0 or bad_scores.numel() == 0:
        return float("nan")
    # Probability that a random good clip is scored higher than a random bad clip.
    cmp = (good_scores[:, None] - bad_scores[None, :]).gt(0).float()
    return float(cmp.mean())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        train_args = _load_train_args(ckpt)
    else:
        train_args = _args_from_cli(args)

    _resolve_vl_model_preset(train_args)
    _init_quant_config_if_needed(train_args)

    # Keep eval stream size configurable from CLI.
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers

    model = build_model(train_args, device=device)
    model = _apply_peft(model, train_args)
    if ckpt is not None and not args.skip_checkpoint_weights:
        _load_checkpoint_state(model, ckpt["model"], getattr(train_args, "peft", "none"))
        print("Evaluation mode: checkpoint-loaded model weights.")
    elif ckpt is not None and args.skip_checkpoint_weights:
        print("Evaluation mode: random initialization using checkpoint architecture/config (weights NOT loaded).")
    else:
        print("Evaluation mode: random initialization from CLI-provided/default architecture.")
    model.eval()
    model.to(device)

    if args.eval_shards:
        eval_loader = webdataset_loader(train_args, args.eval_shards, args.batch_size, args.num_workers)
        pred, ret, aligned = run_stream(model, eval_loader, device, args.max_samples, train_args)
        if ret.numel() > 0:
            _print_sample_table(pred, ret, args.print_samples, args.seed, "true_return")
            _print_core_metrics("eval_shards/returns_target", pred, ret)
        if aligned.numel() > 0:
            _print_sample_table(pred, aligned, args.print_samples, args.seed + 1, "train_aligned_target")
            _print_core_metrics("eval_shards/train_aligned_target", pred, aligned)

    if args.good_shards and args.bad_shards:
        good_loader = webdataset_loader(train_args, args.good_shards, args.batch_size, args.num_workers)
        bad_loader = webdataset_loader(train_args, args.bad_shards, args.batch_size, args.num_workers)
        good_pred, good_ret, good_aligned = run_stream(model, good_loader, device, args.max_samples, train_args)
        bad_pred, bad_ret, bad_aligned = run_stream(model, bad_loader, device, args.max_samples, train_args)
        if good_ret.numel() > 0:
            _print_core_metrics("good_shards/returns_target", good_pred, good_ret)
        if bad_ret.numel() > 0:
            _print_core_metrics("bad_shards/returns_target", bad_pred, bad_ret)
        if good_aligned.numel() > 0:
            _print_core_metrics("good_shards/train_aligned_target", good_pred, good_aligned)
        if bad_aligned.numel() > 0:
            _print_core_metrics("bad_shards/train_aligned_target", bad_pred, bad_aligned)
        gb_acc = _good_bad_pair_accuracy(good_pred, bad_pred)
        print("\n[good_vs_bad]")
        print(f"  PairwiseProb(score_good > score_bad)={gb_acc:.6f}")
        if not math.isnan(gb_acc):
            print(f"  MeanScore(good)={float(good_pred.mean()):.6f}")
            print(f"  MeanScore(bad)={float(bad_pred.mean()):.6f}")
            print(f"  Gap(good-bad)={(float(good_pred.mean()) - float(bad_pred.mean())):+.6f}")

    if not args.eval_shards and not (args.good_shards and args.bad_shards):
        raise ValueError("Provide either --eval_shards, or both --good_shards and --bad_shards.")


if __name__ == "__main__":
    main()
