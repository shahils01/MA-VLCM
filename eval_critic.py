import argparse
import math
from types import SimpleNamespace

import torch

from data_loading import webdataset_loader
from train import _apply_peft, _resolve_vl_model_preset, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MA-VLCM critic predictions against return targets.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt) saved by train.py")
    p.add_argument("--eval_shards", type=str, default="", help="Shard pattern for general eval set")
    p.add_argument("--good_shards", type=str, default="", help="Optional shard pattern containing good trajectories")
    p.add_argument("--bad_shards", type=str, default="", help="Optional shard pattern containing bad trajectories")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=512, help="Max samples to evaluate per dataset stream")
    p.add_argument("--print_samples", type=int, default=40, help="How many random pred-vs-return rows to print")
    p.add_argument("--seed", type=int, default=0)
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
def run_stream(model, loader, device, max_samples):
    preds = []
    rets = []
    for batch in loader:
        inputs = _move_inputs_to_device(batch["inputs"], device)
        robot_obs = batch["robot_obs"].to(device)
        adj = batch["adj"].to(device)
        out = model(inputs, robot_obs, adj)
        ret = batch["returns"].to(device)
        preds.append(out.detach().float().cpu())
        rets.append(ret.detach().float().cpu())
        if sum(x.numel() for x in preds) >= max_samples:
            break

    if not preds:
        return torch.empty(0), torch.empty(0)
    pred = torch.cat(preds, dim=0)[:max_samples]
    ret = torch.cat(rets, dim=0)[:max_samples]
    return pred, ret


def _print_sample_table(pred: torch.Tensor, ret: torch.Tensor, k: int, seed: int):
    if pred.numel() == 0:
        print("No samples to print.")
        return
    k = min(k, pred.numel())
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(pred.numel(), generator=g)[:k]
    print("\nRandom sample comparison (predicted_value vs true_return):")
    print("idx\tpred\ttrue\terror")
    for i in idx.tolist():
        p = float(pred[i])
        t = float(ret[i])
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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = _load_train_args(ckpt)

    _resolve_vl_model_preset(train_args)
    _init_quant_config_if_needed(train_args)

    # Keep eval stream size configurable from CLI.
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers

    model = build_model(train_args, device=device)
    model = _apply_peft(model, train_args)
    _load_checkpoint_state(model, ckpt["model"], getattr(train_args, "peft", "none"))
    model.eval()
    model.to(device)

    if args.eval_shards:
        eval_loader = webdataset_loader(train_args, args.eval_shards, args.batch_size, args.num_workers)
        pred, ret = run_stream(model, eval_loader, device, args.max_samples)
        _print_sample_table(pred, ret, args.print_samples, args.seed)
        _print_core_metrics("eval_shards", pred, ret)

    if args.good_shards and args.bad_shards:
        good_loader = webdataset_loader(train_args, args.good_shards, args.batch_size, args.num_workers)
        bad_loader = webdataset_loader(train_args, args.bad_shards, args.batch_size, args.num_workers)
        good_pred, good_ret = run_stream(model, good_loader, device, args.max_samples)
        bad_pred, bad_ret = run_stream(model, bad_loader, device, args.max_samples)
        _print_core_metrics("good_shards", good_pred, good_ret)
        _print_core_metrics("bad_shards", bad_pred, bad_ret)
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
