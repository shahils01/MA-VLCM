import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from train import build_model, _apply_peft, webdataset_loader

def run_mc_comparison_flow(cli_args, ckpt, args, device, model_dtype):
    print(f"\n--- Running MC Comparison (runs={cli_args.mc_runs}) for both Baseline and LoRA ---")
    
    # Setup test loader
    args.train_shards = cli_args.test_shards
    saved_loss_type = getattr(args, "loss_type", "td")
    args.loss_type = "td"
    test_loader = webdataset_loader(
        args, shards=cli_args.test_shards, batch_size=args.batch_size, 
        num_workers=args.num_workers, shuffle=False
    )
    args.loss_type = saved_loss_type

    def _move_and_cast(tensor_dict):
        out = {}
        for k, v in tensor_dict.items():
            if torch.is_tensor(v):
                v = v.to(device)
                if v.is_floating_point():
                    v = v.to(dtype=model_dtype)
                out[k] = v
            else:
                out[k] = v
        return out

    # Cache top samples on CPU
    print(f"Caching up to {cli_args.max_samples} samples into RAM...")
    batches = []
    num_processed = 0
    true_returns = []
    
    for batch in test_loader:
        # Move to CPU explicitly just to be safe
        b_cpu = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                b_cpu[k] = v.cpu()
            elif isinstance(v, dict):
                b_cpu[k] = {k2: v2.cpu() if torch.is_tensor(v2) else v2 for k2, v2 in v.items()}
            else:
                b_cpu[k] = v
        batches.append(b_cpu)
        
        b_size = batch["robot_obs"].shape[0]
        if "returns" in batch:
            true_returns.append(batch["returns"].cpu().float())
            
        num_processed += b_size
        if cli_args.max_samples and num_processed >= cli_args.max_samples:
            break
            
    if len(true_returns) > 0:
        true_returns = torch.cat(true_returns, dim=0).view(-1).numpy()[:cli_args.max_samples]
    else:
        print("Warning: no 'returns' found in batch. Cannot plot true returns!")
        true_returns = np.zeros(num_processed)[:cli_args.max_samples]

    def eval_model(is_baseline):
        print(f"\nEvaluating {'Baseline' if is_baseline else 'LoRA'} model...")
        saved_peft = getattr(args, "peft", "none")
        if is_baseline:
            args.peft = "none"
        
        model = build_model(args, device=device)
        model = _apply_peft(model, args)
        
        state_dict = ckpt["model"]
        cleaned_sd = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}
        
        if is_baseline:
            custom_prefixes = ("robot_gnn.", "value_head.", "obs_to_lm.")
            baseline_sd = {k: v for k, v in cleaned_sd.items() if k.startswith(custom_prefixes)}
            model.load_state_dict(baseline_sd, strict=False)
        else:
            model.load_state_dict(cleaned_sd, strict=False)
            
        args.peft = saved_peft
        
        model = model.to(device=device, dtype=model_dtype)
        model.eval()
        
        # Enable dropout for MC runs
        def enable_dropout(m):
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        model.apply(enable_dropout)
        
        all_mc_preds = []
        for run_idx in tqdm(range(cli_args.mc_runs), desc=f"{'Baseline' if is_baseline else 'LoRA'} MC Runs"):
            run_preds = []
            with torch.no_grad():
                for batch in batches:
                    inputs = _move_and_cast(batch["inputs"])
                    robot_obs = batch["robot_obs"].to(device=device, dtype=model_dtype)
                    adj = batch["adj"].to(device=device, dtype=model_dtype)
                    
                    pred = model(inputs, robot_obs, adj)
                    run_preds.append(pred.detach().cpu().float())
                    
            run_preds_flat = torch.cat(run_preds, dim=0).view(-1).numpy()[:cli_args.max_samples]
            all_mc_preds.append(run_preds_flat)
            
        del model
        torch.cuda.empty_cache()
        
        # [mc_runs, N]
        all_mc_preds = np.stack(all_mc_preds, axis=0)
        return all_mc_preds.mean(axis=0), all_mc_preds.std(axis=0)

    base_mean, base_std = eval_model(is_baseline=True)
    lora_mean, lora_std = eval_model(is_baseline=False)

    print("\nGenerating prediction interval plot...")
    plot_dir = cli_args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(true_returns)
    idx = np.arange(n)
    
    # 1. True Returns
    ax.plot(idx, true_returns, "-", color="#333333", linewidth=1.5, alpha=0.9, label="True returns")
    
    # 2. Baseline
    ax.plot(idx, base_mean, "-", color="#8DA0CB", linewidth=1.2, alpha=0.9, label="Predicted returns (Baseline)")
    ax.fill_between(idx, base_mean - base_std, base_mean + base_std, color="#8DA0CB", alpha=0.3)
    
    # 3. LoRA
    ax.plot(idx, lora_mean, "-", color="#E78AC3", linewidth=1.2, alpha=0.9, label="Predicted returns (LoRA)")
    ax.fill_between(idx, lora_mean - lora_std, lora_mean + lora_std, color="#E78AC3", alpha=0.3)
    
    ax.set_xlabel("Sample Index", fontsize=13)
    ax.set_ylabel("Return Value", fontsize=13)
    ax.set_title(f"Prediction Intervals (MC Runs: {cli_args.mc_runs}) - Baseline vs LoRA", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    path = os.path.join(plot_dir, "mc_prediction_intervals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot to: {path}")

