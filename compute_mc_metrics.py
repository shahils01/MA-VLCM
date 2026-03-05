"""
Compute Spearman correlation, MSE, MAE, and Mean Prediction Width Interval
from MC dropout CSV files (IID + OOD) for a given environment.

Saves results as a summary CSV in the comparison_plots folder.
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats as scipy_stats


def compute_metrics(csv_path, label=""):
    """Compute all metrics from an MC runs CSV file."""
    df = pd.read_csv(csv_path)

    td_targets = df["td_target"].values
    nstep_returns = df["nstep_return"].values

    # Get prediction columns (run_0, run_1, ...)
    run_cols = [c for c in df.columns if c.startswith("run_")]
    if run_cols:
        runs = df[run_cols].values  # [N, num_runs]
        preds = runs.mean(axis=1)  # mean prediction across MC runs
    else:
        raise ValueError(f"No run columns found in {csv_path}")

    n = len(preds)

    # 1. Spearman correlation (preds vs td_target)
    spearman_corr, spearman_p = spearmanr(preds, td_targets)

    # 2. MSE
    mse = float(np.mean((preds - td_targets) ** 2))

    # 3. MAE
    mae = float(np.mean(np.abs(preds - td_targets)))

    # 4. Mean Prediction Width Interval (95% PI width from linear regression)
    if n > 2:
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            td_targets, preds
        )
        y_pred_for_se = slope * td_targets + intercept
        se_estimate = np.sqrt(np.sum((preds - y_pred_for_se) ** 2) / (n - 2))

        t_val = scipy_stats.t.ppf(0.975, n - 2)
        x_mean = np.mean(td_targets)
        ss_x = np.sum((td_targets - x_mean) ** 2)

        # Prediction interval margin for each point
        pi_margin = (
            t_val * se_estimate * np.sqrt(1 + 1 / n + (td_targets - x_mean) ** 2 / ss_x)
        )
        # Width = 2 * margin, mean over all points
        mean_pi_width = float(np.mean(2 * pi_margin))
    else:
        mean_pi_width = float("nan")

    return {
        "label": label,
        "csv_path": csv_path,
        "n_samples": n,
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p,
        "mse": mse,
        "mae": mae,
        "mean_prediction_interval_width": mean_pi_width,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from MC dropout CSVs and save to comparison_plots."
    )
    parser.add_argument(
        "--iid_csv", type=str, required=True, help="Path to IID mc_runs CSV"
    )
    parser.add_argument(
        "--ood_csv", type=str, required=True, help="Path to OOD mc_runs CSV"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Environment name (e.g., 'offroad' or 'rware')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="0.5B_LoRA",
        help="Model name for labeling (default: 0.5B_LoRA)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_plots",
        help="Output directory for results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Computing metrics for: {args.env_name.upper()} ({args.model_name})")
    print(f"{'='*70}")

    # Compute metrics for IID
    print(f"\n--- IID ---")
    print(f"  CSV: {args.iid_csv}")
    iid_metrics = compute_metrics(args.iid_csv, label=f"{args.model_name}_IID")

    print(f"  N samples:                    {iid_metrics['n_samples']}")
    print(f"  Spearman ρ:                   {iid_metrics['spearman_corr']:.6f}")
    print(f"  MSE:                          {iid_metrics['mse']:.6f}")
    print(f"  MAE:                          {iid_metrics['mae']:.6f}")
    print(
        f"  Mean PI Width:                {iid_metrics['mean_prediction_interval_width']:.6f}"
    )

    # Compute metrics for OOD
    print(f"\n--- OOD ---")
    print(f"  CSV: {args.ood_csv}")
    ood_metrics = compute_metrics(args.ood_csv, label=f"{args.model_name}_OOD")

    print(f"  N samples:                    {ood_metrics['n_samples']}")
    print(f"  Spearman ρ:                   {ood_metrics['spearman_corr']:.6f}")
    print(f"  MSE:                          {ood_metrics['mse']:.6f}")
    print(f"  MAE:                          {ood_metrics['mae']:.6f}")
    print(
        f"  Mean PI Width:                {ood_metrics['mean_prediction_interval_width']:.6f}"
    )

    # Build summary DataFrame
    rows = []
    for metrics, split in [(iid_metrics, "IID"), (ood_metrics, "OOD")]:
        rows.append(
            {
                "Model": args.model_name,
                "Environment": args.env_name,
                "Split": split,
                "N_Samples": metrics["n_samples"],
                "Spearman_Correlation": round(metrics["spearman_corr"], 6),
                "MSE": round(metrics["mse"], 6),
                "MAE": round(metrics["mae"], 6),
                "Mean_PI_Width": round(metrics["mean_prediction_interval_width"], 6),
            }
        )

    summary_df = pd.DataFrame(rows)

    # Save
    out_path = os.path.join(args.output_dir, f"{args.env_name}_metrics.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"  Saved metrics to: {out_path}")
    print(f"{'='*70}")

    # Also print a formatted table
    print(f"\n{summary_df.to_string(index=False)}")
    print()


if __name__ == "__main__":
    main()
