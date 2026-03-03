import argparse
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_data(df, max_samples):
    n = min(len(df), max_samples)

    # 1. Identify target column
    target_cols = ["return", "true_return", "td_target", "reward"]
    target_col = None
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]

    true_vals = df.loc[: n - 1, target_col].values - 50.0

    # 2. Identify prediction columns
    run_cols = [c for c in df.columns if c.startswith("run_")]
    if run_cols:
        preds = df.loc[: n - 1, run_cols].values.mean(axis=1)
    elif "prediction" in df.columns:
        preds = df.loc[: n - 1, "prediction"].values
    else:
        raise ValueError(
            f"Could not find prediction columns in {df.columns}. Found: {df.columns}"
        )

    return true_vals, preds, target_col, n


def main():
    parser = argparse.ArgumentParser(
        description="Compare sample_comparison plots for 2 runs."
    )
    parser.add_argument(
        "--csv1", type=str, required=True, help="Path to first results CSV"
    )
    parser.add_argument(
        "--csv2", type=str, required=True, help="Path to second results CSV"
    )
    parser.add_argument(
        "--label1", type=str, default="Run 1", help="Label for first run"
    )
    parser.add_argument(
        "--label2", type=str, default="Run 2", help="Label for second run"
    )
    parser.add_argument(
        "--max_samples", type=int, default=300, help="Max samples to plot"
    )
    parser.add_argument(
        "--output_dir", type=str, default="comparison_plots", help="Output directory"
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help="Label for target values (e.g. Return)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.csv1}...")
    df1 = pd.read_csv(args.csv1)
    print(f"Loading data from {args.csv2}...")
    df2 = pd.read_csv(args.csv2)

    true1, preds1, tname1, n1 = extract_data(df1, args.max_samples)
    true2, preds2, tname2, n2 = extract_data(df2, args.max_samples)

    n_plot = min(n1, n2)
    target_label = args.target_label or tname1

    # ── Plotting ──────────────────────────────────────────────────────────
    # Single panel plot: sequence comparison
    fig, ax = plt.subplots(figsize=(15, 7))
    idx = np.arange(n_plot)

    # 1. Main Comparison Plot
    ax.plot(
        idx,
        true1[:n_plot],
        "-",
        color="#333333",
        linewidth=2.0,
        alpha=0.9,
        label=f"True {target_label}",
    )
    ax.plot(
        idx,
        preds1[:n_plot],
        "-",
        color="#4C72B0",
        linewidth=1.5,
        alpha=0.8,
        label=args.label1,
    )
    ax.plot(
        idx,
        preds2[:n_plot],
        "-",
        color="#C44E52",
        linewidth=1.5,
        alpha=0.8,
        label=args.label2,
    )

    ax.set_xlabel("Sample Index", fontsize=25)
    ax.set_ylabel("True Return", fontsize=25)
    ax.set_title(
        f"Per-Sample Comparison: {args.label1} vs {args.label2}",
        fontsize=40,
        fontweight="bold",
    )

    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", labelsize=25)

    ax.legend(fontsize=20, loc="upper right", frameon=True, shadow=True)
    ax.grid(True, alpha=0.2, linestyle="--")

    # Check for ground truth difference
    if not np.allclose(true1[:n_plot], true2[:n_plot], atol=1e-5):
        print("Warning: True values in the two CSVs differ! Plotting both.")
        ax.plot(
            idx,
            true2[:n_plot],
            "--",
            color="#555555",
            linewidth=1.0,
            alpha=0.4,
            label=f"True {tname2} (Run 2)",
        )
        ax.legend(fontsize=12)

    fig.tight_layout()
    # Clean filename
    safe_l1 = (
        args.label1.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    )
    safe_l2 = (
        args.label2.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    )
    out_path = os.path.join(
        args.output_dir, f"sample_comparison_{safe_l1}_vs_{safe_l2}.png"
    )
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved beautiful comparison plot to: {out_path}")


if __name__ == "__main__":
    main()
