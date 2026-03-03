import argparse
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot IID vs OOD MC prediction intervals."
    )
    parser.add_argument(
        "--iid_csv", type=str, required=True, help="Path to IID results CSV"
    )
    parser.add_argument(
        "--ood_csv", type=str, required=True, help="Path to OOD results CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--max_samples", type=int, default=100, help="Max samples to plot"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading IID data from: {args.iid_csv}")
    df_iid = pd.read_csv(args.iid_csv)
    print(f"Loading OOD data from: {args.ood_csv}")
    df_ood = pd.read_csv(args.ood_csv)

    def extract_data(df):
        n = min(len(df), args.max_samples)

        if "return" in df.columns:
            target_col = "return"
        elif "td_target" in df.columns:
            target_col = "td_target"
        elif "reward" in df.columns:
            target_col = "reward"
        elif "true_return" in df.columns:
            target_col = "true_return"
        else:
            raise ValueError(f"Could not find a target column in {df.columns}")

        true_returns = df.loc[: n - 1, target_col].values

        # We don't care about MC runs anymore, just use run_0 if available,
        # or avg over runs if they exist to get the point prediction
        cols = [c for c in df.columns if c.startswith("run_")]
        if cols:
            runs = df.loc[: n - 1, cols].values
            preds = runs.mean(axis=1)  # Use the mean prediction
        elif "prediction" in df.columns:
            preds = df.loc[: n - 1, "prediction"].values
        else:
            raise ValueError(f"Could not find prediction columns in {df.columns}")

        return true_returns, preds, n

    try:
        iid_true, iid_preds, n_iid = extract_data(df_iid)
        ood_true, ood_preds, n_ood = extract_data(df_ood)
    except Exception as e:
        print(f"Error extracting data: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    n_plot = min(n_iid, n_ood)

    # Calculate Confidence and Prediction Intervals for the regressions
    def plot_regression_with_intervals(x, y, color, label, marker="o"):
        # Scatter the points
        ax.scatter(
            x,
            y,
            color=color,
            alpha=0.6,
            label=label,
            marker=marker,
            s=30,
            edgecolors="none",
        )

        # Linear fit
        if len(x) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Sort x for plotting smooth curves
            x_sorted = np.sort(x)
            y_pred = slope * x_sorted + intercept

            # Plot the line of best fit
            ax.plot(
                x_sorted,
                y_pred,
                color=color,
                linewidth=2,
                label=f"{label} Fit (slope={slope:.2f})",
            )

            # Local variance calculation via binning
            n_pts = len(x)
            num_bins = min(10, n_pts // 5)
            if num_bins < 3:
                num_bins = 3

            min_x, max_x = np.min(x), np.max(x)
            bins = np.linspace(min_x, max_x, num_bins + 1)

            bin_centers = []
            bin_stds = []

            for i in range(num_bins):
                if i == num_bins - 1:
                    mask = (x >= bins[i]) & (x <= bins[i + 1] + 1e-9)
                else:
                    mask = (x >= bins[i]) & (x < bins[i + 1])

                pts_y = y[mask]
                pts_x = x[mask]

                if len(pts_y) > 1:
                    resids = pts_y - (slope * pts_x + intercept)
                    bin_std = np.std(resids, ddof=1)
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    bin_stds.append(bin_std)

            if len(bin_centers) < 2:
                resids = y - (slope * x + intercept)
                global_std = np.std(resids, ddof=1)
                bin_centers = [min_x, max_x]
                bin_stds = [global_std, global_std]

            from scipy.interpolate import interp1d
            from scipy.ndimage import gaussian_filter1d

            smooth_std_func = interp1d(
                bin_centers, bin_stds, kind="linear", fill_value="extrapolate"
            )
            local_std = smooth_std_func(x_sorted)
            local_std_smoothed = gaussian_filter1d(
                local_std, sigma=max(1, len(x) // 10)
            )

            pi_margin = 1.96 * local_std_smoothed
            pi_margin = np.maximum(pi_margin, 1e-3)

            # Plot prediction intervals as varying width ribbons #
            ax.fill_between(
                x_sorted,
                y_pred - pi_margin,
                y_pred + pi_margin,
                color=color,
                alpha=0.15,
                label=f"{label} Local 95% PI",
            )

    plot_regression_with_intervals(
        iid_true[:n_plot], iid_preds[:n_plot], "#4C72B0", "IID"
    )
    plot_regression_with_intervals(
        ood_true[:n_plot], ood_preds[:n_plot], "#C44E52", "OOD", marker="^"
    )

    # Plot y=x perfect prediction line
    lo = min(
        iid_true[:n_plot].min(),
        ood_true[:n_plot].min(),
        iid_preds[:n_plot].min(),
        ood_preds[:n_plot].min(),
    )
    hi = max(
        iid_true[:n_plot].max(),
        ood_true[:n_plot].max(),
        iid_preds[:n_plot].max(),
        ood_preds[:n_plot].max(),
    )
    margin = (hi - lo) * 0.05
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        "--",
        color="#333333",
        linewidth=1.5,
        alpha=0.8,
        label="y = x (Perfect)",
    )

    ax.set_xlabel("True Return", fontsize=18)
    ax.set_ylabel("Predicted Return", fontsize=18)

    # Decrease ticks to 5 values
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title(
        f"IID vs OOD Predictions & 95% Prediction Intervals",
        fontsize=20,
        fontweight="bold",
    )

    ax.legend(fontsize=15)
    ax.grid(False)  # Turn off grid entirely

    out_path = os.path.join(args.output_dir, "iid_vs_ood_prediction_intervals.png")
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot to: {out_path}")


if __name__ == "__main__":
    main()
