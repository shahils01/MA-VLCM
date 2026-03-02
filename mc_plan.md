# MC Inference Refactoring Plan

The user wants to remove the "Baseline vs LoRA" dual-evaluation from `inference.py` when `--run_mc_comparison` is used. 
Instead, a single `--run_mc_comparison` call should just evaluate the model currently loaded (which is either Baseline or LoRA depending on the checkpoint and `--baseline` flag), and dump the MC runs to a CSV file.
Then, they can run `inference.py` twice (once on IID data, once on OOD data) and use the existing `plot_mc_ood_vs_iid.py` to plot them together.

## Changes to `inference.py`
1. Rename `--run_mc_comparison` to `--run_mc_dropout`. (Optional but clearer, maybe keep it as `--run_mc_comparison` or `--run_mc` for backward compatibility from a few minutes ago. Let's use `--run_mc_dropout`).
2. Update `run_mc_comparison_flow` to `run_mc_dropout_flow`.
3. Inside `run_mc_dropout_flow`:
   - Remove the `eval_model(is_baseline)` loop that explicitly loads both.
   - Use the *already loaded* `model` from `main()`. This means `run_mc_dropout_flow` should accept the `model` object.
   - Run the MC dropout passes using the provided model.
   - Save the results to `mc_runs.csv` in `plot_dir` (perhaps incorporating the checkpoint name or dataset name if we want, but `plot_dir` is usually specific enough).
   - Generate a single simple plot for *this* specific run (mean + std interval) just as a quick visualization, or skip plotting and just save the CSV. Let's just save the CSV and make a basic plot.

## Changes to `plot_mc_ood_vs_iid.py`
- No major changes needed, it already accepts `--iid_csv` and `--ood_csv`. Just ensure the instructions reflect the new workflow.

Let's look at `inference.py` around line 600-800 to see how to integrate this cleanly.
