# Relational native224 experiment drivers

These scripts archive the executed Palmetto training/profile and inference workflow.
They retain the recorded run paths, frozen checkpoint choices, and readiness checks.
Stage the required scripts inside a new scratch run/evaluation directory and update
the recorded roots before reusing them; do not rerun the completed experiment paths.

- `train.sbatch`: fresh Qwen, both task domains, 8 frames, 224×224, 75 epochs,
  LoRA rank32/alpha64, two A100s, batch32 per GPU, accumulation4. Requires completed
  data, preference-loader and distributed-profile receipts.
- `profile.sbatch`, `profile_distributed.py`, `check_distributed_cpu.py`: distributed
  loss/backward, weight synchronization and strict checkpoint reload checks.
- `prepare_evaluation.py` and `prepare.sbatch`: deterministic forward-only holdout
  extraction; 50 GTG and50 AO episodes, excluding training shards and rewinds.
- `run_evaluation.py` and `eval.sbatch`: shell-based inference for exact epochs22/63/75,
  then finite-metric validation and comparison plots.
- `plot_comparison.py`, `plot_training_history.py`: matched episode and logged epoch
  analysis.

[Completed results](../../../docs/experiments/relational_native224_20261014/README.md)
and the [run register](../../../docs/experiments/vlcm_run_register.md) record source
revisions, checkpoint selection, metrics and known AO limitations.

The scripts do not launch jobs when imported; submit them explicitly using Slurm.
Collection/upload/deletion orchestration remains with the separately staged dataset
pipeline. This archive does not trigger dataset replacement.
