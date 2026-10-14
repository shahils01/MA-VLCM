# VLCM run register

> **Input audit:** all three recorded runs used state-augmented task text with collision/reached flags. Their metrics apply to that legacy contract. Task-only critic readiness requires fresh training. [Audit and literature](task_only_input_audit.md).

The last three Qwen VLCM training runs, ordered oldest to newest. Full W&B names link directly to each run. Run 3 status and epoch were refreshed on October 14, 2026 (America/New_York); runs 1 and 2 retain their verified completed status from October 13.

| Run | Exact W&B name | Changes from preceding run | Data: GTG / AO episodes | LoRA rank / alpha | A100 GPUs / effective batch | Status |
|---|---|---|---|---|---|---|
| 1 | [full224_categorical21_signed_suboptimal500_qwen3_vl_clip8_epoch75_20261006_20261006_130727](https://wandb.ai/i2rLAB/VLCM_Training_RWARE/runs/xayryj29) | Reference run: 21-bin categorical CE on signed APF, query readout, attention/max agent pooling, auxiliary goal/risk/collision heads, temporal delta and conditional Lipschitz losses; added 500 AO episodes. | 2,266 / 1,500 | 16 / 32 | 1 / 128 | finished; epoch 75/75 |
| 2 | [smooth_pref_diverse1000_qwen3_vl_clip8_epoch75_20261008_20261011_172911](https://wandb.ai/i2rLAB/VLCM_Training_RWARE/runs/idin1axh) | Added 1,000 diverse AO episodes; causal APF target smoothing; separate trajectory preference head (BCE weight 0.5 every 4 microbatches), with 400 training and 100 validation pairs. | 2,266 / 2,500 | 16 / 32 | 1 / 128 | finished; epoch 75/75 |
| 3 | [relational_native224_2500_qwen3_vl_clip8_epoch75_20261012_20261013_010607](https://wandb.ai/i2rLAB/VLCM_Training_RWARE/runs/nku3ga22) | Replaced all 2,500 AO episodes with native 224x224 collection; per-agent observation tokens, two-layer causal observation-history Transformer, task-context-conditioned visual cross-attention and agent self-attention; LoRA rank 16 to 32 (alpha 32 to 64); one to two A100 GPUs; wider validation coverage. | 2,266 / 2,500 | 32 / 64 | 2 / 256 | finished; epoch 75/75 |

Shared configuration: Qwen3-VL-2B-Instruct; 8-frame clips; 224×224 model input; 75 epochs; both GTG and AO datasets with balanced domain sampling; fresh adapters, heads, optimizer and scheduler; no prior critic checkpoint resume. All three use 21-bin categorical CE over [-1,1], query readout, attention/max aggregation, auxiliary potential weight 0.5, temporal delta weight 1.0 and conditional Lipschitz weight 1.0. Batch size is 32 per GPU with gradient accumulation 4. VJEPA remains on standby.

## Target and validation changes

| Setting | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| APF target | Raw | Past-only Gaussian, sigma 1 frame, window 5 | Same smoothing as run 2 |
| Collision target | Signed APF collision penalty | Immediate -1; reset smoothing history | Same as run 2 |
| Trajectory preference | Disabled | Separate pooled-clip head; BCE 0.5 every 4 microbatches | Same objective, shared new relational representation |
| Preference pairs, train / validation | None | 400 / 100 | 400 / 100 |
| Graph observations | Last frame | Last frame | All eight frames through a causal Transformer |
| Robot tokens | One pooled team token | One pooled team token | One token per real agent |
| Visual grounding | Qwen query and team feature concatenation | Same as run 1 | Per-agent visual cross-attention conditioned on critic query context |
| Validation batches per rank | 40 | 40 | 128 |
| Validation window stride | 1 | 1 | 32 |
| Train / validation episode shards | 3,016 / 750 | 3,825 / 941 | 3,783 / 983 |

Validation trajectory IDs can include forward and synthetic rewind trajectories; they are not a count of distinct real episodes. Validation losses are not a matched comparison across runs because dataset membership, target smoothing and coverage changed. Use the same held-out episodes and target convention when comparing model quality.

## Run records

### Run 1

W&B ID: `xayryj29`. Training job: **16723209**. Remote experiment: `/scratch/aparame/VLCM/ma_vlcm/runs/categorical_qwen_clip8_full_epoch75_20261006_01`.

GTG revision: `a4156c6d887b830d38c9bf6e2aa9b1130414bf2b`. AO revision: `d77e8983b540b9517a4d53bfbed2b848eca22383`.

- Full 75 epochs completed; epoch 34 and 75 evaluated on 100 held-out real episodes.

### Run 2

W&B ID: `idin1axh`. Training job: **17038502**. Remote experiment: `/scratch/aparame/VLCM/ma_vlcm/runs/qwen_smooth_pref_diverse1000_20261008_01`.

GTG revision: `a4156c6d887b830d38c9bf6e2aa9b1130414bf2b`. AO revision: `4391808e20336f235614f518f8cf495a49dbbe96`.

- Collected 1,000 new AO episodes: 594 with collisions and 400 safe successes.
- 75 epochs completed.

### Run 3

W&B ID: `nku3ga22`. Training job: **17141648**. Remote experiment: `/scratch/aparame/VLCM/ma_vlcm/runs/qwen_relational_native224_2500_20261012_01`.

GTG revision: `a4156c6d887b830d38c9bf6e2aa9b1130414bf2b`. AO revision: `8c44ba972e374e50db54eb4e34c7a3ed161e6015`.

- Collected 2,500 new scenes: 1,506 collision episodes and 943 safe successes; all uploaded LFS checksums verified before old AO files were removed.
- Profile 17092951 failed before training with bf16 mask overflow; repaired and validated with 94 tests plus two-process CPU backward/synchronization.
- Replacement two-A100 profile 17141647 passed; strict state reload passed; peak allocated memory about 20.4 GiB per A100 80GB GPU.
- Fresh training 17141648 completed all 75 epochs successfully (Slurm COMPLETED, exit 0:0; W&B finished).
- Best composite validation loss / categorical CE: epoch 22. Best scalar validation MSE: epoch 63. Final checkpoint: epoch 75.
- Matched 100-episode inference comparison (50 GTG, 50 AO) submitted October 14: preparation **17252650** → one-A100 evaluation **17252651**. Preparation and inference both completed successfully (exit 0:0). Evaluated 9,562 windows per checkpoint from frozen validation forward trajectories.
- [Training history plot](relational_native224_20261014/training_history.png). Inference output root on Palmetto: `/scratch/aparame/VLCM/ma_vlcm/evaluations/qwen_relational_native224_best_vs75_20261014`.

## Relational inference results

| Epoch | Overall MAE | GTG MAE | AO MAE | GTG drop recall | AO drop recall |
|---|---:|---:|---:|---:|---:|
| 22 | 0.1048 | 0.0665 | 0.1430 | 83.3% | 46.4% |
| 63 | 0.0611 | 0.0346 | 0.0877 | 92.4% | 40.2% |
| 75 | 0.0550 | 0.0270 | 0.0830 | 96.5% | 36.7% |

Epoch75 ranked best for GTG under the legacy text contract; that does not validate task-only policy deployment. AO remains weak on declining risk potentials; epoch63 has lower AO RMSE (0.167 vs0.179) and better AO drop/recovery direction than75. These are matched diagnostic-holdout results, not policy-learning success rates. [Comparison plot](relational_native224_20261014/comparison_overview.png).

## Maintenance

Keep the JSON sidecar (`vlcm_run_register.json`) and this table in sync. Curated comparison reports live beside this register; raw inference outputs remain in ignored `outputs/` or Palmetto scratch. Append new training runs; retain historical rows. A retry of a pre-training smoke/profile job belongs in the current experiment's event history. Record source configurations and dataset revisions, including any resource or validation changes. Date-stamp status updates; preserve the distinction between requested epochs and completed epochs. W&B display names in this register must remain exact.
