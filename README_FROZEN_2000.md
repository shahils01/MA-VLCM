# Frozen critic on the complete 2,000-episode collection

This prepares a separate **supervised value-regression** experiment on the Isaac
archives. It does not train MAPPO, resume the 900-episode model, or change the
running LoRA job. SmolVLM stays frozen; the visual projection, GAT, fusion,
temporal transformer and value head train from scratch on the expanded data.

## Submit on Palmetto

Update the MA-VLCM code on Palmetto, including `prepare_frozen_2000.py`,
`train_frozen_2000.sh`, `train_frozen_2000_palmetto.sbatch`, `train.py` and
`evaluate.py`. From that checkout, set paths to the uploaded datasets:

```bash
cd /path/to/MA-VLCM
mkdir -p logs
export DATASET="$HOME/Dataset/formation_vlcm_polygon_3to6_10hz_inner00m"
export EXTERNAL_DATASET="$HOME/Dataset/formation_validation_seed7300001"
export CONDA_ENV=MARL
sbatch train_frozen_2000_palmetto.sbatch
```

The template requests **one A100 GPU**, with `NUM_GPUS=1`, 4 CPUs, 24 GB RAM and
12 hours, following this project's existing Palmetto resource conventions.
The independent job/output name is `vlcm_frozen_2000`; outputs go to
`outputs/frozen_2000_<jobid>/`. Set `PROJECT_ROOT` if submitting from elsewhere;
the checkout's `logs` directory must exist before submission. An explicit
`PYTHON_BIN=/path/to/env/bin/python` bypasses conda activation, or `CONDA_BASE`
can point to its installation. Set `HF_HOME` to the existing model cache.
No cluster job was submitted from this local workspace.

W&B is enabled through the existing trainer, with a distinct
`smolvlm-frozen-2000-<jobid>` run name in project `ma-vlcm`. Use existing W&B
login/environment credentials. `WANDB_MODE=offline` saves local logs;
`WANDB_ENABLED=0` disables tracking. Credentials are not written into scripts.
The external metrics are saved as JSON/CSV, independently of W&B availability.

## Dataset and objective

Use the **full-control-reward** `formation_vlcm_polygon_3to6_10hz_inner00m`
collection, not the older sampled-reward `formation_vlcm_polygon_3to6` folder.
The launcher checks exactly 2,000 archives, requires raw control rewards,
validates return interval alignment, and checks the external set's scenario,
reset-stream seed and archive-key independence before starting the encoder.

The local preflight verified:

| Team | Collection episodes | Successful episodes |
| --- | ---: | ---: |
| Triangle, 3 agents | 500 | 299 |
| Square, 4 agents | 500 | 271 |
| Pentagon, 5 agents | 500 | 182 |
| Hexagon, 6 agents | 500 | 113 |
| Total | 2,000 | 865 |

Fourteen timeouts/unfinished trajectories are excluded because this supervised
objective requires a known full return through true termination. The remaining
1,986 episodes split into 1,594 train / 196 internal validation / 196 internal
test, grouped by initial scenario and stratified by stage/team/outcome.
The 60 external archives remain outside every training split. Thus “2,000
episodes” means the source collection, not 2,000 gradient-training episodes.
All four behavior stages and both success/failure outcomes are retained.

The default preserves the old 900-run settings to isolate the effect of the
expanded collection: seed 43, 128-pixel image processing, 6 frames at 5 Hz
(one-second history), gamma 0.99 per 20 Hz control step, raw **full control**
return labels, terminal anchors excluded, batch 64, learning rate 1e-4,
dropout 0.2, width 128, one GAT and one transformer layer, and early-stopping
patience 6 over at most 30 epochs. The backbone revision is pinned to the old
run's `a7da5b986cb59b408707209984f360a5f4ad7e47`.

`--max-episodes 0` uses the entire collection; it does not reuse the old random
900-subset cap. This starts a new head and computes training-only normalization.
Resuming the 900 run is inappropriate because adding data changes the split
membership. Its internal validation scores are not directly comparable with
the new split; compare both selected checkpoints on the same external 60.

## Comparing with the running LoRA model

The **local** `outputs/zero_margin_lora/run.json` records seed 42, 8 frames at
10 Hz (0.7-second history), image size 128, batch 1, accumulation 1 and lr 1e-4.
These differ from the old frozen run in more than trainable backbone weights.
Use the actual Palmetto `run.json` as the authority if its configuration differs.

To match that local LoRA run's split and temporal/image inputs, use:

```bash
FROZEN_PROFILE=lora sbatch train_frozen_2000_palmetto.sbatch
```

This profile changes the frozen run to seed 42 / 8 frames / 10 Hz. Batch size
still defaults to 64 for efficient cached-head training, so that comparison
also includes an optimization difference. To match the recorded single-GPU
LoRA effective batch of 1 as well:

```bash
FROZEN_PROFILE=lora FROZEN_BATCH_SIZE=1 sbatch train_frozen_2000_palmetto.sbatch
```

Match total effective batch across GPU count and accumulation if the Palmetto
LoRA run uses different values. Avoid interpreting different-resolution,
different-context, different-split or different-batch runs as isolating LoRA.
Both profiles train a frozen backbone; neither touches LoRA checkpoint files.
Repeated external evaluation makes these 60 a validation benchmark, not a final
untouched test. Additional five/six-agent successes improve coverage but do not
guarantee lower held-out error or an exact match to every realized return.

## Outputs and 60-episode evaluation

The job caches frozen features in its own `outputs/cache_frozen_2000/`, saves
`preflight.json`, training `run.json`, W&B/epoch metrics, `best.pt`, and one
`epoch_XXX.pt` every two epochs (plus the final/early-stopping epoch). Step saves
are disabled by default. For a complete 20-epoch run this is **11 checkpoint
files**: ten even-epoch snapshots plus the overwritten best file. Optional step
saves overwrite just one `latest.pt`, making 12 files rather than accumulating
one file per step interval. Use `FROZEN_CHECKPOINT_EVERY_STEPS=200` to enable
that rolling recovery checkpoint. A new
`RUN_DIR` is required. `best.pt` is selected only by internal-validation MSE.
These defaults apply to new launches. An already running Palmetto process keeps
the checkpoint intervals recorded in its own `run.json`; changing a local script
does not change that job's startup arguments.
After training, the job automatically runs `evaluate.py --split external
--expected-episodes 60` with that checkpoint. It fails if fewer than 60 complete
evaluation episodes are available, rather than reporting a partial set.

Results are in `<run>/external60/external_metrics.json` and
`external_predictions.csv`: aggregate MSE, MAE, RMSE, R²/correlation,
training-mean baseline, episode-averaged MSE, team-size and outcome metrics, plus
joint groups such as `pentagon_success` and `hexagon_failure`. Joint groups
include episode counts and episode-averaged MSE so long failures cannot silently
dominate the interpretation. This is **VLCM inference**, not MARL-critic output.

The external set has 15 episodes per formation, with 9/9/4/1 successes for
triangle/square/pentagon/hexagon respectively. Successful hexagon error therefore
describes only one episode; aggregate improvements do not establish reliability
on all six-agent successes. Labels are realized discounted future returns; the
model estimates the conditional return of the recorded behavior mixture.

Rerun value estimation independently after copying a checkpoint back, from
MA-VLCM with the correct Python environment:

```bash
python evaluate.py --checkpoint outputs/frozen_2000_JOBID/best.pt \
  --dataset ../Ground_Robot_MARL/datasets/formation_validation_seed7300001 \
  --split external --expected-episodes 60 \
  --cache-dir outputs/cache_frozen_2000 \
  --output-dir outputs/frozen_2000_JOBID/external60_recheck
```

To make per-formation success/failure videos from those predictions:

```bash
python animate_checkpoint.py \
  --predictions outputs/frozen_2000_JOBID/external60/external_predictions.csv \
  --run-dir outputs/frozen_2000_JOBID \
  --external-dataset ../Ground_Robot_MARL/datasets/formation_validation_seed7300001 \
  --by-formation --exclude-terminal \
  --output-dir outputs/frozen_2000_JOBID/animations
```

These replay saved value predictions; return estimation was performed by the
preceding evaluation command. To audit before GPU use, set `PREFLIGHT_ONLY=1`
when running `bash train_frozen_2000.sh` in a Python environment with the project
dependencies; it writes the split/inventory report and stops before training.

Other explicit overrides are `FROZEN_SEED`, `FROZEN_IMAGE_HZ`, `FROZEN_CLIP_LEN`,
`FROZEN_BATCH_SIZE`, `FROZEN_VL_BATCH_SIZE`, `FROZEN_LR`, `FROZEN_EPOCHS`,
`FROZEN_PATIENCE`, `FROZEN_RUN_NAME`, `FROZEN_CACHE_DIR` (Slurm) and `RUN_DIR`.
`FROZEN_CHECKPOINT_EVERY` defaults to 2 epochs and
`FROZEN_CHECKPOINT_EVERY_STEPS` defaults to 0 (disabled).
The forced frozen mode, one process, complete collection and control-return
labels prevent inherited LoRA launcher settings from changing the experiment.
