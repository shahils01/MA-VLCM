# One frozen-backbone VLCM critic for both Python tasks

`train_multitask.py` initializes from the Python formation-trained supervised
`best.pt` and jointly fits formation and go-to-goal **native return** targets.
It keeps the SmolVLM vision/text encoders frozen and trains the existing graph,
visual projection, fusion, time embedding, temporal transformer and value head.
This is supervised value fine-tuning, before using the resulting critic in
policy learning. There is one shared scalar value head; the image/instruction
and robot-state history condition its predictions on the task.

Use a new output directory; the existing formation checkpoint is retained.
Initialize from its weights, with a fresh optimizer. `--resume` is reserved
for epoch-boundary continuation of this same joint run, not for changing datasets.

## Launch

After setting the actual paths on the training machine:

```bash
export FORMATION_DATASET=/path/to/polygon_formation_python_env
export GOTOGOAL_DATASET=/path/to/manyagent_gotogoal_python_2000
export INIT_CHECKPOINT=/path/to/outputs/smolvlm-18918211/best.pt
bash train_multitask_frozen.sh
```

The shell launcher works locally with `PYTHON_BIN` or the activated Python.
On Palmetto, from the `MA-VLCM` checkout, create `logs/` and use:

```bash
mkdir -p logs
CONDA_ENV=internvl sbatch train_multitask_palmetto.sbatch
```

Export path overrides first so Slurm inherits them. Defaults are
one A100, 20 epochs, batch size 64, learning rate 3e-5, bf16 and W&B logging.
`RUN_DIR`, `CACHE_DIR`, `EPOCHS`, `LR`, `BATCH_SIZE`, `NUM_WORKERS`,
`WANDB_PROJECT`, `WANDB_MODE` and `WANDB_ENABLED` are configurable. Set
`WANDB_ENABLED=0` to omit W&B. `NUM_GPUS>1` uses torchrun; request the matching
GPU resources when submitting. Extra CLI flags follow the shell/sbatch name.

The Palmetto launcher now defaults to the confirmed paths:
`/home/anshuln/Dataset/polygon_formation_python_env`,
`/home/anshuln/Dataset/manyagent_gotogoal_python_2000`, and
`/home/anshuln/MA_VLCM/MA-VLCM/outputs/smolvlm-18918211/best.pt`.
Environment variables override those defaults. The shell launcher also accepts
an initialization directory and resolves its `best.pt`.

## Train only the new navigation task

Formation rehearsal is optional. To initialize from the trained formation
critic and update it using **only go-to-goal training episodes** on Palmetto:

```bash
cd /home/anshuln/MA_VLCM/MA-VLCM
mkdir -p logs
CONDA_ENV=internvl sbatch train_multitask_palmetto.sbatch --training-tasks goal_to_goal
```

In this mode, formation samples supply no gradients and no new normalization
statistics. Training traverses the go-to-goal clips without replacement. The
original formation training mean/variance are reused only as evaluation
baselines. Formation validation/test remain available to measure forgetting;
best-checkpoint selection still evaluates both tasks equally. The run records
the precise gradient-bearing tasks and episode counts. Initial pretraining
episodes remain marked as previously seen for external-evaluation checks.

Omit `--training-tasks goal_to_goal` for balanced rehearsal on both tasks.
That continues training from existing weights; it does not restart formation
learning from scratch. New-task-only training can forget formation behavior,
so retention must be measured rather than assumed from the initialization.

The input checkpoint pins the exact backbone model revision, processor size,
clip length and network architecture. Frozen image-language features are cached
once; compatible existing formation feature caches remain reusable. Index
metadata is refreshed for the new task without changing its image-feature keys.
Feature extraction may take most of the initial wall time.

## Dataset and return correctness

The loader now handles both the formation metadata and navigation metadata.
It checks 3,000 formation and 2,000 go-to-goal archives before training. Native
timeouts (`timeout` or `max_steps`) are excluded: their unobserved continuation
returns are not labeled zero. The supplied go-to-goal collection has three
such episodes, leaving 1,997 eligible episodes. Failures ending in collision
or boundary exit are true terminals and are valid training examples.

Targets use all recorded control rewards, excluding the reward leading into
the current state. Gamma .99 is applied **per native control step**: formation
has 20 Hz control and go-to-goal has 10 Hz control. Both supply 10 Hz images.
The model therefore learns task-conditional native returns with distinct
physical discount horizons; no reward scaling or invented intermediate reward
is applied to the data. Terminal anchors are excluded from regression.

The earlier formation train/validation/test assignment is preserved from the
initial checkpoint even if its dataset root moved. New go-to-goal scenarios
are split by whole episode/scenario, stratified by checkpoint, team size and
terminal outcome. Previously held-out formation trajectories never enter
joint training. The initial checkpoint must be the one trained on this same
formation collection, with its saved split metadata.

In the default joint mode, the formation collection has more and longer episodes. Training samples each
task with equal probability at the clip level. Its loss divides each task's
return error by that task's training-only target standard deviation, so a task
with larger numerical returns does not dominate simply because of its scale.
The output still predicts native reward units. Joint observation/output
normalizers are computed from training data only. Affine conversion of the
first graph layer and final scalar layer preserves the initial checkpoint's
predictions while changing those normalizers; the value head is not reset.

## Assessing both tasks

The run writes `before_validation_metrics.json`, per-epoch validation metrics,
`best.pt`, `latest.pt` and periodic epoch checkpoints. Early stopping and best
checkpoint selection use the mean of the two tasks' validation MSE divided by
their respective training return variances. This gives both tasks equal weight.
The initial pretrained model is also a candidate: if no epoch improves on its
joint validation score, `best.pt` retains that initial model (epoch zero).

At the end, the original formation critic and the selected joint critic are
evaluated on the **same held-out test episodes**, once. Results are in
`before_test_metrics.json`, `after_test_metrics.json` and `task_comparison.json`.
Both sets of test predictions are saved as CSV. Metrics include MAE, RMSE,
Pearson correlation, R², training-mean baselines, team size and success/failure
breakdowns **for each task**. Formation outcome tables do not mislabel go-to-goal
episodes as triangles or squares. W&B tracks validation errors separately.
`--skip-test` keeps the test set untouched during preliminary experiments.

Joint fitting is a way to test cross-task value estimation, not evidence that
the current checkpoint already generalizes. The go-to-goal data contains 355
full successes out of 2,000, and its 5-by-5 m collection arena differs from the
source MAPPO policies' 10-by-10 m training arena. Report success/failure and
team-size results separately. Values estimate the dataset behavior-policy
mixture's recorded return; they are not guaranteed current-policy values for
an evolving MARL actor.
