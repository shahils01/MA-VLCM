# Three 0.5B critic experiments for 3 and 5 robots

These are three separate supervised VLCM runs with separate checkpoints:

| Mode | Formation control | Navigation | Agents |
| --- | --- | --- | --- |
| `formation_only` | Python formation dataset | None | 3, 5 |
| `joint_without_obstacles` | Same formation data | Only `without_obstacles` | 3, 5 |
| `joint_with_obstacles` | Same formation data | Only `with_obstacles` | 3, 5 |

“Joint” means one shared critic learns from both tasks in the same training
run. Tasks have equal probability at the clip level. It does not mean running
two unrelated critics on separate GPUs. Run the obstacle-free experiment
before the obstacle experiment. Formation data are unchanged in the third
experiment; obstacle filtering applies only to navigation.

Each run defaults to a pretrained `HuggingFaceTB/SmolVLM-500M-Instruct` encoder
with fresh trainable graph, fusion, temporal and scalar value layers. The 0.5B
image/language backbone is frozen and its features are cached. Default runs
are independent comparisons: the obstacle experiment does not automatically
load weights from the obstacle-free experiment or an older formation critic.

## Palmetto commands

From your updated `MA-VLCM` checkout, with both datasets already downloaded:

```bash
cd /home/anshuln/MA_VLCM/MA-VLCM
mkdir -p logs
export FORMATION_DATASET=/home/anshuln/Dataset/polygon_formation_python_env
export GOTOGOAL_DATASET=/home/anshuln/Dataset/manyagent_gotogoal_10m_3000
export RUN_ROOT="$PWD/outputs/critic35_comparison"
export CACHE_DIR="$PWD/outputs/cache"
unset INIT_CHECKPOINT  # Independent runs; optional transfer is described below.
```

Check all selections without loading the backbone or starting training:

```bash
bash train_35_experiments.sh all --preflight-only
```

Use your activated `internvl` environment for that command, or set
`PYTHON_BIN=/home/anshuln/.conda/envs/internvl/bin/python`.
Each mode writes a `selection_report.json` with raw counts, selected counts
by task/team/noise and exact selected paths. Full training additionally
checks the actual archive agent counts and obstacle geometry.

Submit the baseline and joint obstacle-free experiments:

```bash
CONDA_ENV=internvl sbatch train_35_palmetto.sbatch formation_only
CONDA_ENV=internvl sbatch train_35_palmetto.sbatch joint_without_obstacles
```

After reviewing the obstacle-free results, submit:

```bash
CONDA_ENV=internvl sbatch train_35_palmetto.sbatch joint_with_obstacles
```

To run all three sequentially within one allocation instead:

```bash
CONDA_ENV=internvl sbatch train_35_palmetto.sbatch all
```

The Slurm wrapper follows the existing Palmetto `work1`/Anaconda convention
and requests one A100. Adjust partition, GPU or wall time to your allocation.
The local shell wrapper can also run inside an existing compute allocation:
`bash train_35_experiments.sh formation_only`. Training is not run on the
login node by these submission commands.

Defaults: 30 epochs, batch 64, learning rate 1e-4, bf16, six-frame causal
clips at 10 Hz, early-stopping patience six, seed 42 and W&B enabled.
`EPOCHS`, `BATCH_SIZE`, `LR`, `MIXED_PRECISION`, `NUM_WORKERS`, `NUM_GPUS`,
`VL_BATCH_SIZE`, `CLIP_LEN`, `IMAGE_SIZE`, `SEED`, `RUN_ROOT`, `RUN_DIR`,
`CACHE_DIR`, `WANDB_PROJECT`, `WANDB_ENTITY` and `WANDB_ENABLED` override these.
Request matching GPUs when using `NUM_GPUS>1`.
Use `WANDB_ENABLED=0` for untracked runs. Extra Python flags follow the mode.
With `all`, use `RUN_ROOT` rather than a shared `RUN_DIR`.

Sources may also be Hugging Face dataset IDs:

```bash
export FORMATION_DATASET=shulnak09/polygon_formation_python_env
export GOTOGOAL_DATASET=shulnak09/manyagent_gotogoal_10m_3000
```

The loader downloads matching archive/summary files into the HF cache before
selection. Local downloaded roots are preferable on the HPC to avoid
repeated Hub access and rate limits. No dataset download is needed for a
second experiment using the same local roots.

## Data selection and return targets

The unified navigation collection supplies **1,500 episodes per obstacle
condition**, with **750 per team size** (500 clean, 125 wandering and 125
oscillating). Both clean and noisy data are included. The combined dataset
is expected to contain 3,000 raw navigation archives; this check runs before
team/condition filters. `EXPECTED_GOTOGOAL_EPISODES=0` disables that size
check for another collection. Formation raw counts are reported without
assuming a particular dataset version; use `EXPECTED_FORMATION_EPISODES`
to enforce one.

Timeouts (`timeout`/`max_steps`) and unfinished trajectories are excluded
because their continuation return is unknown. Consequently the navigation
training/evaluation population is 1,500 without obstacles and 1,486 with
obstacles in this collection. Those are totals before train/validation/test
splitting. Collision and boundary failures are valid true-terminal examples.
No episode is discarded merely for failure, wandering or a low reward.

Targets use the complete recorded native control-reward stream, not geometric
progress. Gamma 0.99 is applied per native control step (formation 20 Hz;
navigation 10 Hz). Incoming reward at the current image is excluded. Different
control rates imply different physical discount horizons. No progress reward,
reward-prediction head or auxiliary progress loss is introduced by this change.
Progress/outcome labels are not model input features. The value estimates the
dataset's behavior-policy mixture, not automatically the clean actor's value.

Independent fresh runs split whole scenarios into approximately 80/10/10
train/validation/test, stratified by task, stage, team size and terminal outcome.
These are seeded experiment splits, rather than the collection's 90/5/5 split.
Formation splitting uses its own RNG and is identical across the three runs
when source data, filters and seed match. Navigation scenarios include obstacle
geometry. Train-only statistics normalize observations and per-task loss;
predictions remain in native reward units. The mean of per-task validation
MSE divided by each task's training return variance selects `best.pt`.

Each output directory contains `run.json`, `selection_report.json`,
`metrics.jsonl`, `best.pt`, `latest.pt`, per-task validation/test metrics and
prediction CSVs, and an initial-versus-trained test comparison. Terminal
anchors are excluded from value regression. Use `--skip-test` during tuning.

## DeltaAI: four GPUs and utilization monitoring

DeltaAI uses four H100 GPUs per GH200 node. Use `train_35_deltaai.sbatch`
instead of the Palmetto/A100 wrapper. From the checkout on DeltaAI:

```bash
mkdir -p logs
export FORMATION_DATASET="$HOME/VLCM/datasets/polygon_formation_python_env"
export GOTOGOAL_DATASET="$HOME/VLCM/datasets/manyagent_gotogoal_10m_3000"
sbatch --account=YOUR_PROJECT train_35_deltaai.sbatch joint_without_obstacles
```

This requests four GPUs on one node for up to 24 hours and launches four DDP
workers. It loads `python/miniforge3_pytorch/2.10.0` and activates `base`, or
uses `CONDA_ENV`/`PYTHON_BIN` overrides. The chosen environment must also have
the repository's training dependencies. Dataset path overrides are required
if your DeltaAI directories differ from the examples.

Utilization and memory samples are written every ten seconds to
`logs/gpu_utilization_JOBID.csv`. GPU compute utilization differs from memory
occupancy. A Slurm allocation cannot enforce a 70% compute utilization floor.
The current frozen-backbone pipeline creates image features on rank zero
while other ranks wait, then trains small critic layers from cached features.
Four GPUs may therefore be less efficient than one. No minimum utilization
is claimed, and increasing batch size needs a measured throughput/validation
comparison. Do not change the learning objective simply to keep GPUs busy.

For a one-GPU benchmark of the same experiment:

```bash
NUM_GPUS=1 sbatch --account=YOUR_PROJECT --gpus-per-node=1 \
  --cpus-per-task=8 --mem=32G train_35_deltaai.sbatch joint_without_obstacles
```

Compare samples/second, epoch time and validation quality before choosing
four GPUs for the full run. Parallel feature caching and a profiler-guided
training pass would be further optimizations; they are not implemented by
this batch wrapper.

## Optional sequential transfer and resume

To explicitly initialize a new joint run from the formation-only critic:

```bash
INIT_CHECKPOINT="$RUN_ROOT/formation_only/best.pt" \
  sbatch train_35_palmetto.sbatch joint_without_obstacles
```

Similarly, initialize the obstacle run from the obstacle-free critic by
setting `INIT_CHECKPOINT` to its `best.pt`. This is a **transfer experiment**,
not an independent comparison. It starts a fresh optimizer, preserves the
initial checkpoint's formation holdouts and converts normalizers without
changing the initial predictions. Document the initialization when reporting
results. Always use a new output directory.

For epoch-boundary continuation of the same run, keep the same sources,
selection, seed and original `INIT_CHECKPOINT` setting (if any), set the
original `RUN_ROOT`/`RUN_DIR`, and pass
`--resume /path/to/latest.pt --epochs 50`. Resume rejects changed task,
agent/condition selections or dataset splits. Submit one mode when resuming.

The older `train_multitask_frozen.sh` and legacy warm-start mode remain
available for their original experiments; they do not select these new modes.
