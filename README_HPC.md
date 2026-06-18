# Running MA-VLCM on HPC with Apptainer

This document describes how to build and run the MA-VLCM container after the repo reorganization into `src/`, `scripts/`, `tools/`, and `outputs/`.

## 1. Prerequisites

- Apptainer installed on the cluster
- Git
- Enough scratch space for the container image and caches

## 2. Building the Container

From the repo root:

```bash
cd /path/to/MA-VLCM
apptainer build --fakeroot ma_vlcm.sif ma_vlcm.def
```

If `/tmp` is too small:

```bash
mkdir -p $HOME/scratch/tmp
export APPTAINER_TMPDIR=$HOME/scratch/tmp
apptainer build --fakeroot ma_vlcm.sif ma_vlcm.def
```

## 3. Running Training

### Interactive Run

```bash
cd /path/to/MA-VLCM
export PYTHONPATH="$PWD/src"
apptainer exec --nv -B "$PWD:$PWD" ma_vlcm.sif \
  accelerate launch -m ma_vlcm.train --help
```

### Batch Run

Use the launcher in `scripts/`:

```bash
bash scripts/run_train_vlcm.sh
```

Example SLURM wrapper:

```bash
#!/bin/bash
#SBATCH --job-name=ma_vlcm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --gpus=h100:2

mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

bash scripts/run_train_vlcm.sh
```

Submit with:

```bash
sbatch --mail-type=BEGIN,END,FAIL scripts/submit_train.sh
```

### TurtleBot3 Lab LoRA Fine Tuning

The TB3 lab dataset is stored under `data/tb3_lab` and should contain numeric
WebDataset shards from `0.tar` through `200.tar`.

The TB3-specific launcher resumes from the saved MA-VLCM checkpoint
`checkpoints/NewFinal_0.5B.pt` by default:

```bash
bash scripts/lora_run_train_tb3_lab.sh
```

Submit the Slurm job with:

```bash
sbatch scripts/lora_submit_train_tb3_lab.sh
```

To use a different pretrained or intermediate checkpoint:

```bash
RESUME_CHECKPOINT=/path/to/checkpoint.pt \
  sbatch scripts/lora_submit_train_tb3_lab.sh
```

To change the output directory:

```bash
SAVE_DIR=/scratch/$USER/ma_vlcm_tb3_checkpoints \
  sbatch scripts/lora_submit_train_tb3_lab.sh
```

## 4. Running Inference

```bash
bash scripts/run_inference_vlcm.sh
```

Inference results now default to:

- `outputs/results/`
- `outputs/plots/inference/`

For physical TurtleBot3 live monitoring, run this on the operator machine with
the ROS 2 workspace sourced or available at `/home/adi2440/turtlebot_ws`:

```bash
bash scripts/run_tb3_vlcm_live_monitor.sh checkpoints/NewFinal_0.5B.pt
```

This starts the MARL policy launch, MA-VLCM live inference, and the live plot of
prediction versus cumulative reward.

## 5. Key Launcher Arguments

The training launchers in `scripts/` pass arguments like:

| Argument | Meaning |
| :--- | :--- |
| `--train_shards` | Path or glob for WebDataset `.tar` shards |
| `--offroad_shards` | Optional OFFROAD shard directory or glob |
| `--rware_config` | Label/config name used in prompts and logging |
| `--batch_size` | Per-GPU batch size |
| `--grad_accum_steps` | Gradient accumulation steps |
| `--clip_len` | Number of frames per clip |
| `--num_robots` | Padded maximum number of agents |
| `--robot_obs_dim` | Per-agent feature dimension |
| `--vl_backend` | Vision-language backend |
| `--vl_model_name` | Hugging Face model id |
| `--mixed_precision` | `bf16`, `fp16`, or `no` |
| `--peft` | LoRA/QLoRA mode |
| `--gamma` | Discount factor |
| `--max_return_horizon` | Cap on n-step return horizon |

## 6. Notes

- The shell launchers set `PYTHONPATH="$REPO_ROOT/src"` automatically.
- If you run the Python modules directly, set `PYTHONPATH` yourself or install the package with `pip install -e .`.
- If you add dependencies, rebuild the container or install them at runtime inside the Apptainer image.
