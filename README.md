# MA-VLCM

MA-VLCM is a multi-agent vision-language critic. It combines a short overhead
video, text prompt, per-robot state, and a robot adjacency graph to predict one
team-level scalar.

- TurtleBot3 fine-tuning predicts bounded task progress in `[0, 1]`.
- During MAPPO, the standalone critic wrapper keeps that progress prior and
  adds a separate trainable RL-return head.
- GNN parameters are shared across robots, so one checkpoint supports variable
  team sizes.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Without an editable installation, run commands from the repository root with:

```bash
export PYTHONPATH="$PWD/src"
```

## TurtleBot3 Progress Fine-Tuning

Both TurtleBot datasets use [lora_run_train_tb3_lab.sh](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/scripts/lora_run_train_tb3_lab.sh). The selected profile changes the dataset, checkpoint directory, and W&B run prefix.

| Dataset | Profile | Default Hugging Face repository | Slurm script |
| --- | --- | --- | --- |
| Physical TurtleBot3 lab | `tb3_lab` | `adi2440/tb3-lab-vlcm-progress-v1` | `lora_submit_train_tb3_lab.sh` |
| Isaac Sim TurtleBot3 | `tb3_isaac` | `adi2440/tb3-isaac-vlcm` | `lora_submit_train_tb3_isaac.sh` |

### Physical TurtleBot3 dataset

The lab profile is the default:

```bash
bash scripts/lora_run_train_tb3_lab.sh
```

It loads:

```text
hf://datasets/adi2440/tb3-lab-vlcm-progress-v1/**/*.tar
```

To relabel older reward-based lab shards with `tb3_progress_v1`:

```bash
python scripts/relabel_tb3_progress_dataset.py \
  hf://datasets/adi2440/tb3-lab-vlcm/*.tar \
  --output-dir data/tb3_lab_progress_v1 \
  --repo-id adi2440/tb3-lab-vlcm-progress-v1
```

Omit `--repo-id` to write only local shards.

### Isaac Sim TurtleBot3 dataset

Run locally or interactively with:

```bash
DATASET_PROFILE=tb3_isaac bash scripts/lora_run_train_tb3_lab.sh
```

It loads:

```text
hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar
```

To train directly from a local Isaac collection:

```bash
DATASET_PROFILE=tb3_isaac bash scripts/lora_run_train_tb3_lab.sh \
  /home/adi2440/Desktop/MARL_Shahil_Aditya/VLCM_Data_Collection/TURTLEBOT/data/tb3_isaac_vlcm
```

### Slurm

Create the log directory once before submitting:

```bash
mkdir -p logs
```

Submit the physical-lab dataset:

```bash
sbatch scripts/lora_submit_train_tb3_lab.sh
```

Submit the Isaac Sim dataset:

```bash
sbatch scripts/lora_submit_train_tb3_isaac.sh
```

Both jobs request one H100, 32 CPU cores, 128 GB RAM, and 48 hours by default.
Edit the `#SBATCH` header if the cluster uses different resource names.

## Common Overrides

Use a different Hugging Face repository:

```bash
HF_DATASET_REPO=owner/dataset \
DATASET_PROFILE=tb3_isaac \
bash scripts/lora_run_train_tb3_lab.sh
```

For private repositories, export a token without placing it in a script:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Resume from a specific checkpoint:

```bash
RESUME_CHECKPOINT=/path/to/checkpoint.pt \
bash scripts/lora_run_train_tb3_lab.sh
```

Train without a pretrained checkpoint:

```bash
TRAIN_FROM_SCRATCH=1 bash scripts/lora_run_train_tb3_lab.sh
```

The same variables can be passed to `sbatch`:

```bash
TOTAL_EPOCHS=30 \
SAVE_DIR=/scratch/$USER/ma_vlcm/checkpoints/tb3_isaac \
sbatch scripts/lora_submit_train_tb3_isaac.sh
```

Useful variables:

| Variable | Purpose |
| --- | --- |
| `DATA_DIR` | Local directory, shard glob, or `hf://` pattern |
| `HF_DATASET_REPO` | Hugging Face dataset repository |
| `RESUME_CHECKPOINT` | Pretrained or intermediate `.pt` checkpoint |
| `TRAIN_FROM_SCRATCH=1` | Disable checkpoint resume |
| `TOTAL_EPOCHS` | Final epoch target |
| `SAVE_DIR` | Checkpoint output directory |
| `MA_VLCM_SCRATCH_ROOT` | Root for caches, temporary files, W&B, and checkpoints |
| `NUM_PROCESSES` | Number of Accelerate processes |
| `MIXED_PRECISION` | Usually `bf16`, `fp16`, or `no` |

## Training Behavior

The TurtleBot launcher uses:

- LLaVA-OneVision Qwen2 0.5B
- 16-frame clips
- 8-D robot observations
- LoRA for language and vision attention layers
- `target_mode=progress`
- sigmoid output and MSE loss
- dynamic robot cardinality

Each episode retains its native number of robots. Mixed-cardinality minibatches
are padded only to the largest team in that minibatch. A zero adjacency diagonal
marks padded nodes, excluding them from message passing and team pooling.

Checkpoints default to:

```text
$MA_VLCM_SCRATCH_ROOT/checkpoints/tb3_lab
$MA_VLCM_SCRATCH_ROOT/checkpoints/tb3_isaac
```

The launcher uses Apptainer when `ma_vlcm.sif` is available; otherwise it runs
natively. See [README_HPC.md](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/README_HPC.md) for container and cluster setup.

## Input And Target

For a clip beginning at time `t`, the model receives:

```text
x_t = (V_t, p_t, O_t, A_t)
```

- `V_t`: overhead video frames
- `p_t`: task prompt
- `O_t`: robot observations with shape `[T,N,8]`
- `A_t`: adjacency matrices with shape `[T,N,N]`

The TurtleBot robot row is:

```text
[x, y, cos(yaw), sin(yaw), v_linear, v_angular,
 distance_to_goal, min_neighbor_distance]
```

The model uses the last robot state and graph in the clip, while the vision
backbone processes the video. The offline TurtleBot target is read from
`progress.json`:

```text
agent_progress_i = clamp(
  (initial_dist_i - max(dist_i - goal_radius, 0)) /
  max(initial_dist_i - goal_radius, eps),
  0,
  1
)

team_progress = mean(agent_progress_i)
```

The target becomes `1` for clean success and `0` for terminal failure or an
unsafe collision; otherwise it is the mean team progress.

## Other Training Profiles

The generic RWARE/OFFROAD launcher remains available:

```bash
bash scripts/run_train_vlcm.sh
```

It supports return targets, n-step bootstrapping, contrastive losses, and the
alternate launchers under `scripts/`. Run the full module help for all options:

```bash
PYTHONPATH=src python -m ma_vlcm.train --help
```

## Inference

Offline evaluation:

```bash
bash scripts/run_inference_vlcm.sh
```

Physical TurtleBot3 live monitoring:

```bash
bash scripts/run_tb3_vlcm_live_monitor.sh checkpoints/draft_1_tb3_lab.pt
```

The live workflow consumes the overhead camera and TurtleBot pose/velocity/goal
topics, publishes `/fleet_vlcm/vlcm_prediction`, and writes
`outputs/results/tb3_live_predictions.csv`. It also starts the robot policy, so
run it only from the configured ROS 2 operator machine.

## Repository Layout

- `src/ma_vlcm/`: model, training, inference, and live inference code
- `scripts/`: local, Slurm, inference, upload, and live-monitor launchers
- `tests/`: focused loader tests
- `tools/`: visualization and analysis utilities
- `outputs/`: generated plots, videos, and result files
- `checkpoints/`, `data/`, `logs/`, `wandb/`: generated or downloaded artifacts

Useful checks:

```bash
PYTHONPATH=src python tests/test_tb3_lab_loader.py
bash -n scripts/*.sh
```
