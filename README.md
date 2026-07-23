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

## Backbone Options

The TurtleBot training path supports three backbone profiles:

| Profile | Default model | Modalities used by the critic | Training script |
| --- | --- | --- | --- |
| `llava_onevision` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | video + prompt + robot graph | `lora_run_train_tb3_lab.sh` |
| `qwen3_vl` | `Qwen/Qwen3-VL-2B-Instruct` | video + prompt + robot graph | `lora_run_train_tb3_qwen3_vl.sh` |
| `vjepa2` | `facebook/vjepa2-vitl-fpc64-256` | video + robot graph | `lora_run_train_tb3_vjepa2.sh` |

Qwen3-VL is the larger vision-language comparison. V-JEPA2 is the visual
representation baseline: it intentionally does not consume the language
prompt, and its pooled video tokens are fused directly with the shared GNN team
feature. All three profiles predict the same bounded progress target and remain
independent of the number of robots.

Run Qwen3-VL on the physical dataset:

```bash
bash scripts/lora_run_train_tb3_qwen3_vl.sh
```

Run V-JEPA2 on the Isaac Sim dataset:

```bash
DATASET_PROFILE=tb3_isaac bash scripts/lora_run_train_tb3_vjepa2.sh
```

The corresponding Slurm launchers default to the Isaac Sim dataset:

```bash
mkdir -p logs
sbatch scripts/lora_submit_train_tb3_qwen3_vl.sh
sbatch scripts/lora_submit_train_tb3_vjepa2.sh
```

Select the physical dataset for either Slurm launcher with, for example:

```bash
DATASET_PROFILE=tb3_lab sbatch scripts/lora_submit_train_tb3_qwen3_vl.sh
```

Qwen3-VL defaults to batch size 1 with 16 accumulation steps. V-JEPA2 defaults
to batch size 2 with 8 accumulation steps. Override these with `BATCH_SIZE`,
`GRAD_ACCUM_STEPS`, or `CLIP_LEN`. New Qwen3-VL and V-JEPA2 runs start from
their Hugging Face backbone weights; a LLaVA MA-VLCM checkpoint is not a valid
resume checkpoint for either architecture.

### Fine-tuning modes

`FINETUNE_MODE` controls exactly which pretrained parameters are optimized. The
GNN and progress/value head are trained in every mode.

| `FINETUNE_MODE` | Qwen/LLaVA backbone | V-JEPA2 backbone |
| --- | --- | --- |
| `lora` (default) | language LoRA + vision LoRA | vision LoRA |
| `qlora` | quantized language LoRA + vision LoRA | unsupported |
| `language_lora` | language LoRA; vision frozen | unsupported |
| `vision_lora` | vision LoRA; language frozen | vision LoRA |
| `full` | full language and vision backbone | full video backbone |
| `vision_full` | full vision tower; language frozen | full video backbone |
| `heads_only` | entire pretrained backbone frozen | entire video backbone frozen |

With the default dimensions and LoRA rank 16, the expected trainable counts are:

| Mode | Qwen3-VL-2B | V-JEPA2 ViT-L |
| --- | ---: | ---: |
| `lora` | about 20,430,593 | about 2,429,697 |
| `language_lora` | about 18,030,337 | unsupported |
| `vision_lora` | about 2,998,017 | about 2,429,697 |
| `full` | about 2,128,131,841 | about 303,955,713 |
| `vision_full` | about 407,554,817 | about 303,955,713 |
| `heads_only` | 597,761 | 70,401 |

These include the shared GNN and progress head. For Qwen they also include the
robot-to-language projection. V-JEPA2's unused predictor branch stays frozen
because the critic calls the encoder with `skip_predictor=True`.

Examples:

```bash
# Full Qwen3-VL fine-tuning (high GPU-memory use)
FINETUNE_MODE=full DATASET_PROFILE=tb3_isaac \
  bash scripts/lora_run_train_tb3_qwen3_vl.sh

# Only Qwen's visual LoRA plus the MA-VLCM graph/progress layers
FINETUNE_MODE=vision_lora DATASET_PROFILE=tb3_isaac \
  bash scripts/lora_run_train_tb3_qwen3_vl.sh

# Frozen V-JEPA2 feature extractor; train only GNN and progress head
FINETUNE_MODE=heads_only DATASET_PROFILE=tb3_isaac \
  bash scripts/lora_run_train_tb3_vjepa2.sh
```

For targeted language/V-JEPA attention adapters, set a comma-separated module
suffix list with `LORA_TARGET_MODULES`, for example
`LORA_TARGET_MODULES=q_proj,v_proj`. Each run prints the exact trainable count
for the vision backbone, language backbone, GNN, robot-to-language projection,
and value head before training starts. Custom heads use `HEAD_LR` (default
`3e-4`), a fully trainable language backbone uses `BACKBONE_LR` (default
`1e-5` in `full` mode), and visual parameters use `VISION_LR` (default `1e-5`).

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
| `SAMPLES_PER_EPOCH` | Global number of clip samples consumed per epoch (default `5000`) |
| `VAL_SPLIT` | Fraction of episode shards held out for validation (default `0.2`) |
| `SPLIT_SEED` | Deterministic episode-shard split seed (default `42`) |
| `SAVE_DIR` | Checkpoint output directory |
| `MA_VLCM_SCRATCH_ROOT` | Root for caches, temporary files, W&B, and checkpoints |
| `NUM_PROCESSES` | Number of Accelerate processes |
| `MIXED_PRECISION` | Usually `bf16`, `fp16`, or `no` |
| `BACKBONE_PROFILE` | `llava_onevision`, `qwen3_vl`, or `vjepa2` |
| `VL_MODEL_NAME` | Override the Hugging Face model within a profile |
| `BATCH_SIZE` | Per-process minibatch size |
| `GRAD_ACCUM_STEPS` | Optimizer gradient accumulation steps |
| `CLIP_LEN` | Frames supplied to the video backbone |
| `FINETUNE_MODE` | Select `lora`, `full`, `vision_lora`, `heads_only`, or another mode above |
| `LORA_TARGET_MODULES` | Optional comma-separated language/V-JEPA LoRA module suffixes |
| `HEAD_LR` | Learning rate for the GNN, fusion projection, and progress/value head |
| `BACKBONE_LR` | Learning rate for trainable non-vision language-backbone parameters |
| `VISION_LR` | Learning rate for trainable visual-backbone parameters |

## Training Behavior

The default TurtleBot launcher uses:

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

Remote Hugging Face shard patterns are expanded before splitting, so complete
episode shards—not neighboring clips—are assigned to train or validation. Each
epoch logs its consumed batch count and observed unique episode-ID count.

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

Compare complete held-out episodes with the latest saved LLaVA-OneVision,
Qwen3-VL, and V-JEPA2 epoch checkpoints:

```bash
bash scripts/run_tb3_episode_inference_all.sh
```

Submit the same evaluation as a one-GPU A100 Slurm job:

```bash
mkdir -p logs
LLAVA_CHECKPOINT=/path/to/llava_epoch_20.pt \
sbatch scripts/submit_tb3_episode_inference_all.sh
```

Environment overrides are inherited by the job, and extra evaluator arguments
after the submission-script path are forwarded to the inference launcher.

The launcher deterministically selects five episode shards from the same 20%
episode-level validation split used by training. It processes every sliding
16-frame clip through each model sequentially, so only one backbone occupies
GPU memory at a time. Results are organized under:

```text
outputs/plots/tb3_episode_inference/run_YYYYMMDD_HHMMSS/
├── manifest.json
├── model_metadata.json
├── summary.json
├── episode_01_<source>/
│   ├── progress.csv
│   ├── progress.png
│   ├── episode_progress.mp4
│   └── summary.json
└── ...
```

Each animation shows the original overhead episode video above a growing plot
of ground-truth normalized progress and all three model predictions. If FFmpeg
is unavailable, the evaluator writes a GIF instead.

Checkpoint discovery chooses the newest run timestamp for each backbone and
then the highest epoch saved by that run. Override any model or other setting
with environment variables, for example:

```bash
LLAVA_CHECKPOINT=/path/to/llava_epoch_20.pt \
NUM_EPISODES=5 VIDEO_FPS=5 \
bash scripts/run_tb3_episode_inference_all.sh
```

The current `tb3_isaac` scratch folder contains timestamped Qwen3-VL and
V-JEPA2 epoch checkpoints but no timestamped LLaVA epoch checkpoint, so provide
`LLAVA_CHECKPOINT` if it is stored elsewhere. Use `NO_VIDEO=1` for CSV/PNG-only
evaluation or pass additional Python options after the launcher name.

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
