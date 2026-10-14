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

## Combined Goal-To-Goal And Obstacle Critic Training

The Qwen3-VL and V-JEPA2 launchers default to a balanced two-source TurtleBot
training set:

```text
goal_to_goal:
  hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar
static_obstacles:
  hf://datasets/adi2440/tb3-isaac-avoid-obstacles-vlcm/**/*.tar
```

The legacy goal-to-goal repository contains 3-, 4-, 5-, and 6-agent episodes;
the current obstacle repository contains 3-agent episodes. The loader
recognizes the legacy repository name as `goal_to_goal`, recognizes
zero-padded paths such as `agents_03`, and makes an episode-level split for
every observed domain/cardinality group. Training balances the two task
domains while preserving every available cardinality; the smaller obstacle
domain is repeated to match the goal-to-goal domain. Override the sources later
with a semicolon-delimited `TB3_TRAIN_SOURCES` value.

Both launchers use one physical target definition, `tb3_progress_v2`:

```text
mean_i clamp((initial_goal_distance_i - current_goal_distance_i)
             / max(initial_goal_distance_i, epsilon), 0, 1)
```

For legacy `tb3_progress_v1` shards, the initial distance is taken from the
first frame and the v2 target is recomputed online; the old stored target is
not mixed with the new definition. `SUCCESS_ONLY` defaults to `0` because the
legacy repository does not attach final episode success to every frame. Set it
to `1` only after relabeling all sources with episode-wide success metadata.

Camera inputs are canonicalized before either backbone sees them. The legacy
640x360 frame is center-cropped to its square world-aligned workspace; the
newer square Isaac frame is unchanged. Both are then resized to 336x336. No
rotation or reflection is applied. The audited convention for both sources is
world `+X` to image-right and world `+Y` to image-up, with coordinates in
meters.

Qwen3-VL receives a canonical prompt on every clip that states the domain,
exact number of agents, exact number of static obstacles, task instruction,
and world/image axis convention. Goal-to-goal prompts explicitly say that no
static obstacles are present; obstacle prompts explicitly require avoiding all
three static obstacles and teammate collisions. V-JEPA2 is intentionally the
video-only comparison and therefore does not consume language; it distinguishes
the tasks through canonical video and graph/state inputs.

After the obstacle repository upload is complete, submit directly with:

```bash
sbatch scripts/lora_submit_train_tb3_vjepa2.sh
sbatch scripts/lora_submit_train_tb3_qwen3_vl.sh
```

Configuration-only checks do not allocate a GPU:

```bash
DRY_RUN=1 bash scripts/lora_run_train_tb3_vjepa2.sh
DRY_RUN=1 bash scripts/lora_run_train_tb3_qwen3_vl.sh
```

## Common Overrides

Use a different Hugging Face repository:

```bash
HF_DATASET_REPO=owner/dataset \
DATASET_PROFILE=tb3_isaac \
bash scripts/lora_run_train_tb3_lab.sh
```

For private repositories, export a token without placing it in a script:

```bash
export HF_TOKEN="<your Hugging Face token>"
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

## `tb3_progress_v2` reward-model workflow

The visual MARL reward oracle uses nine values per robot:

```text
[x, y, cos(yaw), sin(yaw), v, w,
 current_goal_distance, min_neighbor_distance, initial_goal_distance]
```

Relabel raw episode shards before stage-one training. The v2 target is
`mean(clamp((initial_distance-current_distance)/initial_distance, 0, 1))`.
The relabeler stores initial distances and episode-wide success metadata in
every frame label, allowing full successful trajectories—not just their last
frames—to pass `--success_only`.

```bash
python scripts/relabel_tb3_progress_dataset.py RAW_SHARDS \
  --output-dir data/tb3_progress_v2

HF_DATASET_REPO=owner/balanced-gtg-obstacles-3to5 \
  bash scripts/lora_run_train_tb3_vjepa2.sh
```

The V-JEPA2 and Qwen3-VL launchers default to `tb3_progress_v2`, 9-D robot
rows, sigmoid progress output, all available 3/4/5/6-agent goal-to-goal data,
the current 3-agent static-obstacle data, and domain-balanced episode splits.
Success-only loading is disabled for compatibility with the legacy labels.
When additional obstacle cardinalities are ready, keep paths organized by task
domain and cardinality, for example `static_obstacles/agents_4/*.tar`. The
loader prevents cross-split shard leakage, preserves native cardinality, and
pads only within each batch using the adjacency diagonal as the node mask.

`ma_vlcm.reward_refinement` provides the stage-two dual objective and ordering
contract: absolute frame-progress MSE plus Bradley–Terry loss on matched
same-task trajectory pairs ordered by terminal success, final team progress,
then completion time. The model also contains a zero-initialized success head;
enable its BCE only with `--success_loss_weight`, and keep learned termination
disabled for the MVP. `temporal_order_accuracy` and
`value_order_correlation` implement the transferable TOPReward criterion for
successful prefixes. Token-probability scoring is intentionally not applied to
V-JEPA2.


## Signed scene potential experiment (`tb3_potential_v1`)

The new opt-in path trains a scene potential in `[-1, 1]` with a `tanh`
output. It does not interpret the potential itself as an instantaneous reward.
For agent i, let q_i be remaining distance outside the goal radius, divided by
its initial remaining distance (floored at the goal radius for short episodes).
Let C_agent and C_obstacle be the maximum squared clearance-risk ramp across
robot pairs and robot/obstacle pairs. A ramp is zero outside the configured
clearance band and reaches one at contact. Then:

```text
U = mean(q_i) + agent_weight * C_agent + obstacle_weight * C_obstacle
Phi = (1 - U) / (1 + U)
```

Safe completion gives Phi=1; a safe initial state typically gives Phi=0;
retreat and increasing risk lower Phi. A collision flag or geometric overlap
sets Phi=-1. Worst-case hazard aggregation prevents one dangerous pair from
being diluted by a larger team. Goal attraction remains a team average: one
agent's progress can outweigh another's retreat. The configurable weights are
initial experiment settings, not calibrated safety guarantees. Potential fields
can have local optima, and Euclidean attraction can penalize a necessary detour.
Use `distance_mode=route_required` when verified route distances are available
for every frame; set `PROGRESS_DISTANCE_MODE=route_required` during training too.

This target replaces the supervised progress label. `reward.json` is not the
training target in `target_mode=progress`; changing that file alone has no effect.
The relabeler preserves original image/state/action/reward bytes in new shards,
replacing only `progress.json`. It refuses to overwrite a source or destination.
Model prompts for this schema describe signed scene quality and omit timestep,
future episode outcome, and termination reason. Endpoint prompts match each
clip's final state. Resume detects schema, activation, or potential-config
changes and resets the value head, optimizer, scheduler, EMA, and epoch count.

Relabel an existing locally cached dataset:

```bash
python scripts/relabel_tb3_potential_dataset.py /path/to/source_shards \
  --output-dir /path/to/new_potential_shards \
  --config configs/potential_v1.json --include-rewind
```

Rewinds append synthetic reversed episodes inside the same shard as their
source, so the normal shard split cannot put the original and rewind on
opposite sides. Rewind velocity/command signs are reversed, original action
arrays are omitted, and samples are marked synthetic. These examples teach
undoing visible progress; they are not simulator-validated collision rollouts
and cannot supply unseen near-collision or obstacle-contact scenes. Include
real failed, stalled, near-collision, and recovery rollouts when available.
The adjacent temporal loss matches signed target changes; it does not impose
monotonic progress. Evaluate held-out retreat, near-contact, collision, stall,
and recovery cases in addition to successful-trajectory correlation.

Obstacle tasks require a verified layout in `state.json` as `obstacles`, or
`--layouts` with entries such as:

```json
{
  "static_obstacles/agents_03/worker_00/episode_000066.tar::ep000066": {
    "obstacles": [{"pos": [0.6, 0.6], "size_m": [0.3, 0.3]}]
  }
}
```

Each key is the shard path relative to the input directory, followed by `::`
and its episode prefix. Circles use `radius_m`; rectangles use `size_m` and must
be axis-aligned. Coordinates and robot radii must share the dataset's world
frame. Missing/empty geometry for an obstacle task is an error. The sampled
Hugging Face obstacle shard lacks layout and route metadata; its seed alone
has not been verified sufficient to reconstruct the original simulator layout.
Do not label unknown layouts using candidate sites from a different code version.

Train only after relabeling and reviewing label/delta audits:

```bash
BACKBONE_PROFILE=vjepa2 bash scripts/lora_run_train_tb3_potential.sh \
  /path/to/new_potential_shards
```

Use `BACKBONE_PROFILE=qwen3_vl` for Qwen. This separate launcher keeps the
existing progress launchers and checkpoints available. Match its potential
config to the one used for relabeling. The normal offline inference path reads
checkpoint activation metadata and supports tanh; live inference also decodes
tanh, but its legacy ground-truth progress plot remains a different quantity.
No training or data upload is performed by adding this feature.

The companion Palmetto MARL wrapper decodes tanh checkpoints and uses
Psi=(Phi+1)/2 for discounted shaping, `scale * (gamma * Psi_next - Psi_current)`.
The affine shift avoids paying positive rewards for a stationary negative Phi.
Signed checkpoints require `vlcm_progress_delta_terminal` and smoothing=0.
True absorbing terminations have next potential zero; time-limit truncations
retain the final observation's potential (using existing bad-mask metadata).
The success/failure anchors provide the task objective. Training a state
potential does not train a policy-specific V/Q return critic; MAPPO's separate
return head still learns returns from rollout rewards. Persistent safety costs
and failure outcomes need their own objective if repeated dangerous states must
continue receiving penalties after the potential saturates.

Relevant distinctions: [VLAC](https://arxiv.org/html/2509.15937) samples both
frame-pair directions. [RoboMeter](https://arxiv.org/html/2603.02115) combines
progress, success, trajectory preferences, and video rewinds. This experiment
uses geometry targets and rewinds, without claiming to reproduce their models
or preference objectives. [SimpleVLA-RL](https://arxiv.org/html/2509.09674) uses
binary trajectory success, while [TOPReward](https://arxiv.org/html/2602.19313)
uses pretrained completion-token probabilities; neither implies that simply
rescaling progress to [-1,1] resolves failure coverage.


### Mandatory combined Isaac critic training

**ALWAYS** train both VJEPA and Qwen VLCM critics with BOTH
`adi2440/tb3-isaac-vlcm` (go-to-goal) and
`adi2440/tb3-isaac-avoid-obstacles-vlcm` (obstacle avoidance).
The backbone launchers and signed-potential launcher enforce both sources,
set both task domains, and enable equal domain sampling. Local relabeled data
must contain `goal_to_goal` and `static_obstacles` shard directories. Missing
cardinalities within the obstacle dataset are allowed; missing domains are not.
The signed training entry point independently checks this contract.

These launchers start fresh VLCM training with `TRAIN_FROM_SCRATCH=1` and never
resume a previous critic checkpoint. Standard pretrained VJEPA/Qwen backbone
weights remain the initialization; critic head, adapters, optimizer, and
scheduler are new. Keep the video clip length at 16. For the combined signed
runs, use batch 32 and four gradient accumulation steps (effective batch 128).
W&B records exact PyTorch current and peak allocated/reserved GPU memory in
GiB, alongside sampled system memory, to support later batch tuning.

Legacy obstacle states lack layouts. `recover_tb3_obstacle_layouts.py` can
recover the three occupied sites from their recorded brown cuboids using the
configured top-down camera (height 5 m, ground width 3.048 m), 0.3 m square
boxes of height 1 m, and five candidate sites. It checks every video frame,
requires an unchanged unambiguous layout, and rejects unsupported canvases or
unexplained brown geometry. The recovered geometry is explicitly audited as
image-derived rather than simulator ground truth. Pass its layout manifest to
the relabeler; do not silently disable obstacle repulsion on these episodes.
Future collection saves the actual simulator layouts directly.


VJEPA host-memory settings: use at least 128 GiB for the current batch-32
training job, four data-loader workers, `LOADER_PREFETCH_FACTOR=1`, and
`DISABLE_PERSISTENT_WORKERS=1`. GPU memory and host RAM are separate budgets.
The previous 64 GiB job completed epoch 1 but exhausted host RAM at epoch 2;
training and validation had eight persistent workers each with two prefetched
batches per worker. The new settings release phase workers and limit queued
batches. Do not infer the host-memory requirement from W&B GPU memory or the
main Python process alone; include all workers and pinned buffers.


### Signed-potential episode inference

Use `scripts/run_tb3_episode_inference_all.sh` with explicit
`QWEN3_VL_CHECKPOINT` and `VJEPA2_CHECKPOINT` paths for new run checkpoints.
Inference preserves the saved dataset type, target schema, and output
activation. Signed plots show the full [-1, 1] range.

Relabeled training shards contain both a demonstration and its synthetic
rewind. Extract each episode into a separate evaluation shard before using
this full-episode evaluator; mixed episodes are rejected to prevent step
collisions. Select them using the original training split manifest, since
resplitting a domain independently changes held-out membership. Synthetic
rewinds are labeled as such, not as real failed-policy episodes.

Legacy progress-model and signed-model native errors use different targets.
Report their ranges and compare score-change signs against the same signed
target transitions when assessing downstream reward usefulness.

## Conditional temporal regularization for signed critics

The Qwen and VJEPA signed launcher now defaults to
`TEMPORAL_CONSISTENCY_WEIGHT=1.0` (previously 0.1) and
`CONDITIONAL_LIPSCHITZ_WEIGHT=1.0`. Both are tunable initial settings, not
validated improvements. Both datasets, fresh critic initialization, batch 32,
four accumulation steps, and 16-frame clips remain required.

The adjacent clips end at t and t+1. With predicted potential f and target Phi:

```
L = 0.5 * [MSE(f_t, Phi_t) + MSE(f_next, Phi_next)]
  + lambda_T * SmoothL1(f_next - f_t, Phi_next - Phi_t; beta=0.02)
  + lambda_L * mean_valid[relu(abs(f_next - f_t) - d(x_t,x_next))^2]
```

Temporal supervision permits both signs of change, including collision-driven
decreases. It does not penalize a legitimate decrease merely for decreasing.
The Lipschitz penalty is one-sided: flatter-than-unit slopes are allowed.
It is a sampled soft constraint, not a certificate of global 1-Lipschitzness.
One-sided penalty rationale: https://arxiv.org/abs/1709.08894.

The condition fixes episode, task/domain/instruction, agent identities/goals,
and obstacle geometry. Same-named episodes in different shards, step resets,
and shard wraparound cannot produce temporal training pairs. Dynamic observed
state text may change within a pair. Pairs with changed static conditions are
excluded from the Lipschitz penalty. Changes of collision status are also
excluded because the existing target has a discontinuous override to -1; they
still receive endpoint and signed-delta supervision. Target semantics and
collision penalties have not been changed.

`d` is a fixed joint input metric, never a trainable feature norm or a target
delta. It combines RGB video RMS at 32x32 across all 16 frames, last-frame
state distance (matching the model's graph consumption), and last adjacency
RMS. Default state scales are 0.1m for positions/current and initial goal
distances, 1 for cos/sin yaw and traversability, 0.22m/s for linear velocity,
and 2.84rad/s for angular velocity. State squared distances are averaged over
nodes, not feature dimensions. Video values are in [0,1], with RMS divided by
0.1; adjacency has unit scale. The joint distance is the square root of the
sum of modality squared distances. Units matter: a 1-Lipschitz claim is always
relative to this declared input metric. Change the spatial unit through
`LIPSCHITZ_POSITION_SCALE_M` and RGB unit through `LIPSCHITZ_PIXEL_SCALE`.

The loader sends only the scalar distance and validity mask to the GPU. Both
losses reuse the existing adjacent-clip model pass, without a third pass or
second-order gradients through the vision-language backbone.

W&B logs separate train and validation epoch MSE, temporal and Lipschitz terms,
valid/excluded pair counts, prediction violation rates, and target violation
rates. Counts/components reduce over distributed ranks. Audit target violation
rates before tightening the metric: too-tight bounds can conflict with hazard
labels. The two illustrative held-out forward episodes checked on 2026-10-05
had no target or prediction bound violations under the initial metric, despite
the latest critics' missed target decreases. This small audit is not full-data
calibration and does not establish that Lipschitz regularization will improve
accuracy. Stronger delta supervision must be evaluated on held-out trajectories.

Regression checks:

```
PYTHONPATH=src python -m pytest -q tests/test_critic_regularization.py \
  tests/test_signed_potential_loader.py tests/test_signed_potential.py \
  tests/test_tb3_training_contract.py tests/test_tb3_combined_compatibility.py \
  tests/test_training_sampling.py
PYTHONPATH=src python tests/test_tb3_lab_loader.py
bash -n scripts/*.sh
```


### Categorical signed-potential critic (2026-10-05)

The signed-potential launcher defaults to categorical cross-entropy over 21
uniform centers from -1 to 1. Soft targets interpolate between adjacent centers;
inference returns their probability-weighted expectation without an extra tanh.
The labels remain geometry-derived `tb3_potential_v1`, not episode time.
Both adjacent endpoints receive categorical supervision; temporal delta and
conditional sampled Lipschitz losses operate on the expected values. W&B logs
CE separately from diagnostic scalar MSE and regularizer components.
`LOSS_TYPE=mse` explicitly retains the previous scalar/tanh baseline.
ALWAYS use both goal and obstacle datasets, fresh critic/adapters/optimizer,
and 16-frame clips. `VAL_SHARDS` supports an explicit episode holdout.
The categorical head size and support are saved in checkpoint arguments and
restored by offline inference; old scalar checkpoints retain their old shape.


### Qwen 8-frame, 75-epoch experiment (2026-10-06)

The user-requested `scripts/lora_submit_train_tb3_qwen8_75.sh RUN_ROOT`
experiment uses 8-frame clips and 75 epochs, fresh critic/adapters/optimizer,
and ALWAYS both `adi2440/tb3-isaac-vlcm` and
`adi2440/tb3-isaac-avoid-obstacles-vlcm`, including the verified 500 appended
suboptimal episodes and the preserved validation split. Batch size is 32,
accumulation is 4, and the sampling budget is 5,000 clips per epoch. Keep the
21-bin categorical signed-potential objective and both regularizer weights
at 1.0. Save and validate every epoch, selecting checkpoints by validation
quality and responses to real negative changes. VJEPA remains on standby.
This explicitly authorized experiment overrides the usual 16-frame default.


### Qwen full accuracy run (2026-10-06)

See [README_full_run.md](README_full_run.md) for the fresh 8-frame, 224x224,
75-epoch experiment, ALWAYS both datasets, profiling and matched ablations.

### Smooth APF + trajectory preference iteration (2026-10-08)

The isolated `qwen_smooth_pref_diverse1000_20261008_01` run ALWAYS trains on both Isaac datasets, fresh Qwen only, 8 frames/224 pixels/75 epochs. Targets use causal Gaussian sigma 1/window 5 smoothing with immediate collision override; raw APF labels remain available. `--apf_target_transform` must match the labels. Optional `--preference_loss_weight`, `--preference_shards` and `--preference_interval` enable a separate pooled-trajectory preference head and BCE, supervised from recorded outcomes. These labels never enter model inputs. Full collection/training contract is in the run's `README_iteration.md`.


## Optional relational Qwen critic

The native224 iteration uses `--relational_critic --critic_readout query --agent_pooling attention_max`: causal histories of all agent observations, individual `<obs>` tokens, visual cross-attention and masked agent attention. It accepts the same video/prompt/robot-observation/adjacency inputs. Checkpoint arguments retain the architecture flag, LoRA rank and preprocessing configuration for standard offline inference and `tools/evaluate_tb3_episode_progress.py`. Defaults preserve legacy checkpoints. Validation clip thinning is disabled for offline episode inference. Native AO collection configures the actual overhead camera and recorded image to 224x224, with the actor observation shape unchanged. Both HF task domains are mandatory for every critic training run.

### Experiment tracking and completed relational evaluation

The [run register](docs/experiments/vlcm_run_register.md) records exact W&B names,
configuration changes, dataset revisions and completed inference comparisons.
The [relational checkpoint report](docs/experiments/relational_native224_20261014/README.md)
compares epochs22/63/75 on100 held-out forward episodes, split equally between
go-to-goal and avoid-obstacles. It includes plots and metrics; checkpoints and raw
trajectories remain outside Git.

The validated75-epoch native224 launch/profile and matched-evaluation scripts are
archived in `scripts/experiments/relational_native224/`. These are the exact
Palmetto experiment drivers, with recorded run paths and readiness checks; they
do not submit jobs automatically. Collection-only helpers are in
`tools/collection/`: native camera capture preserves the actor image shape, and
simulated action perturbations include timed recovery. The Isaac collector
integration remains in the Ground_Robot_MARL project.

## Required TB3 task-only text contract

TB3 critic text contains only the task instruction. Frame-wise collision/reached
flags, outcomes, step counters, current state descriptions and reward components
must never be appended to it. Numerical robot observations and adjacency remain
separate model inputs. All TB3 training and inference entry points share
`ma_vlcm.tb3_task_prompt`; literal `--text_prompt_template` overrides receive no
state suffix. Fresh checkpoints record `text_input_contract=task_instruction_only_v1`.

The earlier runs in the [register](docs/experiments/vlcm_run_register.md) used
legacy state-augmented text. Their collision results do not validate a clean
vision-language critic. See the [input audit and paper comparison](docs/experiments/task_only_input_audit.md).
Existing stored labels remain usable; train a fresh critic with the corrected
loader. Legacy checkpoint inference warns about the changed text distribution.
