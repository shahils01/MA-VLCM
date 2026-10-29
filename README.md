# MA-VLCM: SmolVLM formation value estimator

The main Python files train a signed scalar value from overhead images, the task
instruction, robot states, and dynamic adjacency. The implementation uses
[SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct),
which is based on Idefics3. It does not use the `pilot_skeleton` directory.

For each of 6 or 8 **frames**, SmolVLM jointly encodes the image and instruction,
and a masked GAT encodes the robot graph. A fusion layer combines these features
at each time, then a temporal Transformer uses the observation history to predict
the return at the final frame. At 5 Hz, 6 frames span 1.0 seconds and 8 span 1.4
seconds. Early clips are left-padded with an explicit attention mask. Clips never
cross episode boundaries or include observations after the prediction time.
Training and evaluation exclude terminal anchors: the model input does not
include a termination flag, and the known terminal state's future return is zero.
`predict.py` returns zero directly when given a known terminal frame.

The graph inputs per robot are position x/y, sin/cos of yaw, the two recorded
velocity components, relative assigned-goal x/y, distance to goal, and nearest
robot distance. Attention uses the recorded adjacency plus self-loops. Team
pooling masks padded robots, handles teams of different sizes in one batch, and
preserves robot count. Outcome, progress labels, rewards, checkpoint identity,
and future/terminal metadata are excluded from the input features.

## Return definition: 5 Hz images with 20 Hz control

`--gamma` is a **per-control-step** discount. With gamma=0.99, four control steps
between 5 Hz images give an interval discount of 0.99^4 = 0.960596. The final
capture can have a shorter interval; targets use the recorded timestamps.

For an image at control index c_i, full control-step supervision is:

```text
G_i = R_i + gamma^m_i * G_(i+1),       m_i = c_(i+1) - c_i
R_i = sum(k=0..m_i-1) gamma^k * r_(c_i+k+1)
G_terminal = 0
```

This is available with `--return-source control` only when the archive contains
all per-control-step rewards in `control_rewards.json`. Rewards for each saved
frame describe the interval since the preceding capture. Images may still be
sampled at 5 Hz even when rewards are recorded at 20 Hz.

**The older dataset does not contain the missing 20 Hz rewards.** Its default
`--return-source sampled` uses only the recorded endpoint reward:

```text
R_sampled_i = gamma^(m_i-1) * reward_at_next_saved_image
G_sampled_i = R_sampled_i + gamma^m_i * G_sampled_(i+1)
```

This is the timestamp-weighted sum of observed reward samples, not a reconstruction
of the full control-rate return. The code does not interpolate or multiply the
sampled rewards by four. `run.json` and checkpoints record the selected definition;
sampled and full-return labels are never silently mixed. Timeout/unfinished
episodes are excluded because their remaining return is unknown.

The local `formation_vlcm_polygon_3to6_10hz_inner00m` collection includes
20 Hz rewards despite its 10 Hz image capture. Its [audit](reports/formation_10hz_diversity.md)
found 901 complete episodes with aligned reward intervals. It is an incomplete
collection: it currently contains initial and intermediate policies only. Its
workspace boundary also differs from the older dataset. To train on it while
using images at 5 Hz:

```bash
DATASET=../Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6_10hz_inner00m \
  RETURN_SOURCE=control RUN_DIR=outputs/smolvlm-control bash train.sh
```

Use the cluster's dataset path instead when running on Palmetto. Targets from
the old sampled-reward dataset and this full-reward collection measure different
quantities; their raw MSE values should not be compared directly.

Training minimizes MSE on returns standardized with **training-only** statistics;
the output and reported MSE/MAE remain in the original return units. MSE estimates
the conditional mean of the recorded return. With all four policy stages mixed,
this is a value for the dataset's behavior mixture, rather than the value of a
single specified deployment policy. Use `--policy-stage finetuned` (or another
stage) for a policy-specific experiment; each team size can still have a different
recorded actor.

## Install

Use a CUDA-enabled PyTorch installation compatible with your GPU driver. A fresh
environment avoids changing an older InternVL environment's Transformers version:

```bash
conda create -n vlcm-smol python=3.10
conda activate vlcm-smol
pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The backbone needs Transformers >=4.49. GAT uses pure PyTorch, so torch-geometric,
torch-scatter, FlashAttention and bitsandbytes are not required.

## Local RTX 3090

From the MA-VLCM checkout, with the environment activated:

```bash
# Default dataset is the sibling Ground_Robot_MARL local collection directory.
WANDB_ENABLED=0 bash train.sh

# Or specify any dataset root; archives may be nested below it.
DATASET=/path/to/formation_vlcm_polygon_3to6 CLIP_LEN=8 bash train.sh
```

The default mode freezes SmolVLM and caches every selected frame's image-language
features once. Later epochs train the GAT, fusion, temporal Transformer and scalar
head without keeping the VLM on the GPU. The cache is keyed by source archive
identity, instruction, resolved model revision, image size and frame selection.
The 512-pixel processor limit avoids the model processor's much larger default
image tiling. GPU memory usage still depends on frame microbatch size; lower
`VL_BATCH_SIZE` if needed.

For actual VLM adaptation with LoRA (language-model q/v projections):

```bash
BACKBONE_MODE=lora BATCH_SIZE=1 VL_BATCH_SIZE=1 GRAD_ACCUM_STEPS=8 \
  RUN_DIR=outputs/smolvlm-lora bash train.sh
```

`BACKBONE_MODE=full` trains all backbone weights and needs more memory. LoRA/full
modes read and encode images during training, use gradient checkpointing, and
never use frozen feature caches. Start with a short run to measure VRAM before
increasing clip/batch size. A quick data/model smoke run is:

```bash
WANDB_ENABLED=0 EPOCHS=1 bash train.sh --max-episodes 32
```

To download the older archive dataset from HF instead of copying local files:

```bash
bash train_hf.sh
# Or: DATASET=shulnak09/formation-vlcm-polygon-3to6 bash train.sh
```

The dataset repo stores episode `.tar` files with `.summary.json` (or `.json`)
sidecars, not tabular Hugging Face rows. The loader downloads/cache these files
through huggingface_hub and uses the same archive reader as local training.

## Palmetto

For the completed full-reward **2,000-episode frozen critic** run with automatic
evaluation on the separate 60 archives, use
[`train_frozen_2000_palmetto.sbatch`](train_frozen_2000_palmetto.sbatch) and the
[submission/data/comparison instructions](README_FROZEN_2000.md). This dedicated
template requests one GPU/process and leaves the LoRA launch path unchanged.

Set up the same environment on Palmetto and copy the dataset to shared storage,
for example `/home/anshuln/Dataset/formation_vlcm_polygon_3to6`. The batch template
requests the resources selected in this project: one node/task, 4 CPUs, 12 GB
host RAM, two A100 GPUs and 12 hours. Adjust partition/resources to your allocation.
Both GPUs participate in training through torchrun/Accelerate DDP. Frozen VLM
cache creation runs once on rank zero before distributed head training.

```bash
cd /path/to/MA-VLCM
mkdir -p logs
# Make conda available, or point CONDA_BASE at its installation.
export CONDA_BASE="$(conda info --base)"
export CONDA_ENV=vlcm-smol
export DATASET=/home/anshuln/Dataset/formation_vlcm_polygon_3to6
sbatch train_palmetto.sbatch
```

`PROJECT_ROOT`, `CACHE_DIR`, `RUN_DIR`, `HF_HOME`, `EPOCHS`, `CLIP_LEN`,
`BACKBONE_MODE`, and batch settings can be overridden in the submission environment.
Choose storage with space for the dataset and feature cache. Create `logs` before
submission because Slurm opens log files before running the script.
`train_ddp.sh` launches the same trainer on two local GPUs. Legacy
`train_VLCM.sh` and `train_fsdp.sh` delegate to the new local/DDP entry points;
FSDP is not enabled by these wrappers.

## W&B, checkpoints and evaluation

`train.sh` enables W&B by default. Authenticate with `wandb login` or supply the
key through the job environment. Override `WANDB_PROJECT`, `WANDB_ENTITY` and
`WANDB_RUN_NAME`; use `WANDB_MODE=offline` without network, or `WANDB_ENABLED=0`
to disable tracking. Direct `python train.py` runs opt in with `--wandb`.

The default critic has width 128, one GAT layer, one temporal layer, dropout 0.2,
learning rate 0.0001 and batch size 64 for the frozen backbone. With LoRA/full
backbones, the script defaults to batch size 1. `BATCH_SIZE`, `PATIENCE` and
other settings can be overridden; batch size is per GPU. A scheduler halves the
learning rate after validation plateaus, and early stopping defaults to six
epochs without improvement (`PATIENCE=0` disables stopping). The W&B curves
include validation MSE, MAE, R2 and the constant-prediction baseline MSE.

The first comparison of these settings with the earlier run is recorded in
[training diagnostics](reports/training_diagnostics.md). The comparison uses
the same episode splits and excludes terminal anchors from both models' metrics.

The trainer saves `best.pt`, snapshots every 10 epochs and the final epoch,
including optimizer state. Frozen checkpoints contain only trainable heads and
normalization; LoRA checkpoints add adapters. Full-finetuning checkpoints contain
the full backbone. Do not delete the base-model HF cache if you need offline
inference from head/adapter checkpoints.

`--resume path/to/epoch_010.pt --epochs 30` restores the model/optimizer/scheduler at an epoch
boundary. Keep model, dataset, splits and target definition the same. Checkpoints
do not promise bit-identical RNG continuation. Use the original `RUN_DIR` when
resuming. New runs from `train.sh` use a timestamped output directory, and the
trainer refuses to overwrite an existing run. Checkpoints from the earlier
terminal-inclusive trainer can be evaluated, but cannot be resumed under the
changed sampling scheme. Episode/scenario splits are saved
in `run.json`. Matching initial scenarios are grouped before stratifying by
policy stage, team size and termination reason. Entire test episodes remain held
out until evaluation of the best-validation checkpoint. Pass `--skip-test` during
experiments to avoid evaluating the test split; after choosing a configuration,
evaluate it once using the command below. Outputs include raw-unit
MSE, MAE, RMSE, correlation, R2, constant-prediction baseline MSE, per-team/stage/
outcome metrics, episode-averaged MSE, and prediction CSV files.

```bash
python evaluate.py --checkpoint /path/to/run/best.pt --split test \
  --output-dir outputs/final-test
python predict.py --checkpoint /path/to/run/best.pt \
  --episode /path/to/episode_000001.tar --anchor 10
```

The anchor is the selected-image index. The terminal image's future-return target
is zero. New input observations must use the same physical feature definitions
and task semantics as training.

## Animate checkpoint predictions

To prepare and collect a separate 60-episode validation set with new seeds
(15 episodes per formation), use the existing full-reward collection as the
source of frozen actor and environment settings:

```bash
python3 collect_fresh_validation.py --collect
```

The default output is the sibling
`Ground_Robot_MARL/datasets/formation_validation_seed7300001` directory. The
default uses the intermediate actor with formation seeds 7300001, 7400001,
7500001, and 7600001. It preserves the source scene, 10 Hz images and all
20 Hz rewards, and retains every outcome without success/failure quotas. Run
without `--collect` to prepare only. Existing completed episodes are retained
when resuming the same request. The request is checked against the saved plan,
and seeds are checked against the source collection. New episode IDs avoid
collisions with the saved train/validation/test path keys. Keep this dataset
outside training when using it for model selection.

Evaluate genuinely new archives with the training Python environment:

```bash
python evaluate.py --checkpoint outputs/zero_margin_900/best.pt \
  --dataset ../Ground_Robot_MARL/datasets/formation_validation_seed7300001 \
  --split external --output-dir outputs/validation_seed7300001_evaluation
```

External evaluation caches frozen VLM features once, rejects archives with saved
split path keys, and excludes timeout/unfinished episodes whose full future
return is unknown. MSE is reported on nonterminal anchors by default. New-seed
collection alone does not establish generalization to new tasks or appearances.

`animate_checkpoint.py` displays an episode's overhead images beside causal
predicted value, recorded discounted future return, recorded frame reward, and
recorded team progress. The model predicts future return; it has no separate
immediate-reward or progress output. Curves reveal observations only as playback
advances. Output includes an animation, per-frame CSV, and `manifest.json` with
split membership, prediction source, and nonterminal MSE.

Activate the training environment and install `requirements.txt`. MP4 export
requires the system `ffmpeg` executable; `--format gif` uses Pillow instead.
From this directory, run fresh inference on two saved validation episodes
(one per outcome when available):

```bash
python animate_checkpoint.py --checkpoint outputs/zero_margin_900/best.pt \
  --samples 2 --output-dir outputs/checkpoint_animations/fresh
```

To evaluate your own archive, supply its path. Repeat `--episode` for more
archives. CPU inference works with `--device cpu`, but uncached SmolVLM inference
can be slow. Existing frozen feature caches are reused automatically.

```bash
python animate_checkpoint.py --checkpoint outputs/zero_margin_900/best.pt \
  --episode /path/to/episode_000001.tar \
  --output-dir outputs/checkpoint_animations/custom
```

New archives must use the formation collector's schema: frame `overhead.png`,
`state.json`, `adj.npy`, and `reward.json` fields, plus an adjacent
`episode_000001.summary.json` (or `.json`) sidecar. Include `progress.json` to
display progress. For this full-control-reward checkpoint, completed episodes
also need aligned `control_rewards.json` for every frame. Preserve the training
task, robot-state definitions, physical units, image view, and control/image
timestamps. The script reads the checkpoint's clip length, image sampling,
normalization, and return definition. It labels supplied training episodes as
`train`; newly collected episodes appear as external rather than claiming saved
validation membership. Use `--dataset /relocated/dataset` to resolve saved
validation paths after moving the dataset.

Timeout/unfinished episodes show predictions with unavailable future-return
labels and no MSE. For fresh inference the terminal image shows the raw network
output against the known zero return; it is never replaced by a forced zero.
Terminal frames are excluded from reported MSE, matching validation evaluation.

The existing held-out previews are in
`outputs/checkpoint_animations/validation_preview`. They replay saved CSV
predictions, explicitly labeled in the video, rather than performing new
inference. This mode also works without PyTorch:

```bash
python3 animate_checkpoint.py \
  --predictions outputs/zero_margin_900/validation_predictions.csv \
  --samples 2 --output-dir outputs/checkpoint_animations/validation_preview
```

Saved replay uses `run.json` beside the CSV (override with `--run-dir`) and checks
targets against the selected archive. The existing validation CSV excludes
terminal anchors, so replay adds the recorded terminal frame, reward, progress
and known zero future return without inventing a prediction. The animation holds
the terminal frame for one second to make the outcome visible. Playback otherwise
defaults to real time; use `--speed 0.5` for half speed or `--format gif` for a GIF.

To create named success/failure examples for every formation in the saved
validation split, use `--by-formation`:

```bash
python animate_checkpoint.py --checkpoint outputs/zero_margin_900/best.pt \
  --by-formation --output-dir outputs/checkpoint_animations/by_formation_fresh

# Replay existing best-checkpoint predictions without PyTorch:
python3 animate_checkpoint.py \
  --predictions outputs/zero_margin_900/validation_predictions.csv \
  --by-formation --output-dir outputs/checkpoint_animations/by_formation
```

Files are named `triangle_success.mp4`, `triangle_failure.mp4`,
`square_success.mp4`, and so on, with matching CSVs. Formation names come from
the archive summary. `README.md` indexes the videos and errors; `coverage.json`
records the available validation counts for every formation/outcome. Selection
uses the first sorted archive in each category, without ranking by model error.
Use `--per-case 3` for up to three episodes per category, with numbered filenames.
The current split has no successful hexagon episode; that case is marked
unavailable, and no training/test episode is substituted.

For the separately collected new-seed episodes and their existing best-checkpoint
predictions, select one success and failure from every formation with:

```bash
python3 animate_checkpoint.py --predictions outputs/validation_seed7300001_evaluation/external_predictions.csv \
  --run-dir outputs/zero_margin_900 \
  --external-dataset ../Ground_Robot_MARL/datasets/formation_validation_seed7300001 \
  --by-formation --exclude-terminal \
  --output-dir outputs/checkpoint_animations/fresh_validation_formations
```

This replays stored predictions from a previous fresh checkpoint evaluation;
it does not train the actor or critic. Omit `--exclude-terminal` to see the
recorded terminal reward.

## Data diversity and rewards

Run the audit without downloading a model:

```bash
python audit_dataset.py --dataset /path/to/formation_vlcm_polygon_3to6
```

The checked-in [audit report](reports/formation_diversity.md) describes all 2,000
local episodes, not just one sampled episode. The data has policy-quality and
initial-state variation and both outcomes, so it can support an initial
in-distribution experiment. It has **one instruction**, one formation task,
one spacing/workspace, and strongly uneven success coverage. It cannot establish
generalization across language goals, appearances, tasks or unseen policies.
Collect more independent resets/policies, hard recoveries, collision cases and
successful large-team episodes as needed; collect genuinely different goals and
visual environments before claiming those kinds of generalization. Compare a
state/graph-only baseline and image/language ablations before attributing gains
to the VLM; the robot states already reveal much of the formation geometry.

Keep the raw reward trajectory. The reward need not be differentiable for value
regression: gradients pass through the predicted value and its MSE loss, while
observed returns are fixed labels. Abrupt success/failure penalties carry useful
information. Smoothing can erase or move those events and changes the return
being learned. Training-only target standardization is used instead. A smoothed
curve may be useful for plots, but it is not used as supervision. See the
[standard return/value definitions](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).

## Verification

Python actor training with a VLCM critic is available through
[`train_mappo.py`](train_mappo.py) and [`train_mappo.sh`](train_mappo.sh), including
W&B, MLP/state baselines, 20 Hz states with 5 Hz images, and actor-only exports.
See [Python MAPPO setup and verified limits](README_PYTHON_MAPPO.md).

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic archives and a tiny randomly initialized Idefics3 model; they
do not download the 500M checkpoint or require CUDA. GPU memory/throughput and
cluster execution must be measured on the target machine.
