# MA-VLCM

MA-VLCM is a multi-agent vision-language critic that predicts a scalar policy-quality signal from a short video clip, a structured prompt, and per-agent state/action features. Generic datasets can train return estimates; the TurtleBot3 lab path now trains bounded task progress in `[0, 1]`.

## Workspace Layout

- `src/ma_vlcm/`: core package code
- `scripts/`: shell launchers for training and inference
- `tools/`: analysis and debugging utilities
- `outputs/`: generated plots, CSVs, videos, and other run artifacts
- `train_sample/`: example trajectory data kept with the repo

Current core files:

- [src/ma_vlcm/train.py](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/src/ma_vlcm/train.py)
- [src/ma_vlcm/inference.py](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/src/ma_vlcm/inference.py)
- [src/ma_vlcm/model.py](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/src/ma_vlcm/model.py)
- [scripts/run_train_vlcm.sh](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/scripts/run_train_vlcm.sh)
- [scripts/run_inference_vlcm.sh](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/scripts/run_inference_vlcm.sh)

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

If you do not want an editable install, use:

```bash
export PYTHONPATH="$PWD/src"
```

The shell launchers in `scripts/` already set `PYTHONPATH` for you.

## Training

Launch training with:

```bash
bash scripts/run_train_vlcm.sh
```

The current default launcher uses:

- `--vl_backend llava_onevision`
- `--vl_model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- `clip_len=16`
- `num_robots=8`
- `robot_obs_dim=8`
- `gamma=0.95`
- `max_return_horizon=64`
- LoRA with `contrastive_mse`

There are alternate launchers in `scripts/` for contrastive and LoRA-specific runs.

### TurtleBot3 Lab LoRA Fine Tuning

The current TurtleBot3 lab LoRA launcher expects progression-labeled WebDataset
shards on Hugging Face by default:

```bash
hf://datasets/adi2440/tb3-lab-vlcm-progress-v1/**/*.tar
```

If you collected raw TB3 lab shards or already have the older reward-labeled
dataset, relabel them with `tb3_progress_v1` before training:

```bash
python scripts/relabel_tb3_progress_dataset.py \
  hf://datasets/adi2440/tb3-lab-vlcm/*.tar \
  --output-dir data/tb3_lab_progress_v1 \
  --repo-id adi2440/tb3-lab-vlcm-progress-v1
```

The relabel command uploads to Hugging Face only when `--repo-id` is present.
Without `--repo-id`, it only writes relabeled local `.tar` shards to
`--output-dir`; run it again with `--repo-id` or upload the directory separately
when you are ready to publish.

Fine tune the saved MA-VLCM checkpoint on the progression-labeled TB3 lab data with:

```bash
bash scripts/lora_run_train_tb3_lab.sh
```

The same launcher also trains on the compatible Isaac Sim TurtleBot3 dataset.
Isaac episodes contain 3–6 agents, so this profile uses six model slots and
pads smaller episodes in the loader:

```bash
DATASET_PROFILE=tb3_isaac bash scripts/lora_run_train_tb3_lab.sh
```

The `tb3_lab` profile is the default and uses three model slots. The Isaac
profile defaults to `adi2440/tb3-isaac-vlcm`; both profiles recursively load
root-level or nested `.tar` shards. Override `HF_DATASET_REPO`, `DATA_DIR`, or
the first positional argument as before. A local Isaac collection can be used
directly:

```bash
DATASET_PROFILE=tb3_isaac bash scripts/lora_run_train_tb3_lab.sh \
  /home/adi2440/Desktop/MARL_Shahil_Aditya/VLCM_Data_Collection/TURTLEBOT/data/tb3_isaac_vlcm
```

The TB3 launcher defaults to `--target_mode progress`,
`--value_output_activation sigmoid`, and `--loss_type mse`. It preserves the
old dense reward files in the shards, but trains the value head on
`progress.json` instead of bootstrapped return.

The launcher writes Hugging Face downloads, Transformers model files, Torch
cache files, temporary files, wandb files, and TB3 checkpoints under scratch.
Scratch defaults to `$SCRATCH/ma_vlcm`, then `/scratch/$USER/ma_vlcm`, then
`$SLURM_TMPDIR/ma_vlcm`; set `MA_VLCM_SCRATCH_ROOT` to override it.

To use a different dataset repo or local shard directory:

```bash
HF_DATASET_REPO=your-username/your-tb3-dataset bash scripts/lora_run_train_tb3_lab.sh

bash scripts/lora_run_train_tb3_lab.sh hf://datasets/your-username/your-tb3-dataset/*.tar
```

#### Configuring Hugging Face Dataset & Authentication for Training

- **Dataset Repository**: Pass via `HF_DATASET_REPO` environment variable or as the first positional argument:
  ```bash
  HF_DATASET_REPO="your-username/your-tb3-dataset" bash scripts/lora_run_train_tb3_lab.sh
  ```
- **Authentication Token**: If using a private Hugging Face dataset, export your API token before running:
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
  HF_DATASET_REPO="your-username/your-tb3-dataset" bash scripts/lora_run_train_tb3_lab.sh
  ```

To resume from a different pretrained or intermediate checkpoint, pass it as
the second argument or set `RESUME_CHECKPOINT`:

```bash
RESUME_CHECKPOINT=/scratch/$USER/ma_vlcm/checkpoints/0.5B_LoRA_epoch_3.pt bash scripts/lora_run_train_tb3_lab.sh
```

To train from scratch instead of resuming from the default MA-VLCM checkpoint:

```bash
TRAIN_FROM_SCRATCH=1 bash scripts/lora_run_train_tb3_lab.sh
```

Equivalent forms are also accepted:

```bash
RESUME_CHECKPOINT=none bash scripts/lora_run_train_tb3_lab.sh
bash scripts/lora_run_train_tb3_lab.sh /path/to/local/tb3_lab_progress_v1 none
```

On Slurm, pass the same environment variable:

```bash
TRAIN_FROM_SCRATCH=1 sbatch scripts/lora_submit_train_tb3_lab.sh
```

Outputs are written to `$MA_VLCM_SCRATCH_ROOT/checkpoints/$DATASET_PROFILE`
unless `SAVE_DIR` is set.

On Slurm:

```bash
sbatch scripts/lora_submit_train_tb3_lab.sh
```

## Inference

Run evaluation with:

```bash
bash scripts/run_inference_vlcm.sh
```

Edit these variables in [scripts/run_inference_vlcm.sh](/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM/scripts/run_inference_vlcm.sh) before running:

- `CHECKPOINT`
- `TEST_DATA_DIR`
- `OUTPUT_FILE`
- `PLOT_DIR`
- `CONTAINER_PATH`

By default, inference outputs now go under:

- `outputs/results/`
- `outputs/plots/inference/`

The native fallback path in the inference launcher still uses `--baseline`; remove that flag if you want native fine-tuned LoRA inference instead of the pretrained-backbone baseline.

### Live TurtleBot3 MA-VLCM Monitor

To run the physical TurtleBot3 MARL policy and monitor it with MA-VLCM live:

```bash
cd /home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM
bash scripts/run_tb3_vlcm_live_monitor.sh \
  checkpoints/NewFinal_0.5B.pt
```

The script launches:

- `ros2 launch cv_localization cv_rl_direct.launch.py`
- `python3 -m ma_vlcm.tb3_live_inference`
- `python3 -m ma_vlcm.tb3_live_monitor`

The live inference node listens to the TurtleBot workspace topics already used
for collection:

- `/fleet_vlcm/overhead/compressed`
- `/tb_N/cv_pose`
- `/tb_N/cmd_vel`
- `/tb_N/cv_measured_velocity`
- `/tb_N/mppi_goal`

It publishes JSON predictions to `/fleet_vlcm/vlcm_prediction` and writes
`outputs/results/tb3_live_predictions.csv`. The plot monitor shows MA-VLCM
predictions against live bounded task progress.

## MA-VLCM Inputs And Output

For a clip starting at timestep `t`, the dataloader builds:

`x_t = (V_t, p_t, O_t, A_t)`

where:

- `V_t = (I_t, I_{t+1}, ..., I_{t+T-1})` is the video clip
- `p_t` is the text prompt built from the first frame in the clip
- `O_t in R^(T x N x D)` is the per-agent observation tensor
- `A_t in R^(T x N x N)` is the adjacency tensor

### Robot Observations

For RWARE, each agent row is:

`o_i = [x_i, y_i, dx_i, dy_i, carry_i, a_i, 0, 0]`

with `a_i in {0,1,2,3,4}` for `NOOP`, `FORWARD`, `LEFT`, `RIGHT`, `TOGGLE_LOAD`.

For OFFROAD, each agent row is:

`o_i = [x_i, y_i, cos(yaw_i), sin(yaw_i), v_i, w_i, dist_i, trav_i]`

### Prompt

The prompt is generated from structured state metadata:

- RWARE: timestep, requested shelves, and per-agent position/direction/action/carrying
- OFFROAD: timestep, and per-agent position/yaw/speed/dist-to-goal/traversability/reached/collision
- TB3 lab: timestep, goals, per-agent pose/velocity/distance-to-goal, neighbor spacing, reached/collision flags, and outcome metadata

### What The Forward Pass Uses

Although the loader passes the full `O_t` and `A_t`, the model uses only the last robot state and last graph in the clip:

- `O_t[-1]`
- `A_t[-1]`

The robot team feature is:

`g_t = MeanPool(GNN(O_t[-1][:, :8], A_t[-1]))`

That feature is projected into the language-model embedding space and injected at the `<obs>` token. The VLM pooled feature `z_t` is concatenated with `g_t`, and the value head returns:

`yhat_t = w^T [z_t ; g_t] + b`

So the model output is one scalar per clip. With
`--value_output_activation sigmoid`, the reported scalar is bounded to `[0, 1]`:

`yhat_t in R`

## Training Target

For return training (`--target_mode return`), the loader first computes the
n-step clipped return:

`G_t^(H) = sum_(k=0)^(H-1) gamma^k r_(t+k) prod_(j=0)^(k-1) (1 - d_(t+j))`

Then training bootstraps from the overlapping next clip:

`target_t = G_t^(H) + gamma^T (1 - d_(t+T-1)) yhat^-_(t+1)`

where `yhat^-_(t+1)` is the EMA target-model prediction on the shifted clip.

For TurtleBot3 lab progress training (`--target_mode progress`), the loader
reads `progress.json` directly and does not bootstrap from the next clip. The
`tb3_progress_v1` target is:

`agent_progress_i = clamp((initial_dist_i - max(dist_i - goal_radius, 0)) / max(initial_dist_i - goal_radius, eps), 0, 1)`

`team_progress = mean(agent_progress_i)`

`target = 1.0` for clean success, `0.0` for terminal failure or unsafe
collision, otherwise `team_progress`.

## Outputs

The repo is now organized so generated artifacts are not mixed into the root:

- `outputs/plots/reward_analysis/`
- `outputs/plots/comparison/`
- `outputs/plots/inference/`
- `outputs/results/`
- `outputs/videos/`

## Useful Commands

Training as a module:

```bash
PYTHONPATH=src accelerate launch -m ma_vlcm.train ...
```

Inference as a module:

```bash
PYTHONPATH=src python -m ma_vlcm.inference ...
```

Single-shard visualization:

```bash
PYTHONPATH=src python tools/visualize_shard.py ...
```
