# MA-VLCM

MA-VLCM is a multi-agent vision-language critic that predicts a scalar return estimate for a policy from a short video clip, a structured prompt, and per-agent state/action features.

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

### What The Forward Pass Uses

Although the loader passes the full `O_t` and `A_t`, the model uses only the last robot state and last graph in the clip:

- `O_t[-1]`
- `A_t[-1]`

The robot team feature is:

`g_t = MeanPool(GNN(O_t[-1][:, :8], A_t[-1]))`

That feature is projected into the language-model embedding space and injected at the `<obs>` token. The VLM pooled feature `z_t` is concatenated with `g_t`, and the value head returns:

`yhat_t = w^T [z_t ; g_t] + b`

So the model output is one scalar per clip:

`yhat_t in R`

## Training Target

The loader first computes the n-step clipped return:

`G_t^(H) = sum_(k=0)^(H-1) gamma^k r_(t+k) prod_(j=0)^(k-1) (1 - d_(t+j))`

Then training bootstraps from the overlapping next clip:

`target_t = G_t^(H) + gamma^T (1 - d_(t+T-1)) yhat^-_(t+1)`

where `yhat^-_(t+1)` is the EMA target-model prediction on the shifted clip.

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
