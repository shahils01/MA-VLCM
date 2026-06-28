# Agent Instructions for MA-VLCM

## Workspace Context

- This repository is `MA-VLCM`, a Python package for training and running a
  multi-agent vision-language critic. The model predicts one scalar return
  estimate from a short video clip, structured prompt text, and per-agent
  state/action graph features.
- Treat `/home/adi2440/turtlebot_ws` as the companion ROS 2 workspace. It owns
  the TurtleBot3 hardware stack, launch files, GUI/localization/control nodes,
  and data collection topics that feed this repo. Read its `README.md` before
  changing live TurtleBot3 integration, topic contracts, collection formats, or
  launch assumptions.
- This repo and `/home/adi2440/turtlebot_ws` go hand in hand:
  - TurtleBot workspace runs the fleet policy, localization, GUI, WebDataset
    collector, and ROS topics.
  - MA-VLCM trains/fine-tunes the critic, runs offline inference, and provides
    live MA-VLCM inference/plotting nodes for the TurtleBot runtime.
- Do not edit `/home/adi2440/turtlebot_ws` unless the user explicitly asks for
  cross-workspace changes. Inspect it as needed to keep contracts consistent.

## Important Paths

- `src/ma_vlcm/`: package code.
  - `train.py`: WebDataset loading, training loop, target construction.
  - `inference.py`: offline checkpoint evaluation.
  - `model.py`: multimodal value model.
  - `tb3_live_inference.py`: ROS 2 live inference node for TurtleBot3 lab data.
  - `tb3_live_monitor.py`: live plot/monitor node.
- `scripts/`: launcher scripts for training, inference, Slurm, Hugging Face
  upload, and TurtleBot3 live monitoring.
- `tools/`: analysis, plotting, shard inspection, and visualization utilities.
- `tests/`: focused smoke tests; currently includes TB3 lab loader coverage.
- `README.md`: primary local source of truth for normal use.
- `README_HPC.md`: container/HPC and Slurm workflow notes.
- `/home/adi2440/turtlebot_ws/README.md`: companion ROS/hardware workflow.
- `/home/adi2440/turtlebot_ws/DETAILS.md`: deeper TurtleBot architecture,
  calibration, config, safety, and troubleshooting notes.

## Environment Setup

- Python package target is Python `>=3.10`.
- Install locally with:

```bash
pip install -r requirements.txt
pip install -e .
```

- If not installing editable, run modules with:

```bash
export PYTHONPATH="$PWD/src"
```

- Shell launchers generally set `PYTHONPATH` themselves.
- Training commonly expects CUDA, PyTorch CUDA 12.1 packages, Accelerate,
  Transformers, PEFT/LoRA, bitsandbytes, decord, WebDataset, and W&B.
- HPC/container workflows use `ma_vlcm.sif` built from `ma_vlcm.def`; see
  `README_HPC.md`.

## TurtleBot3 / ROS 2 Integration

- Live TurtleBot3 workflows assume ROS 2 Humble and the workspace at
  `/home/adi2440/turtlebot_ws`.
- The operator PC and robots use `ROS_DOMAIN_ID=30`.
- The TurtleBot workspace controls three TurtleBot3 Burger robots, typically
  `tb_1`, `tb_2`, and `tb_3`.
- The companion workspace launch files of interest are:
  - `cv_localization cv_rl_direct.launch.py`: runs the AERO-MARL policy.
  - `cv_localization cv_rl_vlcm_collect.launch.py`: runs policy plus MA-VLCM
    WebDataset collection.
  - `cv_localization cv_mppi_direct.launch.py`: model-based CBS/ORCA mode.
- The MA-VLCM convenience live monitor launcher is:

```bash
export ROS_DOMAIN_ID=30
bash scripts/run_tb3_vlcm_live_monitor.sh checkpoints/draft_1_tb3_lab.pt
```

- That launcher sources ROS, sources the TurtleBot workspace if available,
  starts the MARL policy launch, starts `ma_vlcm.tb3_live_inference`, and starts
  `ma_vlcm.tb3_live_monitor`.
- Do not run robot bringup, ROS launch files, or live monitor scripts unless the
  user explicitly asks. These commands can command physical robots.

## ROS Topic Contract

The live MA-VLCM path consumes the same TurtleBot topics used for collection:

- `/fleet_vlcm/overhead/compressed`
- `/tb_N/cv_pose`
- `/tb_N/cmd_vel`
- `/tb_N/cv_measured_velocity`
- `/tb_N/mppi_goal`

It publishes predictions to:

- `/fleet_vlcm/vlcm_prediction`

Default live prediction CSV:

- `outputs/results/tb3_live_predictions.csv`

Keep topic names, robot names, and the three-robot assumption synchronized with
`/home/adi2440/turtlebot_ws` if changing live inference or collection code.

## Data and Model Contracts

- Standard model input is `x_t = (V_t, p_t, O_t, A_t)`.
- `V_t` is the clip, `p_t` is structured prompt text, `O_t` is robot
  observation data, and `A_t` is adjacency data.
- The model uses the last robot state and graph in the clip: `O_t[-1]` and
  `A_t[-1]`.
- RWARE/OFFROAD training commonly uses `num_robots=8` and `robot_obs_dim=8`.
- TurtleBot3 lab fine-tuning and live inference use `num_robots=3` and
  `robot_obs_dim=8`.
- TB3 WebDataset samples include files such as:
  - `*.overhead.png`
  - `*.state.json`
  - `*.reward.json`
  - `*.episode_reward.json`
  - `*.adj.npy`
  - `*.dist.npy`
- TB3 lab collection is usually written from the TurtleBot workspace into this
  repo under `data/tb3_lab`, or uploaded to Hugging Face as `.tar` shards.
  Check `scripts/lora_run_train_tb3_lab.sh` for the current default
  `HF_DATASET_REPO` before running training.

## Common Commands

Install and import-check:

```bash
pip install -r requirements.txt
pip install -e .
python -c "import ma_vlcm; print(ma_vlcm.__file__)"
```

Run the current focused smoke test:

```bash
PYTHONPATH=src python tests/test_tb3_lab_loader.py
```

Train with the default mixed RWARE/OFFROAD launcher:

```bash
bash scripts/run_train_vlcm.sh
```

Fine-tune LoRA on TB3 lab data:

```bash
bash scripts/lora_run_train_tb3_lab.sh
```

Run training directly as a module:

```bash
PYTHONPATH=src accelerate launch -m ma_vlcm.train ...
```

Run inference directly as a module:

```bash
PYTHONPATH=src python -m ma_vlcm.inference ...
```

Visualize a shard:

```bash
PYTHONPATH=src python tools/visualize_shard.py ...
```

## Validation Guidance

- For loader or TB3 dataset changes, run:

```bash
PYTHONPATH=src python tests/test_tb3_lab_loader.py
```

- For shell launcher edits, run syntax checks before handing off:

```bash
bash -n scripts/*.sh
```

- For import-sensitive changes, use module execution from repo root with
  `PYTHONPATH=src` unless the package is installed editable.
- Avoid starting full training, Slurm jobs, Apptainer runs, ROS launches, or
  robot bringup as validation unless the user asks for them. They are expensive
  or hardware-facing.

## Generated Files and Large Artifacts

- Do not commit or casually rewrite generated artifacts:
  - `checkpoints/`
  - `outputs/`
  - `logs/`
  - `wandb/`
  - `data/`
  - `*.pt`, `*.pth`, `*.safetensors`, `*.ckpt`
  - `*.sif`
- The repo root contains very large artifacts such as `ma_vlcm.sif`; leave them
  alone unless the user asks for container work.
- Keep new run outputs under `outputs/` or scratch paths rather than the repo
  root.

## Editing Practices

- Prefer existing package structure and module entrypoints over adding root-level
  scripts.
- Keep absolute paths in launchers configurable through environment variables
  when possible. Existing workflows use paths under:
  - `/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM`
  - `/home/adi2440/turtlebot_ws`
  - `/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL`
- Do not add new secrets or tokens to scripts. Use environment variables for
  Hugging Face, W&B, and other credentials.
- If changing dataset schema, robot observation layout, reward semantics, or ROS
  topics, update both code and README documentation. Also check the TurtleBot
  workspace README/DETAILS because collection and live inference may depend on
  the same contract.
- Preserve the safety behavior of hardware-facing scripts: missing checkpoint
  checks, cleanup traps, ROS domain setup, and zero/stop behavior belong in the
  integration path.
- Be careful around files with existing uncommitted changes. Read them first,
  make only task-scoped edits, and do not revert user work.
