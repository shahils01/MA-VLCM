# MA-VLCM: RWARE Training with LLaVA-NeXT-Video

This branch is configured for training a Multimodal Value Model on the Robot Warehouse (RWARE) environment using **LLaVA-NeXT-Video** as the vision-language backbone.

## Quick Start
To run the training, use the provided shell script:
```bash
./run_train_vlcm.sh
```
This script handles:
1. Data directory checks (expects data in `VLCM_Data_Collection/RWARE/data_scratch` or defined via environment variables).
2. Automatic detection of RWARE configuration and number of robots from directory names.
3. Setting up cache directories on scratch space.
4. Launching the training inside an Apptainer/Singularity container.

## Experiment Configuration

### 1. Model Architecture
- **VLM Backbone**: `llava-hf/LLaVA-NeXT-Video-7B-32K-hf` (via `llava_video` backend)
- **Robot Encoder**: 2-layer GNN (Graph Neural Network) processing robot states.
- **Fusion**: Multimodal fusion of video embeddings, text instruction embeddings, and robot graph embeddings.
- **Value Head**: Estimates state value $V(s)$ trained with TD(0) loss.

### 2. Observation Structure
The model processes a sequence of observations (video clip + state history).
- **Video**: 8 frames per clip (`--clip_len 8`).
- **RWARE State**: A 6-dimensional vector for each robot, derived from `state.json`:
  - `[0, 1]`: Position (x, y)
  - `[2, 3]`: Direction vector (dx, dy) - e.g., (0, 1) for North
  - `[4]`: Carrying status (1.0 if carrying a shelf, 0.0 otherwise)
  - `[5]`: Last Action (mapped to integer: NOOP=0, FORWARD=1, LEFT=2, RIGHT=3, TOGGLE_LOAD=4)
- **OFFROAD State**: An 8-dimensional vector per agent, representing continuous navigation commands:
  - `[0, 1]`: Position (x, y)
  - `[2, 3]`: Heading (cos(yaw), sin(yaw))
  - `[4, 5]`: Velocity commands (v_cmd, w_cmd)
  - `[6]`: Distance to Goal
  - `[7]`: Traversability of current position
- **Graph**: Fully connected graph between agents (unless `adj.npy` is provided), allowing the GNN to model agent interactions.

### 3. Reward Structure
The reward function is a combination of the environment reward and a custom safety penalty:
- **Base Reward**: Taken from `reward.json` (sparse reward for successful delivery).
- **Collision Penalty**: A penalty of **-1.0** is added if any two agents are within **3.0 meters** of each other (`dist < 3.0`).
- **Distance Reward**: A dense shaping reward equal to the **negative average minimum distance** from each agent to any requested box.
  - If an agent is already carrying a requested box, its distance cost is 0.
  - Otherwise, it is the Euclidean distance to the nearest requested box.
- **Formula**: $R_{total} = R_{env} + (-1.0 \text{ if collision}) - \frac{1}{N} \sum_{i=1}^{N} \min_{j} \text{dist}(\text{agent}_i, \text{box}_j)$

### 4. Language Prompt
A dynamic text prompt is generated for every step to ground the VLM:

**RWARE Template:**
> "Analyze the robotic warehouse state. Agents must pick up requested boxes and avoid collisions (distance < 3m). Step: {step}. Requested boxes: {requests}. Agent {id}: at {pos}, facing {dir}, action {action}, carrying {yes/no}. ..."

**OFFROAD Template:**
> "You are a vision-language critic model that evaluates multi-agent trajectories by estimating the expected cumulative return. This is an offroad navigation environment with {n_ag} agents traversing rough terrain. Each agent must reach its color-matched goal while minimizing traversability cost and avoiding inter-agent collisions. Given the video frames and agent states below, assess the quality of the current policy. Timestep: {step}. Agent {id} ({color}): position ({x}, {y}), heading {yaw} rad, speed {v} m/s, dist_to_goal {d}m, traversability {t}, reached: {yes/no}, collision: {yes/no}."

## Training Script Arguments
The `run_train_vlcm.sh` script passes the following key arguments to `train.py`:
- `--train_shards`: Primary dataset path (typically RWARE).
- `--offroad_shards`: Optional secondary dataset path to enable Multi-Dataset interleaving training.
- `--dataset_type rware`: Enables the specific RWARE state parsing logic (will round-robin with "offroad" if multi-dataset enabled).
- `--rware_config`: The specific map configuration (e.g., `tiny-2ag-hard`).
- `--vl_backend llava_video`: Selects the LLaVA-NeXT-Video backend.
- `--num_robots`: Automatically detected, or artificially padded to max across datasets (e.g. 5 if mixing RWARE and OFFROAD).
- `--robot_obs_dim 8`: Updated to 8 to accommodate the larger feature set of OFFROAD when training jointly.
- `--batch_size 4`, `--epochs 2`, `--num_workers 1`.

## Installation
```bash
pip install -r requirements.txt
```
*Note: The training script assumes a specific environment setup with Apptainer/Singularity for HPC usage.*
