#!/bin/bash

# Exit on error
set -e

echo "Starting Job on $(hostname)"
echo "Date: $(date)"

# 2. Data Setup
echo "Checking data..."

DATA_DIR="/scratch/aparame/Research/VLCM_Data_Collection/data_scratch"

# User stated data is already extracted in data_scratch
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Directory $DATA_DIR not found!"
    echo "Current directory content:"
    ls -l
    exit 1
fi
echo "Using existing data in $DATA_DIR"


# 4. Run Training
echo "Running Training..."


# Point directly to the root data directory.
# train.py will recursively find all .tar files in all subdirectories (configs).
SHARD_PATTERN="$DATA_DIR"
CONFIG_NAME="mixed-rware"

echo "Using Shard Pattern: $SHARD_PATTERN"

# Define container path
CONTAINER_PATH="$PWD/ma_vlcm.sif"



# Determine optimal cache/tmp location (Prefer /scratch as it has 5TB capacity)
if [ -n "$SCRATCH" ]; then
    BASE_SCRATCH="$SCRATCH"
elif [ -d "/scratch/$USER" ]; then
    BASE_SCRATCH="/scratch/$USER"
else
    # Fallback to current directory if no scratch is found (likely will fail if quota is low)
    BASE_SCRATCH="$PWD"
    echo "WARNING: using $PWD for scratch space. This may fail if quota is low."
fi


# Export variables for the host environment (Apptainer uses these)

export HF_TOKEN=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

# Performance tuning for multi-GPU H100
export NCCL_ALGO=Ring
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false



# Set NUM_ROBOTS to the maximum expected across all configs (e.g. 4).
# train.py will pad observations for configs with fewer robots (e.g. 2ag) to this size.
NUM_ROBOTS=4
echo "Using Max Num Robots: $NUM_ROBOTS"

SAVE_DIR="/scratch/aparame/Research/VLCM_checkpoints"
mkdir -p "$SAVE_DIR"
echo "Saving checkpoints to: $SAVE_DIR"


# Run with Singularity
# We bind the entire BASE_SCRATCH to ensure the container can access the tmp locations if needed
apptainer exec --nv -B "$PWD:$PWD" -B "$BASE_SCRATCH:$BASE_SCRATCH" \
  --env HF_TOKEN="$HF_TOKEN" \
  "$CONTAINER_PATH" accelerate launch --num_processes 4 train.py \
  --train_shards "$SHARD_PATTERN" \
  --dataset_type rware \
  --rware_config "$CONFIG_NAME" \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --clip_len 10 \
  --num_robots "$NUM_ROBOTS" \
  --robot_obs_dim 6 \
  --epochs 50 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf \
  --save_dir "$SAVE_DIR" \
  --num_workers 4 \
  --mixed_precision bf16 \
  --freeze_vl \
  --vision_lr 1e-5 \
  --loss_type contrastive_mse \
  --mse_loss_weight 1.0 \
  --vl_max_text_len 4096

# Tar up results for transfer back (handled by transfer_output_files=checkpoints_rware)
