#!/bin/bash

# Exit on error
set -e

# Data Directory
DATA_DIR="/home/anshul/Research/Postdoc/RL/MA-VLCM/test_data" 

# If Data Already extracted
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Directory $DATA_DIR not found!"
    echo "Current directory content:"
    ls -l
    exit 1
fi
echo "Using existing data in $DATA_DIR"


# RUN Training
echo "Run Training"

# Point to Root Directory
SHARD_PATTERN="$DATA_DIR"
echo "Using Shard Pattern: $SHARD_PATTERN"

# Define Container Path:
CONTAINER_PATH="$PWD/ma_vlvm_container.sif"

NUM_ROBOTS=5
echo "Using Max Num Robots: $NUM_ROBOTS"

# Run with Singularity
# We bind the entire BASE_SCRATCH to ensure the container can access the tmp locations if needed
apptainer exec --nv -B "$PWD:$PWD" -B "$BASE_SCRATCH:$BASE_SCRATCH" \
  --env HF_TOKEN="$HF_TOKEN" \
  "$CONTAINER_PATH" accelerate launch --num_processes 4 train.py \
  --train_shards "$SHARD_PATTERN" \
  --dataset_type rware \
  --rware_config "$CONFIG_NAME" \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --clip_len 8 \
  --num_robots "$NUM_ROBOTS" \
  --robot_obs_dim 6 \
  --epochs 20 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf \
  --save_dir checkpoints_pyenv \
  --num_workers 0 \
  --mixed_precision bf16 \
  --peft qlora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --vl_max_text_len 16384 \