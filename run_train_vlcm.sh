#!/bin/bash

# Exit on error
set -e

echo "Starting Job on $(hostname)"
echo "Date: $(date)"

# 2. Data Setup
echo "Checking data..."

DATA_SOURCE="${1:-local}"

if [ "$DATA_SOURCE" = "huggingface" ]; then
    echo "Using Hugging Face Hub datasets..."
    # IMPORTANT: Replace YOUR_HF_USERNAME and DATASET_NAME with your actual repository path
    # e.g., hf://datasets/aparame/VLCM-RWARE/*.tar
    SHARD_PATTERN="hf://datasets/adi2440/RWARE_Dataset/*.tar"
    OFFROAD_DATA_DIR="hf://datasets/adi2440/OFFROAD_Dataset/*.tar"
else
    DATA_DIR="/scratch/aparame/Research/VLCM_Data_Collection/RWARE/data_scratch"
    # OFFROAD Dataset
    OFFROAD_DATA_DIR="/scratch/aparame/Research/VLCM_Data_Collection/OFFROAD/Varied_Traversability"

    # User stated data is already extracted in data_scratch
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Directory $DATA_DIR not found!"
        echo "Current directory content:"
        ls -l
        exit 1
    fi
    echo "Using existing data in $DATA_DIR"
    echo "Using OFFROAD data in $OFFROAD_DATA_DIR"
    
    # Point directly to the root data directory.
    # train.py will recursively find all .tar files in all subdirectories (configs).
    SHARD_PATTERN="$DATA_DIR"
fi

# 4. Run Training
echo "Running Training..."

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Set NUM_ROBOTS to the maximum expected across all configs (e.g. 8 for OFFROAD max).
# train.py will pad observations for configs with fewer robots (e.g. 2ag RWARE) to this size.
NUM_ROBOTS=8
echo "Using Max Num Robots: $NUM_ROBOTS"

SAVE_DIR="/scratch/aparame/Research/VLCM_checkpoints"
mkdir -p "$SAVE_DIR"
echo "Saving checkpoints to: $SAVE_DIR"


# Run with Singularity
# We bind the entire BASE_SCRATCH to ensure the container can access the tmp locations if needed
apptainer exec --nv -B "$PWD:$PWD" -B "$BASE_SCRATCH:$BASE_SCRATCH" \
  --env HF_TOKEN="$HF_TOKEN" \
  "$CONTAINER_PATH" accelerate launch --num_processes 2 train.py \
  --train_shards "$SHARD_PATTERN" \
  --offroad_shards "$OFFROAD_DATA_DIR" \
  --offroad_num_robots "$NUM_ROBOTS" \
  --dataset_type rware \
  --rware_config "$CONFIG_NAME" \
  --batch_size 10 \
  --grad_accum_steps 2 \
  --clip_len 16 \
  --num_robots "$NUM_ROBOTS" \
  --robot_obs_dim 8 \
  --epochs 10 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf \
  --save_dir "$SAVE_DIR" \
  --num_workers 16 \
  --mixed_precision bf16 \
  --freeze_vl \
  --peft lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --vision_lr 1e-5 \
  --loss_type contrastive_mse \
  --return_mode nstep \
  --mse_loss_weight 0.01 \
  --max_grad_norm 1.0 \
  --samples_per_epoch 50000 \
  --rware_visual_mode rware_only \
  --gamma 0.95 \
  --max_return_horizon 64 \
  --ema_decay 0.995 \
  --vl_max_text_len 4700

# Tar up results for transfer back (handled by transfer_output_files=checkpoints_rware)
#   --vl_backend llava_video \
#   --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf \

# apptainer exec --nv -B "$PWD:$PWD" -B "$BASE_SCRATCH:$BASE_SCRATCH" \
#   --env HF_TOKEN="$HF_TOKEN" \
#   "$CONTAINER_PATH" accelerate launch --num_processes 1 train.py \
#   --train_shards "$SHARD_PATTERN" \
#   --offroad_shards "$OFFROAD_DATA_DIR" \
#   --offroad_num_robots "$NUM_ROBOTS" \
#   --dataset_type rware \
#   --rware_config "$CONFIG_NAME" \
#   --batch_size 8 \
#   --grad_accum_steps 4 \
#   --clip_len 16 \
#   --num_robots "$NUM_ROBOTS" \
#   --robot_obs_dim 8 \
#   --epochs 10 \
#   --vl_backend llava_onevision \
#   --vl_model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
#   --save_dir "$SAVE_DIR" \
#   --num_workers 16 \
#   --mixed_precision bf16 \
#   --freeze_vl \
#   --peft lora \
#   --lora_r 16 \
#   --lora_alpha 32 \
#   --lora_dropout 0.05 \
#   --vision_lr 1e-5 \
#   --loss_type td \
#   --return_mode td \
#   --mse_loss_weight 0.5 \
#   --max_grad_norm 1.0 \
#   --samples_per_epoch 50000 \
#   --vl_max_text_len 7000