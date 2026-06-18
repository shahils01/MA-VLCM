#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "Starting TB3 lab MA-VLCM LoRA job on $(hostname)"
echo "Date: $(date)"

DATA_DIR="${1:-$REPO_ROOT/data/tb3_lab}"
DEFAULT_RESUME_CHECKPOINT="${DEFAULT_RESUME_CHECKPOINT:-$REPO_ROOT/checkpoints/NewFinal_0.5B.pt}"
RESUME_CHECKPOINT="${2:-${RESUME_CHECKPOINT:-$DEFAULT_RESUME_CHECKPOINT}}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs/checkpoints/tb3_lab}"
CONTAINER_PATH="${CONTAINER_PATH:-$REPO_ROOT/ma_vlcm.sif}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: TB3 lab dataset directory not found: $DATA_DIR"
    exit 1
fi

if [ -n "$RESUME_CHECKPOINT" ] && [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: resume checkpoint not found: $RESUME_CHECKPOINT"
    exit 1
fi

mkdir -p "$SAVE_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_CMD=(
  accelerate launch --num_processes "$NUM_PROCESSES" -m ma_vlcm.train
  --train_shards "$DATA_DIR"
  --dataset_type tb3_lab
  --batch_size 4
  --grad_accum_steps 4
  --clip_len 16
  --num_robots 3
  --robot_obs_dim 8
  --epochs 10
  --vl_backend llava_onevision
  --vl_model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  --save_dir "$SAVE_DIR"
  --num_workers 4
  --mixed_precision bf16
  --freeze_vl
  --peft lora
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --vision_lr 1e-5
  --loss_type contrastive_mse
  --return_mode nstep
  --mse_loss_weight 0.01
  --max_grad_norm 1.0
  --samples_per_epoch 5000
  --gamma 0.95
  --max_return_horizon 64
  --ema_decay 0.995
  --vl_max_text_len 4700
)

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD+=(--resume_from "$RESUME_CHECKPOINT")
    echo "Will resume from checkpoint: $RESUME_CHECKPOINT"
fi

echo "Using TB3 lab dataset: $DATA_DIR"
echo "Saving checkpoints to: $SAVE_DIR"

if command -v apptainer >/dev/null 2>&1 && [ -f "$CONTAINER_PATH" ]; then
    if [ -n "$SCRATCH" ]; then
        BASE_SCRATCH="$SCRATCH"
    elif [ -d "/scratch/$USER" ]; then
        BASE_SCRATCH="/scratch/$USER"
    else
        BASE_SCRATCH="$REPO_ROOT"
    fi
    echo "Running inside Apptainer container: $CONTAINER_PATH"
    apptainer exec --nv \
      -B "$REPO_ROOT:$REPO_ROOT" \
      -B "$DATA_DIR:$DATA_DIR" \
      -B "$SAVE_DIR:$SAVE_DIR" \
      -B "$BASE_SCRATCH:$BASE_SCRATCH" \
      --env PYTHONPATH="$PYTHONPATH" \
      --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
      --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      "$CONTAINER_PATH" "${TRAIN_CMD[@]}"
else
    echo "Running natively; set CONTAINER_PATH or install apptainer to use the container."
    "${TRAIN_CMD[@]}"
fi
