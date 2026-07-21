#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "Starting TB3 lab MA-VLCM LoRA job on $(hostname)"
echo "Date: $(date)"

USER_NAME="${USER:-$(id -un 2>/dev/null || echo user)}"
if [ -n "${MA_VLCM_SCRATCH_ROOT:-}" ]; then
    SCRATCH_ROOT="$MA_VLCM_SCRATCH_ROOT"
    BASE_SCRATCH="$SCRATCH_ROOT"
elif [ -n "${SCRATCH:-}" ]; then
    BASE_SCRATCH="$SCRATCH"
    SCRATCH_ROOT="$BASE_SCRATCH/ma_vlcm"
elif [ -d "/scratch/$USER_NAME" ]; then
    BASE_SCRATCH="/scratch/$USER_NAME"
    SCRATCH_ROOT="$BASE_SCRATCH/ma_vlcm"
elif [ -d "/scratch/aparame" ]; then
    BASE_SCRATCH="/scratch/aparame"
    SCRATCH_ROOT="$BASE_SCRATCH/ma_vlcm"
elif [ -n "${SLURM_TMPDIR:-}" ]; then
    BASE_SCRATCH="$SLURM_TMPDIR"
    SCRATCH_ROOT="$BASE_SCRATCH/ma_vlcm"
else
    BASE_SCRATCH="$REPO_ROOT/.scratch"
    SCRATCH_ROOT="$BASE_SCRATCH"
    echo "WARNING: no scratch directory detected; using $SCRATCH_ROOT for caches and checkpoints."
fi

export MA_VLCM_SCRATCH_ROOT="$SCRATCH_ROOT"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/cache/triton}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH_ROOT/apptainer_tmp}"
mkdir -p \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME" \
    "$TRITON_CACHE_DIR" \
    "$WANDB_DIR" \
    "$TMPDIR" \
    "$APPTAINER_TMPDIR"

DATASET_PROFILE="${DATASET_PROFILE:-tb3_lab}"
case "${DATASET_PROFILE,,}" in
    tb3_lab|lab|real)
        DATASET_PROFILE="tb3_lab"
        DEFAULT_HF_DATASET_REPO="adi2440/tb3-lab-vlcm-progress-v1"
        DEFAULT_NUM_ROBOTS=3
        ;;
    tb3_isaac|isaac|isaac_sim|sim)
        DATASET_PROFILE="tb3_isaac"
        DEFAULT_HF_DATASET_REPO="adi2440/tb3-isaac-vlcm"
        DEFAULT_NUM_ROBOTS=6
        ;;
    *)
        echo "ERROR: DATASET_PROFILE must be tb3_lab or tb3_isaac (got: $DATASET_PROFILE)"
        exit 2
        ;;
esac

HF_DATASET_REPO="${HF_DATASET_REPO:-$DEFAULT_HF_DATASET_REPO}"
# Both collectors may keep shards in nested agents_XX/worker_XX folders. The
# MA-VLCM loader downloads and recursively expands this Hugging Face pattern.
DEFAULT_TB3_DATA="${DEFAULT_TB3_DATA:-hf://datasets/$HF_DATASET_REPO/**/*.tar}"
DATA_DIR="${1:-${DATA_DIR:-$DEFAULT_TB3_DATA}}"
NUM_ROBOTS="${NUM_ROBOTS:-$DEFAULT_NUM_ROBOTS}"

DEFAULT_RESUME_CANDIDATES=(
    "/scratch/aparame/VLCM_Data_Collection/checkpoints/NewFinal_0.5B.pt"
    "$SCRATCH_ROOT/checkpoints/NewFinal_0.5B.pt"
    "$REPO_ROOT/checkpoints/NewFinal_0.5B.pt"
)
if [ -z "${DEFAULT_RESUME_CHECKPOINT:-}" ]; then
    DEFAULT_RESUME_CHECKPOINT="${DEFAULT_RESUME_CANDIDATES[0]}"
    for candidate in "${DEFAULT_RESUME_CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            DEFAULT_RESUME_CHECKPOINT="$candidate"
            break
        fi
    done
fi
RESUME_CHECKPOINT="${2:-${RESUME_CHECKPOINT:-$DEFAULT_RESUME_CHECKPOINT}}"
TRAIN_FROM_SCRATCH="${TRAIN_FROM_SCRATCH:-0}"
case "${TRAIN_FROM_SCRATCH,,}" in
    1|true|yes|y|on)
        RESUME_CHECKPOINT=""
        ;;
esac
case "${RESUME_CHECKPOINT,,}" in
    none|null|false|0|scratch|from_scratch)
        RESUME_CHECKPOINT=""
        ;;
esac
SAVE_DIR="${SAVE_DIR:-$SCRATCH_ROOT/checkpoints/$DATASET_PROFILE}"
CONTAINER_PATH="${CONTAINER_PATH:-$REPO_ROOT/ma_vlcm.sif}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${EPOCHS:-20}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-${DATASET_PROFILE}_0.5B}"

case "$DATA_DIR" in
    hf://*|http://*|https://*|pipe:*)
        echo "Using remote TurtleBot dataset: $DATA_DIR"
        ;;
    *)
        if [ ! -d "$DATA_DIR" ] && [[ "$DATA_DIR" != *"*"* ]] && [[ "$DATA_DIR" != *"?"* ]]; then
            echo "ERROR: TurtleBot dataset directory or shard pattern not found: $DATA_DIR"
            exit 1
        fi
        echo "Using local TurtleBot dataset: $DATA_DIR"
        ;;
esac

if [ -n "$RESUME_CHECKPOINT" ] && [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: resume checkpoint not found: $RESUME_CHECKPOINT"
    exit 1
fi

mkdir -p "$SAVE_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_CMD=(
  accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision "$MIXED_PRECISION" -m ma_vlcm.train
  --train_shards "$DATA_DIR"
  --dataset_type tb3_lab
  --batch_size 4
  --grad_accum_steps 4
  --clip_len 16
  --num_robots "$NUM_ROBOTS"
  --robot_obs_dim 8
  --epochs "$TOTAL_EPOCHS"
  --vl_backend llava_onevision
  --vl_model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  --save_dir "$SAVE_DIR"
  --num_workers 4
  --mixed_precision "$MIXED_PRECISION"
  --freeze_vl
  --peft lora
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --vision_lr 1e-5
  --loss_type mse
  --return_mode nstep
  --target_mode progress
  --value_output_activation sigmoid
  --mse_loss_weight 1.0
  --max_grad_norm 1.0
  --samples_per_epoch 5000
  --gamma 0.95
  --max_return_horizon 64
  --ema_decay 0.995
  --vl_max_text_len 4700
  --run_name_prefix "$WANDB_RUN_PREFIX"
)

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD+=(--resume_from "$RESUME_CHECKPOINT")
    echo "Will resume from checkpoint: $RESUME_CHECKPOINT"
else
    echo "Training from scratch: no --resume_from checkpoint will be used."
fi

echo "Using TurtleBot dataset: $DATA_DIR"
echo "Dataset profile: $DATASET_PROFILE"
echo "Model robot slots: $NUM_ROBOTS"
echo "Scratch root: $SCRATCH_ROOT"
echo "Hugging Face cache: $HF_HOME"
echo "Torch cache: $TORCH_HOME"
echo "Saving checkpoints to: $SAVE_DIR"
echo "Total epoch target: $TOTAL_EPOCHS"
echo "Mixed precision: $MIXED_PRECISION"
echo "W&B run prefix: $WANDB_RUN_PREFIX"

if command -v apptainer >/dev/null 2>&1 && [ -f "$CONTAINER_PATH" ]; then
    APPTAINER_BINDS=(-B "$REPO_ROOT:$REPO_ROOT" -B "$BASE_SCRATCH:$BASE_SCRATCH" -B "$SAVE_DIR:$SAVE_DIR")
    if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
        RESUME_DIR=$(dirname "$RESUME_CHECKPOINT")
        APPTAINER_BINDS+=(-B "$RESUME_DIR:$RESUME_DIR")
    fi
    case "$DATA_DIR" in
        hf://*|http://*|https://*|pipe:*) ;;
        *)
            if [ -d "$DATA_DIR" ]; then
                APPTAINER_BINDS+=(-B "$DATA_DIR:$DATA_DIR")
            fi
            ;;
    esac
    APPTAINER_ENVS=(
      --env PYTHONPATH="$PYTHONPATH"
      --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
      --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
      --env HF_HOME="$HF_HOME"
      --env HF_HUB_CACHE="$HF_HUB_CACHE"
      --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
      --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
      --env TORCH_HOME="$TORCH_HOME"
      --env XDG_CACHE_HOME="$XDG_CACHE_HOME"
      --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
      --env WANDB_DIR="$WANDB_DIR"
      --env TMPDIR="$TMPDIR"
    )
    if [ -n "${HF_TOKEN:-}" ]; then
        APPTAINER_ENVS+=(--env HF_TOKEN="$HF_TOKEN")
    fi
    if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
        APPTAINER_ENVS+=(--env HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN")
    fi
    echo "Running inside Apptainer container: $CONTAINER_PATH"
    apptainer exec --nv \
      "${APPTAINER_BINDS[@]}" \
      "${APPTAINER_ENVS[@]}" \
      "$CONTAINER_PATH" "${TRAIN_CMD[@]}"
else
    echo "Running natively; set CONTAINER_PATH or install apptainer to use the container."
    "${TRAIN_CMD[@]}"
fi
