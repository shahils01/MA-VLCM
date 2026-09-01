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
        ;;
    tb3_isaac|isaac|isaac_sim|sim)
        DATASET_PROFILE="tb3_isaac"
        DEFAULT_HF_DATASET_REPO="adi2440/tb3-isaac-vlcm"
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
DATA_DIR="${1:-${DATA_DIR:-${TB3_TRAIN_SOURCES:-$DEFAULT_TB3_DATA}}}"

BACKBONE_PROFILE="${BACKBONE_PROFILE:-llava_onevision}"
case "${BACKBONE_PROFILE,,}" in
    llava_onevision|llava|onevision)
        BACKBONE_PROFILE="llava_onevision"
        DEFAULT_VL_BACKEND="llava_onevision"
        DEFAULT_VL_MODEL_NAME="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        DEFAULT_BATCH_SIZE=4
        DEFAULT_GRAD_ACCUM_STEPS=4
        DEFAULT_CLIP_LEN=16
        DEFAULT_VL_MAX_TEXT_LEN=4700
        BACKBONE_RUN_LABEL="llava_onevision_0.5b"
        ;;
    qwen3_vl|qwen3|qwen)
        BACKBONE_PROFILE="qwen3_vl"
        DEFAULT_VL_BACKEND="qwen3_vl"
        DEFAULT_VL_MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
        DEFAULT_BATCH_SIZE=1
        DEFAULT_GRAD_ACCUM_STEPS=16
        DEFAULT_CLIP_LEN=16
        DEFAULT_VL_MAX_TEXT_LEN=32768
        BACKBONE_RUN_LABEL="qwen3_vl_2b"
        ;;
    vjepa2|v_jepa2|jepa)
        BACKBONE_PROFILE="vjepa2"
        DEFAULT_VL_BACKEND="vjepa2"
        DEFAULT_VL_MODEL_NAME="facebook/vjepa2-vitl-fpc64-256"
        DEFAULT_BATCH_SIZE=2
        DEFAULT_GRAD_ACCUM_STEPS=8
        DEFAULT_CLIP_LEN=16
        DEFAULT_VL_MAX_TEXT_LEN=256
        BACKBONE_RUN_LABEL="vjepa2_vitl"
        ;;
    *)
        echo "ERROR: BACKBONE_PROFILE must be llava_onevision, qwen3_vl, or vjepa2 (got: $BACKBONE_PROFILE)"
        exit 2
        ;;
esac

VL_BACKEND="${VL_BACKEND:-$DEFAULT_VL_BACKEND}"
VL_MODEL_NAME="${VL_MODEL_NAME:-$DEFAULT_VL_MODEL_NAME}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$DEFAULT_GRAD_ACCUM_STEPS}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLIP_LEN="${CLIP_LEN:-$DEFAULT_CLIP_LEN}"
VL_MAX_TEXT_LEN="${VL_MAX_TEXT_LEN:-$DEFAULT_VL_MAX_TEXT_LEN}"
TB3_DATASET_TYPE="${TB3_DATASET_TYPE:-tb3_lab}"
TB3_TARGET_SCHEMA="${TB3_TARGET_SCHEMA:-tb3_progress_v1}"
ROBOT_OBS_DIM="${ROBOT_OBS_DIM:-8}"
SUCCESS_ONLY="${SUCCESS_ONLY:-0}"
TRAINED_AGENT_COUNTS="${TRAINED_AGENT_COUNTS:-3}"
TASK_DOMAINS="${TASK_DOMAINS:-goal_to_goal}"
LAYOUT_SPLIT="${LAYOUT_SPLIT:-seen_train}"
BALANCE_TB3_SOURCES="${BALANCE_TB3_SOURCES:-0}"
TB3_BALANCE_MODE="${TB3_BALANCE_MODE:-domain_cardinality}"
TB3_IMAGE_MODE="${TB3_IMAGE_MODE:-center_square}"
TB3_IMAGE_SIZE="${TB3_IMAGE_SIZE:-336}"
TASK_DOMAIN_CONDITIONING="${TASK_DOMAIN_CONDITIONING:-0}"
TEMPORAL_CONSISTENCY_WEIGHT="${TEMPORAL_CONSISTENCY_WEIGHT:-0.0}"
PROGRESS_DISTANCE_MODE="${PROGRESS_DISTANCE_MODE:-euclidean}"
QWEN_VIDEO_FPS="${QWEN_VIDEO_FPS:-5.0}"
FINETUNE_MODE="${FINETUNE_MODE:-lora}"
case "${FINETUNE_MODE,,}" in
    lora|all_lora)
        FINETUNE_MODE="lora"
        PEFT_MODE="lora"
        LORA_SCOPE="all"
        FREEZE_VL=1
        FREEZE_VISION_TOWER=0
        ;;
    qlora)
        if [ "$BACKBONE_PROFILE" = "vjepa2" ]; then
            echo "ERROR: V-JEPA2 does not support QLoRA."
            exit 2
        fi
        FINETUNE_MODE="qlora"
        PEFT_MODE="qlora"
        LORA_SCOPE="all"
        FREEZE_VL=1
        FREEZE_VISION_TOWER=0
        ;;
    language_lora|language-only-lora|language_only_lora)
        if [ "$BACKBONE_PROFILE" = "vjepa2" ]; then
            echo "ERROR: V-JEPA2 has no language model; language_lora is unavailable."
            exit 2
        fi
        FINETUNE_MODE="language_lora"
        PEFT_MODE="lora"
        LORA_SCOPE="language"
        FREEZE_VL=1
        FREEZE_VISION_TOWER=1
        ;;
    vision_lora|vision-only-lora|vision_only_lora)
        FINETUNE_MODE="vision_lora"
        PEFT_MODE="lora"
        LORA_SCOPE="vision"
        FREEZE_VL=1
        FREEZE_VISION_TOWER=0
        ;;
    full|full_finetune|full-finetune)
        FINETUNE_MODE="full"
        PEFT_MODE="none"
        LORA_SCOPE="all"
        FREEZE_VL=0
        FREEZE_VISION_TOWER=0
        ;;
    vision_full|vision-only-full|vision_only_full)
        FINETUNE_MODE="vision_full"
        PEFT_MODE="none"
        LORA_SCOPE="vision"
        FREEZE_VL=1
        FREEZE_VISION_TOWER=0
        ;;
    heads_only|heads-only|head_only)
        FINETUNE_MODE="heads_only"
        PEFT_MODE="none"
        LORA_SCOPE="all"
        FREEZE_VL=1
        FREEZE_VISION_TOWER=1
        ;;
    *)
        echo "ERROR: FINETUNE_MODE must be lora, qlora, language_lora, vision_lora, full, vision_full, or heads_only (got: $FINETUNE_MODE)"
        exit 2
        ;;
esac
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
HEAD_LR="${HEAD_LR:-3e-4}"
if [ "$FINETUNE_MODE" = "full" ]; then
    DEFAULT_BACKBONE_LR="1e-5"
else
    DEFAULT_BACKBONE_LR="$HEAD_LR"
fi
BACKBONE_LR="${BACKBONE_LR:-$DEFAULT_BACKBONE_LR}"
VISION_LR="${VISION_LR:-1e-5}"

DEFAULT_RESUME_CANDIDATES=(
    "/scratch/aparame/VLCM_Data_Collection/checkpoints/NewFinal_0.5B.pt"
    "$SCRATCH_ROOT/checkpoints/NewFinal_0.5B.pt"
    "$REPO_ROOT/checkpoints/NewFinal_0.5B.pt"
)
if [ "$BACKBONE_PROFILE" = "llava_onevision" ] && [ "$FINETUNE_MODE" = "lora" ] && [ -z "${DEFAULT_RESUME_CHECKPOINT:-}" ]; then
    DEFAULT_RESUME_CHECKPOINT="${DEFAULT_RESUME_CANDIDATES[0]}"
    for candidate in "${DEFAULT_RESUME_CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            DEFAULT_RESUME_CHECKPOINT="$candidate"
            break
        fi
    done
fi
if [ "$BACKBONE_PROFILE" != "llava_onevision" ] || [ "$FINETUNE_MODE" != "lora" ]; then
    DEFAULT_RESUME_CHECKPOINT="${DEFAULT_RESUME_CHECKPOINT:-}"
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
if [ "$BACKBONE_PROFILE" = "llava_onevision" ] && [ "$FINETUNE_MODE" = "lora" ]; then
    DEFAULT_SAVE_DIR="$SCRATCH_ROOT/checkpoints/$DATASET_PROFILE"
elif [ "$FINETUNE_MODE" = "lora" ]; then
    DEFAULT_SAVE_DIR="$SCRATCH_ROOT/checkpoints/$DATASET_PROFILE/$BACKBONE_PROFILE"
else
    DEFAULT_SAVE_DIR="$SCRATCH_ROOT/checkpoints/$DATASET_PROFILE/$BACKBONE_PROFILE/$FINETUNE_MODE"
fi
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-5000}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SAVE_DIR="${SAVE_DIR:-$DEFAULT_SAVE_DIR}"
CONTAINER_PATH="${CONTAINER_PATH:-$REPO_ROOT/ma_vlcm.sif}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${EPOCHS:-20}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
if [ "$FINETUNE_MODE" = "lora" ]; then
    DEFAULT_WANDB_RUN_PREFIX="${DATASET_PROFILE}_${BACKBONE_RUN_LABEL}"
else
    DEFAULT_WANDB_RUN_PREFIX="${DATASET_PROFILE}_${BACKBONE_RUN_LABEL}_${FINETUNE_MODE}"
fi
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-$DEFAULT_WANDB_RUN_PREFIX}"

case "$DATA_DIR" in
    *";"*)
        echo "Using multiple TurtleBot dataset sources: $DATA_DIR"
        ;;
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
  --dataset_type "$TB3_DATASET_TYPE"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --clip_len "$CLIP_LEN"
  --robot_obs_dim "$ROBOT_OBS_DIM"
  --target_schema "$TB3_TARGET_SCHEMA"
  --progress_distance_mode "$PROGRESS_DISTANCE_MODE"
  --qwen_video_fps "$QWEN_VIDEO_FPS"
  --tb3_image_mode "$TB3_IMAGE_MODE"
  --resize_width "$TB3_IMAGE_SIZE"
  --resize_height "$TB3_IMAGE_SIZE"
  --trained_agent_counts "$TRAINED_AGENT_COUNTS"
  --task_domains "$TASK_DOMAINS"
  --tb3_balance_mode "$TB3_BALANCE_MODE"
  --layout_split "$LAYOUT_SPLIT"
  --modalities video,robot_obs,adj
  --epochs "$TOTAL_EPOCHS"
  --vl_backend "$VL_BACKEND"
  --vl_model_name "$VL_MODEL_NAME"
  --save_dir "$SAVE_DIR"
  --num_workers "$NUM_WORKERS"
  --mixed_precision "$MIXED_PRECISION"
  --peft "$PEFT_MODE"
  --lora_scope "$LORA_SCOPE"
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --lr "$HEAD_LR"
  --backbone_lr "$BACKBONE_LR"
  --vision_lr "$VISION_LR"
  --loss_type mse
  --return_mode nstep
  --target_mode progress
  --value_output_activation sigmoid
  --mse_loss_weight 1.0
  --temporal_consistency_loss_weight "$TEMPORAL_CONSISTENCY_WEIGHT"
  --max_grad_norm 1.0
  --samples_per_epoch "$SAMPLES_PER_EPOCH"
  --val_split "$VAL_SPLIT"
  --split_seed "$SPLIT_SEED"
  --gamma 0.95
  --max_return_horizon 64
  --ema_decay 0.995
  --vl_max_text_len "$VL_MAX_TEXT_LEN"
  --run_name_prefix "$WANDB_RUN_PREFIX"
)

if [ "$TASK_DOMAIN_CONDITIONING" = "1" ]; then
    TRAIN_CMD+=(--task_domain_conditioning)
fi

if [ "$SUCCESS_ONLY" = "1" ]; then
    TRAIN_CMD+=(--success_only)
fi

if [ "$BALANCE_TB3_SOURCES" = "1" ]; then
    TRAIN_CMD+=(--balance_tb3_sources)
fi

if [ "$FREEZE_VL" = "1" ]; then
    TRAIN_CMD+=(--freeze_vl)
fi
if [ "$FREEZE_VISION_TOWER" = "1" ]; then
    TRAIN_CMD+=(--freeze_vision_tower)
fi
if [ -n "$LORA_TARGET_MODULES" ]; then
    TRAIN_CMD+=(--lora_target_modules "$LORA_TARGET_MODULES")
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD+=(--resume_from "$RESUME_CHECKPOINT")
    echo "Will resume from checkpoint: $RESUME_CHECKPOINT"
else
    echo "Training from scratch: no --resume_from checkpoint will be used."
fi

echo "Using TurtleBot dataset: $DATA_DIR"
echo "Dataset profile: $DATASET_PROFILE"
echo "Backbone profile: $BACKBONE_PROFILE ($VL_BACKEND, $VL_MODEL_NAME)"
echo "Clip/batch/accumulation/workers: $CLIP_LEN / $BATCH_SIZE / $GRAD_ACCUM_STEPS / $NUM_WORKERS"
echo "Fine-tune mode: $FINETUNE_MODE (PEFT=$PEFT_MODE, LoRA scope=$LORA_SCOPE)"
echo "Learning rates: heads=$HEAD_LR, language=$BACKBONE_LR, vision=$VISION_LR"
echo "Robot cardinality: inferred per episode (minibatches are padded dynamically)"
echo "TB3 source balance mode: $TB3_BALANCE_MODE"
echo "TB3 image canonicalization: $TB3_IMAGE_MODE -> ${TB3_IMAGE_SIZE}x${TB3_IMAGE_SIZE} (no rotation or reflection)"
echo "Task-domain conditioning: $TASK_DOMAIN_CONDITIONING"
echo "Temporal consistency weight: $TEMPORAL_CONSISTENCY_WEIGHT"
echo "Progress distance mode: $PROGRESS_DISTANCE_MODE"
echo "Qwen source-video timestamp rate: ${QWEN_VIDEO_FPS} Hz"
echo "Scratch root: $SCRATCH_ROOT"
echo "Hugging Face cache: $HF_HOME"
echo "Torch cache: $TORCH_HOME"
echo "Saving checkpoints to: $SAVE_DIR"
echo "Total epoch target: $TOTAL_EPOCHS"
echo "Global clips per epoch: $SAMPLES_PER_EPOCH"
echo "Validation split / seed: $VAL_SPLIT / $SPLIT_SEED"
echo "Mixed precision: $MIXED_PRECISION"
echo "W&B run prefix: $WANDB_RUN_PREFIX"

case "${DRY_RUN:-0}" in
    1|true|yes|y|on)
        printf 'Training command:'
        printf ' %q' "${TRAIN_CMD[@]}"
        printf '\n'
        exit 0
        ;;
esac

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
