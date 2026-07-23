#!/bin/bash
# Evaluate the latest LLaVA, Qwen3-VL, and V-JEPA2 critics on five full
# held-out TurtleBot episode shards and generate comparison plots/videos.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

USER_NAME="${USER:-$(id -un)}"
if [ -n "${MA_VLCM_SCRATCH_ROOT:-}" ]; then
    SCRATCH_ROOT="$MA_VLCM_SCRATCH_ROOT"
elif [ -d "/scratch/$USER_NAME" ]; then
    SCRATCH_ROOT="/scratch/$USER_NAME/ma_vlcm"
elif [ -d "/scratch/aparame" ]; then
    SCRATCH_ROOT="/scratch/aparame/ma_vlcm"
else
    SCRATCH_ROOT="$REPO_ROOT/.scratch"
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH_ROOT/checkpoints}"
DATASET="${DATASET:-hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/plots/tb3_episode_inference}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_SEED="${EPISODE_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
CLIP_STRIDE="${CLIP_STRIDE:-1}"
VIDEO_FPS="${VIDEO_FPS:-5}"
DEVICE="${DEVICE:-cuda}"
CONTAINER_PATH="${CONTAINER_PATH:-$REPO_ROOT/ma_vlcm.sif}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH_ROOT/cache/matplotlib}"
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR"

EVAL_ARGS=(
  --checkpoint-root "$CHECKPOINT_ROOT"
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --num-episodes "$NUM_EPISODES"
  --episode-seed "$EPISODE_SEED"
  --split-seed "$SPLIT_SEED"
  --val-split "$VAL_SPLIT"
  --clip-stride "$CLIP_STRIDE"
  --fps "$VIDEO_FPS"
  --device "$DEVICE"
  --models qwen3_vl vjepa2
)

if [ -n "${RUN_NAME:-}" ]; then
    EVAL_ARGS+=(--run-name "$RUN_NAME")
fi
if [ -n "${LLAVA_CHECKPOINT:-}" ]; then
    EVAL_ARGS+=(--llava-checkpoint "$LLAVA_CHECKPOINT")
fi
if [ -n "${QWEN3_VL_CHECKPOINT:-}" ]; then
    EVAL_ARGS+=(--qwen3-vl-checkpoint "$QWEN3_VL_CHECKPOINT")
fi
if [ -n "${VJEPA2_CHECKPOINT:-}" ]; then
    EVAL_ARGS+=(--vjepa2-checkpoint "$VJEPA2_CHECKPOINT")
fi
case "${NO_VIDEO:-0}" in
    1|true|yes|on) EVAL_ARGS+=(--no-video) ;;
esac
EVAL_ARGS+=("$@")

echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Dataset: $DATASET"
echo "Episodes: $NUM_EPISODES (held-out split=$VAL_SPLIT, seed=$EPISODE_SEED)"
echo "Output directory: $OUTPUT_DIR"

if [ -f "$CONTAINER_PATH" ] && command -v apptainer >/dev/null 2>&1; then
    BASE_SCRATCH=$(dirname "$SCRATCH_ROOT")
    CMD=(
      apptainer exec --nv
      -B "$REPO_ROOT:$REPO_ROOT"
      -B "$BASE_SCRATCH:$BASE_SCRATCH"
      --env "PYTHONPATH=$PYTHONPATH"
      --env "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
      --env "HF_HOME=$HF_HOME"
      --env "HF_HUB_CACHE=$HF_HUB_CACHE"
      --env "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
      --env "TORCH_HOME=$TORCH_HOME"
      --env "MPLCONFIGDIR=$MPLCONFIGDIR"
    )
    if [ -n "${HF_TOKEN:-}" ]; then
        CMD+=(--env "HF_TOKEN=$HF_TOKEN")
    fi
    CMD+=(
      "$CONTAINER_PATH"
      python3 "$REPO_ROOT/tools/evaluate_tb3_episode_progress.py"
      "${EVAL_ARGS[@]}"
    )
else
    CMD=(python3 "$REPO_ROOT/tools/evaluate_tb3_episode_progress.py" "${EVAL_ARGS[@]}")
fi

case "${DRY_RUN:-0}" in
    1|true|yes|on)
        printf 'Inference command:'
        printf ' %q' "${CMD[@]}"
        printf '\n'
        exit 0
        ;;
esac

"${CMD[@]}"
