#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
    echo "Provide a relabeled potential-shard directory or glob."
    exit 2
fi
export DATASET_PROFILE=tb3_isaac
export TB3_DATASET_TYPE=tb3_potential_v1
export TB3_TARGET_SCHEMA=tb3_potential_v1
export POTENTIAL_CONFIG="${POTENTIAL_CONFIG:-$REPO_ROOT/configs/potential_v1.json}"
export LOSS_TYPE="${LOSS_TYPE:-categorical_ce}"
if [ "$LOSS_TYPE" = "categorical_ce" ]; then
    export VALUE_DISTRIBUTION_BINS="${VALUE_DISTRIBUTION_BINS:-21}"
    export VALUE_OUTPUT_ACTIVATION=identity
else
    export VALUE_DISTRIBUTION_BINS=0
    export VALUE_OUTPUT_ACTIVATION=tanh
fi
export SUCCESS_ONLY="${SUCCESS_ONLY:-0}"
export PROGRESS_DISTANCE_MODE="${PROGRESS_DISTANCE_MODE:-euclidean}"
# ALWAYS use both source datasets and fresh critics. Default to 16 frames;
# explicitly requested clip-length experiments may override CLIP_LEN.
# These are initial regularization weights, to be assessed on held-out data.
export TEMPORAL_CONSISTENCY_WEIGHT="${TEMPORAL_CONSISTENCY_WEIGHT:-1.0}"
export CONDITIONAL_LIPSCHITZ_WEIGHT="${CONDITIONAL_LIPSCHITZ_WEIGHT:-1.0}"
export LIPSCHITZ_POSITION_SCALE_M="${LIPSCHITZ_POSITION_SCALE_M:-0.1}"
export LIPSCHITZ_PIXEL_SCALE="${LIPSCHITZ_PIXEL_SCALE:-0.1}"
# Fresh VLCM training: NEVER load a previous critic checkpoint.
export TRAIN_FROM_SCRATCH=1
export DEFAULT_RESUME_CHECKPOINT=
export RESUME_CHECKPOINT=
export BACKBONE_PROFILE="${BACKBONE_PROFILE:-vjepa2}"
export SAVE_DIR="${SAVE_DIR:-/scratch/$USER/VLCM/ma_vlcm/checkpoints/tb3_potential_v1/$BACKBONE_PROFILE}"
export WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-tb3_potential_v1_$BACKBONE_PROFILE}"
export TB3_TRAIN_SOURCES="$DATA_DIR"
source "$SCRIPT_DIR/require_both_tb3_sources.sh" "$DATA_DIR"

case "$BACKBONE_PROFILE" in
    vjepa2) exec bash "$SCRIPT_DIR/lora_run_train_tb3_vjepa2.sh" "$DATA_DIR" ;;
    qwen3_vl) exec bash "$SCRIPT_DIR/lora_run_train_tb3_qwen3_vl.sh" "$DATA_DIR" ;;
    *) echo "BACKBONE_PROFILE must be vjepa2 or qwen3_vl"; exit 2 ;;
esac
