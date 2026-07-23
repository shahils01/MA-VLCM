#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE_PROFILE=qwen3_vl
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-12}"

exec bash "$SCRIPT_DIR/lora_run_train_tb3_lab.sh" "$@"
