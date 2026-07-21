#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE_PROFILE=qwen3_vl

exec bash "$SCRIPT_DIR/lora_run_train_tb3_lab.sh" "$@"
