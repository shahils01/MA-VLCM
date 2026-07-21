#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE_PROFILE=vjepa2

exec bash "$SCRIPT_DIR/lora_run_train_tb3_lab.sh" "$@"
