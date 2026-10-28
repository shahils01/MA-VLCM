#!/usr/bin/env bash
set -euo pipefail
export NUM_GPUS="${NUM_GPUS:-2}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
