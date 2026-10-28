#!/usr/bin/env bash
set -euo pipefail
export DATASET="${DATASET:-shulnak09/formation-vlcm-polygon-3to6}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
