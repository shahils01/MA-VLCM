#!/usr/bin/env bash
# Compatibility entry point for the main formation trainer.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
