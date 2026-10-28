#!/usr/bin/env bash
# Legacy entry point: the small SmolVLM pipeline now uses DDP rather than FSDP.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_ddp.sh" "$@"
