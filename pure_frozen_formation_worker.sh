#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
  echo 'The worker and its preflight must run inside srun on a compute node.' >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
export RUN_DIR="$RUN_ROOT/policy_seed1"
"$PYTHON_BIN" "$PROJECT_ROOT/benchmark_frozen_python.py" \
  --checkpoint "$VLCM_CHECKPOINT" --output "$RUN_ROOT/vision_benchmark.json"
VL_BATCH_SIZE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_vl_batch_size"])' "$RUN_ROOT/vision_benchmark.json")"
exec bash "$PROJECT_ROOT/train_frozen_python_value.sh" \
  --device cuda --vl-batch-size "$VL_BATCH_SIZE" --critic-batch-size 256 \
  --eval-interval 10 --eval-episodes 4 \
  --run-name "pure-frozen-python-scratch-seed1-${SLURM_JOB_ID}"
