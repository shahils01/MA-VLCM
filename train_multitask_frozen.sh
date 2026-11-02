#!/usr/bin/env bash
# One supervised value critic, frozen vision/text backbone, both Python tasks.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${FORMATION_DATASET:?Set FORMATION_DATASET to the 3000-episode formation dataset}"
: "${GOTOGOAL_DATASET:?Set GOTOGOAL_DATASET to the 2000-episode go-to-goal dataset}"
CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/outputs/smolvlm-18918211/best.pt}"
if [[ -d "$CHECKPOINT" ]]; then
    CHECKPOINT="$CHECKPOINT/best.pt"
fi
launcher=("$PYTHON_BIN")
if (( ${NUM_GPUS:-1} > 1 )); then
    launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
fi
tracking=()
if [[ "${WANDB_ENABLED:-1}" == 1 ]]; then
    tracking+=(--wandb --wandb-project "${WANDB_PROJECT:-ma-vlcm-multitask}" \
        --wandb-entity "${WANDB_ENTITY:-}" --wandb-run-name "${WANDB_RUN_NAME:-vlcm-joint-${SLURM_JOB_ID:-local}}")
fi
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
exec "${launcher[@]}" "$ROOT/train_multitask.py" \
    --formation-dataset "$FORMATION_DATASET" --gotogoal-dataset "$GOTOGOAL_DATASET" \
    --init-checkpoint "$CHECKPOINT" \
    --output-dir "${RUN_DIR:-$ROOT/outputs/multitask_frozen_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}" \
    --cache-dir "${CACHE_DIR:-$ROOT/outputs/cache}" \
    --epochs "${EPOCHS:-20}" --lr "${LR:-.00003}" \
    --batch-size "${BATCH_SIZE:-64}" --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
    --num-workers "${NUM_WORKERS:-0}" --mixed-precision "${MIXED_PRECISION:-bf16}" \
    --image-hz 10 --gamma .99 --seed "${SEED:-42}" \
    "${tracking[@]}" "$@"
