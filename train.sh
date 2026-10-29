#!/usr/bin/env bash
# Local RTX 3090: activate the vlcm-smol environment, then run bash train.sh.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKBONE_MODE="${BACKBONE_MODE:-frozen}"
if [[ "$BACKBONE_MODE" == frozen ]]; then
  default_batch=64
  default_vl_batch=4
else
  default_batch=1
  default_vl_batch=1
fi
launcher=("$PYTHON_BIN")
if (( ${NUM_GPUS:-1} > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
fi
tracking=()
if [[ "${WANDB_ENABLED:-1}" == 1 ]]; then
  tracking+=(--wandb --wandb-project "${WANDB_PROJECT:-ma-vlcm}" --wandb-run-name "${WANDB_RUN_NAME:-smolvlm-${SLURM_JOB_ID:-local}}" --wandb-entity "${WANDB_ENTITY:-}")
fi
"${launcher[@]}" "$PROJECT_ROOT/train.py" \
  --dataset "${DATASET:-$PROJECT_ROOT/../Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6}" \
  --dataset-revision "${DATASET_REVISION:-main}" \
  --model-id "${MODEL_ID:-HuggingFaceTB/SmolVLM-500M-Instruct}" \
  --model-revision "${MODEL_REVISION:-main}" \
  --backbone-mode "$BACKBONE_MODE" \
  --return-source "${RETURN_SOURCE:-sampled}" \
  --clip-len "${CLIP_LEN:-8}" --image-hz 5 --gamma "${GAMMA:-0.99}" \
  --batch-size "${BATCH_SIZE:-$default_batch}" --vl-batch-size "${VL_BATCH_SIZE:-$default_vl_batch}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --mixed-precision "${MIXED_PRECISION:-bf16}" \
  --epochs "${EPOCHS:-30}" --checkpoint-every "${CHECKPOINT_EVERY:-2}" \
  --patience "${PATIENCE:-6}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --cache-dir "${CACHE_DIR:-$PROJECT_ROOT/outputs/cache}" \
  --output-dir "${RUN_DIR:-$PROJECT_ROOT/outputs/smolvlm-$(date +%Y%m%d-%H%M%S)}" \
  "${tracking[@]}" "$@"
