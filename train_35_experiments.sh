#!/usr/bin/env bash
# Three independent SmolVLM-500M critic experiments; optional explicit warm start.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-formation_only}"
if (( $# )); then shift; fi
case "$EXPERIMENT" in
  formation_only|joint_without_obstacles|joint_with_obstacles|all) ;;
  *) echo "Usage: $0 {formation_only|joint_without_obstacles|joint_with_obstacles|all} [training flags]" >&2; exit 2 ;;
esac
if [[ "$EXPERIMENT" == all ]]; then
  if [[ -n "${RUN_DIR:-}" ]]; then
    echo "Use RUN_ROOT (one directory per experiment), rather than RUN_DIR, with all." >&2
    exit 2
  fi
  export RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/critic35_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
  for name in formation_only joint_without_obstacles joint_with_obstacles; do
    bash "$ROOT/train_35_experiments.sh" "$name" "$@"
  done
  exit
fi
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
PYTHON_BIN="${PYTHON_BIN:-python}"
launcher=("$PYTHON_BIN")
if (( ${NUM_GPUS:-1} > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
fi
tracking=()
if [[ "${WANDB_ENABLED:-1}" == 1 ]]; then
  tracking+=(--wandb --wandb-project "${WANDB_PROJECT:-ma-vlcm-35}" \
    --wandb-entity "${WANDB_ENTITY:-}" --wandb-run-name "${WANDB_RUN_NAME:-${EXPERIMENT}-${SLURM_JOB_ID:-local}}")
fi
initialization=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  checkpoint="$INIT_CHECKPOINT"
  if [[ -d "$checkpoint" ]]; then checkpoint="$checkpoint/best.pt"; fi
  initialization+=(--init-checkpoint "$checkpoint")
fi
navigation=()
if [[ "$EXPERIMENT" != formation_only ]]; then
  navigation+=(--gotogoal-dataset "${GOTOGOAL_DATASET:-/home/anshuln/Dataset/manyagent_gotogoal_10m_3000}" \
    --expected-gotogoal-episodes "${EXPECTED_GOTOGOAL_EPISODES:-3000}")
fi
exec "${launcher[@]}" "$ROOT/train_multitask.py" \
  --experiment "$EXPERIMENT" --num-agents 3 5 \
  --formation-dataset "${FORMATION_DATASET:-/home/anshuln/Dataset/polygon_formation_python_env}" \
  --expected-formation-episodes "${EXPECTED_FORMATION_EPISODES:-0}" \
  --model-id "${MODEL_ID:-HuggingFaceTB/SmolVLM-500M-Instruct}" \
  --model-revision "${MODEL_REVISION:-main}" \
  --output-dir "${RUN_DIR:-${RUN_ROOT:-$ROOT/outputs/critic35_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}/$EXPERIMENT}" \
  --cache-dir "${CACHE_DIR:-$ROOT/outputs/cache}" \
  --epochs "${EPOCHS:-30}" --lr "${LR:-.0001}" --batch-size "${BATCH_SIZE:-64}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" --num-workers "${NUM_WORKERS:-0}" \
  --mixed-precision "${MIXED_PRECISION:-bf16}" --image-hz 10 --gamma .99 \
  --image-size "${IMAGE_SIZE:-512}" --clip-len "${CLIP_LEN:-6}" \
  --vl-batch-size "${VL_BATCH_SIZE:-4}" --seed "${SEED:-42}" \
  "${navigation[@]}" "${initialization[@]}" "${tracking[@]}" "$@"
