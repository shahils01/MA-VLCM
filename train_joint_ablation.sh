#!/usr/bin/env bash
# Same data, seed and loss; change only one scalar head versus K scalar heads.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
if (( $# )); then shift; fi
case "$MODE" in
  shared|task|all) ;;
  *) echo "Usage: $0 {shared|task|all} [training flags]" >&2; exit 2 ;;
esac
if [[ "$MODE" == all ]]; then
  if [[ -n "${RUN_DIR:-}" ]]; then
    echo "Use RUN_ROOT for all; RUN_DIR is for one ablation." >&2; exit 2
  fi
  export RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/joint_heads_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
  for head in shared task; do
    bash "$ROOT/train_joint_ablation.sh" "$head" "$@"
    if [[ "$head" == shared && -f "$RUN_ROOT/shared/run.json" ]]; then
      # Pin the second run to the exact pretrained weights used by the first.
      export MODEL_REVISION="$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_config"]["model_revision"])' "$RUN_ROOT/shared/run.json")"
    fi
  done
  exit
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a AGENT_COUNTS <<< "${NUM_AGENTS:-3 4 5 6}"
export NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
CPU_BUDGET="${SLURM_CPUS_PER_TASK:-8}"
WORKERS_PER_RANK=$(( CPU_BUDGET / NUM_GPUS - 2 ))
if (( WORKERS_PER_RANK < 0 )); then WORKERS_PER_RANK=0; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
launcher=("$PYTHON_BIN")
if (( NUM_GPUS > 1 )); then
  launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS")
fi
tracking=()
if [[ "${WANDB_ENABLED:-1}" == 1 ]]; then
  tracking+=(--wandb --wandb-project "${WANDB_PROJECT:-ma-vlcm-joint-heads}" \
    --wandb-entity "${WANDB_ENTITY:-}" --wandb-run-name "${WANDB_RUN_NAME:-${MODE}-agents${AGENT_COUNTS[*]}-${SEED:-42}-${SLURM_JOB_ID:-local}}")
fi
exec "${launcher[@]}" "$ROOT/train_multitask.py" \
  --experiment joint --num-agents "${AGENT_COUNTS[@]}" \
  --value-head-mode "$MODE" --task-normalization --formation-context --balance-agent-counts \
  --formation-dataset "${FORMATION_DATASET:-/home/anshuln/Dataset/polygon_formation_python_env}" \
  --gotogoal-dataset "${GOTOGOAL_DATASET:-/home/anshuln/Dataset/manyagent_gotogoal_python_2000}" \
  --model-id "${MODEL_ID:-HuggingFaceTB/SmolVLM-500M-Instruct}" --model-revision "${MODEL_REVISION:-main}" \
  --output-dir "${RUN_DIR:-${RUN_ROOT:-$ROOT/outputs/joint_heads_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}/$MODE}" \
  --cache-dir "${CACHE_DIR:-$ROOT/outputs/cache}" \
  --epochs "${EPOCHS:-30}" --lr "${LR:-.0001}" --batch-size "${BATCH_SIZE:-1024}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" --num-workers "${NUM_WORKERS:-$WORKERS_PER_RANK}" \
  --prefetch-factor "${PREFETCH_FACTOR:-2}" --mixed-precision "${MIXED_PRECISION:-bf16}" \
  --image-hz 10 --expected-capture-hz 10 --expected-control-hz "${EXPECTED_CONTROL_HZ:-0}" \
  --gamma "${GAMMA:-.99}" --discount-reference-hz 20 \
  --image-size "${IMAGE_SIZE:-512}" --clip-len "${CLIP_LEN:-8}" \
  --vl-batch-size "${VL_BATCH_SIZE:-16}" --seed "${SEED:-42}" \
  "${tracking[@]}" "$@"
