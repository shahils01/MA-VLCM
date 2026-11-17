#!/usr/bin/env bash
#SBATCH --job-name=marl_frozen_joint
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Two independent one-GPU workers; this is not distributed policy optimization.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export MARL_ROOT="${MARL_ROOT:-$(dirname "$PROJECT_ROOT")/MARL_data_collection}"
export VLCM_CHECKPOINT="${VLCM_CHECKPOINT:-$PROJECT_ROOT/outputs/ckpts/critic35_sequential/joint_with_obstacles/best.pt}"
export DATASET_ROOT="${DATASET_ROOT:-/home/${USER}/Dataset}"
if [[ -z "${VLCM_FORMATION_ROOT:-}" ]]; then
  export VLCM_FORMATION_ROOT="$PROJECT_ROOT/outputs/eval_rollouts/assets/formation"
  if [[ ! -f "$VLCM_FORMATION_ROOT/collection_plan.json" ]]; then
    export VLCM_FORMATION_ROOT="$DATASET_ROOT/polygon_formation_python_env"
  fi
fi
if [[ -z "${VLCM_NAVIGATION_ROOT:-}" ]]; then
  export VLCM_NAVIGATION_ROOT="$(dirname "$PROJECT_ROOT")/manyagent_gotogoal_10m_3000/provenance/clean/without_obstacles"
  if [[ ! -f "$VLCM_NAVIGATION_ROOT/collection_plan.json" ]]; then
    export VLCM_NAVIGATION_ROOT="$DATASET_ROOT/manyagent_gotogoal_10m_3000/provenance/clean/without_obstacles"
  fi
fi
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/frozen_joint_palmetto_${SLURM_JOB_ID:?Submit this script using sbatch}}"
export NUM_ENV_STEPS="${NUM_ENV_STEPS:-5000000}"
export NUM_ENVS="${NUM_ENVS:-16}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-300}"
export VISION_BATCH="${VISION_BATCH:-4}"
export SEED="${SEED:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DISABLE_GIT=true WANDB_DISABLE_CODE=true PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -z "${PYTHON_BIN:-}" ]]; then
  module load anaconda3
  set +u
  source activate "${CONDA_ENV:-internvl}"
  set -u
  export PYTHON_BIN
  PYTHON_BIN="$(command -v python)"
fi
if [[ -d "$PROJECT_ROOT/outputs/training_dependencies" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/outputs/training_dependencies${PYTHONPATH:+:$PYTHONPATH}"
fi
for required in "$VLCM_CHECKPOINT" "$MARL_ROOT/train_frozen_joint_policies.py" \
  "$PROJECT_ROOT/frozen_joint_palmetto_worker.sh" \
  "$VLCM_FORMATION_ROOT/collection_plan.json" "$VLCM_NAVIGATION_ROOT/collection_plan.json"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done
# Verify the frozen source snapshots before reserving separate GPU steps.
"$PYTHON_BIN" - "$MARL_ROOT" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from train_frozen_joint_policies import verify_sources, native_actor_args
verify_sources();native_actor_args()
PY
[[ ! -e "$RUN_ROOT" ]] || { echo "Use a fresh RUN_ROOT: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
printf 'Outputs: %s\nWorker 0: navigation, then physical-obstacle formation\nWorker 1: formation without obstacles\n' "$RUN_ROOT"

workers=()
cleanup() {
  if ((${#workers[@]})); then kill "${workers[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for worker in 0 1; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    --mem=32G --gpus-per-node=a100:1 --gpus-per-task=a100:1 --gpu-bind=single:1 \
    bash "$PROJECT_ROOT/frozen_joint_palmetto_worker.sh" "$worker" \
    >"$RUN_ROOT/worker_${worker}.log" 2>&1 &
  workers+=("$!")
done
status=0
for pid in "${workers[@]}"; do
  wait "$pid" || status=1
done
workers=()
exit "$status"
