#!/usr/bin/env bash
#SBATCH --job-name=vlcm_lora2
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Two independent LoRA critics; each Slurm step has one GPU.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit this launcher using sbatch.' >&2; exit 2
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  module load anaconda3
  set +u
  source activate "${CONDA_ENV:-internvl}"
  set -u
  export PYTHON_BIN="$(command -v python)"
fi
if [[ -d "$PROJECT_ROOT/outputs/training_dependencies" ]]; then
  export PYTHONPATH="$PROJECT_ROOT/outputs/training_dependencies${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FORMATION_DATASET="${FORMATION_DATASET:-/home/anshuln/Dataset/polygon_formation_python_env}"
export GOTOGOAL_DATASET="${GOTOGOAL_DATASET:-/home/anshuln/Dataset/manyagent_gotogoal_10m_3000/without_obstacles}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/joint_lora_no_obstacles_$SLURM_JOB_ID}"
export CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/outputs/cache}"
export CLIP_LEN="${CLIP_LEN:-8}" BATCH_SIZE="${BATCH_SIZE:-1}" VL_BATCH_SIZE="${VL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}" NUM_WORKERS="${NUM_WORKERS:-1}"
export EPOCHS="${EPOCHS:-30}" LR="${LR:-.0001}" SEED="${SEED:-42}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CPUS_PER_CASE="${SLURM_CPUS_PER_TASK:-2}"
CASE_MEM_GB="${CASE_MEM_GB:-28}"
if (( NUM_WORKERS + OMP_NUM_THREADS > CPUS_PER_CASE )); then
  echo 'NUM_WORKERS plus the main CPU thread exceeds --cpus-per-task.' >&2; exit 2
fi
mkdir -p "$RUN_ROOT/logs"
# Verify archive counts and matching team selections before spending GPU time.
env NUM_GPUS=1 NUM_AGENTS='3 4 5 6' WANDB_ENABLED=0 RUN_DIR="$RUN_ROOT/preflight" \
  bash "$PROJECT_ROOT/train_joint_ablation_lora.sh" shared --preflight-only --backbone-mode lora --experiment joint_without_obstacles \
  --expected-formation-episodes 3000 --expected-gotogoal-episodes 3000
# Validate the real model, PEFT adapter and CUDA gradients before the long runs.
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_CASE" \
  --gpus-per-node=a100:1 --gpus-per-task=a100:1 --gpu-bind=single:1 --mem="${CASE_MEM_GB}G" \
  --output="$RUN_ROOT/logs/lora_smoke.out" --error="$RUN_ROOT/logs/lora_smoke.err" \
  env RUN_DIR="$RUN_ROOT/gpu_smoke" NUM_GPUS=1 "$PYTHON_BIN" "$PROJECT_ROOT/tests/lora_gpu_smoke.py"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for head in shared task; do
  name="agents3456_${head}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_CASE" \
    --gpus-per-node=a100:1 --gpus-per-task=a100:1 --gpu-bind=single:1 --mem="${CASE_MEM_GB}G" \
    --output="$RUN_ROOT/logs/$name.out" --error="$RUN_ROOT/logs/$name.err" \
    env NUM_GPUS=1 NUM_AGENTS='3 4 5 6' RUN_DIR="$RUN_ROOT/$name" \
    WANDB_RUN_NAME="lora-$name-$SEED-$SLURM_JOB_ID" \
    bash "$PROJECT_ROOT/train_joint_ablation_lora.sh" "$head" \
      --backbone-mode lora --experiment joint_without_obstacles --require-cuda --expected-visible-gpus 1 &
  pids+=("$!")
done
status=0
for i in 0 1; do
  if wait "${pids[i]}"; then code=0; else code=$?; status=1; fi
  unset 'pids[i]'
  printf '%s\n' "$code" > "$RUN_ROOT/agents3456_$([[ $i == 0 ]] && echo shared || echo task).exitcode"
done
pids=()
exit "$status"
