#!/usr/bin/env bash
#SBATCH --job-name=vlcm_heads2
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Two cases × two CPUs = four CPUs total, with one A100 per case.
set -euo pipefail
CHECK_ONLY=0
case "${1:-}" in
  --check) CHECK_ONLY=1 ;;
  '') ;;
  *) echo 'Usage: sbatch train_joint_3456_two_palmetto.sbatch, or bash train_joint_3456_two_palmetto.sbatch --check'; exit 2 ;;
esac
if [[ "$CHECK_ONLY" == 0 && -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit with sbatch, or use bash train_joint_3456_two_palmetto.sbatch --check.' >&2
  exit 2
fi
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
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
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/joint_3456_two_${SLURM_JOB_ID:-preflight}}"
export CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/outputs/cache}"

# EDIT TRAINING SETTINGS HERE. Environment variables can override these defaults.
export CLIP_LEN="${CLIP_LEN:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4096}"
export VL_BATCH_SIZE="${VL_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-1}"  # One loader worker + one training process per case.
export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-.0001}"
CPUS_PER_CASE="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CASE_MEM_GB="${CASE_MEM_GB:-14}"
CASES=(agents3456_shared agents3456_task)
HEADS=(shared task)
COUNTS=('3 4 5 6' '3 4 5 6')
mkdir -p "$RUN_ROOT/logs"

if [[ "$CHECK_ONLY" == 1 ]]; then
  for i in 0 1; do
    printf 'Checking %s\n' "${CASES[i]}"
    NUM_GPUS=1 NUM_AGENTS="${COUNTS[i]}" RUN_DIR="$RUN_ROOT/preflight/${CASES[i]}" \
      WANDB_ENABLED=0 bash "$PROJECT_ROOT/train_joint_ablation.sh" "${HEADS[i]}" --preflight-only
  done
  exit
fi
if (( NUM_WORKERS + OMP_NUM_THREADS > CPUS_PER_CASE )); then
  echo 'NUM_WORKERS plus the main CPU thread exceeds --cpus-per-task.' >&2; exit 2
fi

step_pids=()
MONITOR_PID=''
cleanup() {
  for pid in "${step_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if command -v nvidia-smi >/dev/null && [[ "${GPU_MONITOR:-1}" == 1 ]]; then
  nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=30 > "$RUN_ROOT/logs/gpu.csv" &
  MONITOR_PID=$!
fi

printf 'Building/reusing image features with two GPUs. Log: %s/logs/cache.out\n' "$RUN_ROOT"
# Stage 1: cache features once using both GPUs and all four CPUs.
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$((2 * CPUS_PER_CASE))" \
  --gpus-per-node=a100:2 --gpus-per-task=a100:2 --gpu-bind=none --mem=0 \
  --output="$RUN_ROOT/logs/cache.out" --error="$RUN_ROOT/logs/cache.err" \
  env NUM_GPUS=2 NUM_AGENTS='3 4 5 6' RUN_DIR="$RUN_ROOT/cache_build" WANDB_ENABLED=0 \
  bash "$PROJECT_ROOT/train_joint_ablation.sh" shared --cache-only --require-cuda --expected-visible-gpus 2
export MODEL_REVISION
MODEL_REVISION="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_config"]["model_revision"])' "$RUN_ROOT/cache_build/cache_report.json")"
SPLIT_MANIFEST="$RUN_ROOT/cache_build/split_manifest.json"

# Stage 2: each independent training case gets one GPU and two CPUs.
for i in 0 1; do
  case_name="${CASES[i]}"
  printf 'Launching %s: agents=%s, batch=%s, CPUs=%s. Log: %s/logs/%s.out\n' \
    "$case_name" "${COUNTS[i]}" "$BATCH_SIZE" "$CPUS_PER_CASE" "$RUN_ROOT" "$case_name"
  # Explicit memory and exact CPU/GPU requests allow both steps to coexist.
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_CASE" \
    --gpus-per-node=a100:1 --gpus-per-task=a100:1 --gpu-bind=single:1 --mem="${CASE_MEM_GB}G" \
    --output="$RUN_ROOT/logs/$case_name.out" --error="$RUN_ROOT/logs/$case_name.err" \
    env NUM_GPUS=1 NUM_AGENTS="${COUNTS[i]}" RUN_DIR="$RUN_ROOT/$case_name" \
    WANDB_RUN_NAME="$case_name-${SEED:-42}-${SLURM_JOB_ID}" \
    bash "$PROJECT_ROOT/train_joint_ablation.sh" "${HEADS[i]}" \
    --require-cuda --expected-visible-gpus 1 --split-manifest "$SPLIT_MANIFEST" &
  step_pids+=("$!")
done

# Wait for every case and report failures instead of silently succeeding.
status=0
for i in 0 1; do
  if wait "${step_pids[i]}"; then
    code=0
  else
    code=$?; status=1
  fi
  unset 'step_pids[i]'
  printf '%s\n' "$code" > "$RUN_ROOT/${CASES[i]}.exitcode"
  printf '%s finished with exit code %s\n' "${CASES[i]}" "$code"
done
step_pids=()
exit "$status"
