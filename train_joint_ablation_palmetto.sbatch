#!/usr/bin/env bash
#SBATCH --job-name=vlcm_heads
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  module load anaconda3
  set +u
  source activate "${CONDA_ENV:-internvl}"
  set -u
fi
# Observe actual utilization; throughput is also recorded per training epoch.
if command -v nvidia-smi >/dev/null && [[ "${GPU_MONITOR:-1}" == 1 ]]; then
  mkdir -p logs
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=30 > "logs/gpu-${SLURM_JOB_ID:-local}.csv" &
  MONITOR_PID=$!
  trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi
bash "$PROJECT_ROOT/train_joint_ablation.sh" "$@"
