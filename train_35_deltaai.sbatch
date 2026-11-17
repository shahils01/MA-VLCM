#!/usr/bin/env bash
#SBATCH --job-name=vlcm_35
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Supply --account=YOUR_PROJECT when submitting; DeltaAI GPUs are H100/GH200.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
module load "${PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  set +u
  conda activate "${CONDA_ENV:-base}"
  set -u
  export PYTHON_BIN="$(command -v python)"
fi
export NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
if [[ ! "$NUM_GPUS" =~ ^[1-4]$ ]]; then
  echo "NUM_GPUS must be an integer from 1 to 4 on a single DeltaAI node." >&2
  exit 2
fi
export FORMATION_DATASET="${FORMATION_DATASET:-$HOME/VLCM/datasets/polygon_formation_python_env}"
export GOTOGOAL_DATASET="${GOTOGOAL_DATASET:-$HOME/VLCM/datasets/manyagent_gotogoal_10m_3000}"
gpu_log="$PROJECT_ROOT/logs/gpu_utilization_${SLURM_JOB_ID:-local}.csv"
mkdir -p "$PROJECT_ROOT/logs"
"$PYTHON_BIN" -c 'import os, torch; n=int(os.environ["NUM_GPUS"]); print("Requested workers:",n,"Visible GPUs:",torch.cuda.device_count()); assert torch.cuda.device_count()==n, "Match NUM_GPUS to the allocated GPU count"'
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv --loop=10 > "$gpu_log" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
echo "GPU utilization samples: $gpu_log"
# Keep this shell so its EXIT trap stops the monitor after training.
bash "$PROJECT_ROOT/train_35_experiments.sh" "$@"
