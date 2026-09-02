#!/bin/bash
#SBATCH --job-name=tb3_vlcm_infer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=a100:1

# Submit the three-backbone, full-episode inference launcher through Slurm.
# Environment overrides such as NUM_EPISODES, and NO_VIDEO
# are inherited by sbatch; positional arguments are forwarded unchanged.

set -euo pipefail

if [ -n "${MA_VLCM_ROOT:-}" ]; then
    REPO_ROOT=$(cd "$MA_VLCM_ROOT" && pwd)
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_tb3_episode_inference_all.sh" ]; then
    REPO_ROOT=$(cd "$SLURM_SUBMIT_DIR" && pwd)
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

cd "$REPO_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

echo "Repository: $REPO_ROOT"
echo "Slurm job: ${SLURM_JOB_ID:-not-set}"
echo "Node list: ${SLURM_JOB_NODELIST:-not-set}"

bash scripts/run_tb3_episode_inference_all.sh "$@"
