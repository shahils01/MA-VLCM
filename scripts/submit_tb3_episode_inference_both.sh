#!/bin/bash
#SBATCH --job-name=tb3_vlcm_infer_both
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gpus=a100:1

# Evaluate five held-out episodes from each Isaac Lab TB3 dataset.  Each
# invocation produces the evaluator's CSVs, PNG plots, and MP4 videos in a
# dataset-specific subdirectory, preventing artifact collisions.

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
mkdir -p logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

NUM_EPISODES="${NUM_EPISODES:-5}"
NUM_FAILED_EPISODES="${NUM_FAILED_EPISODES:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/plots/tb3_episode_inference}"
GOAL_DATASET="${GOAL_DATASET:-hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar}"
OBSTACLE_DATASET="${OBSTACLE_DATASET:-hf://datasets/adi2440/tb3-isaac-avoid-obstacles-vlcm/**/*.tar}"
GOAL_OUTPUT_DIR="${GOAL_OUTPUT_DIR:-$OUTPUT_ROOT/tb3-isaac-vlcm}"
OBSTACLE_OUTPUT_DIR="${OBSTACLE_OUTPUT_DIR:-$OUTPUT_ROOT/tb3-isaac-avoid-obstacles-vlcm}"

run_dataset_evaluation() {
    local label="$1"
    local dataset="$2"
    local output_dir="$3"
    shift 3

    echo "============================================================"
    echo "Evaluating ${label}"
    echo "Dataset: ${dataset}"
    echo "Episodes: ${NUM_EPISODES}"
    echo "Failed episodes: ${NUM_FAILED_EPISODES}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    DATASET="$dataset" OUTPUT_DIR="$output_dir" NUM_EPISODES="$NUM_EPISODES" \
        NUM_FAILED_EPISODES="$NUM_FAILED_EPISODES" \
        bash scripts/run_tb3_episode_inference_all.sh "$@"
}

run_dataset_evaluation "goal-to-goal dataset" "$GOAL_DATASET" "$GOAL_OUTPUT_DIR" "$@"
run_dataset_evaluation "obstacle-avoidance dataset" "$OBSTACLE_DATASET" "$OBSTACLE_OUTPUT_DIR" "$@"

echo "Completed both dataset evaluations."
echo "Goal-to-goal artifacts: $GOAL_OUTPUT_DIR"
echo "Obstacle-avoidance artifacts: $OBSTACLE_OUTPUT_DIR"
