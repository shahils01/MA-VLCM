#!/usr/bin/env bash
#SBATCH --job-name=vlcm_head_videos
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/anshuln/MA_VLCM/MA-VLCM}"
cd "$PROJECT_ROOT"
module load anaconda3
set +u
source activate "${CONDA_ENV:-internvl}"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export VLCM_FFMPEG="${VLCM_FFMPEG:-$HOME/.conda/envs/ma-vlcm/bin/ffmpeg}"
exec python "$PROJECT_ROOT/utils/animate_joint_heads.py" \
  --root "$PROJECT_ROOT/outputs/joint_3456_two_21847848" \
  --output "$PROJECT_ROOT/outputs/VLCM_shared_independent_heads/videos" \
  --per-group 2 --fps 10
