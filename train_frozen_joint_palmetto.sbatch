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
check_only=0
case "${1:-}" in
  --check) check_only=1 ;;
  '') ;;
  *) echo 'Usage: bash train_frozen_joint_palmetto.sbatch --check, or sbatch train_frozen_joint_palmetto.sbatch' >&2; exit 2 ;;
esac
if [[ "$check_only" == 0 && -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit using sbatch, or use --check for CPU-only setup validation.' >&2; exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export VLCM_CODE_ROOT="$PROJECT_ROOT"
export MARL_ROOT="${MARL_ROOT:-$(dirname "$PROJECT_ROOT")/MARL_data_collection}"
if [[ -z "${VLCM_CHECKPOINT:-}" ]]; then
  # Cluster training outputs and workstation copies use different directory layouts.
  export VLCM_CHECKPOINT="$PROJECT_ROOT/outputs/critic35_sequential/joint_with_obstacles/best.pt"
  if [[ ! -f "$VLCM_CHECKPOINT" ]]; then
    export VLCM_CHECKPOINT="$PROJECT_ROOT/outputs/ckpts/critic35_sequential/joint_with_obstacles/best.pt"
  fi
fi
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
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/outputs/frozen_joint_palmetto_${SLURM_JOB_ID:-preflight}}"
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
printf 'Code: %s\nTrainer: %s\nCritic: %s\nFormation snapshot: %s\nNavigation snapshot: %s\n' \
  "$PROJECT_ROOT" "$MARL_ROOT" "$VLCM_CHECKPOINT" "$VLCM_FORMATION_ROOT" "$VLCM_NAVIGATION_ROOT"
missing=0
for required in "$VLCM_CHECKPOINT" "$MARL_ROOT/train_frozen_joint_policies.py" \
  "$PROJECT_ROOT/frozen_joint_palmetto_worker.sh" \
  "$VLCM_FORMATION_ROOT/collection_plan.json" "$VLCM_NAVIGATION_ROOT/collection_plan.json"; do
  if [[ ! -f "$required" ]]; then echo "Missing required file: $required" >&2; missing=1; fi
done
if ((missing)); then
  echo 'Set VLCM_CHECKPOINT, MARL_ROOT, VLCM_FORMATION_ROOT and VLCM_NAVIGATION_ROOT to the actual cluster paths.' >&2
  echo 'Git does not copy ignored checkpoints or source snapshots. Run --check again after correcting the paths.' >&2
  exit 2
fi
# Verify the frozen source snapshots before reserving separate GPU steps.
"$PYTHON_BIN" - "$MARL_ROOT" <<'PY'
import os,sys
from pathlib import Path
import torch
from transformers import AutoProcessor, Idefics3Model
sys.path.insert(0,sys.argv[1])
from train_frozen_joint_policies import verify_sources, native_actor_args, World, make_actor, CASES
from model import ModelConfig, MultimodalValueModel
verify_sources();native_actor_args()
checkpoint=torch.load(os.environ['VLCM_CHECKPOINT'],map_location='cpu',weights_only=True)
cfg=ModelConfig(**checkpoint['model_config'])
trained=checkpoint['run']['args']
if cfg.backbone_mode!='frozen' or cfg.clip_len!=6 or (trained['gamma'],trained['image_hz'],trained['return_source'])!=(.99,10.,'control'):
    raise SystemExit('Expected the frozen six-frame, 10 Hz, gamma .99 joint critic checkpoint')
model=MultimodalValueModel(cfg,load_backbone=False)
model.load_state_dict(checkpoint['model_state'],strict=True)
for case in CASES:
    world=World(case,3);obs=world.reset(42);make_actor(obs.shape[-1]);world.image()
    print(f'PASS {case}: actor observation {obs.shape}, environment and renderer initialized',flush=True)
if os.environ['WANDB_MODE']!='disabled':
    import wandb
print('PASS checkpoint tensors, native source hashes, dependencies and all three policy environments',flush=True)
print('Backbone cache/download, CUDA allocation and W&B authentication still require the compute-node check.',flush=True)
PY
if [[ "$check_only" == 1 ]]; then
  echo 'Setup checks passed. No training job was submitted.'
  exit 0
fi
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
