#!/usr/bin/env bash
#SBATCH --job-name=marl_frozen_joint
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32gb
#SBATCH --array=0-2%2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# One independent policy per array element; at most two elements run at once.
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
if [[ -z "${MARL_ROOT:-}" ]]; then
  export MARL_ROOT="$(dirname "$PROJECT_ROOT")/MARL_data_collection"
  if [[ ! -f "$MARL_ROOT/train_frozen_joint_policies.py" && -f "$HOME/MARL_data_collection/train_frozen_joint_policies.py" ]]; then
    export MARL_ROOT="$HOME/MARL_data_collection"
  fi
fi
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
    export VLCM_FORMATION_ROOT="$PROJECT_ROOT/outputs/frozen_joint_runtime_assets/formation"
    if [[ ! -f "$VLCM_FORMATION_ROOT/collection_plan.json" ]]; then
      export VLCM_FORMATION_ROOT="$DATASET_ROOT/polygon_formation_python_env"
    fi
  fi
fi
if [[ -z "${VLCM_NAVIGATION_ROOT:-}" ]]; then
  export VLCM_NAVIGATION_ROOT="$(dirname "$PROJECT_ROOT")/manyagent_gotogoal_10m_3000/provenance/clean/without_obstacles"
  if [[ ! -f "$VLCM_NAVIGATION_ROOT/collection_plan.json" ]]; then
    export VLCM_NAVIGATION_ROOT="$PROJECT_ROOT/outputs/frozen_joint_runtime_assets/navigation_without_obstacles"
    if [[ ! -f "$VLCM_NAVIGATION_ROOT/collection_plan.json" ]]; then
      export VLCM_NAVIGATION_ROOT="$DATASET_ROOT/manyagent_gotogoal_10m_3000/provenance/clean/without_obstacles"
    fi
  fi
fi
export RUN_ROOT="${RUN_ROOT:-$MARL_ROOT/outputs/frozen_joint_palmetto_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-preflight}}}"
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
case "${SLURM_ARRAY_TASK_ID:-}" in
  0|1|2) task_id="$SLURM_ARRAY_TASK_ID" ;;
  *) echo 'Expected Slurm array index 0, 1 or 2. Submit using sbatch --array=0-2%2.' >&2; exit 2 ;;
esac
cases=(navigation_without_obstacles formation_without_obstacles formation_physical_obstacles)
task="${cases[$task_id]}"
mkdir -p "$RUN_ROOT"
# Each element claims only its own output directory; siblings share RUN_ROOT.
mkdir "$RUN_ROOT/$task" || { echo "Use a fresh output directory: $RUN_ROOT/$task" >&2; exit 2; }
printf 'Array index: %s\nTask: %s\nOutputs: %s/%s\n' "$task_id" "$task" "$RUN_ROOT" "$task"
exec srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --gpus-per-task=a100:1 --gpu-bind=single:1 \
  bash "$PROJECT_ROOT/frozen_joint_palmetto_worker.sh" --array-index "$task_id" \
  >"$RUN_ROOT/worker_${task_id}.log" 2>&1
