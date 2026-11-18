#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --array-index ]]; then
  worker_index="${2:?Expected array index 0, 1 or 2}"
  case "$worker_index" in
    0) cases=(navigation_without_obstacles) ;;
    1) cases=(formation_without_obstacles) ;;
    2) cases=(formation_physical_obstacles) ;;
    *) echo 'Expected array index 0, 1 or 2' >&2; exit 2 ;;
  esac
else
  # Retain compatibility with previously submitted two-worker batch scripts.
  worker_index="${1:?Expected --array-index INDEX}"
  case "$worker_index" in
    0) cases=(navigation_without_obstacles formation_physical_obstacles) ;;
    1) cases=(formation_without_obstacles) ;;
    *) echo 'Expected --array-index INDEX' >&2; exit 2 ;;
  esac
fi
"$PYTHON_BIN" - "$RUN_ROOT" "$worker_index" <<'PY'
import json,os,socket,sys
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit('Each srun worker must see exactly one allocated CUDA GPU')
record=dict(host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
    array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.executable)
Path(sys.argv[1],f'worker_{sys.argv[2]}_hardware.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
status=0
for task in "${cases[@]}"; do
  printf 'Starting %s on worker index %s\n' "$task" "$worker_index"
  if "$PYTHON_BIN" "$MARL_ROOT/train_frozen_joint_policies.py" \
    --case "$task" --output "$RUN_ROOT/$task" --critic "$VLCM_CHECKPOINT" \
    --steps "$NUM_ENV_STEPS" --num-envs "$NUM_ENVS" --seed "$SEED" \
    --device cuda:0 --vision-batch "$VISION_BATCH" \
    --advantage-estimator gae --rollout-steps "$ROLLOUT_STEPS" --gae-lambda 0.95 \
    --ppo-epochs 5 --minibatches 1 --actor-lr 0.0005 \
    --entropy-coef 0.0001 --max-grad-norm 10 --target-kl 0 \
    --wandb-mode "$WANDB_MODE" >"$RUN_ROOT/$task.log" 2>&1; then
    code=0
    printf 'Completed %s\n' "$task"
  else
    code=$?;status="$code"
    printf 'Failed %s (exit %s); see %s/%s.log. Other array tasks are independent.\n' "$task" "$code" "$RUN_ROOT" "$task" >&2
  fi
  printf '%s\n' "$code" >"$RUN_ROOT/$task.exitcode"
done
exit "$status"
