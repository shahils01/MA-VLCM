#!/usr/bin/env bash
set -euo pipefail
case "${1:?Expected worker 0 or 1}" in
  0) cases=(navigation_without_obstacles formation_physical_obstacles) ;;
  1) cases=(formation_without_obstacles) ;;
  *) echo 'Expected worker 0 or 1' >&2; exit 2 ;;
esac
"$PYTHON_BIN" - "$RUN_ROOT" "$1" <<'PY'
import json,os,socket,sys
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit('Each srun worker must see exactly one allocated CUDA GPU')
record=dict(host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.executable)
Path(sys.argv[1],f'worker_{sys.argv[2]}_hardware.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
status=0
for task in "${cases[@]}"; do
  printf 'Starting %s on worker %s\n' "$task" "$1"
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
    code=$?;status=1
    printf 'Failed %s (exit %s); see %s/%s.log. Continuing independent cases.\n' "$task" "$code" "$RUN_ROOT" "$task" >&2
  fi
  printf '%s\n' "$code" >"$RUN_ROOT/$task.exitcode"
done
exit "$status"
