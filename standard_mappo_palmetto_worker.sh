#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Compute allocation required' >&2; exit 2; }
cases=(navigation_without_obstacles formation_without_obstacles formation_physical_obstacles)
index="${SLURM_ARRAY_TASK_ID:?Expected array task 0, 1, or 2}"
case "$index" in 0|1|2) ;; *) exit 2 ;; esac
task="${cases[$index]}"
"$PYTHON_BIN" - "$MARL_ROOT" <<'PY'
import sys,torch
sys.path.insert(0,sys.argv[1])
from train_standard_mappo import make_actor,make_critic,World,central_state,verify_sources
verify_sources()
assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Exactly one allocated GPU is required'
assert 'A100' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
for case in ('navigation_without_obstacles','formation_without_obstacles','formation_physical_obstacles'):
    world=World(case,3);obs=world.reset(42)
    actor=make_actor(obs.shape[-1]);critic,norm=make_critic(central_state(world,obs).size,torch.device('cuda:0'))
    assert any(p.requires_grad for p in critic.parameters())
    print('PASS',case,'obs',obs.shape,'centralized state',central_state(world,obs).shape,flush=True)
print('PASS compute-node sources, fresh native actor, trainable MLP critic and allocated A100; no VLCM loaded',flush=True)
PY
if [[ "${1:-train}" == --check ]]; then exit 0; fi
mkdir -p "$RUN_ROOT"
mkdir "$RUN_ROOT/$task"
"$PYTHON_BIN" - "$RUN_ROOT/$task/hardware.json" <<'PY'
import json,os,socket,sys,torch
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
    array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.executable),indent=2)+'\n')
PY
# Hardware record is outside the fresh trainer output directory.
mv "$RUN_ROOT/$task/hardware.json" "$RUN_ROOT/worker_${index}_hardware.json"
if "$PYTHON_BIN" "$MARL_ROOT/train_standard_mappo.py" --case "$task" --output "$RUN_ROOT/$task" \
    --steps "$NUM_ENV_STEPS" --num-envs 16 --rollout-steps 300 --horizon 400 --seed 1 \
    --ppo-epochs 5 --minibatches 1 --actor-lr .0005 --critic-lr .0005 \
    --entropy-coef .0001 --max-grad-norm 10 --gae-lambda .95 --device cuda:0 \
    --wandb-mode "$WANDB_MODE" >"$RUN_ROOT/$task.log" 2>&1; then
  echo 0 >"$RUN_ROOT/$task.exitcode"
else
  code=$?; echo "$code" >"$RUN_ROOT/$task.exitcode"; exit "$code"
fi
"$PYTHON_BIN" "$MARL_ROOT/compare_frozen_standard_mappo.py" \
  --standard "$RUN_ROOT" --frozen "$FROZEN_POLICIES" --case "$task" \
  --output "$RUN_ROOT/comparison/$task" --samples 100 \
  >"$RUN_ROOT/${task}_comparison.log" 2>&1
