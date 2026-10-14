#!/bin/bash
# ALWAYS use BOTH HF datasets, including when training relabeled local shards.
export TASK_DOMAINS=goal_to_goal,static_obstacles
export BALANCE_TB3_SOURCES=1
export TB3_BALANCE_MODE=domain
export TB3_REQUIRE_BOTH_DOMAINS=1
TB3_CONTRACT_SOURCE="${1:-${DATA_DIR:-${TB3_TRAIN_SOURCES:-}}}"
PYTHONPATH="$SCRIPT_DIR/../src${PYTHONPATH:+:$PYTHONPATH}" python3 - "$TB3_CONTRACT_SOURCE" <<'PY'
import sys
from ma_vlcm.tb3_training_contract import require_both_tb3_sources
require_both_tb3_sources(sys.argv[1])
print('ALWAYS training with BOTH tb3-isaac-vlcm (go-to-goal) and tb3-isaac-avoid-obstacles-vlcm (obstacle avoidance).')
PY
