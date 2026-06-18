#!/bin/bash
# Run the AERO-MARL TurtleBot3 policy and a live MA-VLCM critic monitor together.

set -euo pipefail

MA_VLCM_ROOT="${MA_VLCM_ROOT:-/home/adi2440/Desktop/MARL_Shahil_Aditya/MA-VLCM}"
TURTLEBOT_WS="${TURTLEBOT_WS:-/home/adi2440/turtlebot_ws}"
AERO_MARL_ROOT="${AERO_MARL_ROOT:-/home/adi2440/Desktop/MARL_Shahil_Aditya/AERO-MARL}"
MODEL_DIR="${MODEL_DIR:-$TURTLEBOT_WS/models/transformer_800.pt}"
CHECKPOINT="${1:-${CHECKPOINT:-$MA_VLCM_ROOT/checkpoints/NewFinal_0.5B.pt}}"
OUTPUT_CSV="${OUTPUT_CSV:-$MA_VLCM_ROOT/outputs/results/tb3_live_predictions.csv}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
WINDOW_SIZE="${WINDOW_SIZE:-16}"
INFERENCE_RATE_HZ="${INFERENCE_RATE_HZ:-1.0}"
POLICY_START_DELAY_S="${POLICY_START_DELAY_S:-5}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ma_vlcm_matplotlib}"

export ROS_DOMAIN_ID
export MPLCONFIGDIR
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: MA-VLCM checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$MODEL_DIR" ]; then
    echo "ERROR: AERO-MARL policy checkpoint not found: $MODEL_DIR"
    exit 1
fi

source /opt/ros/humble/setup.bash
if [ -f "$TURTLEBOT_WS/install/setup.bash" ]; then
    source "$TURTLEBOT_WS/install/setup.bash"
fi

export PYTHONPATH="$MA_VLCM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$MPLCONFIGDIR"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
}
trap cleanup EXIT INT TERM

echo "Starting MARL policy launch"
ros2 launch cv_localization cv_rl_direct.launch.py \
  model_dir:="$MODEL_DIR" \
  aero_marl_root:="$AERO_MARL_ROOT" &
pids+=("$!")

sleep "$POLICY_START_DELAY_S"

echo "Starting MA-VLCM live inference from $CHECKPOINT"
cd "$MA_VLCM_ROOT"
"$PYTHON_BIN" -m ma_vlcm.tb3_live_inference \
  --checkpoint "$CHECKPOINT" \
  --window_size "$WINDOW_SIZE" \
  --inference_rate_hz "$INFERENCE_RATE_HZ" \
  --output_csv "$OUTPUT_CSV" &
pids+=("$!")

echo "Starting MA-VLCM live plot monitor"
"$PYTHON_BIN" -m ma_vlcm.tb3_live_monitor \
  --prediction_topic /fleet_vlcm/vlcm_prediction &
pids+=("$!")

wait -n "${pids[@]}"
