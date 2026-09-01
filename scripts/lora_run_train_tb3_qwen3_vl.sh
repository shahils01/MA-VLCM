#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DATASET_PROFILE="${DATASET_PROFILE:-tb3_isaac}"
export BACKBONE_PROFILE=qwen3_vl
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export TB3_DATASET_TYPE="${TB3_DATASET_TYPE:-tb3_progress_v2}"
export TB3_TARGET_SCHEMA="${TB3_TARGET_SCHEMA:-tb3_progress_v2}"
export ROBOT_OBS_DIM="${ROBOT_OBS_DIM:-9}"
export SUCCESS_ONLY="${SUCCESS_ONLY:-0}"
export TRAINED_AGENT_COUNTS="${TRAINED_AGENT_COUNTS:-3,4,5,6}"
export TASK_DOMAINS="${TASK_DOMAINS:-goal_to_goal,static_obstacles}"
export LAYOUT_SPLIT="${LAYOUT_SPLIT:-mixed_goal_and_obstacle_policies}"
export BALANCE_TB3_SOURCES="${BALANCE_TB3_SOURCES:-1}"
export PROGRESS_DISTANCE_MODE="${PROGRESS_DISTANCE_MODE:-route_if_available}"
export TB3_BALANCE_MODE="${TB3_BALANCE_MODE:-domain}"
export TB3_IMAGE_MODE="${TB3_IMAGE_MODE:-center_square}"
export TB3_IMAGE_SIZE="${TB3_IMAGE_SIZE:-144}"
export QWEN_VIDEO_FPS="${QWEN_VIDEO_FPS:-5.0}"
export TB3_TRAIN_SOURCES="${TB3_TRAIN_SOURCES:-hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar;hf://datasets/adi2440/tb3-isaac-avoid-obstacles-vlcm/**/*.tar}"
export WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-tb3_combined_qwen3_vl_2b}"
export TEMPORAL_CONSISTENCY_WEIGHT="${TEMPORAL_CONSISTENCY_WEIGHT:-0.05}"

exec bash "$SCRIPT_DIR/lora_run_train_tb3_lab.sh" "$@"
