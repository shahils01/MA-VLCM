#!/bin/bash

# Exit on error
set -e

echo "Starting Job on $(hostname)"
echo "Date: $(date)"

# 2. Data Setup
echo "Checking data..."

DATA_DIR="/scratch/aparame/Research/VLCM_Data_Collection/data_scratch"

# User stated data is already extracted in data_scratch
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Directory $DATA_DIR not found!"
    echo "Current directory content:"
    ls -l
    exit 1
fi
echo "Using existing data in $DATA_DIR"

# 3. Code Setup
# train.py, model.py, gat.py, requirements.txt are in PWD

# 4. Run Training
echo "Running Training..."

# Logic to find the RWARE config folder.
# Structure: data_scratch/{config_name}/{date_optional}/*.tar
# We look for the first directory inside data_scratch that contains "rware"
# or just the first sub directory.

# Find the first subdirectory in data_scratch
CONFIG_DIR=$(find $DATA_DIR -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -z "$CONFIG_DIR" ]; then
    echo "ERROR: No config directory found in $DATA_DIR"
    find $DATA_DIR
    exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_DIR")
echo "Detected RWARE Config: $CONFIG_NAME"

# Now find where the .tar files are. They might be in CONFIG_DIR or a subdirectory (date).
# We search recursively for .tar files inside CONFIG_DIR
BLOCK_DIR=$(find "$CONFIG_DIR" -name "*.tar" | head -n 1 | xargs dirname)

if [ -z "$BLOCK_DIR" ]; then
    echo "ERROR: No .tar files found inside $CONFIG_DIR"
    exit 1
fi

echo "Found data shards in: $BLOCK_DIR"

# Construct glob pattern for train.py
# If BLOCK_DIR is something like "data_scratch/rware-tiny-2ag/2026-02-09"
# Then pattern is "data_scratch/rware-tiny-2ag/2026-02-09/*.tar"
SHARD_PATTERN="$BLOCK_DIR/*.tar"

echo "Using Shard Pattern: $SHARD_PATTERN"

# Define container path
CONTAINER_PATH="/home/aparame/Research/MA-VLCM/ma_vlcm.sif"

# Run with Singularity
# We intentionally mount the current directory ($PWD) to ensure train.py and data are accessible inside.
# Handle Hugging Face Token (Environment variable or file)

# Determine optimal cache/tmp location (Prefer /scratch as it has 5TB capacity)
if [ -n "$SCRATCH" ]; then
    BASE_SCRATCH="$SCRATCH"
elif [ -d "/scratch/$USER" ]; then
    BASE_SCRATCH="/scratch/$USER"
else
    # Fallback to current directory if no scratch is found (likely will fail if quota is low)
    BASE_SCRATCH="$PWD"
    echo "WARNING: using $PWD for scratch space. This may fail if quota is low."
fi

# Extract number of robots from config name (e.g. rware-tiny-2ag-hard -> 2)
if [[ "$CONFIG_NAME" =~ ([0-9]+)ag ]]; then
    NUM_ROBOTS="${BASH_REMATCH[1]}"
else
    NUM_ROBOTS=2
    echo "WARNING: Could not parse num_robots from $CONFIG_NAME. Defaulting to 2."
fi
echo "Detected Num Robots: $NUM_ROBOTS"


accelerate launch --num_processes 2 train.py  \
  --train_shards "$SHARD_PATTERN" \
  --dataset_type rware \
  --rware_config "$CONFIG_NAME" \
  --batch_size 2 \
  --clip_len 2 \
  --num_robots "$NUM_ROBOTS" \
  --robot_obs_dim 6 \
  --epochs 2 \
  --vl_backend llava_video \
  --vl_model_name llava-hf/LLaVA-NeXT-Video-7B-32K-hf \
  --save_dir checkpoints_rware \
  --num_workers 1