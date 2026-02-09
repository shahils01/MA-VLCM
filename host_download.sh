#!/bin/bash
# Script to download model directly on the HOST machine (bypassing Apptainer)

# 1. Determine Scratch Directory
if [ -n "$SCRATCH" ]; then
    BASE_DIR="$SCRATCH"
elif [ -d "/scratch/$USER" ]; then
    BASE_DIR="/scratch/$USER"
else
    echo "Using PWD as scratch (Fallback)"
    BASE_DIR="$PWD"
fi

WORK_DIR="$BASE_DIR/vlcm_download_env"
echo "Working directory: $WORK_DIR"
mkdir -p "$WORK_DIR"

# 2. Setup Python Virtual Environment in Scratch
# Try loading a newer python module if available (optimistic)
module load python/3.11 2>/dev/null || module load python/3.9 2>/dev/null || module load python/3.8 2>/dev/null || module load anaconda3/2022.05 2>/dev/null || echo "Module load failed or not available, using system python"

ENV_PATH="$WORK_DIR/venv"
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment at $ENV_PATH..."
    python3 -m venv "$ENV_PATH"
fi

# 3. Activate and Install Dependencies
source "$ENV_PATH/bin/activate"
echo "Installing dependencies..."
pip install --upgrade pip
# Python 3.6 needs dataclasses backport
pip install dataclasses
pip install huggingface_hub

# 4. Run the Download Script
# Ensure HF_HOME is set to scratch
export HF_HOME="$BASE_DIR/hf_cache"
mkdir -p "$HF_HOME"

echo "Starting download to $HF_HOME..."
echo "Running download_model.py..."

# Assuming download_model.py is in the current directory or MA-VLCM/
SCRIPT_PATH="download_model.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="MA-VLCM/download_model.py"
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Could not find download_model.py"
    exit 1
fi

python3 "$SCRIPT_PATH"

echo "Done. If successful, the model works."
