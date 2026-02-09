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
ENV_PATH="$WORK_DIR/venv"
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating virtual environment at $ENV_PATH..."
    python3 -m venv "$ENV_PATH"
fi

# 3. Activate and Install Dependencies
source "$ENV_PATH/bin/activate"
echo "Installing huggingface_hub..."
pip install --upgrade pip
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
