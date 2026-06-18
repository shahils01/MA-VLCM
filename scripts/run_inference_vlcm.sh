#!/bin/bash
# run_inference_vlcm.sh – Evaluate a trained MA-VLCM checkpoint on test data.
#
# Usage:
#   bash run_inference_vlcm.sh
#
# Edit CHECKPOINT and TEST_DATA_DIR below to match your setup.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "Starting Inference on $(hostname)"
echo "Date: $(date)"

# ── Paths (edit these) ──────────────────────────────────────────────────────
CHECKPOINT="/scratch/aparame/Research/VLCM_checkpoints/50StepReturn_ContrastiveMSE_20260304_025223_epoch_10.pt"
TEST_DATA_DIR="/scratch/aparame/Research/VLCM_Data_Collection/RWARE/data_test_iid"
OUTPUT_FILE="inference_results.csv"

# ── Environment ─────────────────────────────────────────────────────────────
export HF_TOKEN=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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Container path (set to empty string to run without Apptainer) ───────────
CONTAINER_PATH="$REPO_ROOT/ma_vlcm.sif"

if [ -n "$CONTAINER_PATH" ] && [ -f "$CONTAINER_PATH" ]; then
    # Determine scratch bind path
    if [ -n "$SCRATCH" ]; then
        BASE_SCRATCH="$SCRATCH"
    elif [ -d "/scratch/$USER" ]; then
        BASE_SCRATCH="/scratch/$USER"
    else
        BASE_SCRATCH="$REPO_ROOT"
    fi

    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$(dirname "$OUTPUT_FILE")" "$PLOT_DIR"
    echo "Running via Apptainer container: $CONTAINER_PATH"
    apptainer exec --nv -B "$REPO_ROOT:$REPO_ROOT" -B "$BASE_SCRATCH:$BASE_SCRATCH" \
      --env HF_TOKEN="$HF_TOKEN" \
      "$CONTAINER_PATH" python3 inference.py \
        --checkpoint "$CHECKPOINT" \
        --test_shards "$TEST_DATA_DIR" \
        --batch_size 4 \
        --num_workers 8 \
        --max_samples 200\
        --dataset_type rware\
        --plot_dir "0.5B_b_rware_iid"\
        --run_mc_dropout
else
    echo "Running natively (no container)"
    python3 inference.py \
        --checkpoint "$CHECKPOINT" \
        --test_shards "$TEST_DATA_DIR" \
        --batch_size 4 \
        --num_workers 8 \
        --output_file "$OUTPUT_FILE" \
        --plot_dir "$PLOT_DIR" \
        --max_samples 100 \
        --dataset_type rware
else
    echo "Running natively (no container)"
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$(dirname "$OUTPUT_FILE")" "$PLOT_DIR"
    python3 -m ma_vlcm.inference \
        --checkpoint "$CHECKPOINT" \
        --test_shards "$TEST_DATA_DIR" \
        --batch_size 4 \
        --num_workers 8 \
        --output_file "$OUTPUT_FILE" \
        --plot_dir "$PLOT_DIR" \
        --max_samples 100 \
        --baseline
fi

echo "Done. Results written to: $OUTPUT_FILE"
