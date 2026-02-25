#!/bin/bash
# run_inference_vlcm.sh – Evaluate a trained MA-VLCM checkpoint on test data.
#
# Usage:
#   bash run_inference_vlcm.sh
#
# Edit CHECKPOINT and TEST_DATA_DIR below to match your setup.

set -e

echo "Starting Inference on $(hostname)"
echo "Date: $(date)"

# ── Paths (edit these) ──────────────────────────────────────────────────────
CHECKPOINT="/scratch/aparame/Research/VLCM_checkpoints/ckpt_epoch_3.pt"
TEST_DATA_DIR="/scratch/aparame/Research/VLCM_Data_Collection/data_test"
OUTPUT_FILE="inference_results.csv"

# ── Model / data config (should match training) ────────────────────────────
NUM_ROBOTS=4

# ── Environment ─────────────────────────────────────────────────────────────
export HF_TOKEN=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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Container path (set to empty string to run without Apptainer) ───────────
CONTAINER_PATH="$PWD/ma_vlcm.sif"

if [ -n "$CONTAINER_PATH" ] && [ -f "$CONTAINER_PATH" ]; then
    # Determine scratch bind path
    if [ -n "$SCRATCH" ]; then
        BASE_SCRATCH="$SCRATCH"
    elif [ -d "/scratch/$USER" ]; then
        BASE_SCRATCH="/scratch/$USER"
    else
        BASE_SCRATCH="$PWD"
    fi

    echo "Running via Apptainer container: $CONTAINER_PATH"
    apptainer exec --nv -B "$PWD:$PWD" -B "$BASE_SCRATCH:$BASE_SCRATCH" \
      --env HF_TOKEN="$HF_TOKEN" \
      "$CONTAINER_PATH" python3 inference.py \
        --checkpoint "$CHECKPOINT" \
        --test_shards "$TEST_DATA_DIR" \
        --batch_size 4 \
        --num_workers 8 \
        --output_file "$OUTPUT_FILE" \
        --max_samples 100\
        --baseline
else
    echo "Running natively (no container)"
    python3 inference.py \
        --checkpoint "$CHECKPOINT" \
        --test_shards "$TEST_DATA_DIR" \
        --batch_size 4 \
        --num_workers 8 \
        --output_file "$OUTPUT_FILE" \
        --max_samples 100 \
        --baseline
fi

echo "Done. Results written to: $OUTPUT_FILE"
