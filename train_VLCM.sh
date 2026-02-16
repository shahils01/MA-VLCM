#!/bin/bash

# Exit on error
set -e

# Data Directory
DATA_DIR="/home/anshul/Research/Postdoc/RL/MA-VLCM/test_data" 

# If Data Already extracted
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Directory $DATA_DIR not found!"
    echo "Current directory content:"
    ls -l
    exit 1
fi
echo "Using existing data in $DATA_DIR"


# RUN Training
echo "Run Training"


# Run with Singularity
