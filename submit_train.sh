#!/bin/bash
#SBATCH --job-name=ma_vlcm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --gpus=h100:2

# Ensure logs directory exists
mkdir -p logs

# Set OMP_NUM_THREADS to match cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the training script
bash run_train_vlcm.sh