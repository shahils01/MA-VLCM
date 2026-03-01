#!/bin/bash
#SBATCH --job-name=ma_vlcm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gpus=h100:2

# Ensure logs directory exists
mkdir -p logs

# CPU threading: one OMP thread per physical core avoids oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Unbuffered output so tqdm progress bars appear in real-time
export PYTHONUNBUFFERED=1

# Run the training script
bash run_train_vlcm.sh local /scratch/aparame/Research/VLCM_checkpoints/0.5B_LoRA_20260228_161429_epoch_3.pt
