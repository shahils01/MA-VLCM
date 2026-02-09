# Running MA-VLCM on HPC with Apptainer

This document describes how to build and run the MA-VLCM training container on an HPC cluster.

## 1. Prerequisites

- **Apptainer (formerly Singularity)** installed on your HPC user node.
- **Git** to clone the repository.
- **Sufficient Disk Space** (~10GB) for the container image and cache.

## 2. Building the Container

You need to build the container image (`ma_vlcm.sif`) from the definition file (`ma_vlcm.def`). This usually requires `fakeroot` privileges, which are often available on HPC login nodes via the `--fakeroot` flag.

```bash
# Navigate to the directory containing ma_vlcm.def
cd /path/to/MA-VLCM

# Build the container
apptainer build --fakeroot ma_vlcm.sif ma_vlcm.def
```

If you encounter issues with disk space in `/tmp`, set the `APPTAINER_TMPDIR` environment variable to a location with more space (e.g., your scratch directory):

```bash
mkdir -p $HOME/scratch/tmp
export APPTAINER_TMPDIR=$HOME/scratch/tmp
apptainer build --fakeroot ma_vlcm.sif ma_vlcm.def
```

## 3. Running Training

Once the `ma_vlcm.sif` file is built, you can run the training script using `apptainer run` or `apptainer exec`.

### Interactive Run (for debugging)

```bash
# Request an interactive GPU node (command varies by HPC, e.g., srun --pty ...)
# Then run:
apptainer exec --nv ma_vlcm.sif python3 train.py
```
*Note: The `--nv` flag is crucial for enabling GPU support inside the container.*

### SLURM Batch Script Example

Create a file named `run_train.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ma_vlcm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu  # Change to your cluster's GPU partition

# Load Apptainer module if required
# module load apptainer

# Set up environment
export MA_VLCM_ROOT=$(pwd)

# Run the training
# Note: Bind mount data directories if they are not in your home/cwd
# Example: --bind /scratch/user/data:/data
srun apptainer exec --nv ma_vlcm.sif python3 train.py \
    --train_shards "/path/to/your/data_scratch/" \
    --batch_size 8 \
    --epochs 10 \
    --save_dir checkpoints_hpc
```

Submit the job:
```bash
sbatch run_train.slurm
```

## 4. Troubleshooting

- **CUDA Errors**: Ensure the `--nv` flag is used with `apptainer exec/run`.
- **Import Errors**: If you added new dependencies to `requirements.txt` but they are missing in the container, you need to rebuild the container or install them at runtime (less efficient). To install at runtime:
  ```bash
  apptainer exec --nv ma_vlcm.sif pip install <package>
  ```
  (This installs to a temporary location unless you use `--user` and bind your home dir).
