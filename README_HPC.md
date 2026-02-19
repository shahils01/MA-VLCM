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

Create a file named `submit_train.sh`:

```bash
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
```

Submit the job:
```bash
sbatch --mail-type BEGIN,END,FAIL submit_train.sh
```

### Training Script Arguments (`run_train_vlcm.sh`)

The `run_train_vlcm.sh` script launches the training with several key arguments. Here is what they mean:

| Argument | Meaning |
| :--- | :--- |
| `--train_shards` | Path to the directory or glob pattern containing WebDataset `.tar` shards. |
| `--dataset_type` | Set to `rware` to use specific multi-agent robot dataset logic. |
| `--rware_config` | Label used for the run (e.g., `mixed-rware`). |
| `--batch_size` | Number of samples per GPU per step. |
| `--grad_accum_steps` | Number of steps to accumulate gradients before updating weights (increases effective batch size). |
| `--clip_len` | Number of video frames in each training sample. |
| `--num_robots` | Maximum number of robots expected in any sample (used for observation padding). |
| `--robot_obs_dim` | The dimension of the low-level observation vector for each robot (default `6`). |
| `--epochs` | Total number of training passes over the dataset. |
| `--vl_backend` | The VLM architecture to use (e.g., `llava_video`). |
| `--vl_model_name` | The specific Hugging Face model ID for the VLM backbone. |
| `--save_dir` | Directory where checkpoints and logs will be saved. |
| `--num_workers` | Number of CPU workers for the data loader. |
| `--mixed_precision` | Use `bf16` or `fp16` to speed up training and reduce memory usage on modern GPUs. |
| `--peft` | Parameter-Efficient Fine-Tuning method. `qlora` is used to fit large models on standard GPU memory. |
| `--lora_r` / `--lora_alpha` | Rank and scaling factors for LoRA adapters. |
| `--vl_max_text_len` | Maximum token length for the text prompt/instructions. |

## 4. Troubleshooting

- **CUDA Errors**: Ensure the `--nv` flag is used with `apptainer exec/run`.
- **Import Errors**: If you added new dependencies to `requirements.txt` but they are missing in the container, you need to rebuild the container or install them at runtime (less efficient). To install at runtime:
  ```bash
  apptainer exec --nv ma_vlcm.sif pip install <package>
  ```
  (This installs to a temporary location unless you use `--user` and bind your home dir).
