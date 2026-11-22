# Training MA-VLCM on Kubernetes / NRP Nautilus

This guide starts with your datasets already downloaded from Hugging Face to a
PVC and this repository cloned on that PVC. It covers creating a persistent
Conda environment, launching training, following logs, W&B, saving results,
resuming, and downloading checkpoints to your computer. Commands marked
**local terminal** run on your computer, where `kubectl` is configured.
Commands embedded in Job YAML run inside the container automatically.

The examples use the joint frozen-backbone formation/navigation critic in
[train_multitask_frozen.sh](train_multitask_frozen.sh). They do not train the
MAPPO actor. See [README_MULTITASK.md](README_MULTITASK.md) for the experiment.
For this joint trainer you also need the **formation-trained critic `best.pt`**
from the same formation collection; downloaded dataset actor checkpoints and
Hugging Face SmolVLM weights do not replace that initialization checkpoint.

## 1. Know which files persist

Your container image supplies Linux, CUDA userspace libraries, and a base Conda
installation. The host supplies the NVIDIA driver. `/workspace` is the mounted
PVC and survives a training pod exiting or being deleted. Files written only
to the container's `/tmp`, `/root`, or `/opt/conda` disappear with that container.

Use this layout, changing paths if your PVC differs:

```text
/workspace/
  code/MA-VLCM/                          # Anshul_branch checkout
  datasets/polygon_formation_python_env/ # formation archives + summary sidecars
  datasets/manyagent_gotogoal_python_2000/
  envs/vlcm/                            # persistent Conda environment
  hf-cache/                             # checkpoint-pinned SmolVLM download
  vlcm-cache/                           # dataset indexes + frozen features
  checkpoints/formation-best.pt          # original supervised formation critic
  checkpoints/multitask-<pod-name>/       # each joint training run
  logs/                                 # W&B files, dependency records, exports
```

| Setting | Value used here |
| --- | --- |
| Kubernetes context | `nautilus` |
| Namespace | `anayak` |
| PVC | `vlm-storage` (50 GiB, ReadWriteMany) |
| Mount point in every pod | `/workspace` |
| Workspace pod for browsing/copying | `vlm-workspace` |
| GPU resource | `nvidia.com/a100` |
| HF secret / key | `hf-token` / `HF_TOKEN` |
| W&B secret / key | `wandb-secret` / `WANDB_API_KEY` |

**Local terminal:**

```bash
kubectl config current-context
kubectl --context nautilus -n anayak get pvc,pods,jobs
kubectl --context nautilus -n anayak get resourcequota
kubectl --context nautilus -n anayak exec vlm-workspace -- df -h /workspace
kubectl --context nautilus -n anayak exec vlm-workspace -- ls /workspace/code/MA-VLCM
kubectl --context nautilus -n anayak exec vlm-workspace -- ls -lh /workspace/checkpoints
```

The GPU quota is permission to request GPUs, not an allocation of idle hardware.
Four GPUs in one pod must fit on one node. A pod can remain Pending while it
waits. Use the namespace's allowed default priority unless NRP explicitly
provides a different class for your allocation.

## 2. Create a Conda environment on the PVC once

Create the environment **by prefix**, `/workspace/envs/vlcm`. A plain
`conda create -n vlcm` would normally place it under the container's base Conda
directory, which is not the PVC. Mount the PVC at `/workspace` in every future
pod: an installed Conda environment should not be moved to a new prefix.
See [Conda's prefix environment documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#specifying-a-location-for-an-environment).

Let the setup Job create `/workspace/envs/vlcm`; do not pre-create that final
directory or save YAML files inside it. Keep the manifest in your local
`Nautilus` folder or `/tmp`. If the Job reports that the prefix already exists,
inspect it first. For a verified-empty directory, `rmdir` can remove just that
directory before retrying; it refuses to remove a directory containing files.

The setup Job below uses the same Linux/CUDA image as training, requests **no
GPU**, and creates a fresh Python 3.11 environment. It installs CUDA-enabled
PyTorch even though the setup pod has no GPU. CUDA availability is checked
later in the GPU pod. The PyTorch/torchvision pairing and CUDA wheel index
follow the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2-5-1).

**Local terminal:** paste this entire block. It writes a manifest locally; it
does not modify your training source.

```bash
cat > /tmp/vlcm-env-setup.yaml <<'YAML'
apiVersion: batch/v1
kind: Job
metadata:
  name: vlcm-env-setup
  namespace: anayak
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/arch: amd64
      containers:
        - name: setup
          image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
          workingDir: /workspace/code/MA-VLCM
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              ENV_PREFIX=/workspace/envs/vlcm
              export CONDA_PKGS_DIRS=/tmp/conda-pkgs
              mkdir -p /workspace/envs /workspace/logs
              df -h /workspace
              if [ -e "$ENV_PREFIX" ]; then
                echo "Environment path already exists. Inspect it or choose a new prefix; no overwrite." >&2
                exit 1
              fi
              /opt/conda/bin/conda create -y --prefix "$ENV_PREFIX" \
                --override-channels -c conda-forge python=3.11 pip
              "$ENV_PREFIX/bin/python" -m pip install --no-cache-dir \
                torch==2.5.1 torchvision==0.20.1 \
                --index-url https://download.pytorch.org/whl/cu124
              cat > /tmp/vlcm-constraints.txt <<'CONSTRAINTS'
              torch==2.5.1
              torchvision==0.20.1
              transformers==4.51.3
              accelerate==1.10.1
              peft==0.15.2
              numpy==1.26.4
              CONSTRAINTS
              "$ENV_PREFIX/bin/python" -m pip install --no-cache-dir \
                -c /tmp/vlcm-constraints.txt -r requirements.txt
              "$ENV_PREFIX/bin/python" -m pip check
              "$ENV_PREFIX/bin/python" -c 'import torch, torchvision, accelerate, peft, wandb; from transformers import AutoProcessor, Idefics3Model; from train_multitask import main; print("Environment imports verified; CUDA build:", torch.version.cuda)'
              "$ENV_PREFIX/bin/python" -m pip freeze > /workspace/logs/vlcm-pip-freeze.txt
              /opt/conda/bin/conda list --prefix "$ENV_PREFIX" --explicit \
                > /workspace/logs/vlcm-conda-explicit.txt
              touch "$ENV_PREFIX/.ready"
              du -sh "$ENV_PREFIX"
              echo "Persistent training environment ready."
          resources:
            requests:
              cpu: "2"
              memory: 8Gi
              ephemeral-storage: 10Gi
            limits:
              cpu: "4"
              memory: 8Gi
              ephemeral-storage: 16Gi
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: vlm-storage
YAML

kubectl --context nautilus -n anayak create -f /tmp/vlcm-env-setup.yaml
kubectl --context nautilus -n anayak logs -f job/vlcm-env-setup --pod-running-timeout=5m
kubectl --context nautilus -n anayak get job vlcm-env-setup
```

Proceed when the Job shows `Complete` and logs say the environment is ready.
Installation is done once, not in each GPU job. The environment, CUDA wheels,
model download, and feature cache all need additional space beyond the datasets;
check free space before installing. Conda's temporary package cache above is
kept in the setup container's `/tmp` to avoid duplicating it on the 50 GiB PVC.
No local CUDA toolkit installation or `nvcc` is needed for this workload.

If setup fails after creating the prefix, inspect its logs and use a new prefix
such as `/workspace/envs/vlcm-v2` for a clean retry. Change the setup Job name
and all later environment paths accordingly. Do not install into a shared
environment while another job is training with it.

### Activation and choosing the interpreter

Inside a compatible **Bash-based** pod with this PVC mounted, activation is:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /workspace/envs/vlcm
which python
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Activation only affects that shell. Kubernetes starts a new shell for every
container; your local Conda activation does not carry over. Training jobs below
set both `PATH` and `PYTHON_BIN` to the PVC environment instead. Another option
inside the image is `/opt/conda/bin/conda run --no-capture-output --prefix
/workspace/envs/vlcm python ...`; see [conda run](https://docs.conda.io/projects/conda/en/stable/commands/run.html).

The existing `vlm-workspace` pod uses Alpine. Use it to browse or transfer files;
do **not** use it to test this glibc-based Python environment. A file can exist
there and still report “not found” because Alpine uses a different C runtime.
Use the PyTorch image for setup, training, and Python diagnostics.

## 3. Run the four-A100 training Job

The following Job reuses `/workspace/envs/vlcm`; it does not reinstall packages.
It downloads the SmolVLM revision saved in the formation checkpoint, then
prepares frozen image features across all four GPUs using the existing data
and model functions. That preparation does not train weights or fit normalizers.
The existing DDP training launcher runs after every cache worker succeeds.

This separate cache phase matters: the trainer's own cold-cache path runs
extraction on rank zero while other ranks wait, which can leave GPUs idle and
hit a distributed timeout during a long first extraction. Validation still
runs on rank zero, so four GPUs do not imply a fourfold speedup.

**Local terminal:** paste the full manifest block below.

```bash
cat > /tmp/vlcm-joint-conda-4a100.yaml <<'YAML'
apiVersion: batch/v1
kind: Job
metadata:
  name: vlcm-joint-conda-4a100
  namespace: anayak
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 172800
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/arch: amd64
      containers:
        - name: trainer
          image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
          workingDir: /workspace/code/MA-VLCM
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export PATH="/workspace/envs/vlcm/bin:$PATH"
              export PYTHON_BIN=/workspace/envs/vlcm/bin/python
              test -f /workspace/envs/vlcm/.ready
              test -s "$INIT_CHECKPOINT"
              test -f train_multitask_frozen.sh
              mkdir -p "$CACHE_DIR/features" "$WANDB_DIR"
              export RUN_DIR="/workspace/checkpoints/multitask-${HOSTNAME}"
              export WANDB_RUN_NAME="vlcm-joint-${HOSTNAME}"
              nvidia-smi

              python -c 'import torch, accelerate, wandb; from transformers import Idefics3Model; print("Using persistent Conda environment")'
              python -m pip freeze > "$WANDB_DIR/dependencies-${HOSTNAME}.txt"

              # Download the checkpoint-pinned model once before the workers start.
              python -u - <<'PY'
              import os
              import torch
              from huggingface_hub import snapshot_download
              checkpoint = torch.load(os.environ["INIT_CHECKPOINT"], map_location="cpu", weights_only=True)
              config = checkpoint["model_config"]
              assert config["backbone_mode"] == "frozen"
              assert checkpoint["run"]["args"]["return_source"] == "control"
              assert checkpoint["run"]["args"]["gamma"] == .99
              assert torch.cuda.device_count() == 4
              snapshot_download(config["model_id"], revision=config["model_revision"],
                                ignore_patterns=["*.bin", "*.h5", "*.msgpack", "onnx/*"])
              PY

              # Use all four GPUs to prepare exactly the existing frozen features.
              # Running this before DDP avoids a long rank-0-only cache barrier.
              pids=()
              for device in 0 1 2 3; do
                python -u - "$device" <<'PY' &
              import os
              import sys
              from pathlib import Path
              import torch
              from data_utils import discover_episodes, read_episode, prepare_episode, cache_features, feature_cache_path
              from model import ModelConfig, SmolVLMBackbone

              rank = int(sys.argv[1])
              torch.set_num_threads(2)
              torch.cuda.set_device(rank)
              device = torch.device("cuda", rank)
              checkpoint = torch.load(os.environ["INIT_CHECKPOINT"], map_location="cpu", weights_only=True)
              cfg = ModelConfig(**checkpoint["model_config"])
              refs = []
              for task, variable, count in (("formation_control", "FORMATION_DATASET", 3000),
                                             ("goal_to_goal", "GOTOGOAL_DATASET", 2000)):
                  task_refs = discover_episodes(os.environ[variable])
                  if len(task_refs) != count:
                      raise ValueError(f"Expected {count} {task} archives, found {len(task_refs)}")
                  refs.extend((task, path, summary) for path, summary in task_refs)
              cache = Path(os.environ["CACHE_DIR"])
              encoder = None
              for task, path, summary in refs[rank::4]:
                  ep = read_episode(path, summary, cache / "index")
                  if ep["task_domain"] != task:
                      raise ValueError(f"Wrong task in {path}")
                  if not ep["complete"]:
                      continue
                  ep = prepare_episode(ep, gamma=.99, source="control", image_hz=10.)
                  if feature_cache_path(ep, cfg, cache / "features").exists():
                      continue
                  if encoder is None:
                      encoder = SmolVLMBackbone(cfg, device)
                  cache_features([ep], encoder, cfg, cache / "features")
              print(f"GPU {rank}: feature cache ready", flush=True)
              PY
                pids+=("$!")
              done
              status=0
              for pid in "${pids[@]}"; do
                wait "$pid" || status=1
              done
              if (( status != 0 )); then
                echo "Feature preparation failed; training was not started." >&2
                exit 1
              fi

              # Existing launcher: one shared critic, four DDP ranks, W&B enabled.
              exec bash train_multitask_frozen.sh
          resources:
            requests:
              cpu: "8"
              memory: 32Gi
              ephemeral-storage: 8Gi
              nvidia.com/a100: "4"
            limits:
              cpu: "8"
              memory: 32Gi
              ephemeral-storage: 16Gi
              nvidia.com/a100: "4"
          env:
            - name: FORMATION_DATASET
              value: /workspace/datasets/polygon_formation_python_env
            - name: GOTOGOAL_DATASET
              value: /workspace/datasets/manyagent_gotogoal_python_2000
            - name: INIT_CHECKPOINT
              value: /workspace/checkpoints/formation-best.pt
            - name: CACHE_DIR
              value: /workspace/vlcm-cache
            - name: HF_HOME
              value: /workspace/hf-cache
            - name: NUM_GPUS
              value: "4"
            # 16 per rank gives effective batch 64, matching the one-GPU default.
            - name: BATCH_SIZE
              value: "16"
            - name: NUM_WORKERS
              value: "0"
            - name: OMP_NUM_THREADS
              value: "2"
            - name: TOKENIZERS_PARALLELISM
              value: "false"
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: WANDB_ENABLED
              value: "1"
            - name: WANDB_MODE
              value: online
            - name: WANDB_PROJECT
              value: nautilus-vlm-training
            - name: WANDB_DIR
              value: /workspace/logs
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: HF_TOKEN
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secret
                  key: WANDB_API_KEY
          volumeMounts:
            - name: workspace
              mountPath: /workspace
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: vlm-storage
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 4Gi
YAML

kubectl --context nautilus -n anayak create -f /tmp/vlcm-joint-conda-4a100.yaml
```

## 4. Follow training and diagnose startup

**Local terminal:**

```bash
JOB=vlcm-joint-conda-4a100
kubectl --context nautilus -n anayak get pods -l job-name="$JOB" -o wide
kubectl --context nautilus -n anayak logs -f "job/$JOB" --pod-running-timeout=5m
```

`Ctrl+C` stops following logs; it does not stop training. Save a snapshot locally:

```bash
mkdir -p ./nautilus-results
kubectl --context nautilus -n anayak logs "job/$JOB" > "./nautilus-results/${JOB}.log"
POD=$(kubectl --context nautilus -n anayak get pods -l job-name="$JOB" -o jsonpath='{.items[0].metadata.name}')
kubectl --context nautilus -n anayak describe pod "$POD"
kubectl --context nautilus -n anayak get events \
  --field-selector "involvedObject.name=$POD" --sort-by=.metadata.creationTimestamp
```

When the container is Running, inspect the actual GPUs and interpreter:

```bash
kubectl --context nautilus -n anayak exec "$POD" -- nvidia-smi
kubectl --context nautilus -n anayak exec "$POD" -- \
  /workspace/envs/vlcm/bin/python -c 'import sys,torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

Expect four devices and `cuda.is_available() == True` inside this GPU job.
Startup proceeds through model download, parallel feature preparation, dataset
index/statistics loading, initial validation, then training epochs. A long
cache stage is not an epoch. Later runs reuse compatible caches automatically.

| Symptom | What to check |
| --- | --- |
| `Pending` | `describe pod`: insufficient GPUs/CPU/memory, quota, or node taints. |
| `ContainerCreating` | Image pull and PVC mount events. |
| `CreateContainerConfigError` | Secret names and keys; PVC and namespace spelling. |
| Environment “not found” | Prefix is on PVC, mounted at `/workspace`, and image uses glibc, not Alpine. |
| `ModuleNotFoundError` | `sys.executable`; `.ready` marker; setup Job logs. |
| Dataset count mismatch | Correct dataset roots and archive/summary sidecars; defaults expect 3,000 / 2,000. |
| Initialization/split mismatch | Correct supervised formation `best.pt` and its original formation collection. |
| `OOMKilled` | Host RAM limit, not necessarily GPU VRAM; inspect pod termination reason. |
| CUDA out of memory | Reduce training batch; cache extraction microbatch is pinned by checkpoint config. |
| No epoch or W&B output yet | Check download/cache progress first. |
| `Error` / failed Job | Read logs before deleting it; `backoffLimit: 0` prevents automatic repeats. |

## 5. W&B authentication, metrics, and local records

The Job reads `WANDB_API_KEY` from `wandb-secret`, with online mode and project
`nautilus-vlm-training`. Existing secrets need no further login. Check their
existence without printing credentials:

```bash
kubectl --context nautilus -n anayak get secret wandb-secret hf-token
```

If the W&B secret does not exist, create it from a prompted local value. This
keeps the literal key out of your command history and repository:

```bash
read -r -s -p 'W&B API key: ' vlcm_wandb_key
echo
printf '%s' "$vlcm_wandb_key" | kubectl --context nautilus -n anayak create secret generic wandb-secret \
  --from-file=WANDB_API_KEY=/dev/stdin
unset vlcm_wandb_key
```

Do not run this creation command if that secret already exists. Configure the
analogous `hf-token` secret separately if needed for gated/private downloads.
Keep credentials in Secrets, not `value:` fields in committed YAML.

The training launch gives the run a name based on its pod name. If you use a
team account, add `WANDB_ENTITY` with the team slug to the YAML environment.
The trainer initializes W&B **after feature preparation**, and prints the run
URL then. Look for `wandb` in the log. Model download/cache preparation before
that point will not appear as a training run in W&B.

Current joint metrics include:

- `train/normalized_mse`
- `validation/macro_task_normalized_mse`
- `validation/formation_control/{mse,mae,r2}`
- `validation/goal_to_goal/{mse,mae,r2}`

Local W&B files live under `/workspace/logs/wandb`. Checkpoints remain separate
under `/workspace/checkpoints`; **the trainer does not automatically upload
`best.pt` or `latest.pt` as W&B artifacts**. An online W&B dashboard is not a
backup of those files.

For offline logging, change `WANDB_MODE` to `offline` before submitting. Later,
from a compatible pod with the environment, PVC, network, and W&B secret:

```bash
/workspace/envs/vlcm/bin/wandb sync /workspace/logs/wandb/offline-run-REPLACE_WITH_RUN_DIRECTORY
```

Do this via a finite CPU Job if the GPU training pod has already exited. A new
Job created for checkpoint resume starts a new W&B run unless you separately
configure a supported W&B run-ID/resume policy; model resume alone does not
guarantee continuation of the same dashboard history.

## 6. Checkpoints, reports, and space

The Job sets `RUN_DIR=/workspace/checkpoints/multitask-${HOSTNAME}`. For example,
pod `vlcm-joint-conda-4a100-abcde` writes to
`/workspace/checkpoints/multitask-vlcm-joint-conda-4a100-abcde`.

| File | Meaning |
| --- | --- |
| `best.pt` | Lowest macro task normalized validation MSE; may retain the initial model at epoch zero. |
| `latest.pt` | Last completed training epoch, suitable for epoch-boundary resume. |
| `epoch_002.pt`, etc. | Periodic snapshots; every two epochs by default. |
| `run.json` | Arguments, initialization, data splits, model config, task statistics. |
| `metrics.jsonl` | Training and validation results per epoch. |
| `before_validation_metrics.json` | Initial critic validation. |
| `best_validation_metrics.json` | Selected critic validation after training. |
| `before_test_metrics.json`, `after_test_metrics.json` | Initial/selected model held-out tests after training. |
| `task_comparison.json` | Per-task before/after test comparison. |
| Prediction CSV files | Saved evaluation predictions and native return targets. |

`latest.pt` is written after an epoch completes, not every optimizer step. A
pod lost during an epoch loses that unfinished epoch's progress. The checkpoint
contains model configuration/state and normalizers plus optimizer/scheduler
state and experiment metadata. Frozen SmolVLM weights are referenced by model
ID/revision rather than embedded in the small downstream critic checkpoint.
For later inference, retain the pinned model cache or allow HF download again.

List saved runs and space usage through the workspace pod, even after training:

```bash
kubectl --context nautilus -n anayak exec vlm-workspace -- ls -lh /workspace/checkpoints
kubectl --context nautilus -n anayak exec vlm-workspace -- df -h /workspace
kubectl --context nautilus -n anayak exec vlm-workspace -- \
  du -sh /workspace/envs /workspace/hf-cache /workspace/vlcm-cache /workspace/checkpoints /workspace/logs
```

## 7. Retrieve a checkpoint or complete run locally

Run these commands on **your computer**, not inside a pod. Use the still-running
workspace pod because `kubectl cp`/`exec` cannot enter a completed training
container. Both pods access the same PVC at `/workspace`. `kubectl cp` requires
`tar` in the transfer container; see [Kubernetes copy documentation](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_cp/).

Choose the real directory from the listing above:

```bash
REMOTE_RUN=/workspace/checkpoints/multitask-vlcm-joint-conda-4a100-REPLACE_WITH_POD_SUFFIX
LOCAL_RUN=./nautilus-results/joint-run
mkdir -p "$LOCAL_RUN"

# Selected critic for inference/evaluation.
kubectl --context nautilus -n anayak cp \
  "vlm-workspace:${REMOTE_RUN}/best.pt" "$LOCAL_RUN/best.pt"

# Copy the entire run for metrics, provenance, and resumable checkpoints.
kubectl --context nautilus -n anayak cp \
  "vlm-workspace:${REMOTE_RUN}" "$LOCAL_RUN/full-run"

# Compare these two checksums for the selected checkpoint.
kubectl --context nautilus -n anayak exec vlm-workspace -- sha256sum "$REMOTE_RUN/best.pt"
sha256sum "$LOCAL_RUN/best.pt"

# Preserve the environment recipe alongside the results.
kubectl --context nautilus -n anayak cp \
  vlm-workspace:/workspace/logs/vlcm-pip-freeze.txt "$LOCAL_RUN/vlcm-pip-freeze.txt"
kubectl --context nautilus -n anayak cp \
  vlm-workspace:/workspace/logs/vlcm-conda-explicit.txt "$LOCAL_RUN/vlcm-conda-explicit.txt"
```

For a single binary checkpoint, an alternative avoids remote `tar`:

```bash
kubectl --context nautilus -n anayak exec vlm-workspace -- \
  cat "$REMOTE_RUN/best.pt" > "$LOCAL_RUN/best.pt"
```

Do not add `-t` to binary transfers; a terminal can alter the stream. Copy
`best.pt` after the run finishes for a stable selection; `latest.pt` is intended
for continuation. To upload a local initialization checkpoint, reverse the copy:

```bash
kubectl --context nautilus -n anayak cp ./best.pt \
  vlm-workspace:/workspace/checkpoints/formation-best.pt
```

If `vlm-workspace` is absent, use another **Running** pod mounting `vlm-storage`
at `/workspace`, or recreate your CPU workspace/transfer pod. The PVC remains
independent of the lifetime of the completed training container.

## 8. Resume after interruption or start a different experiment

For a new experiment, submit a new Job name and use a new output directory.
For continuation, use a new Kubernetes Job name but keep the **original**
`RUN_DIR`. Edit a local copy of the training manifest:

1. Change `metadata.name` to a new name, for example `vlcm-joint-conda-resume1`.
2. Replace the automatic `RUN_DIR` assignment with the full original PVC path.
3. Keep the original initialization checkpoint, datasets, split seed, gamma,
   frame settings, and selected training tasks.
4. Replace the final training command with the following, choosing a total
   epoch limit greater than the epoch recorded in `latest.pt`:

```bash
export RUN_DIR=/workspace/checkpoints/multitask-REPLACE_WITH_ORIGINAL_POD_NAME
export EPOCHS=30
exec bash train_multitask_frozen.sh --resume "$RUN_DIR/latest.pt"
```

`EPOCHS=30` means stop at total epoch 30, not run 30 additional epochs. The
trainer checks experiment compatibility and requires the original output path.
Its resume is epoch-boundary continuation; it is not a claim of bit-for-bit
reproduction of an uninterrupted run's RNG trajectory.

Do not use `--resume` to switch datasets. To initialize a different joint run
from a supported formation checkpoint, set `INIT_CHECKPOINT` and a new
`RUN_DIR`, without `--resume`.

## 9. Stop a Job and keep its data

Save the logs before deleting the Job. Deleting the Job stops its pod and
releases GPUs; it does not delete the separately created PVC:

```bash
kubectl --context nautilus -n anayak logs "job/$JOB" > "./nautilus-results/${JOB}.log"
kubectl --context nautilus -n anayak delete job "$JOB"
```

Do not delete `vlm-storage` to stop training. Do not update dependencies or
change the checkout while a job is using them. Kubernetes Job pod templates
are immutable; use a new Job name for changed launch settings. NRP batch jobs
should execute finite computation and exit when done, rather than hold GPUs
with an indefinite sleep: [NRP batch-job guidance](https://nrp.ai/documentation/userdocs/running/jobs/).

## Daily workflow

1. Check PVC space, dataset paths, and the initialization checkpoint.
2. Create the PVC Conda environment once; wait for setup to complete.
3. Submit a uniquely named GPU Job using that environment.
4. Follow Kubernetes logs through download/cache preparation and epochs.
5. Follow the W&B URL once tracking initializes.
6. Retrieve the selected checkpoint, full run metadata, and environment records.
7. Resume from `latest.pt` if necessary, or delete the finished Job to clean up.
