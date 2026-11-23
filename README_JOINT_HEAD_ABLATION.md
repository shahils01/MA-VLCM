# Joint formation/navigation critic: one head versus K heads

Train on `polygon_formation_python_env` and a many-agent go-to-goal collection,
using 3, 4, 5 and 6 robots. Both experiments use the pretrained SmolVLM-500M
image/text encoder, frozen and cached. The graph, fusion, temporal transformer
and value layers train from scratch. This is supervised critic training;
the actor is not trained in this experiment.

| Setting | `shared` | `task` |
|---|---|---|
| Representation | Shared visual + graph + temporal layers | Same |
| Final scalar heads | One | K (two for these datasets) |
| Return normalization | Per task | Per task |
| Model output by default | Native reward units | Native reward units |
| Data splits and sampling | Same seed; equal task and team-size probabilities | Same |
| Formation hold input | Recorded hold fraction and settled fraction | Same |

The final linear layer has shape `(1, d)` or `(K, d)`. Its rows are independent
scalar affine heads; there is no separate full network for each task. K is
`len(cfg.task_names)`, saved in the checkpoint. The CLI currently exposes the
two existing task loaders. Adding a new dataset for another task also requires
its observation/reward adapter and task name; adding a source is not automatic.

## The math in the code

For task k, compute the mean `mu[k]` and standard deviation `sigma[k]` from
nonterminal training return targets only. Keep them fixed throughout the run:

```python
target_z = (return_target - mu[task_id]) / sigma[task_id]
pred_z = shared_head(history)              # ablation A
pred_z = task_heads(history)[task_id]       # ablation B
loss = ((pred_z - target_z) ** 2).mean()
native_value = sigma[task_id] * pred_z + mu[task_id]
```

`model(batch, return_normalized=True)` supplies the normalized prediction for
the loss. `model(batch)` returns native reward units. Task identity is required
for both experiments because both use per-task normalization; `collate_clips`
supplies it automatically. Save/reload uses `task_names`, `task_target_mean`
and `task_target_std`. Use native predictions with native rewards for PPO/GAE.

The shared backbone sees image instructions for the task in both experiments.
Only the selected task head contributes prediction error. The K heads are
evaluated together and gathered by task, including in distributed training.

## Control rates: read the data, preserve physical time

The formation collection has 20 Hz controls and 10 Hz images. The existing
`manyagent_gotogoal_python_2000` manifest records **10 Hz controls and 10 Hz
images**, not 20 Hz. A newer collection may differ. The run records actual
control/image rates and effective discount factors per task in `run.json`.

The launcher defines gamma=.99 at a reference rate of 20 Hz:

```python
control_gamma = gamma ** (20 / recorded_control_hz)
G = sum(control_gamma ** j * future_control_rewards[j] for j in ...)
```

Thus formation uses .99 per .05-second step; 10 Hz navigation uses .9801 per
.1-second step. Both discount the same amount over the same elapsed time.
All native rewards remain in their recorded units. Normalization cannot make
reward definitions or policy distributions identical across tasks.

To require both collections to have genuine 20 Hz controls, set
`EXPECTED_CONTROL_HZ=20`. The run fails on a mismatch. Do not change metadata,
duplicate frames, split rewards or invent intermediate transitions. Matching
20 Hz control data must be collected in a 20 Hz environment, with its rewards
and policy behavior checked under that timestep. Existing mixed-rate data is
usable with the common discount timescale above.

Targets use every `control_rewards.json` reward, excluding incoming rewards
already received at the current image. Success after the 20-step formation
hold is a true terminal; its future return is zero. External timeout episodes
are excluded from this Monte Carlo regression because their continuation is
unobserved. Native terminal failures remain valid examples.

## Palmetto

From the repository root, using an environment satisfying `requirements.txt`:

```bash
mkdir -p logs
export FORMATION_DATASET=/home/anshuln/Dataset/polygon_formation_python_env
export GOTOGOAL_DATASET=/home/anshuln/Dataset/manyagent_gotogoal_python_2000
CONDA_ENV=internvl sbatch train_joint_ablation_palmetto.sbatch all
```

`all` runs `shared`, then `task`, reusing the same frozen image-feature cache.
Use `shared` or `task` to run one experiment. Defaults request two A100s,
16 CPUs and 96 GB RAM on one node. `torchrun` uses both allocated GPUs for
feature extraction and training; validation/test are also sharded across ranks.
With one GPU, override the allocation using `sbatch --gpus-per-node=a100:1 ...`.
`NUM_GPUS` can override process count explicitly.

The critic batch size defaults to 1024 **per GPU**; effective batch size is
`BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS`. The image-encoding batch size is
independent (`VL_BATCH_SIZE=16`). Increase either after measuring memory and
throughput; reduce `VL_BATCH_SIZE` if image encoding runs out of memory.
Changing effective batch size may require learning-rate tuning.

CPU workers decode upcoming images while each GPU encodes its current batch.
Feature extraction is divided by episode across ranks, with complete files
published atomically. During critic fitting, image features are loaded into
CPU RAM once; pinned memory, nonblocking transfers, persistent workers and
prefetching feed the GPU. By default, each rank gets
`max(0, allocated_CPUs / NUM_GPUS - 2)` workers. `NUM_WORKERS` overrides it.
Keep `NUM_GPUS * NUM_WORKERS` within the CPU allocation. More workers are not
always faster for this small, RAM-resident critic.

Useful overrides: `BATCH_SIZE`, `VL_BATCH_SIZE`, `NUM_WORKERS`, `EPOCHS`, `LR`,
`SEED`, `CACHE_DIR`, `RUN_ROOT`, `MODEL_REVISION`, `WANDB_ENABLED=0` and
`EXPECTED_CONTROL_HZ`. `RUN_DIR` selects a single experiment's output only.
Pin `MODEL_REVISION` to a model commit for reproducibility across different
submission dates. Both runs record the resolved model revision.

```bash
# Cheap selection check: summary sidecars only; no GPU/model download.
WANDB_ENABLED=0 bash train_joint_ablation.sh shared --preflight-only

# Resume one run. Keep its original model/data/head/normalization flags.
RUN_DIR=/path/to/original/task sbatch train_joint_ablation_palmetto.sbatch task \
  --resume /path/to/original/task/latest.pt --epochs 60
```

The full run checks rates and recorded hold metadata after indexing archives.
The cheap preflight only checks selection/counts. Logs include per-epoch
training clips/second; `logs/gpu-JOB_ID.csv` samples GPU use and memory every
30 seconds (`GPU_MONITOR=0` disables it). High GPU utilization cannot be
guaranteed: the frozen backbone is only evaluated during caching and the
remaining critic is small. Optimize measured examples/second, and compare
one versus two GPUs if communication dominates cached-feature training.

## Compare the experiments

Use validation macro normalized MSE (equal task weight) to choose checkpoints.
Then compare held-out test native-unit MAE/RMSE, R-squared, team-size and
success/failure results separately for each task. Both runs preserve whole
scenario splits, so overlapping clips never cross partitions. Run several
matched seeds before claiming that separate heads help.

Keep failures, interrupted holds and recovery trajectories in the training
distribution; report their errors separately instead of selecting only
successful demonstrations. The two causal formation features help distinguish
just-achieved geometry from a nearly completed hold. A later state-design
ablation can add per-robot settling latches and longer histories.

The value estimates describe the recorded behavior-policy mixture. They are
initial critic estimates for an evolving actor, not guaranteed on-policy
values. Standard saved-split evaluation, prediction and checkpoint animation
read the new normalization and discount settings. The legacy Python MAPPO
critic wrapper manually assembles its own forward pass; adapt its task routing
and context/normalization before deploying these new checkpoints through it.
