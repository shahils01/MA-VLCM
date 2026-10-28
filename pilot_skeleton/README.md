# Image-language value pilot

This folder contains the separate Iteration 1 Qwen image-language value model,
episode adapter, and training entry point. It is independent of the legacy
`train.py` pipeline. See [PILOT_README.md](../PILOT_README.md) for the target,
data contract, and experiment sequence.
# Iteration 1: image-language value pilot

The pilot reads each episode `.tar` directly from a local collection directory or
from the Hugging Face dataset repo. It uses the overhead images and task
instruction as input, and learns to predict a discounted reward return at the
last frame in each clip. Inputs are sampled at 5 Hz. When available, targets
use the 20 Hz `control_rewards.json` stream. Older archives can train using
sampled frame rewards as a discounted proxy return; those labels do not
reconstruct the full 20 Hz return, so evaluation measures fit to the proxy.
Targets run through true episode termination. When collection images are at
10 Hz, the adapter combines two image intervals while selecting every other
image. Robot observations and adjacency are loaded for the next iteration but
are not passed to the Iteration 1 model.

Install the MA-VLCM requirements, then run from the workspace root:

```bash
python MA-VLCM/pilot_skeleton/test.py
python MA-VLCM/pilot_skeleton/train_stage1_pseudocode.py \
  --dataset shulnak09/formation-vlcm-polygon-3to6 \
  --dataset-revision main --epochs 10
```

Use a local episode directory as the dataset argument to avoid HF downloads.
Pin `--dataset-revision` and `--model-revision` to commit IDs for a recorded
experiment. The run stores cached frozen Qwen features, the best temporal/value
head, episode splits, training MSE, and held-out predictions under
`pilot_runs/iteration1`.

The first run downloads the Qwen2-VL 2B checkpoint and computes one feature per
image. Later runs reuse the feature cache. Only the temporal encoder and scalar
value head are trained.
