# Simplified Critic Model Implementation

## Overview
Added a `--simplified_critic` flag to the training script to bypass the entire language model (and textual descriptions) and directly reason over visual features plus graph-structured robot observations.

## Changes Made
- **`model.py`:**
  - Implemented the `SimplifiedCriticModel` which encodes video frames locally with a `CLIPVisionModel` (`openai/clip-vit-base-patch32`), processes them through a `TemporalTransformer`, and concatenates the resulting temporal token with the Graph Attention Network (`GNN_Model`) robot-team state representation. 
  - Fed the concatenated embedding into a linear `value_head` initialized to 50.0.
  - Adapted the forward pass to return the appropriate single-depth `vlm_feature` if contrastive losses are requested.

- **`train.py`:**
  - Added the `--simplified_critic` CLI flag.
  - Made `build_model` conditionally return `SimplifiedCriticModel` if the flag is provided (defaulting to the CLIP patch32 base model).
  - Modified model parameter grouping logic correctly fallback to `vision_encoder` for when `freeze_vision_tower` is used on the specialized critic.
  - Adapted `SequenceWebDataset` logic. We now use `AutoImageProcessor` to prepare `(Batch*Time, C, H, W)` tensors if purely vision-based (if `text` input isn't present when `simplified_critic` is running). 

## Validation
- Successfully compiled and ran a local dry-run script (`test_simplified_critic.py`) sending synthetic robot observations and images formatting accurately simulating `DataLoader` structure.
- Loss targets and value predictions passed the sanity check gracefully, with dimensions perfectly aligning (`vlm_feature` produced `[B, d_model]`, value `[B]`).


- Training with ```ma_vlcm.sif``` container like:


```bash
 apptainer exec --nv ma_vlcm.sif python3 train.py --simplified_critic --train_shards "/home/adi2440/Desktop/MARL_Shahil_Aditya/VLCM_Data_Collection/RWARE/data_test_rware" --offroad_shards "/home/adi2440/Desktop/MARL_Shahil_Aditya/VLCM_Data_Collection/OFFROAD/Varied_Traversability_IID" --epochs 2
```