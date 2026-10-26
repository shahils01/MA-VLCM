# MA-VLCM value-estimator pilot

This is an implementation plan for a small, staged pilot. Iteration 1 now has an episode-tar adapter, frozen Qwen frame encoder, temporal value head, and a standalone training loop in `pilot_skeleton/`. It remains separate from `train.py`, whose legacy input contract and model differ from this image-language baseline.

## Goal and backbone

Estimate team quality over time from a short sequence of overhead images and the task instruction. Begin with image and language only; add per-robot state and adjacency as graph inputs in Iteration 2.

Start from the pretrained Qwen/Qwen2-VL-2B-Instruct checkpoint. It is a 2B image/video-capable model; see the [official model card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_vl). Do not train the 2B parameters from random initialization. Start with Qwen frozen and train a temporal module plus scalar value head. If this baseline learns useful ordering, compare a separate LoRA run. Reduce sampled frames, image resolution, and batch size before changing the architecture; use accumulation/checkpointing or supported low-bit loading if needed. Any smaller replacement should first be checked on the overhead images, where robots may occupy few pixels.

## Iteration 1: image and language

For a value prediction anchored at time t, input only images up to t, I[t-L+1:t], the instruction l, and valid-frame mask m. Future frames may be used to compute the training return target, but must not enter the input if the estimator will be queried online or used as a critic at time t:

```text
z_k = PoolHidden(QwenVL(image_k, instruction))
h_1 ... h_L = TemporalEncoder(z_1 ... z_L, valid_frame_mask)
v_hat_t = ScalarHead(h_query_or_last)
```

The output should be a scalar prediction, not generated prose. Define its label before training. Preferred target, after validating reward timestamps and control-step aggregation, is a finite-horizon discounted team return:

```text
G_t = sum(j=0..H-1) gamma^j * r_(t+j)
      + gamma^H * bootstrap_(t+H) * V_target(x_(t+H))
bootstrap = 0 for true terminal; 1 for a time-limit truncation if the process continues
```

Do not copy a terminal +1/-1 label onto every frame. If current reward data cannot produce a trustworthy return, label the initial experiment explicitly as a continuous geometric-progress proxy, not a calibrated value function. Preserve boundary/safety outcome as a distinct label or metric so near-boundary formation progress is visible without hiding safety failure.

Begin with masked scalar regression. Optionally compare pairwise ranking for clips from comparable tasks and starting conditions. Split by whole episode (preferably collection seed/start condition) before generating overlapping clips. Evaluate held-out regression error, rank correlation, and calibration; report results separately by formation size and success, incomplete, recovery, and safety-failure groups. Plot predicted value and target over complete episodes.

Avoid treating a smooth rising curve as the definition of value. A learned value should rise when expected future return improves, but it can fall during a bad detour or rise during a recovery. As a cheap representation check, add an auxiliary frame-order/progress-ranking experiment inspired by [GVL (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/54854cf15a24fff9f5134a8641136fe4-Abstract-Conference.html): sample frames from one episode, shuffle their presentation order, and train pairwise ordering/progress supervision against recorded quality or return. GVL reports that directly asking a VLM for ordered frame values can collapse to uninformative monotonic output, while shuffled temporal ordering improves progress predictions. Treat this as a separate auxiliary/ablation because your target is reward/return, not merely demonstration time. Do not impose chronological monotonic ranking on failed or recovery episodes unless the label supports that ordering.

## Iteration 2: robot state and adjacency

Keep the image-language pathway and target fixed. Encode each robot with a shared node encoder, run masked graph attention over recorded adjacency, pool only active robots, fuse one graph vector with the aligned visual-language vector per frame, then use the same temporal encoder and value head:

```text
q_i,k = SharedRobotEncoder(robot_state_i,k)
g_i,k = MaskedGAT(q_1..q_N,k, adjacency_k, robot_mask_k)
g_k = MaskedTeamPool(g_1..g_N,k, robot_mask_k)
u_k = Fusion(z_k, g_k)
h_1..h_L = TemporalEncoder(u_1..u_L, valid_frame_mask)
v_hat_t = ScalarHead(h_query_or_last)
```

Check robot ordering, units, timestamp alignment, adjacency direction/self-loop convention, and inactive-agent masks. Mask absent robots in both attention and pooling. Reuse exactly the same episode splits and targets to compare the graph model fairly against Iteration 1. Add boundary clearance or velocity only when those fields exist consistently in the data schema.

## Data contract

The adapter should produce episode records, then clips that never cross episode boundaries. Every clip identifies episode ID, ordered frame IDs/timestamps, images, language instruction, valid-frame mask, target values, and target-valid mask. Iteration 2 adds robot state shaped [T,N,F], adjacency [T,N,N], and robot mask [T,N]. Preserve units and target-generation metadata.

Validate finite values, monotone timestamps, matching frame counts, image/state alignment, adjacency dimensions, and true-terminal versus time-limit semantics. Count and explain malformed samples; do not silently substitute random data. Keep a manifest with episode, split, outcome, formation size, source revision, and target version. Align every target to the image timestamp and actual reward/control timebase.

## Work sequence

1. Implement the adapter and inspect a few decoded clips by eye, including image, prompt, timestamp, and target alignment.
2. Overfit a tiny set as a pipeline check. Then train the frozen-Qwen temporal/scalar head with episode-balanced batches.
3. Compare with constant-mean and progress-only baselines on a fixed held-out episode split. Inspect curves for successes, incomplete attempts, recoveries, and failures.
4. If the frozen baseline learns useful ordering, run LoRA as a separate experiment on the same split.
5. Add graph/state fusion. Initially freeze Qwen and train the graph/fusion/value modules; compare on the same episodes. Consider LoRA only as another controlled run.

Save model and processor IDs/revisions, code revision, data manifest/hash, split IDs, target definition, and training configuration with each checkpoint. Track metrics by formation size and outcome; aggregate metrics can hide poor performance on larger teams.

## Pseudocode module map

- pilot_skeleton/config_pseudocode.py: model, data, target, and training settings.
- pilot_skeleton/data/episode_adapter_pseudocode.py: episode loading, validation, and clip contract.
- pilot_skeleton/models/image_language_value_pseudocode.py: Iteration 1 backbone, temporal features, scalar readout.
- pilot_skeleton/train_stage1_pseudocode.py: Iteration 1 loss, optimization, logging, and checkpoints.
- pilot_skeleton/models/graph_fusion_value_pseudocode.py: Iteration 2 robot encoder, masked GAT, and fusion.
- pilot_skeleton/train_stage2_pseudocode.py: controlled Iteration 2 training and comparison.
- pilot_skeleton/evaluate_pseudocode.py: episode metrics and value-curve inspection.
- pilot_skeleton/slurm/train_stage1_palmetto.sbatch: Palmetto 2 Slurm launch template.

Implement one module at a time, replacing TODO comments with working code. Keep the existing trainer independent until each new piece is validated. Leave MoE, complex contrastive objectives, joint PPO training, and end-to-end unfrozen 2B fine-tuning out of the first pilot.

## Recommended implementation order

Work through the modules in this order; each step should have a small local smoke run before moving on:

1. `data/episode_adapter_pseudocode.py`: read one local episode or one Hugging Face dataset example; validate fields and timestamps; create a non-overlapping example for inspection.
2. `config_pseudocode.py`: make model ID, image/frame budget, target definition, split manifest, and optimization settings explicit and serializable.
3. `models/image_language_value_pseudocode.py`: use Hugging Face Transformers/AutoProcessor with Qwen2-VL; extract one feature per timestamped image, then causally encode history up to the requested time and predict one scalar.
4. `train_stage1_pseudocode.py`: implement masked regression, checkpointing, metrics, resume, and the optional pairwise loss. Begin with the Qwen weights frozen.
5. `evaluate_pseudocode.py`: evaluate held-out episodes and create time plots before adapting the backbone.
6. `slurm/train_stage1_palmetto.sbatch`: configure the account/partition, GPU, Python environment, storage paths, HF cache/auth, and call the completed training entry point.

For the target, prefer stored dense reward aggregated to the image sampling interval and compute discounted returns with documented terminal/truncation masks. Keep a separate progress target/head for diagnostic experiments. If the reward cannot yet be aligned reliably, do not call the proxy a value estimate. To address weak temporal labels without complicating the first model, add pairwise ordering only after scalar regression works: compare two clips from the same task and comparable starting conditions, and apply a ranking loss to their predicted returns. As a later ablation, add a relative-progress head that predicts the signed change in progress between nearby frames. Do not force all successful trajectories to increase monotonically if they have legitimate recovery or oscillation segments.

## Relevant work and architectural fit

- [RL-VLM-F (ICML 2024)](https://arxiv.org/abs/2402.03681) learns rewards from VLM preferences over image pairs conditioned on a language goal. Its [official code](https://github.com/yufeiwang63/RL-VLM-F) is useful for pairwise-label and reward-model design, though its older environment/dependency stack is not a drop-in training pipeline for this project.
- [Generative Value Learning (GVL)](https://github.com/TOPReward/TOPReward) predicts task-completion percentages from images and is included in the public [TOPReward codebase](https://github.com/TOPReward/TOPReward). This is the closest public code example for visual progress/value supervision. TOPReward itself extracts instruction-token likelihoods as a zero-shot trajectory reward; it is a useful frozen-model baseline, not a substitute for the proposed learned scalar estimator.
- [VLM-AR3L (IJCAI 2026)](https://arxiv.org/abs/2607.00483) combines an absolute state score with relative comparisons of consecutive observations. Its [public implementation](https://github.com/1224K/VLM-AR3L) supports the proposed idea of a scalar level signal plus optional relative progress supervision. It relies on task-specific environments and VLM preference generation, so borrow the objective pattern rather than its full stack.
- [WCM (2026 preprint)](https://arxiv.org/abs/2607.29613) jointly estimates value and predicts future latent dynamics from observation history. Its [repository](https://github.com/sylvestf/WCM) is an advanced follow-up reference. It argues that history plus scalar regression may not teach dynamics robustly; consider a lightweight future-latent auxiliary loss only after the image-language baseline and labels are sound.
- [Offline RL-VLM-F (IROS 2025)](https://arxiv.org/abs/2411.05273) uses VLM preference feedback to label suboptimal offline robot demonstrations and then learns with offline RL. It is relevant to the mixed-quality trajectories here; its [project page](https://sites.google.com/andrew.cmu.edu/offline-rl-vlm-f) links the study materials.

The proposed Iteration 1 model aligns with these works at the representation level: a pretrained visual-language encoder receives images and instruction, temporal aggregation handles partial observability, and a small head outputs a scalar. It is not identical to their supervision: RL-VLM-F/AR3L learn preference-based rewards, GVL predicts progress, TOPReward uses model token likelihoods, and WCM adds future-latent prediction. For a compact, learnable first pipeline, keep scalar return/progress regression as the baseline, add pairwise ranking as the first comparison, then test relative-progress or future-latent objectives as isolated ablations. The multi-robot GAT belongs in Iteration 2, where it can address information missing from pixels, such as exact robot identity, adjacency, velocity, or boundary clearance.

## Hugging Face and Clemson Palmetto 2

The model is fetched through Hugging Face Transformers from the model ID in config; dataset access should be explicit and separate. The formation dataset upload preserves the collection folder layout and stores one `.tar` plus one `.summary.json` per episode; it is not a tabular `datasets.load_dataset()` export. The pilot adapter reads those archives directly from a local directory or downloads an individual archive from the HF dataset repo on demand. Pass the dataset repo ID and a revision (default `main`) or a local dataset directory. Keep episode-level train/validation/test splits separate from archive discovery. For private HF assets, authenticate on the cluster using Clemson-approved secret handling; never put an access token in this repository or a Slurm script.

The Slurm file is a template for Clemson Palmetto 2. Current RCD docs describe `--gpus` requests and the `work1` default for batch jobs; GPU models include H100, L40S, A100, and others. Check your allocation and available hardware with `sinfo`, then edit requested GPU/partition/time and paths to match your account. Build or install the Python environment before submitting the training job, and place Hugging Face cache/checkpoints on approved persistent or scratch storage with enough capacity. Create the log directory before `sbatch` because Slurm opens output files before the script starts. See [Clemson Slurm job submission](https://docs.rcd.clemson.edu/palmetto/job_management/submit/) and [Palmetto 2 scheduler guidance](https://docs.rcd.clemson.edu/palmetto/job_management/sched/).
