# Training the Python MAPPO actor with a VLCM critic

**Implementation update (2026-10-29):** The dedicated loop is now implemented
in `train_mappo.py` / `python_mappo.py`, with on-policy launchers, W&B,
MLP/state-temporal baselines and actor-only exports. Real frozen-SmolVLM GPU
PPO smoke runs and CPU baseline runs passed; converged learning and transfer
gains have not been measured. The [runnable guide](../README_PYTHON_MAPPO.md)
supersedes the proposal below, including normalization/checkpoint details.

## Recommendation

Keep the successful Python dynamics, team reward and 20 Hz actor. Train a centralized VLCM value model on Python rollout images and Python reward targets, then use its values in MAPPO's advantage calculation. Export the actor alone for Isaac deployment. The Python and Isaac renderers do not need identical appearance because the VLCM is used only during Python training. This audit originally described a proposal; see the implementation update above for current runnable code and checks.

The existing Isaac-trained `MA-VLCM/outputs/zero_margin_900/best.pt` can initialize the feature extractor and some learned layers, but its fitted return output is not calibrated to the Python objective. Adapt the head and recompute train-only observation/target statistics for Python data before trusting its values in PPO. Keep the SmolVLM backbone frozen for the first experiment and continue updating the graph/fusion/temporal/value layers from current-policy rollouts. Freezing SmolVLM does not mean freezing the whole critic.

## What was inspected

- `MARL_data_collection/RL_bound_overwatch/envs/FormationControlEnv.py`: canonical Python task.
- `MARL_data_collection/on-policy/onpolicy/scripts/train/train_formation_control.py` and the two polygon training shell entrypoints.
- `MARL_data_collection/on-policy/onpolicy/runner/shared/runner_formation_control.py`, shared base runner, PPO policy/trainer and replay buffer.
- Checkpoint `.../mappo_generic_polygon_3to6_seed1/polygon/step_008001600`, plus its metadata and nearby saved checkpoints.
- Python evaluation renderer, vector-environment reset behavior, and the Isaac formation reward/collector.
- Current MA-VLCM model, dataset sampler and `zero_margin_900` training configuration.

The Python environment and runner have existing uncommitted user changes, including settling behavior and mixed-command support. They were not edited during this investigation.

## Verified native baseline

The saved 8,001,600-step actor was evaluated without training on 15 new episodes per formation, seeds 8300001 through 8300015. Each formation uses the same seed list; these are paired initial scenarios between the two evaluation settings, not 120 independently seeded task streams.

Settings: polygon, 3/4/5/6 active robots padded to six; spacing 1.0 m; slot tolerance 0.05 m; observation divisor 3.0; starts in +/-1.5 m with 0.55 m separation; dt=0.05 s; 300-step limit; deterministic actor actions. The current task requires 20 consecutive all-robot in-tolerance steps. No native training or Isaac training was performed.

| Formation | Success / episodes | Mean raw team return | Mean control steps | Mean final max slot error (m) |
| --- | ---: | ---: | ---: | ---: |
| Triangle | 15/15 | 41.351 | 130.4 | 0.01938 |
| Square | 15/15 | 40.946 | 154.9 | 0.02920 |
| Pentagon | 15/15 | 36.035 | 188.0 | 0.02383 |
| Hexagon | 15/15 | 35.765 | 193.7 | 0.02789 |

A second, paired evaluation with hold=1 and settling reward weights zero also achieved 15/15 successes for each formation. This is a compatibility ablation, not a reconstruction of the exact historical training code. The old checkpoint metadata does not record settling settings or discount. Results are in `MA-VLCM/outputs/python_mappo_baseline/`; there are 120 evaluated episodes across the two settings. Some pentagon trajectories include collision steps, so 100% success should not be read as perfect safety. These small evaluations establish a working reference, not a generalization guarantee.

Reproduce from MA-VLCM using the environment that contains PyTorch and the on-policy dependencies:

```bash
/home/anshul/miniconda3/envs/env_isaaclab/bin/python run_python_mappo_baselines.py \
  --episodes 15 --seed 8300001 --include-instant-success \
  --output-dir outputs/python_mappo_baseline_reproduced
```

This evaluates the existing actor. It does not run a MAPPO training comparison.

## Actual compatibility differences

| Contract | Successful Python path | Existing VLCM / Isaac labels | Decision for the first direct training experiment |
| --- | --- | --- | --- |
| Actor | Shared MAPPO; per-robot 25-value observation, six padded slots | Collector loads the same actor contract | Preserve actor architecture and action distribution |
| Critic state | Shared 52-value state | Six-frame image/instruction + graph with 10 physical features per robot | Build a separate causal adapter; weights are not interchangeable |
| Control | dt=0.05 s, 20 Hz | 20 Hz control rewards, 5 Hz critic anchors | Preserve 20 Hz actor and GAE; train the new head at 20 Hz |
| Gamma | Python config defaults to 0.99; training shell does not override it | Checkpoint records 0.99 per control step | Use explicit 0.99 in every new experiment; historical checkpoint gamma is not saved |
| Reward | Raw progress/error/collision/effort/settling reward, success term +50 | Different bounded components, out-of-bounds penalty, divide by 20, clip to [-1,1] | Use Python rewards directly and learn Python values; no scalar conversion of Isaac values |
| Success | Default 0.05 m tolerance sustained for 20 steps | Stored collection 0.075 m and immediate success | Preserve Python criterion; adapt targets |
| Failure / timeout | No workspace-exit terminal; timeout is a truncation | Workspace exit is terminal; offline trainer excludes timeouts | Use separate terminated/truncated masks and proper timeout bootstrapping |
| Physical scene | Training metadata spacing 1.0, unbounded dynamics, 300-step shell limit | Collection spacing 0.5 and bounded workspace, 20-second limit | Preserve successful Python task for training; evaluate actor transfer separately |
| Images | Existing PIL renderer: colored robots, goal crosses, trails and grid | Isaac overhead checkerboard with black robots | Fit the training-only critic on truthful Python images/instructions |
| Normalization | PPO ValueNorm enabled by default | VLCM has internal training-target standardization and emits raw return units | Normalize exactly once at the PPO boundary, or disable PPO ValueNorm consistently for matched pilot baselines |

In particular, multiplying an Isaac value by 20 or 50 does not fix reward mismatch: clipping, settling penalties, boundary termination and success timing differ. Also, the previously mentioned gamma 0.995 belongs to another Isaac MAPPO YAML, not this Python training default.

## Resolving 5 Hz images and 20 Hz advantages

Five-Hz camera updates do not require five-Hz action updates. Store robot/graph state and the transition reward at every 20 Hz control step. Capture and encode a Python image every four control steps, and encode each image once while the backbone is frozen. At every control step t, query the inexpensive trainable head with a causal history.

To preserve the existing one-second history, use six state samples at indices t-20, t-16, t-12, t-8, t-4, t. For each state sample, attach the most recent camera feature at or before that sample. Across the four control phases the image can be 0, 0.05, 0.10 or 0.15 seconds old. Record that age and make it explicit to the adapted head, or validate a version trained over all four phases. Never use a later image or interpolate with future frames. Reset the history and capture an initial image when an episode resets. Train/evaluate this adapter on all 20 Hz anchors; the old 5 Hz head is not assumed correct at intermediate phases.

`MA-VLCM/vlcm_mappo_timing.py` implements the state/camera mapping and is now used by the dedicated `python_mappo.py` collector and learned head. Its four unit tests cover all control phases, original aligned histories, episode boundaries and stale/future-image prevention. The new collector measures 20 Hz states directly; the old offline archives are not interpolated to fabricate those measurements.

A simpler timing control is to render/encode every 20 Hz state and train at 20 Hz, with the same physical history length; it is slower and is useful for checking the multirate optimization. Lowering actor control to 5 Hz would change the successful control problem and requires its own matched baseline.

## PPO integration points

1. **Rollout collection:** retain actor observations, actual sampled actions and their log probabilities exactly as today. Store the privileged current robot state, frame references and episode identifiers. Preserve terminal observations before vector wrappers auto-reset. Both the subprocess worker and dummy vector wrapper currently return reset observations after done; inserting those as terminal history would cross episodes.
2. **Value inference after the rollout:** with one fixed critic snapshot, encode the unique captured images in batches, construct causal contexts for every transition plus the rollout endpoint, and compute raw team values. Broadcast the same team value to active robots; keep padded robots masked. Do not multiply duplicated team rewards by robot count or average over padded zeros.
3. **Advantages:** fill `buffer.value_preds` in the units expected by the buffer and compute GAE at 20 Hz with gamma=0.99 and the configured lambda=0.95. For true success terminal states bootstrap zero. For rollout cutoffs and true time-limit truncations use the value of the actual final observation, not the reset observation. A finite-horizon formulation with a time-left input is another experiment, but should not be mixed with truncation bootstrapping.
4. **Actor update:** use the existing PPO clipping, entropy, active masks and collected log probabilities. Advantages are detached. This is where the critic affects the actor; evaluating values without this update does not train the actor. The runner currently calls `compute()` then `train()`, which are the natural insertion points.
5. **Critic update:** train the graph/fusion/temporal/value head on the current rollout return/GAE targets using the same cached image features. Keep old rollout values fixed for value clipping. Ensure `evaluate_actions` and `cal_value_loss` do not accidentally call/train the original MLP critic as well.
6. **Normalization:** the current `R_MAPPO.train` denormalizes buffer values with ValueNorm. Feeding it already raw VLCM values would denormalize twice. Either supply correctly normalized values through one common normalizer, or run both the MLP and VLM pilot with ValueNorm/PopArt disabled and use a deliberate train-only target scale in the VLM loss. Save the normalizer in new training checkpoints.
7. **Deployment:** export only actor parameters and observation/action metadata. `R_MAPPOPolicy.act()` already invokes only the actor. No images, VLM, graph critic or return prediction is required by this path.

The existing `base_runner.restore()` loads native `actor.pt` and `critic.pt`. It cannot restore MA-VLCM `best.pt` into `R_Critic`. The native step checkpoints also omit optimizer state and ValueNorm state, so loading them is a warm start rather than exact optimizer/normalizer continuation. Use a dedicated actor warm-start plus a critic initialization/refitting stage; save full new training state for reliable resumes.

## Data and baselines

First collect Python RGB/state/reward rollouts from several existing actor checkpoints, including weaker actors, with fresh episode seeds. Use current physical robot positions and goal offsets; reconstruct world vx/vy from position differences, rather than passing normalized linear/angular action coordinates as world velocities. Keep robot masks, instruction semantics and adjacency definitions explicit. Derive labels from the actual Python reward stream. Python timeouts contain useful failures: include them via explicit n-step/TD bootstrapped targets or a separately documented finite-horizon objective; copying the Isaac offline rule that drops all timeouts would discard the Python failure class.

For an initial direct VLCM experiment, freeze the backbone and fit the head on Python data, then continue adapting the head as the actor changes. Include policy-specific current-rollout data. Do not keep a fitted old-policy value head frozen indefinitely and assume it remains the value of an improving policy. LoRA can follow after this pipeline is validated.

Run the following training comparisons with the same reward, timing, masks, actor initialization, seeds, environment-step budget, PPO settings and held-out evaluation:

| Arm | Centralized critic | Purpose |
| --- | --- | --- |
| B0 | Existing MLP | Successful reference |
| B1 | Graph + temporal state model, no image/language features | Controls for richer state representation and history |
| B2 | Frozen SmolVLM features + graph/temporal/value head trained on Python data | Main direct VLM-critic experiment |
| B3 | B2 with Isaac head initialization followed by Python adaptation | Tests whether the existing critic checkpoint helps |
| B4, later | LoRA + B2 | Tests adapting language-model features after the pipeline works |

Use at least three training seeds and report both environment steps and wall-clock cost. The 8-million-step actor is already at a success ceiling in this small Python test; use a shared earlier checkpoint or training from scratch across all arms to measure learning efficiency. Keep the successful actor as a transfer reference. Track return, success, formation-hold quality, collision rate, completion time, value explained variance and critic errors by team size. Evaluate exported actor-only checkpoints in Isaac with the same task settings and seeds across arms. The earlier 60 episodes have already been used for validation, so use a separately seeded final test for the chosen method.

## Literature context

The actor/centralized-critic split follows [MAPPO](https://arxiv.org/abs/2103.01955); [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) describes fitting values and using them for advantages. Training and execution can use different information, as explored in [asymmetric actor-critic](https://arxiv.org/abs/1710.06542). These references support the architecture, not an empirical claim that SmolVLM will improve this task. That claim requires the matched training comparisons above.
