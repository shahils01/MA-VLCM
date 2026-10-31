# Python MAPPO with the MA-VLCM critic

## Critic training modes

Use the same `--critic-checkpoint` pointing to a trained Python VLCM `critic.pt`
to compare these modes from the same starting weights; use `--actor-from-scratch`
to start each actor from random initialization:

| Mode | CLI option | Trainable critic parameters |
| --- | --- | --- |
| Downstream fine-tuning | Neither freeze flag | Visual projection, GAT, fusion, time/context embeddings, transformer and value head |
| Head-only | `--freeze-critic-representation` | Entire value head only |
| Fully frozen | `--freeze-critic` | None |

The SmolVLM image/text backbone is frozen in all three modes. By default the fully
frozen mode requires a trained Python VLCM checkpoint. The saved Python critic's
head, normalization and value scale are loaded intact. Fully
frozen mode does not construct a critic optimizer, take critic gradients or
perform critic updates; actor learning starts at the first rollout because there
is no critic warmup to perform. `run.json`, checkpoint metadata and
`train/critic_updated` record the mode; critic loss is omitted when no critic
update occurs.

The critic's output varies with each new image and robot-state history, but its
mapping is fixed and cannot adapt to the changing policy or a new reward
function. The saved Python critic already predicts raw Python reward units; the
existing PPO code normalizes advantages across active samples, not rewards or
critic outputs. Advantage normalization does not fix calibration errors or
remove bias from bootstrapping GAE with a stale critic. Trainable value-head
adaptation is the more suitable option when the task or policy distribution
changes; compare fixed-critic experiments on matched rollout and seed budgets.

For a direct frozen Isaac baseline, `--freeze-critic --frozen-isaac-critic`
loads the supervised `outputs/zero_margin_900/best.pt` unchanged, including its
original value head and training-set observation/target normalizers. Its
image-age/hold/settled embedding is zero and frozen. Its output is an Isaac
discounted return, not a Python return: the run combines this value with raw
Python rewards in GAE without calibration, so the mixed-unit TD errors and
critic MSE are not interpretable as Python value accuracy. This is an explicit
experimental baseline, not a calibrated critic or proof of value transfer.
Image domain, prompts and termination rules also differ. The run records this
unit mismatch in `run.json` and prints it at startup. The original network's
target mean and standard deviation are applied; the Python `--value-scale 50`
does not rescale its outputs. No critic parameters or optimizer are trained.

Optionally add `--value-calibration <calibration.json>` to apply each team's
fixed affine map `V_python_estimate = slope[N] * V_isaac + intercept[N]` before
GAE, with no gradients to either the critic or the calibration. The loader
requires a fit for every selected formation, matching discount and the exact
source checkpoint SHA-256; the coefficients and file hash are saved with the
new run and the critic checkpoint. Both current and next values use the map,
while a true terminal still has zero continuation. This changes value units,
not environment rewards, and the fit may be biased because its completed
episodes came from a previously trained actor rather than the new scratch actor.
In the current local calibration, held-out frame MSE improves for each shape,
but remains approximately 100–214 in squared Python return units; timeouts
were excluded. Calibration cannot guarantee accurate current-policy values or
unbiased GAE during subsequent training.

## Optional frozen representation experiment

`--freeze-critic-representation` freezes the visual projection, GAT, fusion,
time embedding, temporal transformer and image-age/hold/settled embedding;
SmolVLM also remains frozen. Only `core.value_head` (LayerNorm, hidden linear,
GELU and final scalar linear) and the actor train. The downstream-training
description below documents the original run and the default without this flag.

Set `--critic-checkpoint` explicitly to a Python MAPPO `best/` directory or its
`critic.pt` to reuse that learned critic, including its head, normalization,
context embedding and value scale. `--actor-from-scratch` still creates a new
actor; it does not load the neighboring `actor.pt`. This is a new experiment
with fresh optimizers, not an exact resume. A pretrained head may need to adapt
to the fresh actor's behavior even though the Python reward units match.

The shell's default remains `outputs/zero_margin_900/best.pt`, which is the
supervised Isaac model and resets the final scalar layer on transfer. The
run name does not select a checkpoint. Startup output and `run.json` record
the actual checkpoint, source training step, and trainable critic parameters.

## End-to-end method and mathematics

This section was checked against the local code, collection manifests, saved
`run.json` files and evaluation results on 2026-10-29. It describes the existing
experiments; updating this document does not launch training or evaluation.

**There is a critic throughout actor training.** In the VLCM experiment, it is
SmolVLM features + GAT + fusion + temporal transformer + a scalar value head.
This replaces the native MAPPO state-vector critic. Only the SmolVLM backbone
is frozen. The downstream critic layers are optimized again on Python
rollouts, and their weights are saved as **`critic.pt`**. The actor is initialized
from scratch and needs only `actor.pt` and local observations at deployment.

### 1. What has actually been collected and trained

| Artifact | Verified local evidence | Interpretation |
| --- | --- | --- |
| Isaac source collection | [collection plan](../Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6_10hz_inner00m/collection_plan.json); 2,000 `.tar` archives and summaries | Complete collection exists, with 159,785 recorded image/state samples |
| Frozen supervised 900 experiment | [run configuration](outputs/zero_margin_900/run.json), `outputs/zero_margin_900/best.pt` | Trained for 11 epochs; epoch 5 selected by validation MSE |
| Full-collection frozen experiment | [preflight report](outputs/frozen_2000_preflight/preflight.json), [Palmetto workflow](README_FROZEN_2000.md) | Preparation verified locally; no completed frozen-2000 training checkpoint was found in local outputs during this audit |
| Separate LoRA experiment | [LoRA configuration](outputs/zero_margin_lora/run.json) and its own `best.pt` | Separate experiment; its existence does not establish Palmetto completion, and its weights are not the source for this MAPPO run |
| Actor-from-scratch Python MAPPO | [recorded run](../MARL_data_collection/on-policy/ckpts/formation_control/mappo_vlcm_actor_scratch_frozen900_seed1_20261029_173044/run.json) | `actor_checkpoint=null`, `actor_initialization=random_from_scratch`, source critic `outputs/zero_margin_900/best.pt` |

The 900 run has **724 train / 87 validation / 87 test episodes** after excluding
2 timeouts, with 28,499 / 3,238 / 3,213 causal clips. Its saved split membership
contains **initial and intermediate policies only**: training has 403 initial
and 321 intermediate episodes. The full collection also contains trained and
fine-tuned behavior, so using all 2,000 changes the behavior distribution as
well as the amount of data. Do not describe the saved 900 model as trained on
all four stages merely because its configuration says `policy_stage=all`.

The answer to “did you retrain VLCM?” has two parts: **yes, its downstream
critic layers are trained again during MAPPO**; a completed, supervised
**frozen-2000** replacement is not verified locally. A Palmetto run must be
verified from its own checkpoint/configuration/results before claiming that
it replaced the 900 initialization. No new retraining was performed for this
documentation update.

### 2. Collect the zero-wall-margin Isaac dataset

The relevant collection is
`Ground_Robot_MARL/datasets/formation_vlcm_polygon_3to6_10hz_inner00m/`.
The collection workflow is implemented in
[formation_dataset_pipeline.py](../Ground_Robot_MARL/scripts/vlcm/formation_dataset_pipeline.py)
and [collect_formation_dataset.py](../Ground_Robot_MARL/scripts/vlcm/collect_formation_dataset.py).
The collection plan, rather than defaults in older README examples, specifies
the settings used for these archives.

1. Take existing Python-trained generic-polygon MAPPO actors at four stages:
   initial (`step_001603200`), intermediate (`step_003600000`), trained
   (`step_008001600`), and stability fine-tuned (`step_001996800` in the separate
   fine-tuning run). These are behavior policies used to generate experience.
2. Transfer the actor into Isaac and roll out teams of 3, 4, 5, and 6 physical
   robots. The plan allocates **125 episodes per stage/team pair**:
   `4 stages × 4 team sizes × 125 = 2,000`. Stage names are provenance labels;
   individual episodes still succeed or fail.
3. Use the recorded Isaac configuration: a **3.048 m × 3.048 m (10 ft × 10 ft)**
   workspace, black robots of diameter **0.205 m**, polygon neighbor spacing
   **0.5 m**, slot tolerance **0.075 m**, 20 Hz control, 10 Hz RGB capture,
   256 × 256 stored images, and a 20-second episode limit (400 controls).
   Success and out-of-bounds events terminate episodes; a time limit truncates.
4. Set `formation_wall_margin=0.0`. This removes the extra safety margin,
   **not the robot radius or boundary rule**. A robot center must remain inside
   `±(1.524 − 0.1025 − 0.0) = ±1.4215 m` on both axes.
5. Capture initial samples and final pre-reset samples. Store every control
   reward between successive camera samples, including the terminal reward,
   so 10 Hz image storage does not lose the intervening 20 Hz reward stream.
6. Keep successes and failures. Do not choose only visually good episodes.
   Verify archive structure, timing and terminal labels before training.

Inventory from the local episode summaries:

| Formation | Episodes | Success | Out of bounds | Timeout |
| --- | ---: | ---: | ---: | ---: |
| Triangle, 3 robots | 500 | 299 | 200 | 1 |
| Square, 4 robots | 500 | 271 | 226 | 3 |
| Pentagon, 5 robots | 500 | 182 | 312 | 6 |
| Hexagon, 6 robots | 500 | 113 | 383 | 4 |
| Total | 2,000 | 865 | 1,121 | 14 |

Each episode archive contains per-sample `overhead.png`, `state.json`,
`adj.npy`, `reward.json`, `control_rewards.json` and associated task/geometry
records. `reward.json` is the reward at the captured control step;
`control_rewards.json` is the ordered interval of control rewards since the
previous image. The state contains poses, velocities, assigned moving targets,
control time and the instruction. Outcome/progress fields are labels for
analysis; the supervised model does not receive future outcomes as inputs.

The recorded instruction is:

> The image shows black ground robots on a 10-by-10 checkerboard grid. Count the robots visible in the image. Three robots must form an equilateral triangle, four a square, five a regular pentagon, and six a regular hexagon, with one robot at each vertex. Reach the assigned centroid-relative target positions with 0.5 m formation spacing, while keeping every robot inside the allowed inner workspace.

These behavior actors were originally learned with the earlier MARL training
system. Therefore the critic's pretraining data contain knowledge from those
policies. The later VLCM experiment tests learning a **new actor from random
weights with that pretrained representation**, not learning without any prior
policy-generated data.

### 3. Define the value target and its future horizon

Let `s_t` be the state before action `a_t`; let `r_t` be the reward returned by
the transition from `s_t` to `s_(t+1)`. Let `T` be the true terminal state index.
The full-control target in [data_utils.py](data_utils.py) is

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k},\qquad
G_T=0,\qquad \gamma=0.99\ \text{per 0.05-second control step}.
$$

**The future horizon is the remaining episode, `H_t = T − t`, not 6 or 8
frames.** The Isaac collection allows up to 400 controls, but successful or
out-of-bounds rollouts usually terminate earlier. Six frames at 5 Hz describe
one second of **past input history**; they do not cap the future return.
For intuition, `1/(1−0.99)=100` controls is a discount-weight scale of about
5 seconds, not a hard cutoff. Rewards beyond that still contribute.

At an image timestamp, all rewards already earned are excluded. If two
selected images are separated by `m` controls, the implemented recursion is

$$
G(t_j)=\sum_{k=0}^{m-1}\gamma^k r_{t_j+k}
       +\gamma^m G(t_{j+1}).
$$

At 10 Hz, normally `m=2`; at 5 Hz, `m=4`. The continuation discount at 5 Hz
is `0.99^4 ≈ 0.960596`, while every reward inside the interval retains its own
control-step discount. Subsampling images never multiplies or interpolates
rewards. The older dataset without `control_rewards.json` supports only a
sampled-reward proxy; this workflow explicitly uses `return_source=control`.

The logged scalar `G_t` is a **realized Monte Carlo return** from one rollout.
The function learned by squared-error regression approximates

$$
V_{\mathcal D}(h_t,p)=\mathbb E_{\mathcal D}[G_t\mid h_t,p],
$$

where `h_t` is the available history, `p` is the instruction, and the data
come from the recorded behavior-policy mixture. We do not observe a separate
exact expected return from the simulator. A single rollout can differ from
its conditional expectation, and this offline value is not automatically
calibrated for a new policy or a different reward function.

The 14 timeouts are excluded from the supervised full-return objective:
future continuation after the time limit was not recorded. With seed 43,
the verified 1,986 complete episodes give **1,594 train / 196 validation /
196 test**. Splits group initial scenarios and stratify by stage, team size
and outcome; neighboring clips from an episode cannot cross splits. Terminal
frames are retained to calculate the last transition reward but excluded as
training anchors, avoiding an abundance of trivial `G_T=0` examples.

### 4. What [-1, 1] means, and why there is no direct return conversion

In the Isaac [reward implementation](../Ground_Robot_MARL/source/distMARL/distMARL/tasks/direct/distmarl_formation/formation_env.py),
the authored one-step reward has the form

$$
r_t^{I}=\operatorname{clip}\left(\frac{R_t^{I}}{20},-1,1\right),
$$

where the raw expression combines formation progress, bounded formation
error, one-time success bonus, collision proximity, out-of-bounds penalty,
action effort and step cost. Current coefficients are 10 for progress,
−0.5 for bounded error, +50 for new success, −10 for proximity cost, −20
for boundary exit, −0.005 for normalized effort and −0.01 per step;
error is clipped to 4 m² and progress to ±0.5 m²/control.

**The one-step reward is clipped to [-1, 1]; the discounted return and
network output are not.** For a finite horizon,

$$
|G_t|\leq\sum_{k=0}^{H_t-1}\gamma^k
=\frac{1-\gamma^{H_t}}{1-\gamma}.
$$

Even if many observed returns happen to lie near [-1, 1], the value head has
an unrestricted linear output, not a final `tanh`.

Clipping destroys information: raw rewards 20 and 50 can both become +1.
Multiplying an old value by 20 cannot undo that, and multiplying it by 50
cannot turn it into a Python-policy value either. The domains also differ in
geometry, success timing and failure rules. Consequently the implementation
**transfers the representation and relearns the Python scalar value**, rather
than trying to invert the Isaac reward transformation.

### 5. Supervised SmolVLM + GAT + temporal value training

Implementation: [model.py](model.py), [gat.py](gat.py),
[data_utils.py](data_utils.py) and [train.py](train.py).

1. Load `HuggingFaceTB/SmolVLM-500M-Instruct` (0.5B), pinned in the saved run to
   revision `a7da5b986cb59b408707209984f360a5f4ad7e47`. For the verified 900 run,
   preprocess images at size 128, select images at 5 Hz, and form six-frame
   causal clips with stride 1. Early clips are padded and masked.
2. Freeze **both image and language backbone weights**. The images and text
   themselves are inputs, not trainable objects. Encode each image together
   with its instruction using SmolVLM's processor/chat template. Extract the
   final non-padding token's hidden representation, dimension 960, without
   generating text or using vocabulary logits:
   `z_j = stop_gradient(B_frozen(I_j, p))`. Cache these features once.
3. For each active robot, build the ten-dimensional vector

   $$
   x_{i,j}=[x,y,\sin\theta,\cos\theta,v_x,v_y,
            g_x-x,g_y-y,\|g-p_i\|,d_{\mathrm{nearest}}].
   $$

   Normalize using statistics computed on the training episodes only.
   Feed the vectors and recorded adjacency into the GAT; mask padded robots
   and include valid self-connections. For each attention head, the layer
   computes scores proportional to
   `aᵀ LeakyReLU(W_s u_i + W_n u_k)`, softmaxes over allowed neighbors,
   aggregates transformed neighbor vectors, and applies a residual/output
   projection and layer normalization. Pool masked mean, masked maximum and
   `log(N)` to produce a team feature `g_j`.
4. Project image-language and graph representations into the shared width 128,
   fuse them, and add the relative timestamp embedding:

   $$
   v_j=P(z_j),\qquad
   f_j=\operatorname{GELU}(W_f[v_j;g_j]+b_f)+e(t_j-t).
   $$

5. Run one temporal-transformer layer with four attention heads over the
   six observations ending at anchor `t`. Use the final token's representation
   `u_t`. Every input timestamp is at or before `t`; the transformer is
   bidirectional within this past-only window, so no future rollout is needed.
6. Apply `LayerNorm → Linear(128,128) → GELU → Linear(128,1)` to obtain
   `q_t`. Undo training-target standardization:

   $$
   \widehat V_t=\mu_G+\sigma_G q_t,\qquad
   L_{\mathrm{sup}}=\frac1B\sum_t
     \left(\frac{\widehat V_t-G_t}{\sigma_G}\right)^2.
   $$

   The saved 900 model uses `μ_G ≈ −0.428456` and `σ_G ≈ 0.577435`.
   These are training-data statistics, not a mapping to [-1, 1]. Reported
   MSE/MAE are in the stored Isaac return units; the optimized MSE is scaled
   by `σ_G²` for conditioning.
7. Optimize the visual projection, GAT, fusion, time embedding, transformer
   and value head; keep SmolVLM frozen. The verified 900 configuration uses
   AdamW, learning rate `1e-4`, weight decay `0.01`, batch 64 and dropout 0.2.
   Choose `best.pt` using internal validation MSE. The proposed baseline900
   frozen-2000 profile repeats these architecture/input settings with the
   expanded split and a newly trained downstream model.

| Component | Supervised frozen experiment | Python VLCM-MAPPO training |
| --- | --- | --- |
| SmolVLM image + language backbone | Frozen pretrained weights; cached features | Same frozen pretrained weights; new Python image features |
| Visual projection | Trained | Loaded from supervised checkpoint, then trained again |
| GAT and team pooling | Trained | Loaded, then trained again |
| Fusion and relative-time embedding | Trained | Loaded, then trained again |
| Temporal transformer | Trained | Loaded, then trained again |
| Intermediate value-head layers | Trained | Loaded, then trained again |
| Final scalar linear layer | Trained for Isaac return | Reset randomly, trained for Python return |
| Image-age/hold/settled embedding | Absent | Newly added and trained |
| Actor | No actor optimization in supervised regression | Initialized randomly and trained with PPO |

The cached Python integration accepts frozen-backbone checkpoints; it does
not load the separate LoRA-trained backbone as its encoder.

### 6. Validate value estimates on recorded rollouts

[evaluate.py](evaluate.py) loads the selected supervised `best.pt`, reconstructs
the saved split or a new external collection, builds the same past-only
features and uses the original training normalization. No optimizer runs.
It saves per-anchor predictions, realized return labels and aggregate metrics.
[animate_checkpoint.py](animate_checkpoint.py) turns these predictions into
formation-specific replay videos.

The external collection was prepared with [collect_fresh_validation.py](collect_fresh_validation.py)
using the **intermediate behavior actor**, 15 episodes for each of four
formations and reset-stream seeds starting at **7300001**, offset by 100000
per formation. It preserves the Isaac task and reward definitions. Its 60
archives contain 23 successes and 37 out-of-bounds failures, with no timeouts.
These are seeded simulator streams, not independently specified seeds for
every individual episode. Preflight checks scenario/seed/archive independence
from the training collection; outcomes were not forced into equal quotas.

Verified results for the **900-run epoch-5 checkpoint**:

| Evaluation | Episodes / anchors | MSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Internal validation | 87 / 3,238 | 0.091362 | 0.179261 | 0.703978 |
| Internal test | 87 / 3,213 | 0.110972 | 0.200388 | 0.666533 |
| External seeded Isaac rollouts | 60 / 2,498 | 0.057100 | 0.161039 | 0.816014 |

Sources: [validation metrics](outputs/zero_margin_900/validation_metrics.json),
[test metrics](outputs/zero_margin_900/test_metrics.json), and
[external metrics](outputs/validation_seed7300001_evaluation/external_metrics.json).
At epoch 5, train MSE was 0.058892; by epoch 11 it fell to 0.037258 while
validation MSE rose to 0.115185. This is why the best checkpoint is epoch 5,
not the last epoch. Aggregate scores do not guarantee accurate values for
every successful formation; inspect per-team/per-outcome errors as well.
Repeatedly using the external 60 for decisions makes it a validation benchmark;
reserve new seeds for a final test.

A reward spike and a value drop can coexist: the final transition earns the
success reward, but **after** that transition `G_T=0` because no reward remains.
If an animation excludes the terminal frame, its instantaneous reward panel
can omit the success spike even though the preceding future-return labels
include it. Reward plots, return plots and progress plots are different
quantities. The value model is a regressor, not a success classifier; the
simulator's termination rule supplies the success label.

### 7. Adapt the representation to the Python task

[python_mappo.py](python_mappo.py) and [train_mappo.py](train_mappo.py) implement
the adaptation. The verified actor-scratch run loads
`outputs/zero_margin_900/best.pt` and performs these initialization steps:

1. Construct the native shared `R_Actor` with random weights: 25 local
   observation values per robot, hidden size 64, and two continuous action
   coordinates (normalized linear and angular velocity). It receives neither
   language nor images nor a value prediction as an action input.
2. Construct `PythonValueCritic` and copy the supervised downstream weights.
   Reset the final scalar regressor, set the old target mean/std to 0/1, and
   replace the robot-feature normalization with zero mean and fixed scales
   `[3,3,1,1,0.22,0.22,3,3,3,3]`.
3. Add a trainable three-input context embedding for camera age, current
   formation-hold fraction and settled fraction. Initialize its linear weights
   and bias to zero. These inputs describe the current state of reward/holding
   logic; they do not include future rewards or eventual success labels.
4. Freeze SmolVLM and encode new Python renders. Use dropout 0 in the
   downstream model for consistent PPO old/new value comparisons; `.eval()`
   disables dropout but does **not** disable parameter gradients.
5. Fit the critic for the first five rollout updates with the actor unchanged,
   then update both actor and downstream critic on fresh rollouts.

The Python simulator keeps the successful task settings: polygon spacing
1 m, random starts in `[-1.5,1.5]²`, minimum start separation 0.55 m, observation
scale 3, and 0.205 m robots. The renderer tiles the same floor texture over a
**6.096 m / 20 ft view**, with black robots and no heading/goal overlays.
This is a camera coverage choice, not a 20 ft boundary imposed on the dynamics.
There are no hard Python arena walls. The image domain therefore resembles
Isaac but is not identical: workspace coverage, robot pixel size, spacing,
rendering and termination rules differ.

The Python history instruction is generated as:

> Overhead view of N black ground robots on a tiled floor. Form a regular N-vertex polygon with neighbor spacing 1 metres. The camera covers 6.096 by 6.096 metres.

`N` is replaced by the active robot count. Selecting 3/4/5/6 robots configures
the regular polygon environment. Arbitrary language commands do not directly
control this actor; changing text alone does not change its assigned goals.

### 8. Python reward, expected value and output scaling

Let `c_t` be the current centroid and `q_i` robot i's assigned polygon offset.
The reported formation error and success condition are

$$
E_t=\frac1N\sum_i\|p_{i,t}-(c_t+q_i)\|^2,\qquad
\max_i\|p_{i,t}-(c_t+q_i)\|\leq0.05\text{ m for 20 consecutive controls}.
$$

Thus `formation_error=0.01` is **0.01 m²**, corresponding to 0.1 m RMS error.
It is not evidence that every robot is within 0.05 m, much less that the team
held that tolerance for one second. Episode formation error metrics aggregate
final errors of completed episodes; success is a stricter event.

In the current [FormationControlEnv.py](../MARL_data_collection/RL_bound_overwatch/envs/FormationControlEnv.py),
the reward returned after action t is

$$
\begin{aligned}
r_t^{P}={}&10(E_t-E_{t+1})-0.5E_{t+1}+50S_{t+1}-10C_{t+1}\\
         &-0.005U_t-0.2U_t^{\mathrm{settled}}-1.0L_{t+1}
           +0.05B_{t+1}-0.01.
\end{aligned}
$$

Here `S` is a newly achieved held formation (one-time bonus), `C` is the
environment collision count, `U` is mean squared two-coordinate action effort,
`U_settled` includes only settled robots but retains the mean over the team,
`L` is the fraction exiting the settling band, and `B` says all robots are
currently within tolerance. Settling latches use the environment's entry/release
radii. The reward is duplicated across active robots; the collector averages
only those active entries, recovering one team reward. Padded slots contribute
neither rewards nor actor losses. The Python reward is **not clipped to [-1,1]**.

The on-policy target function becomes

$$
V^{\pi}(h_t,p)=\mathbb E_{\pi}\left[
\sum_{k\ge0}\gamma^k r^P_{t+k}\mid h_t,p\right]
$$

with continuation ending at a true terminal and bootstrapping at artificial
time limits. The new regressor outputs an unrestricted scalar `q_ψ`, and

$$
V_\psi(h_t,p)=s q_\psi(h_t,p),\qquad s=50.
$$

This is an **invertible conditioning scale for the newly learned Python
value**, not a conversion from the old Isaac prediction. Raw rewards, TD
residuals and GAE use `V`, while critic loss compares `V/50` to its target/50.
There is no ValueNorm or PopArt in this integration. For example, if the new
head outputs `q=0.4`, the estimated Python return is 20; `q` may also exceed 1.

### 9. Collect histories and train MAPPO step by step

The recorded run uses **16 logical Python environments**, 300 controls per
rollout, 20 Hz control and a camera every four controls (5 Hz). This yields
`16 × 300 = 4,800` team transitions per rollout update. The collector steps
the environment objects in a Python loop while batching actor inference;
16 environments does not mean 16 multiprocessing simulator workers.
Teams start distributed over 3–6 robots and are resampled at resets.

1. At every control step, the actor samples an action from each active robot's
   local observation. Store observations, actions and old action log-probabilities.
2. Advance each simulator, record the raw team reward and separate
   `terminated`/`truncated` flags, and append robot state and causal context.
3. Capture images at 5 Hz. At a control step without a new image, pair the
   current states with the latest image available **at or before** each state.
   The six state slots are `t−20,t−16,t−12,t−8,t−4,t` in control indices;
   image ages range from 0 to 0.15 seconds. Mask missing initial history and
   keep histories separate across resets. Recompute graph edges within 1 m.
4. After collection, encode each unique camera frame once with the frozen
   SmolVLM. Batched encoding is possible because the actor did not need images
   to act. Every critic input still contains only that timestep's past, even
   though inference is computed later. This implements a simulated periodic
   camera; it is not a measured real-time 5 Hz inference guarantee.
5. Evaluate old VLCM values for each state and its actual next state, including
   the final pre-reset observation at a timeout. Construct TD residuals and GAE:

   $$
   \delta_t=r_t^P+\gamma(1-d_t^{\mathrm{term}})V_{\mathrm{old}}(h_{t+1})
            -V_{\mathrm{old}}(h_t),
   $$

   $$
   A_t=\delta_t+\gamma\lambda(1-d_t^{\mathrm{done}})A_{t+1},\qquad
   Y_t=V_{\mathrm{old}}(h_t)+A_t,\qquad \lambda=0.95.
   $$

   `done = terminated OR truncated`. At true success, the next-value multiplier
   is zero. At a timeout, bootstrap the final state's value but cut the GAE
   recursion before the reset episode. At a rollout cutoff, bootstrap the
   continuing state and preserve its history. Therefore MAPPO does not require
   complete Monte Carlo rollouts to termination to train the critic.
6. Repeat the same team advantage for active agents, normalize it across
   active actor samples, and optimize the shared actor with clipped PPO:

   $$
   \rho_{i,t}(\theta)=\frac{\pi_\theta(a_{i,t}\mid o_{i,t})}
                            {\pi_{\mathrm{old}}(a_{i,t}\mid o_{i,t})},
   $$

   $$
   L_{\mathrm{actor}}=-\mathbb E\left[
     \min\{\rho_{i,t}\bar A_t,
            \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\bar A_t\}
     \right]-\beta\mathcal H,\qquad \epsilon=0.2.
   $$

   Advantages are detached scalars. There is no backpropagation through the
   simulator or frozen SmolVLM into the actor. A better-than-expected outcome
   increases the probability of the sampled action through this loss.
7. Update the downstream VLCM critic toward the fixed GAE targets `Y_t`.
   With `v=V_ψ/50`, `v_old=V_old/50`, `y=Y/50` and
   `v_clip=v_old+clip(v−v_old,−0.2,0.2)`, the implemented loss is

   $$
   L_{\mathrm{critic}}=\mathbb E\left[
     \max\{\operatorname{Huber}_{10}(v-y),
             \operatorname{Huber}_{10}(v_{\mathrm{clip}}-y)\}\right].
   $$

   Clipping here limits value changes in the loss; it does not bound return
   predictions. The 0.2 clip corresponds to 10 raw reward units at scale 50.
   The recorded run uses actor/critic learning rates `5e-4`, five actor PPO
   epochs and five critic epochs per rollout, critic batches of 64, and
   entropy coefficient `1e-4`. The first five rollout updates train only the
   critic; actor updates begin afterward.
8. Repeat with new on-policy experience as both actor and critic change.
   Evaluate actor success separately on fixed validation resets. A low critic
   loss against `Y_t` is not sufficient evidence of good task success because
   `Y_t` itself contains bootstrapped estimates.

Increasing environments supplies more trajectories per update; it does not by
itself guarantee low-bias GAE. Critic accuracy, gamma, lambda, rollout lengths
and correct terminal/timeout handling also determine the estimates.

### 10. How this differs from a traditional MAPPO critic

| Question | Traditional state-vector critic in this repository | VLCM critic in this experiment |
| --- | --- | --- |
| Centralized value input | 52-dimensional shared state | Overhead image + task text + robot graph/history + timing/holding context |
| Network | Native `R_Critic`, an MLP for this non-recurrent configuration | Frozen SmolVLM, trainable projection/GAT/fusion/transformer/value head |
| Pretraining | Can initialize a state critic from scratch | Starts with offline Isaac return representations; resets scalar output for Python |
| On-policy supervision | Environment rewards and bootstrapped targets | Environment rewards and bootstrapped targets |
| Policy optimization | PPO using advantages | PPO using advantages |
| Deployment | Actor can run without centralized critic | Actor can run without VLCM or images |
| Saved value artifact | Native critic weights | Python wrapper checkpoint named `critic.pt` |

The algorithm still has an RL critic: **VLCM is that critic**. There is no
second, native `R_Critic` determining advantages in the `critic=vlcm` branch.
Importing `R_Critic` in the module supports the explicit `critic=mlp` baseline;
it does not make it active in the VLCM branch. The VLM is not authoring rewards:
the Python environment supplies rewards, and the multimodal critic learns to
predict their discounted future accumulation.

The actor receives task geometry through its local observations. Its ability
to operate without VLCM at deployment comes from policy optimization, not from
copying the critic into its network or calling a language model for each action.

### 11. Checkpoints, evaluation videos and deployment

Three different artifacts must not be confused:

| File | Contents and intended use |
| --- | --- |
| `MA-VLCM/outputs/zero_margin_900/best.pt` | Supervised Isaac downstream model, normalization, configuration and training state; frozen SmolVLM base weights are loaded separately |
| `<MAPPO run>/best/actor.pt` or `checkpoints/step_.../actor.pt` | Learned native actor state dictionary; deployable with the matching observation/action contract |
| Matching `critic.pt` in the same MAPPO checkpoint directory | `python_mappo_vlcm_v1`: adapted VLCM downstream weights, context embedding, scale and configuration; loaded with `load_python_critic` |

The actor-scratch run directory is
`MARL_data_collection/on-policy/ckpts/formation_control/mappo_vlcm_actor_scratch_frozen900_seed1_20261029_173044/`.
Both **`best/actor.pt` and `best/critic.pt` exist locally**. `best` is selected by
mean actor validation success, with mean return as the tiebreaker, so it may be
older than the latest periodic snapshot. The run also saves optimizer artifacts
and metadata; frozen 0.5B backbone tensors are not copied into every checkpoint.

[evaluate_python_vlcm_actor.py](evaluate_python_vlcm_actor.py) loads a matching
actor/critic pair, resets the Python environment to seeded random positions,
and lets the actor alone choose actions. Afterward it computes VLCM values
from the recorded causal histories and compares them to observed discounted
Python rewards. It saves per-episode `.mp4` and `.csv` files plus `summary.json`.
There is no training during this evaluation. Selecting five episodes for each
of triangle/square/pentagon creates 15 videos; a 500-step override extends only
that evaluation time limit. Success can terminate an episode earlier, and this
override does not change the training run's recorded 300-step limit.

For a timeout, the evaluator labels the observed sum as **partial return to
timeout** and does not report it as a complete Monte Carlo value target. For a
true terminal, it sets terminal continuation value to zero by definition;
that plotted endpoint is not a raw network prediction. A deterministic actor
rollout also differs from the stochastic policy used during PPO training, so
a single video is a diagnostic rather than a calibrated estimate of the full
training-policy expectation.

To show that learning is progressing, compare saved actor checkpoints on the
same evaluation seeds and report success rate, completion length, maximum slot
error, collisions and return. To compare critic architectures, use the matched
MLP/state-temporal/VLCM baselines described below, with the same actor
initialization, reward/task settings, seeds and transition budget. A visually
plausible polygon or a declining value loss alone does not establish success.

## Operational launchers and implementation notes

Run from `/home/anshul/Research/Postdoc`:

```bash
bash MA-VLCM/train_mappo.sh
```

This starts training with online W&B logging to `formation-mappo-vlcm`, using
the local frozen `outputs/zero_margin_900/best.pt` as the critic's representation
initialization. The native shared MAPPO actor trains from scratch. Configure
W&B credentials in your environment or use `WANDB_MODE=offline` for local logs.
The script uses the existing `env_isaaclab` Python; override `PYTHON_BIN` if needed.

For the main actor-learning experiment, explicitly enforce random initialization:

```bash
bash MARL_data_collection/on-policy/onpolicy/scripts/train_generic_polygon_vlcm.sh --actor-from-scratch
```

Do not set `ACTOR_CHECKPOINT` for this run. The flag rejects any simultaneous
actor-checkpoint argument, and `run.json` records `actor_initialization` and a
null actor checkpoint/hash. The pretrained supervised VLCM representation is
still used; its Python value output adapts on current rollouts. A successful
pretrained actor is only a separate adaptation/control experiment and cannot
establish that the VLCM taught an actor from scratch.

For a new environment, install `MA-VLCM/requirements-mappo.txt`; keep the sibling
`MARL_data_collection` checkout available because its actor/environment are
imported directly.

The Python entrypoint is `MA-VLCM/train_mappo.py`. Launchers also exist under
the requested `on-policy` tree:

```bash
bash MARL_data_collection/on-policy/onpolicy/scripts/train_generic_polygon_vlcm.sh
/home/anshul/miniconda3/envs/env_isaaclab/bin/python \
  MARL_data_collection/on-policy/onpolicy/scripts/train/train_formation_vlcm.py --help
```

Command-line flags at the end of the shell command override its defaults.
Output directories must be new. Invoking the Python entrypoint without
`--critic-checkpoint` initializes the graph/temporal head from scratch; the
shell supplies the local checkpoint.

## Warm-start the successful actor

```bash
ACTOR_CHECKPOINT="$PWD/MARL_data_collection/on-policy/ckpts/formation_control/mappo_generic_polygon_3to6_seed1/polygon/step_008001600" \
ACTOR_LR=0.00005 WANDB_MODE=online \
bash MA-VLCM/train_mappo.sh --run-name vlcm-warmstart-seed1
```

This loads actor weights, creates new optimizers and initializes a new scalar
critic output. It is a warm start, not an exact training resume. The lower
actor rate is suggested for adaptation; the scratch default retains 0.0005.
Initially only the critic updates for five rollouts (`--critic-warmup-updates`),
then both networks learn. Use an earlier actor or scratch training across all
arms to measure learning efficiency: the successful actor has little room to
improve on the previous small Python evaluation.

## What trains

`python_mappo.py` imports the actual `on-policy` `R_Actor` and the canonical
`RL_bound_overwatch` environment. Its dedicated collector and clipped PPO update
replace the value path without modifying the working native training runner.
It does not use the old MARL critic for the VLCM arm.

The real SmolVLM image/language encoder is **frozen**; each camera frame is
encoded once. The visual projection, GAT, fusion, temporal transformer and
value head train on current Python rollouts. This cached-feature path rejects
LoRA/full-backbone checkpoints. Freezing SmolVLM does not freeze the critic head.
These are the default fine-tuning settings; the following mode freezes all
critic layers before the value head.

### Frozen representation, adaptive value head

Run a separate actor-from-scratch experiment from this repository directory:

```bash
bash train_mappo.sh --actor-from-scratch --freeze-critic-representation \
  --run-name vlcm-frozen-representation-seed1
```

This requires `--critic-checkpoint` (supplied by the shell launcher). SmolVLM,
visual projection, GAT, fusion, time embedding and temporal transformer remain
frozen. Only `core.value_head` (LayerNorm, linear, GELU, scalar linear) and the
actor learn. The intermediate value-head weights transfer from pretraining;
the last scalar layer is reset as in the default run. The existing critic
warmup, raw reward targets and fixed physical-unit observation scales remain.
This freezes the transferred weights, not the original supervised input/output
normalization contract.

With the default supervised Isaac checkpoint, the Python-only age/hold/settled
context embedding is frozen at its zero initialization, so these extra inputs
have no effect. A Python critic checkpoint instead preserves its learned embedding.
The default fine-tuning mode learns that embedding; account for this difference
when interpreting the comparison. `run.json` records the adaptation mode and
the exact trainable parameter names/count, and saved critics restore the mode.

The scalar output is **unbounded**, not restricted to `[-1, 1]`: neither the
supervised nor Python value head ends in tanh. The existing `--value-scale 50`
multiplies the scalar prediction and divides predictions/targets in the loss;
it does not clip rewards or restrict return estimates to `[-50, 50]`. The head
learns discounted returns, whose range differs from immediate rewards.

Use matched seeds, actor initialization and training budgets when comparing
with the default fine-tuned representation. This experiment tests representation
reuse; evidence for a vision benefit additionally requires a matched state-only
baseline. Successful training with a frozen representation alone does not prove
that images improve learning.

The Isaac checkpoint initializes the visual/GAT/fusion/temporal/time layers and
intermediate value head. The last scalar layer is reset, Isaac target statistics
are discarded, and observation normalizers use physical-unit scales
`[3,3,1,1,.22,.22,3,3,3,3]`. Isaac's clipped rewards, boundary termination and
success timing differ, so its old scalar predictions are not calibrated Python
returns. Representation transfer still needs adaptation to the Python domain.

The actor keeps its native 25-value local observation and six padded slots;
images and value predictions are not actor inputs. Detached critic advantages
drive actor PPO updates. Exported `actor.pt` therefore deploys independently
through the existing actor-only Python/Isaac inference contract.

## Reward and timing

| Item | Default |
| --- | --- |
| Teams | 3/4/5/6 active robots, six padded slots |
| Successful task settings | Spacing 1 m; starts ±1.5 m; initial separation 0.55 m; observation divisor 3 |
| Success | All robots within 0.05 m for 20 consecutive controls |
| Control / camera | 20 Hz state/actions/rewards; 5 Hz images |
| Context | Six states at `t-20,t-16,t-12,t-8,t-4,t`: one second once full |
| Image association | Latest image at/before each state; explicit age 0–0.15 s |
| Graph | Distance edges within 1 m; self connections; padded robots masked |
| Discount / GAE | gamma 0.99 per 0.05 s; lambda 0.95 |
| Reward | Original raw Python reward, including +50 success |
| True terminal | Retain final reward, bootstrap zero |
| Timeout | Bootstrap actual final pre-reset state, cut GAE before reset |
| Rollout cutoff | Bootstrap continuing state; retain episode history |

The ten robot features match MA-VLCM: world x/y, sin/cos yaw, measured world
vx/vy, assigned-goal relative x/y, goal distance, nearest-robot distance.
Additional causal inputs encode image age, current formation-hold fraction and
current settled fraction. Goals use the current environment slot assignment;
velocity uses position differences, not normalized unicycle commands.

Team reward is averaged over active agents only; their rewards are duplicates.
The same team advantage goes to active actors; padded zeros never enter policy
loss or graph pooling. The critic trains once per team transition.

Images are encoded after collection because the actor does not use them. Each
value uses only images already available at that simulated time. This models a
periodic 5 Hz camera, not arbitrary wall-clock inference latency. Initial
history is padded/masked, resets get their own camera, and no future images or
outcome labels enter the inputs. Dropout is disabled for consistent old/new
value comparisons while gradients remain enabled.

Targets are `V_old + GAE`, using
`delta = r + gamma*(1-terminated)*V_old(actual_next_state) - V_old(state)` and
GAE recursion masked by `1-done`. They estimate the current policy's discounted
return through TD/GAE, rather than a new fixed-horizon Monte Carlo dataset.

Predictions, GAE and reported rewards stay in **raw Python reward units**.
All three arms disable ValueNorm/PopArt and use `--value-scale 50`: the network
output is multiplied by 50; both sides of the clipped Huber loss are divided by
50. This conditions learning without changing rewards/gamma. The value clip is
0.2 in normalized units (10 raw units at scale 50). These normalization and
timeout settings differ from the legacy native runner, so retain its original
training script as a separate historical reference.

## Floor and workspace recommendation

Keep the successful **physical task and actor state scaling** for the first
experiment. Training uses
`env.render(size=256, view_size_m=6.096, tile_floor=True)` to repeat the original
3.048 m floor texture at its physical scale across a 20 ft camera view. Black
robots remain 0.205 m in diameter, without headings/IDs/goals. Tiling changes
pixels, not boundaries, actions or reset distributions. The renderer's default
outside training remains the original single floor.

The Python task has no hard arena walls: reset centers can approach a 10 ft
camera edge, and robots can move beyond it. A wider view avoids many cropped
robots but cannot guarantee coverage. Monitor
`rollout/outside_robot_frame_fraction` and increase `--view-size-m` if needed.
Coordinates are not clamped into the image; `training_view.png` records the view.

Floor matching helps appearance but does **not** preserve robot pixel scale:
at equal resolution, a 20 ft view makes robots half as wide as a 10 ft view.
At 256 pixels they are roughly 8.6 pixels wide in 20 ft. Higher resolution costs
more encoder time. Adapt the head on actual Python images; shrinking the task
to 10 ft solely to match the old critic should be a separate comparison because
it also changes the state/reward distribution. The training-only critic need
not interpret Isaac images during deployment; retain the existing actor transfer
mapping and evaluate it separately.

## Baselines and logging

```bash
CRITIC=mlp            SEED=1 bash MA-VLCM/train_mappo.sh
CRITIC=state-temporal SEED=1 bash MA-VLCM/train_mappo.sh
CRITIC=vlcm           SEED=1 bash MA-VLCM/train_mappo.sh
```

Keep actor initialization, seeds, task settings, PPO budget, warmup and evaluation
matched. `mlp` uses the native 52-value `R_Critic` without images. `state-temporal`
uses the same GAT/history/context as VLCM but zero visual features and no encoder;
its shell also loads the Isaac representation. Comparing state-temporal and
VLCM isolates the image/language contribution. The MLP also differs in history
and available hold/context inputs. Invoke the Python entrypoint without a critic
checkpoint for a scratch-head ablation, matching the remaining arguments. Use
at least three training seeds before judging gains.

Periodic deterministic **actor-only** evaluation uses fixed seeds starting from
9300001, separate from training resets. Per-formation metrics include success,
return, completion steps, collision steps and formation error. These repeatedly
used seeds are validation for checkpoint selection; reserve final-test seeds.

W&B/`metrics.jsonl` include actor loss/KL/clipping/base-Gaussian entropy, critic
loss/raw MSE/explained variance **before** each update, per-formation episode and
evaluation statistics, image ages, cropping fraction and timing. Critic errors
are against GAE training targets, not held-out Monte Carlo returns; their decline
alone does not prove better policy performance. The native Gaussian entropy
proxy can be negative for a narrow policy.

## Outputs and checks

`checkpoints/step_XXXXXXXXX/` contains native deployable `actor.pt`, Python
`critic.pt` (format `python_mappo_vlcm_v1`), optimizer artifacts and
`training_metadata.json`. `best/` selects mean validation success then mean
return. The new critic includes timing/hold embeddings and the fixed output
scale; it differs from both supervised `best.pt` and native MLP checkpoints.
Use `python_mappo.load_python_critic` for critic inference. No exact environment/
RNG/optimizer resume CLI is provided.

The shell launcher saves every two **rollout updates**, each followed by its
configured PPO epochs. With 16 environments and 300 controls per rollout that
is every 9,600 environment transitions. `--keep-checkpoints 2` retains only the
latest two periodic directories plus `best/`; older periodic directories from
this new run are removed after a replacement has been saved. Each directory has
actor, critic and optimizer `.pt` files. Frozen backbone weights and image
histories are not duplicated into these checkpoints. Set `KEEP_CHECKPOINTS=0`
only if deliberately keeping every periodic snapshot.

Actor-only validation runs before training (`initial_evaluation.json`), so the
successful initialization remains eligible as the best actor. The first five
rollouts fit the Python critic before actor updates begin. `wandb_run.json`
records the live run URL; the process launch sidecar and log for locally started
runs are alongside the checkpoint directory.

Verified locally:

- Unit tests cover terminal success reward, timeout bootstrap, rollout cutoff,
  causal image age, resets, masks, feature-cache reuse, actor/GAT/transformer
  weight updates and actor/critic checkpoint round trips.
- Real frozen SmolVLM CUDA: two updates/32 transitions, four formations, timeout
  resets, actor-only evaluation and offline W&B in
  `outputs/mappo_integration_smoke_vlcm/`.
- Real CUDA default 256-pixel view with full one-second contexts: 128 transitions
  and two PPO epochs in `outputs/mappo_integration_smoke_vlcm_256/`.
- CPU MLP/state-temporal PPO runs in `outputs/mappo_integration_smoke_mlp/` and
  `outputs/mappo_integration_smoke_state_temporal/`.
- Both repository test suites passed: 32 MA-VLCM tests and 12 Python environment/
  renderer tests. The real 256-pixel run changed 13 actor tensors, 4 visual
  projection tensors, 11 GAT tensors and 12 transformer tensors relative to
  their initial checkpoints. Its exported actor also loaded directly in the
  existing evaluator and completed one square episode (162 steps, success,
  no collisions); one episode is a compatibility check, not a benchmark.

These are integration checks, not converged training or evidence of VLM gains.
Short timeout tests intentionally prevent task completion. Full multi-seed
learning and Isaac transfer comparisons remain experiments to run.

```bash
cd MA-VLCM
/home/anshul/miniconda3/envs/env_isaaclab/bin/python -m unittest discover -s tests -p 'test_python_mappo.py' -v
```

