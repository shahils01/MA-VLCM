# Improving MA-VLCM training

The training target is a discounted **realized return** from a recorded
trajectory. With squared-error regression the model estimates a conditional
mean for the recorded behavior-policy mixture, conditional on the information
it receives. It is not automatically the value of an arbitrary actor. Frozen
critic MAPPO can visit different states and use different actions.

## What the current evidence says

The `critic35_sequential` fresh experiments use the same 3/5-agent formation
validation episodes. Best-checkpoint formation MSE/MAE/R² were 50.91/3.76/0.794
for formation-only, 80.41/5.45/0.675 for joint without obstacles and
74.63/5.45/0.698 for joint with obstacles. This is evidence of a joint-training
tradeoff, not proof of its cause. Their checkpoint selection score is the
equal-task average of task MSE divided by training return variance.

Current formation selection already includes naive, intermediate and successful
policies, with clean, mild and moderate disturbances. The 3/5 experiment uses
1,500 formation episodes (750 each); each navigation condition uses up to
1,500 episodes (750 per team) before removing incomplete/timeout episodes. The
older `reports/formation_diversity.md` describes a different 2,000-episode,
128-pixel Isaac collection and should not be quoted as a count for the current
Python dataset. Inspect `selection_report.json` and outcome/stage breakdowns
for the exact run being studied.

The formation collector records hold duration and per-agent settling flags, but
the ten critic robot features currently omit them. The hold counter changes
termination and success reward. At 10 Hz the six-frame critic window spans
0.5 s; the original 20-control-step success hold is 1 s. The actor's relative
observations also omit absolute boundary clearance, while the current 10-ft
dataset terminates on boundary exit. Neither fact proves a particular failure
mode; both motivate controlled tests.

## Data collection and return labels

1. Preserve the existing data as a baseline. Audit clips and **episodes** by
   task, team size, policy/checkpoint, outcome, obstacle count, distance to
   boundary, hold duration and time to termination. Report target scales,
   success rates, seed identities and frame counts. Check for leakage at the
   scenario level, not just neighboring clips.
2. Add independent seeds and policy snapshots from early scratch actors,
   intermediate actors and competent actors. Include purposeful near-boundary,
   collision, recovery, obstacle-interaction and near-success trajectories.
   Sample all 3/4/5/6 sizes if those are deployment targets; retain a separate
   3/5-only experiment to measure OOD generalization. Preserve failures and
   their true rewards; do not equate dataset size with diversity.
3. Collect matched-start **multiple continuations under the same policy** to
   estimate conditional return distributions and quantify trajectory noise.
   For frozen-critic actor training, separately evaluate returns on fresh
   scratch-actor rollouts and actor snapshots. Policy identity or a policy
   fingerprint may be needed if one network is meant to estimate many policies.
4. If the task changes to a 10-step formation hold, regenerate or verify
   replay-based labels under the new environment: termination and the success
   bonus must happen at the first 10-step hold. Recompute every preceding
   discounted target. Truncating old reward arrays alone is insufficient if
   the success bonus was issued later. Verify reward/terminal semantics and
   keep old 20-step experiments separate.
5. Save raw per-control-step rewards, actions, terminal vs timeout status,
   configuration, seeds and causal observations. Do not assign zero future
   continuation to a timeout; either exclude it from full-return supervision,
   extend it, or use a clearly documented and evaluated bootstrap.

## Model and loss experiments

Use identical source splits and freeze the SmolVLM backbone. Start from the
existing per-task variance-scaled MSE, which targets a conditional mean; do not
assume changing loss alone fixes distribution shift. Compare:

* **A:** current shared model, task-balanced per-clip MSE.
* **B:** shared representation plus explicit task ID and separate formation
  and navigation value heads, with train-only task normalizers. Warm-start
  from formation-only as a separate condition; retain comparable scratch runs.
* **C:** B plus causal hold/settled-state inputs (recorded offline and available
  online) and task-relevant boundary/obstacle information. Update the full
  preprocessing and frozen inference contracts together, then retrain offline.
* **D:** a state-only GAT+temporal critic, a vision/language-only critic and
  the full model, retrained with matched splits. Test shuffled images and
  instructions only as sensitivity analyses, since they can be out of
  distribution. Report performance and inference cost.

For each, keep train-only normalization and original reward units; log both
clip-weighted and episode-weighted errors. Try lower learning rates and stronger
regularization with early stopping, because the joint runs' training loss fell
while validation worsened. Use a formation metric or a constrained joint
selection criterion if formation degradation matters; select only on validation.
Task-gradient cosine similarities can test for competing gradients before
trying gradient surgery. Huber loss is an optional outlier-sensitivity ablation;
it does not generally estimate the conditional mean and can suppress important
success/failure targets. A separate quantile or distributional head can report
return uncertainty; retain and evaluate an explicitly mean-valued output when
the goal is expected return.

## Evaluation and deployment

Use held-out whole episodes and report per-task and per-team MAE, RMSE, bias,
R², constant baselines and success/failure/near-terminal slices, with
episode-level bootstrap intervals. A single realized return is a noisy sample
of the expected value; repeated continuations help separate error from outcome
variability. Avoid using final test data for checkpoint selection.

Finally compare fully frozen critics in scratch-actor MAPPO under identical
reward/termination contracts, seed lists and budgets; use at least three
independent training seeds where feasible and many held-out evaluation episodes.
Report success, return, collisions, boundary exits and compute, not regression
scores alone. When comparing Monte Carlo and GAE advantages, handle timeouts and
equalize other PPO settings.

## Code map

* `train.py` and `train.sh`: single-task supervised training.
* `train_multitask.py` and `train_35_experiments.sh`: single/joint task selection,
  task-balanced sampling, losses, validation and checkpointing.
* `model.py` and `gat.py`: frozen SmolVLM encoder, graph, fusion, temporal
  Transformer and scalar head; `data_utils.py` loads clips and return targets.
* `utils/`: supporting prediction, evaluation, data inspection, audits,
  calibration and analysis commands. Run as `python utils/<command>.py ...`.
* `train_mappo.py`/`python_mappo.py`: separate MARL actor-training integration.
  The checked-in launchers, snapshot tooling and frozen-rollout integration
  remain in the repository root to preserve their existing cross-project paths.

Further details: `README_EXPERIMENTS_35.md`, `README_MULTITASK.md`, and
`README_PYTHON_MAPPO.md` section 5. Palmetto workload code, including all
Python and preflight checks, must run inside Slurm on a compute node.
