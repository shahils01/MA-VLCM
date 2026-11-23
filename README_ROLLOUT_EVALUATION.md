# Matched rollout evaluations

Run from this repository with the PyTorch/Transformers environment:

```bash
python run_rollout_evaluation.py all
```

Outputs are in `outputs/eval_rollouts/` (already ignored by Git). The runner resumes
completed episode/score/video files. Use a new output directory when changing the
collection configuration. Formation assets must first be downloaded under
`outputs/eval_rollouts/assets/formation/` from
`shulnak09/polygon_formation_python_env`: `source/`, `checkpoints/`, and
`collection_plan.json`. Navigation uses the self-contained frozen sources and
actors under the merged dataset's `provenance/clean/<condition>/` directories.
The backbone uses the revision recorded in the critic, not a substitute model.

Phases can also run separately: `collect-navigation`, `collect-formation`,
`audit`, `score`, `render`. They are separate processes because the frozen formation and
navigation packages have the same Python package names and different contents.

## Cases and selection

Each scenario/team cell has three episodes, shared by both joint critics:

| Scenario | Team sizes | Selection |
|---|---|---|
| Native formation | 3, 5, 4, 6 | First three fixed fresh reset seeds |
| Navigation without obstacles | 3, 5, 4, 6 | First three fixed fresh reset seeds |
| Navigation with obstacles | 3, 5, 4, 6 | First three fixed fresh reset seeds |
| Formation, visual obstacle clutter | 3, 5 | Same resets as native formation |
| Formation, physical obstacle failure events | 3, 4, 5, 6 | Same resets as native formation; 3/5 also match visual-clutter layouts |
| Clean formation recovery counterpart | 3, 5 | Matched to accepted noisy recovery seeds |
| Formation, temporary noise and recovery | 3, 5 | First three seeds where both clean and noisy runs succeed |

This produces 66 trajectory videos when all recovery examples are found. Each
video contains both critic predictions, recorded return-to-go, incoming reward,
and team progress. A single video per trajectory avoids duplicating the same
actor rollout for each critic. Episodes retain the nine-member tar schema and
full control-step traces. Three samples illustrate behavior; they are too few
for strong population accuracy or success-rate claims.

Navigation preserves the native 10 x 10 m arena, goals, obstacles, dynamics and
rewards. Initial robot **centers** must lie inside [-4,4]^2, with pairwise center
distance >=1.5 m. Every saved episode is checked from its actual initial state;
failure raises an error. This is not a 1.5 m footprint-to-footprint gap.

Formation retains its original physical workspace: the 10-by-10 checkerboard
covers 3.048 m, and robot centers must stay within +/-1.4215 m. Enlarging this
workspace would introduce an additional distribution shift. Its native actor was
trained on 3–6 agents; the **critics** were restricted to 3/5. Navigation actors
were trained on odd teams, so 4/6 changes the actor distribution as well.

## ID and OOD interpretation

3/5-agent native scenarios are fresh samples from the original scenario
generators using a best-actor subgroup of the training behavior mixture. They
are not a replication of the whole mixture or the saved held-out test set.
Formation training included clean, mild, and moderate noise; navigation training
included clean, wandering and oscillating behavior. Do not describe all noisy
trajectories as unseen.

Obstacle navigation is condition-shift OOD for the no-obstacle critic. Removing
obstacles from the obstacle critic is also condition-shift OOD: its source
configuration requires at least one obstacle. Team counts 4/6 are unseen by both critics, but their
formation shapes are mentioned in the shared training instruction; this tests
unseen cardinality/geometry rather than unseen shape vocabulary.

Formation visual clutter changes only the image. Physical obstacles introduce
three random static disks, sampled clear of initial robot footprints and initial
assigned slots. First disk contact terminates the episode and subtracts the native
formation collision coefficient from each active robot's reward. This explicitly
changes the task's reward/termination rule; it is not a pure appearance ablation.
The actor has no obstacle sensor and the critic receives obstacle information
through the image only. Formation language is held fixed across these tests.

Noise recovery uses a correlated action burst at native control steps 41–100,
then restores the unmodified deterministic actor. This differs from the early
independent bursts used in formation training. All attempted clean/noisy outcomes
are stored in `recovery_attempts_agents_*.json`; successful pairs are selected
for the requested recovery illustrations. Failed attempts are not silently
counted as successful or used for population accuracy estimates. Up to 250
candidates per team are attempted; a shortfall is reported, not filled with
edited trajectories.

## Return estimation and reports

`training_agent_audit.json` checks selected episode counts and actual train-split
paths for all three critics. `scoring_provenance.json` records weight hashes,
epochs, model configuration and training signatures. `evaluation_results.json`
and per-episode `predictions/` JSON/CSV files contain causal frame predictions and native
return targets. `summary_metrics.json` separates task, team, condition and critic,
and provides pooled error plus episode-macro MAE. `paired_noise_results.json`
compares matched clean/noisy initial returns and predictions. The generated
`README.md` links every video.

Targets use all future control rewards, gamma=.99 per native step: 20 Hz for
formation and 10 Hz for navigation. Current incoming reward is excluded. True
terminal continuation targets are zero; raw terminal network outputs are shown
but excluded from errors. Timeout continuation is unknown, so timeout return
targets and error scores remain unavailable. Normalizers remain checkpoint-owned.

A recorded trajectory supplies a **realized** return, not the expected return
under repeated stochastic continuations. Clean and noisy paired initial inputs
are identical, so their initial critic predictions should agree even if the
unannounced future perturbation changes the realized return. Test sensitivity
after the perturbation becomes observable. Recovery does not mathematically
guarantee lower return: reward shaping and changed completion time can produce
either sign. For a policy-value claim, add repeated continuations from restored
full states under one explicitly specified policy/noise process.

## Explicit obstacle cues

`python utils/evaluate_obstacle_cues.py all` creates three matched variants of the six
existing visual-only formation-obstacle episodes: language-only, red outlines
only, and language plus outlines. Both frozen joint critics score all variants.
Results and 18 additional videos are saved in
`outputs/eval_rollouts/obstacle_cues/README.md`.

The instruction identifies the gray disks and explicitly states their visual-only
semantics. It does not add an avoidance objective or imply changed rewards.
Geometry, actor actions, full control traces and return targets remain identical;
the audit checks the physical state and label payloads against the original.
The annotation is an input-sensitivity ablation, not an obstacle-aware actor.
