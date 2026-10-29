# Python MAPPO with the MA-VLCM critic

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
