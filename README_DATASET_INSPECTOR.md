# Compare formation and navigation episode archives

`inspect_dataset.py` serves a read-only, CPU-only comparison viewer. It does
not load VLM weights, modify archives, extract tar members to disk, or send
data to an external service. It uses the existing VLCM training return code.

## Run on Palmetto

With the training Python environment activated, on the login host:

```bash
cd /home/anshuln/MA_VLCM/MA-VLCM
python inspect_dataset.py \
  --dataset formation /home/anshuln/Dataset/polygon_formation_python_env \
  --dataset navigation /home/anshuln/Dataset/manyagent_gotogoal_10m_3000 \
  --port 8765
```

This only indexes summaries initially. It opens an archive when you select
an episode. Keep the terminal open. On your laptop, forward the same port
to the **same login host** where the viewer runs:

```bash
ssh -N -L 8765:127.0.0.1:8765 anshuln@YOUR_PALMETTO_LOGIN_HOST
```

Open **http://127.0.0.1:8765** in your laptop browser. Use the concrete login
hostname from your SSH session if the generic login alias can reach different
hosts. No GPU allocation is needed. For DeltaAI use your `anayak2` account,
the actual DeltaAI login hostname, and your `/u/anayak2/VLCM/datasets/...` paths.

To start with particular archives instead of scanning dataset roots:

```bash
python inspect_dataset.py \
  --episode /path/to/formation/episode_000001.tar \
  --episode /path/to/formation/episode_000501.tar \
  --episode /path/to/navigation/episode_000002.tar \
  --episode /path/to/navigation/episode_002001.tar
```

Each archive needs its `.summary.json` sidecar (or legacy `.json`). Dataset
roots are local directories. The viewer deliberately does not download an
entire Hugging Face dataset when you mistype a path.

## Compare three or four episodes

The viewer initially opens up to four cases, preferring 3/5-agent examples
from each dataset. They are the first matching episodes, without success
filtering. Every panel has independent dataset, agent-count, obstacle,
noise, outcome and path-search selectors. Select an episode and click
**Load episode**. Remove one panel to compare three, or add a panel up to four.

Playback is synchronized in physical seconds. Scrub the time slider or click
and drag on a plot. Episodes with shorter durations keep their final image
and are explicitly marked ended; curves do not get padded with fabricated data.

Plot groups:

- **Reward / return / progress:** every recorded native team control reward,
  full-control discounted future return, mean and least-agent progress.
- **Team and individual progress:** per-agent progress and reached fraction.
- **Safety:** nearest robot center distance, obstacle surface clearance when
  footprint metadata exist, and robot-center margin to the workspace boundary.
- **Motion:** distance to each assigned goal/slot and physical linear/angular
  command vectors. Commands are distinct from actual Cartesian velocity.
- **Noise and graph:** recorded normalized perturbation magnitude, cumulative
  native reward, and non-self graph density.
- **Formation geometry:** recorded native slot MSE, RMS slot distance,
  pair-distance shape error, and stability-hold steps versus required hold.
  Pair-distance shape error is a derived geometric diagnostic, not a new reward.

Change gamma/image Hz and click **Recompute returns**. Use gamma .99 and image
Hz 10 to match the three current experiment configurations. Formation uses
20 Hz native control, navigation uses 10 Hz: the same numerical gamma has
different discount horizons in seconds. Images remain at their recorded
cadence; the return anchors follow the selected training image cadence.

`G_t` is a **recorded Monte Carlo return target**, not a direct measurement of
the expected value under every possible continuation. It excludes the reward
leading into the current state. The endpoint reward of a terminal transition
belongs to preceding states; the known true-terminal state's future return
is zero. Timeouts (`timeout`, `max_steps`) have unknown continuation returns,
so the viewer leaves their targets blank, matching their exclusion from
current training. Missing/misaligned control rewards are never replaced by
sampled rewards. The native reward curve still shows observed rewards.

## What to verify in an episode tar

1. **Visual and language agreement:** robot count, formation/goal assignment,
   arena size, colors, heading lines, obstacles, physical scale, image framing
   and task instruction match. Human review is still needed for these visual
   semantics; schema validation cannot count objects in an image reliably.
2. **Nine fields per frame:** `overhead.png`, `state.json`, `dist.npy`, `adj.npy`,
   `actions.npy`, `reward.json`, `control_rewards.json`, `progress.json`, `task.txt`.
   IDs/order and all per-agent dimensions must align.
3. **Time and reward alignment:** increasing timestamps; integer control steps
   per image interval; empty initial control-reward interval; all later native
   rewards retained once; frame reward matches the last reward in its interval.
   Check terminal spikes against arrival, collision or boundary events.
4. **Return semantics:** compare full control returns at gamma .99. Do not
   accidentally sum only 10 Hz endpoint rewards for the 20 Hz formation task,
   include incoming/past reward at the anchor, or mark timeouts as zero-value
   terminals. The viewer also checks the undiscounted episode reward sum.
5. **Geometry and dynamics labels:** goal distance from positions; pairwise
   `dist.npy`; binary symmetric NxN adjacency; declared threshold/self-loops;
   finite Nx2 action arrays and physical command agreement. The 1 m VLCM graph
   differs from the navigation actor's 5 m visibility graph. Navigation initial
   region/separation and stopped-after-arrival rules are checked when declared.
6. **Progress and team completion:** progress stays in [0,1], per-agent means
   agree with the team label, and the stored geometric progress definition
   matches. Mean progress can hide a stuck robot. A high mean is not team
   success. Formation robots may have to remain settled for a hold interval;
   they do not necessarily stop immediately after entering a slot.
7. **Data integrity:** images decode at a consistent size; finite model states;
   no duplicate tar members; final terminal handling; recorded tar SHA-256
   matches if a checksum is supplied. Historical absolute paths inside frozen
   tar metadata do not replace the actual selected local archive path.
8. **Dataset-level checks:** verify task/team/condition/noise/outcome coverage,
   scenario-separated splits, no duplicate scenes across holdouts, train-only
   normalizers and matched formation holdouts across experiments. A four-episode
   viewer does not establish full dataset coverage or prove absence of leakage.

Expand **Training-data checks** for per-episode findings. **Export report**
saves series/checks as JSON. A missing field is reported; successful plotting
alone is not evidence that the archive is suitable for training. The geometric
safety curves use captured endpoints, not continuous swept collision replay.

For a headless report without the web server:

```bash
python inspect_dataset.py --episode /path/to/episode_000001.tar \
  --gamma .99 --image-hz 10 --report reports/episode_checks.json
```

## Existing verification and value-visualization tools

- `MARL_data_collection/dataset_tools/validate_python_gotogoal_dataset.py`:
  full native clean-collection actor/noise/dynamics/reward and pixel replay.
  Its `noisy/` counterpart validates the perturbed collection. Use the original
  frozen tools with their matching original collection layouts; the consolidated
  dataset preserves those source plans/audits in `provenance/` and verifies
  byte-identical archive copies. This viewer does not replace native replay.
- `MA-VLCM/audit_dataset.py`: dataset coverage and distribution audit; its older
  default report is formation-specific and uses sampled reward proxies.
- `MA-VLCM/animate_checkpoint.py`: learned checkpoint value versus recorded
  return/reward/progress, or replay of saved evaluation predictions. Use this
  after training when you want the model's prediction, not just target labels.
- `MA-VLCM/analyze_saved_run.py`: saved prediction/loss analysis and reward/
  progress figures. `Ground_Robot_MARL/scripts/vlcm/policy_dataset.py` provides
  the original archive schema and progress serialization.

## Additional supervision to test

Keep native return regression as the baseline. Candidate **separate auxiliary
heads**, trained only on training episodes and evaluated in an ablation, are:

1. Per-agent progress plus team mean/minimum and reached fraction, to expose
   which robot is limiting completion. Labels use the actual state geometry,
   not elapsed time divided by episode length for noisy/failed rollouts.
2. Eventual team success, alongside current all-agent completion. These are
   different labels: a future success label must never become a model input.
3. Formation slot error and settled-hold progress for formation control;
   goal distance and clearance for navigation. Use task masks for signals that
   do not apply to both tasks.
4. Collision/boundary risk over a specified future horizon. Build labels from
   native events and censor cases without enough observed future, rather than
   marking an unobserved future safe at truncation.
5. Time to success, with survival/censoring treatment for failures/timeouts.
6. Same-task trajectory preferences from measured outcomes/returns, preferably
   matched scenarios. A noisy trajectory is not automatically worse, and native
   numerical returns from different tasks are not directly comparable rankings.

Progress estimates completion now; value estimates future reward. They should
not be forced equal. Auxiliary prediction is different from adding progress
to the environment reward (which would change the return objective). Terminal,
success, progress, reward, noise identity and checkpoint labels remain excluded
from input features. No extra training head or reward has been enabled by
installing this viewer.

Two qualifications matter for the current architecture. SmolVLM is frozen, so
auxiliary losses on the existing fused representation train the critic layers,
not the VLM backbone itself. Also, normalized geometric progress depends on
the reset distance. A six-frame clip later in an episode need not reveal that
distance: consider adding the initial frame/initial-distance context before
requiring an exact progress prediction. Current distance is already a graph
input; merely reconstructing it through a fused head can be a trivial shortcut
rather than evidence of improved visual understanding. A vision-only auxiliary
head is a separate experiment if visual grounding is the intended objective.
