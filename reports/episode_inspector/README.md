# Four recorded episode examples

`comparison.png` shows the first available 3- and 5-agent episode for each
task, with recorded images at one second, all native team control rewards,
full-control Monte Carlo return targets (gamma .99), and mean/minimum progress.
These cases were not chosen for successful outcomes. All four happen to fail;
rising progress does not imply eventual team success.

Formation examples were downloaded from
`shulnak09/polygon_formation_python_env`:

- 3 robots: `formation_control/agents_03/worker_00/episode_000000.tar`.
- 5 robots: `formation_control/agents_05/worker_02/episode_000500.tar`.

The two small downloaded archives/sidecars are local demonstration data in
`formation_samples/`, excluded from Git. Navigation examples come from the
existing combined `manyagent_gotogoal_10m_3000` folder; no archives were modified.

`four_episode_audit.json` contains the exact plotted series, frame metadata
and per-episode checks. The live server was checked against all four real
episodes: 21 checks per formation episode and 24 per navigation episode passed.
Additional real navigation episodes with obstacles and action perturbations
also passed the label/schema checks. Four inspector tests and six checkpoint
animation tests passed, including timeout handling. JavaScript syntax and
every plot-group/clock mapping were checked against the real report data.
The comparison figure was visually reviewed; full browser visual automation
was unavailable in this session.

See `../../README_DATASET_INSPECTOR.md` for running the four-panel viewer on
Palmetto/DeltaAI and forwarding its localhost port to your browser.
