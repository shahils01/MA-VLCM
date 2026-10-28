# Formation dataset diversity audit

Observed 2,000 episodes and 68,562 frames; 497 successes (24.9%).

| Policy stage | Robots | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: | ---: |
| finetuned | 3 | 125 | 108 | 86.4% |
| finetuned | 4 | 125 | 80 | 64.0% |
| finetuned | 5 | 125 | 50 | 40.0% |
| finetuned | 6 | 125 | 27 | 21.6% |
| initial | 3 | 125 | 0 | 0.0% |
| initial | 4 | 125 | 0 | 0.0% |
| initial | 5 | 125 | 0 | 0.0% |
| initial | 6 | 125 | 0 | 0.0% |
| intermediate | 3 | 125 | 37 | 29.6% |
| intermediate | 4 | 125 | 55 | 44.0% |
| intermediate | 5 | 125 | 9 | 7.2% |
| intermediate | 6 | 125 | 2 | 1.6% |
| trained | 3 | 125 | 88 | 70.4% |
| trained | 4 | 125 | 24 | 19.2% |
| trained | 5 | 125 | 14 | 11.2% |
| trained | 6 | 125 | 3 | 2.4% |

Outcomes: `{'out_of_bounds': 1498, 'success': 497, 'timeout': 5}`. Recorded collision frames: 1641 across 106 episodes.

First-frame image sizes: `{'[128, 128]': 2000}`.

Unique instructions: 1; initial scenarios: 2000; repeated-scenario groups: 0; stream seeds: 17.

Initial robot positions occupy 64/64 bins over [-1.35, 1.35] m. Occupancy demonstrates spatial variation, not independence or broad task coverage.

Patterns: `{'polygon': 2000}`; spacings: `[0.5]`; robot-center bounds: `[(-1.1975, 1.1975, -1.1975, 1.1975)]`.

Capture/control rates: `[5.0]` / `[20.0]` Hz. Episodes with a raw control reward stream: 0. Timeout/unfinished episodes excluded from return training: 5.

Sampled rewards: `{'min': -1.0, 'p05': -0.01325743063353002, 'median': -0.0015392707427963614, 'p95': -0.00022671267142868612, 'max': 1.0, 'mean': -0.0177015018176968, 'std': 0.16975129759185406}`.

Sampled proxy returns: `{'min': -1.1044514179229736, 'p05': -0.9238967299461365, 'median': -0.3934599459171295, 'p95': 0.759227740764618, 'max': 1.0, 'mean': -0.2616780162654895, 'std': 0.5180672176279836}`.

Suitable for an initial in-distribution return-regression experiment. It does not establish generalization to new tasks, instructions, visual environments, robot embodiments or behavior policies. Success/failure coverage must be checked per team size. Use scenario-separated validation and compare against a constant-return baseline.

The current data has one common instruction and one recorded formation task, spacing and workspace. Additional paraphrases alone would not demonstrate task-conditioned value prediction. To claim broader language or visual generalization, collect genuinely different goals/instructions, layouts, appearances, disturbances and independent policy/reset seeds. Retain successes, recoveries, near-boundary states and collision cases without erasing their terminal penalties.

Mixed policy stages yield the conditional expected return of the recorded behavior mixture, not the value of one specified deployment policy. Use --policy-stage to restrict training when needed.

Reward differentiability is not required for supervised value regression. Keep raw rewards and terminal events; standardize return targets using the training split. Smoothing across a collision, boundary exit or termination changes the objective and can spread future information into earlier labels.
