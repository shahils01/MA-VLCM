# Formation dataset diversity audit

Observed 903 episodes and 70,900 frames; 212 successes (23.5%).

| Policy stage | Robots | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: | ---: |
| initial | 3 | 125 | 0 | 0.0% |
| initial | 4 | 125 | 1 | 0.8% |
| initial | 5 | 125 | 0 | 0.0% |
| initial | 6 | 125 | 0 | 0.0% |
| intermediate | 3 | 125 | 74 | 59.2% |
| intermediate | 4 | 125 | 93 | 74.4% |
| intermediate | 5 | 125 | 43 | 34.4% |
| intermediate | 6 | 28 | 1 | 3.6% |

Outcomes: `{'out_of_bounds': 689, 'success': 212, 'timeout': 2}`. Recorded collision frames: 924 across 21 episodes.

First-frame image sizes: `{'[256, 256]': 903}`.

Unique instructions: 1; initial scenarios: 903; repeated-scenario groups: 0; stream seeds: 8.

Initial robot positions occupy 64/64 bins over [-1.35, 1.35] m. Occupancy demonstrates spatial variation, not independence or broad task coverage.

Patterns: `{'polygon': 903}`; spacings: `[0.5]`; robot-center bounds: `[(-1.4215, 1.4215, -1.4215, 1.4215)]`.

Capture/control rates: `[10.0]` / `[20.0]` Hz. Episodes with a raw control reward stream: 903. Timeout/unfinished episodes excluded from return training: 2.

Sampled rewards: `{'min': -1.0, 'p05': -0.017812632769346238, 'median': -0.0035507349530234933, 'p95': -0.0005752421129727734, 'max': 1.0, 'mean': -0.01212419386819033, 'std': 0.11236469510829743}`.

Sampled proxy returns: `{'min': -1.2276983261108398, 'p05': -0.9187141090631485, 'median': -0.37133336067199707, 'p95': 0.7133418947458267, 'max': 1.0, 'mean': -0.2654786437480946, 'std': 0.4923697404052441}`.

Suitable for an initial in-distribution return-regression experiment. It does not establish generalization to new tasks, instructions, visual environments, robot embodiments or behavior policies. Success/failure coverage must be checked per team size. Use scenario-separated validation and compare against a constant-return baseline.

The current data has one common instruction and one recorded formation task, spacing and workspace. Additional paraphrases alone would not demonstrate task-conditioned value prediction. To claim broader language or visual generalization, collect genuinely different goals/instructions, layouts, appearances, disturbances and independent policy/reset seeds. Retain successes, recoveries, near-boundary states and collision cases without erasing their terminal penalties.

Mixed policy stages yield the conditional expected return of the recorded behavior mixture, not the value of one specified deployment policy. Use --policy-stage to restrict training when needed.

Reward differentiability is not required for supervised value regression. Keep raw rewards and terminal events; standardize return targets using the training split. Smoothing across a collision, boundary exit or termination changes the objective and can spread future information into earlier labels.

All 901 complete episodes with control streams passed interval alignment checks. Across 35029 nonterminal clips, mean full return is -0.430, mean sampled proxy is -0.272, and their mean absolute difference is 0.157. These statistics are based on the recorded team rewards.
