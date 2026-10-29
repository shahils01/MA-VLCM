# Formation dataset diversity audit

Observed 2,000 episodes and 159,785 frames; 865 successes (43.2%).

| Policy stage | Robots | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: | ---: |
| finetuned | 3 | 125 | 120 | 96.0% |
| finetuned | 4 | 125 | 109 | 87.2% |
| finetuned | 5 | 125 | 102 | 81.6% |
| finetuned | 6 | 125 | 68 | 54.4% |
| initial | 3 | 125 | 0 | 0.0% |
| initial | 4 | 125 | 1 | 0.8% |
| initial | 5 | 125 | 0 | 0.0% |
| initial | 6 | 125 | 0 | 0.0% |
| intermediate | 3 | 125 | 74 | 59.2% |
| intermediate | 4 | 125 | 93 | 74.4% |
| intermediate | 5 | 125 | 43 | 34.4% |
| intermediate | 6 | 125 | 14 | 11.2% |
| trained | 3 | 125 | 105 | 84.0% |
| trained | 4 | 125 | 68 | 54.4% |
| trained | 5 | 125 | 37 | 29.6% |
| trained | 6 | 125 | 31 | 24.8% |

Outcomes: `{'out_of_bounds': 1121, 'success': 865, 'timeout': 14}`. Recorded collision frames: 4935 across 125 episodes.

First-frame image sizes: `{'[256, 256]': 2000}`.

Unique instructions: 1; initial scenarios: 2000; repeated-scenario groups: 0; stream seeds: 18.

Initial robot positions occupy 64/64 bins over [-1.35, 1.35] m. Occupancy demonstrates spatial variation, not independence or broad task coverage.

Patterns: `{'polygon': 2000}`; spacings: `[0.5]`; robot-center bounds: `[(-1.4215, 1.4215, -1.4215, 1.4215)]`.

Capture/control rates: `[10.0]` / `[20.0]` Hz. Episodes with a raw control reward stream: 2000. Timeout/unfinished episodes excluded from return training: 14.

Sampled rewards: `{'min': -1.0, 'p05': -0.017433851584792135, 'median': -0.002345125423744321, 'p95': -0.0004532921884674583, 'max': 1.0, 'mean': -0.006281977714216297, 'std': 0.11157932981126284}`.

Sampled proxy returns: `{'min': -1.2276983261108398, 'p05': -0.8837961375713348, 'median': -0.2275746390223503, 'p95': 0.8323635041713714, 'max': 1.0, 'mean': -0.08233000403028645, 'std': 0.5421872251372278}`.

Suitable for an initial in-distribution return-regression experiment. It does not establish generalization to new tasks, instructions, visual environments, robot embodiments or behavior policies. Success/failure coverage must be checked per team size. Use scenario-separated validation and compare against a constant-return baseline.

The current data has one common instruction and one recorded formation task, spacing and workspace. Additional paraphrases alone would not demonstrate task-conditioned value prediction. To claim broader language or visual generalization, collect genuinely different goals/instructions, layouts, appearances, disturbances and independent policy/reset seeds. Retain successes, recoveries, near-boundary states and collision cases without erasing their terminal penalties.

Mixed policy stages yield the conditional expected return of the recorded behavior mixture, not the value of one specified deployment policy. Use --policy-stage to restrict training when needed.

Reward differentiability is not required for supervised value regression. Keep raw rewards and terminal events; standardize return targets using the training split. Smoothing across a collision, boundary exit or termination changes the objective and can spread future information into earlier labels.

All 1986 complete episodes with control streams passed interval alignment checks. Across 77992 nonterminal clips, mean full return is -0.209, mean sampled proxy is -0.084, and their mean absolute difference is 0.125. These statistics are based on the recorded team rewards.
