# Task-only critic input audit

The TB3 loader previously appended current robot state to the language instruction,
including `reached` and `collision` flags. Legacy progress prompts also included
step/outcome/termination metadata. Signed-potential prompts removed the latter
fields but retained per-agent flags. The custom prompt override appended the same
state block. The live inference adapter duplicated this behavior.

This violates the intended text contract: the task instruction is the only
natural-language input. Current collision status directly overlaps with signed
potential supervision, so the earlier collision-detection results can be achieved
through an unintended input cue. Their dependence on that cue has not yet been
measured by an ablation. They do not establish visual collision understanding.

## Literature checked

| Method | Critic inputs / reward source |
|---|---|
| [RoboMeter, §III-B](https://arxiv.org/html/2603.02115v1#S3.SS2) | Instruction and image/video observations; paired videos during preference training. Progress, success and preference are supervised outputs. |
| [TOPReward, §3.1](https://arxiv.org/html/2602.19313v1#S3.SS1) | Video prefix plus task instruction in a completion query; affirmative-token log probability supplies the score. |
| [VLAC, §3.1.1](https://arxiv.org/html/2509.15937v1#S3.SS1.SSS1) | Two observed images and a task goal description produce relative progress; completion is predicted separately. |
| [SimpleVLA-RL, §3.2](https://arxiv.org/html/2509.09674v1#S3.SS2) | Environment task success supplies a binary trajectory reward for GRPO. It is not a learned dense vision-language critic baseline. |

These input formulations separate the task description from supervision. Their
success labels/targets should not be confused with language inputs containing a
success or collision answer.

## Corrected implementation

`src/ma_vlcm/tb3_task_prompt.py` defines `task_instruction_only_v1`. It constructs
TB3 task text from task identity and static task cardinality, never current poses,
distances, velocities, reached/collision flags, episode outcomes, timestamps or
reward/progress/risk labels. Literal instruction overrides are accepted without
state concatenation; overrides containing metadata assignments or dynamic state
placeholders are rejected. Training, episode inference and the live adapter use
the same builder. The live adapter was edited but was not executed on hardware.

The separate numerical robot-observation/adjacency channels remain available as
requested; no new collision/success/reward features were added to them. The
existing nine-dimensional row contains pose, velocity, distance-to-goal,
minimum-neighbor distance and initial goal distance. Input-boundary tests change
flags and labels while keeping visual/numerical observations fixed, and check
that Qwen text/tensors, robot observations and adjacency remain identical.

New training checkpoints record the text contract in their arguments. Evaluation
of a legacy checkpoint emits a warning and records the checkpoint and current
text contracts separately. Sanitizing its inputs is a distribution-shift
ablation, not a substitute for fresh training. Readiness conclusions from the
previous state-augmented evaluations are withdrawn for the task-only setup.

The dataset labels, collision metadata used to build training targets, APF
smoothing and preference supervision remain stored separately. The datasets do
not need recollection merely to remove the prompt leakage. The old experiment
source snapshots remain unchanged for reproducibility. No replacement training
or evaluation jobs were submitted as part of this audit.

Validation: task-prompt invariance tests, actual Qwen model-input invariance with
legacy/relational token layouts and default/custom instructions, existing TB3
loader/prompt/inference regressions, and the standalone TB3 loader smoke test.

The focused checks cover57 unique passing tests, plus the standalone TB3 loader
smoke test. The task-domain regression also keeps GTG instructions mentioning
"no static obstacles" distinct from AO instructions.
