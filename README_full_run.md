# Qwen full accuracy experiment, 2026-10-06

ALWAYS both `adi2440/tb3-isaac-vlcm` and `adi2440/tb3-isaac-avoid-obstacles-vlcm`.
Uses their verified signed APF shards, including 500 uploaded suboptimal 3-agent
GTG AO episodes, preserving the original and grouped scenario holdouts.
Fresh pretrained Qwen3-VL-2B + new LoRA/graph/query/heads/optimizer/scheduler;
no earlier critic checkpoint is loaded. VJEPA stays on standby.

75 epochs; 8 frames; 224x224 canonical images (processor grid checked on GPU);
5,000 clips per epoch; same 128 effective batch; A100 profile selects the fastest
safe microbatch/workers/checkpointing configuration. Candidate peaks must stay
under 80% allocated and 90% reserved VRAM. No profile weights are saved or reused.

21-bin soft categorical CE over [-1,1] remains the scalar potential objective.
Learned critic query replaces the final non-padding assistant-prompt token,
after video and state; final-token readout receives full causal context.
Agent attention plus masked feature maxima preserves individual hazard evidence.
Four auxiliary targets are log1p(goal_cost), agent_risk, obstacle_risk, collision;
Smooth-L1 continuous components and balanced collision BCE, summed with weight .5.
APF components ONLY supervise outputs, never feed model inputs. Inputs retain
the deployment-compatible nine-feature agent observations.

Paired endpoint CE and signed delta Smooth-L1 (weight 1, beta .02) use randomly
selected 1/2/4-frame gaps. Training-only event sampling keeps all decreases,
recoveries, or collision windows and keeps ordinary windows with probability 1/3.
Event detection uses absolute adjacent target changes >= .03. Training CE/delta
weights are 3 for paired endpoint changes or collision plateaus, 1 otherwise.
Both signs receive identical emphasis. Tail pairs retain terminal collisions.
Validation is unfiltered, gap 1, and unweighted for meaningful comparison.
Conditional empirical Lipschitz weight stays 1; changed conditions/collision
status are exempt from that penalty but retain endpoint/delta/auxiliary losses.

Epoch validation logs drop/recovery direction recall, false-drop rate, change MAE,
component losses, memory and throughput. Full episode inference via the existing
shell script measures amplitude and delay; it uses causal complete windows only,
without future pairs, event filtering, or auxiliary label requirements.

Ablations in ablation_plan.json are evaluated after full-run checkpoints exist.
Remove one feature per fresh matched run, keeping data, split, clip length,
epoch budget and effective batch fixed. Keep cheaper changes only if scalar
accuracy, collision penalties, drop amplitude/latency and recovery recall remain
acceptable. Do not select by categorical CE alone. No ablation jobs are submitted
before the full-run reference is validated.
