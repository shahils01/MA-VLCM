"""Evaluate formation actors trained with the fully frozen Python VLCM critic."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from evaluate_python_vlcm_actor import load_pair
from python_mappo import make_env, python_collection_boundary

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT.parent / "MARL_data_collection/on-policy/ckpts/formation_control/mappo_python_vlcm_frozen_seed1_20261101_160325"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_provenance(run, checkpoints):
    contract = json.loads((run / "run.json").read_text())
    source_path = Path(contract["args"]["critic_checkpoint"])
    assert sha256(source_path) == contract["source_critic_sha256"]
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    assert Path(source["run"]["args"]["dataset"]).name == "polygon_formation_python_env"
    assert sum(len(rows) for rows in source["run"]["splits"].values()) == 3000
    assert contract["args"]["actor_from_scratch"]
    assert contract["args"]["freeze_critic"]
    assert contract["args"]["frozen_python_supervised"]
    assert not contract["args"]["value_calibration"]
    assert contract["critic_trainable_parameter_count"] == 0
    env_path = ROOT.parent / "MARL_data_collection/RL_bound_overwatch/envs/FormationControlEnv.py"
    assert sha256(env_path) == contract["environment_sha256"]
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(row.get("train/critic_updated", 0) == 0 for row in metrics)
    checks = []
    for checkpoint in checkpoints:
        critic = torch.load(checkpoint / "critic.pt", map_location="cpu", weights_only=True)
        state = critic["model_state"]
        assert all(torch.equal(v, state["core." + k]) for k, v in source["model_state"].items())
        assert all(torch.count_nonzero(v) == 0 for k, v in state.items() if k.startswith("context_embedding."))
        optimizers = torch.load(checkpoint / "optimizers.pt", map_location="cpu", weights_only=True)
        assert optimizers["critic_optimizer"] is None
        checks.append(dict(checkpoint=str(checkpoint),steps=critic["steps"],
                           all_supervised_critic_tensors_exactly_unchanged=True,
                           matched_tensor_count=len(source["model_state"]),
                           context_adapter_exactly_zero=True,critic_optimizer_absent=True))
    return dict(passed=True,source_critic=str(source_path),source_critic_sha256=sha256(source_path),
                supervised_dataset_episodes=3000,matching_training_environment_sha256=sha256(env_path),
                no_recorded_critic_updates=True,actor_initialized_from_scratch=True,checkpoints=checks)


def rollout(actor, args, count, seed, root, label, checkpoint_steps, make_video):
    env = make_env(args, count)
    obs, _, _ = env.reset(seed=seed)
    initial_positions = env.pos.copy()
    initial_yaws = env.theta.copy()
    name = f"{label}_agents_{count:02d}_seed_{seed}"
    positions, yaws, errors = [env.pos.copy()], [env.theta.copy()], [env._slot_errors().copy()]
    actions, per_agent_rewards, team_rewards = [], [], []
    collisions, video_frames = 0, 0
    video_path = root / "videos" / f"{name}.mp4"
    writer = imageio.get_writer(video_path, fps=1/(args.control_dt*args.camera_stride),
                               codec="libx264",quality=8,ffmpeg_log_level="error") if make_video else None

    def frame(outcome):
        nonlocal video_frames
        image = Image.fromarray(env.render(size=256,view_size_m=args.view_size_m,tile_floor=args.tile_floor))
        canvas = Image.new("RGB", (512,576), "white")
        canvas.paste(image.resize((512,512),Image.Resampling.NEAREST),(0,64))
        draw = ImageDraw.Draw(canvas)
        draw.text((10,8),f"{label}: {checkpoint_steps:,} training steps | {count} agents | seed {seed}",fill="black")
        draw.text((10,25),f"t={env.steps*args.control_dt:.2f}s | step {env.steps}/{args.env_max_steps} | {outcome}",fill="black")
        draw.text((10,42),f"Worst slot error {env._slot_errors().max():.3f} m / {args.formation_tolerance:.3f} m | hold {env.stable_team_steps}/{args.formation_hold_steps}",fill="black")
        writer.append_data(np.asarray(canvas))
        video_frames += 1

    if writer:
        frame("initial")
    try:
        for _ in range(args.env_max_steps):
            with torch.inference_mode():
                action, _, _ = actor(obs,np.zeros((6,1,args.hidden_size),np.float32),
                                     np.ones((6,1),np.float32),deterministic=True)
            action = action.cpu().numpy()
            obs, _, reward, dones, infos, _ = env.step(action)
            boundary = python_collection_boundary(env,args)
            success = bool(infos[0]["stable_formation_achieved"] and not boundary)
            done = bool(dones[0] or boundary)
            reason = "out_of_bounds" if boundary else "success" if success else "timeout" if done else "running"
            positions.append(env.pos.copy())
            yaws.append(env.theta.copy())
            errors.append(env._slot_errors().copy())
            actions.append(action)
            per_agent_rewards.append(reward[:count,0].copy())
            team_rewards.append(float(reward[:count].mean()))
            collisions += bool(infos[0]["collision_detected"])
            if writer and (env.steps % args.camera_stride == 0 or done):
                frame(reason)
            if done:
                break
    finally:
        if writer:
            writer.close()
    np.savez_compressed(root/"traces"/f"{name}.npz",positions=np.asarray(positions),
                        yaws=np.asarray(yaws),slot_errors=np.asarray(errors),
                        actions=np.asarray(actions),per_agent_rewards=np.asarray(per_agent_rewards),
                        team_rewards=np.asarray(team_rewards),seed=seed,dt=args.control_dt,
                        slot_assignment=env.slot_assignment)
    result = dict(checkpoint=label,training_steps=checkpoint_steps,num_agents=count,seed=seed,
                  success=success,outcome=reason,steps=env.steps,seconds=env.steps*args.control_dt,
                  team_return=float(np.sum(team_rewards)),
                  discounted_team_return=float(np.sum(np.asarray(team_rewards)*args.gamma**np.arange(len(team_rewards)))),
                  collision_steps=collisions,initial_max_slot_error_m=float(np.max(errors[0])),
                  final_max_slot_error_m=float(np.max(errors[-1])),final_formation_mse_m2=float(env._formation_error()),
                  stable_hold_steps=int(env.stable_team_steps),video=str(video_path.relative_to(root)) if writer else "",
                  video_frames=video_frames,trace=f"traces/{name}.npz")
    env.close()
    return result, initial_positions, initial_yaws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--steps",type=int,default=1401600)
    parser.add_argument("--episodes-per-team",type=int,default=50)
    parser.add_argument("--videos-per-team",type=int,default=2)
    parser.add_argument("--seed",type=int,default=97000001)
    parser.add_argument("--output-dir",type=Path,default=ROOT.parent/"MARL_data_collection/on-policy/eval_outputs/formation_frozen_python3000_step1401600")
    opt = parser.parse_args()
    if opt.episodes_per_team<1 or not 0<=opt.videos_per_team<=opt.episodes_per_team:
        parser.error("Require positive episode count and 0 <= videos <= episodes per team")
    torch.set_num_threads(1)
    run, root = opt.run.resolve(), opt.output_dir.resolve()
    checkpoints = {f"step{opt.steps}":run/"checkpoints"/f"step_{opt.steps:09d}","best":run/"best"}
    provenance = check_provenance(run,list(checkpoints.values()))
    root.mkdir(parents=True,exist_ok=False)
    for name in ("checkpoints","videos","traces"):
        (root/name).mkdir()
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    shutil.copy2(run/"run.json",root/"training_run.json")
    rows, hashes, starts = [], {}, {}
    for label,checkpoint in checkpoints.items():
        snapshot = root/"checkpoints"/label
        snapshot.mkdir()
        for name in ("actor.pt","critic.pt","training_metadata.json"):
            shutil.copy2(checkpoint/name,snapshot/name)
        actor, critic, args, metadata, digest = load_pair(snapshot,torch.device("cpu"))
        actor.requires_grad_(False).eval()
        critic.requires_grad_(False).eval()
        hashes[label] = dict(actor_sha256=digest[0],critic_sha256=digest[1])
        for count in (3,4,5,6):
            for index in range(opt.episodes_per_team):
                seed = opt.seed + 10000*count + index
                assert seed not in {args.eval_seed+1000*count+i for i in range(args.eval_episodes)}
                row,positions,yaws = rollout(actor,args,count,seed,root,label,
                                             metadata["total_environment_steps"],index<opt.videos_per_team)
                key = (count,seed)
                if key in starts:
                    assert np.array_equal(positions,starts[key][0]) and np.array_equal(yaws,starts[key][1])
                else:
                    starts[key] = (positions,yaws)
                rows.append(row)
                if (index+1)%10==0:
                    selected = [r for r in rows if r["checkpoint"]==label and r["num_agents"]==count]
                    print(f"{label} n={count}: {index+1}/{opt.episodes_per_team}, successes={sum(r['success'] for r in selected)}",flush=True)
        assert sha256(snapshot/"actor.pt")==digest[0] and sha256(snapshot/"critic.pt")==digest[1]
        assert not any(p.requires_grad or p.grad is not None for p in actor.parameters())
        del actor, critic
    with (root/"episodes.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for label in checkpoints:
        summaries[label] = {}
        for count in (3,4,5,6):
            selected = [r for r in rows if r["checkpoint"]==label and r["num_agents"]==count]
            summaries[label][str(count)] = dict(episodes=len(selected),successes=sum(r["success"] for r in selected),
                success_rate=float(np.mean([r["success"] for r in selected])),outcomes=dict(Counter(r["outcome"] for r in selected)),
                mean_team_return=float(np.mean([r["team_return"] for r in selected])),
                mean_steps=float(np.mean([r["steps"] for r in selected])),
                mean_final_max_slot_error_m=float(np.mean([r["final_max_slot_error_m"] for r in selected])),
                episodes_with_collision=sum(r["collision_steps"]>0 for r in selected))
    report = dict(run=str(run),provenance_verified=True,episodes_per_team=opt.episodes_per_team,
                  seed=opt.seed,paired_initial_conditions=True,deterministic_actions=True,
                  policy_only=True,critic_used_to_choose_actions=False,model_updates=False,
                  episode_max_steps=args.env_max_steps,formation_tolerance_m=args.formation_tolerance,
                  formation_hold_steps=args.formation_hold_steps,checkpoint_hashes=hashes,
                  by_checkpoint=summaries,episodes=rows)
    (root/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for i,(label,groups) in enumerate(summaries.items()):
        x = np.arange(4)+(i-.5)*.35
        axes[0].bar(x,[100*groups[str(n)]["success_rate"] for n in (3,4,5,6)],width=.35,label=label)
        axes[1].bar(x,[groups[str(n)]["mean_final_max_slot_error_m"] for n in (3,4,5,6)],width=.35,label=label)
        axes[2].bar(x,[groups[str(n)]["mean_team_return"] for n in (3,4,5,6)],width=.35,label=label)
    for ax,title in zip(axes,("Success rate (%)","Mean final worst-robot slot error (m)","Mean native team return")):
        ax.set(title=title,xlabel="Number of robots",xticks=np.arange(4),xticklabels=[3,4,5,6])
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylim(0,100)
    axes[0].legend()
    axes[1].axhline(args.formation_tolerance,color="red",linestyle="--",label="Tolerance")
    fig.savefig(root/"policy_performance.png",dpi=160)
    plt.close(fig)
    lines = ["# Formation policy trained with the fully frozen Python-3000 VLCM critic","",
             f"{opt.episodes_per_team} fresh seeds per team size and checkpoint, with matched initial conditions. Deterministic actor inference; no critic inference or training is needed to choose actions. Native training configuration, including the 400-step horizon, workspace boundary rule, 0.075 m slot tolerance and 20-step hold requirement, is retained.","",
             "`provenance.json` verifies the 3,000-episode source critic, all 41 source tensors unchanged, zero context adapter, no critic optimizer and no recorded critic updates. The `best` checkpoint was selected during training on separate fixed evaluation seeds, not on this new evaluation.","",
             "| Checkpoint | Agents | Successes | Success rate | Mean steps | Mean final worst slot error (m) | Mean team return |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label,groups in summaries.items():
        for count,metrics in groups.items():
            lines.append(f"| {label} | {count} | {metrics['successes']}/{metrics['episodes']} | {100*metrics['success_rate']:.1f}% | {metrics['mean_steps']:.1f} | {metrics['mean_final_max_slot_error_m']:.3f} | {metrics['mean_team_return']:.2f} |")
    lines += ["","Videos show the first two predetermined seeds per team and checkpoint, without filtering outcomes. All native per-agent rewards, applied actions and positions are saved in `traces/`; per-episode scores are in `episodes.csv`."]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summaries,indent=2),flush=True)
    print(f"Saved policy evaluation to {root}",flush=True)


if __name__ == "__main__":
    main()
