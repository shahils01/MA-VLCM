"""Collect matched ID/OOD rollouts, score frozen critics, and render comparisons.

Collection phases run in separate processes to keep the two frozen environment
packages isolated. No actor or critic is updated. See README_ROLLOUT_EVALUATION.md.
"""
import argparse
from collections import Counter, defaultdict
import csv
from io import BytesIO
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/eval_rollouts"
MODELS = ("joint_without_obstacles", "joint_with_obstacles")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_records(root):
    records = []
    for path in sorted((root / "episodes").rglob("episode_*.summary.json")):
        row = json.loads(path.read_text())
        row["summary_path"] = str(path)
        records.append(row)
    return records


def distribution(model, case, n):
    shifts = []
    if n not in (3, 5):
        shifts.append("agent_count")
    if case not in {"formation_clean", "formation_recovery_clean_pair"} and case.startswith("formation_"):
        shifts.append(case.removeprefix("formation_"))
    if case == "navigation_with_obstacles" and model == "joint_without_obstacles":
        shifts.append("obstacles_added")
    if case == "navigation_without_obstacles" and model == "joint_with_obstacles":
        shifts.append("obstacles_removed")
    return "OOD_" + "_and_".join(shifts) if shifts else "ID_best_actor_subgroup"


def navigation_start_audit(positions):
    xy = np.asarray(positions, dtype=float)[:, :2]
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    result = dict(start_centers_inside_8m=bool((np.abs(xy) <= 4. + 1e-6).all()),
                  min_start_center_separation_m=float(distance.min()))
    if not result["start_centers_inside_8m"] or result["min_start_center_separation_m"] < 1.5 - 1e-6:
        raise ValueError(f"Navigation start contract failed: {result}")
    return result


def collect_navigation(root, samples):
    import torch
    tools = ROOT.parent / "MARL_data_collection/dataset_tools"
    sys.path.insert(0, str(tools))
    import collect_python_gotogoal_dataset as collector
    torch.set_num_threads(1)
    for ci, condition in enumerate(("without_obstacles", "with_obstacles")):
        frozen = ROOT.parent / "manyagent_gotogoal_10m_3000/provenance/clean" / condition
        plan = json.loads((frozen / "collection_plan.json").read_text())
        collector.verify_sources(frozen, plan)
        config = dict(plan["environment_config"], start_grid_size=8., min_agent_dist=1.5)
        if config["grid"] != [[-5., 5.], [-5., 5.]]:
            raise ValueError("Expected a 10 x 10 m navigation arena")
        policy = dict(plan["checkpoints"][0], checkpoint=str(frozen / "checkpoints/best/actor.pt"))
        policy.pop("checkpoint_relative", None)
        case = "navigation_" + condition
        output = root / "episodes" / case
        output.mkdir(parents=True, exist_ok=True)
        for n in (3, 5, 4, 6):
            env, actor, args = collector.load_runtime(frozen, config, n, policy)
            for index in range(samples):
                job = dict(episode_id=800000 + ci*10000 + n*100 + index,
                           seed=117000000 + ci*10000 + n*100 + index,
                           noise_seed=127000000 + ci*10000 + n*100 + index,
                           stage="best", num_agents=n, noise="clean", worker_id=0)
                _, summary_path = collector.paths(output, job)
                if summary_path.exists():
                    continue
                result = collector.rollout(output, job, policy, config, env, actor, args)
                with np.load(result["control_trace"]) as trace:
                    audit = navigation_start_audit(trace["positions"][0])
                result.update(eval_case=case, condition=condition, start_audit=audit,
                              collection_note="Fresh native reset; fixed deterministic best actor; no success filtering")
                write_json(summary_path, result)
                print(case, n, index, result["termination_reason"], audit, flush=True)


def formation_setup(root):
    frozen = root / "assets/formation"
    sys.path.insert(0, str(frozen / "source"))
    import collect_python_formation_dataset as collector
    from onpolicy_actor import load_actor
    import torch
    plan = json.loads((frozen / "collection_plan.json").read_text())
    for name, expected in plan["source_hashes"].items():
        if collector.sha256(frozen/name) != expected:
            raise ValueError(f"Formation source hash mismatch: {name}")
    for item in plan["checkpoints"]:
        path = frozen / "checkpoints" / item["stage"] / "actor.pt"
        if collector.sha256(path) != item["sha256"]:
            raise ValueError("Formation actor hash mismatch")
    policy = dict(next(p for p in plan["checkpoints"] if p["stage"] == "successful"))
    policy["checkpoint"] = str(frozen / "checkpoints/successful/actor.pt")
    actor = load_actor(Path(policy["checkpoint"]), torch.device("cpu")).eval()
    torch.set_num_threads(1)
    return frozen, collector, actor, policy


def sample_formation_obstacles(env, seed):
    """Fixed reset-only layout; keep starts and assigned initial slots clear."""
    rng = np.random.default_rng(seed + 17000000)
    goals = env._compute_swarm_centroid() + env._assigned_formation_config()
    protected = np.concatenate((env.pos, goals))
    obstacles = []
    for _ in range(10000):
        candidate = np.r_[rng.uniform(-1.2, 1.2, 2), rng.uniform(.09, .16)]
        if np.min(np.linalg.norm(protected - candidate[:2], axis=1)) < candidate[2] + env.rob_diam/2 + .08:
            continue
        if any(np.linalg.norm(candidate[:2]-o[:2]) < candidate[2]+o[2]+.1 for o in obstacles):
            continue
        obstacles.append(candidate)
        if len(obstacles) == 3:
            return np.asarray(obstacles)
    raise ValueError("Could not sample collision-free formation obstacle layout")


def formation_episode(frozen, collector, actor, policy, output, job, case, render=True):
    """Native formation dynamics; physical-obstacle mode adds an explicit failure event."""
    import torch
    env = collector.environment(frozen, job["num_agents"])
    n = env.active_num_agents
    obs, _, _ = env.reset(seed=job["seed"])
    initial = env._slot_errors().astype(np.float32)
    initial_positions = env.pos.copy()
    obstacles = sample_formation_obstacles(env, job["seed"]) if "obstacles" in case else np.empty((0, 3))
    physical = case == "formation_physical_obstacles"
    noisy = case == "formation_noise_recovery"
    rng = np.random.default_rng(job["noise_seed"])
    noise_state = np.zeros((n, 2), np.float32)
    frames, trace, interval = [], defaultdict(list), []
    zeros = np.zeros((6, 2), np.float32)
    reason, success, touched = "", False, False

    def capture(step, reward, actions, policy_action, delta, velocities, rewards):
        for key, value in dict(positions=env.pos, yaws=env.theta, applied_actions=actions,
                               policy_actions=policy_action, sampled_noise=delta, rewards=rewards,
                               velocities=velocities).items():
            trace[key].append(np.asarray(value).copy())
        if render and (step % 2 == 0 or reason):
            f = collector.frame_from_env(env, step=step, reward=reward, interval=interval,
                velocities=velocities, applied=actions, policy=policy_action, noise=delta, reason=reason, job=job)
            f.sample_index = len(frames)
            if len(obstacles):
                image = Image.open(BytesIO(f.png)).convert("RGB")
                draw = ImageDraw.Draw(image)
                # Native camera covers 3.048 m; draw obstacles before black robot footprints.
                scale = image.width / 3.048
                for x, y, radius in obstacles:
                    px, py = (x/3.048+.5)*image.width, (.5-y/3.048)*image.height
                    r = radius*scale
                    draw.ellipse((px-r, py-r, px+r, py+r), fill="#505050", outline="#aaaaaa")
                r = env.rob_diam/2*scale
                for x, y in env.pos:
                    px, py = (x/3.048+.5)*image.width, (.5-y/3.048)*image.height
                    draw.ellipse((px-r, py-r, px+r, py+r), fill="black")
                stream = BytesIO(); image.save(stream, format="PNG")
                f.png = stream.getvalue()
            f.task_metadata.update(eval_case=case, obstacles=obstacles.tolist(),
                obstacle_semantics="contact terminates with native collision coefficient penalty" if physical else "visual only; native reward/dynamics unchanged",
                noise_window_control_steps=[41, 100] if noisy else None)
            frames.append(f)
            interval.clear()

    capture(0, 0., zeros, zeros, zeros, np.zeros((n, 2)), np.zeros(6))
    for step in range(1, env.max_steps+1):
        with torch.inference_mode():
            policy_action = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        policy_action[n:] = 0.
        delta = np.zeros_like(policy_action)
        if noisy and 41 <= step <= 100:
            # Correlated late burst differs from the independent early training bursts.
            noise_state = .8*noise_state + rng.normal(size=(n, 2))*np.array([.20, .45])
            envelope = min((step-40)/10, (101-step)/10, 1.)
            delta[:n] = noise_state*envelope
        applied = np.clip(policy_action + delta, -1., 1.).astype(np.float32)
        previous = env.pos.copy()
        obs, _, rewards, dones, infos, _ = env.step(applied)
        rewards = rewards[:, 0].copy()
        contact = bool(len(obstacles) and np.any(np.linalg.norm(env.pos[:, None]-obstacles[None, :, :2], axis=-1)
                                               <= env.rob_diam/2 + obstacles[None, :, 2]))
        touched |= contact
        out = collector.outside(env)
        success = bool(infos[0]["stable_formation_achieved"] and not out and not (physical and contact))
        if physical and contact:
            # Explicit novel task rule, separate from the visual-clutter control.
            rewards[:n] -= env.collision_penalty_w
            if infos[0].get("first_success_step") == step:
                rewards[:n] -= env.success_reward
            reason = "obstacle_collision"
        elif out or dones[0]:
            reason = "out_of_bounds" if out else "success" if success else "timeout"
        reward = float(rewards[:n].mean())
        interval.append(reward)
        capture(step, reward, applied, policy_action, delta, (env.pos-previous)/env.dt, rewards)
        if reason:
            break
    result = dict(job, eval_case=case, success=success, termination_reason=reason,
                  control_steps=env.steps, sim_time_s=env.steps*env.dt,
                  episode_return=float(np.asarray(trace["rewards"])[1:, :n].mean(axis=1).sum()),
                  obstacles=obstacles.tolist(), obstacle_contact=touched,
                  initial_positions=initial_positions.tolist(), max_final_slot_error_m=float(env._slot_errors().max()))
    if not render:
        return result
    tar_path, summary_path = collector.episode_paths(output, "formation_control", n, 0, job["episode_id"])
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    writer = collector.WebDatasetTarWriter(tar_path)
    try:
        for i, frame in enumerate(frames):
            sample = collector.build_sample(frame, initial_distances=initial,
                episode_id=job["episode_id"], episode_seed=job["seed"], worker_id=0,
                task_name="Python-FormationControl-v0", task_domain="formation_control",
                task_instruction=collector.formation_task_instruction("polygon", spacing_m=.5),
                checkpoint=policy, control_hz=20., capture_hz=10., episode_success=success, is_last=i==len(frames)-1)
            sample["state.json"].update(environment="python_formation_control", noise_profile=job["noise"])
            sample["state.json"]["episode_meta"].update(terminated=bool(reason and reason!="timeout" and i==len(frames)-1),
                truncated=bool(reason=="timeout" and i==len(frames)-1))
            writer.write(f"ep{job['episode_id']:06d}_step{i:05d}", sample)
        writer.close()
    except BaseException:
        writer.abort(); raise
    trace_path = tar_path.with_suffix(".control.npz")
    np.savez_compressed(trace_path, **{k:np.asarray(v) for k,v in trace.items()},
                        sample_control_steps=[f.physics_step for f in frames], dt=env.dt)
    result.update(samples=len(frames), schema=collector.DATASET_SCHEMA, num_agents=n, checkpoint=policy,
                  task_domain="formation_control", shard=str(tar_path), control_trace=str(trace_path),
                  tar_sha256=collector.sha256(tar_path), summary_path=str(summary_path))
    write_json(summary_path, result)
    return result


def collect_formation(root, samples):
    frozen, collector, actor, policy = formation_setup(root)
    for n in (3, 5, 4, 6):
        for index in range(samples):
            job = dict(episode_id=900000+n*100+index, seed=118000000+n*100+index,
                       noise_seed=128000000+n*100+index, num_agents=n, stage="successful", noise="clean", worker_id=0)
            for case in ("formation_clean", "formation_physical_obstacles") + (("formation_visual_obstacles",) if n in (3, 5) else ()):
                output = root/"episodes"/case
                _, summary_path = collector.episode_paths(output, "formation_control", n, 0, job["episode_id"])
                if summary_path.exists():
                    continue
                row = formation_episode(frozen, collector, actor, policy, output, job, case)
                print(case, n, index, row["termination_reason"], flush=True)
    # Requested recovery examples are explicitly outcome-conditioned. Save every
    # rejected attempt; do not include this subset in population success estimates.
    for n in (3, 5):
        attempt_path = root/f"recovery_attempts_agents_{n:02d}.json"
        attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else []
        accepted = [a for a in attempts if a["accepted"]]
        for attempt in range(len(attempts), 250):
            if len(accepted) >= samples:
                break
            job = dict(episode_id=950000+n*1000+attempt, seed=119000000+n*1000+attempt,
                       noise_seed=129000000+n*1000+attempt, num_agents=n, stage="successful", noise="late_correlated_burst", worker_id=0)
            clean = formation_episode(frozen, collector, actor, policy, root, dict(job, noise="clean"), "formation_clean", render=False)
            noisy = formation_episode(frozen, collector, actor, policy, root, job, "formation_noise_recovery", render=False)
            accepted_now = bool(clean["success"] and noisy["success"] and noisy["control_steps"] > 100)
            attempt_row = dict(seed=job["seed"], clean=clean, noisy=noisy, accepted=accepted_now)
            attempts.append(attempt_row)
            write_json(attempt_path, attempts)
            if accepted_now:
                accepted.append(attempt_row)
                print("recovery accepted", n, job["seed"], "attempt", attempt+1, flush=True)
        for accepted_row in accepted[:samples]:
            job = {k:accepted_row["noisy"][k] for k in ("episode_id", "seed", "noise_seed", "num_agents", "stage", "noise", "worker_id")}
            for case in ("formation_recovery_clean_pair", "formation_noise_recovery"):
                output = root/"episodes"/case
                _, summary_path = collector.episode_paths(output, "formation_control", n, 0, job["episode_id"])
                if not summary_path.exists():
                    row = formation_episode(frozen, collector, actor, policy, output,
                        dict(job, noise=job["noise"] if case=="formation_noise_recovery" else "clean"), case)
                    print(case, n, row["termination_reason"], flush=True)
        if len(accepted) < samples:
            print(f"Only {len(accepted)} successful recovery pairs for {n} agents after 250 attempts; all attempts retained", flush=True)


def verify_training(root, checkpoints):
    report = {}
    for name in ("formation_only",) + MODELS:
        directory = checkpoints/name
        run = json.loads((directory/"run.json").read_text())
        selection = json.loads((directory/"selection_report.json").read_text())
        counts = {task: info["by_num_agents"] for task, info in selection["tasks"].items()}
        if any(set(info) != {"3", "5"} for info in counts.values()):
            raise ValueError(f"Unexpected training teams for {name}: {counts}")
        # Verify the actual train split too, rather than relying on launch flags.
        team_counts = Counter()
        for path in run["splits"]["train"]:
            team = next(part for part in Path(path).parts if part.startswith("agents_"))
            team_counts[team] += 1
        if set(team_counts) != {"agents_03", "agents_05"}:
            raise ValueError(f"Unexpected train split: {team_counts}")
        report[name] = dict(selected_counts=counts, actual_train_split_team_counts=dict(team_counts))
    write_json(root/"training_agent_audit.json", report)


def audit_rollouts(root, samples):
    """Check archive targets independently against the full control trace."""
    from animate_checkpoint import load_frames
    records = read_records(root)
    groups = Counter((r["eval_case"], r["num_agents"]) for r in records)
    expected = {(case, n) for case in ("formation_clean", "navigation_without_obstacles", "navigation_with_obstacles")
                for n in (3, 4, 5, 6)}
    expected |= {("formation_physical_obstacles", n) for n in (3, 4, 5, 6)}
    expected |= {(case, n) for case in ("formation_visual_obstacles",
                 "formation_recovery_clean_pair", "formation_noise_recovery") for n in (3, 5)}
    if set(groups) != expected or any(groups[key] != samples for key in expected):
        raise ValueError(f"Incomplete collection: expected {samples} episodes for each of {len(expected)} cells, got {groups}")
    checked = []
    lookup = {(r["eval_case"], r["num_agents"], r["seed"]):r for r in records}
    for row in records:
        path = Path(row["shard"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["tar_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {path}")
        frames, _ = load_frames(path, dict(gamma=.99, return_source="control", image_hz=10.))
        with np.load(row["control_trace"]) as trace:
            rewards = trace["rewards"][:, :row["num_agents"]].mean(axis=1)
            dt = float(trace["dt"])
            positions = trace["positions"]
            if row["eval_case"].startswith("navigation_"):
                navigation_start_audit(positions[0])
            native_return = np.zeros(len(rewards))
            for t in range(len(rewards)-2, -1, -1):
                native_return[t] = float(rewards[t+1]) + .99*native_return[t+1]
            for f in frames:
                step = round(f["time_s"]/dt)
                if not np.isclose(f["reward"], rewards[step], atol=1e-5):
                    raise ValueError(f"Incoming reward mismatch: {path} at {step}")
                if f["target_return"] is not None and not np.isclose(f["target_return"], native_return[step], atol=1e-4):
                    raise ValueError(f"Full-control return mismatch: {path} at {step}")
            if row["eval_case"] == "formation_visual_obstacles":
                clean = lookup[("formation_clean", row["num_agents"], row["seed"])]
                with np.load(clean["control_trace"]) as baseline:
                    for key in ("positions", "rewards", "applied_actions"):
                        if not np.array_equal(trace[key], baseline[key]):
                            raise ValueError("Visual-only obstacles changed native dynamics/rewards")
            if row["eval_case"] == "formation_physical_obstacles":
                clean = lookup[("formation_clean", row["num_agents"], row["seed"])]
                with np.load(clean["control_trace"]) as baseline:
                    length = len(trace["positions"])
                    for key in ("positions", "applied_actions"):
                        if not np.array_equal(trace[key], baseline[key][:length]):
                            raise ValueError("Physical-obstacle prefix differs from matched native rollout")
                    expected_rewards = baseline["rewards"][:length].copy()
                    if row["termination_reason"] == "obstacle_collision":
                        expected_rewards[-1, :row["num_agents"]] -= 10.
                        if clean["termination_reason"] == "success" and clean["control_steps"] == row["control_steps"]:
                            expected_rewards[-1, :row["num_agents"]] -= 50.
                        obstacles = np.asarray(row["obstacles"])
                        contact = np.linalg.norm(positions[-1, :, None]-obstacles[None, :, :2], axis=-1) <= .205/2+obstacles[None, :, 2]
                        if not contact.any():
                            raise ValueError("Obstacle failure without disk contact")
                    if not np.allclose(trace["rewards"], expected_rewards, atol=1e-5):
                        raise ValueError("Physical-obstacle reward rule does not match native prefix plus documented penalty")
            if row["eval_case"] == "formation_noise_recovery":
                clean = lookup[("formation_recovery_clean_pair", row["num_agents"], row["seed"])]
                with np.load(clean["control_trace"]) as baseline:
                    if not np.array_equal(positions[:41], baseline["positions"][:41]):
                        raise ValueError("Noise pair diverged before intervention")
                if not row["success"] or not clean["success"] or not np.any(trace["sampled_noise"][41:101]):
                    raise ValueError("Expected an actual perturbed successful recovery pair")
                if np.any(trace["sampled_noise"][101:]):
                    raise ValueError("Noise continued after intervention window")
        checked.append(dict(case=row["eval_case"], num_agents=row["num_agents"], seed=row["seed"], frames=len(frames)))
    report = dict(passed=True, episodes=len(records), cells=len(groups), episodes_per_cell=samples,
        checks=["archive SHA256", "incoming native reward", "independent full-control return recurrence",
                "navigation 8m/1.5m initial geometry", "visual-only dynamics equivalence",
                "physical-obstacle matched native prefix, disk contact and explicit reward rule",
                "noise-pair identical prefix, intervention, clean recovery"], checked=checked)
    write_json(root/"rollout_audit.json", report)
    print(f"Audited {len(records)} episodes across {len(groups)} cells", flush=True)


def score(root, checkpoints, device_name):
    import torch
    from torch.utils.data import DataLoader
    from data_utils import (read_episode, prepare_episode, frame_indices, cache_features,
                            feature_cache_path, FormationClipDataset, collate_clips)
    from model import ModelConfig, SmolVLMBackbone, MultimodalValueModel
    torch.set_num_threads(4)
    verify_training(root, checkpoints)
    records = read_records(root)
    if not records:
        raise ValueError("Collect rollouts first")
    device = torch.device(device_name)
    saved = {name:torch.load(checkpoints/name/"best.pt", map_location="cpu", weights_only=True) for name in MODELS}
    provenance = dict(checkpoints={name:dict(path=str((checkpoints/name/"best.pt").resolve()),
        sha256=hashlib.sha256((checkpoints/name/"best.pt").read_bytes()).hexdigest(),
        epoch=checkpoint["epoch"], training_signature=checkpoint["run"]["signature"])
        for name, checkpoint in saved.items()}, model_config=saved[MODELS[0]]["model_config"])
    provenance_path = root/"scoring_provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Existing predictions belong to different checkpoint weights")
    write_json(provenance_path, provenance)
    cfg = ModelConfig(**saved[MODELS[0]]["model_config"])
    if any(s["model_config"] != saved[MODELS[0]]["model_config"] for s in saved.values()):
        raise ValueError("Critics need matching feature configurations for a shared cache")
    backbone = SmolVLMBackbone(cfg, device).eval()
    cache = root/"cache"
    models = {}
    for name, checkpoint in saved.items():
        model = MultimodalValueModel(cfg, device, load_backbone=False).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        models[name] = model.eval()
    results = []
    for index, row in enumerate(records):
        path = Path(row["shard"])
        destination = root/"predictions"/row["eval_case"]/f"agents_{row['num_agents']:02d}"/f"{path.stem}.json"
        if destination.exists():
            previous = json.loads(destination.read_text())
            for name in MODELS:
                previous["metrics"][name]["distribution"] = distribution(name, row["eval_case"], row["num_agents"])
            write_json(destination, previous)
            results.append(previous); continue
        raw = read_episode(path, row["summary_path"], cache/"index")
        args = saved[MODELS[0]]["run"]["args"]
        if raw["complete"]:
            ep = prepare_episode(raw, args["gamma"], args["return_source"], args["image_hz"])
        else:
            selected = frame_indices(raw, args["image_hz"])
            ep = dict(raw, selected_indices=selected, returns=torch.zeros(len(selected)))
        with torch.inference_mode():
            cache_features([ep], backbone, cfg, cache/"features")
        dataset = FormationClipDataset([ep], cfg.clip_len, 1, cache/"features", cfg)
        values = {name:[] for name in MODELS}
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=64, collate_fn=collate_clips):
                batch.pop("metadata")
                batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                # Training evaluates the heads under BF16 on CUDA as well.
                for name, model in models.items():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                        values[name].extend(model(batch).float().cpu().tolist())
        times = ep["timestamps"][ep["selected_indices"]].tolist()
        target = ep["returns"].tolist() if ep["complete"] else [None]*len(times)
        valid = len(times)-int(ep["complete"])
        metrics = {}
        for name, predicted in values.items():
            metrics[name] = dict(distribution=distribution(name, row["eval_case"], row["num_agents"]),
                                 scored_frames=valid if ep["complete"] else 0)
            if ep["complete"] and valid:
                error = np.asarray(predicted[:valid])-np.asarray(target[:valid])
                metrics[name].update(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                    initial_predicted_return=float(predicted[0]), initial_recorded_return=float(target[0]))
        result = dict(episode=row, times=times, target_return=target, predicted_return=values,
                      metrics=metrics, complete=ep["complete"], gamma=args["gamma"],
                      note="Single realized return, not repeated-continuation expected value; true-terminal anchors excluded; no timeout-zero substitution")
        write_json(destination, result)
        results.append(result)
        write_json(root/"evaluation_results.json", results)
        print(f"Scored {index+1}/{len(records)} {row['eval_case']} n={row['num_agents']} {path.stem}", flush=True)
    write_json(root/"evaluation_results.json", results)
    for result in results:
        row = result["episode"]
        path = root/"predictions"/row["eval_case"]/f"agents_{row['num_agents']:02d}"/(Path(row["shard"]).stem+".csv")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "recorded_return", *MODELS, "true_terminal", "included_in_error"])
            for i, time in enumerate(result["times"]):
                terminal = bool(result["complete"] and i==len(result["times"])-1)
                writer.writerow([time, result["target_return"][i],
                    *[result["predicted_return"][name][i] for name in MODELS], terminal,
                    bool(result["complete"] and not terminal)])


def plot_panel(draw, bounds, times, series, cursor, ylabel):
    x0, y0, x1, y1 = bounds
    finite = [v for _, values, _ in series for v in values if v is not None and np.isfinite(v)]
    low, high = (min(finite), max(finite)) if finite else (0., 1.)
    margin = max((high-low)*.12, .03); low -= margin; high += margin
    left, right, top, bottom = x0+53, x1-12, y0+32, y1-24
    draw.text((x0+4, y0+5), ylabel, fill="black")
    for k in range(4):
        value = low+(high-low)*k/3
        y = bottom-(bottom-top)*k/3
        draw.line((left, y, right, y), fill="#dddddd")
        draw.text((x0+2, y-5), f"{value:.2f}", fill="#555555")
    duration = max(times[-1], .1)
    for label, values, color in series:
        points = [(left+t/duration*(right-left), bottom-(v-low)/(high-low)*(bottom-top))
                  for t,v in zip(times[:cursor+1], values[:cursor+1]) if v is not None and np.isfinite(v)]
        if len(points)>1: draw.line(points, fill=color, width=2)
        if points:
            px, py = points[-1]
            draw.ellipse((px-3, py-3, px+3, py+3), fill=color)
    x = left+times[cursor]/duration*(right-left)
    draw.line((x, top, x, bottom), fill="#666666")
    draw.text((left, bottom+8), "0 s", fill="black")
    draw.text((right-50, bottom+8), f"{duration:.1f} s", fill="black")


def render_result(result, root):
    from animate_checkpoint import load_frames
    row = result["episode"]
    frames, _ = load_frames(Path(row["shard"]), dict(gamma=result["gamma"], return_source="control", image_hz=10.))
    if not np.allclose([f["time_s"] for f in frames], result["times"]):
        raise ValueError("Video and prediction timestamps differ")
    video = root/"videos"/row["eval_case"]/f"agents_{row['num_agents']:02d}"/(Path(row["shard"]).stem+".mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    stamp = video.with_suffix(".render.json")
    signature = hashlib.sha256(json.dumps([2, result], sort_keys=True).encode()).hexdigest()
    if video.exists() and stamp.exists() and json.loads(stamp.read_text()).get("signature") == signature:
        return str(video.relative_to(root))
    colors = ("#1674c8", "#159246")
    times = result["times"]
    predictions = result["predicted_return"]
    series = [("Recorded return", result["target_return"], "#e48518")]
    series += [(name, predictions[name], colors[i]) for i,name in enumerate(MODELS)]
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "1200x760", "-r", "10", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    frame_times = np.r_[np.arange(0., times[-1], .1), times[-1]+np.arange(0., 1., .1)]
    try:
        for timestamp in frame_times:
            index = min(np.searchsorted(times, timestamp, side="right")-1, len(frames)-1)
            image = Image.new("RGB", (1200, 760), "white")
            scene = Image.fromarray(frames[index]["image"]).resize((500, 500), Image.Resampling.NEAREST)
            image.paste(scene, (10, 95))
            draw = ImageDraw.Draw(image)
            draw.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
            draw.text((15, 10), f"{row['eval_case']} | {row['num_agents']} agents | seed {row['seed']}", fill="black")
            draw.text((15, 32), f"t={times[index]:.2f}s | outcome: {row['termination_reason']} | gamma={result['gamma']} per native control step", fill="black")
            draw.text((15, 54), "Blue: joint WITHOUT obstacles   Green: joint WITH obstacles   Orange: recorded return-to-go", fill="black")
            draw.text((15, 74), "Trajectory fixed by MAPPO actor. Both critics score identical causal prefixes; neither controls these actions.", fill="black")
            plot_panel(draw, (525, 95, 1195, 365), times, series, index, "Future discounted return (native task units)")
            plot_panel(draw, (525, 375, 1195, 550), times,
                       [("Reward", [f["reward"] for f in frames], "#c34242")], index, "Recorded incoming control-step reward")
            plot_panel(draw, (525, 560, 1195, 735), times,
                       [("Progress", [f["progress"] for f in frames], "#9356af")], index, "Recorded team progress (not the return target)")
            for i,name in enumerate(MODELS):
                m = result["metrics"][name]
                metric = f"RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}" if "rmse" in m else "Return unknown: timeout; no regression score"
                draw.text((15, 610+i*36), f"{name}: {m['distribution']}", fill=colors[i])
                draw.text((15, 625+i*36), metric, fill=colors[i])
            draw.text((15, 704), "One rollout gives a realized return, not a Monte Carlo expected value.", fill="black")
            draw.text((15, 720), "True terminal network output is shown raw, excluded from error metrics.", fill="black")
            if index==len(frames)//2:
                image.save(video.with_suffix(".png"))
            process.stdin.write(np.asarray(image).tobytes())
    finally:
        process.stdin.close()
    if process.wait()!=0:
        video.unlink(missing_ok=True); raise RuntimeError("Video encoding failed")
    write_json(stamp, dict(signature=signature))
    return str(video.relative_to(root))


def render_all(root):
    results = json.loads((root/"evaluation_results.json").read_text())
    lines = ["# Matched ID/OOD critic rollout evaluation", "",
        "Three planned samples per team and case; recovery pairs are explicitly selected successful recoveries (all attempted seeds retained).",
        "Both frozen joint critics score identical causal observations. No policy or critic training occurs.",
        "Errors use known full native returns and exclude terminal anchors. Timeouts remain unscored.",
        "These are illustrative single-rollout return comparisons, not statistically powered expected-value estimates.", "",
        "| Case | Agents | Seed | Outcome | No-obstacle RMSE | Obstacle RMSE | Video |", "|---|---:|---:|---|---:|---:|---|"]
    for result in results:
        row = result["episode"]
        video = render_result(result, root)
        score = [f"{result['metrics'][m]['rmse']:.3f}" if "rmse" in result["metrics"][m] else "unknown (timeout)" for m in MODELS]
        lines.append(f"| {row['eval_case']} | {row['num_agents']} | {row['seed']} | {row['termination_reason']} | {score[0]} | {score[1]} | [MP4]({video}) |")
        print("Rendered", video, flush=True)
    pairs = defaultdict(dict)
    for result in results:
        row = result["episode"]
        if row["eval_case"] in {"formation_recovery_clean_pair", "formation_noise_recovery"}:
            pairs[(row["num_agents"], row["seed"])][row["eval_case"]] = result
    paired = []
    for (n,seed), pair in pairs.items():
        if len(pair)!=2: continue
        clean, noisy = pair["formation_recovery_clean_pair"], pair["formation_noise_recovery"]
        common_end = min(clean["times"][-1], noisy["times"][-1])
        aligned = []
        for time in (0., 2., 3., 5., 7.5, 10.):
            if time >= common_end: continue
            aligned.append(dict(time_s=time,
                noisy_minus_clean_recorded_return=float(np.interp(time, noisy["times"], noisy["target_return"])
                                                       -np.interp(time, clean["times"], clean["target_return"])),
                noisy_minus_clean_predicted_return={m:float(np.interp(time,noisy["times"],noisy["predicted_return"][m])
                                                          -np.interp(time,clean["times"],clean["predicted_return"][m])) for m in MODELS}))
        paired.append(dict(num_agents=n, seed=seed, clean_recorded_G0=clean["target_return"][0],
            noisy_recorded_G0=noisy["target_return"][0],
            noisy_minus_clean_recorded_G0=noisy["target_return"][0]-clean["target_return"][0],
            initial_prediction_differences={m:noisy["predicted_return"][m][0]-clean["predicted_return"][m][0] for m in MODELS},
            aligned_time_comparisons=aligned))
    write_json(root/"paired_noise_results.json", paired)
    groups = defaultdict(list)
    for result in results:
        row = result["episode"]
        for name in MODELS:
            groups[(row["eval_case"], row["num_agents"], name)].append(result)
    aggregates = []
    for (case, n, name), group in sorted(groups.items()):
        errors = [np.asarray(r["predicted_return"][name][:-1])-np.asarray(r["target_return"][:-1])
                  for r in group if r["complete"]]
        valid = np.concatenate(errors) if errors else np.array([])
        aggregates.append(dict(case=case, num_agents=n, critic=name, episodes=len(group),
            outcomes=dict(Counter(r["episode"]["termination_reason"] for r in group)),
            complete_episodes=len(errors), scored_frames=len(valid),
            pooled_mae=float(np.abs(valid).mean()) if len(valid) else None,
            pooled_rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else None,
            episode_macro_mae=float(np.mean([np.abs(e).mean() for e in errors])) if errors else None))
    write_json(root/"summary_metrics.json", aggregates)
    with (root/"summary_metrics.csv").open("w", newline="") as stream:
        fields = [k for k in aggregates[0] if k!="outcomes"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(aggregates)
    table = {(r["case"], r["num_agents"], r["critic"]):r for r in aggregates}
    lines += ["", "## Summary of these illustrative samples", "",
        "Pooled RMSE excludes true-terminal anchors. Compare within one task/condition/team row; native reward scales differ across tasks.", "",
        "| Case | Agents | No-obstacle critic RMSE | Obstacle critic RMSE |", "|---|---:|---:|---:|"]
    for case,n in sorted({(r["case"],r["num_agents"]) for r in aggregates}):
        values = [table[(case,n,m)]["pooled_rmse"] for m in MODELS]
        cells = [f"{v:.3f}" if v is not None else "unknown" for v in values]
        lines.append(f"| {case} | {n} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Paired successful noise recoveries", "",
        "All rows below ended successfully in both versions. Negative delta means the realized initial return fell under the sampled disturbance; this does not establish an expected-value effect.", "",
        "| Agents | Seed | Clean recorded G0 | Noisy recorded G0 | Noisy minus clean |", "|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['num_agents']} | {row['seed']} | {row['clean_recorded_G0']:.3f} | {row['noisy_recorded_G0']:.3f} | {row['noisy_minus_clean_recorded_G0']:.3f} |")
    lines += ["", "## Interpretation", "",
        "Agent counts 4 and 6 are unseen by the critics; formation actors were trained on 3–6, while navigation actors used odd teams.",
        "3/5-agent no-obstacle navigation is OOD for the obstacle critic (training layouts had at least one obstacle); obstacle navigation is OOD for the no-obstacle critic.",
        "Formation's 10-by-10 checkerboard covers 3.048 m physically; its native workspace is retained. Navigation's arena is 10 x 10 m.",
        "Visual formation obstacles change pixels only. Physical obstacles add collision termination and a team penalty equal to the native collision coefficient; they also change the task rules.",
        "Formation instructions are held fixed to isolate scene/dynamics changes. No obstacle geometry is added to the structured critic inputs.",
        "Noise is a late correlated burst at control steps 41–100, then the native actor recovers. Training already included early independent mild/moderate noise.",
        "Paired clean/noisy initial inputs are identical: an unannounced future perturbation cannot be predicted at t=0. Inspect divergence after the perturbation is observed.",
        "Recovery samples are conditional illustrations, not an estimate of recovery probability. See recovery_attempts_agents_*.json and paired_noise_results.json.", ""]
    if (root/"obstacle_cues/README.md").exists():
        lines += ["[Visual-only obstacle language/highlight comparisons](obstacle_cues/README.md)", ""]
    (root/"README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect-navigation", "collect-formation", "audit", "score", "render", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=ROOT/"outputs/ckpts/critic35_sequential")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples < 1: parser.error("samples must be positive")
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.phase=="all":
        for phase in ("collect-navigation", "collect-formation", "audit", "score", "render"):
            subprocess.run([sys.executable, __file__, phase, "--output", str(root), "--checkpoints", str(args.checkpoints),
                            "--samples", str(args.samples), "--device", args.device], check=True)
    elif args.phase=="collect-navigation": collect_navigation(root, args.samples)
    elif args.phase=="collect-formation": collect_formation(root, args.samples)
    elif args.phase=="audit": audit_rollouts(root, args.samples)
    elif args.phase=="score": score(root, args.checkpoints, args.device)
    elif args.phase=="render": render_all(root)


if __name__ == "__main__":
    main()
