"""Collect and audit eight fresh task rollouts, then evaluate a frozen VLCM."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
COLLECTORS = ROOT.parent/"Ground_Robot_MARL/scripts/vlcm"
SOURCES = {"formation_control":ROOT/"outputs/python_frozen_external_10",
           "goal_to_goal":ROOT.parent/"Ground_Robot_MARL/datasets/manyagent_gotogoal_python_2000"}


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_collection(root,task,jobs):
    source=SOURCES[task]
    destination=root/"samples"/("formation" if task=="formation_control" else "navigation")
    if (destination/"collection_plan.json").exists():
        plan=json.loads((destination/"collection_plan.json").read_text())
        if plan["jobs"] != jobs:
            raise ValueError("Resume jobs differ from the original collection")
        return destination,plan
    destination.mkdir(parents=True)
    shutil.copytree(source/"source",destination/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(source/"checkpoints",destination/"checkpoints",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    plan=json.loads((source/"collection_plan.json").read_text())
    request=json.loads((root/"request.json").read_text())
    if request.get("max_steps_override",0):
        plan["environment_config"]["max_steps"]=request["max_steps_override"]
    for policy in plan["checkpoints"]:
        policy["checkpoint"]=str(destination/"checkpoints"/policy["stage"]/"actor.pt")
        assert digest(policy["checkpoint"])==policy["sha256"]
        assert digest(Path(policy["checkpoint"]).with_name("training_metadata.json"))==policy["metadata_sha256"]
    plan.update(episodes=4,jobs=jobs,intended_use="Fresh external frozen-VLCM evaluation; no training or outcome filtering",
                source_collection=str(source.resolve()))
    (destination/"collection_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    return destination,plan


def collect_task(root,task):
    request=json.loads((root/"request.json").read_text())
    jobs=request["jobs"][task]
    destination,plan=prepare_collection(root,task,jobs)
    sys.path.insert(0,str(COLLECTORS))
    sys.path.insert(0,str(destination/"source"))
    policies={p["stage"]:p for p in plan["checkpoints"]}
    summaries,audits=[],[]
    if task=="formation_control":
        import collect_python_formation_dataset as collector
        # The original collector reads a module-level configuration. Override
        # only the time limit, retaining all native dynamics and termination.
        collector.CONFIG.update(plan["environment_config"])
        import validate_python_formation_dataset as validator
        for job in jobs:
            sidecar=destination/task/f"agents_{job['num_agents']:02d}"/f"worker_{job['worker_id']:02d}"/f"episode_{job['episode_id']:06d}.summary.json"
            summary=json.loads(sidecar.read_text()) if sidecar.exists() else collector.collect_job(destination,job,policies[job["stage"]])
            audits.append(validator.audit_episode(destination,job))
            summaries.append(summary)
            print(f"Collected/audited {task} n={job['num_agents']} seed={job['seed']}: {summary['termination_reason']}, {summary['control_steps']} controls",flush=True)
    else:
        import collect_python_gotogoal_dataset as collector
        import validate_python_gotogoal_dataset as validator
        for job in jobs:
            runtime=collector.load_runtime(destination,plan["environment_config"],job["num_agents"],policies[job["stage"]])
            try:
                _,sidecar=collector.paths(destination,job)
                summary=json.loads(sidecar.read_text()) if sidecar.exists() else collector.rollout(destination,job,policies[job["stage"]],plan["environment_config"],*runtime)
                audits.append(validator.audit_episode(destination,job,policies[job["stage"]],plan["environment_config"],runtime))
                summaries.append(summary)
                print(f"Collected/audited {task} n={job['num_agents']} seed={job['seed']}: {summary['termination_reason']}, {summary['control_steps']} controls",flush=True)
            finally:
                runtime[0].close()
    (destination/"external_collection_report.json").write_text(json.dumps(dict(passed=True,episodes=summaries,audits=audits),indent=2)+"\n")


def prepare_request(root,checkpoint_path,seed,resume,max_steps=0):
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError("Output directory already exists; use --resume or choose another")
        request=json.loads((root/"request.json").read_text())
        if (request["checkpoint_sha256"]!=digest(checkpoint_path) or request["seed"]!=seed
                or request.get("max_steps_override",0)!=max_steps):
            raise ValueError("Resume requires the same checkpoint, random seed and step limit")
        return request
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    assert checkpoint["model_config"]["backbone_mode"]=="frozen"
    assert checkpoint["run"]["args"]["return_source"]=="control"
    rng=np.random.default_rng(seed)
    reset_seeds=rng.integers(100000000,200000000,size=8).tolist()
    assert len(set(reset_seeds))==8
    jobs={}
    for index,(task,stage) in enumerate((("formation_control","successful"),("goal_to_goal","best"))):
        source_plan=json.loads((SOURCES[task]/"collection_plan.json").read_text())
        source_seeds={job["seed"] for job in source_plan["jobs"]}
        jobs[task]=[dict(episode_id=800000+index*100+n,num_agents=n,stage=stage,noise="clean",
                         seed=reset_seeds[index*4+n-3],noise_seed=reset_seeds[index*4+n-3]+50000000,
                         worker_id=n-3) for n in (3,4,5,6)]
        assert not source_seeds & {job["seed"] for job in jobs[task]}
    from data_utils import episode_path_key
    seen={episode_path_key(p) for group in (checkpoint["run"]["splits"],checkpoint["run"].get("pretrained_splits",{}))
          for paths in group.values() for p in paths}
    for task,rows in jobs.items():
        for job in rows:
            key=(task,f"agents_{job['num_agents']:02d}",f"worker_{job['worker_id']:02d}",f"episode_{job['episode_id']:06d}.tar")
            assert key not in seen
    snapshot=root/"checkpoint"
    snapshot.mkdir()
    shutil.copy2(checkpoint_path,snapshot/"best.pt")
    request=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=digest(checkpoint_path),checkpoint_epoch=checkpoint["epoch"],
                 seed=seed,jobs=jobs,max_steps_override=max_steps,
                 termination="Native success, collision and boundary termination retained; time limit is only a maximum",
                 actor_selection="Best/finetuned behavior checkpoint of each source collection; deterministic clean actions",
                 selection="Exactly one fresh seeded episode per task/team size; no resampling for success",
                 note="Recorded discounted returns are individual observed outcomes, not a statistical estimate of the expected return")
    (root/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    return request


def render_curve_video(root,episode,predictions,targets,selected,summary,metrics):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from data_utils import read_images
    task=episode["task_domain"]
    n=episode["num_agents"]
    name=f"{task}_agents_{n:02d}"
    output=root/"videos"
    output.mkdir(exist_ok=True)
    times=episode["timestamps"][selected].numpy()
    fig,ax=plt.subplots(figsize=(5.12,5.12),dpi=100)
    ax.plot(times,targets,label="Realized discounted return",color="#d08000")
    ax.plot(times,predictions,label="Frozen VLCM value",color="#2467b2")
    ax.set(xlabel="Time (s)",ylabel="Native discounted return",xlim=(0.,max(.1,float(times[-1]))))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    title=(f"MAE {metrics['mae']:.2f} | RMSE {metrics['rmse']:.2f}" if metrics else "Timeout: targets are partial returns")
    ax.set_title(title,fontsize=10)
    fig.tight_layout()
    canvas=FigureCanvasAgg(fig)
    canvas.draw()
    chart=np.asarray(canvas.buffer_rgba())[...,:3].copy()
    box=ax.get_window_extent()
    fig.savefig(root/"plots"/f"{name}.png",dpi=150)
    pngs=read_images(episode,selected)
    fps=episode["capture_hz"]
    with imageio.get_writer(str(output/f"{name}.mp4"),format="FFMPEG",fps=fps,
                            codec="libx264",quality=8,macro_block_size=16,ffmpeg_log_level="error") as writer:
        for index,(pixels,t,pred,target) in enumerate(zip(pngs,times,predictions,targets)):
            frame=Image.new("RGB",(1024,576),"white")
            frame.paste(pixels.resize((512,512),Image.Resampling.NEAREST),(0,64))
            frame.paste(Image.fromarray(chart),(512,64))
            draw=ImageDraw.Draw(frame)
            label="Formation control" if task=="formation_control" else "Go-to-goal"
            draw.text((12,9),f"{label} | {n} agents | seed {summary['seed']} | frozen VLCM",fill="black")
            draw.text((12,27),f"t={t:.2f}s | prediction={pred:.3f} | realized return={target:.3f} | outcome={summary['termination_reason']}",fill="black")
            draw.text((12,45),"Causal model inputs; future rewards used only as offline labels. Terminal anchors excluded from metrics.",fill="black")
            x=512+box.x0+float(t)/max(.1,float(times[-1]))*box.width
            draw.line((x,64+512-box.y1,x,64+512-box.y0),fill="#555555",width=1)
            writer.append_data(np.asarray(frame))
    plt.close(fig)
    return str((output/f"{name}.mp4").relative_to(root))


def reuse_identical_features(root,previous):
    """Reuse only an identical frozen image/instruction computation."""
    import hashlib
    from data_utils import (discover_episodes,read_episode,prepare_episode,
                            feature_cache_path)
    from model import ModelConfig

    checkpoint=torch.load(root/"checkpoint/best.pt",map_location="cpu",weights_only=True)
    original=torch.load(previous/"checkpoint/best.pt",map_location="cpu",weights_only=True)
    if checkpoint["model_config"]!=original["model_config"]:
        raise ValueError("Feature reuse requires identical model configurations")
    device="cuda" if torch.cuda.is_available() else "cpu"
    if json.loads((previous/"alignment_summary.json").read_text())["device"]!=device:
        raise ValueError("Feature reuse requires the same inference device/precision")
    cfg=ModelConfig(**checkpoint["model_config"])
    args=checkpoint["run"]["args"]

    def episodes(directory):
        return [prepare_episode(read_episode(p,s),args["gamma"],"control",args["image_hz"])
                for p,s in discover_episodes(directory/"samples")]

    def image_digests(ep):
        hashes=[]
        with open(ep["path"],"rb") as archive:
            for i in ep["selected_indices"]:
                offset,size=ep["image_offsets"][i]
                archive.seek(offset)
                hashes.append(hashlib.sha256(archive.read(size)).hexdigest())
        return hashes

    old={(ep["task_domain"],ep["num_agents"]):ep for ep in episodes(previous)}
    copied=[]
    for ep in episodes(root):
        source=old.get((ep["task_domain"],ep["num_agents"]))
        if source is None or source["prompt"]!=ep["prompt"]:
            continue
        if image_digests(source)!=image_digests(ep):
            continue
        src=feature_cache_path(source,cfg,previous/"cache/features")
        if not src.exists():
            continue
        features=torch.load(src,map_location="cpu",weights_only=True)
        assert features.shape==(len(ep["selected_indices"]),cfg.vl_hidden_dim)
        assert torch.isfinite(features).all()
        dst=feature_cache_path(ep,cfg,root/"cache/features")
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        copied.append(dict(task=ep["task_domain"],num_agents=ep["num_agents"],frames=len(features),
                           identical_prompt=True,every_input_png_sha256_identical=True))
    (root/"feature_reuse_audit.json").write_text(json.dumps(dict(
        previous_evaluation=str(previous),identical_model_configuration=True,
        same_inference_device_and_precision=True,reused_episodes=copied),indent=2)+"\n")
    print(f"Verified and reused identical frozen image/instruction features for {len(copied)} episodes",flush=True)


def evaluate(root,threads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from transformers.utils import logging
    logging.disable_progress_bar()
    from data_utils import (FormationClipDataset,collate_clips,discover_episodes,read_episode,
                            prepare_episode,feature_cache_path,read_images,save_tensor_file)
    from model import ModelConfig,MultimodalValueModel,SmolVLMBackbone
    from train import regression_metrics
    torch.set_num_threads(threads)
    checkpoint=torch.load(root/"checkpoint/best.pt",map_location="cpu",weights_only=True)
    cfg=ModelConfig(**checkpoint["model_config"])
    request=json.loads((root/"request.json").read_text())
    args=checkpoint["run"]["args"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refs=discover_episodes(root/"samples")
    if len(refs)!=8:
        raise ValueError(f"Expected eight episodes, found {len(refs)}")
    episodes=[prepare_episode(read_episode(path,summary,root/"cache/index"),args["gamma"],"control",args["image_hz"]) for path,summary in refs]
    if {(ep["task_domain"],ep["num_agents"]) for ep in episodes}!={(task,n) for task in SOURCES for n in (3,4,5,6)}:
        raise ValueError("Missing task/team-size coverage")
    feature_dir=root/"cache/features"
    missing=[ep for ep in episodes if not feature_cache_path(ep,cfg,feature_dir).exists()]
    if missing:
        encoder=SmolVLMBackbone(cfg,device).requires_grad_(False).eval()
        for ep in missing:
            blocks=[]
            indices=ep["selected_indices"]
            print(f"Encoding {ep['task_domain']} n={ep['num_agents']}: {len(indices)} frames on {device}",flush=True)
            for start in range(0,len(indices),cfg.vl_batch_size):
                images=read_images(ep,indices[start:start+cfg.vl_batch_size])
                with torch.inference_mode():
                    features=encoder(images,[ep["prompt"]]*len(images)).detach().cpu().half()
                blocks.append(features)
                if start%32==0:
                    print(f"  frames {min(start+len(images),len(indices))}/{len(indices)}",flush=True)
            feature_dir.mkdir(parents=True,exist_ok=True)
            save_tensor_file(torch.cat(blocks),feature_cache_path(ep,cfg,feature_dir))
        del encoder
        if device.type=="cuda":
            torch.cuda.empty_cache()
    model=MultimodalValueModel(cfg,device,load_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state"],strict=True)
    model.requires_grad_(False).eval()
    assert not any(p.requires_grad for p in model.parameters())
    (root/"plots").mkdir(exist_ok=True)
    (root/"predictions").mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(14,15),constrained_layout=True)
    results=[]
    by_task=defaultdict(list)
    for ep in episodes:
        ds=FormationClipDataset([ep],cfg.clip_len,1,feature_dir,cfg,include_terminal=True)
        values=[]
        with torch.inference_mode():
            for batch in DataLoader(ds,batch_size=64,collate_fn=collate_clips):
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                precision=torch.autocast("cuda",dtype=getattr(torch,cfg.vl_dtype)) if device.type=="cuda" and cfg.vl_dtype!="float32" else nullcontext()
                with precision:
                    values.extend(model(batch).float().cpu().tolist())
        predicted=np.asarray(values)
        actual=ep["returns"].numpy()
        assert np.isfinite(predicted).all()
        summary=json.loads(Path(ep["path"]).with_suffix(".summary.json").read_text())
        eligible=np.arange(len(actual))<len(actual)-1 if ep["complete"] else np.zeros(len(actual),bool)
        task,n=ep["task_domain"],ep["num_agents"]
        baseline=checkpoint["run"]["task_statistics"][task]["target_mean"]
        metrics=regression_metrics(predicted[eligible],actual[eligible],baseline) if eligible.any() else None
        times=ep["timestamps"][ep["selected_indices"]].numpy()
        name=f"{task}_agents_{n:02d}"
        with (root/"predictions"/f"{name}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["anchor","time_s","predicted_value","realized_discounted_return","error","included_in_metrics","terminal","return_is_partial"])
            writer.writeheader()
            for i,(t,p,target) in enumerate(zip(times,predicted,actual)):
                writer.writerow(dict(anchor=i,time_s=float(t),predicted_value=float(p),realized_discounted_return=float(target),
                    error=float(p-target),included_in_metrics=bool(eligible[i]),terminal=bool(ep["complete"] and i==len(actual)-1),return_is_partial=not ep["complete"]))
                if eligible[i]:
                    by_task[task].append((float(p),float(target)))
        video=render_curve_video(root,ep,predicted,actual,ep["selected_indices"],summary,metrics)
        result=dict(task=task,num_agents=n,seed=summary["seed"],outcome=summary["termination_reason"],
            success=summary["success"],complete=ep["complete"],samples=len(actual),control_steps=summary["control_steps"],
            episode=str(Path(ep["path"]).relative_to(root)),video=video,
            metrics=metrics,initial_prediction=float(predicted[0]),initial_realized_return=float(actual[0]),
            raw_terminal_prediction=float(predicted[-1]) if ep["complete"] else None)
        results.append(result)
        ax=axes[n-3,0 if task=="formation_control" else 1]
        ax.plot(times,actual,label="Realized discounted return",color="#d08000")
        ax.plot(times,predicted,label="Frozen VLCM value",color="#2467b2")
        if ep["complete"]:
            ax.scatter(times[-1],predicted[-1],marker="x",color="#2467b2",label="Terminal (excluded)")
        score=f"MAE {metrics['mae']:.2f}" if metrics else "partial return"
        ax.set(title=f"{task}, {n} agents: {summary['termination_reason']} | {score}",xlabel="Time (s)",ylabel="Discounted return")
        ax.grid(alpha=.25)
        print(json.dumps(result),flush=True)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Eight fresh rollouts: frozen VLCM predictions vs recorded future returns")
    fig.savefig(root/"value_alignment.png",dpi=150)
    plt.close(fig)
    pooled={task:regression_metrics([v[0] for v in rows],[v[1] for v in rows],checkpoint["run"]["task_statistics"][task]["target_mean"]) for task,rows in by_task.items()}
    report=dict(checkpoint=str(root/"checkpoint/best.pt"),checkpoint_epoch=checkpoint["epoch"],
        checkpoint_sha256=digest(root/"checkpoint/best.pt"),frozen_all_layers=True,device=str(device),
        gamma=args["gamma"],return_source="Full future native control rewards; incoming reward excluded",
        max_steps_override=request.get("max_steps_override",0),
        control_hz_by_task={task:sorted({ep["control_hz"] for ep in episodes if ep["task_domain"]==task}) for task in SOURCES},
        note="Exactly one freshly seeded rollout per task/team size. Values are compared to realized returns, not a statistical estimate of expectation. No outcome filtering, terminal value override, calibration, fitting or gradient updates. Terminal anchors and all timeout-return labels are excluded from scores.",
        by_task=pooled,episodes=results)
    (root/"alignment_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Frozen multitask VLCM: eight fresh rollouts","",report["note"],"",
           f"Checkpoint: `{json.loads((root/'request.json').read_text())['checkpoint']}`, epoch {checkpoint['epoch']}. Evaluation device: {device}. All model layers are frozen.","",
           f"Maximum native control steps per episode: {request.get('max_steps_override') or 'original collection setting'}. Success, collision and boundary termination remain active; a larger limit does not extend episodes that have already terminated.","",
           "| Task | Nonterminal frames | MAE | RMSE | R² |",
           "| --- | --- | --- | --- | --- |"]
    for task,metrics in pooled.items():
        lines.append(f"| {task} | {metrics['count']} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} | {metrics['r2']:.3f} |")
    lines += ["","Formation predictions follow these rollouts more closely than go-to-goal predictions. Raw terminal predictions are often far from zero; terminal outputs are preserved in the CSVs and plots but excluded from the scores, matching the training evaluation convention. Bootstrap values must be masked at true terminal states.","",
           "| Task | Robots | Outcome | MAE | RMSE | Initial prediction | Initial recorded return | Video |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in results:
        m=row["metrics"]
        mae=f"{m['mae']:.3f}" if m else "excluded"
        rmse=f"{m['rmse']:.3f}" if m else "excluded"
        lines.append(f"| {row['task']} | {row['num_agents']} | {row['outcome']} | {mae} | {rmse} | {row['initial_prediction']:.3f} | {row['initial_realized_return']:.3f} | [MP4]({row['video']}) |")
    lines += ["","All sampled archives and full native control traces are in `samples/`. Each source collection has a replay-audit report, frozen behavior policies, environment and serializer snapshots. `predictions/` contains every raw network prediction, including terminal predictions; `value_alignment.png` and `plots/` show the traces. The critic's checkpoint and all model parameters remain frozen throughout.","",
              "Task summaries pool frames, so longer episodes contribute more observations. Eight episodes are a diagnostic; they do not establish performance across the full task distribution. Images stored in the archives are 256 × 256; model preprocessing and the eight-frame causal clips match the checkpoint configuration."]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(pooled,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"outputs/multitask_frozen_19174095")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/multitask_frozen_19174095_external_8")
    parser.add_argument("--seed",type=int,default=20261103)
    parser.add_argument("--torch-threads",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-steps",type=int,default=0,help="Override the native time limit; 0 keeps the collection setting. True terminations remain active.")
    parser.add_argument("--reuse-features-from",type=Path,default=None,help="Reuse visual features only after comparing model configuration, prompts and every input PNG byte")
    parser.add_argument("--collect-task",choices=tuple(SOURCES),default=None)
    args=parser.parse_args()
    if args.max_steps<0:
        parser.error("--max-steps must be nonnegative")
    root=args.output_dir.resolve()
    if args.collect_task:
        collect_task(root,args.collect_task)
        return
    checkpoint=args.checkpoint.resolve()
    if checkpoint.is_dir():
        checkpoint=checkpoint/"best.pt"
    request=prepare_request(root,checkpoint,args.seed,args.resume,args.max_steps)
    for task in SOURCES:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"--output-dir",str(root),"--collect-task",task],check=True)
    reports=[json.loads((root/"samples"/name/"external_collection_report.json").read_text()) for name in ("formation","navigation")]
    assert all(report["passed"] and len(report["episodes"])==4 for report in reports)
    (root/"collection_audit.json").write_text(json.dumps(dict(passed=True,episodes=8,tasks=list(SOURCES),fresh_reset_seeds=[j["seed"] for rows in request["jobs"].values() for j in rows]),indent=2)+"\n")
    if args.reuse_features_from:
        reuse_identical_features(root,args.reuse_features_from.resolve())
    evaluate(root,args.torch_threads)
    assert digest(checkpoint)==request["checkpoint_sha256"],"Source checkpoint changed"
    print(f"Completed eight frozen-critic evaluations: {root}",flush=True)


if __name__=="__main__":
    main()
