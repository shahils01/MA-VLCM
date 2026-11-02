from collections import Counter
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from data_utils import (FormationClipDataset, collate_clips, feature_cache_path,
                        prepare_episode, read_episode, training_statistics)
from model import ModelConfig, MultimodalValueModel
from train_multitask import (JointClipDataset, adapt_normalizers, archive_key,
                             joint_splits, selection_score, training_partitions)

torch.set_num_threads(1)


def fixture_episode(root, index, task, reason=None):
    directory=Path(root)/task/"agents_03/worker_00"
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"episode_{index:06d}.tar"
    reason=reason or ("success" if index%2 else "out_of_bounds")
    with tarfile.open(path,"w") as archive:
        for step in range(4):
            agents=[]
            for i in range(3):
                pos=[index*.01+step*.02,i*.2]
                goal=[.7,i*.2+.3]
                agents.append(dict(pos=pos,yaw=.1*i,vel=[.2,0.],goal_pos=goal,
                    dist_to_goal=float(np.linalg.norm(np.asarray(goal)-pos)),min_neighbor_dist=.2))
            state=dict(agents=agents,capture_hz=10.,control_hz=20. if task=="formation_control" else 10.,
                task_domain=task,task_instruction="Form a polygon." if task=="formation_control" else "Reach matching goals.",
                episode_meta=dict(sim_time_s=step*.1,done=step==3))
            if task=="formation_control":
                state["formation"]=dict(robot_center_bounds_m=[-1.,1.,-1.,1.],pattern="polygon",formation_spacing_m=.5)
            else:
                state["navigation"]=dict(grid=[[-2.5,2.5],[-2.5,2.5]])
            png=BytesIO()
            Image.new("RGB",(32,32),(step*40,index*5,80)).save(png,format="PNG")
            adjacency=BytesIO()
            np.save(adjacency,np.ones((3,3),np.float32))
            reward=0. if step==0 else (1. if reason=="success" else -1.) if step==3 else -.02
            fields={"state.json":json.dumps(state).encode(),"overhead.png":png.getvalue(),
                    "adj.npy":adjacency.getvalue(),"reward.json":json.dumps(reward).encode(),
                    "control_rewards.json":json.dumps([] if step==0 else ([.01,reward] if task=="formation_control" else [reward])).encode()}
            for suffix,data in fields.items():
                member=tarfile.TarInfo(f"ep{index:06d}_step{step:05d}.{suffix}")
                member.size=len(data)
                archive.addfile(member,BytesIO(data))
    summary=path.with_suffix(".summary.json")
    summary.write_text(json.dumps(dict(episode_id=index,num_agents=3,success=reason=="success",
        termination_reason=reason,seed=index,checkpoint=dict(stage="finetuned" if task=="formation_control" else "best",sha256="fixture"))))
    return path,summary


class JointValueTests(unittest.TestCase):
    def test_both_schemas_and_native_control_rates_and_timeout_exclusion(self):
        with tempfile.TemporaryDirectory() as root:
            formation=read_episode(*fixture_episode(root,1,"formation_control"))
            navigation=read_episode(*fixture_episode(root,1,"goal_to_goal"))
            timeout=read_episode(*fixture_episode(root,2,"goal_to_goal","max_steps"))
            self.assertEqual(formation["robot_obs"].shape,navigation["robot_obs"].shape)
            self.assertEqual(navigation["task_domain"],"goal_to_goal")
            self.assertEqual(navigation["bounds"],[-2.5,2.5,-2.5,2.5])
            self.assertFalse(timeout["complete"])
            self.assertNotEqual(formation["scenario_id"],navigation["scenario_id"])
            for ep in (formation,navigation):
                prepared=prepare_episode(ep,.5,"control",10.)
                controls=sum(ep["control_rewards"][1:],[])
                expected=sum(.5**i*r for i,r in enumerate(controls))
                self.assertAlmostEqual(prepared["returns"][0].item(),expected,places=6)

    def test_old_heldout_splits_survive_relocation_and_mixed_scenario_conflicts_fail(self):
        episodes=[]
        old={"splits":dict(train=[],validation=[],test=[])}
        for i,name in enumerate(old["splits"]):
            path=f"/new/formation_control/agents_03/worker_00/episode_{i:06d}.tar"
            old["splits"][name].append(path.replace("/new/","/old/"))
            episodes.append(dict(path=path,task_domain="formation_control",scenario_id=str(i),stage="finetuned",num_agents=3,termination_reason="success"))
        for i in range(12):
            episodes.append(dict(path=f"/new/goal_to_goal/episode_{i}.tar",task_domain="goal_to_goal",scenario_id=str(i),stage="best",num_agents=3,termination_reason="success"))
        result=joint_splits(episodes,old)
        for name in old["splits"]:
            saved=[ep for ep in result[name] if ep["task_domain"]=="formation_control"]
            self.assertEqual(archive_key(saved[0]["path"]),archive_key(old["splits"][name][0]))
        navigation_only=training_partitions(result,["goal_to_goal"])
        self.assertEqual({ep["task_domain"] for ep in navigation_only["train"]},{"goal_to_goal"})
        for name in ("validation","test"):
            self.assertEqual(navigation_only[name],result[name])
        self.assertIn("formation_control",{ep["task_domain"] for ep in result["train"]})
        episodes[1]["scenario_id"]=episodes[0]["scenario_id"]
        with self.assertRaisesRegex(ValueError,"scenario crosses"):
            joint_splits(episodes,old)

    def test_normalizer_conversion_preserves_value_predictions_and_keeps_representation(self):
        torch.manual_seed(7)
        cfg=ModelConfig(vl_hidden_dim=8,d_model=16,dropout=0)
        model=MultimodalValueModel(cfg,load_backbone=False).eval()
        model.obs_mean.copy_(torch.randn(10))
        model.obs_std.copy_(torch.rand(10)+.5)
        model.target_mean.fill_(4.)
        model.target_std.fill_(3.)
        batch=dict(vl_features=torch.randn(2,6,8),robot_obs=torch.randn(2,6,4,10),
            adj=torch.ones(2,6,4,4),robot_mask=torch.ones(2,6,4,dtype=torch.bool),
            frame_mask=torch.ones(2,6,dtype=torch.bool),times=torch.arange(-5,1).float().repeat(2,1))
        before=model(batch).detach()
        old_graph=model.graph.layers[0].source.weight.detach().clone()
        statistics=dict(obs_mean=torch.randn(10),obs_std=torch.rand(10)+.5,
                        target_mean=torch.tensor(-8.),target_std=torch.tensor(12.))
        adapt_normalizers(model,statistics)
        torch.testing.assert_close(model(batch),before,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(model.graph.layers[0].source.weight,old_graph,rtol=0,atol=0)
        for key,value in statistics.items():
            torch.testing.assert_close(getattr(model,key),value)

    def test_task_weights_equalize_unequal_clip_populations_and_macro_selection(self):
        with tempfile.TemporaryDirectory() as root:
            refs=[prepare_episode(read_episode(*fixture_episode(root,i,"formation_control")),source="control",image_hz=10.) for i in range(5)]
            refs.append(prepare_episode(read_episode(*fixture_episode(root,10,"goal_to_goal")),source="control",image_hz=10.))
            stats={task:dict(target_mean=0.,target_std=1.) for task in ("formation_control","goal_to_goal")}
            ds=JointClipDataset(refs,6,include_terminal=False,task_statistics=stats)
            mass=Counter()
            for weight,(index,_) in zip(ds.sample_weights,ds.samples):
                mass[refs[index]["task_domain"]]+=float(weight)
            self.assertAlmostEqual(mass["formation_control"],mass["goal_to_goal"])
            score=selection_score(dict(by_task={"formation_control":dict(mse=1.),"goal_to_goal":dict(mse=400.)}),
                                  dict(formation_control=dict(target_std=1.),goal_to_goal=dict(target_std=20.)))
            self.assertEqual(score,1.)

    def test_joint_training_checkpoint_and_before_after_reports_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            episodes=[]
            old_splits=dict(train=[],validation=[],test=[])
            for task in ("formation_control","goal_to_goal"):
                for i in range(9):
                    path,summary=fixture_episode(root/"data",i,task)
                    episodes.append(prepare_episode(read_episode(path,summary),.99,"control",10.))
                    if task=="formation_control":
                        old_splits[("train","validation","test")[i//3]].append(str(path))
            cfg=ModelConfig(model_id="synthetic-test",vl_hidden_dim=8,d_model=16,
                            image_size=32,clip_len=6,dropout=0,vl_dtype="float32")
            initial=MultimodalValueModel(cfg,load_backbone=False)
            checkpoint=root/"initial.pt"
            torch.save(dict(model_config=asdict(cfg),model_state=initial.checkpoint_state(),run=dict(
                args=dict(return_source="control",gamma=.99),splits=old_splits)),checkpoint)
            original=checkpoint.read_bytes()
            features=root/"cache/features"
            features.mkdir(parents=True)
            for ep in episodes:
                torch.save(torch.randn(len(ep["selected_indices"]),8),feature_cache_path(ep,cfg,features))
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/"train_multitask.py"),
                "--formation-dataset",str(root/"data/formation_control"),
                "--gotogoal-dataset",str(root/"data/goal_to_goal"),
                "--init-checkpoint",str(checkpoint),"--cache-dir",str(root/"cache"),
                "--output-dir",str(root/"output"),"--expected-formation-episodes","9",
                "--expected-gotogoal-episodes","9","--epochs","2","--batch-size","8","--mixed-precision","no"]
            completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(completed.returncode,0,completed.stdout)
            self.assertEqual(checkpoint.read_bytes(),original)
            comparison=json.loads((root/"output/task_comparison.json").read_text())
            self.assertEqual(set(comparison),{"formation_control","goal_to_goal"})
            for task in comparison:
                self.assertGreater(comparison[task]["after"]["count"],0)
                self.assertTrue(np.isfinite(comparison[task]["after"]["mae"]))
            saved=torch.load(root/"output/best.pt",map_location="cpu",weights_only=True)
            self.assertEqual(saved["model_config"]["backbone_mode"],"frozen")
            self.assertTrue(any(not torch.equal(initial.state_dict()[key],value) for key,value in saved["model_state"].items()))
            metrics=json.loads((root/"output/after_test_metrics.json").read_text())
            self.assertEqual(set(metrics["by_formation_outcome"]) <= {"triangle_success","triangle_failure"},True)
            self.assertIn("by_outcome",metrics["by_task"]["goal_to_goal"])
            latest=torch.load(root/"output/latest.pt",map_location="cpu",weights_only=True)
            self.assertFalse(torch.equal(initial.state_dict()["graph.layers.0.source.weight"],latest["model_state"]["graph.layers.0.source.weight"]))
            evaluate_command=[sys.executable,str(Path(__file__).resolve().parents[1]/"evaluate.py"),
                "--checkpoint",str(root/"output/best.pt"),"--cache-dir",str(root/"cache"),
                "--output-dir",str(root/"standalone_evaluation"),"--split","test"]
            evaluation=subprocess.run(evaluate_command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(evaluation.returncode,0,evaluation.stdout)
            standalone=json.loads((root/"standalone_evaluation/test_metrics.json").read_text())
            self.assertEqual(set(standalone["by_task"]),{"formation_control","goal_to_goal"})
            resume=command.copy()
            resume[resume.index("--epochs")+1]="3"
            resume += ["--resume",str(root/"output/latest.pt")]
            continuation=subprocess.run(resume,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(continuation.returncode,0,continuation.stdout)
            self.assertEqual(torch.load(root/"output/latest.pt",map_location="cpu",weights_only=True)["epoch"],3)
            navigation_only=command.copy()
            navigation_only[navigation_only.index("--output-dir")+1]=str(root/"navigation_only")
            navigation_only += ["--training-tasks","goal_to_goal"]
            transfer=subprocess.run(navigation_only,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(transfer.returncode,0,transfer.stdout)
            transfer_run=json.loads((root/"navigation_only/run.json").read_text())
            self.assertEqual(set(transfer_run["training_episode_counts"]),{"goal_to_goal"})
            self.assertTrue(all("goal_to_goal" in path for path in transfer_run["splits"]["train"]))
            self.assertTrue(any("formation_control" in path for path in transfer_run["splits"]["test"]))
            self.assertEqual(transfer_run["task_statistics"]["formation_control"]["target_mean"],initial.target_mean.item())
            self.assertEqual(checkpoint.read_bytes(),original)
            transfer_results=json.loads((root/"navigation_only/task_comparison.json").read_text())
            self.assertEqual(set(transfer_results),{"formation_control","goal_to_goal"})


if __name__=="__main__":
    unittest.main()
