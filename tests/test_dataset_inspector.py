"""Inspector uses real training targets and rejects misaligned archive labels."""
import hashlib
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from data_utils import prepare_episode, read_episode
from inspect_dataset import Catalog, inspect_episode, load_archive, make_handler


def fixture(root, task="goal_to_goal", reason="success", bad_controls=False, bad_dist=False):
    path=Path(root)/task/"agents_03/worker_00/episode_000000.tar"
    path.parent.mkdir(parents=True,exist_ok=True)
    initial=np.array([[-1.,-1.8],[-1.,0.],[-1.,1.8]])
    goals=initial+[1.,0.]
    reward_sum=0
    with tarfile.open(path,"w") as archive:
        for step in range(4):
            pos=initial+[step/3,0.]
            distances=np.linalg.norm(goals-pos,axis=-1)
            pair=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
            commands=np.zeros((3,2))
            agents=[dict(id=i,pos=pos[i].tolist(),yaw=0.,vel=[0.,0.],goal_pos=goals[i].tolist(),
                dist_to_goal=float(distances[i]),min_neighbor_dist=1.8,reached=step==3,action_vector=[0.,0.]) for i in range(3)]
            chunk=[] if step==0 else [2.*step-1,2.*step]
            reward=chunk[-1] if chunk else 0.
            reward_sum+=sum(chunk)
            state=dict(agents=agents,capture_hz=10.,control_hz=20.,task_domain=task,
                task_instruction="Guide all 3 robots to their assigned goals.",
                episode_meta=dict(sim_time_s=step/10,done=step==3))
            if task=="goal_to_goal":
                state["navigation"]=dict(grid=[[-5,5],[-5,5]],obstacles=[],robot_diameter_m=.3,
                    adjacency_distance_m=1.,adjacency_self_loops=True,start_region_bounds_m=[[-4,4],[-4,4]],
                    minimum_initial_agent_separation_m=1.5,sampled_action_noise=[[0.,0.]]*3)
            else:
                state["formation"]=dict(robot_center_bounds_m=[-5,5,-5,5],pattern="polygon")
            progress=dict(schema="tb3_progress_v2",initial_distances=[1.]*3,
                team_progress=step/3,per_agent_progress=[step/3]*3)
            png=BytesIO(); Image.new("RGB",(32,32),(step*50,0,0)).save(png,format="PNG")
            fields={"state.json":state,"reward.json":reward,"control_rewards.json":chunk[:1] if bad_controls and step==1 else chunk,
                    "progress.json":progress,"task.txt":state["task_instruction"],"overhead.png":png.getvalue(),
                    "dist.npy":pair+1 if bad_dist else pair,"adj.npy":pair<1.,"actions.npy":commands}
            for field,value in fields.items():
                if isinstance(value,np.ndarray):
                    stream=BytesIO(); np.save(stream,value,allow_pickle=False); raw=stream.getvalue()
                elif isinstance(value,bytes): raw=value
                elif isinstance(value,str): raw=value.encode()
                else: raw=json.dumps(value).encode()
                member=tarfile.TarInfo(f"ep000000_step{step:05d}.{field}"); member.size=len(raw)
                archive.addfile(member,BytesIO(raw))
    summary=path.with_suffix(".summary.json")
    summary.write_text(json.dumps(dict(episode_id=0,num_agents=3,seed=42,success=reason=="success",
        termination_reason=reason,episode_return=reward_sum,condition="without_obstacles",
        checkpoint=dict(stage="best",sha256="fixture"),tar_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    return path,summary


class InspectorTests(unittest.TestCase):
    def tearDown(self): load_archive.cache_clear()

    def test_both_tasks_pass_and_match_training_return_with_subsampling(self):
        with tempfile.TemporaryDirectory() as root:
            for task in ("formation_control","goal_to_goal"):
                path,summary=fixture(root,task)
                report=inspect_episode(path,summary,.5,5.)
                self.assertEqual([c for c in report["checks"] if c["status"]!="pass"],[])
                expected=prepare_episode(read_episode(path,summary),.5,"control",5.)
                np.testing.assert_allclose(report["series"]["target_return"],expected["returns"].numpy())
                self.assertEqual(report["series"]["return_time"],[0.,.2,.3])
                self.assertEqual(report["series"]["cumulative_reward"][-1],21.)
                self.assertEqual(report["series"]["target_return"][-1],0.)
                json.dumps(report,allow_nan=False)

    def test_max_steps_never_gets_zero_continuation_target(self):
        with tempfile.TemporaryDirectory() as root:
            report=inspect_episode(*fixture(root,reason="max_steps"),gamma=.99)
            self.assertFalse(report["complete"])
            self.assertTrue(all(v is None for v in report["series"]["target_return"]))
            self.assertEqual(next(c for c in report["checks"] if c["name"]=="Known continuation return")["status"],"warning")

    def test_control_count_and_distance_corruption_are_reported(self):
        with tempfile.TemporaryDirectory() as root:
            report=inspect_episode(*fixture(root,bad_controls=True,bad_dist=True))
            failures={c["name"] for c in report["checks"] if c["status"]=="error"}
            self.assertTrue({"Pairwise distances","Full control reward stream"} <= failures)
            self.assertTrue(all(v is None for v in report["series"]["target_return"]))

    def test_http_catalog_episode_frame_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as root:
            path,summary=fixture(root)
            catalog=Catalog(episodes=[str(path)])
            server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(catalog))
            worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
            url=f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url+"/api/catalog") as response:
                    self.assertEqual(len(json.load(response)),1)
                with urlopen(url+"/api/episode?id=0") as response:
                    self.assertEqual(json.load(response)["task"],"goal_to_goal")
                with urlopen(url+"/api/frame?id=0&frame=2") as response:
                    self.assertEqual(Image.open(BytesIO(response.read())).size,(32,32))
                with self.assertRaises(HTTPError): urlopen(url+"/api/frame?id=0&frame=-1")
                with self.assertRaises(HTTPError): urlopen(url+"/api/episode?id=../../etc/passwd")
            finally:
                server.shutdown(); server.server_close(); worker.join(timeout=2)


if __name__=="__main__": unittest.main()
