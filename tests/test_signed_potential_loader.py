import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import asdict

import torch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from ma_vlcm.potential import PotentialConfig
from ma_vlcm.train import (SequenceWebDataset, _apply_value_output_activation,
    _progress_target_from_sample, _target_contract_changed, _build_tb3_prompt)


class FakeProcessor:
    def __call__(self, videos, return_tensors):
        return {"pixel_values_videos":torch.zeros((1,len(videos),3,8,8))}


class SignedPotentialLoaderTests(unittest.TestCase):
    def test_tanh_and_checkpoint_migration(self):
        args=SimpleNamespace(target_mode="progress",target_schema="tb3_potential_v1",
                             value_output_activation="tanh",potential_config=asdict(PotentialConfig()))
        values=_apply_value_output_activation(torch.tensor([-2.,0.,2.]),args)
        self.assertLess(values[0].item(),0.)
        self.assertGreater(values[2].item(),0.)
        checkpoint={"args":dict(vars(args))}
        self.assertFalse(_target_contract_changed(checkpoint,args))
        checkpoint["args"]["target_schema"]="tb3_progress_v2"
        self.assertTrue(_target_contract_changed(checkpoint,args))
        checkpoint["args"]=dict(vars(args))
        checkpoint["args"]["potential_config"]=dict(args.potential_config,agent_weight=4.)
        self.assertTrue(_target_contract_changed(checkpoint,args))

    def test_label_schema_and_configuration_are_strict(self):
        config=asdict(PotentialConfig())
        sample={"progress.json":{"schema":"tb3_potential_v1","target":-.5,"potential_config":config}}
        self.assertEqual(_progress_target_from_sample(sample,{},"tb3_potential_v1",potential_config=config),-.5)
        with self.assertRaisesRegex(ValueError,"legacy"):
            _progress_target_from_sample({}, {},"tb3_potential_v1")
        with self.assertRaisesRegex(ValueError,"configuration"):
            _progress_target_from_sample(sample,{},"tb3_potential_v1",potential_config={})

    def test_signed_prompt_has_no_time_or_outcome_shortcut(self):
        text=_build_tb3_prompt({"agents":[{"id":0}],"task_domain":"goal_to_goal"},{},[],123,"success","all_reached",target_schema="tb3_potential_v1")
        self.assertIn("assigned goal",text)
        self.assertNotIn("potential",text)
        self.assertNotIn("progress",text)
        self.assertNotIn("Timestep:",text)
        self.assertNotIn("outcome token",text)
        self.assertNotIn("all_reached",text)

    def test_loader_preserves_negative_labels_and_adjacent_targets(self):
        config=asdict(PotentialConfig())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"episode.tar"
            with tarfile.open(path,"w") as tar:
                for step,target in enumerate((-.2,-.4,-.6)):
                    state={"task_domain":"goal_to_goal","agents":[{"id":0,"pos":[0.,0.],"dist_to_goal":1.,"initial_goal_distance":1.}],"episode_meta":{"episode_id":"ep","step":step,"done":False}}
                    label={"schema":"tb3_potential_v1","target":target,"initial_distances":[1.],"potential_config":config,"episode_success":False}
                    stream=io.BytesIO()
                    Image.new("RGB",(8,8)).save(stream,format="PNG")
                    for suffix,payload in (("state.json",json.dumps(state).encode()),("progress.json",json.dumps(label).encode()),("overhead.png",stream.getvalue())):
                        info=tarfile.TarInfo(f"ep_step{step:05d}.{suffix}")
                        info.size=len(payload)
                        tar.addfile(info,io.BytesIO(payload))
            dataset=SequenceWebDataset(str(path),clip_len=2,clip_stride=1,text_mode="raw",robot_source="obs",reward_reduce="mean",done_reduce="any",dataset_type="tb3_progress_v2",target_mode="progress",target_schema="tb3_potential_v1",potential_config=config,vlm_processor=FakeProcessor(),vl_backend="vjepa2",robot_obs_dim=9,num_robots=None,max_num_robots=None,include_next=True,resize_width=8,resize_height=8)
            sample=next(iter(dataset))
            self.assertAlmostEqual(sample["progress"].item(),-.4,places=6)
            self.assertAlmostEqual(sample["next_progress"].item(),-.6,places=6)


if __name__=="__main__":
    unittest.main()
