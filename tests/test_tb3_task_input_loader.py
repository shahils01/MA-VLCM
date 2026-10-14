"""Metamorphic test of the actual Qwen loader model-input boundary."""
import io
import json
import tarfile
from dataclasses import asdict
import pytest
import torch
from PIL import Image
from ma_vlcm.potential import PotentialConfig
from ma_vlcm.train import SequenceWebDataset


class RecordingQwenProcessor:
    def __init__(self):self.text=[]
    def apply_chat_template(self,messages,**kwargs):
        text=messages[0]['content'][1]['text'];self.text.append(text)
        return {'input_ids':torch.tensor([[1,2,3,4,5]]),
                'attention_mask':torch.ones(1,5,dtype=torch.long),
                'pixel_values_videos':torch.zeros(1,2,3,8,8)}


def write_shard(path,flag):
    with tarfile.open(path,'w') as tar:
        for step in range(3):
            state={'task_domain':'static_obstacles','num_agents':3,'num_static_obstacles':3,
                'agents':[{'id':i,'pos':[i*.4,0.],'goal_pos':[1.,1.],'yaw':0.,'vel':[0.,0.],
                           'dist_to_goal':1.,'initial_goal_distance':1.5,'min_neighbor_dist':.4,
                           'collision':flag,'reached':flag} for i in range(3)],
                'episode_meta':{'episode_id':'same','step':step,'success':flag,
                                'failure':flag,'outcome':'success' if flag else 'running',
                                'termination_reason':'collision' if flag else ''}}
            label={'schema':'tb3_potential_v1','target':-1. if flag else .4,
                   'potential_config':asdict(PotentialConfig())}
            image=io.BytesIO();Image.new('RGB',(8,8),color=(10,20,30)).save(image,format='PNG')
            for suffix,data in [('state.json',json.dumps(state).encode()),
                                ('progress.json',json.dumps(label).encode()),
                                ('reward.json',json.dumps(-100 if flag else 1).encode()),
                                ('overhead.png',image.getvalue())]:
                member=tarfile.TarInfo(f'ep_step{step:05d}.{suffix}');member.size=len(data)
                tar.addfile(member,io.BytesIO(data))


@pytest.mark.parametrize('relational',[False,True])
@pytest.mark.parametrize('template',[None,'Reach assigned goals while avoiding obstacles and teammates.'])
def test_collision_outcome_and_reward_changes_never_change_qwen_inputs(tmp_path,relational,template):
    samples=[];texts=[]
    for flag in [False,True]:
        path=tmp_path/f'{flag}.tar';write_shard(path,flag)
        processor=RecordingQwenProcessor()
        ds=SequenceWebDataset(str(path),2,1,'raw','obs','mean','any',
            vlm_processor=processor,vl_backend='qwen3_vl',relational_critic=relational,
            dataset_type='tb3_potential_v1',target_mode='progress',
            target_schema='tb3_potential_v1',robot_obs_dim=9,num_robots=None,
            text_prompt_template=template,resize_width=8,resize_height=8)
        samples.append(next(iter(ds)));texts.append(processor.text[0])
    assert texts[0]==texts[1]
    assert texts[0].count('<obs>')==(3 if relational else 1)
    for key in ['collision:','reached:','outcome','termination','dist_to_goal','reward:','success:']:
        assert key not in texts[0]
    for key in samples[0]['inputs']:
        assert torch.equal(samples[0]['inputs'][key],samples[1]['inputs'][key])
    for key in ['robot_obs','adj']:assert torch.equal(samples[0][key],samples[1][key])
    # Different labels are still available to the loss and evaluator.
    assert samples[0]['progress'].item()==pytest.approx(.4)
    assert samples[1]['progress'].item()==pytest.approx(-1.)
