import importlib.util
from pathlib import Path
import pytest
import torch
from ma_vlcm.target_smoothing import smooth_targets
from ma_vlcm.trajectory_preference import trajectory_preference_loss
from ma_vlcm.train import SequenceWebDataset


def test_smoothing_is_causal_and_collision_is_immediate():
    prefix = [0, .2, .8, .1]
    short = smooth_targets(prefix, [False]*4)
    long = smooth_targets(prefix + [-.9, 1], [False]*6)
    assert short == long[:4]
    assert short[2] < .8
    assert smooth_targets([.8, -.5, .2], [False, True, False]) == [.8, -1., .2]
    assert smooth_targets([.4]*12, [False]*12) == pytest.approx([.4]*12)
    with pytest.raises(ValueError): smooth_targets([float('nan')], [False])


def test_preference_orientation_and_gradient():
    scores = torch.tensor([.8]*4 + [.1]*4, requires_grad=True)
    batch = {'preference_pair_id':torch.ones(8), 'preference_side':torch.tensor([0]*4+[1]*4), 'preference_label':torch.ones(8)}
    loss, accuracy = trajectory_preference_loss(scores,batch)
    assert accuracy == 1
    swapped = dict(batch, preference_side=1-batch['preference_side'], preference_label=1-batch['preference_label'])
    assert trajectory_preference_loss(scores,swapped)[0].item() == pytest.approx(loss.item())
    loss.backward()
    assert (scores.grad[:4] < 0).all() and (scores.grad[4:] > 0).all()
    with pytest.raises(ValueError):trajectory_preference_loss(scores,dict(batch, preference_pair_id=torch.arange(8)))


def test_old_checkpoint_configuration_preserved():
    from ma_vlcm.model import ModelConfig
    assert not ModelConfig().trajectory_preference_head


def test_loader_transform_is_explicit():
    ds = SequenceWebDataset([],8,8,'structured','state','mean','any', dataset_type='tb3_potential_v1', target_schema='tb3_potential_v1', apf_target_transform='causal_gaussian_sigma1_window5_collision_override')
    assert ds.apf_target_transform.startswith('causal_gaussian')


def test_preference_head_reaches_shared_encoder():
    from test_critic_improvements import tiny_model
    model = tiny_model(critic_readout='query', agent_pooling='attention_max', trajectory_preference_head=True)
    obs=torch.randn(8,2,3,9); adj=torch.eye(3).repeat(8,2,1,1)
    inputs={'input_ids':torch.tensor([[1,19,2]]*8), 'attention_mask':torch.ones(8,3,dtype=torch.long)}
    out=model(inputs,obs,adj,return_features=True)
    batch={'preference_pair_id':torch.ones(8),'preference_side':torch.tensor([0]*4+[1]*4),'preference_label':torch.ones(8)}
    trajectory_preference_loss(out['preference_score'],batch)[0].backward()
    assert model.preference_head[0].weight.grad.abs().sum()>0
    assert model.robot_gnn.linear.weight.grad.abs().sum()>0
    assert model.critic_query.grad.abs().sum()>0


def test_pair_tar_loader_preserves_clip_groups_and_labels(tmp_path):
    import io,json,tarfile
    from dataclasses import asdict
    from PIL import Image
    from itertools import islice
    from ma_vlcm.potential import PotentialConfig
    from ma_vlcm.target_smoothing import TRANSFORM
    from test_critic_improvements import FakeProcessor
    path=tmp_path/'pair.tar'
    with tarfile.open(path,'w') as tar:
        for side in range(2):
            for step in range(32):
                state={'task_domain':'goal_to_goal','agents':[{'id':0,'pos':[step*.01,0.],'goal':[1.,0.],'initial_goal_distance':1.,'dist_to_goal':1.}], 'episode_meta':{'step':step,'done':False}}
                label={'schema':'tb3_potential_v1','target':.1,'raw_target':.2,'target_transform':TRANSFORM,'goal_cost':1.,'agent_risk':0.,'obstacle_risk':0.,'collision':False,'potential_config':asdict(PotentialConfig()),'preference':{'pair_id':77,'side':side,'label':1}}
                image=io.BytesIO();Image.new('RGB',(8,8)).save(image,format='PNG')
                for suffix,data in [('state.json',json.dumps(state).encode()),('progress.json',json.dumps(label).encode()),('overhead.png',image.getvalue())]:
                    info=tarfile.TarInfo(f'pref77_side{side}_step{step:05d}.{suffix}');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    ds=SequenceWebDataset(str(path),8,8,'raw','obs','mean','any',vlm_processor=FakeProcessor(),dataset_type='tb3_potential_v1',target_mode='progress',target_schema='tb3_potential_v1',vl_backend='vjepa2',robot_obs_dim=9,num_robots=None,resize_width=8,resize_height=8,apf_target_transform=TRANSFORM)
    clips=list(islice(iter(ds),8))
    assert [c['preference_side'].item() for c in clips]==[0]*4+[1]*4
    assert [c['preference_pair_id'].item() for c in clips]==[77]*8
    assert all(c['progress'].item()==pytest.approx(.1) and c['raw_progress'].item()==pytest.approx(.2) for c in clips)
    assert all(c['robot_obs'].shape[0]==8 for c in clips)
    ds.apf_target_transform='raw'
    with pytest.raises(ValueError,match='transform'):next(iter(ds))


def test_collector_modes_recover_and_do_not_mutate_policy_actions():
    spec=importlib.util.spec_from_file_location('diversity',Path(__file__).resolve().parents[1]/'tools/collection/collection_diversity.py')
    if not Path(spec.origin).exists():pytest.skip('collector tests only in staged pipeline')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actions=torch.ones(3,3,2)*.4; original=actions.clone();steps=torch.tensor([0.,120.,270.])
    for mode in ('clean','noise_recover','stall_recover','retreat_recover','straight_recover'):
        result=module.perturb_actions(actions,steps,30,mode)
        assert torch.equal(result[0],actions[0]) and torch.equal(result[2],actions[2])
        assert torch.equal(actions,original)
        if mode!='clean':assert not torch.equal(result[1],actions[1])


def test_training_step_updates_preference_head_and_reports_accuracy(tmp_path):
    from unittest.mock import patch
    from types import SimpleNamespace
    from accelerate import Accelerator
    from test_critic_improvements import TinyAuxCritic
    from ma_vlcm.train import run_epoch
    class Critic(TinyAuxCritic):
        def __init__(self):
            super().__init__();self.preference_head=torch.nn.Linear(1,1)
        def forward(self,inputs,obs,adj,return_features=False):
            result=super().forward(inputs,obs,adj,return_features)
            result['preference_score']=self.preference_head(obs[:,-1,0,:1]).view(-1)
            return result
    model=Critic();obs=torch.zeros(2,2,1,9);adj=torch.ones(2,2,1,1)
    batch={'inputs':{},'robot_obs':obs,'adj':adj,'reward':torch.zeros(2),'done':torch.zeros(2),'progress':torch.tensor([.1,.3])}
    po=torch.zeros(8,2,1,9);po[:4,-1,0,0]=1
    pb={'inputs':{},'robot_obs':po,'adj':torch.ones(8,2,1,1),'preference_pair_id':torch.ones(8),'preference_side':torch.tensor([0]*4+[1]*4),'preference_label':torch.ones(8)}
    prefdir=tmp_path/'train';prefdir.mkdir();(prefdir/'dummy.tar').touch()
    args=SimpleNamespace(target_mode='progress',temporal_consistency_loss_weight=0.,conditional_lipschitz_loss_weight=0.,loss_type='categorical_ce',success_loss_weight=0.,value_output_activation='identity',value_support_min=-1.,value_support_max=1.,debug_save_video=False,max_grad_norm=0.,potential_aux_loss_weight=0.,event_loss_boost=1.,event_delta_threshold=.03,preference_loss_weight=.5,preference_interval=4,preference_shards=str(tmp_path),clip_len=8)
    before=model.preference_head.weight.detach().clone()
    with patch('ma_vlcm.train.webdataset_loader',return_value=[pb]):
        loss=run_epoch(model,[batch],torch.optim.SGD(model.parameters(),lr=.1),Accelerator(cpu=True),1,.95,args,train=True)
    assert not torch.equal(model.preference_head.weight,before)
    assert run_epoch.last_metrics['train/trajectory_preference_pairs']==1
    assert run_epoch.last_metrics['train/trajectory_preference_ce']>0
    run_epoch._preference_loaders={}
