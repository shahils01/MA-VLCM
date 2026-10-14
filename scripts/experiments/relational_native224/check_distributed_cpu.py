"""Two-process test of actual progress/pair/preference training backward."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from accelerate import Accelerator,DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
run=Path(sys.argv[1]);sys.path.insert(0,str(run/'source/tests'))
from test_critic_improvements import tiny_model
from ma_vlcm.train import run_epoch
accelerator=Accelerator(cpu=True,gradient_accumulation_steps=4,
 kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
 dataloader_config=DataLoaderConfiguration(dispatch_batches=False))
assert accelerator.num_processes==2
torch.manual_seed(42)
model=tiny_model(critic_readout='query',agent_pooling='attention_max',relational_critic=True,
 temporal_dropout=0.,potential_aux_heads=True,trajectory_preference_head=True)
model.success_head.requires_grad_(False)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
model,optimizer=accelerator.prepare(model,optimizer)
def batch(size):
 obs=torch.randn(size,8,3,9)
 return dict(inputs=dict(input_ids=torch.tensor([[18,18,1,19,19,19,2]]*size),attention_mask=torch.ones(size,7,dtype=torch.long)),
  robot_obs=obs,adj=torch.eye(3).repeat(size,8,1,1),reward=torch.zeros(size),done=torch.zeros(size),
  progress=torch.linspace(-.5,.5,size),raw_progress=torch.linspace(-.5,.5,size),
  potential_components=torch.tensor([[.3,.2,.4,0.]]*size))
b=batch(2);n=batch(2)
b.update(next_inputs=n['inputs'],next_robot_obs=n['robot_obs'],next_adj=n['adj'],
 next_progress=n['progress']+.05,next_raw_progress=n['progress']+.05,next_potential_components=n['potential_components'])
pb=batch(8);pb.update(preference_pair_id=torch.ones(8),preference_side=torch.tensor([0]*4+[1]*4),preference_label=torch.ones(8))
args=SimpleNamespace(target_mode='progress',temporal_consistency_loss_weight=1.,conditional_lipschitz_loss_weight=0.,
 loss_type='categorical_ce',success_loss_weight=0.,value_output_activation='identity',value_support_min=-1.,value_support_max=1.,
 debug_save_video=False,max_grad_norm=1.,potential_aux_loss_weight=.5,event_loss_boost=3.,event_delta_threshold=.03,
 preference_loss_weight=.5,preference_interval=4,preference_shards=str(run/'cpu_preferences'),clip_len=8,
 relational_critic=True,ema_decay=.995)
if accelerator.is_main_process:
 p=Path(args.preference_shards)/'train';p.mkdir(parents=True,exist_ok=True);(p/'dummy.tar').touch()
accelerator.wait_for_everyone()
with patch('ma_vlcm.train.webdataset_loader',return_value=[pb]*3):
 loss=run_epoch(model,[b]*8,optimizer,accelerator,1,.95,args,train=True,max_steps=8)
assert torch.isfinite(torch.tensor(loss))
checksum=torch.stack([p.detach().sum() for p in model.parameters() if p.requires_grad]).sum().view(1)
checks=accelerator.gather(checksum);assert torch.allclose(checks,checks[0].expand_as(checks),rtol=1e-5,atol=.001)
if accelerator.is_main_process:
 (run/'CPU_DDP_READY.json').write_text(json.dumps(dict(passed=True,processes=2,microbatches=8,loss=loss,preference_pairs=run_epoch.last_metrics['train/trajectory_preference_pairs']))+'\n')
 print('Two-process progress, temporal and preference backward passed',flush=True)
