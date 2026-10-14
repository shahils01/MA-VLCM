"""Real two-GPU profile: training, preference, validation and strict state reload."""
import json,math,shlex,sys,time
from pathlib import Path
import torch
from accelerate import Accelerator,DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from ma_vlcm.train import parse_args,build_model,_apply_peft,webdataset_loader,run_epoch,_reset_value_head_for_progress
run=Path(sys.argv[1])
command=next(line.split(':',1)[1].strip() for line in (run/'dry_run.txt').read_text().splitlines() if line.startswith('Training command:'))
tokens=shlex.split(command);sys.argv=['profile']+tokens[tokens.index('ma_vlcm.train')+1:];args=parse_args()
assert args.relational_critic and args.lora_r==32 and args.epochs==75 and args.clip_len==8 and not args.resume_from
assert args.batch_size==32 and args.grad_accum_steps==4 and args.resize_width==args.resize_height==224
accelerator=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=4,
 kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
 dataloader_config=DataLoaderConfiguration(dispatch_batches=False,non_blocking=True))
assert accelerator.num_processes==2, 'profile must exercise two real distributed GPUs'
torch.manual_seed(42)
model=_apply_peft(build_model(args,accelerator.device),args).to(accelerator.device)
_reset_value_head_for_progress(model,accelerator)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay)
train=webdataset_loader(args,args.train_shards,args.batch_size,args.num_workers,shuffle=True,dataset_type=args.dataset_type)
val=webdataset_loader(args,args.val_shards,args.batch_size,args.num_workers,shuffle=False,dataset_type=args.dataset_type)
model,optimizer,train,val=accelerator.prepare(model,optimizer,train,val)
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter()
loss=run_epoch(model,train,optimizer,accelerator,1,args.gamma,args,train=True,max_steps=12)
train_metrics=dict(run_epoch.last_metrics)
assert math.isfinite(loss) and train_metrics['train/trajectory_preference_pairs']>=3
train_seconds=time.perf_counter()-start
validation_loss=run_epoch(model,val,optimizer,accelerator,1,args.gamma,args,train=False,max_steps=8)
assert math.isfinite(validation_loss)
raw=accelerator.unwrap_model(model)
# Trainable weights must be synchronized across ranks after optimizer steps.
checksum=torch.stack([p.detach().float().sum() for p in raw.parameters() if p.requires_grad]).sum().view(1)
checksums=accelerator.gather(checksum)
assert torch.allclose(checksums,checksums[0].expand_as(checksums),atol=.01,rtol=1e-5)
checkpoint=run/'profiling/profile_checkpoint.pt'
if accelerator.is_main_process:
 torch.save({'model':accelerator.get_state_dict(model),'args':vars(args),'profile_only':True},checkpoint)
accelerator.wait_for_everyone()
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
raw.load_state_dict(state['model'],strict=True);del state
batch=next(iter(val));model.eval()
inputs={k:v.to(accelerator.device) if torch.is_tensor(v) else v for k,v in batch['inputs'].items()}
with torch.no_grad(),accelerator.autocast():
 output=model(inputs,batch['robot_obs'].to(accelerator.device),batch['adj'].to(accelerator.device),return_features=True)
assert torch.isfinite(output['value']).all()
gib=1024**3
local={'rank':accelerator.process_index,'device_name':torch.cuda.get_device_name(accelerator.device),
 'device_total_gib':torch.cuda.get_device_properties(accelerator.device).total_memory/gib,
 'peak_allocated_gib':torch.cuda.max_memory_allocated()/gib,'peak_reserved_gib':torch.cuda.max_memory_reserved()/gib}
assert local['peak_allocated_gib']<.9*local['device_total_gib'], 'profile exceeded memory margin'
(run/'profiling'/('rank_'+str(accelerator.process_index)+'.json')).write_text(json.dumps(local,indent=2)+'\n')
accelerator.wait_for_everyone()
if accelerator.is_main_process:
 result={'status':'passed','num_processes':2,'batch_size_per_gpu':32,'grad_accum_steps':4,
  'effective_batch_size':256,'train_loss':loss,'val_loss':validation_loss,'lora_r':32,
  'relational_critic':True,'train_seconds':train_seconds,'global_clips_per_second':12*32*2/train_seconds,
  'train_metrics':train_metrics,'strict_checkpoint_reload':True,
  'ranks':[json.loads((run/'profiling'/('rank_'+str(i)+'.json')).read_text()) for i in range(2)]}
 (run/'PROFILE_READY.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
accelerator.wait_for_everyone()
