import hashlib,json,shutil,sys
from pathlib import Path
run=Path(sys.argv[1]);p=json.loads((run/'PROFILE_READY.json').read_text());ready=json.loads((run/'READY.json').read_text())
assert ready['always_both'] and ready['train_from_scratch'] and not ready['resume_checkpoint']
assert p['clip_len']==8 and p['epochs']==75 and p['image_size']==ready['image_size'] and p['effective_batch_size']==128
assert p['value_and_gradient_equivalence']['passed']
assert (run/'CHECKS_READY.json').is_file()
for name in ['batch_size','grad_accum_steps','num_workers','disable_gradient_checkpointing','qwen_logits_to_keep']:ready[name]=p[name]
ready['optimization_profile']=str(run/'PROFILE_READY.json')
(run/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
home=Path('/home/aparame/Research/MA-VLCM');backup=Path('/scratch/aparame/VLCM/ma_vlcm/code_backups/qwen_clip8_full_epoch75_20261006')
installed=[];skipped=[]
for relative in ['src/ma_vlcm/train.py','src/ma_vlcm/model.py','src/ma_vlcm/distributional_value.py','src/ma_vlcm/critic_improvements.py','scripts/lora_run_train_tb3_lab.sh','scripts/lora_submit_train_tb3_qwen8_full_75.sh','tools/evaluate_tb3_episode_progress.py','tools/finalize_qwen8_full_profile.py','tests/test_critic_improvements.py']:
    dest=home/relative;original=backup/relative
    matches=(dest.exists() and original.exists() and dest.read_bytes()==original.read_bytes()) or (not dest.exists() and not original.exists())
    if matches:
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/'source'/relative,dest);installed.append(relative)
    else:skipped.append(relative)
shutil.copy2(run/'source/README_full_run.md',home/'README_full_run.md')
readme=home/'README.md'; text=readme.read_text()
link='\n\n### Qwen full accuracy run (2026-10-06)\n\nSee [README_full_run.md](README_full_run.md) for the fresh 8-frame, 224x224,\n75-epoch experiment, ALWAYS both datasets, profiling and matched ablations.\n'
if '### Qwen full accuracy run (2026-10-06)' not in text:readme.write_text(text+link)
manifest=json.loads((run/'experiment_manifest.json').read_text());manifest['training_settings']=ready
manifest['optimization_profile']={k:v for k,v in p.items() if k!='all_candidates'}
manifest['installed_home_files']=installed;manifest['preserved_home_changes']=skipped
manifest['source_files']={str(path.relative_to(run/'source')):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((run/'source').rglob('*')) if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts}
(run/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'batch_size':p['batch_size'],'grad_accum_steps':p['grad_accum_steps'],'installed':installed,'preserved_home_changes':skipped}))
