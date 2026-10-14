"""Freeze a matched, forward-only diagnostic subset from the training holdout."""
import collections, hashlib, io, json, random, re, shutil, tarfile
from pathlib import Path
RUN=Path('/scratch/aparame/VLCM/ma_vlcm/runs/qwen_relational_native224_2500_20261012_01')
ROOT=Path('/scratch/aparame/VLCM/ma_vlcm/evaluations/qwen_relational_native224_best_vs75_20261014')
ROOT.mkdir(parents=True,exist_ok=True)
assert not (ROOT/"PREPARATION_COMPLETE.json").exists(), "Already prepared"
if not (ROOT/'source').exists():shutil.copytree(RUN/'source',ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','.git'))
shutil.copy(RUN/'inference_history_epochs.json',ROOT/'training_history.json')
split=json.loads((RUN/'split_manifest.json').read_text())
assert not set(split['train']) & set(split['validation'])
candidates=[]
for i,path in enumerate(split['validation']):
    with tarfile.open(path) as archive:
        members=[m for m in archive.getmembers() if m.isfile()]
        groups=collections.defaultdict(list)
        for m in members:
            if m.name.endswith('.progress.json'):
                match=re.match(r'(.+)_step_?(\d+)\.progress\.json$',m.name)
                if match and 'rewind' not in match[1]: groups[match[1]].append((int(match[2]),m))
        for prefix,rows in groups.items():
            rows.sort(); labels={}
            for step,m in rows:
                payload=json.load(archive.extractfile(m))
                labels[str(step)]={k:payload.get(k) for k in ('target','raw_target','collision','episode_success','synthetic_rewind','target_transform')}
            if len(labels)<8 or any(x.get('synthetic_rewind',False) for x in labels.values()):continue
            state_member=next(m for m in members if m.name.startswith(prefix+'_step') and m.name.endswith('.state.json'))
            state=json.load(archive.extractfile(state_member))
            if state.get('synthetic_rewind',False):continue
            domain=state['task_domain'];meta=state.get('episode_meta',{});cp=state.get('checkpoint',{})
            values=[labels[str(step)]['target'] for step,m in rows]
            assert all(x.get('target_transform')=='causal_gaussian_sigma1_window5_collision_override' for x in labels.values())
            collision=any(x['collision'] for x in labels.values());success=bool(labels[str(rows[-1][0])]['episode_success'])
            uid=hashlib.sha256((path+'::'+prefix).encode()).hexdigest()[:12]+'_'+prefix
            dest=ROOT/'episodes'/domain/(uid+'.tar')
            record={'source_signed_shard':path,'relative_source':str(Path(path).relative_to(RUN/'data/validation')),
                'episode_prefix':prefix,'domain':domain,'agents':len(state['agents']), 'frames':len(rows),
                'scenario_id':meta.get('scenario_id'),'collision':collision,'episode_success':success,
                'max_drop':max([0.]+[values[j-1]-values[j] for j in range(1,len(values))]),
                'targets':labels,'episode_tar':str(dest),'source_id':'_'.join(dest.with_suffix('').parts[-3:]),
                'policy':Path(cp.get('source_directory','unknown')).name,'behavior':cp.get('behavior_mode','unknown'),
                'cohort':domain if cp.get('behavior_mode','clean')=='clean' else 'new_suboptimal'}
            candidates.append(record)
    if i%100==0:print('Indexed validation shards',i,flush=True)
rng=random.Random(20261014)
selected=[]
def pick(pool,n):
    # Round-robin outcome/collision/behavior strata, random within each stratum.
    buckets=collections.defaultdict(list)
    for r in pool:buckets[(r['episode_success'],r['collision'],r['behavior'])].append(r)
    for b in buckets.values():rng.shuffle(b)
    out=[];keys=sorted(buckets)
    while len(out)<n:
        before=len(out)
        for k in keys:
            if buckets[k] and len(out)<n:out.append(buckets[k].pop())
        assert len(out)>before,('insufficient eligible held-out episodes',n)
    return out
for agents,n in [(3,13),(4,13),(5,12),(6,12)]:
    selected+=pick([r for r in candidates if r['domain']=='goal_to_goal' and r['agents']==agents],n)
ao=[r for r in candidates if r['domain']=='static_obstacles']
policies=sorted({r['policy'] for r in ao});assert len(policies)==5,policies
for policy in policies:selected+=pick([r for r in ao if r['policy']==policy],10)
assert len(selected)==100 and len({r['source_id'] for r in selected})==100
for row in selected:
    dest=Path(row['episode_tar']);dest.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(row['source_signed_shard']) as src,tarfile.open(dest,'w') as out:
        for m in src.getmembers():
            if m.isfile() and m.name.startswith(row['episode_prefix']+'_step'):
                out.addfile(m,src.extractfile(m))
    row['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
checkpoints={}
for epoch in [22,63,75]:
    paths=list((RUN/'checkpoints/qwen3_vl').glob('*_epoch_'+str(epoch)+'.pt'));assert len(paths)==1
    config=json.loads(paths[0].with_name(paths[0].stem+'_info.json').read_text())['config']
    assert config['relational_critic'] and (config['clip_len'],config['resize_width'],config['resize_height'],config['lora_r'])==(8,224,224,32)
    checkpoints['epoch'+str(epoch)]=str(paths[0])
manifest={'training_run':str(RUN),'training_job_id':17141648,'wandb_run':'i2rLAB/VLCM_Training_RWARE/nku3ga22',
    'checkpoints':checkpoints,'episodes':selected,
    'coverage':{'episodes':100,'by_domain':dict(collections.Counter(r['domain'] for r in selected)),
      'by_domain_agents':dict(collections.Counter(r['domain']+'_'+str(r['agents']) for r in selected)),
      'ao_policy':dict(collections.Counter(r['policy'] for r in selected if r['domain']=='static_obstacles')),
      'success':sum(r['episode_success'] for r in selected),'collision':sum(r['collision'] for r in selected)},
    'selection':'Seed20261014, forward-only from frozen validation split. GTG agent-count stratified; AO five behavior policies balanced. Outcome/collision/behavior round-robin strata. Diagnostic subset, not population-weighted.',
    'target':'Training-time causal Gaussian APF labels; no extra inference filtering or episode normalization',
    'checkpoint_selection':{'epoch22':'minimum composite validation loss and categorical CE','epoch63':'minimum scalar validation MSE','epoch75':'final checkpoint'}}
(ROOT/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'PREPARATION_COMPLETE.json').write_text(json.dumps({'status':'completed','episodes':100,'forward_only':True,'source_split':'validation','coverage':manifest['coverage']},indent=2))
print(json.dumps(manifest['coverage']),flush=True)
