"""Summarize matched causal predictions; no normalization or smoothing."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]);manifest=json.loads((root/'evaluation_manifest.json').read_text())
families=list(manifest['checkpoints']);records=[];curves={}
for episode in manifest['episodes']:
    matched={}
    for family in families:
        paths=list((root/'results'/family).glob('episode_*_'+episode['source_id']+'/progress.csv'))
        assert len(paths)==1, (family,episode['source_id'],paths)
        with paths[0].open() as handle:
            rows=[r for r in csv.DictReader(handle) if r.get('prediction_qwen3_vl')]
        steps=np.array([int(r['step']) for r in rows])
        t=np.array([float(r['target']) for r in rows]);p=np.array([float(r['prediction_qwen3_vl']) for r in rows])
        assert len(t)>0 and np.isfinite(p).all() and np.isfinite(t).all()
        assert np.allclose(t,[episode['targets'][str(s)]['target'] for s in steps],atol=1e-6)
        collision=np.array([episode['targets'][str(s)]['collision'] for s in steps],dtype=bool)
        dt,dp=np.diff(t),np.diff(p)
        assert np.all(np.diff(steps)==1), 'Change metrics require adjacent frames'
        drops=dt<=-.03;recoveries=dt>=.03;quiet=np.abs(dt)<.03;events=np.abs(dt)>=.03
        metric={'checkpoint':family,'source_id':episode['source_id'],'domain':episode['domain'],
            'cohort':episode['cohort'],'samples':len(t),'mae':float(np.mean(np.abs(p-t))),
            'mse':float(np.mean((p-t)**2)),'rmse':float(np.sqrt(np.mean((p-t)**2))),
            'change_mae':float(np.mean(np.abs(dp-dt))) if len(dt) else None,
            'drop_pairs':int(drops.sum()),'drop_hits':int((dp[drops]<0).sum()),
            'recovery_pairs':int(recoveries.sum()),'recovery_hits':int((dp[recoveries]>0).sum()),
            'quiet_pairs':int(quiet.sum()),'false_drops':int((dp[quiet]<=-.03).sum()),
            'abrupt_pairs':int(events.sum()),'abrupt_change_abs_error_sum':float(np.abs(dp[events]-dt[events]).sum()),
            'collision_frames':int(collision.sum()),'collision_abs_error_sum':float(np.abs(p[collision]-t[collision]).sum()),
            'collision_negative_predictions':int((p[collision]<0).sum())}
        records.append(metric);matched[family]=(steps,t,p)
    for family in families[1:]:
        assert np.array_equal(matched[families[0]][0],matched[family][0])
        assert np.allclose(matched[families[0]][1],matched[family][1],atol=1e-6)
    curves[episode['source_id']]=matched

def ratio(a,b):return a/b if b else None
def aggregate(items):
    total=lambda key:sum(r[key] for r in items)
    return {'episodes':len(items),'frames':total('samples'),
        'episode_balanced_mae':float(np.mean([r['mae'] for r in items])),
        'episode_balanced_rmse':float(np.sqrt(np.mean([r['mse'] for r in items]))),
        'episode_balanced_change_mae':float(np.mean([r['change_mae'] for r in items if r['change_mae'] is not None])),
        'drop_recall':ratio(total('drop_hits'),total('drop_pairs')),'drop_pairs':total('drop_pairs'),
        'recovery_recall':ratio(total('recovery_hits'),total('recovery_pairs')),'recovery_pairs':total('recovery_pairs'),
        'false_drop_rate':ratio(total('false_drops'),total('quiet_pairs')),'quiet_pairs':total('quiet_pairs'),
        'abrupt_change_mae':ratio(total('abrupt_change_abs_error_sum'),total('abrupt_pairs')),'abrupt_pairs':total('abrupt_pairs'),
        'collision_mae':ratio(total('collision_abs_error_sum'),total('collision_frames')),
        'collision_negative_prediction_rate':ratio(total('collision_negative_predictions'),total('collision_frames')),
        'collision_frames':total('collision_frames')}

summary={'coverage':manifest['coverage'],'change_threshold':.03,
         'selection':manifest['selection'],'note':'Stratified diagnostic holdout; recall is pair-pooled with denominators; value errors are episode-balanced. Targets were causally smoothed during preparation; predictions are unfiltered.',
         'checkpoints':{}}
for family in families:
    subset=[r for r in records if r['checkpoint']==family]
    summary['checkpoints'][family]={'overall':aggregate(subset),
        'by_domain':{d:aggregate([r for r in subset if r['domain']==d]) for d in ['goal_to_goal','static_obstacles']},
        'by_cohort':{d:aggregate([r for r in subset if r['cohort']==d]) for d in sorted({r['cohort'] for r in subset})}}
(root/'episode_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
(root/'metrics_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


# Ground-truth-only example selection, no selection on model errors.
examples=[]
for domain in ['goal_to_goal','static_obstacles']:
    for collision in [False,True]:
        pool=[r for r in manifest['episodes'] if r['domain']==domain and r['collision']==collision]
        examples.append(max(pool,key=lambda r:r['max_drop']))
colors={'epoch22':'#16885b','epoch63':'#8a4ca8','epoch75':'#df7d22'}
def draw_examples(examples,path,title):
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,episode in zip(axes.flat,examples):
        for family in families:
            steps,t,p=curves[episode['source_id']][family]
            if family==families[0]:ax.plot(steps,t,color='#1867b8',lw=2.3,label='Smoothed APF target')
            ax.plot(steps,p,color=colors[family],lw=1.5,label=family.replace('epoch','Epoch '))
        ax.set(title=episode['domain'].replace('_',' ')+' · '+str(episode['agents'])+' agents · '+('collision' if episode['collision'] else 'no collision')+' · '+episode['episode_prefix'],
            xlabel='Episode step',ylabel='Signed potential',ylim=(-1.05,1.05))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(title,fontsize=15)
    fig.text(.5,.015,'8 frames · 224×224 · causal predictions · no output filtering or episode normalization',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(path,dpi=170);plt.close(fig)
draw_examples(examples,root/'comparison_overview.png','Relational Qwen: epochs 22, 63 and 75 · matched 100-episode holdout')
for domain in ['goal_to_goal','static_obstacles']:
    pool=[r for r in manifest['episodes'] if r['domain']==domain]
    if domain=='goal_to_goal':
        examples=[max([r for r in pool if r['agents']==a],key=lambda r:r['max_drop']) for a in [3,4,5,6]]
    else:
        examples=[max([r for r in pool if r['episode_success']==success and r['collision']==collision],key=lambda r:r['max_drop'])
            for success,collision in [(True,False),(True,True),(False,False),(False,True)]
            if any(r['episode_success']==success and r['collision']==collision for r in pool)]
    draw_examples(examples,root/(domain+'_comparison.png'),domain.replace('_',' ').title()+' · epochs 22 / 63 / 75')
# Per-cardinality metrics complement the aggregate domain comparison.
for family in families:
    subset=[r for r in records if r['checkpoint']==family]
    ids={r['source_id']:r for r in manifest['episodes']}
    summary['checkpoints'][family]['by_domain_agents']={
        domain+'_'+str(a):aggregate([r for r in subset if r['domain']==domain and ids[r['source_id']]['agents']==a])
        for domain in ['goal_to_goal','static_obstacles'] for a in [3,4,5,6]
        if any(r['domain']==domain and ids[r['source_id']]['agents']==a for r in subset)}
(root/'metrics_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
