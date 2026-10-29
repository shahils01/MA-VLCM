"""Evaluate the supplied Python MAPPO actor on four formations, without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent/'MARL_data_collection/on-policy'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'ckpts/formation_control/mappo_generic_polygon_3to6_seed1/polygon/step_008001600')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/python_mappo_baseline'))
    p.add_argument('--episodes',type=int,default=15)
    p.add_argument('--seed',type=int,default=8300001)
    p.add_argument('--include-instant-success',action='store_true')
    a=p.parse_args()
    if a.episodes < 1: p.error('--episodes must be positive')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    checkpoint=a.checkpoint.resolve()
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT.parent)+os.pathsep+str(ROOT)
    modes=[('current_hold20',[])]
    if a.include_instant_success:
        modes.append(('instant_success_no_settling',['--formation_hold_steps','1','--settled_action_penalty_w','0','--leave_settle_penalty_w','0','--stable_team_reward_w','0']))
    commands=[]
    for mode,extra in modes:
        for n in (3,4,5,6):
            target=out/f'{mode}_agents_{n:02d}.json'
            cmd=[sys.executable,str(ROOT/'onpolicy/scripts/eval/eval_formation_control.py'),
                 '--checkpoint',str(checkpoint),'--algorithm_name','mappo','--pattern','polygon',
                 '--num_agents',str(n),'--max_num_agents','6','--formation_spacing','1.0',
                 '--formation_tolerance','0.05','--initial_position_range','1.5',
                 '--min_start_separation','0.55','--observation_scale','3.0','--dt','0.05',
                 '--eval_episodes',str(a.episodes),'--env_max_steps','300','--seed',str(a.seed),
                 '--n_training_threads','1','--no_animation','--metrics_path',str(target),*extra]
            commands.append(dict(mode=mode,num_agents=n,argv=cmd))
            with target.with_suffix('.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(mode,n,json.loads(target.read_text())['summary'],flush=True)
    metadata=dict(checkpoint=str(checkpoint),actor_sha256=hashlib.sha256((checkpoint/'actor.pt').read_bytes()).hexdigest(),
                  environment_sha256=hashlib.sha256((ROOT.parent/'RL_bound_overwatch/envs/FormationControlEnv.py').read_bytes()).hexdigest(),
                  commands=commands,python=sys.executable,episodes_per_case=a.episodes,seed_start=a.seed,
                  interpretation='Evaluation only. Instant-success mode is a compatibility ablation, not an exact historical environment reconstruction.')
    (out/'evaluation_contract.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
