"""Collect separate, newly seeded Isaac rollouts using a frozen source actor/scene.

No training, uploading, outcome filtering, or relabeling of existing episodes.
Preparation uses the standard library; --collect requires Isaac Lab and a GPU.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent.parent / 'Ground_Robot_MARL'


def prepare(source, output, stage, episodes, seed):
    source, output = source.resolve(), output.resolve()
    source_plan = json.loads((source / 'collection_plan.json').read_text())
    entries = [e for e in source_plan['checkpoints'] if e['stage'] == stage]
    entries.sort(key=lambda e: e['num_agents'])
    if [e['num_agents'] for e in entries] != [3, 4, 5, 6]:
        raise ValueError('Source must contain one entry per team size for the requested policy stage')
    old_seeds = {int(e['seed']) for e in source_plan['checkpoints']}
    for path in source.rglob('episode_*.summary.json'):
        old_seeds.add(int(json.loads(path.read_text())['seed']))
    seeds = [seed + 100000 * i for i in range(4)]
    if old_seeds.intersection(seeds):
        raise ValueError('Requested seeds overlap the source collection')
    plan_path = output / 'collection_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (plan.get('validation_request') != dict(source=str(source), stage=stage, per_formation=episodes, base_seed=seed)):
            raise ValueError('Existing output has another collection request')
        return plan
    output.mkdir(parents=True, exist_ok=True)
    actor_source = Path(entries[0]['checkpoint'])
    actor = output / 'checkpoints' / stage / 'actor.pt'
    actor.parent.mkdir(parents=True, exist_ok=True)
    for name in ['actor.pt', 'training_metadata.json']:
        shutil.copy2(actor_source.with_name(name), actor.with_name(name))
    if hashlib.sha256(actor.read_bytes()).hexdigest() != entries[0]['sha256']:
        raise ValueError('Source actor does not match its frozen collection plan')
    policy = deepcopy(next(p for p in source_plan['policies'] if p['stage'] == stage))
    policy['checkpoint'] = str(actor)
    plan = deepcopy(source_plan)
    plan['policies'] = [policy]
    plan['checkpoints'] = []
    for i, entry in enumerate(entries):
        entry = deepcopy(entry)
        entry.update(checkpoint=str(actor), worker_id=i, seed=seeds[i], episodes=episodes,
                     start_episode_id=seed * 100 + i * episodes)
        plan['checkpoints'].append(entry)
    plan.update(total_episodes=4 * episodes, hf_repo=None,
                validation_request=dict(source=str(source), stage=stage, per_formation=episodes, base_seed=seed),
                intended_use='external validation only; keep separate from training',
                source_plan=str(source / 'collection_plan.json'))
    (output / 'checkpoint_scores.json').write_text((source / 'checkpoint_scores.json').read_text())
    plan_path.write_text(json.dumps(plan, indent=2))
    (output / 'README.md').write_text(
        '# Fresh formation validation rollouts\n\n'
        f'Planned: {episodes} new episodes for each of triangle, square, pentagon and hexagon.\n\n'
        f'Actor stage: {stage}; base seed: {seed}. Seeds per formation: {seeds}.\n\n'
        'The source environment configuration, image resolution, capture rate, actor and reward semantics are preserved. '
        'The source pilot scores describe the previously measured frozen actors, not the new validation episodes. '
        'All new outcomes are retained; success/failure quotas are not imposed. '
        'Episodes share a seeded simulator stream within each formation; the seed is not a per-episode independent reset seed. '
        'High episode IDs avoid collisions with saved train/validation/test path keys.\n\n'
        'Do not add this dataset to training if it is used for model selection.\n')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 'datasets/formation_vlcm_polygon_3to6_10hz_inner00m')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'datasets/formation_validation_seed7300001')
    parser.add_argument('--stage', choices=['initial', 'intermediate', 'trained', 'finetuned'], default='intermediate')
    parser.add_argument('--episodes-per-formation', type=int, default=15)
    parser.add_argument('--seed', type=int, default=7300001)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--isaaclab-sh', type=Path, default=Path('/home/anshul/IsaacLab/isaaclab.sh'))
    parser.add_argument('--conda-prefix', type=Path, default=Path('/home/anshul/miniconda3/envs/env_isaaclab'))
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if min(args.episodes_per_formation, args.num_envs) < 1 or args.seed < 0:
        parser.error('Episode and environment counts must be positive; seed nonnegative')
    root = args.output_dir.resolve()
    plan = prepare(args.source, root, args.stage, args.episodes_per_formation, args.seed)
    print(f'Plan: {root / "collection_plan.json"}', flush=True)
    if not args.collect:
        print('Prepared only. Add --collect to generate new simulator episodes.', flush=True)
        return
    env = os.environ.copy()
    env.update(CONDA_PREFIX=str(args.conda_prefix), PYTHONNOUSERSITE='1',
               DISTMARL_TURTLEBOT_USD=str(PROJECT / 'usd/turtlebot3.usd'))
    if env.get('TERM', 'dumb') == 'dumb':
        env['TERM'] = 'xterm-256color'
    for entry in plan['checkpoints']:
        subprocess.run([str(args.isaaclab_sh), '-p', str(PROJECT / 'scripts/vlcm/collect_formation_dataset.py'),
                        '--backend', 'onpolicy', '--plan', str(root / 'collection_plan.json'),
                        '--checkpoint-label', entry['label'], '--num-envs', str(args.num_envs), '--headless'],
                       cwd=PROJECT, env=env, check=True)
    subprocess.run([sys.executable, str(PROJECT / 'scripts/vlcm/formation_dataset_pipeline.py'),
                    'verify', '--output-dir', str(root)], cwd=PROJECT, env=env, check=True)
    groups = {}
    for path in sorted(root.rglob('episode_*.summary.json')):
        summary = json.loads(path.read_text())
        shape = summary['formation']['target_shape']
        groups.setdefault(shape, []).append(dict(archive=str(path.with_suffix('').with_suffix('.tar')),
                                                   seed=summary['seed'], success=summary['success'],
                                                   termination_reason=summary['termination_reason']))
    (root / 'episodes_by_formation.json').write_text(json.dumps(groups, indent=2))
    print(f'Collected {sum(map(len, groups.values()))} episodes: {root}', flush=True)


if __name__ == '__main__':
    main()
