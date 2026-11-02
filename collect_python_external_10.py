"""Collect 10 held-out Python formation rollouts, exactly three with action noise."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
COLLECTOR_DIR = ROOT.parent / 'Ground_Robot_MARL/scripts/vlcm'
sys.path.insert(0, str(COLLECTOR_DIR))
from collect_python_formation_dataset import collect_job, prepare
from validate_python_formation_dataset import audit_episode


def main():
    output = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/python_frozen_external_10').resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'Refusing to overwrite nonempty directory: {output}')
    plan = prepare(output, episodes=12, seed=93000000)
    # Fresh reset/noise seeds lie well outside training's 10310000–10312999.
    cases = [
        ('naive', 3, 'clean'), ('naive', 4, 'clean'), ('naive', 5, 'clean'),
        ('intermediate', 3, 'clean'), ('intermediate', 4, 'clean'),
        ('intermediate', 6, 'mild'), ('successful', 3, 'clean'),
        ('successful', 4, 'clean'), ('successful', 5, 'moderate'),
        ('successful', 6, 'mild'),
    ]
    jobs = [dict(episode_id=100000 + i, seed=93000000 + i,
                 noise_seed=103000000 + i, stage=stage, num_agents=n,
                 noise=noise, worker_id=i)
            for i, (stage, n, noise) in enumerate(cases)]
    plan.update(episodes=10, jobs=jobs, intended_use='held-out external evaluation only',
                training_seed_range=[10310000, 10312999])
    (output / 'collection_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    manifest_path = output / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(episodes=10, intended_use=plan['intended_use'], training_seed_range=plan['training_seed_range'])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    policies = {p['stage']: p for p in plan['checkpoints']}
    summaries = []
    for job in jobs:
        summary = collect_job(output, job, policies[job['stage']])
        summaries.append(summary)
        print(f"episode {job['episode_id']}: {job['num_agents']} robots, {job['stage']}, "
              f"{job['noise']}, {summary['termination_reason']}", flush=True)
    audits = [audit_episode(str(output), job) for job in jobs]
    result = dict(episodes=len(summaries), noise_profiles=dict(Counter(r['noise'] for r in summaries)),
                  outcomes=dict(Counter(r['termination_reason'] for r in summaries)),
                  control_steps=sum(r['control_steps'] for r in summaries),
                  frames=sum(r['samples'] for r in summaries),
                  audit_passed=len(audits) == 10, reset_seeds=[r['seed'] for r in summaries],
                  training_seed_range=plan['training_seed_range'],
                  source_collector_sha256=hashlib.sha256((output/'source/collect_python_formation_dataset.py').read_bytes()).hexdigest())
    (output / 'validation_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
