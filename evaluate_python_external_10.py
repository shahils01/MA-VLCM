"""Evaluate a frozen supervised VLCM on the held-out Python formation episodes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import read_episode

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'outputs/python_frozen_external_10')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/python_frozen_external_10_evaluation')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'outputs/cache_python_external_10')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['model_config']['backbone_mode'] != 'frozen':
        raise ValueError('Expected a frozen-backbone supervised VLCM checkpoint')
    run_args = checkpoint['run']['args']
    if run_args['return_source'] != 'control':
        raise ValueError('Expected a checkpoint trained on full control-step returns')
    dataset = args.dataset.resolve()
    collection = json.loads((dataset / 'validation_report.json').read_text())
    if not collection.get('audit_passed') or collection['episodes'] != 10:
        raise ValueError('Expected the audited 10-episode held-out collection')
    if collection['noise_profiles'].get('clean', 0) != 7:
        raise ValueError('Expected seven clean and three noisy episodes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--checkpoint', str(checkpoint_path),
                    '--dataset', str(dataset), '--split', 'external', '--expected-episodes', '10',
                    '--cache-dir', str(args.cache_dir), '--output-dir', str(args.output_dir),
                    '--batch-size', str(args.batch_size)], check=True)
    with (args.output_dir / 'external_predictions.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    per_episode = defaultdict(list)
    for row in rows:
        per_episode[row['episode']].append(row)
    if len(per_episode) != 10:
        raise ValueError(f'Expected predictions for 10 episodes, got {len(per_episode)}')
    groups = defaultdict(list)
    results = []
    fig, axes = plt.subplots(5, 2, figsize=(14, 17), constrained_layout=True)
    for ax, (name, samples) in zip(axes.flat, sorted(per_episode.items())):
        path = Path(name)
        summary = json.loads(path.with_suffix('.summary.json').read_text())
        episode = read_episode(path, path.with_suffix('.summary.json'))
        from data_utils import frame_indices
        selected = frame_indices(episode, run_args['image_hz'])
        samples.sort(key=lambda r: int(r['anchor']))
        predicted = np.asarray([float(r['predicted_value']) for r in samples])
        actual = np.asarray([float(r['target_return']) for r in samples])
        times = [float(episode['timestamps'][selected[int(r['anchor'])]]) for r in samples]
        error = predicted - actual
        category = 'clean' if summary['noise'] == 'clean' else 'perturbed'
        groups[category].extend(error.tolist())
        groups[summary['noise']].extend(error.tolist()) if category != 'clean' else None
        results.append(dict(episode_id=summary['episode_id'], seed=summary['seed'],
                            num_agents=summary['num_agents'], stage=summary['stage'],
                            noise=summary['noise'], outcome=summary['termination_reason'],
                            anchors=len(samples), mse=float(np.mean(error**2)),
                            mae=float(np.mean(np.abs(error))), initial_prediction=float(predicted[0]),
                            initial_realized_return=float(actual[0])))
        ax.plot(times, actual, label='Realized discounted return', linewidth=1.5)
        ax.plot(times, predicted, label='Predicted value', linewidth=1.5)
        ax.set_title(f"{summary['episode_id']}: {summary['num_agents']} robots, {summary['noise']}, {summary['termination_reason']}")
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Discounted return'); ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Held-out Python trajectories: predicted value and realized return')
    fig.savefig(args.output_dir / 'value_alignment.png', dpi=160)
    plt.close(fig)
    result = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  checkpoint_epoch=checkpoint['epoch'], checkpoint_training_dataset=run_args.get('dataset'),
                  gamma=run_args['gamma'], return_source=run_args['return_source'],
                  image_hz=run_args['image_hz'], dataset=str(dataset),
                  note=('Each realized discounted return is one observed outcome, not the expected return. '
                        'Episode samples are few and per-frame errors within an episode are correlated.'),
                  by_noise={key: dict(anchors=len(values), mse=float(np.mean(np.square(values))),
                                      mae=float(np.mean(np.abs(values)))) for key, values in groups.items()},
                  episodes=results)
    (args.output_dir / 'alignment_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['by_noise'], indent=2))
    print(f'Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
