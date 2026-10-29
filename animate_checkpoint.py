"""Animate causal checkpoint value estimates on formation episode archives.

Fresh inference is the default. --predictions replays an existing prediction CSV
without PyTorch; its title and manifest explicitly identify saved predictions.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def episode_key(path):
    return tuple(Path(path).parts[-3:])


def split_membership(path, run):
    matches = [name for name, paths in run['splits'].items()
               if episode_key(path) in {episode_key(p) for p in paths}]
    return ', '.join(matches) if matches else 'external (not in saved splits)'


def read_summary(path):
    sidecar = Path(path).with_suffix('.summary.json')
    if not sidecar.exists():
        sidecar = Path(path).with_suffix('.json')
    return json.loads(sidecar.read_text())


def formation_name(summary):
    name = summary.get('formation', {}).get('target_shape')
    if not name:
        name = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon'}.get(summary.get('num_agents'), 'unknown')
    return re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_') or 'unknown'


def select_formation_cases(candidates, per_case):
    groups = defaultdict(list)
    for path in sorted(candidates):
        summary = read_summary(path)
        groups[(formation_name(summary), bool(summary['success']))].append(path)
    paths, coverage = [], []
    for formation in sorted({key[0] for key in groups}):
        for success in (True, False):
            eligible = groups[(formation, success)]
            selected = eligible[:per_case]
            paths.extend(selected)
            coverage.append(dict(formation=formation, outcome='success' if success else 'failure',
                                 available=len(eligible), selected=len(selected),
                                 episodes=[str(p) for p in selected]))
    return paths, coverage


def load_frames(path, run_args):
    """Read selected observations/labels without extracting files or importing torch."""
    frames = []
    with tarfile.open(path, 'r:') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        names = sorted(n for n in members if n.endswith('.state.json'))
        if not names:
            raise ValueError(f'No state frames in {path}')
        def read(name):
            return json.load(archive.extractfile(members[name]))
        states = [read(n) for n in names]
        capture_hz = states[0]['capture_hz']
        ratio = capture_hz / run_args['image_hz']
        stride = round(ratio)
        if stride < 1 or abs(ratio - stride) > 1e-5:
            raise ValueError('Checkpoint image_hz must divide archive capture_hz')
        indices = list(range(0, len(names), stride))
        if indices[-1] != len(names) - 1:
            indices.append(len(names) - 1)
        summary_path = Path(path).with_suffix('.summary.json')
        if not summary_path.exists():
            summary_path = Path(path).with_suffix('.json')
        summary = json.loads(summary_path.read_text())
        complete = bool(states[-1]['episode_meta']['done']) and summary['termination_reason'] != 'timeout'
        controls = None
        if complete and run_args['return_source'] == 'control':
            try:
                controls = [read(n[:-len('state.json')] + 'control_rewards.json') for n in names]
            except KeyError as exc:
                raise ValueError('This checkpoint needs full control_rewards.json labels; do not substitute sampled rewards') from exc
        for anchor, index in enumerate(indices):
            stem = names[index][:-len('state.json')]
            state = states[index]
            with Image.open(BytesIO(archive.extractfile(members[stem + 'overhead.png']).read())) as image:
                pixels = np.asarray(image.convert('RGB')).copy()
            progress = read(stem + 'progress.json') if stem + 'progress.json' in members else {}
            frames.append(dict(anchor=anchor, time_s=float(state['episode_meta']['sim_time_s']),
                               reward=float(read(stem + 'reward.json')), progress=progress.get('team_progress'),
                               target_return=None, terminal=complete and index == len(names) - 1,
                               image=pixels))
        if complete:
            gamma = run_args['gamma']
            target = 0.0
            frames[-1]['target_return'] = target
            for i in range(len(frames) - 2, -1, -1):
                steps = round((frames[i + 1]['time_s'] - frames[i]['time_s']) * states[0]['control_hz'])
                if steps < 1:
                    raise ValueError('Invalid frame timestamps')
                if controls is None:
                    reward = gamma ** (steps - 1) * frames[i + 1]['reward']
                else:
                    interval = [r for chunk in controls[indices[i] + 1:indices[i + 1] + 1] for r in chunk]
                    if len(interval) != steps:
                        raise ValueError('Control reward count does not match timestamps')
                    reward = sum(gamma ** k * r for k, r in enumerate(interval))
                target = reward + gamma ** steps * target
                frames[i]['target_return'] = target
    return frames, summary


def infer(checkpoint, episode_path, cache_dir, device_name, batch_size):
    """Use exactly the checkpoint's causal clip construction and normalization."""
    import torch
    from torch.utils.data import DataLoader
    from data_utils import (FormationClipDataset, collate_clips, discover_episodes,
                            feature_cache_path, frame_indices, prepare_episode, read_episode)
    from model import ModelConfig, MultimodalValueModel
    cfg = ModelConfig(**checkpoint['model_config'])
    run_args = checkpoint['run']['args']
    path, summary = discover_episodes(str(episode_path))[0]
    episode = read_episode(path, summary, Path(cache_dir) / 'index')
    # Incomplete rollouts may be visualized, but their truncated returns aren't ground truth.
    if episode['complete']:
        episode = prepare_episode(episode, run_args['gamma'], run_args['return_source'], run_args['image_hz'])
    else:
        indices = frame_indices(episode, run_args['image_hz'])
        episode = dict(episode, selected_indices=indices, returns=torch.zeros(len(indices)))
    feature_dir = Path(cache_dir) / 'features'
    cached = cfg.backbone_mode == 'frozen' and feature_cache_path(episode, cfg, feature_dir).exists()
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = MultimodalValueModel(cfg, device, load_backbone=not cached).to(device)
    # Backbone weights are intentionally omitted from frozen/LoRA checkpoints.
    incompatible = model.load_state_dict(checkpoint['model_state'], strict=cached)
    if any(not k.startswith('backbone.') for k in incompatible.missing_keys) or incompatible.unexpected_keys:
        raise ValueError(f'Checkpoint head mismatch: {incompatible}')
    model.eval()
    dataset = FormationClipDataset([episode], cfg.clip_len, 1, feature_dir if cached else None, cfg)
    result = {}
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_clips):
            metadata = batch.pop('metadata')
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
            precision = torch.autocast('cuda', dtype=getattr(torch, cfg.vl_dtype)) if device.type == 'cuda' and cfg.vl_dtype != 'float32' else nullcontext()
            with precision:
                values = model(batch).float().cpu().tolist()
            result.update({int(info['anchor']): value for info, value in zip(metadata, values)})
    return result


def saved_values(path, episode_path, frames):
    with Path(path).open() as stream:
        rows = [r for r in csv.DictReader(stream) if episode_key(r['episode']) == episode_key(episode_path)]
    if not rows:
        raise ValueError(f'No predictions for {episode_path} in {path}')
    result = {}
    for row in rows:
        anchor = int(row['anchor'])
        if anchor in result or not 0 <= anchor < len(frames):
            raise ValueError('Duplicate or invalid prediction anchor')
        target = frames[anchor]['target_return']
        if target is not None and not np.isclose(float(row['target_return']), target, atol=1e-5, rtol=1e-5):
            raise ValueError('Saved CSV targets do not match this archive/checkpoint return definition')
        result[anchor] = float(row['predicted_value'])
    return result


def animation_frames(frames, values, include_terminal=True):
    """Show the observed terminal frame even when evaluation omitted its prediction."""
    return [dict(f, predicted_value=values.get(f['anchor'])) for f in frames
            if (include_terminal or not f['terminal']) and (f['anchor'] in values or f['terminal'])]


def render(frames, values, path, title, fps, speed, fmt, include_terminal=True):
    selected = animation_frames(frames, values, include_terminal=include_terminal)
    times = np.array([f['time_s'] for f in selected])
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.3])
    overhead = fig.add_subplot(grid[:, 0])
    image = overhead.imshow(selected[0]['image'])
    overhead.axis('off')
    text = overhead.set_title('', fontsize=10)
    axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    series = [([np.nan if f['predicted_value'] is None else f['predicted_value'] for f in selected], 'Predicted value', 'tab:blue', 0),
              ([np.nan if f['target_return'] is None else f['target_return'] for f in selected], 'Recorded future return', 'tab:orange', 0),
              ([f['reward'] for f in selected], 'Recorded frame reward', 'tab:red', 1),
              ([np.nan if f['progress'] is None else f['progress'] for f in selected], 'Recorded team progress', 'tab:green', 2)]
    lines = []
    for data, label, color, panel in series:
        line, = axes[panel].plot([], [], color=color, label=label, marker='.', markersize=3)
        lines.append((line, data))
    for i, ax in enumerate(axes):
        data = np.array([v for values_, _, _, panel in series if panel == i for v in values_])
        data = data[np.isfinite(data)]
        low, high = (float(data.min()), float(data.max())) if len(data) else (0, 1)
        margin = max((high - low) * .12, .02)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlim(times[0], max(times[-1], times[0] + .2))
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', fontsize=8)
    axes[-1].set_xlabel('Episode time (s)')
    cursors = [ax.axvline(times[0], color='grey', linewidth=.8) for ax in axes]
    fig.suptitle(title, fontsize=11)
    # Uniform playback clock, so shorter terminal intervals preserve their duration.
    playback = np.arange(times[0], times[-1], speed / fps)
    playback = np.append(playback, times[-1])
    frame_indices = np.searchsorted(times, playback, side='right') - 1
    if selected[-1]['terminal']:
        # A terminal event can be only one control step long; hold it for visibility.
        frame_indices = np.concatenate((frame_indices, np.full(fps, len(selected) - 1)))
    def update(frame_index):
        f = selected[frame_index]
        image.set_data(f['image'])
        value = 'prediction unavailable' if f['predicted_value'] is None else f"value={f['predicted_value']:.3f}"
        terminal = ' | terminal' if f['terminal'] else ''
        if f['terminal'] and f['predicted_value'] is not None:
            terminal += ': raw network output'
        text.set_text(f"t={f['time_s']:.2f}s | {value}{terminal}")
        for line, data in lines:
            line.set_data(times[:frame_index + 1], data[:frame_index + 1])
        for cursor in cursors:
            cursor.set_xdata([f['time_s'], f['time_s']])
        return [image, text] + [line for line, _ in lines] + cursors
    animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']) if fmt == 'mp4' else PillowWriter(fps=fps)
    animation.save(str(path), writer=writer, dpi=100)
    plt.close(fig)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/zero_margin_900/best.pt'))
    parser.add_argument('--episode', type=Path, action='append', help='Repeat for multiple custom episode archives')
    parser.add_argument('--samples', type=int, default=2, help='Number of saved validation episodes when --episode is omitted')
    parser.add_argument('--by-formation', action='store_true', help='Select validation success/failure cases for every formation and use descriptive filenames')
    parser.add_argument('--per-case', type=int, default=1, help='Episodes per formation/outcome with --by-formation')
    parser.add_argument('--predictions', type=Path, help='Replay saved CSV predictions instead of fresh checkpoint inference')
    parser.add_argument('--run-dir', type=Path, help='run.json location for saved-CSV mode; default: CSV parent')
    parser.add_argument('--dataset', type=Path, help='Relocated dataset root, preserving each episode last three path components')
    parser.add_argument('--external-dataset', type=Path, help='Select episodes from this external dataset instead of the saved validation split')
    parser.add_argument('--cache-dir', default='outputs/cache')
    parser.add_argument('--device', default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/checkpoint_animations'))
    parser.add_argument('--format', choices=['mp4', 'gif'], default='mp4')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--speed', type=float, default=1, help='Playback speed, 1 = real time')
    parser.add_argument('--exclude-terminal', action='store_true', help='Stop before the terminal frame and its reward')
    args = parser.parse_args()
    if min(args.samples, args.per_case, args.batch_size, args.fps, args.speed) <= 0:
        parser.error('samples, batch-size, fps and speed must be positive')
    if args.by_formation and args.episode:
        parser.error('--by-formation selects saved validation episodes; use --episode separately for custom archives')
    if args.external_dataset and args.episode:
        parser.error('--external-dataset selects archives directly; do not combine it with --episode')
    if args.external_dataset and args.dataset:
        parser.error('--dataset relocates a saved split; use --external-dataset alone for a new dataset')
    checkpoint = None
    if args.predictions:
        run = json.loads(((args.run_dir or args.predictions.parent) / 'run.json').read_text())
        source = f'SAVED PREDICTIONS: {args.predictions}'
    else:
        try:
            import torch
        except ImportError:
            parser.error('Fresh inference requires PyTorch. Activate the training environment, or use --predictions for a labeled replay.')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        run = checkpoint['run']
        source = f'FRESH INFERENCE: {args.checkpoint} (epoch {checkpoint["epoch"]})'
    relocated = None
    if args.dataset:
        relocated = defaultdict(list)
        for path in args.dataset.rglob('episode_*.tar'):
            relocated[episode_key(path)].append(path)
    def locate(path):
        if relocated is None:
            return Path(path)
        matches = relocated.get(episode_key(path), [])
        if len(matches) != 1:
            raise ValueError(f'Expected one relocated archive for {path}, found {len(matches)}')
        return matches[0]
    paths = args.episode
    coverage = []
    if not paths:
        candidates = (sorted(args.external_dataset.rglob('episode_*.tar')) if args.external_dataset
                      else [locate(p) for p in sorted(run['splits']['validation'])])
        if args.predictions:
            with args.predictions.open() as stream:
                keys = {episode_key(r['episode']) for r in csv.DictReader(stream)}
            candidates = [p for p in candidates if episode_key(p) in keys]
        if args.by_formation:
            paths, coverage = select_formation_cases(candidates, args.per_case)
        else:
            # Include different outcomes when available, with deterministic selection.
            by_outcome = defaultdict(list)
            for path in candidates:
                by_outcome[bool(read_summary(path)['success'])].append(path)
            paths = []
            while len(paths) < args.samples and any(by_outcome.values()):
                for outcome in sorted(by_outcome):
                    if by_outcome[outcome] and len(paths) < args.samples:
                        paths.append(by_outcome[outcome].pop(0))
    if not paths:
        parser.error('No matching validation episodes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if coverage:
        (args.output_dir / 'coverage.json').write_text(json.dumps(coverage, indent=2))
        for case in coverage:
            print(f"{case['formation']} {case['outcome']}: {case['available']} available, {case['selected']} selected", flush=True)
    manifest = []
    case_counts = defaultdict(int)
    for index, path in enumerate(paths):
        frames, summary = load_frames(path, run['args'])
        membership = split_membership(path, run)
        print(f'{path}: {membership}; {source}', flush=True)
        values = saved_values(args.predictions, path, frames) if args.predictions else infer(checkpoint, path, args.cache_dir, args.device, args.batch_size)
        formation = formation_name(summary)
        outcome = 'success' if summary['success'] else 'failure'
        stem = f'{index:02d}_{path.parent.parent.name}_{path.parent.name}_{path.stem}'
        if args.by_formation:
            case_counts[(formation, outcome)] += 1
            stem = f'{formation}_{outcome}'
            if args.per_case > 1:
                stem += f'_{case_counts[(formation, outcome)]:02d}'
        output = args.output_dir / f'{stem}.{args.format}'
        title = f'{formation.title()} | {outcome} ({summary["termination_reason"]}) | {path.stem} | {membership}\n{source}\nValue estimates discounted future reward; reward/progress panels are recorded labels'
        selected = render(frames, values, output, title, args.fps, args.speed, args.format,
                          include_terminal=not args.exclude_terminal)
        scored = [r for r in selected if r['target_return'] is not None and not r['terminal']]
        mse = float(np.mean([(r['predicted_value'] - r['target_return']) ** 2 for r in scored])) if scored else None
        record = dict(episode=str(path.resolve()), membership=membership, source=source,
                      formation=formation, outcome=outcome, termination_reason=summary['termination_reason'],
                      checkpoint=str(args.checkpoint) if checkpoint else None,
                      return_source=run['args']['return_source'], gamma=run['args']['gamma'],
                      animation=str(output), nonterminal_mse=mse, scored_frames=len(scored),
                      rendered_frames=len(selected), terminal_output='excluded' if args.exclude_terminal else 'raw network when present; otherwise unavailable')
        with (args.output_dir / f'{stem}.csv').open('w', newline='') as stream:
            fields = ['anchor', 'time_s', 'predicted_value', 'target_return', 'reward', 'progress', 'terminal']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(selected)
        manifest.append(record)
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'Saved {output}; nonterminal MSE={mse}', flush=True)
    if coverage:
        lines = ['# Validation animations by formation', '', source, '',
                 'Selection: first archive in sorted path order per formation/outcome; no selection by prediction accuracy.', '',
                 '| Formation | Outcome | Available episodes | Animation | Nonterminal MSE |',
                 '| --- | --- | ---: | --- | ---: |']
        for case in coverage:
            records = [r for r in manifest if r['formation'] == case['formation'] and r['outcome'] == case['outcome']]
            if not records:
                lines.append(f"| {case['formation']} | {case['outcome']} | {case['available']} | Unavailable in selected dataset | — |")
            for record in records:
                name = Path(record['animation']).name
                score = f"{record['nonterminal_mse']:.5f}" if record['nonterminal_mse'] is not None else 'unavailable'
                lines.append(f"| {case['formation']} | {case['outcome']} | {case['available']} | [{name}]({name}) | {score} |")
        lines.extend(['', 'CSV files share the animation names. Reward/progress panels are recorded labels; the critic predicts discounted future return.',
                      'Missing cases are not replaced by training or test episodes.', ''])
        (args.output_dir / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
