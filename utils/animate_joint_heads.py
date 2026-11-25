"""Render matched held-out predictions from two frozen value-head runs."""
import argparse
import csv
import json
import os
import tarfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.ffmpeg_path'] = os.environ.get('VLCM_FFMPEG', 'ffmpeg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image


def predictions(path):
    groups = defaultdict(dict)
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups[(row['task_domain'], int(row['num_agents']), row['episode'])][int(row['anchor'])] = row
    return groups


def frames(path, image_hz):
    with tarfile.open(path, 'r:') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        names = sorted(n for n in members if n.endswith('.state.json'))
        def get(name):
            return archive.extractfile(members[name]).read()
        first = json.loads(get(names[0]))
        ratio = float(first['capture_hz']) / image_hz
        stride = round(ratio)
        if stride < 1 or abs(ratio - stride) > 1e-5:
            raise ValueError(f'Invalid capture frequency in {path}')
        indices = list(range(0, len(names), stride))
        if indices[-1] != len(names) - 1:
            indices.append(len(names) - 1)
        result = []
        for index in indices:
            stem = names[index][:-len('state.json')]
            state = json.loads(get(names[index]))
            progress_name = stem + 'progress.json'
            progress = json.loads(get(progress_name)).get('team_progress') if progress_name in members else None
            image = np.asarray(Image.open(BytesIO(get(stem + 'overhead.png'))).convert('RGB'))
            result.append((float(state['episode_meta']['sim_time_s']), float(json.loads(get(stem + 'reward.json'))), progress, image))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=2)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    shared = predictions(args.root / 'agents3456_shared/after_test_predictions.csv')
    task = predictions(args.root / 'agents3456_task/after_test_predictions.csv')
    config = json.loads((args.root / 'agents3456_shared/run.json').read_text())
    image_hz = float(config['args']['image_hz'])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for domain in ('formation_control', 'goal_to_goal'):
        for count in (3, 4, 5, 6):
            keys = sorted(k for k in shared.keys() & task.keys() if k[0] == domain and k[1] == count)
            if len(keys) < args.per_group:
                raise ValueError(f'Insufficient matched episodes: {domain} {count}')
            # Prefer a successful and unsuccessful example if both exist.
            keys.sort(key=lambda k: (next(iter(shared[k].values()))['success'] != 'True', k[2]))
            successes = [k for k in keys if next(iter(shared[k].values()))['success'] == 'True']
            failures = [k for k in keys if next(iter(shared[k].values()))['success'] != 'True']
            chosen = ([successes[0], failures[0]] if args.per_group == 2 and successes and failures else keys[:args.per_group])
            for idx, key in enumerate(chosen, 1):
                _, _, episode = key
                a, b = shared[key], task[key]
                if a.keys() != b.keys():
                    raise ValueError(f'Prediction anchor mismatch: {episode}')
                data = frames(Path(episode), image_hz)
                anchors = sorted(a)
                if any(i >= len(data) or not np.isclose(float(a[i]['target_return']), float(b[i]['target_return']), rtol=1e-6, atol=1e-6) for i in anchors):
                    raise ValueError(f'Prediction target or frame mismatch: {episode}')
                times = np.array([data[i][0] for i in anchors])
                values = [[float(a[i]['predicted_value']) for i in anchors], [float(b[i]['predicted_value']) for i in anchors], [float(a[i]['target_return']) for i in anchors], [data[i][1] for i in anchors], [np.nan if data[i][2] is None else float(data[i][2]) for i in anchors]]
                fig = plt.figure(figsize=(13, 7), constrained_layout=True)
                grid = fig.add_gridspec(3, 2, width_ratios=(1, 1.25))
                ax_image = fig.add_subplot(grid[:, 0]); ax_image.axis('off')
                image = ax_image.imshow(data[anchors[0]][3])
                axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
                plots = []
                for y, label, color, panel in zip(values, ('Shared head', 'Task-dependent head', 'Observed future return', 'Recorded reward', 'Team progress'), ('tab:blue', 'tab:orange', 'black', 'tab:red', 'tab:green'), (0, 0, 0, 1, 2)):
                    line, = axes[panel].plot([], [], label=label, color=color)
                    plots.append((line, y))
                for panel, ax in enumerate(axes):
                    vals = np.array([v for y, p in zip(values, (0, 0, 0, 1, 2)) if p == panel for v in y]); vals = vals[np.isfinite(vals)]
                    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0., 1.)
                    margin = max((hi-lo)*.1, .02); ax.set_ylim(lo-margin, hi+margin); ax.set_xlim(times[0], max(times[-1], times[0]+.1)); ax.grid(alpha=.2); ax.legend(fontsize=8)
                axes[-1].set_xlabel('Episode time (s)')
                outcome = 'success' if next(iter(a.values()))['success'] == 'True' else 'failure'
                fig.suptitle(f'{domain} | {count} agents | {outcome} | held-out test episode {Path(episode).stem}')
                out = args.output / f'{domain}_{count}agents_{idx:02d}_{outcome}.mp4'
                writer = FFMpegWriter(fps=args.fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
                playback = np.arange(times[0], times[-1] + .5/args.fps, 1/args.fps)
                with writer.saving(fig, str(out), dpi=90):
                    for t in playback:
                        j = min(max(np.searchsorted(times, t, side='right') - 1, 0), len(anchors)-1)
                        image.set_data(data[anchors[j]][3]); ax_image.set_title(f't = {times[j]:.2f} s')
                        for line, y in plots: line.set_data(times[:j+1], y[:j+1])
                        writer.grab_frame()
                plt.close(fig)
                manifest.append(dict(episode=episode, task=domain, agents=count, outcome=outcome, video=out.name, anchors=len(anchors)))
                (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                print(f'Created {out}', flush=True)


if __name__ == '__main__':
    main()
