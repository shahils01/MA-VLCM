"""Checks for archive timing, return labels and saved prediction alignment."""
from io import BytesIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from animate_checkpoint import animation_frames, load_frames, saved_values, split_membership, select_formation_cases, formation_name


class AnimationTests(unittest.TestCase):
    def write_archive(self, root, done=True, reason='success', controls=True):
        path = Path(root) / 'agents_03' / 'worker_00' / 'episode_000001.tar'
        path.parent.mkdir(parents=True)
        with tarfile.open(path, 'w') as archive:
            for i, time in enumerate([0, .1, .2, .3]):
                state = dict(capture_hz=10, control_hz=20,
                             episode_meta=dict(sim_time_s=time, done=done and i == 3))
                image = BytesIO()
                Image.new('RGB', (8, 8), (i * 40, 0, 0)).save(image, format='PNG')
                fields = {'state.json': json.dumps(state).encode(), 'overhead.png': image.getvalue(),
                          'reward.json': json.dumps(float(i)).encode(),
                          'progress.json': json.dumps(dict(team_progress=i / 3)).encode()}
                if controls:
                    fields['control_rewards.json'] = json.dumps([] if i == 0 else [1., 1.]).encode()
                for suffix, value in fields.items():
                    member = tarfile.TarInfo(f'frame{i:05d}.{suffix}')
                    member.size = len(value)
                    archive.addfile(member, BytesIO(value))
        path.with_suffix('.summary.json').write_text(json.dumps(dict(success=True, termination_reason=reason)))
        return path

    def test_control_returns_subsampling_and_short_terminal_interval(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_archive(root)
            frames, _ = load_frames(path, dict(image_hz=5, gamma=.5, return_source='control'))
            self.assertEqual([f['time_s'] for f in frames], [0, .2, .3])
            self.assertEqual([f['target_return'] for f in frames], [1.96875, 1.5, 0])
            self.assertTrue(frames[-1]['terminal'])
            self.assertEqual(int(frames[1]['image'][0, 0, 0]), 80)

    def test_timeout_has_no_ground_truth_return(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_archive(root, reason='timeout', controls=False)
            frames, _ = load_frames(path, dict(image_hz=5, gamma=.5, return_source='control'))
            self.assertTrue(all(f['target_return'] is None for f in frames))
            self.assertFalse(frames[-1]['terminal'])

    def test_terminal_reward_is_shown_without_inventing_a_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_archive(root)
            frames, _ = load_frames(path, dict(image_hz=5, gamma=.5, return_source='control'))
            selected = animation_frames(frames, {0: .4, 1: .6})
            self.assertEqual(len(selected), 3)
            self.assertTrue(selected[-1]['terminal'])
            self.assertEqual(selected[-1]['reward'], 3.0)
            self.assertEqual(selected[-1]['target_return'], 0.0)
            self.assertIsNone(selected[-1]['predicted_value'])
            clipped = animation_frames(frames, {0: .4, 1: .6, 2: .7}, include_terminal=False)
            self.assertEqual(len(clipped), 2)
            self.assertFalse(any(row['terminal'] for row in clipped))

    def test_saved_csv_checks_archive_target_and_split(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_archive(root)
            frames, _ = load_frames(path, dict(image_hz=5, gamma=.5, return_source='sampled'))
            csv_path = Path(root) / 'predictions.csv'
            header = 'episode,anchor,predicted_value,target_return\n'
            csv_path.write_text(header + f'{path},0,.4,{frames[0]["target_return"]}\n')
            self.assertEqual(saved_values(csv_path, path, frames), {0: .4})
            csv_path.write_text(header + f'{path},0,.4,999\n')
            with self.assertRaises(ValueError):
                saved_values(csv_path, path, frames)
            self.assertEqual(split_membership(path, {'splits': {'validation': [str(path)]}}), 'validation')
            self.assertEqual(split_membership(path, {'splits': {'train': [str(path)]}}), 'train')

    def test_control_checkpoint_rejects_missing_control_labels(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_archive(root, controls=False)
            with self.assertRaisesRegex(ValueError, 'full control_rewards'):
                load_frames(path, dict(image_hz=5, gamma=.5, return_source='control'))

    def test_formation_selection_reports_missing_outcomes(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for index, (formation, success) in enumerate([('triangle', True), ('triangle', False),
                                                        ('hexagon', False), ('triangle', True)]):
                path = Path(root) / f'episode_{index:06d}.tar'
                path.with_suffix('.summary.json').write_text(json.dumps(dict(
                    formation=dict(target_shape=formation), success=success)))
                paths.append(path)
            selected, coverage = select_formation_cases(paths, 1)
            self.assertEqual(set(selected), set(paths[:3]))
            missing = next(c for c in coverage if c['formation'] == 'hexagon' and c['outcome'] == 'success')
            self.assertEqual(missing['available'], 0)
            self.assertEqual(missing['selected'], 0)
            triangle_success = next(c for c in coverage if c['formation'] == 'triangle' and c['outcome'] == 'success')
            self.assertEqual(triangle_success['available'], 2)
            self.assertEqual(formation_name(dict(num_agents=4)), 'square')


if __name__ == '__main__':
    unittest.main()
