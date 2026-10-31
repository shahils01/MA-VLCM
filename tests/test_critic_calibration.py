import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibrate_critic_by_formation import calibrate, read_episodes


class CalibrationTests(unittest.TestCase):
    def test_episode_split_fit_and_exclude_timeouts_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for i in range(5):
                name = f'triangle_{i}.csv'
                entries.append(dict(formation='triangle', measurements=name, seed=i,
                                    return_complete=True, outcome='success'))
                with (root / name).open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'predicted_value', 'observed_discounted_return', 'return_complete'])
                    writer.writerows([[0, i + 1, 2 * (i + 1) + 3, 'True'],
                                      [1, i + 2, 2 * (i + 2) + 3, 'True'],
                                      [2, 0, 0, 'True']])
            entries.append(dict(formation='square', measurements='square_timeout.csv', seed=1,
                                return_complete=False, outcome='timeout'))
            (root / 'summary.json').write_text(json.dumps(dict(episodes=entries, gamma=.99)))
            summary, groups, counts = read_episodes(root)
            report, rows = calibrate(summary, groups, counts, 43, .4)
            triangle = report['formations']['triangle']
            self.assertEqual(triangle['status'], 'exploratory_fit')
            self.assertAlmostEqual(triangle['slope'], 2.)
            self.assertAlmostEqual(triangle['intercept'], 3.)
            self.assertAlmostEqual(triangle['after']['mse'], 0.)
            self.assertGreater(triangle['before']['mse'], 0.)
            self.assertFalse(set(triangle['fit_episodes']) & set(triangle['test_episodes']))
            self.assertTrue(all(row['step'] != 2 for row in rows))
            self.assertEqual(report['formations']['square']['counts']['timeouts'], 1)
            self.assertEqual(report['formations']['square']['status'], 'insufficient_complete_episodes')
            self.assertEqual(len(rows), 4)

    def test_constant_predictions_cannot_fit(self):
        from calibrate_critic_by_formation import fit_affine
        ep = dict(predictions=np.ones(2), returns=np.zeros(2))
        with self.assertRaises(ValueError):
            fit_affine([ep, ep])


if __name__ == '__main__':
    unittest.main()
