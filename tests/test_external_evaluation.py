from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_utils import discover_episodes, feature_cache_path, prepare_episode, read_episode
from model import ModelConfig, MultimodalValueModel
from test_value_pipeline import write_episode
import evaluate


class ExternalEvaluationTests(unittest.TestCase):
    def test_external_metrics_exclude_timeouts_and_reject_saved_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, cache, output = root / 'external', root / 'cache', root / 'evaluation'
            data.mkdir()
            first, _ = write_episode(data, 100, control=True)
            _, timeout = write_episode(data, 101, control=True)
            summary = json.loads(timeout.read_text())
            summary['termination_reason'] = 'timeout'
            timeout.write_text(json.dumps(summary))
            cfg = ModelConfig(model_revision='unit-test', vl_dtype='float32', vl_hidden_dim=8,
                              d_model=16, clip_len=6, dropout=0)
            for path, sidecar in discover_episodes(data):
                ep = prepare_episode(read_episode(path, sidecar, cache / 'index'), source='control')
                features = feature_cache_path(ep, cfg, cache / 'features')
                features.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.randn(4,8).half(), features)
            model = MultimodalValueModel(cfg, load_backbone=False)
            checkpoint = dict(epoch=1, model_config=asdict(cfg), model_state=model.checkpoint_state(),
                              run=dict(args=dict(dataset=str(data), dataset_revision='main', gamma=.99,
                                                 return_source='control', image_hz=5, clip_stride=1),
                                       splits=dict(train=[], validation=[], test=[]),
                                       return_definition='Synthetic full control-step rewards'))
            path = root / 'best.pt'
            torch.save(checkpoint, path)
            argv = ['evaluate.py', '--checkpoint',str(path),'--dataset',str(data),
                    '--split','external','--cache-dir',str(cache),'--output-dir',str(output)]
            with patch.object(sys, 'argv', argv):
                evaluate.main()
            metrics = json.loads((output / 'external_metrics.json').read_text())
            self.assertEqual(metrics['episodes'],1)
            self.assertEqual(metrics['count'],3)
            self.assertEqual(metrics['excluded_timeout_or_unfinished_episodes'],1)
            self.assertFalse(metrics['include_terminal'])
            self.assertEqual(metrics['by_formation_outcome']['triangle_success']['episodes'],1)
            self.assertAlmostEqual(metrics['by_formation_outcome']['triangle_success']['mse'],metrics['mse'])
            with patch.object(sys, 'argv', argv + ['--expected-episodes','2']), self.assertRaisesRegex(ValueError,'Expected 2 complete'):
                evaluate.main()
            checkpoint['run']['splits']['train'] = [str(first)]
            torch.save(checkpoint,path)
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError,'overlap'):
                evaluate.main()


if __name__ == '__main__':
    unittest.main()
