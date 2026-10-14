"""Bounded epochs must release their workers before another phase starts."""
import gc
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import IterableDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ma_vlcm.train import parse_args, webdataset_loader

class TinyDataset(IterableDataset):
    def __init__(self,**kwargs): pass
    def __iter__(self):
        for step in range(4):
            yield {'episode_id':str(step),'inputs':{'pixel_values_videos':torch.zeros(2,3,8,8)},
                   'robot_obs':torch.zeros(3,9),'adj':torch.eye(3),
                   'reward':torch.tensor(0.),'done':torch.tensor(0.)}

class WorkerLifecycleTests(unittest.TestCase):
    def test_train_break_and_validation_release_workers_each_epoch(self):
        with patch.object(sys,'argv',['train','--vl_backend','vjepa2',
                     '--loader_prefetch_factor','1','--disable_persistent_workers']):
            args=parse_args()
        with patch('ma_vlcm.train.SequenceWebDataset',TinyDataset):
            train=webdataset_loader(args,[],2,2,dataset_type='tb3_potential_v1')
            val=webdataset_loader(args,[],2,2,dataset_type='tb3_potential_v1')
        for epoch in range(2):
            iterator=iter(train)
            workers=list(iterator._workers)
            next(iterator)  # Training consumes a bounded subset, like run_epoch.
            del iterator
            gc.collect()
            for worker in workers: worker.join(timeout=3)
            self.assertFalse(any(worker.is_alive() for worker in workers))
            validation=iter(val)
            workers=list(validation._workers)
            self.assertEqual(len(list(validation)),4)
            for worker in workers:worker.join(timeout=3)
            self.assertFalse(any(worker.is_alive() for worker in workers))

if __name__=='__main__':unittest.main()
