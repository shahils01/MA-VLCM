import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ma_vlcm.tb3_training_contract import require_both_tb3_sources

class CombinedTrainingContractTests(unittest.TestCase):
    def test_missing_domain_rejected(self):
        for source in ('hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar',
                       'hf://datasets/adi2440/tb3-isaac-avoid-obstacles-vlcm/**/*.tar'):
            with self.assertRaisesRegex(ValueError,'ALWAYS'):
                require_both_tb3_sources(source)
    def test_both_hf_sources(self):
        self.assertEqual(require_both_tb3_sources('hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar;hf://datasets/adi2440/tb3-isaac-avoid-obstacles-vlcm/**/*.tar'),{'goal_to_goal','static_obstacles'})
    def test_local_empty_domain_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for domain in ('goal_to_goal','static_obstacles'):
                (root/domain).mkdir()
            (root/'goal_to_goal/example.tar').touch()
            with self.assertRaises(ValueError):require_both_tb3_sources(root)
            (root/'static_obstacles/example.tar').touch()
            self.assertEqual(require_both_tb3_sources(root),{'goal_to_goal','static_obstacles'})

if __name__=='__main__':unittest.main()
