import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from collect_fresh_validation import prepare


class ValidationPlanTests(unittest.TestCase):
    def source(self, root):
        source = root / 'source'
        actor = source / 'checkpoints/intermediate/actor.pt'
        actor.parent.mkdir(parents=True)
        actor.write_bytes(b'frozen actor test fixture')
        actor.with_name('training_metadata.json').write_text('{}')
        digest = hashlib.sha256(actor.read_bytes()).hexdigest()
        entries = [dict(stage='intermediate', num_agents=n, checkpoint=str(actor), seed=10001 + n,
                        episodes=125, label=f'intermediate_agents_{n:02}', sha256=digest,
                        environment_config=dict(num_agents=n, formation_spacing=.5)) for n in (3,4,5,6)]
        plan = dict(checkpoints=entries, policies=[dict(stage='intermediate', checkpoint=str(actor))],
                    capture_hz=10, total_episodes=500)
        (source / 'collection_plan.json').write_text(json.dumps(plan))
        (source / 'checkpoint_scores.json').write_text('[]')
        return source

    def test_sixty_new_ids_seeds_and_preserved_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            original = (source / 'collection_plan.json').read_bytes()
            plan = prepare(source, root / 'validation', 'intermediate', 15, 7300001)
            self.assertEqual(sum(e['episodes'] for e in plan['checkpoints']),60)
            self.assertEqual([e['num_agents'] for e in plan['checkpoints']],[3,4,5,6])
            self.assertEqual(len({e['seed'] for e in plan['checkpoints']}),4)
            ids = [i for e in plan['checkpoints'] for i in range(e['start_episode_id'], e['start_episode_id'] + e['episodes'])]
            self.assertEqual(len(set(ids)),60)
            self.assertGreater(min(ids),1000000)
            self.assertTrue(all(e['environment_config']['formation_spacing']==.5 for e in plan['checkpoints']))
            self.assertEqual((source / 'collection_plan.json').read_bytes(),original)
            self.assertEqual(prepare(source, root / 'validation', 'intermediate', 15, 7300001),plan)
            with self.assertRaisesRegex(ValueError,'another collection request'):
                prepare(source, root / 'validation', 'intermediate', 16, 7300001)

    def test_seed_reuse_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=self.source(root)
            with self.assertRaisesRegex(ValueError,'overlap'):
                prepare(source,root/'validation','intermediate',15,10004)


if __name__ == '__main__':
    unittest.main()
