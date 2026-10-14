import unittest
from ma_vlcm.dataset_metadata import canonical_task_domain

class HistoricalMetadataTests(unittest.TestCase):
    def test_legacy_domains_resolved_from_verified_paths(self):
        for domain in ('goal_to_goal','static_obstacles'):
            state={'agents':[{}, {}, {}]}
            self.assertEqual(canonical_task_domain(state,f'/verified/{domain}/agents_03/episode.tar'),domain)
            self.assertNotIn('task_domain',state)

    def test_conflicts_and_unknown_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'conflicts'):
            canonical_task_domain({'task_domain':'goal_to_goal','agents':[{}, {}, {}]},'/verified/static_obstacles/agents_03/episode.tar')
        with self.assertRaisesRegex(ValueError,'cannot identify'):
            canonical_task_domain({'agents':[{}, {}, {}]},'/unknown/episode.tar')
        with self.assertRaisesRegex(ValueError,'agent count'):
            canonical_task_domain({'agents':[{}]},'/verified/static_obstacles/agents_03/episode.tar')

if __name__=='__main__':unittest.main()
