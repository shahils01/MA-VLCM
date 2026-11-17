"""Evaluation contracts independent of backbone inference."""
import unittest
from run_rollout_evaluation import navigation_start_audit, distribution, DEFAULT_OUTPUT, formation_setup, formation_episode


class EvaluationContracts(unittest.TestCase):
    def test_navigation_contract_rejects_outside_and_too_close(self):
        result = navigation_start_audit([[-4., -4.], [4., 4.], [0., 0.]])
        self.assertTrue(result['start_centers_inside_8m'])
        for invalid in ([[-4.01, 0.], [0., 0.]], [[0., 0.], [1.49, 0.]]):
            with self.assertRaisesRegex(ValueError, 'start contract'):
                navigation_start_audit(invalid)

    def test_distribution_labels_do_not_call_obstacle_transfer_iid(self):
        self.assertEqual(distribution('joint_without_obstacles', 'navigation_with_obstacles', 3), 'OOD_obstacles_added')
        self.assertTrue(distribution('joint_with_obstacles', 'formation_clean', 4).startswith('OOD_agent_count'))
        self.assertTrue(distribution('joint_with_obstacles', 'formation_noise_recovery', 3).startswith('OOD_'))

    @unittest.skipUnless((DEFAULT_OUTPUT/'assets/formation/source/collect_python_formation_dataset.py').exists(), 'requires downloaded frozen actor/source')
    def test_clean_rollout_matches_original_native_collector(self):
        frozen, collector, actor, policy = formation_setup(DEFAULT_OUTPUT)
        job = dict(episode_id=999999, seed=118000300, noise_seed=128000300,
                   num_agents=3, stage='successful', noise='clean', worker_id=0)
        original = collector.rollout(DEFAULT_OUTPUT, job, collector.environment(frozen, 3), actor, policy, render=False)
        evaluated = formation_episode(frozen, collector, actor, policy, DEFAULT_OUTPUT, job, 'formation_clean', render=False)
        self.assertEqual(original['termination_reason'], evaluated['termination_reason'])
        self.assertEqual(original['control_steps'], evaluated['control_steps'])
        self.assertAlmostEqual(original['episode_return'], evaluated['episode_return'], places=5)


if __name__ == '__main__':
    unittest.main()
