import copy
import unittest

from prepare_frozen_2000 import validate_collections


class FrozenPreflightTests(unittest.TestCase):
    def collections(self):
        def row(index, n):
            return dict(num_agents=n, scenario_id=f"scenario-{index}", seed=index,
                        path=f"/dataset/agents_{n}/worker_00/episode_{index}.tar",
                        control_rewards=[[], [.1]], complete=True)
        return [row(i, 3+i%4) for i in range(8)], [row(100+i, 3+i) for i in range(4)]

    def test_independent_balanced_collection_and_actual_size_are_required(self):
        train, external = self.collections()
        validate_collections(train, external, expected_training=8, external_per_team=1)
        with self.assertRaisesRegex(ValueError, "Expected 2000"):
            validate_collections(train, external)
        external[0]["num_agents"] = 4
        with self.assertRaisesRegex(ValueError, "per team size"):
            validate_collections(train, external, expected_training=8, external_per_team=1)

    def test_leakage_and_incomplete_or_sampled_return_labels_are_rejected(self):
        train, original = self.collections()
        changes = [("scenario_id", train[0]["scenario_id"], "scenarios overlap"),
                   ("seed", train[0]["seed"], "seeds overlap"),
                   ("path", train[0]["path"], "path keys overlap"),
                   ("complete", False, "true termination"),
                   ("control_rewards", None, "control reward stream")]
        for key, value, message in changes:
            with self.subTest(key=key):
                external = copy.deepcopy(original)
                external[0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_collections(train, external, expected_training=8, external_per_team=1)


if __name__ == "__main__":
    unittest.main()
