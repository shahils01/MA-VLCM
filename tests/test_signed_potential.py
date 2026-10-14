"""CPU-only geometry and shard invariants, with no model downloads."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from ma_vlcm.potential import PotentialConfig, compute_potential, transition_reward
from relabel_tb3_potential_dataset import relabel_shard


def state(distance, second_x=None, obstacles=None):
    result = {"agents": [{"id": 0, "pos": [0., 0.], "dist_to_goal": distance}],
              "task_domain": "goal_to_goal"}
    if second_x is not None:
        result["agents"].append({"id": 1, "pos": [second_x, 0.], "dist_to_goal": distance})
    if obstacles is not None:
        result.update(task_domain="static_obstacles", obstacles=obstacles)
    return result


class SignedPotentialTests(unittest.TestCase):
    def test_forward_retreat_and_safe_goal(self):
        config = PotentialConfig(goal_radius_m=0.)
        values = [compute_potential(state(d), {0: 1.}, config)["target"] for d in (1., .5, 0., 2.)]
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[2], 1.)
        self.assertLess(values[3], 0.)
        self.assertGreater(transition_reward(values[0],values[1]), 0.)
        self.assertLess(transition_reward(values[1],values[0]), 0.)
        self.assertEqual(transition_reward(values[0],values[0]),0.)

    def test_pair_risk_is_not_diluted_by_team_size(self):
        initial = {0: 1., 1: 1.}
        safe = compute_potential(state(.5, 1.), initial)
        near = compute_potential(state(.5, .3), initial)
        collision = compute_potential(state(.5, .2), initial)
        self.assertLess(near["target"],safe["target"])
        self.assertEqual(collision["target"],-1.)
        self.assertLess(transition_reward(safe["target"],near["target"]),0.)

    def test_box_surface_clearance_and_missing_layout(self):
        safe = state(.5, obstacles=[{"pos": [1.,0.],"size_m":[.3,.3]}])
        near = state(.5, obstacles=[{"pos": [.35,0.],"size_m":[.3,.3]}])
        a = compute_potential(safe,{0:1.})
        b = compute_potential(near,{0:1.})
        self.assertLess(b["target"],a["target"])
        missing = state(.5)
        missing["task_domain"] = "static_obstacles"
        with self.assertRaisesRegex(ValueError,"lacks geometry"):
            compute_potential(missing,{0:1.})
        with self.assertRaisesRegex(ValueError,"nonempty"):
            compute_potential(missing,{0:1.},obstacles=[])

    def test_route_metric_and_invalid_inputs(self):
        config = PotentialConfig(distance_mode="route_required", goal_radius_m=0.)
        s = state(.5)
        with self.assertRaisesRegex(ValueError,"route_required"):
            compute_potential(s,{0:2.},config)
        s["agents"][0]["route_dist_to_goal"] = 3.
        self.assertLess(compute_potential(s,{0:2.},config)["target"],0.)
        s["agents"][0]["pos"][0] = float("nan")
        with self.assertRaisesRegex(ValueError,"finite"):
            compute_potential(s,{0:2.},config)

    def test_labels_do_not_depend_on_future_success_or_time(self):
        a = state(.5)
        b = state(.5)
        a["episode_meta"] = {"step": 1, "success": False}
        b["episode_meta"] = {"step": 100, "success": True}
        self.assertEqual(compute_potential(a,{0:1.}),compute_potential(b,{0:1.}))

    def test_relabel_preserves_source_and_rewards_and_keeps_rewind_in_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp)/"source.tar", Path(tmp)/"new.tar"
            with tarfile.open(source,"w") as tar:
                for step, distance in enumerate((1., .5, .1)):
                    s = state(distance)
                    s["agents"][0]["vel"] = [0.2,0.3]
                    s["agents"][0]["linear_velocity_mps"] = .2
                    s["episode_meta"] = {"step": step, "done": step==2, "success":step==2}
                    for suffix,payload in (("state.json",json.dumps(s).encode()),("reward.json",b"123.0")):
                        info=tarfile.TarInfo(f"ep_step{step:05d}.{suffix}")
                        info.size=len(payload)
                        tar.addfile(info,io.BytesIO(payload))
            original=source.read_bytes()
            report=relabel_shard(source,destination,PotentialConfig(),include_rewind=True)
            self.assertEqual(report["rewind_negative_deltas"],2)
            self.assertEqual(source.read_bytes(),original)
            with tarfile.open(destination) as tar:
                self.assertEqual(tar.extractfile("ep_step00000.reward.json").read(),b"123.0")
                rewind=json.load(tar.extractfile("ep_rewind_step00000.state.json"))
                self.assertEqual(rewind["agents"][0]["vel"],[-.2,-.3])
                self.assertEqual(rewind["agents"][0]["linear_velocity_mps"],-.2)
                self.assertFalse(rewind["episode_meta"]["success"])
            with self.assertRaisesRegex(ValueError,"new shard"):
                relabel_shard(source,destination,PotentialConfig())

    def test_transition_bounds(self):
        for gamma in (0., .5, .995, 1.):
            for a in (-1.,0.,1.):
                for b in (-1.,0.,1.):
                    self.assertLessEqual(abs(transition_reward(a,b,gamma)),1.)


if __name__ == "__main__":
    unittest.main()
