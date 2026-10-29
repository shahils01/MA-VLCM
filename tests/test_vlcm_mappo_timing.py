import unittest
import numpy as np
from vlcm_mappo_timing import context_indices


class TimingTests(unittest.TestCase):
    def test_all_control_phases_have_current_state_and_past_images(self):
        cameras=np.arange(0,45,4)
        for anchor in range(20,40):
            row=context_indices(anchor,0,cameras)
            self.assertEqual(row['state_indices'][-1],anchor)
            self.assertTrue(row['frame_mask'].all())
            self.assertTrue(np.all(cameras[row['camera_indices']] <= row['state_indices']))
            self.assertAlmostEqual(row['times'][0],-1.0)
            self.assertAlmostEqual(row['image_age'][-1],(anchor%4)*.05)

    def test_padding_never_crosses_an_episode_boundary(self):
        row=context_indices(15,10,np.array([0,4,8,10,14,18]))
        self.assertTrue(np.all(row['state_indices']>=10))
        self.assertTrue(np.all(row['camera_indices']>=3))
        self.assertEqual(row['frame_mask'].tolist(),[False,False,False,False,True,True])

    def test_camera_aligned_history_matches_original_one_second_clip(self):
        row=context_indices(20,0,np.arange(0,25,4))
        np.testing.assert_array_equal(row['state_indices'],np.arange(0,21,4))
        np.testing.assert_array_equal(row['camera_indices'],np.arange(6))
        np.testing.assert_array_equal(row['image_age'],np.zeros(6))

    def test_missing_reset_frame_and_stale_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'initial camera'):
            context_indices(15,10,np.array([0,4,8,14]))
        with self.assertRaisesRegex(ValueError,'image-age'):
            context_indices(20,0,np.array([0,4,8]))


if __name__=='__main__':
    unittest.main()
