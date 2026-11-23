"""Cues must not silently change the physical task or its labels."""
from io import BytesIO
import json
import unittest
from PIL import Image
from utils.evaluate_obstacle_cues import cue_sample


class ObstacleCueTests(unittest.TestCase):
    def fields(self):
        stream=BytesIO(); Image.new('RGB',(64,64),'white').save(stream,format='PNG')
        state=dict(task_instruction='Form a triangle.', agents=[dict(pos=[0.,0.])], episode_meta=dict(done=False))
        return {'overhead.png':stream.getvalue(),'state.json':json.dumps(state).encode(),
                'task.txt':b'Form a triangle.','reward.json':b'1.25','control_rewards.json':b'[1.25]',
                'actions.npy':b'unchanged action bytes','progress.json':b'{"team_progress":0.2}'}

    def test_language_changes_only_prompt_and_metadata(self):
        original=self.fields(); new=cue_sample(original,[[.5,0.,.12]],'language')
        self.assertEqual(new['overhead.png'],original['overhead.png'])
        self.assertIn(b'visual-only',new['task.txt'])
        self.assertNotIn(b'avoid',new['task.txt'])
        for key in ('reward.json','control_rewards.json','actions.npy','progress.json'):
            self.assertEqual(new[key],original[key])
        self.assertEqual(json.loads(new['state.json'])['agents'],json.loads(original['state.json'])['agents'])

    def test_highlight_keeps_instruction_and_image_geometry(self):
        original=self.fields(); new=cue_sample(original,[[.5,0.,.12]],'highlight')
        self.assertEqual(new['task.txt'],original['task.txt'])
        self.assertNotEqual(new['overhead.png'],original['overhead.png'])
        self.assertEqual(Image.open(BytesIO(new['overhead.png'])).size,(64,64))
        self.assertEqual(new['reward.json'],original['reward.json'])


if __name__=='__main__': unittest.main()
