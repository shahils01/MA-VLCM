"""Causal 20 Hz state / 5 Hz image indices used by python_mappo.py.

This helper only constructs histories; train_mappo.py trains the critic using
these same asynchronous histories on fresh Python MAPPO rollouts.
"""
import numpy as np


def context_indices(anchor, episode_start, camera_steps, clip_len=6,
                    state_stride=4, control_dt=.05, max_image_age_steps=3):
    if not isinstance(anchor, (int, np.integer)) or not isinstance(episode_start, (int, np.integer)):
        raise ValueError('State indices must be integers')
    if anchor < episode_start or episode_start < 0 or min(clip_len,state_stride) < 1 or control_dt <= 0:
        raise ValueError('Invalid state index or timing configuration')
    camera_steps = np.asarray(camera_steps)
    if camera_steps.ndim != 1 or not np.issubdtype(camera_steps.dtype, np.integer):
        raise ValueError('camera_steps must be a one-dimensional integer sequence')
    if len(camera_steps) == 0 or np.any(np.diff(camera_steps) <= 0):
        raise ValueError('Camera steps must be nonempty and strictly increasing')
    valid_cameras = np.flatnonzero((camera_steps >= episode_start) & (camera_steps <= anchor))
    if not len(valid_cameras) or camera_steps[valid_cameras[0]] != episode_start:
        raise ValueError('Each episode must have its own initial camera frame')
    requested = anchor - np.arange(clip_len - 1, -1, -1) * state_stride
    frame_mask = requested >= episode_start
    states = np.maximum(requested, episode_start)
    local = np.searchsorted(camera_steps[valid_cameras],states,side='right') - 1
    cameras = valid_cameras[local]
    age_steps = states - camera_steps[cameras]
    if np.any(age_steps < 0):
        raise ValueError('Future camera frame selected')
    if max_image_age_steps is not None and np.any(age_steps[frame_mask] > max_image_age_steps):
        raise ValueError('Camera gap exceeds the trained image-age limit')
    return dict(state_indices=states, camera_indices=cameras, frame_mask=frame_mask,
                times=(states - anchor) * control_dt, image_age=age_steps * control_dt)
