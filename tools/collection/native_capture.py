"""Collection-only overhead capture resolution; actor tensor shape is unchanged."""
def configure_native_overhead(env_cfg, size):
    size = int(size)
    if size < 128:
        raise ValueError('capture resolution must be at least 128')
    cameras = [getattr(env_cfg, name) for name in ('global_camera_cfg', 'tiled_camera')
               if getattr(env_cfg, name, None) is not None]
    if len(cameras) != 1:
        raise ValueError('expected exactly one registered overhead camera')
    actor_size = tuple(env_cfg.actor_camera_image_size)
    cameras[0].width = cameras[0].height = size
    env_cfg.vlcm_camera_image_size = (size, size)
    if hasattr(env_cfg, 'camera_image_size'):
        env_cfg.camera_image_size = (size, size)
    for name in ('state_space', 'observation_space'):
        space = getattr(env_cfg, name, None)
        if isinstance(space, dict) and 'image' in space:
            import gymnasium as gym
            space['image'] = gym.spaces.Box(low=0., high=1., shape=(size, size, 3), dtype=float)
    assert tuple(env_cfg.actor_camera_image_size) == actor_size
    return {'native_render_size': [size, size], 'saved_image_size': [size, size],
            'actor_image_size': list(actor_size), 'upscaled': False}
