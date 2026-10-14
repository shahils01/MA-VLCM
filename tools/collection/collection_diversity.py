"""Simulated behavior perturbations with timed recovery, never physical controls."""
import torch


def perturb_actions(actions, steps, hz, mode, noise_std=.75):
    if mode == 'clean':
        return actions
    if hz <= 0 or noise_std < 0 or actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError('invalid collection perturbation inputs')
    result = actions.clone()
    if mode == 'noise_recover':
        active = ((steps / hz) % 12 >= 3) & ((steps / hz) % 12 < 6)
        noisy = (actions + noise_std * torch.randn_like(actions)).clamp(-3, 3)
        return torch.where(active[:, None, None], noisy, actions)
    phase = (steps / hz) % 18
    active = (phase >= 3) & (phase < 7)
    if mode == 'stall_recover':
        result[:] = 0
    elif mode == 'retreat_recover':
        result[..., 0] = -2.0
        result[..., 1] = 0
    elif mode == 'straight_recover':
        result[..., 0] = 2.0
        result[..., 1] = 0
    else:
        raise ValueError('unknown perturbation mode')
    return torch.where(active[:, None, None], result, actions)
