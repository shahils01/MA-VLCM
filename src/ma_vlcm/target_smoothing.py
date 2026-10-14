"""Causal training-label filter; reset independently for every real/rewind episode."""
import math
TRANSFORM = 'causal_gaussian_sigma1_window5_collision_override'


def smooth_targets(values, collisions, sigma=1.0, window=5):
    if len(values) != len(collisions) or sigma <= 0 or window < 1:
        raise ValueError('invalid causal smoothing configuration')
    result = []
    for i, value in enumerate(values):
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('APF targets must be finite in [-1,1]')
        if collisions[i]:
            result.append(-1.0)
            continue
        # Do not bleed the absorbing collision penalty into a safe recovery.
        begin = max(0, i - window + 1)
        for j in range(i - 1, begin - 1, -1):
            if collisions[j]:
                begin = j + 1
                break
        weights = [math.exp(-.5*((i-j)/sigma)**2) for j in range(begin, i+1)]
        result.append(sum(w*values[j] for j,w in zip(range(begin,i+1),weights))/sum(weights))
    return result
