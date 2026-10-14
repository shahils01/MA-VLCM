"""ALWAYS train Isaac VLCM critics on both task domains."""
import glob
from pathlib import Path

REQUIRED_DOMAINS = ('goal_to_goal', 'static_obstacles')
REQUIRED_REPOS = ('adi2440/tb3-isaac-vlcm', 'adi2440/tb3-isaac-avoid-obstacles-vlcm')


def require_both_tb3_sources(sources):
    """Reject missing or unidentifiable domains before allocating a backbone."""
    if isinstance(sources, (list, tuple)):
        paths = [str(p) for p in sources]
    else:
        paths = []
        for source in str(sources).split(';'):
            source = source.strip()
            if source.startswith('hf://datasets/'):
                paths.append(source)
            elif Path(source).is_dir():
                paths.extend(str(p) for p in Path(source).rglob('*.tar'))
            elif any(c in source for c in '*?['):
                paths.extend(glob.glob(source, recursive=True))
            elif Path(source).is_file() and source.endswith('.tar'):
                paths.append(source)
    domains = set()
    for path in paths:
        normalized = path.lower().replace('-', '_')
        if 'static_obstacles' in normalized or 'avoid_obstacles' in normalized:
            domains.add('static_obstacles')
        elif 'goal_to_goal' in normalized or 'tb3_isaac_vlcm' in normalized:
            domains.add('goal_to_goal')
    if domains != set(REQUIRED_DOMAINS):
        raise ValueError('ALWAYS train with BOTH adi2440/tb3-isaac-vlcm and '
                         'adi2440/tb3-isaac-avoid-obstacles-vlcm (or their relabeled '
                         'goal_to_goal/static_obstacles shards); missing: '
                         + ', '.join(sorted(set(REQUIRED_DOMAINS) - domains)))
    return domains
