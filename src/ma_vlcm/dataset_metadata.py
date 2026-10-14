"""Resolve historical episode domains from verified source layout, fail on conflict."""
from .sampling import tb3_shard_stratum


def canonical_task_domain(state, source):
    domain, agents = tb3_shard_stratum(source, (3,4,5,6), ('goal_to_goal','static_obstacles'))
    stored = str(state.get('task_domain','')).strip().lower().replace('-','_').replace(' ','_')
    stored = {'gtg':'goal_to_goal', 'go_to_goal':'goal_to_goal', 'avoid_obstacles':'static_obstacles'}.get(stored,stored)
    if stored and stored not in ('goal_to_goal','static_obstacles'):
        raise ValueError('unrecognized stored task domain')
    if domain is None:
        raise ValueError('cannot identify task domain from verified shard source')
    if stored and stored != domain:
        raise ValueError('task domain conflicts with verified shard source')
    if agents is not None and len(state['agents']) != agents:
        raise ValueError('agent count conflicts with verified shard source')
    return domain
