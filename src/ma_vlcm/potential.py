"""Geometry-grounded state potentials; independent of episode time/outcome."""

from dataclasses import dataclass
import math


POTENTIAL_SCHEMA = "tb3_potential_v1"
ROUTE_KEYS = ("route_dist_to_goal", "route_distance_to_goal", "geodesic_dist_to_goal",
              "geodesic_distance_to_goal", "path_dist_to_goal", "path_distance_to_goal")


@dataclass(frozen=True)
class PotentialConfig:
    robot_radius_m: float = 0.105
    goal_radius_m: float = 0.2
    clearance_band_m: float = 0.3
    agent_weight: float = 2.0
    obstacle_weight: float = 2.0
    distance_mode: str = "euclidean"

    def __post_init__(self):
        for key in ("robot_radius_m", "goal_radius_m", "clearance_band_m",
                    "agent_weight", "obstacle_weight"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.robot_radius_m <= 0 or self.clearance_band_m <= 0:
            raise ValueError("robot radius and clearance band must be positive")
        if self.distance_mode not in ("euclidean", "route_required"):
            raise ValueError("distance_mode must be euclidean or route_required")


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value, name):
    if value is None or len(value) < 2:
        raise ValueError(f"{name} needs x/y coordinates")
    return (_finite(value[0], name), _finite(value[1], name))


def goal_distance(agent, config):
    key = "dist_to_goal"
    if config.distance_mode == "route_required":
        key = next((k for k in ROUTE_KEYS if agent.get(k) is not None), None)
        if key is None:
            raise ValueError("route_required needs route distance for every frame/agent")
    if key not in agent:
        raise ValueError(f"missing goal distance: {key}")
    distance = _finite(agent[key], key)
    if distance < 0:
        raise ValueError("goal distance cannot be negative")
    return distance


def obstacle_clearance(position, obstacle, radius):
    """Surface clearance for circles or unrotated axis-aligned rectangles."""
    center = _point(obstacle.get("pos"), "obstacle.pos")
    if "radius_m" in obstacle:
        obstacle_radius = _finite(obstacle["radius_m"], "obstacle.radius_m")
        if obstacle_radius <= 0:
            raise ValueError("obstacle radius must be positive")
        return math.dist(position, center) - obstacle_radius - radius
    size = _point(obstacle.get("size_m"), "obstacle.size_m")
    if min(size) <= 0:
        raise ValueError("obstacle sizes must be positive")
    if abs(_finite(obstacle.get("yaw", 0.0), "obstacle.yaw")) > 1e-8:
        raise ValueError("rotated rectangles are unsupported; supply verified geometry")
    dx = abs(position[0] - center[0]) - size[0] / 2
    dy = abs(position[1] - center[1]) - size[1] / 2
    signed_distance = math.hypot(max(dx, 0), max(dy, 0)) + min(max(dx, dy), 0)
    return signed_distance - radius


def compute_potential(state, initial_distances, config=PotentialConfig(), obstacles=None):
    """Return a bounded potential, with worst-case rather than averaged hazards.

    U = mean(normalized remaining goal distance) + weighted clearance costs.
    Phi = (1-U)/(1+U), so safe success=1, safe start~0 and retreat<0.
    Obstacle geometry must be explicit (an empty list means no obstacles).
    """
    agents = sorted(state.get("agents", []), key=lambda a: a["id"])
    if not agents or len({a["id"] for a in agents}) != len(agents):
        raise ValueError("state requires nonempty agents with unique IDs")
    if set(initial_distances) != {a["id"] for a in agents}:
        raise ValueError("initial distance IDs must match the episode agents")
    if obstacles is None:
        obstacles = state.get("obstacles")
    domain = str(state.get("task_domain", "")).replace("-", "_")
    instruction = str(state.get("task_instruction", "")).lower()
    needs_obstacles = domain == "static_obstacles" or "obstacle" in instruction
    if config.obstacle_weight > 0 and obstacles is None and needs_obstacles:
        raise ValueError("obstacle task lacks geometry; provide an episode layout manifest")
    obstacles = obstacles or []
    if config.obstacle_weight > 0 and needs_obstacles and not obstacles:
        raise ValueError("obstacle task requires a nonempty verified obstacle layout")
    positions = [_point(a.get("pos"), "agent.pos") for a in agents]
    ratios = []
    for agent in agents:
        initial = _finite(initial_distances[agent["id"]], "initial distance")
        if initial < 0:
            raise ValueError("initial distance cannot be negative")
        ratios.append(max(goal_distance(agent, config) - config.goal_radius_m, 0.0)
                      / max(initial - config.goal_radius_m, config.goal_radius_m, 1e-6))
    pair_clearances = [math.dist(p, q) - 2 * config.robot_radius_m
                       for i, p in enumerate(positions) for q in positions[i+1:]]
    obstacle_clearances = [obstacle_clearance(p, o, config.robot_radius_m)
                           for p in positions for o in obstacles]
    def risk(clearances):
        return max((max(0.0, min(1.0, 1 - c/config.clearance_band_m))**2
                    for c in clearances), default=0.0)
    agent_risk, obstacle_risk = risk(pair_clearances), risk(obstacle_clearances)
    collision = any(a.get("collision", False) for a in agents) or any(
        c <= 0 for c in pair_clearances + obstacle_clearances)
    goal_cost = sum(ratios) / len(ratios)
    energy = goal_cost + config.agent_weight * agent_risk + config.obstacle_weight * obstacle_risk
    value = -1.0 if collision else 2.0 / (1 + energy) - 1.0
    result = {"schema": POTENTIAL_SCHEMA, "target": value, "goal_cost": goal_cost,
            "agent_risk": agent_risk, "obstacle_risk": obstacle_risk,
            "collision": collision, "initial_distances": [initial_distances[a["id"]] for a in agents],
            "agent_ids": [a["id"] for a in agents]}
    if config.distance_mode == "route_required":
        result["initial_route_distances"] = list(result["initial_distances"])
    return result


def transition_reward(current, following, gamma=1.0):
    """Bound a transition delta to [-1,1]; terminal handling belongs to MARL."""
    current, following = _finite(current, "current"), _finite(following, "following")
    gamma = _finite(gamma, "gamma")
    if not 0 <= gamma <= 1 or max(abs(current), abs(following)) > 1:
        raise ValueError("potentials must be in [-1,1] and gamma in [0,1]")
    return (gamma * following - current) / (1 + gamma)
