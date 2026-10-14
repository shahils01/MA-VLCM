"""Task-only TB3 text: frame state and supervision never enter the instruction."""
import re
import warnings

TEXT_INPUT_CONTRACT = "task_instruction_only_v1"

# Explicit task overrides are literal instructions, never state templates.
_STATE_FIELDS = re.compile(
    r"\b(?:collision|reached|success|failure|done|reward|progress|target|"
    r"raw_target|goal_cost|agent_risk|obstacle_risk|timestep|step|"
    r"episode[_ ]outcome(?: token)?|termination[_ ]reason|position|"
    r"dist_to_goal|min_neighbor_dist|heading|linear_vel|angular_vel|goal_pos)"
    r"\s*[:=]", re.IGNORECASE,
)


def validate_task_instruction(instruction):
    text = str(instruction).strip()
    if not text:
        raise ValueError("Task instruction must not be empty")
    if _STATE_FIELDS.search(text) or re.search(r"\{[^{}]+\}", text):
        raise ValueError("Task text must contain only the task instruction, without state flags, labels or dynamic placeholders")
    return text


def tb3_task_context(state_json, sample=None):
    """Return a canonical domain-specific instruction and prompt metadata."""

    agents = state_json.get("agents", [])
    num_agents = int(state_json.get("num_agents", len(agents)))
    task_name = str(state_json.get("task", "")).strip()
    stored_instruction = str(state_json.get("task_instruction", "")).strip()
    if not stored_instruction and sample is not None:
        task_value = sample.get("task.txt")
        if isinstance(task_value, bytes):
            stored_instruction = task_value.decode("utf-8", errors="replace").strip()
        elif isinstance(task_value, str):
            stored_instruction = task_value.strip()

    domain = str(state_json.get("task_domain", "")).strip().lower().replace("-", "_")
    evidence = f"{task_name} {stored_instruction}".lower().replace("-", "_")
    if not domain:
        if "static obstacle" in evidence or re.search(r"marl(?:_hpc)?_v(?:1|3(?:\.0)?)", evidence):
            domain = "static_obstacles"
        else:
            domain = "goal_to_goal"

    if domain == "static_obstacles":
        num_obstacles = int(state_json.get("num_static_obstacles", 3))
        instruction = (
            f"Coordinate exactly {num_agents} TurtleBot3 agents so every agent reaches "
            f"its assigned goal while avoiding all {num_obstacles} static obstacles "
            "and collisions with teammates."
        )
    else:
        num_obstacles = 0
        instruction = (
            f"Coordinate exactly {num_agents} TurtleBot3 agents so every agent reaches "
            "its assigned goal while avoiding collisions with teammates. This "
            "goal-to-goal environment has no static obstacles."
        )
    return domain, num_agents, num_obstacles, instruction



def build_tb3_task_prompt(state_json, sample=None, task_instruction=None):
    """Use task identity/cardinality only; never serialize observation fields."""
    if task_instruction is not None:
        return validate_task_instruction(task_instruction)
    return validate_task_instruction(tb3_task_context(state_json, sample)[3])


def warn_legacy_text_contract(args):
    """Make sanitized inference on state-augmented checkpoints explicit."""
    contract = args.get("text_input_contract") if isinstance(args, dict) else getattr(args, "text_input_contract", None)
    if contract != TEXT_INPUT_CONTRACT:
        warnings.warn(
            "Checkpoint predates task-only text: its training prompts may contain "
            "state/collision/reached flags. Current inputs are task-only. This is "
            "a distribution-shift diagnostic, not validation of a cleanly trained "
            "vision-language critic; fresh training is required.",
            UserWarning, stacklevel=2,
        )
        return True
    return False
