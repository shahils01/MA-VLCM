import json
import tarfile
import numpy as np

def generate_offroad_prompt(json_path, clip_len=5):
    print("==============================================")
    print("OFFROAD TEXT PROMPT (clip_len=5)")
    print("==============================================")
    
    with open(json_path, 'r') as f:
        state_json = json.load(f)
        
    agents = state_json.get("agents", [])
    n_ag = len(agents)
    step_idx = state_json.get("episode_meta", {}).get("step", 0)
    
    obs_lines = []
    for i, ag in enumerate(agents[:5]):
        ag_id = ag.get("id", i)
        color = ag.get("color", "unknown")
        pos = ag.get("pos", [0.0, 0.0])
        yaw = ag.get("yaw", 0.0)
        vel = ag.get("vel", [0.0, 0.0])
        v = np.linalg.norm(vel)
        dist_to_goal = ag.get("dist_to_goal", 0.0)
        traversability = ag.get("traversability", 0.0)
        reached = "yes" if ag.get("reached", False) else "no"
        collision = "yes" if ag.get("collision", False) else "no"

        obs_lines.append(
            f"Agent {ag_id} ({color}): position ({pos[0]:.2f}, {pos[1]:.2f}), "
            f"heading {yaw:.2f} rad, speed {v:.2f} m/s, dist_to_goal {dist_to_goal:.2f}m, "
            f"traversability {traversability:.2f}, reached: {reached}, collision: {collision}."
        )

    header = (
        "You are a vision-language critic model that evaluates"
        " multi-agent trajectories by estimating the expected"
        " cumulative return."
        f" This is an offroad navigation environment with {n_ag}"
        " agents traversing rough terrain."
        " Each agent must reach its color-matched goal while"
        " minimizing traversability cost and avoiding inter-agent collisions."
        " Given the video frames and agent states below, assess"
        " the quality of the current policy. "
    )
    
    # Simulate clip_len by duplicating the same line for simplicity of the prompt demo
    full_prompt = header
    for step in range(clip_len):
        full_prompt += f"Timestep: {step_idx + step}. " + " ".join(obs_lines) + " "
        
    print(full_prompt)
    print()

def generate_rware_prompt(tar_path, clip_len=5):
    print("==============================================")
    print("RWARE TEXT PROMPT (clip_len=5)")
    print("==============================================")
    
    # Extract the first state.json from the tar file
    state_json = None
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith(".state.json"):
                f = tar.extractfile(member)
                state_json = json.loads(f.read().decode('utf-8'))
                break
                
    if not state_json:
        print("No state.json found in RWARE tar.")
        return
        
    agents = state_json.get("agents", [])
    n_ag = len(agents)
    step_idx = state_json.get("step", 0)
    
    obs_lines = []
    for i, ag in enumerate(agents):
        ag_id = ag.get("id", i)
        pos = ag.get("pos", [0.0, 0.0])
        dir_val = ag.get("dir", [0, 1])
        dirs = {(1, 0): "SOUTH", (0, 1): "EAST", (-1, 0): "NORTH", (0, -1): "WEST"}
        facing = dirs.get((dir_val[0], dir_val[1]), "UNKNOWN")
        carrying = "yes" if ag.get("carrying", False) else "no"
        actions = ["NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE_LOAD"]
        action_idx = ag.get("action", 0)
        if isinstance(action_idx, int):
            action_str = actions[action_idx] if action_idx < len(actions) else "UNKNOWN"
        else:
            action_str = str(action_idx)

        obs_lines.append(
            f"Agent {ag_id}: at {pos}, facing {facing}, action {action_str}, carrying {carrying}."
        )

    requested = state_json.get("requested_boxes", [])
    
    header = (
        "You are a vision-language critic model that evaluates"
        " multi-agent trajectories by"
        " estimating the expected"
        " cumulative return."
        f" This is a robotic warehouse"
        f" environment with {n_ag}"
        f" agents (hard"
        f" difficulty, config: tiny-2ag-hard)."
        " Agents must navigate to"
        " requested shelf locations,"
        " pick up the correct boxes,"
        " deliver them to the goal"
        " area, and avoid collisions"
        " (distance < 3m)."
        " Given the video frames and"
        " agent states below, assess"
        " the quality of the current"
        " policy. "
    )
    
    full_prompt = header
    for step in range(clip_len):
        full_prompt += f"Timestep: {step_idx + step}. Requested boxes: {requested}. " + " ".join(obs_lines) + " "
        
    print(full_prompt)
    print()

if __name__ == "__main__":
    offroad_json = "/Users/aditya/Desktop/VLCM/VLCM_Data_Collection/OFFROAD/data_test/shard-000024/traj_230_step_0000.state.json"
    rware_tar = "/Users/aditya/Desktop/VLCM/VLCM_Data_Collection/RWARE/data_test/trajectory_005945_success.tar"
    
    generate_rware_prompt(rware_tar, clip_len=5)
    generate_offroad_prompt(offroad_json, clip_len=5)
