"""Read-only, CPU episode comparison dashboard; no VLM weights are loaded."""

# Support both direct execution and imports from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tarfile
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from data_utils import OBSERVATION_NAMES, discover_episodes, prepare_episode, read_episode

FIELDS = ("overhead.png", "state.json", "dist.npy", "adj.npy", "actions.npy",
          "reward.json", "control_rewards.json", "progress.json", "task.txt")


def finite_list(values):
    """JSON has no NaN/Inf; unavailable diagnostics are explicit null values."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return float(values) if np.isfinite(values) else None
    return [finite_list(value) for value in values]


@lru_cache(maxsize=8)
def load_archive(path, summary_path):
    ep = read_episode(path, summary_path)
    frames = []
    images = []
    with tarfile.open(path, "r:") as archive:
        members = {member.name: member for member in archive if member.isfile()}
        if len(members) != sum(member.isfile() for member in archive.getmembers()):
            raise ValueError("Duplicate archive members")
        for frame in ep["frame_ids"]:
            def get(suffix):
                member = members.get(f"{frame}.{suffix}")
                if member is None:
                    return None
                raw = archive.extractfile(member).read()
                if suffix.endswith(".npy"):
                    return np.load(BytesIO(raw), allow_pickle=False)
                return json.loads(raw) if suffix.endswith(".json") else raw.decode("utf-8")
            row = {suffix:get(suffix) for suffix in FIELDS if suffix != "overhead.png"}
            member = members[f"{frame}.overhead.png"]
            images.append((member.offset_data, member.size))
            frames.append(row)
    return ep, frames, images


def inspect_episode(path, summary_path, gamma=.99, image_hz=10.):
    if not 0 < gamma <= 1 or not np.isfinite(image_hz) or image_hz <= 0:
        raise ValueError("Use finite gamma in (0,1] and a positive image rate")
    ep, frames, images = load_archive(str(path), str(summary_path))
    summary = json.loads(Path(summary_path).read_text())
    checks = []
    def check(name, passed, detail, level="error"):
        checks.append(dict(name=name, status="pass" if passed else level, detail=str(detail)))
    times = ep["timestamps"].numpy()
    n, length = ep["num_agents"], len(frames)
    hz = ep["control_hz"]
    states = [row["state.json"] for row in frames]
    positions = np.array([[agent["pos"] for agent in state["agents"]] for state in states])
    goals = np.array([[agent["goal_pos"] for agent in state["agents"]] for state in states])
    actual_distances = np.linalg.norm(goals-positions, axis=-1)
    distances = np.array([[agent["dist_to_goal"] for agent in state["agents"]] for state in states])
    progress = [row["progress.json"] or {} for row in frames]
    team_progress = [row.get("team_progress", row.get("target")) for row in progress]
    per_progress = [row.get("per_agent_progress", state.get("progress",{}).get("per_agent", [None]*n))
                    for row,state in zip(progress,states)]
    min_progress = [min(row) if len(row)==n and all(x is not None for x in row) else None for row in per_progress]
    reached = [[agent.get("reached") for agent in state["agents"]] for state in states]
    reached_fraction = [float(np.mean(row)) if all(x is not None for x in row) else None for row in reached]
    pair_distances = np.linalg.norm(positions[:,:,None,:]-positions[:,None,:,:],axis=-1)
    pair_distances[:,np.arange(n),np.arange(n)] = np.inf
    nearest = pair_distances.min(axis=(1,2)) if n>1 else np.full(length,np.nan)
    commands = np.array([[agent.get("action_vector",[np.nan,np.nan]) for agent in state["agents"]] for state in states])
    nav = states[0].get("navigation",{})
    diameter = nav.get("robot_diameter_m")
    obstacle_clearance = []
    boundary_margin = []
    for state,pos in zip(states,positions):
        obstacles = np.asarray(state.get("navigation",{}).get("obstacles",[]),dtype=float).reshape(-1,3)
        if len(obstacles) and diameter is not None:
            gaps=np.linalg.norm(pos[:,None,:]-obstacles[None,:,:2],axis=-1)-obstacles[None,:,2]-diameter/2
            obstacle_clearance.append(float(gaps.min()))
        else:
            obstacle_clearance.append(None)
        bounds = np.asarray(ep["bounds"]).reshape(2,2)
        boundary_margin.append(float(np.minimum(pos-bounds[:,0],bounds[:,1]-pos).min()))
    noise=[]
    for state in states:
        task_metadata=state.get("navigation",state.get("formation",{}))
        delta=task_metadata.get("sampled_action_noise")
        noise.append(float(np.linalg.norm(np.asarray(delta),axis=-1).mean()) if delta is not None else None)
    missing = Counter(field for row in frames for field in FIELDS if field != "overhead.png" and row[field] is None)
    check("Nine-field archive schema", not missing, dict(missing) if missing else f"{length} frames with all nine members")
    check("Finite training observations/rewards", np.isfinite(ep["robot_obs"].numpy()).all() and np.isfinite(ep["rewards"].numpy()).all(), "No NaN or infinity in model observations or recorded rewards")
    check("Agent count", summary.get("num_agents",n)==n and all(len(s["agents"])==n for s in states), f"{n} physical robots")
    check("Increasing timestamps", bool(np.all(np.diff(times)>0)), f"Capture {ep['capture_hz']:g} Hz; control {hz:g} Hz")
    steps = np.diff(times)*hz
    check("Native control clock", bool(np.allclose(steps,np.round(steps),atol=1e-5)), "Every image interval contains an integer number of control steps")
    check("Distance labels", np.allclose(actual_distances,distances,atol=2e-5), "Stored goal distances agree with positions/goals")
    check("Stable instruction", all(state["task_instruction"]==ep["prompt"] for state in states) and
          all(row["task.txt"] in (None,ep["prompt"]) for row in frames), "State instruction and task.txt agree throughout the episode")
    check("Terminal flags", all(not state["episode_meta"].get("done",False) for state in states[:-1]) and
          bool(states[-1]["episode_meta"].get("done",False)), "Only the final frame is marked done; timeout still means truncated")
    check("Known continuation return", ep["complete"], f"Outcome: {ep['termination_reason']}; timeouts/unfinished episodes have unknown continuation", "warning")
    progress_ok = all(value is not None and np.isfinite(value) and -.00001 <= value <= 1.00001 for value in team_progress)
    progress_ok &= all(value is not None and np.isfinite(value) and -.00001 <= value <= 1.00001 for row in per_progress for value in row)
    means_match = all(len(row)==n and all(v is not None for v in row) and team is not None and
                      np.isclose(np.mean(row),team,atol=1e-5) for row,team in zip(per_progress,team_progress))
    check("Progress labels", progress_ok and means_match, "Finite team progress in [0,1], equal to mean per-agent progress")
    initial_distances = progress[0].get("initial_distances")
    if initial_distances is not None and progress[0].get("schema")=="tb3_progress_v2":
        initial = np.asarray(initial_distances)
        expected = np.clip((initial-distances)/np.maximum(initial,1e-6),0,1)
        check("Geometric progress definition", np.allclose(np.asarray(per_progress,dtype=float),expected,atol=2e-5),
              "Recorded progress agrees with clipped fractional reduction in initial goal distance")
    distance_matrix_ok = adjacency_ok = action_ok = True
    for i,row in enumerate(frames):
        expected = pair_distances[i].copy(); np.fill_diagonal(expected,0.)
        matrix = row["dist.npy"]
        distance_matrix_ok &= matrix is not None and matrix.shape==(n,n) and np.allclose(matrix,expected,atol=2e-5)
        adj = row["adj.npy"]
        valid_adj = adj is not None and adj.shape==(n,n) and np.isfinite(adj).all() and np.isin(adj,[0,1]).all() and np.array_equal(adj,adj.T)
        task_metadata=states[i].get("navigation",states[i].get("formation",{}))
        threshold = task_metadata.get("adjacency_distance_m")
        if valid_adj and threshold is not None:
            # Preserve the recorded float precision at strict threshold boundaries.
            expected_adj = (matrix if matrix is not None and matrix.shape==(n,n) else expected) < threshold
            if not task_metadata.get("adjacency_self_loops",True):
                np.fill_diagonal(expected_adj,False)
            valid_adj = np.array_equal(adj,expected_adj)
        adjacency_ok &= bool(valid_adj)
        action = row["actions.npy"]
        action_ok &= action is not None and action.shape==(n,2) and np.isfinite(action).all() and np.allclose(action,commands[i],atol=1e-6)
    check("Pairwise distances", distance_matrix_ok, "dist.npy is NxN and agrees with robot positions")
    check("Graph adjacency", adjacency_ok, "Symmetric binary NxN graph; declared navigation threshold/self-loops checked when available")
    check("Executed actions", action_ok, "Finite Nx2 actions.npy agrees with recorded physical command vectors")
    image_shapes = []
    with Path(path).open("rb") as stream:
        for offset,size in images:
            stream.seek(offset)
            with Image.open(BytesIO(stream.read(size))) as image:
                image_shapes.append((image.size,image.mode))
                image.verify()
    check("Images decode consistently", len(set(image_shapes))==1, str(image_shapes[0]))
    controls = ep["control_rewards"]
    control_times, control_values = [], []
    controls_ok = controls is not None and len(controls[0])==0
    if controls is not None:
        for i,chunk in enumerate(controls[1:],1):
            expected_steps = round(float(times[i]-times[i-1])*hz)
            controls_ok &= len(chunk)==expected_steps and bool(np.isfinite(chunk).all())
            control_times.extend((times[i]-(len(chunk)-1-np.arange(len(chunk)))/hz).tolist())
            control_values.extend(chunk)
    check("Full control reward stream", controls_ok, "Initial interval empty; later interval counts match timestamps and control Hz")
    if controls_ok:
        check("Reward endpoint alignment", all(not chunk or np.isclose(chunk[-1],float(ep["rewards"][i]),atol=1e-5)
                                               for i,chunk in enumerate(controls)),
              "Each captured frame reward equals the last native control reward in its interval")
    check("Capture cadence", np.allclose(np.diff(times)[:-1],1/ep["capture_hz"],atol=1e-5), "Regular captures; shorter final interval permitted")
    if controls is not None and "episode_return" in summary:
        check("Undiscounted episode return", np.isclose(sum(control_values),summary["episode_return"],rtol=1e-5,atol=1e-4),
              f"Recorded controls sum={sum(control_values):.6g}; summary={summary['episode_return']:.6g}")
    if ep["task_domain"]=="goal_to_goal":
        check("Obstacle condition", summary.get("condition",ep["navigation_condition"])==ep["navigation_condition"], ep["navigation_condition"])
        if nav.get("start_region_bounds_m"):
            bounds=np.asarray(nav["start_region_bounds_m"])
            check("Initial sampling region", bool(((positions[0]>=bounds[:,0]-1e-5)&(positions[0]<=bounds[:,1]+1e-5)).all()), str(bounds.tolist()))
        if nav.get("minimum_initial_agent_separation_m"):
            minimum=nav["minimum_initial_agent_separation_m"]
            check("Initial robot separation", nearest[0]>=minimum-1e-5, f"Observed {nearest[0]:.3f} m; required {minimum:g} m")
        arrived = np.asarray(reached,dtype=bool)
        stopped_after = arrived[:-1]
        check("Stopped after arrival", bool(np.all(np.linalg.norm(np.diff(positions,axis=0),axis=-1)[stopped_after] < 2e-5)), "Robots already reached in the previous state do not move")
    selected_indices = list(range(length))
    returns = [None]*length
    if controls_ok and ep["complete"]:
        prepared = prepare_episode(ep,gamma,"control",image_hz)
        selected_indices = prepared["selected_indices"]
        returns = prepared["returns"].tolist()
    else:
        ratio = ep["capture_hz"]/image_hz
        stride = round(ratio)
        if stride<1 or not np.isclose(ratio,stride):
            raise ValueError("Image Hz must divide the recorded capture rate")
        selected_indices = list(range(0,length,stride))
        if selected_indices[-1]!=length-1: selected_indices.append(length-1)
        returns = [None]*len(selected_indices)
    series = dict(time=times, frame_reward=ep["rewards"].numpy(), team_progress=team_progress,
        min_progress=min_progress, per_agent_progress=per_progress, reached_fraction=reached_fraction,
        goal_distance=distances, min_robot_distance=nearest, obstacle_clearance=obstacle_clearance,
        boundary_margin=boundary_margin, linear_command=commands[:,:,0], angular_command=commands[:,:,1],
        noise_magnitude=noise, control_time=control_times, control_reward=control_values,
        cumulative_reward=np.cumsum(control_values), return_time=times[selected_indices], target_return=returns)
    series["rms_slot_distance"] = np.sqrt(np.mean(distances**2,axis=-1))
    series["formation_mse"]=[state.get("formation",{}).get("mean_squared_slot_error_m2") for state in states]
    series["stable_steps"]=[state.get("formation",{}).get("stable_team_steps") for state in states]
    series["required_hold_steps"]=[state.get("formation",{}).get("formation_hold_steps") for state in states]
    if all(value is not None for value in series["formation_mse"]):
        check("Native formation error",np.allclose(series["formation_mse"],np.mean(distances**2,axis=-1),atol=2e-5),
              "Recorded native mean-squared slot error matches goal/slot distances")
    if ep["task_domain"]=="formation_control" and n>1:
        shape_error=[]
        for pos,target in zip(positions,goals):
            a=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
            b=np.linalg.norm(target[:,None]-target[None,:],axis=-1)
            shape_error.append(float(np.sqrt(np.mean((a-b)[np.triu_indices(n,1)]**2))))
        series["shape_error"]=shape_error
    else:
        series["shape_error"]=[None]*length
    density=[]
    for row in frames:
        adj=row["adj.npy"]
        density.append(float(adj[~np.eye(n,dtype=bool)].mean()) if adj is not None and n>1 else None)
    series["graph_density"]=density
    if summary.get("tar_sha256"):
        digest=hashlib.sha256()
        with open(path,"rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
        check("Archive SHA-256",digest.hexdigest()==summary["tar_sha256"],"Matches the recorded archive checksum")
    return dict(path=str(path), task=ep["task_domain"], agents=n, outcome=ep["termination_reason"],
        success=ep["success"], complete=ep["complete"], noise=summary.get("noise","unspecified"),
        condition=ep.get("navigation_condition"), stage=ep["stage"], prompt=ep["prompt"],
        frames=length, capture_hz=ep["capture_hz"], control_hz=hz, image_hz=image_hz, gamma=gamma,
        duration=float(times[-1]), start_time=float(times[0]), image_size=ep["image_size"],
        checkpoint_sha256=ep["policy_id"], collision_distance_m=diameter,
        model_inputs=["overhead.png","task instruction",list(OBSERVATION_NAMES),"adjacency","relative clip timestamps"],
        labels_not_inputs=["reward","discounted return","progress","success/failure","termination","checkpoint identity","noise profile"],
        checks=checks, series={key:finite_list(values) for key,values in series.items()},
        snapshot=[dict(time=float(times[i]),agents=state["agents"],
                       navigation=state.get("navigation"),formation=state.get("formation"),
                       episode_meta=state["episode_meta"]) for i,state in enumerate(states)],
        return_definition="Recorded full-control Monte Carlo return; incoming reward excluded; gamma per native control step. Not a learned expected value. Unknown at timeouts.")


class Catalog:
    def __init__(self, datasets=(), episodes=()):
        self.records=[]
        self.paths={}
        for label,source in list(datasets)+[("Selected episodes",path) for path in episodes]:
            if not Path(source).exists():
                raise FileNotFoundError(f"Local dataset/archive does not exist: {source}")
            for path,summary in discover_episodes(source):
                key=str(path.resolve())
                if key in self.paths.values(): continue
                info=json.loads(summary.read_text())
                token=str(len(self.records))
                condition=info.get("condition") or next((p for p in path.parts if p in {"with_obstacles","without_obstacles"}),"")
                task=info.get("task_domain") or next((p for p in path.parts if p in {"formation_control","goal_to_goal"}),"")
                self.records.append(dict(id=token,dataset=label,name=path.name,path=str(path),summary=str(summary),
                    task=task,agents=info.get("num_agents"),noise=info.get("noise","unspecified"),
                    condition=condition,outcome=info["termination_reason"],success=info["success"]))
                self.paths[token]=key
        if not self.records: raise ValueError("No episode archives found")


def make_handler(catalog):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                route=urlparse(self.path)
                query=parse_qs(route.query)
                if route.path=="/":
                    content=Path(__file__).with_name("dataset_inspector.html").read_bytes()
                    mime="text/html; charset=utf-8"
                elif route.path=="/api/catalog":
                    content=json.dumps(catalog.records).encode(); mime="application/json"
                elif route.path in {"/api/episode","/api/frame"}:
                    token=query.get("id",[""])[0]
                    if token not in catalog.paths: raise ValueError("Unknown episode ID")
                    record=catalog.records[int(token)]
                    if route.path=="/api/episode":
                        report=inspect_episode(record["path"],record["summary"],
                            float(query.get("gamma",[.99])[0]),float(query.get("image_hz",[10])[0]))
                        content=json.dumps(report,allow_nan=False).encode(); mime="application/json"
                    else:
                        _,_,images=load_archive(record["path"],record["summary"])
                        index=int(query.get("frame",[0])[0])
                        if not 0<=index<len(images): raise ValueError("Frame outside episode")
                        offset,size=images[index]
                        with open(record["path"],"rb") as stream:
                            stream.seek(offset); content=stream.read(size)
                        mime="image/png"
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Type",mime)
                self.send_header("Content-Length",str(len(content)))
                self.send_header("Cache-Control","no-store")
                self.send_header("X-Content-Type-Options","nosniff")
                self.end_headers(); self.wfile.write(content)
            except (ValueError,KeyError,OSError,AssertionError,tarfile.TarError,TypeError,IndexError) as error:
                content=json.dumps(dict(error=str(error))).encode()
                self.send_response(400); self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(content))); self.end_headers(); self.wfile.write(content)
        def log_message(self,fmt,*args):
            if "/api/frame" not in str(args): super().log_message(fmt,*args)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",nargs=2,action="append",default=[],metavar=("LABEL","LOCAL_ROOT"))
    parser.add_argument("--episode",action="append",default=[],help="Local tar path; repeat to preload up to four")
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--report",type=Path,help="Write JSON checks/series for --episode paths and exit")
    parser.add_argument("--gamma",type=float,default=.99)
    parser.add_argument("--image-hz",type=float,default=10.)
    args=parser.parse_args()
    if not args.dataset and not args.episode: parser.error("Supply at least one --dataset or --episode")
    for path in args.episode:
        if not Path(path).is_file(): parser.error(f"Local episode archive does not exist: {path}")
    if args.report:
        if not args.episode: parser.error("--report requires explicit --episode paths")
        reports=[]
        for path in args.episode:
            refs=discover_episodes(path)
            if len(refs)!=1: parser.error("Each --episode must identify exactly one tar")
            reports.append(inspect_episode(*refs[0],args.gamma,args.image_hz))
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(reports,allow_nan=False,indent=2)+"\n")
        print(f"Wrote {args.report}")
        return
    for _,source in args.dataset:
        if not Path(source).exists(): parser.error(f"Local dataset does not exist: {source}")
    catalog=Catalog(args.dataset,args.episode)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(catalog))
    print(f"Indexed {len(catalog.records)} episodes. Open http://127.0.0.1:{server.server_port}",flush=True)
    print("Read-only CPU inspection; no VLM weights or dataset modifications. Ctrl+C to stop.",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=="__main__": main()
