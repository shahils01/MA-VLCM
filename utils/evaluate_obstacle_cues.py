"""Matched visual-only formation obstacle cues for frozen critic evaluation.

Changes only instructions and/or image overlays on existing rollouts; actor
actions, robot states, rewards and returns are preserved byte-for-byte.
"""

# Support both direct execution and imports from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_rollout_evaluation import ROOT, DEFAULT_OUTPUT, MODELS, score, render_result, write_json

VARIANTS = ("language", "highlight", "language_highlight")


def obstacle_instruction(original, count, highlighted=False):
    appearance = "dark gray circular obstacles outlined in red" if highlighted else "dark gray circular obstacles"
    return (original.rstrip() + f" There are {count} {appearance} in this scene. "
            "They are visual-only scene markers: they do not block robot motion or "
            "affect collision handling or rewards. Keep the original formation objective.")


def highlight_obstacles(png, obstacles, view_size_m=3.048):
    image = Image.open(BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for x, y, radius in obstacles:
        cx, cy = (x/view_size_m+.5)*width, (.5-y/view_size_m)*height
        rx, ry = radius/view_size_m*width+2, radius/view_size_m*height+2
        draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), outline="#ff3030", width=2)
    stream = BytesIO(); image.save(stream, format="PNG")
    return stream.getvalue()


def cue_sample(fields, obstacles, variant):
    """Transform one frame, without modifying any physical observation/label."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown cue variant: {variant}")
    result = dict(fields)
    state = json.loads(fields["state.json"])
    prompt = state["task_instruction"]
    if variant in {"language", "language_highlight"}:
        prompt = obstacle_instruction(prompt, len(obstacles), highlighted=variant=="language_highlight")
        state["task_instruction"] = prompt
        result["task.txt"] = prompt.encode()
    state["evaluation_input_cue"] = dict(variant=variant, visual_only=True)
    result["state.json"] = json.dumps(state, sort_keys=True).encode()
    if variant in {"highlight", "language_highlight"}:
        result["overhead.png"] = highlight_obstacles(fields["overhead.png"], obstacles)
    return result


def prepare_variants(parent, output):
    originals = sorted((parent/"episodes/formation_visual_obstacles").rglob("episode_*.summary.json"))
    if not originals:
        raise ValueError("Run the original visual-only formation evaluations first")
    audits = []
    for summary_path in originals:
        original = json.loads(summary_path.read_text())
        source = Path(original["shard"])
        for variant in VARIANTS:
            case = f"formation_visual_obstacles_{variant}"
            destination = output/"episodes"/case/"formation_control"/f"agents_{original['num_agents']:02d}"/"worker_00"/source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            target_summary = destination.with_suffix(".summary.json")
            if target_summary.exists():
                audits.append(json.loads(target_summary.read_text())["cue_audit"]); continue
            with tarfile.open(source, "r:") as archive:
                frames = {}
                for member in archive:
                    if member.isfile():
                        key, suffix = member.name.split(".", 1)
                        frames.setdefault(key, {})[suffix] = archive.extractfile(member).read()
            temporary = destination.with_suffix(".tar.partial")
            with tarfile.open(temporary, "w") as archive:
                for key, fields in frames.items():
                    transformed = cue_sample(fields, original["obstacles"], variant)
                    for suffix, content in transformed.items():
                        if suffix not in {"state.json", "task.txt", "overhead.png"}:
                            assert content == fields[suffix], suffix
                        if suffix == "state.json":
                            state = json.loads(content); old = json.loads(fields[suffix])
                            state.pop("evaluation_input_cue")
                            state["task_instruction"] = old["task_instruction"]
                            assert state == old, "Physical state changed"
                        if variant=="language":
                            assert transformed["overhead.png"]==fields["overhead.png"]
                        if variant=="highlight":
                            assert transformed["task.txt"]==fields["task.txt"]
                        member = tarfile.TarInfo(f"{key}.{suffix}"); member.size=len(content)
                        archive.addfile(member, BytesIO(content))
            temporary.replace(destination)
            trace = destination.with_suffix(".control.npz")
            shutil.copy2(original["control_trace"], trace)
            if hashlib.sha256(trace.read_bytes()).digest() != hashlib.sha256(Path(original["control_trace"]).read_bytes()).digest():
                raise ValueError("Control trace changed while copying")
            audit = dict(passed=True, variant=variant, source=str(source), frames=len(frames),
                         physical_states_and_actions_unchanged=True, reward_and_progress_labels_unchanged=True,
                         full_control_trace_identical=True)
            row = dict(original, eval_case=case, shard=str(destination), control_trace=str(trace),
                summary_path=str(target_summary), cue_variant=variant, cue_audit=audit,
                tar_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            write_json(target_summary, row)
            audits.append(audit)
    write_json(output/"input_cue_audit.json", dict(passed=True, variants=len(audits), checked=audits))
    source = Path(json.loads(originals[0].read_text())["shard"])
    examples = [("Baseline", source)]
    for variant in VARIANTS:
        path = output/"episodes"/f"formation_visual_obstacles_{variant}"/Path(*source.parts[-4:])
        examples.append((variant.replace("_", " + ").title(), path))
    preview = Image.new("RGB", (1040, 295), "white")
    draw = ImageDraw.Draw(preview)
    draw.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for i, (label, path) in enumerate(examples):
        with tarfile.open(path) as archive:
            member = next(m for m in archive if m.name.endswith(".overhead.png"))
            scene = Image.open(BytesIO(archive.extractfile(member).read())).convert("RGB")
        preview.paste(scene, (i*260, 30)); draw.text((i*260+5, 6), label, fill="black")
    preview.save(output/"input_examples.png")


def report(parent, output):
    baseline = {(r["episode"]["num_agents"], r["episode"]["seed"]):r for r in
        json.loads((parent/"evaluation_results.json").read_text()) if r["episode"]["eval_case"]=="formation_visual_obstacles"}
    results = json.loads((output/"evaluation_results.json").read_text())
    lines = ["# Visual-only formation obstacles: language and visual cues", "",
        "Three original episodes per team (3 and 5 robots), evaluated by both frozen joint critics.",
        "Trajectories, physical state, actions, rewards and return targets are identical across variants.",
        "The original gray disks are visual-only; these are critic-input ablations, not obstacle-avoidance policies.", "",
        "| Cue | Image | Instruction |", "|---|---|---|",
        "| Baseline | Original gray disks | Original formation task |",
        "| Language | Original gray disks | Explicit visual-only obstacle description |",
        "| Highlight | Red outlines around disks | Original formation task |",
        "| Language + highlight | Red outlines | Explicit visual-only description mentioning outlines |", "",
        "[Compare the four visual inputs](input_examples.png). The language-only variant intentionally leaves the image unchanged.", "",
        "## Added instruction", "",
        "> There are 3 dark gray circular obstacles in this scene. They are visual-only scene markers: they do not block robot motion or affect collision handling or rewards. Keep the original formation objective.", "",
        "The combined variant additionally identifies the red outlines. No reward, success or future outcome appears in the instruction.", "",
        "## Videos and return-estimation error", "",
        "| Cue | Agents | Seed | No-obstacle RMSE | Obstacle RMSE | Video |", "|---|---:|---:|---:|---:|---|"]
    comparisons = []
    for result in results:
        row = result["episode"]
        old = baseline[(row["num_agents"], row["seed"])]
        if result["times"] != old["times"] or result["target_return"] != old["target_return"]:
            raise ValueError("Cue variants must preserve the original return targets and timestamps")
        video = render_result(result, output)
        errors = [result["metrics"][m]["rmse"] for m in MODELS]
        lines.append(f"| {row['cue_variant']} | {row['num_agents']} | {row['seed']} | {errors[0]:.3f} | {errors[1]:.3f} | [MP4]({video}) |")
        for model in MODELS:
            prediction_delta = np.asarray(result["predicted_return"][model][:-1])-np.asarray(old["predicted_return"][model][:-1])
            comparisons.append(dict(cue=row["cue_variant"], num_agents=row["num_agents"], seed=row["seed"], critic=model,
                baseline_rmse=old["metrics"][model]["rmse"], cue_rmse=result["metrics"][model]["rmse"],
                rmse_change=result["metrics"][model]["rmse"]-old["metrics"][model]["rmse"],
                mean_absolute_prediction_change=float(np.abs(prediction_delta).mean())))
    write_json(output/"cue_comparison.json", comparisons)
    with (output/"cue_comparison.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    lines += ["", "## Interpretation", "",
        "Because rewards and trajectories are unchanged, the correct realized return is invariant to these cues. Prediction changes measure sensitivity; improvement is not guaranteed.",
        "The image and instruction are jointly encoded by the VLM. These frozen critics were not trained on explicit obstacle descriptions for formation, so language cues are another input-distribution shift.",
        "Highlight-only isolates the visual change; language-only isolates the prompt change. The combined condition also describes the outlines.",
        "The fixed MAPPO actor does not consume these images or instructions. Actual avoidance would require an obstacle-aware actor/controller and a physically defined obstacle task.",
        "Three episodes per team are illustrative, not a statistically powered comparison. See cue_comparison.csv for paired errors and prediction changes.", ""]
    (output/"README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "report", "all"))
    parser.add_argument("--parent", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=ROOT/"outputs/ckpts/critic35_sequential")
    parser.add_argument("--device", default="cuda")
    args=parser.parse_args(); parent=args.parent.resolve(); output=parent/"obstacle_cues"
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"prepare", "all"}: prepare_variants(parent, output)
    if args.phase in {"score", "all"}: score(output, args.checkpoints, args.device)
    if args.phase in {"report", "all"}: report(parent, output)


if __name__=="__main__":
    main()
