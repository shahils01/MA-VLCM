import numpy as np
import pytest
from PIL import Image

from ma_vlcm.train import (
    _build_tb3_prompt,
    _canonicalize_tb3_image,
    _progress_target_from_sample,
    _split_shard_sources,
    _tb3_task_context,
)


def test_combined_hugging_face_sources_are_split_without_splitting_globs():
    sources = _split_shard_sources(
        "hf://datasets/adi2440/tb3-isaac-vlcm/**/*.tar;"
        "hf://datasets/adi2440/tb3-isaac-avoid-obstacles-vlcm/**/*.tar"
    )
    assert len(sources) == 2
    assert sources[0].endswith("/**/*.tar")
    assert "avoid-obstacles" in sources[1]


def test_v2_target_uses_one_geometric_definition_for_legacy_and_new_labels():
    state = {
        "agents": [
            {"id": 0, "dist_to_goal": 1.0},
            {"id": 1, "dist_to_goal": 3.0},
        ]
    }
    initial = [2.0, 6.0]
    legacy = {"progress.json": {"schema": "tb3_progress_v1", "target": 0.1}}
    modern = {"progress.json": {"schema": "tb3_progress_v2", "target": 0.5}}
    assert _progress_target_from_sample(
        legacy, state, "tb3_progress_v2", initial
    ) == pytest.approx(0.5)
    assert _progress_target_from_sample(
        modern, state, "tb3_progress_v2", initial
    ) == pytest.approx(0.5)


def test_camera_canonicalization_center_crops_without_rotation_or_reflection():
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, 2:6, 0] = np.arange(16, dtype=np.uint8).reshape(4, 4)
    image = Image.fromarray(pixels, mode="RGB")
    canonical = _canonicalize_tb3_image(image, "center_square", 4, 4)
    np.testing.assert_array_equal(np.asarray(canonical)[:, :, 0], pixels[:, 2:6, 0])


def test_task_prompts_distinguish_domain_agent_count_and_obstacle_count():
    goal_state = {
        "environment": "simulation",
        "num_agents": 3,
        "agents": [{"id": index} for index in range(3)],
        "task_instruction": "Navigate every TurtleBot3 to its assigned goal.",
    }
    obstacle_state = {
        **goal_state,
        "task": "MARL-HPC-v1",
        "task_domain": "static_obstacles",
        "num_static_obstacles": 3,
    }
    goal_context = _tb3_task_context(goal_state)
    obstacle_context = _tb3_task_context(obstacle_state)
    assert goal_context[:3] == ("goal_to_goal", 3, 0)
    assert obstacle_context[:3] == ("static_obstacles", 3, 3)

    prompt = _build_tb3_prompt(
        obstacle_state,
        {},
        ["Agent 0."],
        step_idx=7,
        outcome="running",
        termination_reason="",
    )
    assert "exactly 3 agents" in prompt
    assert "exactly 3 static obstacles" in prompt
    assert "Task domain: static_obstacles" in prompt
    assert "+X points right in the image, +Y points up in the image" in prompt
