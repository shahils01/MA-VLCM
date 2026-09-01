import numpy as np
import pytest
import torch
from PIL import Image

from ma_vlcm.train import (
    _agent_progress_distance,
    _effective_progress_distance_mode,
    _build_tb3_prompt,
    _canonicalize_tb3_image,
    _collate_vlm_inputs,
    _progress_target_from_sample,
    _qwen_video_metadata,
    _split_shard_sources,
    _task_domain_id_from_text,
    _temporal_consistency_loss,
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


def test_qwen_video_metadata_uses_five_hz_source_timestamps():
    metadata = _qwen_video_metadata(6, 5.0)[0]
    assert metadata["fps"] == pytest.approx(5.0)
    assert metadata["total_num_frames"] == 6
    assert metadata["duration"] == pytest.approx(1.2)
    assert metadata["frames_indices"] == [0, 1, 2, 3, 4, 5]
    timestamps = [
        index / metadata["fps"] for index in metadata["frames_indices"]
    ]
    assert timestamps == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    with pytest.raises(ValueError, match="positive"):
        _qwen_video_metadata(6, 0.0)


def test_qwen_minibatch_collation_flattens_all_visual_patches():
    items = []
    for index in range(16):
        pixels = torch.zeros((1, 512, 12)) if index % 2 else torch.zeros((512, 12))
        grid = torch.tensor([[8, 8, 8]]) if index % 2 else torch.tensor([8, 8, 8])
        items.append(
            {
                "input_ids": torch.arange(3 + index % 2),
                "attention_mask": torch.ones(3 + index % 2, dtype=torch.long),
                "pixel_values_videos": pixels,
                "video_grid_thw": grid,
            }
        )

    batch = _collate_vlm_inputs(items, "qwen3_vl")

    assert batch["input_ids"].shape == (16, 4)
    assert batch["pixel_values_videos"].shape == (8192, 12)
    assert batch["video_grid_thw"].shape == (16, 3)
    assert batch["video_grid_thw"].prod(dim=-1).sum().item() == 8192


def test_qwen_minibatch_collation_rejects_patch_grid_mismatch():
    with pytest.raises(RuntimeError, match="patch/grid mismatch"):
        _collate_vlm_inputs(
            [
                {
                    "input_ids": torch.arange(3),
                    "attention_mask": torch.ones(3, dtype=torch.long),
                    "pixel_values_videos": torch.zeros((16, 12)),
                    "video_grid_thw": torch.tensor([8, 8, 8]),
                }
            ],
            "qwen3_vl",
        )


def test_route_progress_prefers_route_metadata_with_legacy_fallback():
    route_state = {
        "agents": [
            {
                "id": 0,
                "dist_to_goal": 1.0,
                "initial_goal_distance": 2.0,
                "route_dist_to_goal": 8.0,
                "initial_route_dist_to_goal": 10.0,
            }
        ]
    }
    assert _progress_target_from_sample(
        {},
        route_state,
        "tb3_progress_v2",
        [10.0],
        progress_distance_mode="route_if_available",
    ) == pytest.approx(0.2)
    assert _progress_target_from_sample(
        {},
        route_state,
        "tb3_progress_v2",
        [2.0],
        progress_distance_mode="euclidean",
    ) == pytest.approx(0.5)
    assert _agent_progress_distance(
        {"dist_to_goal": 0.7}, mode="route_if_available"
    ) == pytest.approx(0.7)
    assert _effective_progress_distance_mode(
        route_state["agents"], {}, "route_if_available"
    ) == "route_required"
    assert _effective_progress_distance_mode(
        [{"dist_to_goal": 0.7, "route_dist_to_goal": 1.2}],
        {},
        "route_if_available",
    ) == "euclidean"
    with pytest.raises(ValueError, match="route_required"):
        _agent_progress_distance({"dist_to_goal": 0.7}, mode="route_required")


def test_prompt_domain_ids_and_temporal_delta_consistency():
    assert _task_domain_id_from_text("Task domain: goal_to_goal.") == 1
    assert _task_domain_id_from_text("Task domain: static_obstacles.") == 2
    assert _task_domain_id_from_text("unspecified task") == 0

    prediction = torch.tensor([0.2, 0.5])
    next_prediction = torch.tensor([0.3, 0.45])
    target = torch.tensor([0.1, 0.6])
    next_target = torch.tensor([0.2, 0.55])
    assert _temporal_consistency_loss(
        prediction, next_prediction, target, next_target
    ).item() == pytest.approx(0.0, abs=1e-7)


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
