from __future__ import annotations

import json

import numpy as np
import pytest

from zoomgen.camera import build_camera_frames, look_at
from zoomgen.color import process_frame
from zoomgen.config import GeneratorConfig
from zoomgen.random_camera import camera_random_seed, randomize_camera_config
from zoomgen.zoom import allocate_frame_counts, build_schedule


@pytest.fixture
def cfg(tmp_path):
    ply = tmp_path / "dummy.ply"
    ply.touch()
    lenses = []
    for name, limits, exposure, strength, offset in zip(
        ("ultrawide", "wide", "telephoto"),
        ((0.5, 0.95), (1.0, 2.95), (3.0, 5.0)),
        (-0.12, 0.08, -0.03),
        (0.15, -0.1, 0.0),
        ((-0.002, 0.0, 0.0), (0.0, 0.0, 0.0), (0.002, 0.0, 0.0)),
    ):
        lenses.append(
            {
                "name": name,
                "zoom_min": limits[0],
                "zoom_max": limits[1],
                "camera_center_offset_ratio": offset,
                "temporal_color_jump": {"strength": strength, "position": 0.5},
                "color": {
                    "exposure_ev": exposure,
                    "color_correction_matrix": np.eye(3).tolist(),
                    "noise_std": 0.001,
                },
            }
        )
    return GeneratorConfig.model_validate(
        {
            "input": {"ply_path": ply},
            "output": {"directory": tmp_path / "out"},
            "render": {},
            "video": {"width": 80, "height": 48, "fps": 10, "total_frames": 60},
            "scene_analysis": {},
            "camera": {
                "initial_view": {
                    "position": [2.0, 0.0, 0.0],
                    "look_at": [0.0, 0.0, 0.0],
                    "roll_deg": 7.0,
                },
                "motion": {"enabled": False},
            },
            "zoom": {
                "lens_switch": {
                    "color_mode": "crossfade",
                    "color_transition_frames": 2,
                    "camera_offset_mode": "crossfade",
                    "camera_transition_frames": 2,
                },
                "lenses": lenses,
            },
        }
    )


def _camera_frames(cfg):
    return build_camera_frames(
        cfg,
        build_schedule(cfg),
        np.array([2.0, 0.0, 0.0]),
        np.zeros(3),
        1.0,
    )


@pytest.mark.parametrize("curve", ["linear", "smoothstep"])
@pytest.mark.parametrize("seed", range(16))
def test_randomized_zoom_and_actual_focal_lengths_remain_ordered(cfg, curve, seed):
    cfg.zoom.curve = curve
    randomized, _ = randomize_camera_config(cfg, seed)
    schedule = build_schedule(randomized)
    original_frames = _camera_frames(cfg)
    frames = _camera_frames(randomized)

    assert len(schedule) == cfg.video.total_frames
    assert np.all(np.diff([frame.zoom_ratio for frame in schedule]) > 0.0)
    assert np.all(np.diff([frame.fx for frame in frames]) > 0.0)
    assert np.all(np.diff([frame.fy for frame in frames]) > 0.0)
    assert schedule[0].zoom_ratio == cfg.zoom.lenses[0].zoom_min
    assert schedule[-1].zoom_ratio == cfg.zoom.lenses[-1].zoom_max
    for index in (0, -1):
        assert frames[index].fx == original_frames[index].fx
        assert frames[index].fy == original_frames[index].fy
        assert frames[index].fov_y == original_frames[index].fov_y

    assert [lens.name for lens in randomized.zoom.lenses] == [
        lens.name for lens in cfg.zoom.lenses
    ]
    original_gaps = [
        incoming.zoom_min - outgoing.zoom_max
        for outgoing, incoming in zip(cfg.zoom.lenses, cfg.zoom.lenses[1:])
    ]
    randomized_gaps = [
        incoming.zoom_min - outgoing.zoom_max
        for outgoing, incoming in zip(randomized.zoom.lenses, randomized.zoom.lenses[1:])
    ]
    np.testing.assert_allclose(randomized_gaps, original_gaps, rtol=1e-10, atol=1e-12)
    assert randomized.zoom.frame_allocation == cfg.zoom.frame_allocation
    assert randomized.zoom.lens_switch == cfg.zoom.lens_switch


def test_randomization_is_reproducible_and_does_not_mutate_input(cfg):
    original = cfg.model_dump(mode="json")
    first, first_metadata = randomize_camera_config(cfg, 123)
    second, second_metadata = randomize_camera_config(cfg, 123)
    different, different_metadata = randomize_camera_config(cfg, 124)

    assert cfg.model_dump(mode="json") == original
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first_metadata == second_metadata
    assert first_metadata["seed"] == 123
    assert first_metadata["version"]
    assert len(first_metadata["config_sha256"]) == 64
    assert different_metadata["config_sha256"] != first_metadata["config_sha256"]
    assert first.model_dump(mode="json") != different.model_dump(mode="json")
    assert first.zoom.lenses[0].zoom_max != different.zoom.lenses[0].zoom_max
    assert first.camera.initial_view == cfg.camera.initial_view
    assert first.camera.initialization == cfg.camera.initialization
    assert first.camera.fov_y_deg_at_1x == cfg.camera.fov_y_deg_at_1x

    first.zoom.lenses[0].color.exposure_ev += 1.0
    assert cfg.model_dump(mode="json") == original
    assert second.model_dump(mode="json") != first.model_dump(mode="json")


def test_camera_seed_depends_on_scene_camera_and_base_seed():
    inputs = [
        (3407, "scene_a", "00001"),
        (3407, "scene_b", "00001"),
        (3407, "scene_a", "00002"),
        (3408, "scene_a", "00001"),
        (3407, "ab", "c"),
        (3407, "a", "bc"),
    ]
    seeds = [camera_random_seed(*arguments) for arguments in inputs]
    assert seeds == [camera_random_seed(*arguments) for arguments in inputs]
    assert len(set(seeds)) == len(inputs)
    assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)


def test_static_capture_has_distinct_lens_centers_without_rotating_scene(cfg):
    cfg.zoom.lens_switch.camera_offset_mode = "hard"
    randomized, _ = randomize_camera_config(cfg, 45)
    frames = _camera_frames(randomized)
    schedule = build_schedule(randomized)
    expected_rotation = look_at(
        np.array([2.0, 0.0, 0.0]), np.zeros(3), cfg.camera.initial_view.roll_deg
    )[:3, :3]

    assert randomized.camera.motion.enabled
    assert not any(randomized.camera.motion.translation_amplitude_ratio)
    assert not any(randomized.camera.motion.rotation_amplitude_deg)
    positions = []
    for lens_index in range(len(randomized.zoom.lenses)):
        lens_frames = [
            frame for frame, item in zip(frames, schedule) if item.lens_index == lens_index
        ]
        positions.append(lens_frames[0].position)
        for frame in lens_frames:
            np.testing.assert_array_equal(frame.position, lens_frames[0].position)
            np.testing.assert_array_equal(frame.c2w[:3, :3], expected_rotation)
    assert all(
        np.linalg.norm(first - second) > 0.0
        for first, second in zip(positions, positions[1:])
    )
    assert not cfg.camera.motion.enabled


@pytest.mark.parametrize("equal_exposure", [False, True])
def test_lens_exposure_differences_and_temporal_jump_direction_survive(cfg, equal_exposure):
    if equal_exposure:
        for lens in cfg.zoom.lenses:
            lens.color.exposure_ev = 0.0
    for seed in range(30):
        randomized, _ = randomize_camera_config(cfg, seed)
        for previous, current, old_previous, old_current in zip(
            randomized.zoom.lenses,
            randomized.zoom.lenses[1:],
            cfg.zoom.lenses,
            cfg.zoom.lenses[1:],
        ):
            difference = current.color.exposure_ev - previous.color.exposure_ev
            original = old_current.color.exposure_ev - old_previous.color.exposure_ev
            assert abs(difference) >= max(abs(original) * 0.6, 0.04) - 1e-12
            if original:
                assert difference * original > 0.0
        for original, lens in zip(cfg.zoom.lenses, randomized.zoom.lenses):
            assert np.sign(lens.temporal_color_jump.strength) == np.sign(
                original.temporal_color_jump.strength
            )


def test_processed_images_are_finite_reproducible_and_change_at_lens_switches(cfg):
    cfg.zoom.lens_switch.color_mode = "hard"
    for lens in cfg.zoom.lenses:
        lens.temporal_color_jump.strength = 0.0
    randomized, _ = randomize_camera_config(cfg, 321)
    gray = np.linspace(0.05, 0.65, 16 * 24, dtype=np.float32).reshape(16, 24)
    image = np.stack([gray * 0.9, gray, gray * 0.8], axis=-1)
    outputs = []
    for frame in build_schedule(randomized):
        if frame.local_index:
            continue
        encoded, linear, _ = process_frame(image, frame, randomized)
        assert encoded.shape == image.shape
        assert np.isfinite(encoded).all()
        assert np.isfinite(linear).all()
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0
        np.testing.assert_array_equal(encoded, process_frame(image, frame, randomized)[0])
        outputs.append(encoded)
    assert len(outputs) == 3
    assert all(
        not np.allclose(first, second)
        for first, second in zip(outputs, outputs[1:])
    )


@pytest.mark.parametrize("mode", ["few_frames", "minimal_frames", "explicit", "single_lens"])
def test_randomization_respects_frame_and_transition_constraints(cfg, mode):
    raw = cfg.model_dump(mode="python")
    if mode == "few_frames":
        raw["video"]["total_frames"] = 18
        raw["zoom"]["temporal_color_adjustment"]["transition_frames"] = 1
    elif mode == "minimal_frames":
        raw["video"]["total_frames"] = 6
        for index, lens in enumerate(raw["zoom"]["lenses"]):
            lens["zoom_min"] = 0.5 + 1.1 * index
            lens["zoom_max"] = 1.5 + 1.1 * index
        raw["zoom"]["temporal_color_adjustment"]["transition_frames"] = 1
    elif mode == "explicit":
        raw["video"]["total_frames"] = 9
        raw["zoom"]["frame_allocation"]["mode"] = "explicit"
        raw["zoom"]["temporal_color_adjustment"]["transition_frames"] = 2
        for lens in raw["zoom"]["lenses"]:
            lens["frame_count"] = 3
            lens["temporal_color_jump"]["position"] = 0.74
    else:
        raw["video"]["total_frames"] = 2
        raw["zoom"]["lenses"] = raw["zoom"]["lenses"][:1]
    constrained = GeneratorConfig.model_validate(raw)

    for seed in range(30):
        randomized, _ = randomize_camera_config(constrained, seed)
        GeneratorConfig.model_validate(randomized.model_dump(mode="python"))
        schedule = build_schedule(randomized)
        assert len(schedule) == constrained.video.total_frames
        assert all(count >= 2 for count in allocate_frame_counts(randomized))
        assert np.all(np.diff([frame.zoom_ratio for frame in schedule]) > 0.0)
        assert schedule[0].zoom_ratio == constrained.zoom.lenses[0].zoom_min
        assert schedule[-1].zoom_ratio == constrained.zoom.lenses[-1].zoom_max
        assert randomized.zoom.frame_allocation == constrained.zoom.frame_allocation
        assert randomized.zoom.temporal_color_adjustment == constrained.zoom.temporal_color_adjustment
        if mode == "explicit":
            assert allocate_frame_counts(randomized) == [3, 3, 3]


@pytest.mark.parametrize(
    "recorded, requested, expected",
    [
        ("legacy", "disabled", True),
        ("disabled", "disabled", True),
        ("enabled", "enabled", True),
        ("legacy", "enabled", False),
        ("disabled", "enabled", False),
        ("enabled", "disabled", False),
        ("enabled", "new_seed", False),
        ("enabled", "new_config", False),
        ("enabled", "new_version", False),
    ],
)
def test_video_resume_requires_matching_random_camera(
    cfg, monkeypatch, recorded, requested, expected
):
    from render_validated_zoom_dataset import _video_is_complete

    metadata = {"version": 1, "seed": 123, "config_sha256": "a" * 64}
    modes = {
        "disabled": None,
        "enabled": metadata,
        "new_seed": {**metadata, "seed": 124},
        "new_config": {**metadata, "config_sha256": "b" * 64},
        "new_version": {**metadata, "version": 2},
    }
    cfg.output.directory.mkdir()
    (cfg.output.directory / cfg.output.video_filename).touch()
    summary = {
        "traversal_pullback": cfg.camera.traversal_pullback.model_dump(mode="json")
    }
    if recorded != "legacy":
        summary["random_camera"] = modes[recorded]
    (cfg.output.directory / "generation_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    monkeypatch.setattr(
        "zoomgen.video.probe_video",
        lambda path: {
            "frame_count": cfg.video.total_frames,
            "width": cfg.video.width,
            "height": cfg.video.height,
            "fps": cfg.video.fps,
        },
    )

    assert _video_is_complete(cfg, modes[requested]) is expected
    if requested == "disabled":
        assert _video_is_complete(cfg) is expected


def test_random_camera_fingerprint_survives_output_reclassification(cfg, tmp_path):
    original_directory = tmp_path / "scene" / "invaild" / "position_0000"
    accepted_directory = tmp_path / "scene" / "vaild" / "position_0000"
    camera_to_world = look_at(
        np.array([2.0, 0.0, 0.0]), np.zeros(3), cfg.camera.initial_view.roll_deg
    )
    raw = cfg.model_dump(mode="python")
    raw["output"]["directory"] = original_directory
    raw["camera"]["initialization"]["resolved_pose"] = {
        "position": camera_to_world[:3, 3].tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "fov_y_deg_at_1x": cfg.camera.fov_y_deg_at_1x,
        "source": f"traversal_camera_json:{original_directory / 'lens_0000' / 'camera.json'}",
    }
    original = GeneratorConfig.model_validate(raw)
    _, original_metadata = randomize_camera_config(original, 123)

    raw["output"]["directory"] = accepted_directory
    raw["camera"]["initialization"]["resolved_pose"]["source"] = (
        f"traversal_camera_json:{accepted_directory / 'lens_0000' / 'camera.json'}"
    )
    accepted = GeneratorConfig.model_validate(raw)
    _, accepted_metadata = randomize_camera_config(accepted, 123)
    assert accepted_metadata == original_metadata

    accepted.zoom.lenses[0].color.exposure_ev += 0.1
    _, different_color_metadata = randomize_camera_config(accepted, 123)
    assert different_color_metadata["config_sha256"] != original_metadata["config_sha256"]

    raw["camera"]["initialization"]["resolved_pose"]["position"][0] += 0.01
    raw["camera"]["initialization"]["resolved_pose"]["camera_to_world"][0][3] += 0.01
    different_pose = GeneratorConfig.model_validate(raw)
    _, different_pose_metadata = randomize_camera_config(different_pose, 123)
    assert different_pose_metadata["config_sha256"] != original_metadata["config_sha256"]


@pytest.mark.parametrize("curve", ["linear", "smoothstep"])
def test_single_lens_keeps_exact_non_binary_zoom_endpoints(cfg, curve):
    raw = cfg.model_dump(mode="python")
    raw["video"]["total_frames"] = 7
    raw["zoom"]["curve"] = curve
    raw["zoom"]["lenses"] = raw["zoom"]["lenses"][:1]
    raw["zoom"]["lenses"][0]["zoom_min"] = 0.03
    raw["zoom"]["lenses"][0]["zoom_max"] = 0.29
    single_lens = GeneratorConfig.model_validate(raw)

    for seed in range(5):
        randomized, _ = randomize_camera_config(single_lens, seed)
        schedule = build_schedule(randomized)
        assert schedule[0].zoom_ratio == 0.03
        assert schedule[-1].zoom_ratio == 0.29
        assert np.all(np.diff([frame.zoom_ratio for frame in schedule]) > 0.0)
