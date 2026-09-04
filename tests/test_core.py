from __future__ import annotations


from copy import deepcopy
import json

import cv2
import numpy as np
import torch
import pytest
import yaml
from plyfile import PlyData, PlyElement
from pydantic import ValidationError

from zoomgen.camera import CameraFrame, build_camera_frames, intrinsics, look_at
from zoomgen.color import linear_color_stage, process_frame
from zoomgen.config import GeneratorConfig, load_config_parts
from zoomgen.pipeline import coverage_metrics
from zoomgen.quality import screen_space_quality_metrics
from zoomgen.scene import GaussianScene, GaussianTensors
from zoomgen.zoom import allocate_frame_counts, build_schedule, temporal_state


def config_dict(tmp_path):
    ply = tmp_path / "dummy.ply"
    ply.touch()
    color = {
        "exposure_ev": 0.0,
        "white_balance_gains": [1.0, 1.0, 1.0],
        "color_correction_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "black_level": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "gamma": 2.2,
        "vignette_strength": 0.0,
        "noise_std": 0.0,
    }
    return {
        "input": {"ply_path": str(ply)},
        "output": {"directory": str(tmp_path / "out"), "overwrite": True},
        "render": {"device": "cuda", "background_rgb": [0, 0, 0], "sh_degree": "auto"},
        "video": {"width": 320, "height": 180, "fps": 10, "total_frames": 30},
        "scene_analysis": {
            "opacity_threshold": 0.02,
            "position_quantiles": [0.01, 0.99],
            "analysis_max_samples": 2000,
            "io_chunk_size": 100,
        },
        "camera": {
            "fov_y_deg_at_1x": 55,
            "initial_view": {"azimuth_deg": 0, "elevation_deg": 0},
            "auto_fit": {"color_transition_frames": 0} if False else {},
            "motion": {
                "enabled": True,
                "translation_amplitude_ratio": [0.003, 0.002, 0.001],
                "rotation_amplitude_deg": [0.2, 0.25, 0.15],
                "control_point_interval_seconds": 1.0,
                "seed": 3407,
            },
        },
        "zoom": {
            "curve": "linear",
            "frame_allocation": {"mode": "proportional_to_zoom_span"},
            "lens_switch": {
                "color_mode": "crossfade", "color_transition_frames": 3,
                "camera_offset_mode": "crossfade", "camera_transition_frames": 3,
            },
            "temporal_color_adjustment": {
                "max_exposure_jump_ev": 1.0, "transition_frames": 0,
                "couple_noise_to_iso": True, "noise_growth_per_ev": 0.5,
            },
            "lenses": [
                {
                    "name": "uw", "zoom_min": 0.5, "zoom_max": 0.95, "frame_count": None,
                    "camera_center_offset_ratio": [-0.002, 0, 0],
                    "temporal_color_jump": {"strength": 0.15, "position": 0.35},
                    "color": deepcopy(color),
                },
                {
                    "name": "w", "zoom_min": 1.0, "zoom_max": 2.95, "frame_count": None,
                    "camera_center_offset_ratio": [0, 0, 0],
                    "temporal_color_jump": {"strength": -0.1, "position": 0.55},
                    "color": deepcopy(color),
                },
                {
                    "name": "l", "zoom_min": 3.0, "zoom_max": 5.0, "frame_count": None,
                    "camera_center_offset_ratio": [0.002, 0, 0],
                    "temporal_color_jump": {"strength": 0.2, "position": 0.7},
                    "color": deepcopy(color),
                },
            ],
        },
    }


@pytest.fixture
def cfg(tmp_path):
    return GeneratorConfig.model_validate(config_dict(tmp_path))


def test_proportional_schedule_endpoints_monotonic_and_count(cfg):
    counts = allocate_frame_counts(cfg)
    assert counts == [3, 13, 14]
    schedule = build_schedule(cfg)
    assert len(schedule) == 30
    assert [(schedule[sum(counts[:i])].zoom_ratio,
             schedule[sum(counts[:i + 1]) - 1].zoom_ratio) for i in range(3)] == [
        (0.5, 0.95), (1.0, 2.95), (3.0, 5.0)
    ]
    assert np.all(np.diff([x.zoom_ratio for x in schedule]) > 0)


def test_explicit_allocation(cfg):
    raw = cfg.model_dump(mode="python")
    raw["zoom"]["frame_allocation"]["mode"] = "explicit"
    for lens, count in zip(raw["zoom"]["lenses"], [5, 10, 15]):
        lens["frame_count"] = count
    explicit = GeneratorConfig.model_validate(raw)
    assert allocate_frame_counts(explicit) == [5, 10, 15]


def test_bad_explicit_sum_rejected(cfg):
    raw = cfg.model_dump(mode="python")
    raw["zoom"]["frame_allocation"]["mode"] = "explicit"
    for lens, count in zip(raw["zoom"]["lenses"], [5, 10, 14]):
        lens["frame_count"] = count
    with pytest.raises(ValidationError, match="sum"):
        GeneratorConfig.model_validate(raw)


def test_intrinsics_scale_exactly_with_zoom(cfg):
    a = intrinsics(cfg, 1.0)
    b = intrinsics(cfg, 5.0)
    assert b[0] == pytest.approx(a[0] * 5)
    assert b[1] == pytest.approx(a[1] * 5)


@pytest.mark.parametrize("position,index", [(0.0, 0), (1.0, 9), (0.5, 4)])
def test_jump_position_mapping(cfg, position, index):
    lens = cfg.zoom.lenses[0].model_copy(
        update={"temporal_color_jump": cfg.zoom.lenses[0].temporal_color_jump.model_copy(
            update={"position": position}
        )}
    )
    state = temporal_state(lens, index, 10, cfg)
    assert state["jump_local_index"] == index


@pytest.mark.parametrize("strength,gain", [(1.0, 2.0), (-1.0, 0.5)])
def test_jump_ev_gain_extremes(cfg, strength, gain):
    lens = cfg.zoom.lenses[0].model_copy(
        update={"temporal_color_jump": cfg.zoom.lenses[0].temporal_color_jump.model_copy(
            update={"strength": strength, "position": 0.0}
        )}
    )
    state = temporal_state(lens, 0, 10, cfg)
    assert state["jump_ev"] == pytest.approx(strength)
    assert state["gain"] == pytest.approx(gain)


def test_zero_strength_identical(cfg):
    lens = cfg.zoom.lenses[0].model_copy(
        update={"temporal_color_jump": cfg.zoom.lenses[0].temporal_color_jump.model_copy(
            update={"strength": 0.0}
        )}
    )
    before = temporal_state(lens, 0, 10, cfg)
    after = temporal_state(lens, 9, 10, cfg)
    assert before["gain"] == after["gain"] == 1.0
    assert not before["applied"] and not after["applied"]


def test_hard_jump_exact_frame(cfg):
    lens = cfg.zoom.lenses[0]
    jump = round(lens.temporal_color_jump.position * 9)
    assert not temporal_state(lens, jump - 1, 10, cfg)["applied"]
    assert temporal_state(lens, jump, 10, cfg)["progress"] == 1.0


def test_smooth_jump_finishes_in_requested_frames(cfg):
    raw = cfg.model_dump(mode="python")
    raw["zoom"]["frame_allocation"]["mode"] = "explicit"
    for lens in raw["zoom"]["lenses"]:
        lens["frame_count"] = 10
    raw["zoom"]["temporal_color_adjustment"]["transition_frames"] = 3
    smooth = GeneratorConfig.model_validate(raw)
    lens = smooth.zoom.lenses[0]
    jump = round(lens.temporal_color_jump.position * 9)
    assert temporal_state(lens, jump, 10, smooth)["progress"] == pytest.approx(1 / 3)
    assert temporal_state(lens, jump + 2, 10, smooth)["progress"] == 1.0


def test_positive_iso_coupling_increases_noise(cfg):
    lens = cfg.zoom.lenses[0].model_copy(
        update={
            "color": cfg.zoom.lenses[0].color.model_copy(update={"noise_std": 0.01}),
            "temporal_color_jump": cfg.zoom.lenses[0].temporal_color_jump.model_copy(
                update={"strength": 1.0, "position": 0.0}
            ),
        }
    )
    assert temporal_state(lens, 0, 10, cfg)["noise_std"] > 0.01


def test_crossfade_endpoints(cfg):
    schedule = build_schedule(cfg)
    first_w = next(x for x in schedule if x.lens_name == "w")
    w = [x for x in schedule if x.lens_name == "w"]
    assert first_w.color_blend == 0.0
    assert w[2].color_blend == 1.0
    assert first_w.camera_blend == 0.0


def test_color_configs_produce_different_results(cfg):
    image = np.full((4, 4, 3), 0.3, np.float32)
    a = cfg.zoom.lenses[0].color
    b = a.model_copy(update={"exposure_ev": 1.0, "saturation": 0.5})
    assert not np.array_equal(linear_color_stage(image, a, 1.0),
                              linear_color_stage(image, b, 1.0))


def test_noise_and_trajectory_reproducible(cfg):
    schedule = build_schedule(cfg)
    p, t, radius = np.array([2.0, 0, 0]), np.zeros(3), 1.0
    a = build_camera_frames(cfg, schedule, p, t, radius)
    b = build_camera_frames(cfg, schedule, p, t, radius)
    np.testing.assert_allclose([x.position for x in a], [x.position for x in b])
    image = np.full((8, 8, 3), 0.3, np.float32)
    lens = cfg.zoom.lenses[0]
    cfg.zoom.lenses[0].color.noise_std = 0.01
    out1 = process_frame(image, schedule[0], cfg)[0]
    out2 = process_frame(image, schedule[0], cfg)[0]
    np.testing.assert_array_equal(out1, out2)


def test_camera_trajectory_is_continuous(cfg):
    frames = build_camera_frames(
        cfg, build_schedule(cfg), np.array([2.0, 0, 0]), np.zeros(3), 1.0
    )
    steps = np.linalg.norm(np.diff([x.position for x in frames], axis=0), axis=1)
    assert steps.max() < 0.02


def test_coverage_and_largest_component():
    alpha = np.ones((10, 10), np.float32)
    alpha[:2] = 0
    coverage, largest = coverage_metrics(alpha, 0.02)
    assert coverage == 0.8
    assert largest == 0.2


def test_invalid_jump_range_rejected(tmp_path):
    raw = config_dict(tmp_path)
    raw["zoom"]["lenses"][0]["temporal_color_jump"]["strength"] = 1.1
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(raw)


def _write_tiny_3dgs(path, include_rest=True):
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
             "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    if include_rest:
        names += [f"f_rest_{i}" for i in range(9)]
    dtype = [(x, "f4") for x in names]
    v = np.zeros(7, dtype=dtype)
    v["x"] = [-1, -0.5, 0, 0.5, 1, 1000, 0]
    v["y"] = [0, 0.1, 0, -0.1, 0, 1000, 0]
    v["z"] = [0, 0, 0.1, 0, 0, 1000, 0]
    for x in ("scale_0", "scale_1", "scale_2"):
        v[x] = -3
    v["rot_0"] = 1
    v["opacity"] = [4, 4, 4, 4, 4, -10, np.nan]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)


def test_robust_scene_analysis_rejects_outlier_and_nonfinite(tmp_path):
    raw = config_dict(tmp_path)
    ply = tmp_path / "tiny.ply"
    _write_tiny_3dgs(ply)
    raw["input"]["ply_path"] = str(ply)
    raw["scene_analysis"]["position_quantiles"] = [0.01, 0.99]
    scene = GaussianScene(ply, GeneratorConfig.model_validate(raw))
    assert scene.analysis.nonfinite_count == 1
    assert scene.analysis.low_opacity_count == 1
    assert scene.analysis.radius < 10
    assert scene.analysis.effective_gaussian_count <= 5

def test_dc_only_3dgs_falls_back_to_sh_degree_zero(tmp_path):
    raw = config_dict(tmp_path)
    ply = tmp_path / "tiny_dc_only.ply"
    _write_tiny_3dgs(ply, include_rest=False)
    raw["input"]["ply_path"] = str(ply)
    cfg = GeneratorConfig.model_validate(raw)

    with pytest.warns(RuntimeWarning, match="falling back to DC-only"):
        scene = GaussianScene(ply, cfg)

    assert scene.rest_names == []
    assert scene.available_sh_degree == 0
    assert scene.analysis.sh_degree == 0

    # Exercise the tensor-loading path without requiring a CUDA device. The
    # production renderer receives the same degree-0 tensor on CUDA.
    cfg.render.device = "cpu"
    tensors = scene.load_tensors()
    assert tensors.sh_degree == 0
    assert tensors.shs.shape == (scene.analysis.effective_gaussian_count, 1, 3)


def test_crossfade_keeps_incoming_lens_temporal_jump(cfg):
    raw = cfg.model_dump(mode="python")
    raw["zoom"]["lenses"][1]["temporal_color_jump"] = {"strength": 1.0, "position": 0.0}
    jumped = GeneratorConfig.model_validate(raw)
    raw["zoom"]["lenses"][1]["temporal_color_jump"]["strength"] = 0.0
    unjumped = GeneratorConfig.model_validate(raw)
    frame = [x for x in build_schedule(jumped) if x.lens_name == "w"][1]
    image = np.full((8, 8, 3), 0.25, np.float32)
    with_jump = process_frame(image, frame, jumped)[0]
    without_jump = process_frame(image, frame, unjumped)[0]
    assert frame.color_blend > 0
    assert not np.array_equal(with_jump, without_jump)


def test_plain_point_cloud_is_rejected(tmp_path):
    path = tmp_path / "plain.ply"
    vertices = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(path)
    raw = config_dict(tmp_path)
    raw["input"]["ply_path"] = str(path)
    with pytest.raises(ValueError, match="not a valid 3D Gaussian"):
        GaussianScene(path, GeneratorConfig.model_validate(raw))

def test_manual_pose_rotation_representations_conflict(tmp_path):
    raw = config_dict(tmp_path)
    raw["camera"]["initialization"] = {
        "scene_type": "manual",
        "mode": "manual",
        "manual_pose": {
            "position": [0, 0, 0],
            "look_at": [0, 0, -1],
            "yaw_deg": 0,
            "pitch_deg": 0,
            "fov_y_deg": 55,
        },
    }
    with pytest.raises(ValidationError, match="cannot both"):
        GeneratorConfig.model_validate(raw)


def test_manual_pose_is_preserved_and_matrix_is_right_handed(tmp_path):
    from zoomgen.alignment import initialize_manual

    raw = config_dict(tmp_path)
    ply = tmp_path / "tiny_manual.ply"
    _write_tiny_3dgs(ply)
    raw["input"]["ply_path"] = str(ply)
    raw["camera"]["initialization"] = {
        "scene_type": "manual",
        "mode": "manual",
        "manual_pose": {
            "position": [0.1, 0.2, 0.3],
            "look_at": [0.1, 0.2, -1.0],
            "up": [0, 1, 0],
            "fov_y_deg": 61,
        },
    }
    cfg = GeneratorConfig.model_validate(raw)
    base = initialize_manual(cfg, GaussianScene(ply, cfg))
    np.testing.assert_array_equal(base.position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(base.c2w[:3, 2], [0, 0, -1])
    assert np.linalg.det(base.c2w[:3, :3]) == pytest.approx(1.0)
    assert base.fov_y_deg_at_1x == 61


def test_motion_disabled_has_bitwise_identical_extrinsics(cfg):
    cfg.camera.motion.enabled = False
    frames = build_camera_frames(
        cfg, build_schedule(cfg), np.array([2.0, 0.0, 0.0]), np.zeros(3), 1.0
    )
    first = frames[0].c2w
    assert all(np.array_equal(first, frame.c2w) for frame in frames)
    assert all(np.array_equal(frames[0].position, frame.position) for frame in frames)


def test_root_transform_applies_to_analysis_samples(tmp_path):
    from scipy.spatial.transform import Rotation

    ply = tmp_path / "tiny_root.ply"
    _write_tiny_3dgs(ply)
    raw = config_dict(tmp_path)
    raw["input"]["ply_path"] = str(ply)
    identity_cfg = GeneratorConfig.model_validate(raw)
    source = GaussianScene(ply, identity_cfg).effective_position_sample()
    raw["scene"] = {
        "root_transform": {
            "translation": [10, 20, 30],
            "rotation_euler_deg": [0, 0, 90],
            "rotation_order": "xyz",
            "scale": 2,
        }
    }
    transformed_cfg = GeneratorConfig.model_validate(raw)
    transformed = GaussianScene(ply, transformed_cfg).effective_position_sample()
    expected = (
        2 * (Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix() @ source.T).T
        + np.array([10, 20, 30])
    )
    np.testing.assert_allclose(
        np.sort(transformed, axis=0), np.sort(expected, axis=0), atol=1e-5
    )


def test_reference_preprocessing_preserves_aspect_by_crop(tmp_path, cfg):
    from zoomgen.alignment import preprocess_reference

    path = tmp_path / "reference.png"
    image = np.full((100, 100, 3), 127, np.uint8)
    image[:5] = 255
    assert cv2.imwrite(str(path), image)
    processed, meta = preprocess_reference(path, None, 160, 90)
    assert processed.shape == (90, 160, 3)
    assert meta["aspect_operation"]["mode"] == "center_crop_height"
    assert meta["auto_border_removed"]


def test_structural_loss_identical_image_is_minimal(cfg):
    from zoomgen.alignment import _structural_features, structural_loss

    rng = np.random.default_rng(4)
    image = rng.random((32, 48, 3), dtype=np.float32)
    same, _ = structural_loss(image, _structural_features(image), cfg)
    shifted, _ = structural_loss(
        np.roll(image, 5, axis=1), _structural_features(image), cfg
    )
    assert same < 1e-5
    assert shifted > same


def test_object_base_view_remains_outside():
    from zoomgen.camera import base_view

    class C:
        class camera:
            class initial_view:
                position = None
                look_at = None
                azimuth_deg = 0
                elevation_deg = 0

    position, target, distance = base_view(C(), np.zeros(3), 2.0)
    assert distance == 4.0
    assert np.linalg.norm(position - target) == 4.0


def test_seed_orientation_is_always_evaluated(cfg):
    from zoomgen.alignment import _candidate_orientations

    cfg.camera.initialization.interior.directions_per_position = 24
    seed = cfg.camera.initialization.reference_match.seed_pose
    seed.yaw_deg = 147.82784118453603
    seed.pitch_deg = -13.228101298338052
    candidates = _candidate_orientations(cfg, np.zeros(3), 1.0)
    assert any(
        candidate.yaw_deg == seed.yaw_deg
        and candidate.pitch_deg == seed.pitch_deg
        for candidate in candidates
    )


def test_reference_similarity_threshold_is_bounded(tmp_path):
    raw = config_dict(tmp_path)
    raw["camera"]["initialization"] = {
        "reference_match": {"minimum_similarity_score": 1.01}
    }
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        GeneratorConfig.model_validate(raw)


def _write_zoom_config_parts(tmp_path, cfg):
    raw = cfg.model_dump(mode="json")
    render = dict(raw["render"])
    background_rgb = render.pop("background_rgb")

    camera_lens_keys = {
        "name",
        "zoom_min",
        "zoom_max",
        "frame_count",
        "camera_center_offset_ratio",
    }
    color_lens_keys = {"name", "temporal_color_jump", "color"}
    scene = {
        key: raw[key]
        for key in ("input", "scene", "output", "video", "scene_analysis")
    }
    scene["render"] = render
    camera = {
        "camera": raw["camera"],
        "zoom": {
            "curve": raw["zoom"]["curve"],
            "frame_allocation": raw["zoom"]["frame_allocation"],
            "lens_switch": {
                key: raw["zoom"]["lens_switch"][key]
                for key in ("camera_offset_mode", "camera_transition_frames")
            },
            "lenses": [
                {key: value for key, value in lens.items() if key in camera_lens_keys}
                for lens in raw["zoom"]["lenses"]
            ],
        },
    }
    color = {
        "color": {
            "background_rgb": background_rgb,
            "lens_switch": {
                key: raw["zoom"]["lens_switch"][key]
                for key in ("color_mode", "color_transition_frames")
            },
            "temporal_color_adjustment": raw["zoom"][
                "temporal_color_adjustment"
            ],
            "lenses": [
                {key: value for key, value in lens.items() if key in color_lens_keys}
                for lens in raw["zoom"]["lenses"]
            ],
        }
    }

    paths = [tmp_path / name for name in ("scene.yaml", "camera.yaml", "color.yaml")]
    for path, fragment in zip(paths, (scene, camera, color)):
        path.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")
    return paths, color


def test_zoom_split_configs_recompose_same_model(tmp_path):
    combined = GeneratorConfig.model_validate(config_dict(tmp_path))
    paths, _ = _write_zoom_config_parts(tmp_path, combined)

    split = load_config_parts(*paths, check_output=False)

    assert split.model_dump(mode="json") == combined.model_dump(mode="json")


def test_zoom_split_rejects_mismatched_lens_names(tmp_path):
    combined = GeneratorConfig.model_validate(config_dict(tmp_path))
    paths, color = _write_zoom_config_parts(tmp_path, combined)
    color["color"]["lenses"][0]["name"] = "different"
    paths[2].write_text(yaml.safe_dump(color, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="lens names do not match"):
        load_config_parts(*paths, check_output=False)



def _write_traversal_camera_json(path, cfg):
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    raw = {
        "scene_name": "test",
        "camera": {
            "position": [1.0, 2.0, 3.0],
            "camera_to_world": c2w.tolist(),
        },
        "intrinsics": {
            "fixed_across_traversal": True,
            "width": 100,
            "height": 80,
            "fx": 40.0,
            "fy": 40.0,
            "cx": 55.0,
            "cy": 36.0,
            "fov_y_deg": 61.0,
        },
        "scene_root_transform": cfg.scene.root_transform.model_dump(mode="json"),
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return c2w, raw


def test_camera_json_initializes_zoom_pose_and_intrinsics(tmp_path):
    combined = GeneratorConfig.model_validate(config_dict(tmp_path))
    paths, _ = _write_zoom_config_parts(tmp_path, combined)
    camera_json = tmp_path / "camera.json"
    expected_c2w, _ = _write_traversal_camera_json(camera_json, combined)

    loaded = load_config_parts(
        *paths, camera_json_path=camera_json, check_output=False
    )

    resolved = loaded.camera.initialization.resolved_pose
    assert resolved is not None
    np.testing.assert_allclose(resolved.camera_to_world, expected_c2w)
    np.testing.assert_allclose(resolved.position, expected_c2w[:3, 3])
    assert resolved.fov_y_deg_at_1x == pytest.approx(61.0)
    assert loaded.camera.fov_y_deg_at_1x == pytest.approx(61.0)
    np.testing.assert_allclose(loaded.camera.principal_point_offset_px, [16.0, -9.0])
    assert resolved.source == f"traversal_camera_json:{camera_json.resolve()}"
    assert loaded.camera.initialization.scene_type == "interior"


def test_camera_json_rejects_mismatched_scene_root_transform(tmp_path):
    combined = GeneratorConfig.model_validate(config_dict(tmp_path))
    paths, _ = _write_zoom_config_parts(tmp_path, combined)
    camera_json = tmp_path / "camera.json"
    _, raw = _write_traversal_camera_json(camera_json, combined)
    raw["scene_root_transform"]["rotation_euler_deg"] = [180.0, 0.0, 0.0]
    camera_json.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="root_transform does not match"):
        load_config_parts(
            *paths, camera_json_path=camera_json, check_output=False
        )


def test_screen_space_quality_metrics_detect_large_and_overlapping_gaussians():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.05]], dtype=torch.float32
    )
    gaussians = GaussianTensors(
        xyz=xyz,
        scales=torch.tensor(
            [[0.8, 0.35, 0.35], [0.6, 0.3, 0.3]], dtype=torch.float32
        ),
        rotations=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        opacity=torch.ones((2, 1), dtype=torch.float32),
        shs=torch.zeros((2, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    position = np.array([0.0, 0.0, -3.0])
    target = np.zeros(3)
    c2w = look_at(position, target, world_up=np.array([0.0, -1.0, 0.0]))
    fov = np.deg2rad(60.0)
    focal = 60.0 / np.tan(fov / 2.0)
    camera = CameraFrame(
        position,
        target,
        c2w,
        np.linalg.inv(c2w),
        focal,
        focal,
        60.0,
        60.0,
        fov,
        fov,
        0.01,
        20.0,
        np.zeros(3),
    )

    metrics = screen_space_quality_metrics(
        gaussians,
        camera,
        torch.tensor([50, 40], dtype=torch.int32),
        120,
        120,
    )

    assert metrics["visible_gaussian_count"] == 2
    assert metrics["projection_candidate_count"] == 2
    assert metrics["max_projected_area_px2"] > 0
    assert metrics["max_projected_area_ratio"] > 0
    assert metrics["max_projected_major_axis_to_image_diagonal_ratio"] > 0
    assert metrics["largest_single_gaussian_dominated_pixel_ratio"] > 0
    assert (
        metrics["top5_gaussians_dominated_pixel_ratio"]
        >= metrics["largest_single_gaussian_dominated_pixel_ratio"]
    )
    assert metrics["multi_layer_overlap_pixel_ratio"] > 0
    assert metrics["max_overlap_layer_count"] == 2
    assert len(metrics["top_dominant_gaussians"]) == 2
