from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import scene_traversal as traversal_module
from scene_traversal import (
    CapturePlan,
    PositionPlan,
    TraversalConfig,
    _camera_frame,
    _capture_metadata,
    load_traversal_config_parts,
    capture_paths,
    plan_traversal,
    run_traversal,
)


def traversal_config(tmp_path) -> TraversalConfig:
    ply = tmp_path / "scene.ply"
    ply.touch()
    raw = {
        "input": {"ply_path": str(ply)},
        "scene": {"scene_type": "interior"},
        "output": {
            "root_directory": str(tmp_path / "captures"),
            "scene_name": "room",
        },
        "initialization": {
            "world_up": [0, 1, 0],
            "reference_match": {
                "preview_width": 120,
                "preview_height": 120,
                "seed_pose": {
                    "position": [0, 0, 0],
                    "yaw_deg": 30,
                    "pitch_deg": -10,
                    "roll_deg": 1,
                    "fov_y_deg": 70,
                },
                "search_radius_ratio": 0.15,
                "yaw_search_range_deg": [-90.0, 90.0],
                "pitch_search_range_deg": [-12.5, 12.5],
                "roll_search_range_deg": [-2.5, 2.5],
            },
            "traversal": {
                "position_count_k": 3,
                "images_per_position_l": 5,
                "random_seed": 123,
                "max_position_sampling_attempts": 1000,
                "minimum_clearance_radius_ratio": 0.005,
                "maximum_clearance_radius_ratio": 0.20,
                "minimum_position_separation_radius_ratio": 0.02,
            },
        },
    }
    return TraversalConfig.model_validate(raw)


class FakeInteriorScene:
    def __init__(self):
        axis = np.linspace(-5.0, 5.0, 11)
        points = np.array(np.meshgrid(axis, axis, axis)).T.reshape(-1, 3)
        self._points = points
        self.analysis = SimpleNamespace(
            aabb_min=np.array([-5.0, -5.0, -5.0]),
            aabb_max=np.array([5.0, 5.0, 5.0]),
            center=np.zeros(3),
            radius=10.0,
        )

    def effective_position_sample(self):
        return self._points


def test_traversal_plan_is_k_times_l_and_reproducible(tmp_path):
    cfg = traversal_config(tmp_path)
    scene = FakeInteriorScene()
    positions_a, captures_a = plan_traversal(cfg, scene)
    positions_b, captures_b = plan_traversal(cfg, scene)

    assert len(positions_a) == 3
    assert len(captures_a) == 15
    np.testing.assert_allclose(
        np.stack([item.position for item in positions_a]),
        np.stack([item.position for item in positions_b]),
    )
    np.testing.assert_allclose(
        [item.yaw_deg for item in captures_a],
        [item.yaw_deg for item in captures_b],
    )

    match = cfg.initialization.reference_match
    seed = match.seed_pose
    yaw_min, yaw_max = match.yaw_search_range_deg
    pitch_min, pitch_max = match.pitch_search_range_deg
    roll_min, roll_max = match.roll_search_range_deg
    for capture in captures_a:
        assert seed.yaw_deg + yaw_min <= capture.yaw_deg <= seed.yaw_deg + yaw_max
        assert seed.pitch_deg + pitch_min <= capture.pitch_deg <= seed.pitch_deg + pitch_max
        assert seed.roll_deg + roll_min <= capture.roll_deg <= seed.roll_deg + roll_max
        assert capture.fov_y_deg == seed.fov_y_deg
    cameras = [_camera_frame(cfg, scene, capture) for capture in captures_a]
    first = cameras[0]
    assert all(camera.fx == first.fx for camera in cameras)
    assert all(camera.fy == first.fy for camera in cameras)
    assert all(camera.cx == first.cx for camera in cameras)
    assert all(camera.cy == first.cy for camera in cameras)
    assert all(camera.fov_x == first.fov_x for camera in cameras)
    assert all(camera.fov_y == first.fov_y for camera in cameras)
    assert any(
        not np.array_equal(camera.c2w, first.c2w) for camera in cameras[1:]
    )


def test_capture_path_has_scene_position_lens_hierarchy(tmp_path):
    cfg = traversal_config(tmp_path)
    image, metadata = capture_paths(cfg, "position_0002", "lens_0004")
    expected = tmp_path / "captures" / "room" / "position_0002" / "lens_0004"
    assert image == expected / "image.png"
    assert metadata == expected / "camera.json"


def test_unknown_scene_type_is_rejected(tmp_path):
    cfg = traversal_config(tmp_path).model_dump(mode="python")
    cfg["scene"]["scene_type"] = "outdoor"
    with pytest.raises(ValidationError):
        TraversalConfig.model_validate(cfg)


def test_legacy_initialization_scene_type_is_migrated(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["initialization"]["scene_type"] = raw["scene"].pop("scene_type")

    cfg = TraversalConfig.model_validate(raw)

    assert cfg.scene.scene_type == "interior"
    assert not hasattr(cfg.initialization, "scene_type")


def test_conflicting_legacy_scene_type_is_rejected(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["initialization"]["scene_type"] = "object"

    with pytest.raises(ValidationError, match="conflicts"):
        TraversalConfig.model_validate(raw)


def test_scalar_angle_ranges_are_migrated(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["initialization"]["reference_match"].update(
        {
            "yaw_search_range_deg": 360.0,
            "pitch_search_range_deg": 35.0,
            "roll_search_range_deg": 15.0,
        }
    )

    cfg = TraversalConfig.model_validate(raw)
    match = cfg.initialization.reference_match

    assert match.yaw_search_range_deg == (-180.0, 180.0)
    assert match.pitch_search_range_deg == (-17.5, 17.5)
    assert match.roll_search_range_deg == (-7.5, 7.5)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("yaw_search_range_deg", [-181.0, 0.0]),
        ("pitch_search_range_deg", [0.0, 181.0]),
        ("roll_search_range_deg", [20.0, -20.0]),
    ],
)
def test_angle_ranges_must_be_ordered_and_bounded(tmp_path, field_name, value):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["initialization"]["reference_match"][field_name] = value

    with pytest.raises(ValidationError, match=field_name):
        TraversalConfig.model_validate(raw)


def test_random_quality_strategy_metadata_and_validation(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    sampling = raw["initialization"]["traversal"]
    sampling["strategy"] = "random"
    sampling.pop("position_count_k")
    sampling["max_position_sampling_attempts"] = 20
    sampling["minimum_position_separation_radius_ratio"] = 0.0
    sampling["random"] = {
        "target_count_k": 2,
        "topiq_nr_threshold_l": 0.6,
        "device": "cuda",
    }
    cfg = TraversalConfig.model_validate(raw)
    scene = FakeInteriorScene()
    positions, captures = plan_traversal(cfg, scene)
    assert len(positions) > 10
    assert len(captures) == len(positions) * sampling["images_per_position_l"]
    capture = captures[0]
    camera = _camera_frame(cfg, scene, capture)
    image_path, _ = capture_paths(cfg, capture.position.label, capture.lens_label)

    metadata = _capture_metadata(
        cfg,
        scene,
        capture,
        camera,
        image_path,
        geometry_quality={},
        topiq_nr_score=0.7,
    )

    assert metadata["sampling"]["strategy"] == "random"
    assert metadata["image_quality"] == {
        "metric": "topiq_nr",
        "score": 0.7,
        "threshold_l": 0.6,
        "accepted": True,
    }




def test_object_traversal_is_external_center_facing_and_reproducible(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    initialization = raw["initialization"]
    raw["scene"]["scene_type"] = "object"
    initialization["reference_match"]["seed_pose"]["position"] = None
    initialization["reference_match"]["seed_pose"]["yaw_deg"] = 0.0
    initialization["reference_match"]["seed_pose"]["pitch_deg"] = 0.0
    initialization["reference_match"]["yaw_search_range_deg"] = [-10.0, 10.0]
    initialization["reference_match"]["pitch_search_range_deg"] = [-5.0, 5.0]
    initialization["traversal"]["position_count_k"] = 8
    initialization["traversal"]["images_per_position_l"] = 4
    initialization["traversal"]["minimum_position_separation_radius_ratio"] = 0.2
    initialization["object"] = {
        "distance_range_radius_ratio": [1.5, 2.0],
        "azimuth_range_deg": [-180.0, 180.0],
        "elevation_range_deg": [0.0, 90.0],
    }
    cfg_a = TraversalConfig.model_validate(raw)
    cfg_b = TraversalConfig.model_validate(raw)
    scene = FakeInteriorScene()

    positions_a, captures_a = plan_traversal(cfg_a, scene)
    positions_b, captures_b = plan_traversal(cfg_b, scene)

    assert len(positions_a) == 8
    assert len(captures_a) == 32
    np.testing.assert_allclose(
        np.stack([item.position for item in positions_a]),
        np.stack([item.position for item in positions_b]),
    )
    distances = np.linalg.norm(
        np.stack([item.position for item in positions_a]) - scene.analysis.center,
        axis=1,
    )
    assert np.all(distances >= 1.5 * scene.analysis.radius)
    assert np.all(distances <= 2.0 * scene.analysis.radius)
    heights = np.stack([item.position for item in positions_a])[:, 1]
    assert np.all(heights >= scene.analysis.center[1])
    assert all(
        not np.all(
            (item.position >= scene.analysis.aabb_min)
            & (item.position <= scene.analysis.aabb_max)
        )
        for item in positions_a
    )
    assert positions_a[0].seed_position_source == "automatic_object_shell"

    cameras = [_camera_frame(cfg_a, scene, capture) for capture in captures_a]
    first = cameras[0]
    assert all(camera.fx == first.fx for camera in cameras)
    assert all(camera.fy == first.fy for camera in cameras)
    assert all(camera.cx == first.cx for camera in cameras)
    assert all(camera.cy == first.cy for camera in cameras)
    for capture, camera in zip(captures_a, cameras):
        desired = scene.analysis.center - capture.position.position
        desired /= np.linalg.norm(desired)
        assert float(np.dot(camera.c2w[:3, 2], desired)) > math.cos(math.radians(15.0))

    image_path, _ = capture_paths(
        cfg_a, captures_a[0].position.label, captures_a[0].lens_label
    )
    geometry_quality = {
        "max_projected_area_px2": 123.0,
        "top5_gaussians_dominated_pixel_ratio": 0.2,
    }
    metadata = _capture_metadata(
        cfg_a,
        scene,
        captures_a[0],
        cameras[0],
        image_path,
        geometry_quality,
    )
    assert metadata["scene_type"] == "object"
    assert metadata["sampling"]["strategy"] == "object_shell"
    assert metadata["sampling"]["camera_inside_robust_aabb"] is False
    assert metadata["geometry_quality"] == geometry_quality


def test_object_traversal_rejects_an_interior_seed(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["scene"]["scene_type"] = "object"
    raw["initialization"]["reference_match"]["seed_pose"]["position"] = [0, 0, 0]
    cfg = TraversalConfig.model_validate(raw)
    with pytest.raises(ValueError, match="upper-hemisphere shell"):
        plan_traversal(cfg, FakeInteriorScene())


def test_object_traversal_rejects_a_lower_hemisphere_seed(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["scene"]["scene_type"] = "object"
    raw["initialization"]["reference_match"]["seed_pose"]["position"] = [
        0,
        -15,
        0,
    ]
    cfg = TraversalConfig.model_validate(raw)

    with pytest.raises(ValueError, match="upper-hemisphere shell"):
        plan_traversal(cfg, FakeInteriorScene())


def test_object_traversal_rejects_lower_hemisphere_elevation_range(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["scene"]["scene_type"] = "object"
    raw["initialization"]["object"]["elevation_range_deg"] = [-1.0, 45.0]

    with pytest.raises(ValidationError, match="upper hemisphere"):
        TraversalConfig.model_validate(raw)


def test_unsafe_output_label_is_rejected(tmp_path):
    cfg = traversal_config(tmp_path).model_dump(mode="python")
    cfg["output"]["position_label_format"] = "../position_{index}"
    with pytest.raises(ValidationError, match="safe path component"):
        TraversalConfig.model_validate(cfg)


def _write_yaml(path, value):
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_split_configs_recompose_the_same_model(tmp_path):
    cfg = traversal_config(tmp_path)
    raw = cfg.model_dump(mode="json")
    scene_path = tmp_path / "scene.yaml"
    camera_path = tmp_path / "camera.yaml"
    color_path = tmp_path / "color.yaml"
    _write_yaml(
        scene_path,
        {
            key: raw[key]
            for key in ("input", "scene", "output", "scene_analysis")
        },
    )
    _write_yaml(camera_path, {"initialization": raw["initialization"]})
    _write_yaml(
        color_path,
        {key: raw[key] for key in ("render", "image")},
    )

    recomposed = load_traversal_config_parts(
        scene_path, camera_path, color_path, check_output=False
    )
    assert recomposed.model_dump(mode="json") == raw


def test_split_config_rejects_cross_owned_keys(tmp_path):
    cfg = traversal_config(tmp_path)
    raw = cfg.model_dump(mode="json")
    scene_path = tmp_path / "scene.yaml"
    camera_path = tmp_path / "camera.yaml"
    color_path = tmp_path / "color.yaml"
    _write_yaml(
        scene_path,
        {
            "input": raw["input"],
            "output": raw["output"],
            "initialization": raw["initialization"],
        },
    )
    _write_yaml(camera_path, {"initialization": raw["initialization"]})
    _write_yaml(color_path, {"render": raw["render"], "image": raw["image"]})
    with pytest.raises(ValueError, match="owned by another config"):
        load_traversal_config_parts(
            scene_path, camera_path, color_path, check_output=False
        )



def test_null_seed_pose_auto_selects_reproducible_seed(tmp_path):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    raw["initialization"]["reference_match"]["seed_pose"] = None
    cfg_a = TraversalConfig.model_validate(raw)
    cfg_b = TraversalConfig.model_validate(raw)
    scene = FakeInteriorScene()

    positions_a, captures_a = plan_traversal(cfg_a, scene)
    positions_b, captures_b = plan_traversal(cfg_b, scene)

    seed_a = np.asarray(cfg_a.initialization.reference_match.seed_pose.position)
    seed_b = np.asarray(cfg_b.initialization.reference_match.seed_pose.position)
    np.testing.assert_allclose(seed_a, seed_b)
    assert np.all(seed_a >= scene.analysis.aabb_min)
    assert np.all(seed_a <= scene.analysis.aabb_max)
    assert positions_a[0].seed_position_source == "automatic_clearance_density"
    assert positions_b[0].seed_position_source == "automatic_clearance_density"
    np.testing.assert_allclose(
        np.stack([item.position for item in positions_a]),
        np.stack([item.position for item in positions_b]),
    )
    np.testing.assert_allclose(
        [item.yaw_deg for item in captures_a],
        [item.yaw_deg for item in captures_b],
    )
    seed_pose = cfg_a.initialization.reference_match.seed_pose
    assert seed_pose.yaw_deg == 0.0
    assert seed_pose.pitch_deg == 0.0
    assert seed_pose.roll_deg == 0.0
    assert seed_pose.fov_y_deg == 96.0

    capture = captures_a[0]
    camera = _camera_frame(cfg_a, scene, capture)
    image_path, _ = capture_paths(
        cfg_a, capture.position.label, capture.lens_label
    )
    metadata = _capture_metadata(cfg_a, scene, capture, camera, image_path)
    assert metadata["sampling"]["seed_position_source"] == "automatic_clearance_density"
    assert "seed_position_source" not in metadata["intrinsics"]


def test_explicit_seed_outside_robust_aabb_is_still_rejected(tmp_path):
    cfg = traversal_config(tmp_path)
    cfg.initialization.reference_match.seed_pose.position = [20.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="outside the transformed robust scene AABB"):
        plan_traversal(cfg, FakeInteriorScene())


def test_random_quality_run_rejects_equal_threshold_and_stops_at_k(
    tmp_path, monkeypatch
):
    raw = traversal_config(tmp_path).model_dump(mode="python")
    sampling = raw["initialization"]["traversal"]
    sampling["strategy"] = "random"
    sampling["random"] = {
        "target_count_k": 2,
        "topiq_nr_threshold_l": 0.6,
        "device": "cuda",
    }
    cfg = TraversalConfig.model_validate(raw)
    position = PositionPlan(
        0,
        "position_0000",
        np.zeros(3),
        np.zeros(3),
        1.0,
        "configured",
    )
    captures = [
        CapturePlan(position, index, f"lens_{index:04d}", 0.0, 0.0, 0.0, 70.0)
        for index in range(4)
    ]

    class FakeAnalysis:
        effective_gaussian_count = 1
        aabb_min = np.full(3, -1.0)
        aabb_max = np.full(3, 1.0)
        radius = 1.0

        def as_dict(self):
            return {"effective_gaussian_count": 1}

    class FakeScene:
        def __init__(self, *_args):
            self.analysis = FakeAnalysis()

        def load_tensors(self):
            return object()

    class FakeRenderer:
        def __init__(self, *_args):
            self.g = object()

        def render(self, *_args, **_kwargs):
            return np.zeros((2, 2, 3), dtype=np.float32), object()

    scores = iter([0.6, 0.7, 0.8, 0.9])
    saved_metadata = []
    written_json = {}
    monkeypatch.setattr(traversal_module, "GaussianScene", FakeScene)
    monkeypatch.setattr(traversal_module, "GaussianRenderer", FakeRenderer)
    monkeypatch.setattr(
        traversal_module, "plan_traversal", lambda *_args: ([position], captures)
    )
    monkeypatch.setattr(traversal_module, "_camera_frame", lambda *_args: object())
    monkeypatch.setattr(
        traversal_module, "screen_space_quality_metrics", lambda *_args: {}
    )
    monkeypatch.setattr(
        traversal_module, "_create_topiq_nr_metric", lambda _device: (object(), object())
    )
    monkeypatch.setattr(
        traversal_module,
        "_topiq_nr_score",
        lambda *_args: next(scores),
    )
    monkeypatch.setattr(
        traversal_module,
        "_capture_metadata",
        lambda *_args: {
            "sampling": {"strategy": "random"},
            "image_quality": {"score": _args[-1]},
        },
    )
    monkeypatch.setattr(
        traversal_module,
        "_save_capture_pair",
        lambda _image, _json, _rgb, metadata, _cfg: saved_metadata.append(metadata),
    )
    monkeypatch.setattr(
        traversal_module,
        "_write_json",
        lambda path, value: written_json.__setitem__(path.name, value),
    )

    summary = run_traversal(cfg)

    assert [item["image_quality"]["score"] for item in saved_metadata] == [0.7, 0.8]
    assert summary["random_quality_result"] == {
        "target_count_k": 2,
        "topiq_nr_threshold_l": 0.6,
        "acceptance_condition": "topiq_nr > topiq_nr_threshold_l",
        "attempted_count": 3,
        "accepted_count": 2,
        "rejected_count": 1,
        "completed": True,
    }
    assert written_json["traversal_summary.json"] == summary
