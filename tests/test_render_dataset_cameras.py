from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import render_dataset_cameras as render_module
from render_dataset_cameras import (
    camera_frame,
    capture_metadata,
    capture_paths,
    discover_scene_inputs,
    load_camera_entries,
    render_dataset,
)


def _camera_entry(
    image_name: str,
    camera_id: int,
    *,
    position=(0.0, 0.0, 0.0),
    width=8,
    height=6,
    fx=5.0,
    fy=4.0,
) -> dict:
    return {
        "id": camera_id,
        "img_name": image_name,
        "width": width,
        "height": height,
        "position": list(position),
        "rotation": np.eye(3).tolist(),
        "fx": fx,
        "fy": fy,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discovery_uses_root_cameras_and_highest_numeric_iteration(tmp_path):
    data = tmp_path / "scene"
    (data / "point_cloud" / "iteration_9").mkdir(parents=True)
    (data / "point_cloud" / "iteration_30").mkdir(parents=True)
    (data / "point_cloud" / "iteration_bad").mkdir(parents=True)
    (data / "point_cloud" / "iteration_9" / "point_cloud.ply").touch()
    expected = data / "point_cloud" / "iteration_30" / "point_cloud.ply"
    expected.touch()
    (data / "input.ply").touch()
    _write_json(data / "cameras.json", [_camera_entry("frame_1", 1)])

    discovered = discover_scene_inputs(data)

    assert discovered.cameras_path == (data / "cameras.json").resolve()
    assert discovered.ply_path == expected.resolve()
    assert discovered.iteration == 30


def test_camera_loading_naturally_orders_names_and_preserves_source_index(tmp_path):
    path = tmp_path / "cameras.json"
    entries = [
        _camera_entry("frame_00010", 10, position=(1, 2, 3)),
        _camera_entry("frame_00002", 2, position=(4, 5, 6)),
    ]
    _write_json(path, entries)

    cameras = load_camera_entries(path)

    assert [camera.image_name for camera in cameras] == ["frame_00002", "frame_00010"]
    assert [camera.source_index for camera in cameras] == [1, 0]
    assert [camera.source_id for camera in cameras] == [2, 10]
    assert cameras[0].principal_point_source == "assumed_image_center"
    np.testing.assert_array_equal(cameras[0].c2w[:3, 3], [4, 5, 6])


def test_camera_frame_uses_c2w_without_axis_flip_or_transpose(tmp_path):
    path = tmp_path / "camera.json"
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    entry = _camera_entry("frame", 7, position=(1, 2, 3), width=10, height=8, fx=5, fy=4)
    entry["rotation"] = rotation.tolist()
    _write_json(path, entry)

    camera = load_camera_entries(path)[0]
    frame = camera_frame(camera)

    np.testing.assert_allclose(frame.c2w[:3, :3], rotation)
    np.testing.assert_allclose(frame.w2c @ frame.c2w, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(frame.target, [2, 2, 3])
    assert math.degrees(frame.fov_x) == pytest.approx(90.0)
    assert math.degrees(frame.fov_y) == pytest.approx(90.0)
    assert frame.cx == 5.0
    assert frame.cy == 4.0


def test_nested_traversal_camera_json_is_accepted(tmp_path):
    path = tmp_path / "camera.json"
    c2w = np.eye(4)
    c2w[:3, 3] = [3, 4, 5]
    _write_json(
        path,
        {
            "position_index": 12,
            "position_label": "position_0012",
            "image_relative_path": "position_0012/lens_0000/image.png",
            "camera": {"position": [3, 4, 5], "camera_to_world": c2w.tolist()},
            "intrinsics": {
                "width": 640,
                "height": 480,
                "fx": 500,
                "fy": 510,
                "cx": 319.5,
                "cy": 239.5,
                "near": 0.02,
                "far": 50,
            },
        },
    )

    camera = load_camera_entries(path)[0]

    assert camera.source_id == 12
    assert camera.principal_point_source == "camera_json"
    np.testing.assert_array_equal(camera.c2w, c2w)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("width", 0, "width"),
        ("fx", float("nan"), "fx"),
        ("position", [0, 1], "position"),
        ("rotation", [[1, 0], [0, 1]], "rotation"),
    ],
)
def test_invalid_camera_fields_are_rejected(tmp_path, field, value, match):
    entry = _camera_entry("broken", 0)
    entry[field] = value
    path = tmp_path / "cameras.json"
    _write_json(path, [entry])

    with pytest.raises(ValueError, match=match):
        load_camera_entries(path)


def test_capture_metadata_matches_traversal_layout(tmp_path):
    path = tmp_path / "cameras.json"
    entry = _camera_entry("frame_00001", 4)
    _write_json(path, [entry])
    camera = load_camera_entries(path)[0]
    frame = camera_frame(camera)
    scene_directory = tmp_path / "output" / "scene"
    image_path, metadata_path = capture_paths(scene_directory, 3)

    metadata = capture_metadata(
        camera,
        frame,
        scene_name="scene",
        scene_type="object",
        position_index=3,
        scene_directory=scene_directory,
        image_path=image_path,
        cameras_path=path,
        fixed_intrinsics=True,
    )

    assert image_path == scene_directory / "position_0003" / "lens_0000" / "image.png"
    assert metadata_path.name == "camera.json"
    assert metadata["position_label"] == "position_0003"
    assert metadata["lens_label"] == "lens_0000"
    assert metadata["image_relative_path"] == "position_0003/lens_0000/image.png"
    assert metadata["sampling"]["source_camera_id"] == 4
    assert metadata["intrinsics"]["principal_point_source"] == "assumed_image_center"
    np.testing.assert_allclose(
        np.asarray(metadata["camera"]["world_to_camera"])
        @ np.asarray(metadata["camera"]["camera_to_world"]),
        np.eye(4),
        atol=1e-12,
    )


def test_render_dataset_writes_pairs_and_resumes(tmp_path, monkeypatch):
    data = tmp_path / "example_scene"
    ply = data / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.touch()
    _write_json(
        data / "cameras.json",
        [
            _camera_entry("frame_00010", 10, position=(0.2, 0, 0)),
            _camera_entry("frame_00002", 2, position=(0.1, 0, 0)),
        ],
    )

    class FakeAnalysis:
        effective_gaussian_count = 7
        aabb_min = np.full(3, -1.0)
        aabb_max = np.full(3, 1.0)

        def as_dict(self):
            return {"effective_gaussian_count": self.effective_gaussian_count}

    seen_configs = []

    class FakeScene:
        def __init__(self, path, cfg):
            assert path == ply.resolve()
            assert cfg.scene_analysis.filter_gaussians is False
            seen_configs.append(cfg)
            self.analysis = FakeAnalysis()

        def load_tensors(self):
            return object()

    render_calls = []

    class FakeRenderer:
        def __init__(self, _tensors, cfg):
            self.cfg = cfg

        def render(self, frame, *, width, height):
            render_calls.append((frame.position.copy(), width, height))
            return np.full((height, width, 3), 0.25, dtype=np.float32)

    monkeypatch.setattr(render_module, "GaussianScene", FakeScene)
    monkeypatch.setattr(render_module, "GaussianRenderer", FakeRenderer)
    output = tmp_path / "outputs"

    first = render_dataset(data, output)

    scene_directory = output / data.name
    assert first["rendered_count"] == 2
    assert first["skipped_existing_count"] == 0
    assert first["scene_type"] == "interior"
    assert [item[0].tolist() for item in render_calls] == [[0.1, 0, 0], [0.2, 0, 0]]
    image_path = scene_directory / "position_0000" / "lens_0000" / "image.png"
    metadata_path = image_path.with_name("camera.json")
    assert cv2.imread(str(image_path)).shape == (6, 8, 3)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sampling"]["source_image_name"] == "frame_00002"
    assert (scene_directory / "render_summary.json").is_file()

    second = render_dataset(data, output)

    assert second["rendered_count"] == 0
    assert second["skipped_existing_count"] == 2
    assert len(render_calls) == 2
    assert len(seen_configs) == 2


def test_partial_capture_requires_overwrite(tmp_path, monkeypatch):
    data = tmp_path / "scene"
    ply = data / "point_cloud" / "iteration_1" / "point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.touch()
    _write_json(data / "cameras.json", [_camera_entry("frame_1", 1)])
    image_path, _ = capture_paths(tmp_path / "out" / data.name, 0)
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    with pytest.raises(ValueError, match="partial output"):
        render_dataset(data, tmp_path / "out")


def test_shell_wrapper_forwards_two_paths_and_optional_arguments(tmp_path):
    capture = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${CAPTURE_PATH}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    wrapper = Path(__file__).parents[1] / "scripts" / "render_dataset_cameras.sh"
    environment = {
        **os.environ,
        "PYTHON": str(fake_python),
        "CAPTURE_PATH": str(capture),
    }

    result = subprocess.run(
        [
            str(wrapper),
            "/input/data",
            "/output/root",
            "--max-cameras",
            "1",
        ],
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[1:] == ["/input/data", "/output/root", "--max-cameras", "1"]
    assert arguments[0].endswith("/render_dataset_cameras.py")
