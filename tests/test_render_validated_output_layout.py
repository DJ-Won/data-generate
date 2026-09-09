import json

import pytest

from render_validated_zoom_dataset import (
    _move_position_output,
    _position_paths,
    _rebase_path_values,
    _remove_invalid_cache,
)


def test_position_paths_uses_requested_classification_directory(tmp_path):
    directory = tmp_path / "scene" / "invaild" / "position_0007"
    paths = _position_paths(directory)

    assert paths.directory == directory
    assert paths.capture_image == directory / "lens_0000" / "image.png"
    assert paths.camera_json == directory / "lens_0000" / "camera.json"
    assert paths.validation_report == (
        directory / "validation" / "validation_report.json"
    )
    assert paths.scene_config == directory / "validation" / "scene_config.yaml"


def test_rebase_path_values_updates_nested_paths_only(tmp_path):
    source = tmp_path / "scene" / "invaild" / "position_0000"
    destination = tmp_path / "scene" / "vaild" / "position_0000"
    outside = tmp_path / "scene" / "pipeline_summary.json"
    value = {
        "directory": str(source),
        "nested": [str(source / "zoom_video.mp4"), {"outside": str(outside)}],
    }

    rebased = _rebase_path_values(value, source, destination)

    assert rebased["directory"] == str(destination)
    assert rebased["nested"][0] == str(destination / "zoom_video.mp4")
    assert rebased["nested"][1]["outside"] == str(outside)


def test_move_position_output_rebases_persisted_json_paths(tmp_path):
    source = tmp_path / "scene" / "invaild" / "position_0000"
    destination = tmp_path / "scene" / "vaild" / "position_0000"
    paths = _position_paths(source)
    paths.validation_directory.mkdir(parents=True)
    report = {"image": str(source / "validation" / "frame_000000.png")}
    summary = {"video": str(source / "zoom_video.mp4")}
    paths.validation_report.write_text(json.dumps(report), encoding="utf-8")
    (source / "generation_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    moved = _move_position_output(paths, destination)

    assert not source.exists()
    assert moved.directory == destination
    assert json.loads(moved.validation_report.read_text(encoding="utf-8")) == {
        "image": str(destination / "validation" / "frame_000000.png")
    }
    assert json.loads(
        (destination / "generation_summary.json").read_text(encoding="utf-8")
    ) == {"video": str(destination / "zoom_video.mp4")}


def test_move_position_output_refuses_to_replace_existing_target(tmp_path):
    source = tmp_path / "scene" / "invaild" / "position_0000"
    destination = tmp_path / "scene" / "vaild" / "position_0000"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "source.marker").touch()
    (destination / "destination.marker").touch()

    with pytest.raises(RuntimeError, match="target already exists"):
        _move_position_output(_position_paths(source), destination)

    assert (source / "source.marker").is_file()
    assert (destination / "destination.marker").is_file()


def test_remove_invalid_cache_preserves_other_scene_outputs(tmp_path):
    scene_directory = tmp_path / "scene"
    invalid_file = scene_directory / "invaild" / "position_0000" / "cache.bin"
    valid_file = scene_directory / "vaild" / "position_0001" / "zoom_video.mp4"
    scene_file = scene_directory / "pipeline_summary.json"
    invalid_file.parent.mkdir(parents=True)
    valid_file.parent.mkdir(parents=True)
    invalid_file.touch()
    valid_file.touch()
    scene_file.touch()

    assert _remove_invalid_cache(scene_directory) is True

    assert not (scene_directory / "invaild").exists()
    assert valid_file.is_file()
    assert scene_file.is_file()
    assert _remove_invalid_cache(scene_directory) is False


def test_remove_invalid_cache_refuses_symlink(tmp_path):
    scene_directory = tmp_path / "scene"
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    external_file = external_directory / "keep.bin"
    external_file.touch()
    scene_directory.mkdir()
    (scene_directory / "invaild").symlink_to(
        external_directory, target_is_directory=True
    )

    with pytest.raises(RuntimeError, match="symlinked invalid cache"):
        _remove_invalid_cache(scene_directory)

    assert external_file.is_file()
