#!/usr/bin/env python3
"""Render a trained 3DGS scene from its exported cameras.json poses."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from zoomgen.camera import CameraFrame
from zoomgen.config import RenderConfig, SceneAnalysisConfig, SceneConfig
from zoomgen.renderer import GaussianRenderer
from zoomgen.scene import GaussianScene


_ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")
_NATURAL_NUMBER_PATTERN = re.compile(r"(\d+)")


@dataclass(frozen=True)
class SceneInputs:
    data_path: Path
    cameras_path: Path
    ply_path: Path
    iteration: int | None


@dataclass(frozen=True)
class SourceCamera:
    source_index: int
    source_id: Any
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    principal_point_source: str
    near: float
    far: float
    c2w: np.ndarray


def _positive_int(entry: dict, key: str, source: str) -> int:
    raw = entry.get(key)
    if isinstance(raw, bool):
        raise ValueError(f"camera {source} has invalid {key}: {raw!r}")
    try:
        value = int(raw)
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"camera {source} has invalid {key}: {raw!r}") from exc
    if value <= 0 or not math.isfinite(numeric) or numeric != value:
        raise ValueError(f"camera {source} has invalid {key}: {raw!r}")
    return value


def _positive_float(entry: dict, key: str, source: str) -> float:
    raw = entry.get(key)
    if isinstance(raw, bool):
        raise ValueError(f"camera {source} has invalid {key}: {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"camera {source} has invalid {key}: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"camera {source} has invalid {key}: {raw!r}")
    return value


def _finite_vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite array with shape {shape}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite array with shape {shape}")
    return array


def _project_to_rotation(value: Any, source: str) -> np.ndarray:
    matrix = _finite_vector(value, (3, 3), f"camera {source} rotation")
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _source_identifier(value: Any, fallback: int) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return str(value)


def _graphdeco_camera(entry: dict, index: int) -> SourceCamera:
    source = str(entry.get("img_name", entry.get("id", index)))
    width = _positive_int(entry, "width", source)
    height = _positive_int(entry, "height", source)
    fx = _positive_float(entry, "fx", source)
    fy = _positive_float(entry, "fy", source)
    cx = float(entry.get("cx", width / 2.0))
    cy = float(entry.get("cy", height / 2.0))
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError(f"camera {source} has non-finite principal point")
    principal_point_source = str(
        entry.get(
            "principal_point_source",
            "camera_json"
            if "cx" in entry and "cy" in entry
            else "assumed_image_center",
        )
    )
    near = float(entry.get("near", 0.01))
    far = float(entry.get("far", 100.0))
    if not (math.isfinite(near) and math.isfinite(far) and 0.0 < near < far):
        raise ValueError(f"camera {source} must satisfy 0 < near < far")

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = _project_to_rotation(entry.get("rotation"), source)
    c2w[:3, 3] = _finite_vector(
        entry.get("position"), (3,), f"camera {source} position"
    )
    return SourceCamera(
        source_index=index,
        source_id=_source_identifier(entry.get("id"), index),
        image_name=str(entry.get("img_name", f"camera_{index:05d}")),
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        principal_point_source=principal_point_source,
        near=near,
        far=far,
        c2w=c2w,
    )


def _traversal_camera(entry: dict, index: int) -> SourceCamera:
    camera = entry.get("camera")
    intrinsics = entry.get("intrinsics")
    if not isinstance(camera, dict) or not isinstance(intrinsics, dict):
        raise ValueError(f"camera entry {index} has invalid nested camera metadata")
    source = str(entry.get("position_label", entry.get("position_index", index)))
    c2w = _finite_vector(
        camera.get("camera_to_world"), (4, 4), f"camera {source} camera_to_world"
    )
    if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"camera {source} camera_to_world has invalid homogeneous row")
    c2w = c2w.copy()
    c2w[:3, :3] = _project_to_rotation(c2w[:3, :3], source)
    c2w[:3, 3] = _finite_vector(
        camera.get("position", c2w[:3, 3]), (3,), f"camera {source} position"
    )
    normalized = {
        **intrinsics,
        "id": entry.get("position_index", index),
        "img_name": entry.get("image_relative_path", source),
        "position": c2w[:3, 3],
        "rotation": c2w[:3, :3],
        "principal_point_source": intrinsics.get(
            "principal_point_source",
            "camera_json",
        ),
    }
    parsed = _graphdeco_camera(normalized, index)
    return SourceCamera(**{**parsed.__dict__, "c2w": c2w})


def _natural_name_key(camera: SourceCamera) -> tuple:
    parts = _NATURAL_NUMBER_PATTERN.split(camera.image_name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    ) + ((2, camera.source_index),)


def load_camera_entries(path: str | Path) -> list[SourceCamera]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict) and isinstance(document.get("cameras"), list):
        entries = document["cameras"]
    elif isinstance(document, dict):
        entries = [document]
    else:
        raise ValueError(f"camera JSON must contain an object or list: {resolved}")
    if not entries:
        raise ValueError(f"camera JSON contains no cameras: {resolved}")

    cameras = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"camera entry {index} must be a JSON object")
        if "camera" in entry or "intrinsics" in entry:
            cameras.append(_traversal_camera(entry, index))
        else:
            cameras.append(_graphdeco_camera(entry, index))
    return sorted(cameras, key=_natural_name_key)


def discover_scene_inputs(data_path: str | Path) -> SceneInputs:
    data = Path(data_path).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"data path is not a directory: {data}")

    cameras_path = next(
        (data / name for name in ("cameras.json", "camera.json") if (data / name).is_file()),
        None,
    )
    if cameras_path is None:
        recursive = sorted(
            path for path in data.rglob("camera*.json") if path.is_file()
        )
        if len(recursive) != 1:
            raise ValueError(
                f"expected cameras.json or camera.json below {data}; found {len(recursive)} candidates"
            )
        cameras_path = recursive[0]

    iteration_candidates: list[tuple[int, Path]] = []
    point_cloud_root = data / "point_cloud"
    if point_cloud_root.is_dir():
        for directory in point_cloud_root.iterdir():
            match = _ITERATION_PATTERN.fullmatch(directory.name)
            candidate = directory / "point_cloud.ply"
            if match and candidate.is_file():
                iteration_candidates.append((int(match.group(1)), candidate))
    if iteration_candidates:
        iteration, ply_path = max(iteration_candidates, key=lambda item: item[0])
    elif (data / "point_cloud.ply").is_file():
        iteration, ply_path = None, data / "point_cloud.ply"
    else:
        recursive_ply = sorted(
            path for path in data.rglob("point_cloud.ply") if path.is_file()
        )
        if len(recursive_ply) == 1:
            ply_path = recursive_ply[0]
            match = _ITERATION_PATTERN.fullmatch(ply_path.parent.name)
            iteration = int(match.group(1)) if match else None
        else:
            all_ply = sorted(path for path in data.glob("*.ply") if path.is_file())
            if len(all_ply) != 1:
                raise ValueError(
                    f"expected a trained point_cloud.ply below {data}; found {len(recursive_ply)} candidates"
                )
            iteration, ply_path = None, all_ply[0]
    return SceneInputs(data, cameras_path.resolve(), ply_path.resolve(), iteration)


def camera_frame(camera: SourceCamera) -> CameraFrame:
    position = camera.c2w[:3, 3].copy()
    look_direction = camera.c2w[:3, 2]
    fov_x = 2.0 * math.atan(camera.width / (2.0 * camera.fx))
    fov_y = 2.0 * math.atan(camera.height / (2.0 * camera.fy))
    return CameraFrame(
        position=position,
        target=position + look_direction,
        c2w=camera.c2w.copy(),
        w2c=np.linalg.inv(camera.c2w),
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        fov_x=fov_x,
        fov_y=fov_y,
        near=camera.near,
        far=camera.far,
        camera_center_offset=np.zeros(3, dtype=np.float64),
    )


def _training_uses_white_background(data_path: Path) -> bool:
    cfg_args = data_path / "cfg_args"
    if not cfg_args.is_file():
        return False
    try:
        expression = ast.parse(cfg_args.read_text(encoding="utf-8"), mode="eval").body
    except (OSError, SyntaxError, UnicodeError):
        return False
    if not isinstance(expression, ast.Call):
        return False
    for keyword in expression.keywords:
        if keyword.arg == "white_background" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is True
    return False


def _renderer_config(inputs: SceneInputs, first_camera: SourceCamera) -> SimpleNamespace:
    white_background = _training_uses_white_background(inputs.data_path)
    return SimpleNamespace(
        scene=SceneConfig(),
        scene_analysis=SceneAnalysisConfig(filter_gaussians=False),
        render=RenderConfig(
            device="cuda",
            background_rgb=(1.0, 1.0, 1.0) if white_background else (0.0, 0.0, 0.0),
            sh_degree="auto",
            antialiasing=False,
        ),
        video=SimpleNamespace(width=first_camera.width, height=first_camera.height),
    )


def _fixed_intrinsics(cameras: list[SourceCamera]) -> bool:
    first = cameras[0]
    return all(
        camera.width == first.width
        and camera.height == first.height
        and math.isclose(camera.fx, first.fx, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(camera.fy, first.fy, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(camera.cx, first.cx, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(camera.cy, first.cy, rel_tol=1e-9, abs_tol=1e-9)
        for camera in cameras[1:]
    )


def _infer_scene_type(cameras: list[SourceCamera], scene: GaussianScene) -> tuple[str, float]:
    positions = np.stack([camera.c2w[:3, 3] for camera in cameras])
    inside = np.all(positions >= scene.analysis.aabb_min, axis=1) & np.all(
        positions <= scene.analysis.aabb_max, axis=1
    )
    inside_ratio = float(np.mean(inside))
    return ("interior" if inside_ratio >= 0.5 else "object"), inside_ratio


def capture_paths(scene_directory: Path, position_index: int) -> tuple[Path, Path]:
    directory = scene_directory / f"position_{position_index:04d}" / "lens_0000"
    return directory / "image.png", directory / "camera.json"


def capture_metadata(
    camera: SourceCamera,
    frame: CameraFrame,
    *,
    scene_name: str,
    scene_type: str,
    position_index: int,
    scene_directory: Path,
    image_path: Path,
    cameras_path: Path,
    fixed_intrinsics: bool,
) -> dict:
    position_label = f"position_{position_index:04d}"
    return {
        "scene_name": scene_name,
        "scene_type": scene_type,
        "position_index": position_index,
        "position_label": position_label,
        "lens_index": 0,
        "lens_label": "lens_0000",
        "image_relative_path": str(image_path.relative_to(scene_directory)),
        "camera": {
            "coordinate_convention": "OpenCV/COLMAP camera-to-world; +x right, +y down, +z forward",
            "position": frame.position.tolist(),
            "look_direction": frame.c2w[:3, 2].tolist(),
            "look_at_target": frame.target.tolist(),
            "yaw_deg": None,
            "pitch_deg": None,
            "roll_deg": None,
            "camera_to_world": frame.c2w.tolist(),
            "world_to_camera": frame.w2c.tolist(),
        },
        "intrinsics": {
            "fixed_across_traversal": fixed_intrinsics,
            "width": camera.width,
            "height": camera.height,
            "fx": frame.fx,
            "fy": frame.fy,
            "cx": frame.cx,
            "cy": frame.cy,
            "principal_point_source": camera.principal_point_source,
            "fov_x_deg": math.degrees(frame.fov_x),
            "fov_y_deg": math.degrees(frame.fov_y),
            "near": frame.near,
            "far": frame.far,
        },
        "sampling": {
            "strategy": "input_camera_json",
            "source_camera_json": str(cameras_path),
            "source_camera_index": camera.source_index,
            "source_camera_id": camera.source_id,
            "source_image_name": camera.image_name,
            "output_order": "natural_source_image_name",
        },
        "geometry_quality": None,
        "image_quality": None,
        "scene_root_transform": SceneConfig().root_transform.model_dump(mode="json"),
    }


def _encoded_rgb_image(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def _write_json(path: Path, value: dict | list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _save_capture_pair(
    image_path: Path, metadata_path: Path, rgb: np.ndarray, metadata: dict
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_tmp = image_path.with_name(f".{image_path.stem}.tmp.png")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.stem}.tmp.json")
    try:
        image = _encoded_rgb_image(rgb)
        ok = cv2.imwrite(
            str(image_tmp), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not ok:
            raise RuntimeError(f"failed to save rendered image: {image_path}")
        _write_json(metadata_tmp, metadata)
        image_tmp.replace(image_path)
        metadata_tmp.replace(metadata_path)
    finally:
        if image_tmp.exists():
            image_tmp.unlink()
        if metadata_tmp.exists():
            metadata_tmp.unlink()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.json")
    try:
        _write_json(temporary, value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_dataset(
    data_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    max_cameras: int | None = None,
) -> dict:
    inputs = discover_scene_inputs(data_path)
    cameras = load_camera_entries(inputs.cameras_path)
    if max_cameras is not None:
        if max_cameras <= 0:
            raise ValueError("max_cameras must be positive")
        cameras = cameras[:max_cameras]

    scene_name = inputs.data_path.name
    scene_directory = Path(output_path).expanduser().resolve() / scene_name
    existing_pairs: list[bool] = []
    for index in range(len(cameras)):
        image_path, metadata_path = capture_paths(scene_directory, index)
        image_exists, metadata_exists = image_path.exists(), metadata_path.exists()
        if image_exists != metadata_exists and not overwrite:
            raise ValueError(
                f"partial output exists for position_{index:04d}; use --overwrite to repair it"
            )
        existing_pairs.append(image_exists and metadata_exists)

    cfg = _renderer_config(inputs, cameras[0])
    print(f"Loading scene: {inputs.ply_path}")
    scene = GaussianScene(inputs.ply_path, cfg)
    scene_type, camera_inside_aabb_ratio = _infer_scene_type(cameras, scene)
    fixed_intrinsics = _fixed_intrinsics(cameras)
    print(
        f"Loading {scene.analysis.effective_gaussian_count} finite Gaussians on {cfg.render.device}"
    )
    renderer = GaussianRenderer(scene.load_tensors(), cfg)

    rendered_count = 0
    skipped_count = 0
    captures = []
    for index, camera in enumerate(tqdm(cameras, desc="Rendering input cameras")):
        image_path, metadata_path = capture_paths(scene_directory, index)
        status = "rendered"
        if existing_pairs[index] and not overwrite:
            skipped_count += 1
            status = "skipped_existing"
        else:
            cfg.video.width = camera.width
            cfg.video.height = camera.height
            frame = camera_frame(camera)
            rgb = renderer.render(frame, width=camera.width, height=camera.height)
            metadata = capture_metadata(
                camera,
                frame,
                scene_name=scene_name,
                scene_type=scene_type,
                position_index=index,
                scene_directory=scene_directory,
                image_path=image_path,
                cameras_path=inputs.cameras_path,
                fixed_intrinsics=fixed_intrinsics,
            )
            _save_capture_pair(image_path, metadata_path, rgb, metadata)
            rendered_count += 1
        captures.append(
            {
                "position_index": index,
                "position_label": f"position_{index:04d}",
                "source_camera_index": camera.source_index,
                "source_camera_id": camera.source_id,
                "source_image_name": camera.image_name,
                "image": str(image_path.relative_to(scene_directory)),
                "metadata": str(metadata_path.relative_to(scene_directory)),
                "status": status,
            }
        )

    summary = {
        "scene_name": scene_name,
        "scene_type": scene_type,
        "data_path": str(inputs.data_path),
        "input_ply_path": str(inputs.ply_path),
        "input_camera_json_path": str(inputs.cameras_path),
        "iteration": inputs.iteration,
        "output_scene_directory": str(scene_directory),
        "camera_order": "natural_source_image_name",
        "camera_count": len(cameras),
        "rendered_count": rendered_count,
        "skipped_existing_count": skipped_count,
        "camera_inside_robust_aabb_ratio": camera_inside_aabb_ratio,
        "fixed_intrinsics": fixed_intrinsics,
        "background_rgb": list(cfg.render.background_rgb),
        "antialiasing": cfg.render.antialiasing,
        "scene_root_transform": cfg.scene.root_transform.model_dump(mode="json"),
        "scene_analysis": scene.analysis.as_dict(),
        "captures": captures,
    }
    _write_json_atomic(scene_directory / "render_summary.json", summary)
    print(
        f"Render complete: {rendered_count} rendered, {skipped_count} skipped -> {scene_directory}"
    )
    return summary


def _positive_cli_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a 3DGS data directory using its exported camera JSON"
    )
    parser.add_argument("data_path", help="directory containing cameras.json and point_cloud/")
    parser.add_argument("output_path", help="output root; a scene-name subdirectory is created")
    parser.add_argument("--overwrite", action="store_true", help="replace existing capture pairs")
    parser.add_argument(
        "--max-cameras", type=_positive_cli_int, default=None,
        help="render only the first N naturally ordered cameras (for smoke tests)",
    )
    args = parser.parse_args()
    try:
        render_dataset(
            args.data_path,
            args.output_path,
            overwrite=args.overwrite,
            max_cameras=args.max_cameras,
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
