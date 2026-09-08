#!/usr/bin/env python3
"""Export a zoom-video output directory as a COLMAP dataset for 3DGS."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


SH_C0 = 0.28209479177387814
REQUIRED_POINT_FIELDS = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert frames and camera_trajectory.json from the zoom generator "
            "to a COLMAP/3DGS dataset."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Zoom generator output directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Dataset destination (default: update input_dir in place)",
    )
    parser.add_argument(
        "--point-cloud",
        type=Path,
        help="Source 3DGS PLY (default: resolve it from generation_summary.json)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10_000,
        help="Maximum number of initialization points (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Point sampling seed")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy frames instead of creating an images symlink",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated images/sparse/0 export",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_source_ply(
    explicit_path: Path | None, summary: dict[str, Any], project_root: Path
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    elif summary.get("input_ply_path"):
        recorded = Path(summary["input_ply_path"])
        candidates.append(recorded)
        if "3dscene" in recorded.parts:
            suffix = Path(*recorded.parts[recorded.parts.index("3dscene") :])
            candidates.append(project_root / suffix)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(path) for path in candidates) or "none"
    raise FileNotFoundError(f"could not resolve the source PLY; attempted: {attempted}")


def validate_inputs(
    input_dir: Path, trajectory: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[list[Path], int, int]:
    if not trajectory:
        raise ValueError("camera_trajectory.json contains no frames")

    video = summary.get("video", {})
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("generation_summary.json has invalid video dimensions")

    frames_dir = input_dir / "frames"
    image_paths: list[Path] = []
    for expected_index, frame in enumerate(trajectory):
        frame_index = int(frame["frame_index"])
        if frame_index != expected_index:
            raise ValueError(
                f"trajectory frame indices must be contiguous; expected {expected_index}, "
                f"got {frame_index}"
            )
        image_path = frames_dir / f"frame_{frame_index:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing frame: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"{image_path.name} is {image.size}, expected {(width, height)}"
                )
        image_paths.append(image_path)

        c2w = np.asarray(frame["camera_to_world"], dtype=np.float64)
        w2c = np.asarray(frame["world_to_camera"], dtype=np.float64)
        if c2w.shape != (4, 4) or w2c.shape != (4, 4):
            raise ValueError(f"frame {frame_index} camera matrices must be 4x4")
        if not np.allclose(w2c @ c2w, np.eye(4), rtol=1e-7, atol=1e-7):
            raise ValueError(f"frame {frame_index} camera matrices are not inverses")
        rotation = w2c[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-7, atol=1e-7):
            raise ValueError(f"frame {frame_index} rotation is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
            raise ValueError(f"frame {frame_index} rotation determinant is not +1")
        intrinsics = np.asarray(
            [frame["fx"], frame["fy"], frame["cx"], frame["cy"]], dtype=np.float64
        )
        if not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
            raise ValueError(f"frame {frame_index} has invalid intrinsics")

    extra_images = sorted(set(frames_dir.glob("frame_*.png")) - set(image_paths))
    if extra_images:
        raise ValueError(f"found {len(extra_images)} frames not present in the trajectory")
    return image_paths, width, height


def rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    # SciPy returns (x, y, z, w); COLMAP stores Hamilton quaternions as (w, x, y, z).
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = xyzw[[3, 0, 1, 2]]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def build_camera_records(
    trajectory: list[dict[str, Any]], width: int, height: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cameras: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for index, frame in enumerate(trajectory):
        record_id = index + 1
        w2c = np.asarray(frame["world_to_camera"], dtype=np.float64)
        cameras.append(
            {
                "id": record_id,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "params": np.asarray(
                    [frame["fx"], frame["fy"], frame["cx"], frame["cy"]],
                    dtype=np.float64,
                ),
            }
        )
        images.append(
            {
                "id": record_id,
                "qvec": rotation_to_qvec(w2c[:3, :3]),
                "tvec": w2c[:3, 3].copy(),
                "camera_id": record_id,
                "name": f"frame_{index:06d}.png",
            }
        )
    return cameras, images


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def sample_initial_points(
    ply_path: Path,
    summary: dict[str, Any],
    resolved_config: dict[str, Any],
    reference_frame: dict[str, Any],
    width: int,
    height: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    ply = PlyData.read(str(ply_path), mmap="r")
    if "vertex" not in ply:
        raise ValueError(f"source PLY has no vertex table: {ply_path}")
    vertices = ply["vertex"].data
    fields = set(vertices.dtype.names or ())
    missing = sorted(REQUIRED_POINT_FIELDS - fields)
    if missing:
        raise ValueError(f"source PLY is missing fields: {missing}")

    scene = summary["scene"]
    bounds = scene["position_filter_bounds"]
    lower = np.asarray(bounds["min"], dtype=np.float64)
    upper = np.asarray(bounds["max"], dtype=np.float64)
    opacity_storage = str(scene["opacity_storage"])
    opacity_threshold = float(resolved_config["scene_analysis"]["opacity_threshold"])

    root = summary["scene_root_transform"]
    root_rotation = Rotation.from_euler(
        root["rotation_order"], root["rotation_euler_deg"], degrees=True
    ).as_matrix()
    root_scale = float(root["scale"])
    root_translation = np.asarray(root["translation"], dtype=np.float64)

    w2c = np.asarray(reference_frame["world_to_camera"], dtype=np.float64)
    fx = float(reference_frame["fx"])
    fy = float(reference_frame["fy"])
    cx = float(reference_frame["cx"])
    cy = float(reference_frame["cy"])
    near = float(reference_frame["near"])
    far = float(reference_frame["far"])

    rng = np.random.default_rng(seed)
    selected_keys = np.empty(0, dtype=np.float64)
    selected_xyz = np.empty((0, 3), dtype=np.float32)
    selected_rgb = np.empty((0, 3), dtype=np.uint8)
    eligible_count = 0
    chunk_size = 250_000

    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start : start + chunk_size]
        raw_xyz = np.column_stack([chunk["x"], chunk["y"], chunk["z"]]).astype(
            np.float64, copy=False
        )
        xyz = root_scale * (raw_xyz @ root_rotation.T) + root_translation
        raw_opacity = np.asarray(chunk["opacity"], dtype=np.float32)
        alpha = (
            stable_sigmoid(raw_opacity)
            if opacity_storage == "logit"
            else np.clip(raw_opacity, 0.0, 1.0)
        )
        dc = np.column_stack(
            [chunk["f_dc_0"], chunk["f_dc_1"], chunk["f_dc_2"]]
        ).astype(np.float32, copy=False)

        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(alpha) & np.isfinite(dc).all(axis=1)
        valid = finite & (alpha >= opacity_threshold)
        valid &= np.all(xyz >= lower, axis=1) & np.all(xyz <= upper, axis=1)

        camera_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        depth = camera_xyz[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            image_x = fx * camera_xyz[:, 0] / depth + cx
            image_y = fy * camera_xyz[:, 1] / depth + cy
        valid &= (depth > near) & (depth < far)
        valid &= (image_x >= 0.0) & (image_x < width)
        valid &= (image_y >= 0.0) & (image_y < height)

        if not np.any(valid):
            continue
        valid_xyz = xyz[valid].astype(np.float32)
        valid_rgb = np.rint(np.clip(dc[valid] * SH_C0 + 0.5, 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
        keys = rng.random(len(valid_xyz))
        eligible_count += len(valid_xyz)

        all_keys = np.concatenate((selected_keys, keys))
        all_xyz = np.concatenate((selected_xyz, valid_xyz), axis=0)
        all_rgb = np.concatenate((selected_rgb, valid_rgb), axis=0)
        if len(all_keys) > max_points:
            keep = np.argpartition(all_keys, max_points - 1)[:max_points]
            selected_keys = all_keys[keep]
            selected_xyz = all_xyz[keep]
            selected_rgb = all_rgb[keep]
        else:
            selected_keys = all_keys
            selected_xyz = all_xyz
            selected_rgb = all_rgb

    if len(selected_xyz) == 0:
        raise ValueError("no source Gaussian centers lie inside the rendered camera frustum")
    order = np.argsort(selected_keys)
    return selected_xyz[order], selected_rgb[order], eligible_count


def write_cameras(cameras: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            handle.write(
                struct.pack(
                    "<iiQQdddd",
                    camera["id"],
                    1,  # PINHOLE
                    camera["width"],
                    camera["height"],
                    *camera["params"],
                )
            )

    with (output_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in cameras:
            params = " ".join(f"{value:.17g}" for value in camera["params"])
            handle.write(
                f"{camera['id']} PINHOLE {camera['width']} {camera['height']} {params}\n"
            )


def write_images(images: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            handle.write(
                struct.pack(
                    "<idddddddi",
                    image["id"],
                    *image["qvec"],
                    *image["tvec"],
                    image["camera_id"],
                )
            )
            handle.write(image["name"].encode("utf-8") + b"\0")
            handle.write(struct.pack("<Q", 0))

    with (output_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for image in images:
            pose = " ".join(
                f"{value:.17g}" for value in np.concatenate((image["qvec"], image["tvec"]))
            )
            handle.write(
                f"{image['id']} {pose} {image['camera_id']} {image['name']}\n\n"
            )


def write_points(xyz: np.ndarray, rgb: np.ndarray, output_dir: Path) -> None:
    with (output_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(xyz)))
        for point_id, (point, color) in enumerate(zip(xyz, rgb), start=1):
            handle.write(
                struct.pack(
                    "<QdddBBBdQ",
                    point_id,
                    *point.astype(np.float64),
                    *color,
                    0.0,
                    0,
                )
            )

    with (output_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        handle.write(f"# Number of points: {len(xyz)}, mean track length: 0\n")
        for point_id, (point, color) in enumerate(zip(xyz, rgb), start=1):
            handle.write(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )

    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
        str(output_dir / "points3D.ply")
    )


def prepare_images(input_dir: Path, output_dir: Path, copy_images: bool) -> str:
    source = input_dir / "frames"
    destination = output_dir / "images"
    if copy_images:
        shutil.copytree(source, destination)
        return "copy"
    relative_source = os.path.relpath(source, start=output_dir)
    destination.symlink_to(relative_source, target_is_directory=True)
    return "relative_symlink"


def remove_existing_export(output_dir: Path) -> None:
    images = output_dir / "images"
    model = output_dir / "sparse" / "0"
    manifest = output_dir / "colmap_export.json"
    if images.is_symlink() or images.is_file():
        images.unlink()
    elif images.is_dir():
        shutil.rmtree(images)
    if model.is_dir():
        shutil.rmtree(model)
    elif model.exists():
        model.unlink()
    if manifest.exists():
        manifest.unlink()


def main() -> None:
    args = parse_args()
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else input_dir
    )
    trajectory = load_json(input_dir / "camera_trajectory.json")
    summary = load_json(input_dir / "generation_summary.json")
    with (input_dir / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        resolved_config = yaml.safe_load(handle)
    image_paths, width, height = validate_inputs(input_dir, trajectory, summary)

    project_root = Path(__file__).resolve().parents[1]
    source_ply = resolve_source_ply(args.point_cloud, summary, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_paths = [
        output_dir / "images",
        output_dir / "sparse" / "0",
        output_dir / "colmap_export.json",
    ]
    if any(path.exists() or path.is_symlink() for path in export_paths):
        if not args.overwrite:
            raise FileExistsError(
                "COLMAP export already exists; pass --overwrite to replace generated output"
            )
        remove_existing_export(output_dir)

    cameras, images = build_camera_records(trajectory, width, height)
    reference_frame = min(trajectory, key=lambda frame: float(frame["fx"]))
    xyz, rgb, eligible_count = sample_initial_points(
        source_ply,
        summary,
        resolved_config,
        reference_frame,
        width,
        height,
        args.max_points,
        args.seed,
    )

    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".colmap-export-", dir=sparse_dir))
    try:
        write_cameras(cameras, temporary_dir)
        write_images(images, temporary_dir)
        write_points(xyz, rgb, temporary_dir)
        temporary_dir.rename(sparse_dir / "0")
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    image_mode = prepare_images(input_dir, output_dir, args.copy_images)
    camera_centers = np.asarray([frame["camera_position"] for frame in trajectory])
    center_span = np.ptp(camera_centers, axis=0)
    manifest = {
        "format": "COLMAP sparse model for 3D Gaussian Splatting",
        "source_directory": str(input_dir),
        "source_point_cloud": str(source_ply),
        "image_count": len(image_paths),
        "camera_count": len(cameras),
        "camera_model": "PINHOLE",
        "image_width": width,
        "image_height": height,
        "focal_length_range": [
            min(float(frame["fx"]) for frame in trajectory),
            max(float(frame["fx"]) for frame in trajectory),
        ],
        "camera_center_axis_span": center_span.tolist(),
        "point_count": len(xyz),
        "eligible_source_point_count": eligible_count,
        "point_sampling_seed": args.seed,
        "points_are_initialization_priors": True,
        "points_have_tracks": False,
        "image_storage": image_mode,
        "coordinate_convention": "COLMAP/OpenCV: +x right, +y down, +z forward",
        "notes": [
            "Each frame has its own camera record because focal length changes every frame.",
            "Camera extrinsics are static in this generated sequence; use point-cloud-based "
            "scene-radius initialization when the 3DGS implementation provides it.",
        ],
    }
    with (output_dir / "colmap_export.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Exported {len(images)} images and {len(xyz)} points to {output_dir}")
    print(f"COLMAP model: {output_dir / 'sparse' / '0'}")
    print(f"Images: {output_dir / 'images'} ({image_mode})")


if __name__ == "__main__":
    main()
