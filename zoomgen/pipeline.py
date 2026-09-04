from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

from .alignment import (
    AlignmentSelectionRequired,
    BasePose,
    align_reference,
    infer_scene_type,
    initialize_interior,
    initialize_manual,
    resolved_pose_config,
)
from .camera import CameraFrame, base_view, build_camera_frames, intrinsics, look_at
from .color import process_frame
from .config import GeneratorConfig, prepare_output, save_resolved_config
from .renderer import GaussianRenderer
from .scene import GaussianScene
from .video import FFmpegWriter, probe_video
from .zoom import FrameSchedule, allocate_frame_counts, build_schedule, temporal_state


def _json_dump(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def coverage_metrics(alpha: np.ndarray, threshold: float) -> tuple[float, float]:
    foreground = alpha > threshold
    coverage = float(foreground.mean())
    background = ~foreground
    labels, count = ndimage.label(background, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        largest = 0.0
    else:
        sizes = np.bincount(labels.ravel())
        largest = float(sizes[1:].max() / background.size) if len(sizes) > 1 else 0.0
    return coverage, largest


def _static_camera(cfg: GeneratorConfig, position: np.ndarray, target: np.ndarray,
                   radius: float, zoom: float) -> CameraFrame:
    c2w = look_at(position, target, cfg.camera.initial_view.roll_deg)
    fx, fy, cx, cy, fov_x, fov_y = intrinsics(cfg, zoom)
    near = max(radius * cfg.camera.clipping.near_radius_ratio, 1e-5)
    far = max(radius * cfg.camera.clipping.far_radius_ratio, np.linalg.norm(position - target) + 2 * radius)
    return CameraFrame(position, target, c2w, np.linalg.inv(c2w), fx, fy, cx, cy, fov_x, fov_y,
                       near, far, np.zeros(3))


def auto_fit(cfg: GeneratorConfig, renderer: GaussianRenderer, initial_position: np.ndarray,
             target: np.ndarray, radius: float, warnings_out: list[str]) -> tuple[np.ndarray, dict]:
    af = cfg.camera.auto_fit
    direction = (initial_position - target) / np.linalg.norm(initial_position - target)
    initial_distance = float(np.linalg.norm(initial_position - target))
    if not af.enabled:
        return initial_position, {"enabled": False, "distance": initial_distance}
    lo, hi = max(0.10 * radius, 1e-4), max(20.0 * radius, initial_distance * 2.0)
    best = None
    history = []
    for iteration in tqdm(range(af.max_iterations), desc="Auto-fitting camera", unit="render"):
        distance = (lo + hi) / 2.0
        pos = target + direction * distance
        cam = _static_camera(cfg, pos, target, radius, cfg.zoom.lenses[0].zoom_min)
        alpha = renderer.render(cam, af.preview_width, af.preview_height, alpha_only=True)
        cov, bg = coverage_metrics(alpha, af.alpha_threshold)
        score = (abs(cov - af.target_coverage_ratio)
                 + 2.0 * max(0.0, bg - af.max_largest_background_component_ratio)
                 + 2.0 * max(0.0, af.minimum_coverage_ratio - cov))
        item = {"iteration": iteration, "distance": distance, "coverage_ratio": cov,
                "largest_background_component_ratio": bg, "score": score}
        history.append(item)
        if best is None or score < best[0]:
            best = (score, pos.copy(), item)
        if cov < af.target_coverage_ratio or bg > af.max_largest_background_component_ratio:
            hi = distance
        else:
            lo = distance
    assert best is not None
    chosen = best[2]
    met = (chosen["coverage_ratio"] >= af.minimum_coverage_ratio
           and chosen["largest_background_component_ratio"] <= af.max_largest_background_component_ratio)
    if not met:
        message = (
            "auto-fit could not satisfy all coverage constraints; using the best scored distance "
            f"(coverage={chosen['coverage_ratio']:.4f}, "
            f"largest_background={chosen['largest_background_component_ratio']:.4f})"
        )
        warnings_out.append(message)
        warnings.warn(message)
    return best[1], {"enabled": True, "constraints_met": met, "chosen": chosen, "history": history}


def precheck_trajectory(
    cfg: GeneratorConfig,
    scene: GaussianScene,
    renderer: GaussianRenderer,
    schedule: list[FrameSchedule],
    base: BasePose,
    warnings_out: list[str],
) -> tuple[BasePose, list[CameraFrame], list[tuple[float, float]], float]:
    af = cfg.camera.auto_fit
    motion_scale = 1.0
    best = None
    interior = base.allowed_aabb is not None
    if interior:
        threshold_cov = cfg.camera.initialization.interior.minimum_coverage_ratio
        threshold_bg = cfg.camera.initialization.interior.maximum_largest_background_ratio
        sample = scene.effective_position_sample(200_000, max(0.1, cfg.scene_analysis.opacity_threshold))
        clearance_tree = cKDTree(sample)
        minimum_clearance = (
            cfg.camera.initialization.interior.minimum_clearance_radius_ratio
            * scene.analysis.radius
        )
    else:
        threshold_cov = af.minimum_coverage_ratio
        threshold_bg = af.max_largest_background_component_ratio
        clearance_tree = None
        minimum_clearance = 0.0
    attempts = 1 if not cfg.camera.motion.enabled else af.precheck_max_attempts
    current = base
    for attempt in range(attempts):
        frames = build_camera_frames(
            cfg,
            schedule,
            current.position,
            current.target,
            scene.analysis.radius,
            motion_scale,
            base_c2w=current.c2w,
            fov_y_deg_at_1x=current.fov_y_deg_at_1x,
            allowed_aabb=current.allowed_aabb,
            clip_radius=current.local_radius,
        )
        metrics = []
        for camera in tqdm(frames, desc=f"Trajectory precheck {attempt + 1}", unit="frame"):
            alpha = renderer.render(camera, af.preview_width, af.preview_height, alpha_only=True)
            metrics.append(coverage_metrics(alpha, af.alpha_threshold))
        cov = np.array([x[0] for x in metrics])
        bg = np.array([x[1] for x in metrics])
        if clearance_tree is not None:
            clearances = np.asarray(
                clearance_tree.query(np.asarray([x.position for x in frames]), k=1)[0]
            )
            clearance_penalty = float(np.mean(np.maximum(minimum_clearance - clearances, 0)))
            clearance_ok = bool(np.all(clearances >= minimum_clearance))
        else:
            clearance_penalty = 0.0
            clearance_ok = True
        score = float(
            np.mean(np.maximum(threshold_cov - cov, 0))
            + np.mean(np.maximum(bg - threshold_bg, 0))
            + clearance_penalty
        )
        if best is None or score < best[0]:
            best = (score, current, frames, metrics, motion_scale)
        if (
            np.all(cov >= threshold_cov)
            and np.all(bg <= threshold_bg)
            and clearance_ok
        ):
            return current, frames, metrics, motion_scale
        motion_scale *= 0.5
        if not interior and current.source == "object_auto" and attempt >= 1:
            # Object mode retains the legacy distance adjustment.
            target = current.target
            direction = (current.position - target) / np.linalg.norm(current.position - target)
            position = target + direction * np.linalg.norm(current.position - target) * 0.9
            c2w = look_at(position, target, cfg.camera.initial_view.roll_deg)
            current = BasePose(
                c2w,
                current.fov_y_deg_at_1x,
                current.source,
                None,
                current.local_radius,
                current.clearance,
            )
    assert best is not None
    message = (
        "trajectory precheck could not satisfy every coverage/background/clearance "
        "constraint; using the best attempt without moving an interior/manual base pose"
    )
    warnings_out.append(message)
    warnings.warn(message)
    return best[1], best[2], best[3], best[4]


def _save_image(path: Path, rgb: np.ndarray) -> None:
    u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), params):
        raise RuntimeError(f"failed to write frame: {path}")


def resolve_base_pose(
    cfg: GeneratorConfig,
    scene: GaussianScene,
    renderer: GaussianRenderer,
    warning_messages: list[str],
) -> tuple[BasePose, dict]:
    init = cfg.camera.initialization
    if init.resolved_pose is not None:
        resolved = init.resolved_pose
        c2w = np.asarray(resolved.camera_to_world, dtype=np.float64)
        allowed = (
            (scene.analysis.aabb_min, scene.analysis.aabb_max)
            if init.scene_type in ("interior", "reference_image")
            else None
        )
        base = BasePose(
            c2w,
            resolved.fov_y_deg_at_1x,
            resolved.source,
            allowed,
            scene.analysis.radius * (0.2 if allowed is not None else 1.0),
            math.nan,
            resolved.candidate_index,
        )
        return base, {"strategy": "resolved_pose", "automatic_position_change": False}
    if init.mode == "manual" or init.scene_type == "manual":
        return initialize_manual(cfg, scene), {
            "strategy": "manual",
            "automatic_position_change": False,
        }
    if init.mode == "reference_image" or init.scene_type == "reference_image":
        base = align_reference(cfg, scene, renderer, cfg.output.directory)
        return base, {
            "strategy": "reference_image",
            "automatic_position_change": False,
            "alignment": base.alignment,
        }
    scene_type = init.scene_type
    if scene_type == "auto":
        scene_type, message = infer_scene_type(cfg, scene)
        if message:
            warning_messages.append(message)
            warnings.warn(message)
    if scene_type == "object":
        start_pos, target, _ = base_view(
            cfg, scene.analysis.center, scene.analysis.radius
        )
        fitted_pos, fit_meta = auto_fit(
            cfg,
            renderer,
            start_pos,
            target,
            scene.analysis.radius,
            warning_messages,
        )
        c2w = look_at(fitted_pos, target, cfg.camera.initial_view.roll_deg)
        return BasePose(
            c2w,
            cfg.camera.fov_y_deg_at_1x,
            "object_auto",
            None,
            scene.analysis.radius,
            math.nan,
            target_override=target,
        ), {"strategy": "object", "object_auto_fit": fit_meta}
    return initialize_interior(cfg, scene, renderer), {
        "strategy": "interior",
        "automatic_position_change": False,
    }


def run(cfg: GeneratorConfig) -> dict:
    prepare_output(cfg)
    save_resolved_config(cfg)
    warning_messages: list[str] = []
    print(f"Loading and analyzing scene: {cfg.input.ply_path}")
    scene = GaussianScene(cfg.input.ply_path, cfg)
    print(
        f"Effective Gaussians: {scene.analysis.effective_gaussian_count:,} "
        f"/ {scene.analysis.original_gaussian_count:,}"
    )
    gaussians = scene.load_tensors()
    renderer = GaussianRenderer(gaussians, cfg)
    schedule = build_schedule(cfg)
    base, fit_meta = resolve_base_pose(cfg, scene, renderer, warning_messages)
    base, cameras, preview_metrics, motion_scale = precheck_trajectory(
        cfg, scene, renderer, schedule, base, warning_messages
    )
    cfg.camera.initialization.resolved_pose = resolved_pose_config(base)
    cfg.camera.fov_y_deg_at_1x = base.fov_y_deg_at_1x
    save_resolved_config(cfg)

    writer = FFmpegWriter(cfg.output.directory / cfg.output.video_filename, cfg.video)
    frame_meta = []
    camera_meta = []
    ext = cfg.output.frame_format
    try:
        iterator = zip(schedule, cameras, preview_metrics)
        for s, camera, preview_cov in tqdm(
            iterator, total=len(schedule), desc="Rendering video", unit="frame"
        ):
            rgb = renderer.render(camera)
            encoded, linear_processed, color_meta = process_frame(rgb, s, cfg)
            u8 = np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)
            writer.append(u8)
            alpha = renderer.render(camera, alpha_only=True) if cfg.output.save_alpha_masks else None
            if cfg.output.save_processed_frames:
                _save_image(
                    cfg.output.directory / "frames" / f"frame_{s.frame_index:06d}.{ext}", encoded
                )
            if cfg.output.save_linear_frames:
                np.save(
                    cfg.output.directory / "linear_frames" / f"frame_{s.frame_index:06d}.npy",
                    linear_processed.astype(np.float16),
                )
            if alpha is not None:
                _save_image(
                    cfg.output.directory / "alpha_masks" / f"frame_{s.frame_index:06d}.png",
                    np.repeat(alpha[..., None], 3, axis=2),
                )
            item = {
                "frame_index": s.frame_index,
                "timestamp_seconds": s.frame_index / cfg.video.fps,
                "lens_name": s.lens_name,
                "lens_local_frame_index": s.local_index,
                "lens_frame_count": s.lens_frame_count,
                "zoom_ratio": s.zoom_ratio,
                "fov_x": camera.fov_x,
                "fov_y": camera.fov_y,
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "camera_to_world": camera.c2w.tolist(),
                "world_to_camera": camera.w2c.tolist(),
                "camera_position": camera.position.tolist(),
                "look_at_target": camera.target.tolist(),
                "camera_center_offset": camera.camera_center_offset.tolist(),
                "near": camera.near,
                "far": camera.far,
                "coverage_ratio": preview_cov[0],
                "largest_background_component_ratio": preview_cov[1],
                **color_meta,
            }
            frame_meta.append(item)
            camera_meta.append({
                k: item[k] for k in (
                    "frame_index", "timestamp_seconds", "fx", "fy", "cx", "cy", "fov_x", "fov_y",
                    "camera_to_world", "world_to_camera", "camera_position", "look_at_target",
                    "near", "far",
                )
            })
        writer.close()
    except BaseException:
        writer.abort()
        raise

    probe = probe_video(cfg.output.directory / cfg.output.video_filename)
    expected = {
        "frame_count": cfg.video.total_frames,
        "width": cfg.video.width,
        "height": cfg.video.height,
    }
    for key, value in expected.items():
        if probe[key] != value:
            raise RuntimeError(f"encoded video {key} mismatch: expected {value}, got {probe[key]}")
    if not math.isclose(probe["fps"], cfg.video.fps, rel_tol=1e-3, abs_tol=1e-3):
        raise RuntimeError(
            f"encoded video FPS mismatch: expected {cfg.video.fps}, got {probe['fps']}"
        )

    counts = allocate_frame_counts(cfg)
    lens_ranges = []
    cursor = 0
    for lens, count in zip(cfg.zoom.lenses, counts):
        state = temporal_state(lens, count - 1, count, cfg)
        jump_local = state["jump_local_index"]
        lens_ranges.append({
            "name": lens.name,
            "global_frame_start": cursor,
            "global_frame_end": cursor + count - 1,
            "frame_count": count,
            "jump_local_frame": jump_local,
            "jump_global_frame": cursor + jump_local,
            "jump_ev": state["jump_ev"],
            "final_exposure_gain": 2.0 ** state["jump_ev"],
        })
        cursor += count
    coverage = np.array([x[0] for x in preview_metrics])
    summary = {
        "input_ply_path": str(cfg.input.ply_path),
        "scene": scene.analysis.as_dict(),
        "scene_root_transform": cfg.scene.root_transform.model_dump(mode="json"),
        "initialization": fit_meta,
        "auto_fit": fit_meta.get("object_auto_fit"),
        "base_pose": {
            "source": base.source,
            "camera_to_world": base.c2w.tolist(),
            "world_to_camera": np.linalg.inv(base.c2w).tolist(),
            "position": base.position.tolist(),
            "look_direction": base.c2w[:3, 2].tolist(),
            "fov_y_deg_at_1x": base.fov_y_deg_at_1x,
            "camera_inside_robust_aabb": bool(
                np.all(base.position >= scene.analysis.aabb_min)
                and np.all(base.position <= scene.analysis.aabb_max)
            ),
            "distance_to_scene_center": float(
                np.linalg.norm(base.position - scene.analysis.center)
            ),
            "distance_to_nearest_effective_gaussian": base.clearance,
        },
        "final_camera_distance": float(
            np.linalg.norm(base.position - scene.analysis.center)
        ),
        "trajectory_motion_scale": motion_scale,
        "coverage_ratio": {
            "min": float(coverage.min()),
            "mean": float(coverage.mean()),
            "max": float(coverage.max()),
        },
        "video": {
            **probe,
            "duration_seconds": cfg.video.total_frames / cfg.video.fps,
            "filename": cfg.output.video_filename,
        },
        "lenses": lens_ranges,
        "seed": cfg.camera.motion.seed,
        "warnings": warning_messages,
    }
    if cfg.output.save_metadata:
        _json_dump(cfg.output.directory / "camera_trajectory.json", camera_meta)
        _json_dump(cfg.output.directory / "frame_metadata.json", frame_meta)
        _json_dump(cfg.output.directory / "generation_summary.json", summary)
    print(f"Done: {cfg.output.directory / cfg.output.video_filename}")
    return summary
