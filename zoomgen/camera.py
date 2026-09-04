from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from .config import GeneratorConfig
from .zoom import FrameSchedule


@dataclass
class CameraFrame:
    position: np.ndarray
    target: np.ndarray
    c2w: np.ndarray
    w2c: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    fov_x: float
    fov_y: float
    near: float
    far: float
    camera_center_offset: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return v / n


def look_at(
    position: np.ndarray,
    target: np.ndarray,
    roll_deg: float = 0.0,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Build OpenCV/COLMAP camera-to-world: +x right, +y down, +z forward."""
    forward = normalize(target - position)
    up = normalize(
        np.asarray(world_up, dtype=np.float64)
        if world_up is not None
        else np.array([0.0, 0.0, 1.0])
    )
    if abs(float(np.dot(forward, up))) > 0.98:
        basis = np.eye(3)[int(np.argmin(np.abs(forward)))]
        up = normalize(basis - forward * np.dot(basis, forward))
    right = normalize(np.cross(forward, up))
    down = normalize(np.cross(forward, right))
    rot = np.stack([right, down, forward], axis=1)
    if roll_deg:
        rot = rot @ Rotation.from_rotvec(
            np.deg2rad(roll_deg) * np.array([0.0, 0.0, 1.0])
        ).as_matrix()
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot
    c2w[:3, 3] = position
    return c2w


def direction_from_yaw_pitch(
    yaw_deg: float, pitch_deg: float, world_up: np.ndarray
) -> np.ndarray:
    """Yaw around world_up; pitch positive toward world_up."""
    up = normalize(np.asarray(world_up, dtype=np.float64))
    seed = np.array([0.0, 0.0, -1.0])
    if abs(float(np.dot(seed, up))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    base = normalize(seed - up * np.dot(seed, up))
    right = normalize(np.cross(base, up))
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    horizontal = math.cos(yaw) * base + math.sin(yaw) * right
    return normalize(math.cos(pitch) * horizontal + math.sin(pitch) * up)


def pose_from_yaw_pitch(
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    world_up: np.ndarray,
) -> np.ndarray:
    direction = direction_from_yaw_pitch(yaw_deg, pitch_deg, world_up)
    return look_at(position, position + direction, roll_deg, world_up)


def default_view(center: np.ndarray, radius: float, azimuth_deg: float, elevation_deg: float,
                 distance: float) -> tuple[np.ndarray, np.ndarray]:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return center + distance * direction, center.copy()


def base_view(cfg: GeneratorConfig, center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    view = cfg.camera.initial_view
    if view.position is not None:
        pos = np.asarray(view.position, dtype=np.float64)
        target = np.asarray(view.look_at, dtype=np.float64)
        direction = normalize(pos - target)
        distance = float(np.linalg.norm(pos - target))
        # Preserve direction and target; auto-fit is allowed to change distance.
        return target + direction * distance, target, distance
    distance = max(2.0 * radius, 1e-6)
    pos, target = default_view(center, radius, view.azimuth_deg, view.elevation_deg, distance)
    return pos, target, distance


def intrinsics(
    cfg: GeneratorConfig,
    zoom: float,
    width: int | None = None,
    height: int | None = None,
    fov_y_deg_at_1x: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    w = width or cfg.video.width
    h = height or cfg.video.height
    # Scale focal length with resolution for preview renders.
    base_fov = (
        cfg.camera.fov_y_deg_at_1x
        if fov_y_deg_at_1x is None
        else fov_y_deg_at_1x
    )
    fy_1 = (h / 2.0) / math.tan(math.radians(base_fov) / 2.0)
    fy = zoom * fy_1
    fx = fy  # square pixels
    sx = w / cfg.video.width
    sy = h / cfg.video.height
    dx, dy = cfg.camera.principal_point_offset_px
    cx = w / 2.0 + dx * sx
    cy = h / 2.0 + dy * sy
    fov_x = 2.0 * math.atan(w / (2.0 * fx))
    fov_y = 2.0 * math.atan(h / (2.0 * fy))
    return fx, fy, cx, cy, fov_x, fov_y


def _smooth_controls(cfg: GeneratorConfig, count: int) -> tuple[np.ndarray, np.ndarray]:
    motion = cfg.camera.motion
    duration = max((count - 1) / cfg.video.fps, 1.0 / cfg.video.fps)
    n_ctrl = max(4, int(math.ceil(duration / motion.control_point_interval_seconds)) + 3)
    ctrl_t = np.linspace(0.0, duration, n_ctrl)
    frame_t = np.linspace(0.0, duration, count)
    rng = np.random.default_rng(motion.seed)
    trans_ctrl = rng.uniform(-1.0, 1.0, size=(n_ctrl, 3))
    rot_ctrl = rng.uniform(-1.0, 1.0, size=(n_ctrl, 3))
    # Start at the unperturbed pose and remove endpoint drift.
    trans_ctrl[0] = rot_ctrl[0] = 0.0
    trans = CubicSpline(ctrl_t, trans_ctrl, bc_type="natural")(frame_t)
    rot = CubicSpline(ctrl_t, rot_ctrl, bc_type="natural")(frame_t)
    trans /= np.maximum(1.0, np.max(np.abs(trans), axis=0, keepdims=True))
    rot /= np.maximum(1.0, np.max(np.abs(rot), axis=0, keepdims=True))
    return trans, rot


def motion_values(cfg: GeneratorConfig, count: int) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.camera.motion.enabled:
        return np.zeros((count, 3)), np.zeros((count, 3))
    if cfg.camera.motion.type == "low_frequency_sine":
        t = np.linspace(0.0, 1.0, count)
        phases = np.random.default_rng(cfg.camera.motion.seed).uniform(0, 2 * np.pi, 6)
        trans = np.stack([np.sin(2 * np.pi * (i + 1) * t / 3 + phases[i]) for i in range(3)], 1)
        rot = np.stack([np.sin(2 * np.pi * (i + 1) * t / 4 + phases[i + 3]) for i in range(3)], 1)
        trans[0] = rot[0] = 0.0
        return trans, rot
    return _smooth_controls(cfg, count)


def build_camera_frames(
    cfg: GeneratorConfig,
    schedule: list[FrameSchedule],
    base_position: np.ndarray,
    target: np.ndarray,
    radius: float,
    motion_scale: float = 1.0,
    base_c2w: np.ndarray | None = None,
    fov_y_deg_at_1x: float | None = None,
    allowed_aabb: tuple[np.ndarray, np.ndarray] | None = None,
    clip_radius: float | None = None,
) -> list[CameraFrame]:
    count = len(schedule)
    trans_unit, rot_unit = motion_values(cfg, count)
    if base_c2w is None:
        base_c2w = look_at(base_position, target, cfg.camera.initial_view.roll_deg)
    else:
        base_c2w = np.asarray(base_c2w, dtype=np.float64).copy()
        base_position = base_c2w[:3, 3].copy()
    local_axes = base_c2w[:3, :3]
    trans_amp = (
        radius * np.asarray(cfg.camera.motion.translation_amplitude_ratio) * motion_scale
    )
    rot_amp = np.asarray(cfg.camera.motion.rotation_amplitude_deg) * motion_scale
    local_extent = radius if clip_radius is None else clip_radius
    near = max(
        cfg.camera.clipping.near_min,
        local_extent * cfg.camera.clipping.near_radius_ratio,
    )
    if cfg.camera.clipping.mode == "auto_local":
        far = max(near * 10.0, local_extent * cfg.camera.clipping.far_radius_ratio)
    else:
        far = max(
            radius * cfg.camera.clipping.far_radius_ratio,
            np.linalg.norm(base_position - target) + 2 * radius,
        )
    optical_distance = max(float(np.linalg.norm(target - base_position)), local_extent)
    frames: list[CameraFrame] = []
    for i, s in enumerate(schedule):
        if cfg.camera.motion.enabled:
            lens = cfg.zoom.lenses[s.lens_index]
            offset = np.asarray(lens.camera_center_offset_ratio, dtype=np.float64)
            if s.transition_from_lens is not None and s.camera_blend < 1.0:
                prev = np.asarray(
                    cfg.zoom.lenses[s.transition_from_lens].camera_center_offset_ratio
                )
                offset = prev * (1.0 - s.camera_blend) + offset * s.camera_blend
            local_translation = trans_unit[i] * trans_amp + offset * radius
            delta = Rotation.from_euler(
                "xyz", rot_unit[i] * rot_amp, degrees=True
            ).as_matrix()
        else:
            # Explicit guarantee: disabled motion means every frame has identical extrinsics.
            offset = np.zeros(3, dtype=np.float64)
            local_translation = np.zeros(3, dtype=np.float64)
            delta = np.eye(3)
        position = base_position + local_axes @ local_translation
        if allowed_aabb is not None:
            lo, hi = allowed_aabb
            position = np.clip(position, lo, hi)
        c2w = np.eye(4)
        c2w[:3, :3] = local_axes @ delta
        c2w[:3, 3] = position
        optical_target = position + c2w[:3, 2] * optical_distance
        fx, fy, cx, cy, fov_x, fov_y = intrinsics(
            cfg, s.zoom_ratio, fov_y_deg_at_1x=fov_y_deg_at_1x
        )
        frames.append(
            CameraFrame(
                position,
                optical_target,
                c2w,
                np.linalg.inv(c2w),
                fx,
                fy,
                cx,
                cy,
                fov_x,
                fov_y,
                near,
                far,
                offset * radius,
            )
        )
    return frames


def projection_matrix(frame: CameraFrame, width: int, height: int) -> np.ndarray:
    """Graphdeco perspective matrix, including an off-center principal point."""
    n, f = frame.near, frame.far
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 2.0 * frame.fx / width
    p[1, 1] = 2.0 * frame.fy / height
    p[0, 2] = 2.0 * frame.cx / width - 1.0
    p[1, 2] = 2.0 * frame.cy / height - 1.0
    p[2, 2] = f / (f - n)
    p[2, 3] = -(f * n) / (f - n)
    p[3, 2] = 1.0
    return p
