from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .camera import (
    CameraFrame,
    direction_from_yaw_pitch,
    intrinsics,
    look_at,
    pose_from_yaw_pitch,
)
from .config import GeneratorConfig, ResolvedPoseConfig
from .renderer import GaussianRenderer
from .scene import GaussianScene


class AlignmentSelectionRequired(RuntimeError):
    def __init__(self, message: str, report_path: Path):
        super().__init__(message)
        self.report_path = report_path


@dataclass
class Candidate:
    position: np.ndarray
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    fov_y_reference_deg: float
    c2w: np.ndarray
    loss: float = math.inf
    coverage_ratio: float = 0.0
    largest_background_ratio: float = 1.0
    clearance: float = 0.0
    components: dict | None = None
    render_rgb: np.ndarray | None = None

    def as_dict(self, index: int, scene: GaussianScene) -> dict:
        look = self.c2w[:3, 2]
        a = scene.analysis
        return {
            "candidate_index": index,
            "score": self.loss,
            "reference_similarity_score": 1.0 / (1.0 + self.loss),
            "position": self.position.tolist(),
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "fov_y_reference_deg": self.fov_y_reference_deg,
            "camera_to_world": self.c2w.tolist(),
            "look_direction": look.tolist(),
            "coverage_ratio": self.coverage_ratio,
            "largest_background_component_ratio": self.largest_background_ratio,
            "distance_to_nearest_effective_gaussian": self.clearance,
            "camera_inside_robust_aabb": bool(
                np.all(self.position >= a.aabb_min) and np.all(self.position <= a.aabb_max)
            ),
            "loss_components": self.components or {},
        }


@dataclass
class BasePose:
    c2w: np.ndarray
    fov_y_deg_at_1x: float
    source: str
    allowed_aabb: tuple[np.ndarray, np.ndarray] | None
    local_radius: float
    clearance: float
    candidate_index: int | None = None
    alignment: dict | None = None
    target_override: np.ndarray | None = None

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def target(self) -> np.ndarray:
        if self.target_override is not None:
            return self.target_override
        return self.position + self.c2w[:3, 2] * self.local_radius


def _write_json(path: Path, value: dict | list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    image = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save alignment image: {path}")


def coverage_metrics(alpha: np.ndarray, threshold: float) -> tuple[float, float]:
    foreground = alpha > threshold
    coverage = float(foreground.mean())
    labels, count = ndimage.label(
        ~foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if not count:
        return coverage, 0.0
    sizes = np.bincount(labels.ravel())
    largest = float(sizes[1:].max() / alpha.size) if len(sizes) > 1 else 0.0
    return coverage, largest


def _camera_frame(
    cfg: GeneratorConfig,
    c2w: np.ndarray,
    fov_y_deg: float,
    width: int,
    height: int,
    near: float,
    far: float,
) -> CameraFrame:
    fy = (cfg.video.height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    fx = fy
    dx, dy = cfg.camera.principal_point_offset_px
    cx = cfg.video.width / 2.0 + dx
    cy = cfg.video.height / 2.0 + dy
    fov_x = 2.0 * math.atan(cfg.video.width / (2.0 * fx))
    position = c2w[:3, 3].copy()
    target = position + c2w[:3, 2]
    return CameraFrame(
        position,
        target,
        c2w,
        np.linalg.inv(c2w),
        fx,
        fy,
        cx,
        cy,
        fov_x,
        math.radians(fov_y_deg),
        near,
        far,
        np.zeros(3),
    )


def _structural_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    # Global contrast normalization keeps room layout while suppressing exposure/gamma shifts.
    normalized = (gray - float(np.mean(gray))) / (float(np.std(gray)) + 1e-5)
    normalized = np.clip(normalized, -2.5, 2.5) / 5.0 + 0.5
    normalized = cv2.GaussianBlur(normalized, (0, 0), 0.7)
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    gradient /= float(np.quantile(gradient, 0.95)) + 1e-5
    return np.clip(normalized, 0.0, 1.0), np.clip(gradient, 0.0, 1.0)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    aa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    bb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    value = ((2 * mu_a * mu_b + c1) * (2 * ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (aa + bb + c2)
    )
    return float(np.clip(np.mean(value), -1.0, 1.0))


def structural_loss(
    rendered: np.ndarray, reference_features: tuple[np.ndarray, np.ndarray], cfg: GeneratorConfig
) -> tuple[float, dict]:
    gray, grad = _structural_features(rendered)
    ref_gray, ref_grad = reference_features
    gray_loss = float(np.mean(np.abs(gray - ref_gray)))
    gradient_loss = float(np.mean(np.abs(grad - ref_grad)))
    ssim_loss = 1.0 - _ssim(gray, ref_gray)
    weights = cfg.camera.initialization.reference_match.loss
    total = (
        weights.grayscale_weight * gray_loss
        + weights.gradient_weight * gradient_loss
        + weights.ssim_weight * ssim_loss
    )
    return total, {
        "grayscale": gray_loss,
        "gradient": gradient_loss,
        "one_minus_ssim": ssim_loss,
        "weighted_total": total,
    }


def _auto_trim_border(image: np.ndarray) -> tuple[np.ndarray, list[int], bool]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    row_std = gray.std(axis=1)
    row_mean = gray.mean(axis=1)
    col_std = gray.std(axis=0)
    col_mean = gray.mean(axis=0)
    uniform_rows = (row_std < 2.0) & ((row_mean < 8) | (row_mean > 247))
    uniform_cols = (col_std < 2.0) & ((col_mean < 8) | (col_mean > 247))
    top = 0
    while top < h // 4 and uniform_rows[top]:
        top += 1
    bottom = h
    while bottom > 3 * h // 4 and uniform_rows[bottom - 1]:
        bottom -= 1
    left = 0
    while left < w // 4 and uniform_cols[left]:
        left += 1
    right = w
    while right > 3 * w // 4 and uniform_cols[right - 1]:
        right -= 1
    changed = any((top, h - bottom, left, w - right))
    return image[top:bottom, left:right], [left, top, right - left, bottom - top], changed


def preprocess_reference(
    path: Path,
    crop_xywh: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read reference image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]
    if crop_xywh is not None:
        x, y, w, h = crop_xywh
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > original_w or y + h > original_h:
            raise ValueError(f"invalid reference crop_xywh for {path}")
        rgb = rgb[y:y + h, x:x + w]
        used_crop = [x, y, w, h]
        auto_border = False
    else:
        rgb, used_crop, auto_border = _auto_trim_border(rgb)
    h, w = rgb.shape[:2]
    target_aspect = width / height
    source_aspect = w / h
    if source_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x = (w - new_w) // 2
        rgb = rgb[:, x:x + new_w]
        aspect_operation = {"mode": "center_crop_width", "offset": x, "size": [new_w, h]}
    elif source_aspect < target_aspect:
        new_h = int(round(w / target_aspect))
        y = (h - new_h) // 2
        rgb = rgb[y:y + new_h]
        aspect_operation = {"mode": "center_crop_height", "offset": y, "size": [w, new_h]}
    else:
        aspect_operation = {"mode": "none", "size": [w, h]}
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    result = resized.astype(np.float32) / 255.0
    return result, {
        "image_path": str(path),
        "original_size": [original_w, original_h],
        "used_crop_xywh": used_crop,
        "auto_border_removed": auto_border,
        "aspect_operation": aspect_operation,
        "processed_size": [width, height],
        "resampling": "area; no non-uniform stretch before aspect crop",
    }


class InteriorSpace:
    def __init__(self, cfg: GeneratorConfig, scene: GaussianScene):
        self.cfg = cfg
        self.scene = scene
        self.icfg = cfg.camera.initialization.interior
        threshold = max(0.10, cfg.scene_analysis.opacity_threshold)
        self.points = scene.effective_position_sample(300_000, threshold)
        if len(self.points) < 100:
            self.points = scene.effective_position_sample(300_000)
        if not len(self.points):
            raise RuntimeError("no effective points available for interior initialization")
        self.tree = cKDTree(self.points)
        lo_q, hi_q = self.icfg.position_quantiles
        self.lo = np.quantile(self.points, lo_q, axis=0)
        self.hi = np.quantile(self.points, hi_q, axis=0)
        self.radius = scene.analysis.radius
        self.min_clearance = self.icfg.minimum_clearance_radius_ratio * self.radius
        self.max_clearance = self.icfg.maximum_clearance_radius_ratio * self.radius

    def clearance(self, position: np.ndarray) -> float:
        return float(self.tree.query(position, k=1, workers=-1)[0])

    def local_radius(self, position: np.ndarray) -> float:
        k = min(4096, len(self.points))
        distances = np.asarray(self.tree.query(position, k=k, workers=-1)[0])
        return max(self.radius * 0.05, float(np.quantile(distances, 0.95)))

    def valid(self, position: np.ndarray) -> tuple[bool, float]:
        inside = bool(np.all(position >= self.lo) and np.all(position <= self.hi))
        clearance = self.clearance(position)
        return inside and self.min_clearance <= clearance <= self.max_clearance, clearance

    def candidate_positions(self) -> list[tuple[np.ndarray, float]]:
        count = self.icfg.candidate_position_count
        seed = self.cfg.camera.initialization.reference_match.seed_pose
        rng = np.random.default_rng(self.cfg.camera.motion.seed)
        candidates: list[np.ndarray] = []
        if seed.position is not None:
            center = np.asarray(seed.position, dtype=np.float64)
            candidates.append(np.clip(center, self.lo, self.hi))
            radius = self.cfg.camera.initialization.reference_match.search_radius_ratio * self.radius
            attempts = max(50, count * 20)
            for _ in range(attempts):
                if len(candidates) >= count:
                    break
                offset = rng.uniform(-radius, radius, 3)
                candidates.append(np.clip(center + offset, self.lo, self.hi))
        else:
            res = self.icfg.voxel_resolution
            extent = np.maximum(self.hi - self.lo, 1e-6)
            occupied = np.zeros((res, res, res), dtype=bool)
            indices = np.floor((self.points - self.lo) / extent * res).astype(int)
            indices = np.clip(indices, 0, res - 1)
            occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True
            voxel_size = extent / res
            clearance_grid = ndimage.distance_transform_edt(
                ~occupied, sampling=tuple(voxel_size)
            )
            eligible = np.argwhere(
                (clearance_grid >= self.min_clearance)
                & (clearance_grid <= self.max_clearance)
            )
            if len(eligible):
                order = rng.permutation(len(eligible))
                for idx in eligible[order[: min(len(eligible), count * 8)]]:
                    jitter = rng.uniform(0.15, 0.85, 3)
                    candidates.append(self.lo + (idx + jitter) / res * extent)
                    if len(candidates) >= count:
                        break
        accepted: list[tuple[np.ndarray, float]] = []
        for position in candidates:
            valid, clearance = self.valid(position)
            if valid:
                accepted.append((np.asarray(position, dtype=np.float64), clearance))
        attempts = 0
        while len(accepted) < count and attempts < count * 100:
            position = rng.uniform(self.lo, self.hi)
            valid, clearance = self.valid(position)
            if valid:
                accepted.append((position, clearance))
            attempts += 1
        if not accepted:
            raise RuntimeError(
                "interior free-space search found no candidate with configured clearance; "
                "adjust clearance ratios or root transform"
            )
        return accepted[:count]


def _fov_at_zoom(fov_y_1x_deg: float, zoom: float) -> float:
    return math.degrees(
        2.0 * math.atan(math.tan(math.radians(fov_y_1x_deg) / 2.0) / zoom)
    )


def _fov_1x_from_reference(fov_y_reference_deg: float, zoom: float) -> float:
    return math.degrees(
        2.0 * math.atan(zoom * math.tan(math.radians(fov_y_reference_deg) / 2.0))
    )


def _candidate_orientations(
    cfg: GeneratorConfig, position: np.ndarray, clearance: float
) -> list[Candidate]:
    init = cfg.camera.initialization
    match = init.reference_match
    count = init.interior.directions_per_position
    up = np.asarray(init.interior.world_up, dtype=np.float64)
    seed = match.seed_pose
    if seed.yaw_deg is not None:
        # The supplied seed must itself be evaluated. An even-sized linspace
        # otherwise straddles zero and can omit the one pose the user knows.
        if count == 1:
            yaw_offsets = np.zeros(1, dtype=np.float64)
        else:
            yaw_offsets = np.linspace(
                -match.yaw_search_range_deg / 2,
                match.yaw_search_range_deg / 2,
                count,
                endpoint=True,
            )
            yaw_offsets[int(np.argmin(np.abs(yaw_offsets)))] = 0.0
        yaws = seed.yaw_deg + yaw_offsets
    else:
        yaws = np.linspace(-180.0, 180.0, count, endpoint=False)
    if seed.pitch_deg is not None:
        center_pitch = seed.pitch_deg
    else:
        center_pitch = 0.0
    pitch_pattern = np.array([0.0, -0.5, 0.5, -1.0, 1.0])
    pitches = center_pitch + pitch_pattern[np.arange(count) % len(pitch_pattern)] * (
        match.pitch_search_range_deg / 2
    )
    if seed.yaw_deg is not None and seed.pitch_deg is not None:
        seed_index = int(np.argmin(np.abs(yaws - seed.yaw_deg)))
        pitches[seed_index] = seed.pitch_deg
    # Keep FOV independent from direction in coarse search. It is swept during
    # local refinement so every yaw/pitch receives the same unbiased base FOV.
    fov_1x = float(seed.fov_y_deg or cfg.camera.fov_y_deg_at_1x)
    result = []
    for yaw, pitch in zip(yaws, pitches):
        fov_reference = _fov_at_zoom(
            fov_1x, init.reference.reference_zoom_ratio
        )
        c2w = pose_from_yaw_pitch(
            position, float(yaw), float(pitch), seed.roll_deg, up
        )
        result.append(
            Candidate(
                position.copy(),
                float(yaw),
                float(pitch),
                seed.roll_deg,
                fov_reference,
                c2w,
                clearance=clearance,
            )
        )
    return result


def _evaluate_reference_candidate(
    candidate: Candidate,
    references: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]],
    relative_rotations: list[tuple[float, float, float]],
    cfg: GeneratorConfig,
    renderer: GaussianRenderer,
    space: InteriorSpace,
) -> Candidate:
    match = cfg.camera.initialization.reference_match
    width, height = match.preview_width, match.preview_height
    local_radius = space.local_radius(candidate.position)
    near = max(
        cfg.camera.clipping.near_min,
        local_radius * cfg.camera.clipping.near_radius_ratio,
    )
    far = max(near * 10, local_radius * cfg.camera.clipping.far_radius_ratio)
    frame = _camera_frame(
        cfg, candidate.c2w, candidate.fov_y_reference_deg, width, height, near, far
    )
    alpha = renderer.render(frame, width, height, alpha_only=True)
    candidate.coverage_ratio, candidate.largest_background_ratio = coverage_metrics(
        alpha, cfg.camera.auto_fit.alpha_threshold
    )
    icfg = cfg.camera.initialization.interior
    coverage_penalty = 2.0 * max(
        0.0, icfg.minimum_coverage_ratio - candidate.coverage_ratio
    ) + 2.0 * max(
        0.0,
        candidate.largest_background_ratio - icfg.maximum_largest_background_ratio,
    )
    losses = []
    components = []
    first_render = None
    for ref_i, (_, ref_features) in enumerate(references):
        relative = relative_rotations[ref_i]
        c2w = candidate.c2w.copy()
        c2w[:3, :3] = c2w[:3, :3] @ Rotation.from_euler(
            "xyz", relative, degrees=True
        ).as_matrix()
        frame = _camera_frame(
            cfg, c2w, candidate.fov_y_reference_deg, width, height, near, far
        )
        rendered = renderer.render(frame, width, height)
        if first_render is None:
            first_render = rendered
        value, detail = structural_loss(rendered, ref_features, cfg)
        losses.append(value)
        components.append(detail)
    candidate.loss = float(np.mean(losses) + coverage_penalty)
    candidate.components = {
        "per_reference": components,
        "coverage_penalty": coverage_penalty,
        "structural_mean": float(np.mean(losses)),
    }
    candidate.render_rgb = first_render
    return candidate


def _evaluate_coverage_candidate(
    candidate: Candidate,
    cfg: GeneratorConfig,
    renderer: GaussianRenderer,
    space: InteriorSpace,
) -> Candidate:
    width = cfg.camera.auto_fit.preview_width
    height = cfg.camera.auto_fit.preview_height
    local_radius = space.local_radius(candidate.position)
    fov_1x = _fov_1x_from_reference(
        candidate.fov_y_reference_deg,
        cfg.camera.initialization.reference.reference_zoom_ratio,
    )
    wide_fov = _fov_at_zoom(fov_1x, cfg.zoom.lenses[0].zoom_min)
    near = max(cfg.camera.clipping.near_min, local_radius * cfg.camera.clipping.near_radius_ratio)
    far = max(near * 10, local_radius * cfg.camera.clipping.far_radius_ratio)
    frame = _camera_frame(cfg, candidate.c2w, wide_fov, width, height, near, far)
    alpha = renderer.render(frame, width, height, alpha_only=True)
    cov, bg = coverage_metrics(alpha, cfg.camera.auto_fit.alpha_threshold)
    candidate.coverage_ratio = cov
    candidate.largest_background_ratio = bg
    icfg = cfg.camera.initialization.interior
    candidate.loss = abs(1.0 - cov) + 2.0 * max(
        0.0, icfg.minimum_coverage_ratio - cov
    ) + 2.0 * max(0.0, bg - icfg.maximum_largest_background_ratio)
    candidate.components = {"coverage_objective": candidate.loss}
    return candidate


def _refine_candidates(
    candidates: list[Candidate],
    references: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]],
    relative_rotations: list[tuple[float, float, float]],
    cfg: GeneratorConfig,
    renderer: GaussianRenderer,
    space: InteriorSpace,
) -> list[Candidate]:
    match = cfg.camera.initialization.reference_match
    if not match.refine_iterations:
        return candidates
    rng = np.random.default_rng(cfg.camera.motion.seed + 991)
    up = np.asarray(cfg.camera.initialization.interior.world_up)
    refined = []
    for base_i, original in enumerate(
        tqdm(candidates, desc="Refining Top-K", unit="candidate")
    ):
        best = original
        if match.optimize_fov:
            fov_1x_values = [
                match.fov_y_range_deg[0],
                cfg.camera.fov_y_deg_at_1x,
                sum(match.fov_y_range_deg) / 2.0,
                match.fov_y_range_deg[1],
            ]
            for fov_1x in fov_1x_values:
                fov_ref = _fov_at_zoom(
                    fov_1x,
                    cfg.camera.initialization.reference.reference_zoom_ratio,
                )
                trial = Candidate(
                    best.position.copy(),
                    best.yaw_deg,
                    best.pitch_deg,
                    best.roll_deg,
                    fov_ref,
                    best.c2w.copy(),
                    clearance=best.clearance,
                )
                trial = _evaluate_reference_candidate(
                    trial, references, relative_rotations, cfg, renderer, space
                )
                if trial.loss < best.loss:
                    best = trial
        for iteration in range(match.refine_iterations):
            decay = 1.0 - iteration / max(match.refine_iterations, 1)
            position = best.position.copy()
            if match.optimize_position:
                radius = match.search_radius_ratio * space.radius * 0.12 * max(decay, 0.1)
                position += rng.uniform(-radius, radius, 3)
                position = np.clip(position, space.lo, space.hi)
            valid, clearance = space.valid(position)
            if not valid:
                continue
            yaw, pitch, roll = best.yaw_deg, best.pitch_deg, best.roll_deg
            if match.optimize_rotation:
                yaw += rng.normal(0, match.yaw_search_range_deg * 0.04 * max(decay, 0.1))
                pitch += rng.normal(0, match.pitch_search_range_deg * 0.04 * max(decay, 0.1))
                roll += rng.normal(0, match.roll_search_range_deg * 0.10 * max(decay, 0.1))
                roll = float(np.clip(roll, -match.roll_search_range_deg, match.roll_search_range_deg))
            fov = best.fov_y_reference_deg
            if match.optimize_fov:
                fov += rng.normal(0, 3.0 * max(decay, 0.1))
                ref_zoom = cfg.camera.initialization.reference.reference_zoom_ratio
                lo_ref = _fov_at_zoom(match.fov_y_range_deg[0], ref_zoom)
                hi_ref = _fov_at_zoom(match.fov_y_range_deg[1], ref_zoom)
                fov = float(np.clip(fov, min(lo_ref, hi_ref), max(lo_ref, hi_ref)))
            c2w = pose_from_yaw_pitch(position, yaw, pitch, roll, up)
            trial = Candidate(position, yaw, pitch, roll, fov, c2w, clearance=clearance)
            trial = _evaluate_reference_candidate(
                trial, references, relative_rotations, cfg, renderer, space
            )
            if trial.loss < best.loss:
                best = trial
        refined.append(best)
    return sorted(refined, key=lambda x: x.loss)


def _contact_sheet(candidates: list[Candidate], width: int, height: int) -> np.ndarray:
    cell_w, cell_h = width, height + 28
    cols = min(4, len(candidates))
    rows = math.ceil(len(candidates) / cols)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.float32)
    for i, candidate in enumerate(candidates):
        row, col = divmod(i, cols)
        render = candidate.render_rgb
        if render is None:
            continue
        sheet[row * cell_h:row * cell_h + height, col * cell_w:(col + 1) * cell_w] = render
        cv2.putText(
            sheet,
            f"#{i:02d} loss={candidate.loss:.4f}",
            (col * cell_w + 5, row * cell_h + height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (1.0, 1.0, 1.0),
            1,
            cv2.LINE_AA,
        )
    return sheet


def _diagnostics(
    pose: BasePose,
    scene: GaussianScene,
    space: InteriorSpace | None,
    coverage: float | None,
    background: float | None,
    similarity: float | None,
) -> dict:
    p = pose.position
    a = scene.analysis
    nearest = space.clearance(p) if space is not None else None
    near = max(
        pose.local_radius * scene.cfg.camera.clipping.near_radius_ratio,
        scene.cfg.camera.clipping.near_min,
    )
    far = max(near * 10, pose.local_radius * scene.cfg.camera.clipping.far_radius_ratio)
    return {
        "scene_center": a.center.tolist(),
        "robust_scene_radius": a.radius,
        "robust_aabb": {"min": a.aabb_min.tolist(), "max": a.aabb_max.tolist()},
        "camera_position": p.tolist(),
        "camera_inside_robust_aabb": bool(
            np.all(p >= a.aabb_min) and np.all(p <= a.aabb_max)
        ),
        "distance_to_scene_center": float(np.linalg.norm(p - a.center)),
        "distance_to_nearest_effective_gaussian": nearest,
        "look_direction": pose.c2w[:3, 2].tolist(),
        "fov_y_deg_at_1x": pose.fov_y_deg_at_1x,
        "near": near,
        "far": far,
        "coverage_ratio": coverage,
        "largest_background_component_ratio": background,
        "reference_similarity_score": similarity,
    }


def align_reference(
    cfg: GeneratorConfig,
    scene: GaussianScene,
    renderer: GaussianRenderer,
    output_directory: Path,
) -> BasePose:
    match = cfg.camera.initialization.reference_match
    reference_cfg = cfg.camera.initialization.reference
    align_dir = output_directory / "camera_alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    references = []
    preprocess_meta = []
    for i, path in enumerate(reference_cfg.paths()):
        image, meta = preprocess_reference(
            path, reference_cfg.crop_xywh, match.preview_width, match.preview_height
        )
        references.append((image, _structural_features(image)))
        preprocess_meta.append(meta)
        _save_rgb(align_dir / f"reference_processed_{i:02d}.png", image)
        if i == 0:
            _save_rgb(align_dir / "reference_processed.png", image)
    relative = reference_cfg.relative_rotation_euler_deg or [
        (0.0, 0.0, 0.0)
    ] * len(references)
    space = InteriorSpace(cfg, scene)

    # Reproduce the previous object-style exterior initialization for diagnostics.
    old_position = scene.analysis.center + np.array([2.0 * scene.analysis.radius, 0, 0])
    old_c2w = look_at(old_position, scene.analysis.center)
    old_fov = _fov_at_zoom(
        cfg.camera.fov_y_deg_at_1x, reference_cfg.reference_zoom_ratio
    )
    old_local = scene.analysis.radius
    old_near = max(cfg.camera.clipping.near_min, old_local * cfg.camera.clipping.near_radius_ratio)
    old_far = max(old_near * 10, old_local * cfg.camera.clipping.far_radius_ratio)
    old_frame = _camera_frame(
        cfg, old_c2w, old_fov, match.preview_width, match.preview_height, old_near, old_far
    )
    old_render = renderer.render(old_frame, match.preview_width, match.preview_height)
    old_alpha = renderer.render(
        old_frame, match.preview_width, match.preview_height, alpha_only=True
    )
    old_cov, old_bg = coverage_metrics(old_alpha, cfg.camera.auto_fit.alpha_threshold)
    old_loss, old_components = structural_loss(old_render, references[0][1], cfg)
    _save_rgb(align_dir / "current_before.png", old_render)

    positions = space.candidate_positions()
    coarse: list[Candidate] = []
    total = len(positions) * cfg.camera.initialization.interior.directions_per_position
    progress = tqdm(total=total, desc="Interior reference coarse search", unit="view")
    for position, clearance in positions:
        for candidate in _candidate_orientations(cfg, position, clearance):
            candidate = _evaluate_reference_candidate(
                candidate, references, relative, cfg, renderer, space
            )
            coarse.append(candidate)
            progress.update(1)
    progress.close()
    coarse.sort(key=lambda x: x.loss)
    top = coarse[: min(match.top_k, len(coarse))]
    top = _refine_candidates(top, references, relative, cfg, renderer, space)
    for i, candidate in enumerate(top):
        _save_rgb(align_dir / f"candidate_{i:02d}.png", candidate.render_rgb)
    _save_rgb(
        align_dir / "contact_sheet.png",
        _contact_sheet(top, match.preview_width, match.preview_height),
    )
    best = top[0]
    reference = references[0][0]
    aligned = np.clip(best.render_rgb, 0, 1)
    _save_rgb(align_dir / "best_aligned.png", aligned)
    _save_rgb(align_dir / "overlay_reference_aligned.png", 0.5 * reference + 0.5 * aligned)
    difference = np.mean(np.abs(reference - aligned), axis=2)
    difference_rgb = cv2.applyColorMap(
        np.clip(difference * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    _save_rgb(
        align_dir / "difference_map.png",
        cv2.cvtColor(difference_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
    )
    pose_dicts = [x.as_dict(i, scene) for i, x in enumerate(top)]
    _write_json(align_dir / "candidate_poses.json", pose_dicts)

    score_gap = top[1].loss - top[0].loss if len(top) > 1 else math.inf
    best_similarity = 1.0 / (1.0 + top[0].loss)
    ambiguous_by_score_gap = (
        len(top) > 1 and score_gap <= match.ambiguity_score_margin
    )
    low_confidence = best_similarity < match.minimum_similarity_score
    ambiguous = ambiguous_by_score_gap or low_confidence
    selected = match.selected_candidate_index
    if selected is not None:
        if selected >= len(top):
            raise ValueError(
                f"selected_candidate_index {selected} is outside generated Top-K ({len(top)})"
            )
        best = top[selected]
        ambiguous = False
    local_radius = space.local_radius(best.position)
    fov_1x = _fov_1x_from_reference(
        best.fov_y_reference_deg, reference_cfg.reference_zoom_ratio
    )
    base = BasePose(
        best.c2w,
        fov_1x,
        "reference_image",
        (space.lo, space.hi),
        local_radius,
        best.clearance,
        selected if selected is not None else 0,
    )
    report = {
        "coordinate_convention": {
            "vectors": "column vectors",
            "camera_matrix": "camera-to-world",
            "camera_axes": "OpenCV/COLMAP: +x right, +y down, +z forward",
            "image_origin": "top-left",
            "root_transform": "p_world = translation + scale * R_euler * p_ply",
            "root_euler_order": cfg.scene.root_transform.rotation_order,
        },
        "reference_preprocessing": preprocess_meta,
        "before": {
            **_diagnostics(
                BasePose(old_c2w, cfg.camera.fov_y_deg_at_1x, "legacy_object", None,
                         old_local, math.nan),
                scene, space, old_cov, old_bg, 1.0 / (1.0 + old_loss),
            ),
            "reference_loss": old_loss,
            "loss_components": old_components,
        },
        "best": _diagnostics(
            base,
            scene,
            space,
            best.coverage_ratio,
            best.largest_background_ratio,
            1.0 / (1.0 + best.loss),
        ),
        "best_reference_loss": best.loss,
        "best_similarity_before_manual_selection": best_similarity,
        "score_gap_to_second": score_gap,
        "ambiguity_score_margin": match.ambiguity_score_margin,
        "minimum_similarity_score": match.minimum_similarity_score,
        "ambiguous_by_score_gap": ambiguous_by_score_gap,
        "low_confidence": low_confidence,
        "ambiguous": ambiguous,
        "selected_candidate_index": selected,
        "candidate_count_evaluated": len(coarse),
        "top_k": len(top),
    }
    base.alignment = report
    _write_json(align_dir / "alignment_report.json", report)
    if ambiguous and selected is None:
        raise AlignmentSelectionRequired(
            "reference alignment is ambiguous or below the configured similarity threshold; "
            "inspect contact_sheet.png and set "
            "camera.initialization.reference_match.selected_candidate_index",
            align_dir / "alignment_report.json",
        )
    return base


def initialize_interior(
    cfg: GeneratorConfig, scene: GaussianScene, renderer: GaussianRenderer
) -> BasePose:
    space = InteriorSpace(cfg, scene)
    candidates = []
    for position, clearance in tqdm(
        space.candidate_positions(), desc="Interior position search", unit="position"
    ):
        for candidate in _candidate_orientations(cfg, position, clearance):
            candidates.append(_evaluate_coverage_candidate(candidate, cfg, renderer, space))
    candidates.sort(key=lambda x: x.loss)
    best = candidates[0]
    local_radius = space.local_radius(best.position)
    fov_1x = _fov_1x_from_reference(
        best.fov_y_reference_deg,
        cfg.camera.initialization.reference.reference_zoom_ratio,
    )
    return BasePose(
        best.c2w,
        fov_1x,
        "interior_auto",
        (space.lo, space.hi),
        local_radius,
        best.clearance,
    )


def initialize_manual(cfg: GeneratorConfig, scene: GaussianScene) -> BasePose:
    pose = cfg.camera.initialization.manual_pose
    position = np.asarray(pose.position, dtype=np.float64)
    if pose.look_at is not None:
        c2w = look_at(
            position,
            np.asarray(pose.look_at, dtype=np.float64),
            pose.roll_deg,
            np.asarray(pose.up, dtype=np.float64),
        )
    else:
        c2w = pose_from_yaw_pitch(
            position,
            float(pose.yaw_deg),
            float(pose.pitch_deg),
            pose.roll_deg,
            np.asarray(pose.up, dtype=np.float64),
        )
    sample = scene.effective_position_sample(200_000)
    tree = cKDTree(sample)
    clearance = float(tree.query(position)[0])
    local_radius = max(scene.analysis.radius * 0.05, float(tree.query(position, k=min(4096, len(sample)))[0][-1]))
    return BasePose(c2w, float(pose.fov_y_deg), "manual", None, local_radius, clearance)


def infer_scene_type(cfg: GeneratorConfig, scene: GaussianScene) -> tuple[str, str | None]:
    points = scene.effective_position_sample(100_000)
    tree = cKDTree(points)
    center_clearance = float(tree.query(scene.analysis.center)[0]) / scene.analysis.radius
    extents = scene.analysis.aabb_max - scene.analysis.aabb_min
    aspect = float(extents.max() / max(extents.min(), 1e-9))
    if center_clearance < 0.01 and aspect < 2.0:
        return "object", None
    return "interior", (
        "scene_type=auto inferred interior; automatic classification is uncertain "
        f"(normalized center clearance={center_clearance:.4f}, AABB aspect={aspect:.3f})"
    )


def resolved_pose_config(base: BasePose) -> ResolvedPoseConfig:
    return ResolvedPoseConfig(
        position=tuple(float(x) for x in base.position),
        camera_to_world=tuple(
            tuple(float(x) for x in row) for row in base.c2w
        ),
        fov_y_deg_at_1x=base.fov_y_deg_at_1x,
        source=base.source,
        candidate_index=base.candidate_index,
    )
