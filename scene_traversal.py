from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import Field, field_validator, model_validator
from scipy.spatial import cKDTree
from tqdm import tqdm

from zoomgen.camera import CameraFrame, pose_from_yaw_pitch
from zoomgen.config import (
    ClippingConfig,
    InputConfig,
    RenderConfig,
    SceneAnalysisConfig,
    SceneConfig,
    StrictModel,
)
from zoomgen.renderer import GaussianRenderer
from zoomgen.quality import screen_space_quality_metrics
from zoomgen.scene import GaussianScene


class TraversalOutputConfig(StrictModel):
    root_directory: Path
    scene_name: str = Field(min_length=1)
    position_label_format: str = "position_{index:04d}"
    lens_label_format: str = "lens_{index:04d}"
    image_filename: str = "image.png"
    metadata_filename: str = "camera.json"
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_paths(self):
        _safe_label(self.scene_name, "scene_name")
        _safe_label(self.position_label_format.format(index=0), "position_label_format")
        _safe_label(self.lens_label_format.format(index=0), "lens_label_format")
        if Path(self.image_filename).name != self.image_filename:
            raise ValueError("image_filename must be a filename, not a path")
        if Path(self.metadata_filename).name != self.metadata_filename:
            raise ValueError("metadata_filename must be a filename, not a path")
        if Path(self.image_filename).suffix.lower() != ".png":
            raise ValueError("image_filename must use the .png extension")
        if Path(self.metadata_filename).suffix.lower() != ".json":
            raise ValueError("metadata_filename must use the .json extension")
        return self


class TraversalImageConfig(StrictModel):
    gamma: float = Field(default=1.0, gt=0.0)
    png_compression: int = Field(default=3, ge=0, le=9)


class TraversalSeedPoseConfig(StrictModel):
    position: tuple[float, float, float] | None = None
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    fov_y_deg: float = Field(default=96.0, gt=1.0, lt=179.0)


class TraversalReferenceMatchConfig(StrictModel):
    preview_width: int = Field(default=120, gt=0)
    preview_height: int = Field(default=120, gt=0)
    seed_pose: TraversalSeedPoseConfig = Field(
        default_factory=TraversalSeedPoseConfig
    )

    search_radius_ratio: float = Field(default=0.15, gt=0.0)
    yaw_search_range_deg: tuple[float, float] = (-90.0, 90.0)
    pitch_search_range_deg: tuple[float, float] = (-12.5, 12.5)
    roll_search_range_deg: tuple[float, float] = (-2.5, 2.5)

    @field_validator("seed_pose", mode="before")
    @classmethod
    def default_seed_pose(cls, value):
        return {} if value is None else value

    @field_validator(
        "yaw_search_range_deg",
        "pitch_search_range_deg",
        "roll_search_range_deg",
        mode="before",
    )
    @classmethod
    def migrate_scalar_angle_range(cls, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            half_range = float(value) / 2.0
            return (-half_range, half_range)
        return value

    @model_validator(mode="after")
    def validate_ranges(self):
        for field_name in (
            "yaw_search_range_deg",
            "pitch_search_range_deg",
            "roll_search_range_deg",
        ):
            lower, upper = getattr(self, field_name)
            if not -180.0 <= lower <= upper <= 180.0:
                raise ValueError(
                    f"{field_name} must satisfy -180 <= lower <= upper <= 180"
                )
        pitch_lo = self.seed_pose.pitch_deg + self.pitch_search_range_deg[0]
        pitch_hi = self.seed_pose.pitch_deg + self.pitch_search_range_deg[1]
        if pitch_lo <= -89.0 or pitch_hi >= 89.0:
            raise ValueError(
                "seed pitch plus pitch search range must stay within (-89, 89)"
            )
        return self


class RandomQualitySamplingConfig(StrictModel):
    target_count_k: int = Field(default=10, ge=1)
    topiq_nr_threshold_l: float = Field(default=0.5, allow_inf_nan=False)
    device: str = Field(default="cuda", min_length=1)


class TraversalSamplingConfig(StrictModel):
    strategy: Literal["planned", "random"] = "planned"
    position_count_k: int = Field(default=10, ge=1)
    images_per_position_l: int = Field(default=24, ge=1)
    random_seed: int = 3407
    max_position_sampling_attempts: int = Field(default=20000, ge=1)
    minimum_clearance_radius_ratio: float = Field(default=0.005, gt=0.0)
    maximum_clearance_radius_ratio: float = Field(default=0.20, gt=0.0)
    minimum_position_separation_radius_ratio: float = Field(default=0.02, ge=0.0)
    random: RandomQualitySamplingConfig = Field(
        default_factory=RandomQualitySamplingConfig
    )

    @model_validator(mode="after")
    def validate_clearance(self):
        if self.minimum_clearance_radius_ratio >= self.maximum_clearance_radius_ratio:
            raise ValueError("minimum clearance must be smaller than maximum clearance")
        return self


class ObjectCoverageConfig(StrictModel):
    minimum_pixel_ratio: float = Field(default=0.85, ge=0.0, le=1.0)
    alpha_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    preview_width: int = Field(default=128, gt=0)
    preview_height: int = Field(default=128, gt=0)
    max_iterations: int = Field(default=8, ge=2)
    exterior_margin_radius_ratio: float = Field(default=0.01, ge=0.0)


class ObjectTraversalConfig(StrictModel):
    distance_range_radius_ratio: tuple[float, float] = (1.5, 2.5)
    azimuth_range_deg: tuple[float, float] = (-180.0, 180.0)
    elevation_range_deg: tuple[float, float] = (0.0, 90.0)
    center_yaw_jitter_range_deg: tuple[float, float] = (-4.0, 4.0)
    center_pitch_jitter_range_deg: tuple[float, float] = (-4.0, 4.0)
    roll_jitter_range_deg: tuple[float, float] = (-2.5, 2.5)
    random_candidate_multiplier: int = Field(default=50, ge=1)
    coverage: ObjectCoverageConfig = Field(default_factory=ObjectCoverageConfig)

    @model_validator(mode="after")
    def validate_ranges(self):
        distance_lo, distance_hi = self.distance_range_radius_ratio
        if not 1.0 < distance_lo <= distance_hi:
            raise ValueError(
                "object distance range must satisfy 1 < minimum <= maximum"
            )
        azimuth_lo, azimuth_hi = self.azimuth_range_deg
        if not azimuth_lo < azimuth_hi or azimuth_hi - azimuth_lo > 360.0:
            raise ValueError(
                "object azimuth range must be increasing and span at most 360 degrees"
            )
        elevation_lo, elevation_hi = self.elevation_range_deg
        if not 0.0 <= elevation_lo <= elevation_hi <= 90.0:
            raise ValueError(
                "object elevation range must stay within the upper hemisphere "
                "[0, 90] degrees"
            )
        for field_name in (
            "center_yaw_jitter_range_deg",
            "center_pitch_jitter_range_deg",
            "roll_jitter_range_deg",
        ):
            lower, upper = getattr(self, field_name)
            if not -180.0 <= lower <= upper <= 180.0:
                raise ValueError(
                    f"object {field_name} must satisfy -180 <= lower <= upper <= 180"
                )
        return self


class TraversalSceneConfig(SceneConfig):
    scene_type: Literal["interior", "object"]


class TraversalInitializationConfig(StrictModel):
    world_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    reference_match: TraversalReferenceMatchConfig
    traversal: TraversalSamplingConfig = TraversalSamplingConfig()
    object: ObjectTraversalConfig = ObjectTraversalConfig()
    clipping: ClippingConfig = ClippingConfig(mode="auto_local")

    @model_validator(mode="after")
    def validate_world_up(self):
        if sum(x * x for x in self.world_up) < 1e-18:
            raise ValueError("world_up must be non-zero")
        return self


class TraversalConfig(StrictModel):
    input: InputConfig
    scene: TraversalSceneConfig
    output: TraversalOutputConfig
    render: RenderConfig = RenderConfig()
    image: TraversalImageConfig = TraversalImageConfig()
    scene_analysis: SceneAnalysisConfig = SceneAnalysisConfig()
    initialization: TraversalInitializationConfig

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_scene_type(cls, value):
        if not isinstance(value, dict):
            return value
        initialization = value.get("initialization")
        if not isinstance(initialization, dict) or "scene_type" not in initialization:
            return value
        migrated = dict(value)
        migrated_initialization = dict(initialization)
        legacy_scene_type = migrated_initialization.pop("scene_type")
        migrated_scene = dict(migrated.get("scene") or {})
        configured_scene_type = migrated_scene.get("scene_type")
        if (
            configured_scene_type is not None
            and configured_scene_type != legacy_scene_type
        ):
            raise ValueError(
                "scene.scene_type conflicts with legacy initialization.scene_type"
            )
        migrated_scene["scene_type"] = legacy_scene_type
        migrated["scene"] = migrated_scene
        migrated["initialization"] = migrated_initialization
        return migrated

    @model_validator(mode="after")
    def validate_input(self):
        if not self.input.ply_path.is_file():
            raise ValueError(f"input PLY does not exist: {self.input.ply_path}")
        return self


@dataclass(frozen=True)
class PositionPlan:
    index: int
    label: str
    position: np.ndarray
    offset_from_seed: np.ndarray
    clearance: float
    seed_position_source: str
    look_at_target: np.ndarray | None = None


@dataclass(frozen=True)
class CapturePlan:
    position: PositionPlan
    lens_index: int
    lens_label: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    fov_y_deg: float


def _safe_label(value: str, field: str) -> str:
    if not value or value in (".", "..") or Path(value).name != value:
        raise ValueError(f"{field} must resolve to one safe path component")
    return value


def _read_yaml_mapping(path: str | Path, label: str) -> dict:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} config must contain a YAML mapping: {resolved}")
    return raw


def _validate_loaded_config(raw: dict, check_output: bool) -> TraversalConfig:
    cfg = TraversalConfig.model_validate(raw)
    scene_directory = cfg.output.root_directory / cfg.output.scene_name
    if (
        check_output
        and scene_directory.exists()
        and any(scene_directory.iterdir())
        and not cfg.output.overwrite
    ):
        raise ValueError(
            f"output scene directory is not empty: {scene_directory}; "
            "set output.overwrite=true to replace matching capture files"
        )
    return cfg


def load_traversal_config(
    path: str | Path, *, check_output: bool = True
) -> TraversalConfig:
    raw = _read_yaml_mapping(path, "combined")
    return _validate_loaded_config(raw, check_output)


def load_traversal_config_parts(
    scene_path: str | Path,
    camera_path: str | Path,
    color_path: str | Path,
    *,
    check_output: bool = True,
) -> TraversalConfig:
    specifications = (
        (
            "scene",
            scene_path,
            {"input", "scene", "output", "scene_analysis"},
            {"input", "output"},
        ),
        ("camera", camera_path, {"initialization"}, {"initialization"}),
        ("color", color_path, {"render", "image"}, {"render", "image"}),
    )
    merged: dict = {}
    for label, path, allowed, required in specifications:
        fragment = _read_yaml_mapping(path, label)
        unexpected = sorted(set(fragment) - allowed)
        if unexpected:
            raise ValueError(
                f"{label} config contains keys owned by another config: {unexpected}"
            )
        missing = sorted(required - set(fragment))
        if missing:
            raise ValueError(f"{label} config is missing required keys: {missing}")
        duplicates = sorted(set(merged) & set(fragment))
        if duplicates:
            raise ValueError(f"duplicate top-level config keys: {duplicates}")
        merged.update(fragment)
    return _validate_loaded_config(merged, check_output)


def _renderer_adapter(cfg: TraversalConfig) -> SimpleNamespace:
    return SimpleNamespace(
        scene=cfg.scene,
        scene_analysis=cfg.scene_analysis,
        render=cfg.render,
        video=SimpleNamespace(
            width=cfg.initialization.reference_match.preview_width,
            height=cfg.initialization.reference_match.preview_height,
        ),
    )


def _uniform_ball_offset(rng: np.random.Generator, radius: float) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= norm
    return direction * radius * float(rng.random() ** (1.0 / 3.0))


class _MinimumSeparationGrid:
    """Near-constant-time minimum-distance checks for accepted positions."""

    def __init__(self, minimum_separation: float):
        self.minimum_separation = minimum_separation
        self.cells: dict[tuple[int, int, int], list[np.ndarray]] = {}

    def add_if_separated(self, candidate: np.ndarray) -> bool:
        if self.minimum_separation <= 0.0:
            return True
        key_array = np.floor(candidate / self.minimum_separation).astype(np.int64)
        key = tuple(int(value) for value in key_array)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
                    for other in self.cells.get(neighbor, ()):
                        if float(np.linalg.norm(candidate - other)) < self.minimum_separation:
                            return False
        self.cells.setdefault(key, []).append(candidate.copy())
        return True


def _world_basis(
    world_up: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(world_up, dtype=np.float64)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-12:
        raise ValueError("world_up must be non-zero")
    up /= up_norm
    horizontal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if abs(float(np.dot(horizontal, up))) > 0.95:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    horizontal -= up * np.dot(horizontal, up)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-12:
        raise ValueError("could not construct an object traversal basis")
    horizontal /= horizontal_norm
    right = np.cross(horizontal, up)
    return horizontal, right, up



def _sample_object_positions(
    cfg: TraversalConfig, scene: GaussianScene
) -> list[PositionPlan]:
    match = cfg.initialization.reference_match
    sampling = cfg.initialization.traversal
    object_cfg = cfg.initialization.object
    center = np.asarray(scene.analysis.center, dtype=np.float64)
    radius = float(scene.analysis.radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("object traversal requires a finite positive scene radius")

    points = scene.effective_position_sample()
    if not len(points):
        raise RuntimeError("no effective points are available for object position sampling")
    tree = cKDTree(points)
    horizontal, right, up = _world_basis(cfg.initialization.world_up)
    distance_lo, distance_hi = object_cfg.distance_range_radius_ratio
    elevation_lo, elevation_hi = np.radians(object_cfg.elevation_range_deg)
    sin_elevation_lo = math.sin(float(elevation_lo))
    sin_elevation_hi = math.sin(float(elevation_hi))
    azimuth_lo, azimuth_hi = np.radians(object_cfg.azimuth_range_deg)
    minimum_separation = sampling.minimum_position_separation_radius_ratio * radius
    distance_min = distance_lo * radius
    distance_max = distance_hi * radius
    distance_tolerance = max(radius, 1.0) * 1e-12
    rng = np.random.default_rng(
        np.random.SeedSequence([sampling.random_seed, 0x0B1EC7])
    )
    configured_seed = match.seed_pose.position
    seed = (
        np.asarray(configured_seed, dtype=np.float64)
        if configured_seed is not None
        else None
    )
    seed_source = (
        "configured_object_exterior"
        if seed is not None
        else "automatic_object_shell"
    )
    accepted: list[PositionPlan] = []
    separation_grid = _MinimumSeparationGrid(minimum_separation)

    def add_candidate(candidate: np.ndarray) -> bool:
        nonlocal seed
        center_offset = candidate - center
        center_distance = float(np.linalg.norm(center_offset))
        height_above_center = float(np.dot(center_offset, up))
        if not (
            distance_min - distance_tolerance
            <= center_distance
            <= distance_max + distance_tolerance
        ):
            return False
        if height_above_center < -distance_tolerance:
            return False
        if not separation_grid.add_if_separated(candidate):
            return False
        if seed is None:
            seed = candidate.copy()
            match.seed_pose.position = tuple(float(value) for value in seed)
        index = len(accepted)
        label = _safe_label(
            cfg.output.position_label_format.format(index=index),
            "position_label_format",
        )
        clearance = float(tree.query(candidate, k=1, workers=-1)[0])
        accepted.append(
            PositionPlan(
                index,
                label,
                candidate.copy(),
                candidate - seed,
                clearance,
                seed_source,
                center.copy(),
            )
        )
        return True

    if seed is not None and not add_candidate(seed):
        raise ValueError(
            "object reference_match.seed_pose.position must lie within the "
            "configured upper-hemisphere shell"
        )

    if sampling.strategy == "planned":
        target_position_count = sampling.position_count_k
    else:
        # Coverage failures do not consume the expensive TOPIQ-NR budget. Keep
        # the valid positions found during the configured sampling attempts as
        # a cheap reserve, and visit a new direction before another lens at the
        # same position.
        target_position_count = sampling.max_position_sampling_attempts

    # Irrational increments give every prefix broad directional coverage while
    # retaining a deterministic random phase for each configured seed.
    phase = rng.random(3)
    increments = (0.6180339887498949, 0.7548776662466927, 0.5698402909980532)
    for attempt in range(sampling.max_position_sampling_attempts):
        if len(accepted) >= target_position_count:
            break
        azimuth_fraction = (phase[0] + attempt * increments[0]) % 1.0
        elevation_fraction = (phase[1] + attempt * increments[1]) % 1.0
        distance_fraction = (phase[2] + attempt * increments[2]) % 1.0
        azimuth = azimuth_lo + azimuth_fraction * (azimuth_hi - azimuth_lo)
        sin_elevation = (
            sin_elevation_lo
            + elevation_fraction * (sin_elevation_hi - sin_elevation_lo)
        )
        elevation = math.asin(float(sin_elevation))
        radial = (
            math.cos(elevation)
            * (
                math.cos(azimuth) * horizontal
                + math.sin(azimuth) * right
            )
            + math.sin(elevation) * up
        )
        distance_ratio = distance_lo + (
            distance_hi - distance_lo
        ) * distance_fraction**2
        add_candidate(center + distance_ratio * radius * radial)

    if sampling.strategy == "planned" and len(accepted) != sampling.position_count_k:
        raise RuntimeError(
            "object traversal could only sample "
            f"{len(accepted)}/{sampling.position_count_k} external positions after "
            f"{sampling.max_position_sampling_attempts} attempts; adjust object "
            "distance/elevation/azimuth ranges or minimum position separation"
        )
    if not accepted:
        raise RuntimeError(
            "random object traversal found no valid positions within max_position_sampling_attempts"
        )
    return accepted


def _automatic_seed_position(
    cfg: TraversalConfig,
    scene: GaussianScene,
    tree: cKDTree,
    lo: np.ndarray,
    hi: np.ndarray,
    search_radius: float,
    minimum_clearance: float,
    maximum_clearance: float,
) -> tuple[np.ndarray, int, int]:
    sampling = cfg.initialization.traversal
    midpoint = (lo + hi) / 2.0
    analysis_center = np.asarray(
        getattr(scene.analysis, "center", midpoint), dtype=np.float64
    )
    auto_rng = np.random.default_rng(
        np.random.SeedSequence([sampling.random_seed, 0xA570])
    )
    position_hint = (
        sampling.position_count_k
        if sampling.strategy == "planned"
        else min(32, sampling.max_position_sampling_attempts)
    )
    candidate_count = min(256, max(32, position_hint * 8))
    candidates = [np.clip(analysis_center, lo, hi)]
    if not np.allclose(candidates[0], midpoint):
        candidates.append(midpoint)
    remaining = candidate_count - len(candidates)
    if remaining > 0:
        candidates.extend(auto_rng.uniform(lo, hi, size=(remaining, 3)))
    candidate_array = np.asarray(candidates, dtype=np.float64)

    probe_count = min(512, max(128, position_hint * 16))
    directions = auto_rng.normal(size=(probe_count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1e-12)
    radii = search_radius * np.cbrt(auto_rng.random(probe_count))
    offsets = directions * radii[:, None]
    probes = candidate_array[:, None, :] + offsets[None, :, :]
    inside = np.all((probes >= lo) & (probes <= hi), axis=2)
    clearances = np.full(inside.shape, np.nan, dtype=np.float64)
    if np.any(inside):
        clearances[inside] = tree.query(probes[inside], k=1, workers=-1)[0]
    valid = (
        inside
        & (clearances >= minimum_clearance)
        & (clearances <= maximum_clearance)
    )
    valid_counts = valid.sum(axis=1)
    if int(valid_counts.max()) == 0:
        raise RuntimeError(
            "automatic seed search found no neighborhood with valid camera "
            "clearance; adjust clearance ratios, root transform, or robust AABB"
        )

    span = np.maximum(hi - lo, 1e-12)
    boundary_margin = np.min(
        np.minimum(candidate_array - lo, hi - candidate_array) / span,
        axis=1,
    )
    center_distance = np.linalg.norm(
        (candidate_array - midpoint) / span, axis=1
    )
    best_index = max(
        range(len(candidate_array)),
        key=lambda index: (
            int(valid_counts[index]),
            float(boundary_margin[index]),
            -float(center_distance[index]),
            -index,
        ),
    )
    return (
        candidate_array[best_index].copy(),
        int(valid_counts[best_index]),
        probe_count,
    )


def _sample_interior_positions(
    cfg: TraversalConfig, scene: GaussianScene
) -> list[PositionPlan]:
    match = cfg.initialization.reference_match
    sampling = cfg.initialization.traversal
    lo, hi = scene.analysis.aabb_min, scene.analysis.aabb_max
    configured_seed = match.seed_pose.position
    if configured_seed is not None:
        seed = np.asarray(configured_seed, dtype=np.float64)
        if not bool(np.all(seed >= lo) and np.all(seed <= hi)):
            raise ValueError(
                "reference_match.seed_pose.position is outside the transformed "
                "robust scene AABB"
            )
        seed_source = "configured"
    else:
        seed = None
        seed_source = "automatic_clearance_density"

    points = scene.effective_position_sample()
    if not len(points):
        raise RuntimeError("no effective points are available for interior position sampling")
    tree = cKDTree(points)
    radius = float(scene.analysis.radius)
    search_radius = match.search_radius_ratio * radius
    minimum_clearance = sampling.minimum_clearance_radius_ratio * radius
    maximum_clearance = sampling.maximum_clearance_radius_ratio * radius
    minimum_separation = sampling.minimum_position_separation_radius_ratio * radius
    if seed is None:
        seed, valid_probe_count, probe_count = _automatic_seed_position(
            cfg,
            scene,
            tree,
            lo,
            hi,
            search_radius,
            minimum_clearance,
            maximum_clearance,
        )
        match.seed_pose.position = tuple(float(value) for value in seed)
        print(
            "Auto-selected traversal seed "
            f"{seed.tolist()} ({valid_probe_count}/{probe_count} valid probes)"
        )

    rng = np.random.default_rng(sampling.random_seed)
    accepted: list[PositionPlan] = []
    separation_grid = _MinimumSeparationGrid(minimum_separation)

    for _ in range(sampling.max_position_sampling_attempts):
        if sampling.strategy == "planned" and len(accepted) >= sampling.position_count_k:
            break
        offset = _uniform_ball_offset(rng, search_radius)
        candidate = seed + offset
        if not bool(np.all(candidate >= lo) and np.all(candidate <= hi)):
            continue
        clearance = float(tree.query(candidate, k=1, workers=-1)[0])
        if not minimum_clearance <= clearance <= maximum_clearance:
            continue
        if not separation_grid.add_if_separated(candidate):
            continue
        index = len(accepted)
        label = _safe_label(
            cfg.output.position_label_format.format(index=index),
            "position_label_format",
        )
        accepted.append(
            PositionPlan(index, label, candidate, offset, clearance, seed_source)
        )

    if sampling.strategy == "planned" and len(accepted) != sampling.position_count_k:
        raise RuntimeError(
            "interior traversal could only sample "
            f"{len(accepted)}/{sampling.position_count_k} valid positions after "
            f"{sampling.max_position_sampling_attempts} attempts; adjust search radius, "
            "clearance, separation, or seed position"
        )
    if not accepted:
        raise RuntimeError(
            "random interior traversal found no valid positions within max_position_sampling_attempts"
        )
    return accepted


def sample_positions(cfg: TraversalConfig, scene: GaussianScene) -> list[PositionPlan]:
    if cfg.scene.scene_type == "object":
        return _sample_object_positions(cfg, scene)
    return _sample_interior_positions(cfg, scene)




def _yaw_pitch_from_direction(
    direction: np.ndarray, world_up: tuple[float, float, float]
) -> tuple[float, float]:
    horizontal, right, up = _world_basis(world_up)
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("object camera position cannot equal its look-at target")
    direction /= norm
    vertical = float(np.clip(np.dot(direction, up), -1.0, 1.0))
    pitch = math.asin(vertical)
    flat = direction - vertical * up
    flat_norm = float(np.linalg.norm(flat))
    if flat_norm < 1e-12:
        yaw = 0.0
    else:
        flat /= flat_norm
        yaw = math.atan2(float(np.dot(flat, right)), float(np.dot(flat, horizontal)))
    return math.degrees(yaw), math.degrees(pitch)


def sample_orientations(
    cfg: TraversalConfig, position: PositionPlan
) -> list[CapturePlan]:
    match = cfg.initialization.reference_match
    count = cfg.initialization.traversal.images_per_position_l
    seed = match.seed_pose
    rng = np.random.default_rng(
        np.random.SeedSequence([cfg.initialization.traversal.random_seed, position.index])
    )

    if cfg.scene.scene_type == "object":
        if position.look_at_target is None:
            raise RuntimeError("object traversal position is missing its look-at target")
        object_cfg = cfg.initialization.object
        base_yaw, base_pitch = _yaw_pitch_from_direction(
            position.look_at_target - position.position,
            cfg.initialization.world_up,
        )
        yaw_center = base_yaw
        pitch_center = base_pitch
        yaw_range = object_cfg.center_yaw_jitter_range_deg
        pitch_range = object_cfg.center_pitch_jitter_range_deg
        roll_range = object_cfg.roll_jitter_range_deg
        roll_center = 0.0
    else:
        yaw_center = seed.yaw_deg
        pitch_center = seed.pitch_deg
        yaw_range = match.yaw_search_range_deg
        pitch_range = match.pitch_search_range_deg
        roll_range = match.roll_search_range_deg
        roll_center = seed.roll_deg

    yaw_edges = np.linspace(
        yaw_center + yaw_range[0],
        yaw_center + yaw_range[1],
        count + 1,
    )
    yaws = rng.uniform(yaw_edges[:-1], yaw_edges[1:])
    rng.shuffle(yaws)
    pitches = rng.uniform(
        pitch_center + pitch_range[0],
        pitch_center + pitch_range[1],
        count,
    )
    if cfg.scene.scene_type == "object":
        pitches = np.clip(pitches, -88.9, 0.0)
    rolls = rng.uniform(
        roll_center + roll_range[0],
        roll_center + roll_range[1],
        count,
    )

    captures = []
    for index in range(count):
        label = _safe_label(
            cfg.output.lens_label_format.format(index=index),
            "lens_label_format",
        )
        captures.append(
            CapturePlan(
                position,
                index,
                label,
                float(yaws[index]),
                float(pitches[index]),
                float(rolls[index]),
                float(seed.fov_y_deg),
            )
        )
    return captures


def plan_traversal(
    cfg: TraversalConfig, scene: GaussianScene
) -> tuple[list[PositionPlan], list[CapturePlan]]:
    positions = sample_positions(cfg, scene)
    captures = [
        capture
        for position in positions
        for capture in sample_orientations(cfg, position)
    ]
    expected = len(positions) * cfg.initialization.traversal.images_per_position_l
    if len(captures) != expected:
        raise RuntimeError(f"internal traversal plan mismatch: expected {expected}, got {len(captures)}")
    return positions, captures


def capture_paths(
    cfg: TraversalConfig, position_label: str, lens_label: str
) -> tuple[Path, Path]:
    directory = (
        cfg.output.root_directory
        / cfg.output.scene_name
        / _safe_label(position_label, "position_label")
        / _safe_label(lens_label, "lens_label")
    )
    return directory / cfg.output.image_filename, directory / cfg.output.metadata_filename


def _camera_frame(
    cfg: TraversalConfig, scene: GaussianScene, capture: CapturePlan
) -> CameraFrame:
    position = capture.position.position
    c2w = pose_from_yaw_pitch(
        position,
        capture.yaw_deg,
        capture.pitch_deg,
        capture.roll_deg,
        np.asarray(cfg.initialization.world_up, dtype=np.float64),
    )
    width = cfg.initialization.reference_match.preview_width
    height = cfg.initialization.reference_match.preview_height
    fov_y = math.radians(capture.fov_y_deg)
    fy = (height / 2.0) / math.tan(fov_y / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    clipping = cfg.initialization.clipping
    near = max(
        clipping.near_min,
        capture.position.clearance * clipping.near_radius_ratio,
    )
    far = max(near * 10.0, scene.analysis.radius * clipping.far_radius_ratio)
    target_distance = 1.0
    if capture.position.look_at_target is not None:
        target_distance = float(
            np.linalg.norm(capture.position.look_at_target - position)
        )
    target = position + target_distance * c2w[:3, 2]
    return CameraFrame(
        position.copy(),
        target,
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
        np.zeros(3, dtype=np.float64),
    )


def _object_exterior_distance(
    scene: GaussianScene,
    outward_direction: np.ndarray,
    margin_radius_ratio: float,
) -> float:
    center = np.asarray(scene.analysis.center, dtype=np.float64)
    lower = np.asarray(scene.analysis.aabb_min, dtype=np.float64)
    upper = np.asarray(scene.analysis.aabb_max, dtype=np.float64)
    boundary_distances = []
    for axis, component in enumerate(outward_direction):
        if component > 1e-12:
            boundary_distances.append((upper[axis] - center[axis]) / component)
        elif component < -1e-12:
            boundary_distances.append((lower[axis] - center[axis]) / component)
    positive_distances = [
        float(distance)
        for distance in boundary_distances
        if math.isfinite(distance) and distance >= 0.0
    ]
    if not positive_distances:
        raise RuntimeError("could not find the object AABB boundary along the camera ray")
    radius = float(scene.analysis.radius)
    margin = margin_radius_ratio * radius
    # Keep a zero configured margin numerically outside the inclusive AABB.
    strict_exterior_epsilon = max(radius, 1.0) * 1e-9
    return max(
        min(positive_distances) + margin + strict_exterior_epsilon,
        np.finfo(np.float64).eps,
    )


def _object_camera_at_distance(
    cfg: TraversalConfig,
    scene: GaussianScene,
    camera: CameraFrame,
    distance: float,
    clearance_tree: cKDTree,
) -> tuple[CameraFrame, float]:
    center = np.asarray(scene.analysis.center, dtype=np.float64)
    outward = np.asarray(camera.position, dtype=np.float64) - center
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm < 1e-12:
        raise ValueError("object camera position cannot equal the scene center")
    outward /= outward_norm
    position = center + float(distance) * outward
    c2w = np.asarray(camera.c2w, dtype=np.float64).copy()
    c2w[:3, 3] = position
    clearance = float(clearance_tree.query(position, k=1, workers=-1)[0])
    clipping = cfg.initialization.clipping
    near = max(clipping.near_min, clearance * clipping.near_radius_ratio)
    far = max(
        near * 10.0,
        float(scene.analysis.radius) * clipping.far_radius_ratio,
        float(distance) + 2.0 * float(scene.analysis.radius),
    )
    target = position + max(float(distance), 1e-6) * c2w[:3, 2]
    moved = CameraFrame(
        position,
        target,
        c2w,
        np.linalg.inv(c2w),
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.fov_x,
        camera.fov_y,
        near,
        far,
        camera.camera_center_offset.copy(),
    )
    return moved, clearance


def _object_gaussian_coverage_ratio(
    renderer: GaussianRenderer,
    camera: CameraFrame,
    coverage_cfg: ObjectCoverageConfig,
    width: int | None = None,
    height: int | None = None,
) -> float:
    alpha = renderer.render(
        camera,
        width or coverage_cfg.preview_width,
        height or coverage_cfg.preview_height,
        alpha_only=True,
    )
    ratio = float(np.mean(np.asarray(alpha) > coverage_cfg.alpha_threshold))
    if not math.isfinite(ratio):
        raise RuntimeError("object Gaussian coverage produced a non-finite ratio")
    return ratio


def _fit_object_camera_coverage(
    cfg: TraversalConfig,
    scene: GaussianScene,
    renderer: GaussianRenderer,
    camera: CameraFrame,
    clearance_tree: cKDTree,
) -> tuple[CameraFrame, dict]:
    coverage_cfg = cfg.initialization.object.coverage
    center = np.asarray(scene.analysis.center, dtype=np.float64)
    outward = np.asarray(camera.position, dtype=np.float64) - center
    initial_distance = float(np.linalg.norm(outward))
    if initial_distance < 1e-12:
        raise ValueError("object camera position cannot equal the scene center")
    outward /= initial_distance
    minimum_distance = _object_exterior_distance(
        scene,
        outward,
        coverage_cfg.exterior_margin_radius_ratio,
    )
    distance_tolerance = max(initial_distance, minimum_distance, 1.0) * 1e-12
    if minimum_distance > initial_distance + distance_tolerance:
        raise RuntimeError(
            "initial object camera is inside the configured AABB exterior margin: "
            f"distance={initial_distance:.6g}, required>={minimum_distance:.6g}"
        )
    minimum_distance = min(minimum_distance, initial_distance)
    preview_history: list[dict] = []

    def evaluate(
        distance: float,
        history: list[dict],
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[CameraFrame, float, float]:
        moved, clearance = _object_camera_at_distance(
            cfg, scene, camera, distance, clearance_tree
        )
        ratio = _object_gaussian_coverage_ratio(
            renderer,
            moved,
            coverage_cfg,
            width,
            height,
        )
        history.append(
            {
                "distance_to_center": float(distance),
                "pixel_ratio": ratio,
                "distance_to_nearest_effective_gaussian": clearance,
            }
        )
        return moved, ratio, clearance

    initial_result = evaluate(
        initial_distance,
        preview_history,
    )
    initial_ratio = initial_result[1]
    target_ratio = coverage_cfg.minimum_pixel_ratio

    def fit_at_resolution(
        starting_result: tuple[CameraFrame, float, float],
        starting_distance: float,
        history: list[dict],
        width: int | None,
        height: int | None,
        distance_hint: float | None = None,
    ) -> tuple[CameraFrame, float, float, float, bool]:
        camera_at_distance, ratio, clearance = starting_result
        if ratio >= target_ratio:
            return camera_at_distance, ratio, clearance, starting_distance, True
        if starting_distance <= minimum_distance + 1e-12:
            return camera_at_distance, ratio, clearance, starting_distance, False

        minimum_result = evaluate(
            minimum_distance,
            history,
            width,
            height,
        )
        minimum_camera, minimum_ratio, minimum_clearance = minimum_result
        if minimum_ratio < target_ratio:
            if (minimum_ratio, minimum_distance) > (ratio, starting_distance):
                return (
                    minimum_camera,
                    minimum_ratio,
                    minimum_clearance,
                    minimum_distance,
                    False,
                )
            return camera_at_distance, ratio, clearance, starting_distance, False

        failing_distance = starting_distance
        passing_camera = minimum_camera
        passing_ratio = minimum_ratio
        passing_clearance = minimum_clearance
        passing_distance = minimum_distance
        if (
            distance_hint is not None
            and passing_distance + 1e-12 < distance_hint < failing_distance - 1e-12
            and len(history) < coverage_cfg.max_iterations
        ):
            hint_camera, hint_ratio, hint_clearance = evaluate(
                distance_hint,
                history,
                width,
                height,
            )
            if hint_ratio >= target_ratio:
                passing_camera = hint_camera
                passing_ratio = hint_ratio
                passing_clearance = hint_clearance
                passing_distance = distance_hint
            else:
                failing_distance = distance_hint
        while len(history) < coverage_cfg.max_iterations:
            candidate_distance = 0.5 * (failing_distance + passing_distance)
            if (
                candidate_distance <= passing_distance + 1e-12
                or candidate_distance >= failing_distance - 1e-12
            ):
                break
            candidate_camera, candidate_ratio, candidate_clearance = evaluate(
                candidate_distance,
                history,
                width,
                height,
            )
            if candidate_ratio >= target_ratio:
                passing_camera = candidate_camera
                passing_ratio = candidate_ratio
                passing_clearance = candidate_clearance
                passing_distance = candidate_distance
            else:
                failing_distance = candidate_distance
        return (
            passing_camera,
            passing_ratio,
            passing_clearance,
            passing_distance,
            True,
        )

    (
        best_camera,
        best_ratio,
        best_clearance,
        best_distance,
        constraints_met,
    ) = fit_at_resolution(
        initial_result,
        initial_distance,
        preview_history,
        None,
        None,
    )

    preview_constraints_met = constraints_met
    preview_selected_distance = best_distance
    preview_selected_ratio = best_ratio
    output_width = cfg.initialization.reference_match.preview_width
    output_height = cfg.initialization.reference_match.preview_height
    output_history: list[dict] = []
    if (output_width, output_height) != (
        coverage_cfg.preview_width,
        coverage_cfg.preview_height,
    ):
        output_starting_result = evaluate(
            initial_distance,
            output_history,
            output_width,
            output_height,
        )
        (
            best_camera,
            best_ratio,
            best_clearance,
            best_distance,
            constraints_met,
        ) = fit_at_resolution(
            output_starting_result,
            initial_distance,
            output_history,
            output_width,
            output_height,
            preview_selected_distance,
        )

    final_distance = float(np.linalg.norm(best_camera.position - center))
    metadata = {
        "method": "farthest_passing_bracketed_search",
        "alpha_threshold": coverage_cfg.alpha_threshold,
        "minimum_pixel_ratio": target_ratio,
        "preview_size": [coverage_cfg.preview_width, coverage_cfg.preview_height],
        "output_validation_size": [output_width, output_height],
        "preview_constraints_met": preview_constraints_met,
        "output_constraints_met": constraints_met,
        "constraints_met": constraints_met,
        "camera_moved_closer": bool(final_distance < initial_distance - 1e-12),
        "initial_position": camera.position.tolist(),
        "final_position": best_camera.position.tolist(),
        "initial_distance_to_center": initial_distance,
        "minimum_exterior_distance_to_center": minimum_distance,
        "final_distance_to_center": final_distance,
        "initial_pixel_ratio": initial_ratio,
        "preview_selected_distance_to_center": preview_selected_distance,
        "preview_selected_pixel_ratio": preview_selected_ratio,
        "final_pixel_ratio": best_ratio,
        "final_distance_to_nearest_effective_gaussian": best_clearance,
        "preview_evaluation_count": len(preview_history),
        "output_validation_count": len(output_history),
        "evaluation_count": len(preview_history) + len(output_history),
        "history": preview_history,
        "output_validation_history": output_history,
    }
    return best_camera, metadata


def _write_json(path: Path, value: dict | list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _encoded_rgb_image(rgb: np.ndarray, cfg: TraversalConfig) -> np.ndarray:
    encoded = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / cfg.image.gamma)
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def _create_topiq_nr_metric(device: str):
    try:
        import pyiqa
        import torch
    except ImportError as error:
        raise RuntimeError(
            "random traversal requires pyiqa and torch; install pyiqa in the "
            "active traversal environment"
        ) from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"random traversal requested device {device!r}, but CUDA is unavailable"
        )
    # The CFANet checkpoint includes the ResNet50 backbone, so avoid a
    # redundant timm checkpoint download before loading the IQA weights.
    return (
        pyiqa.create_metric(
            "topiq_nr",
            device=device,
            backbone_pretrain=False,
        ),
        torch,
    )


def _topiq_nr_score(
    rgb: np.ndarray,
    cfg: TraversalConfig,
    metric,
    torch_module,
    device: str,
) -> float:
    image = np.ascontiguousarray(_encoded_rgb_image(rgb, cfg))
    try:
        tensor = torch_module.from_numpy(image)
    except RuntimeError as error:
        if "Numpy is not available" not in str(error):
            raise
        # PyTorch built against NumPy 1.x cannot share memory with NumPy 2.x.
        tensor = torch_module.tensor(image.tolist(), dtype=torch_module.uint8)
    tensor = (
        tensor.permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch_module.float32)
        .div_(255.0)
    )
    with torch_module.inference_mode():
        score = float(metric(tensor).item())
    if not math.isfinite(score):
        raise RuntimeError(f"TOPIQ-NR returned a non-finite score: {score}")
    return score


def _save_capture_pair(
    image_path: Path, metadata_path: Path, rgb: np.ndarray, metadata: dict, cfg: TraversalConfig
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_tmp = image_path.with_name(f".{image_path.stem}.tmp.png")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.stem}.tmp.json")
    try:
        image = _encoded_rgb_image(rgb, cfg)
        ok = cv2.imwrite(
            str(image_tmp),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, cfg.image.png_compression],
        )
        if not ok:
            raise RuntimeError(f"failed to save traversal image: {image_path}")
        _write_json(metadata_tmp, metadata)
        image_tmp.replace(image_path)
        metadata_tmp.replace(metadata_path)
    finally:
        if image_tmp.exists():
            image_tmp.unlink()
        if metadata_tmp.exists():
            metadata_tmp.unlink()


def _capture_metadata(
    cfg: TraversalConfig,
    scene: GaussianScene,
    capture: CapturePlan,
    camera: CameraFrame,
    image_path: Path,
    geometry_quality: dict | None = None,
    topiq_nr_score: float | None = None,
    object_coverage: dict | None = None,
) -> dict:
    scene_directory = cfg.output.root_directory / cfg.output.scene_name
    offset_from_seed = capture.position.offset_from_seed
    distance_to_seed = float(np.linalg.norm(offset_from_seed))
    distance_to_nearest = capture.position.clearance
    sampling_metadata = {
        "strategy": (
            "random"
            if cfg.initialization.traversal.strategy == "random"
            else (
                "object_shell" if cfg.scene.scene_type == "object"
                else "interior_clearance"
            )
        ),
        "seed_position": list(cfg.initialization.reference_match.seed_pose.position),
        "seed_position_source": capture.position.seed_position_source,
        "offset_from_seed": offset_from_seed.tolist(),
        "distance_to_seed": distance_to_seed,
        "distance_to_nearest_effective_gaussian": distance_to_nearest,
        "object_center": (
            capture.position.look_at_target.tolist()
            if capture.position.look_at_target is not None else None
        ),
        "camera_inside_robust_aabb": bool(
            np.all(camera.position >= scene.analysis.aabb_min)
            and np.all(camera.position <= scene.analysis.aabb_max)
        ),
        "random_seed": cfg.initialization.traversal.random_seed,
    }
    if cfg.scene.scene_type == "object":
        seed_position = np.asarray(
            cfg.initialization.reference_match.seed_pose.position,
            dtype=np.float64,
        )
        actual_offset = np.asarray(camera.position, dtype=np.float64) - seed_position
        sampling_metadata.update(
            {
                "planned_position": capture.position.position.tolist(),
                "planned_offset_from_seed": offset_from_seed.tolist(),
                "planned_distance_to_seed": distance_to_seed,
                "planned_distance_to_nearest_effective_gaussian": distance_to_nearest,
                "offset_from_seed": actual_offset.tolist(),
                "distance_to_seed": float(np.linalg.norm(actual_offset)),
                "distance_to_nearest_effective_gaussian": (
                    object_coverage.get(
                        "final_distance_to_nearest_effective_gaussian",
                        distance_to_nearest,
                    )
                    if object_coverage is not None
                    else distance_to_nearest
                ),
            }
        )
        sampling_metadata["gaussian_coverage"] = object_coverage

    return {
        "scene_name": cfg.output.scene_name,
        "scene_type": cfg.scene.scene_type,
        "position_index": capture.position.index,
        "position_label": capture.position.label,
        "lens_index": capture.lens_index,
        "lens_label": capture.lens_label,
        "image_relative_path": str(image_path.relative_to(scene_directory)),
        "camera": {
            "coordinate_convention": "OpenCV/COLMAP camera-to-world; +x right, +y down, +z forward",
            "position": camera.position.tolist(),
            "look_direction": camera.c2w[:3, 2].tolist(),
            "look_at_target": camera.target.tolist(),
            "yaw_deg": capture.yaw_deg,
            "pitch_deg": capture.pitch_deg,
            "roll_deg": capture.roll_deg,
            "camera_to_world": camera.c2w.tolist(),
            "world_to_camera": camera.w2c.tolist(),
        },
        "intrinsics": {
            "fixed_across_traversal": True,
            "width": cfg.initialization.reference_match.preview_width,
            "height": cfg.initialization.reference_match.preview_height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "fov_x_deg": math.degrees(camera.fov_x),
            "fov_y_deg": capture.fov_y_deg,
            "near": camera.near,
            "far": camera.far,
        },
        "sampling": sampling_metadata,
        "geometry_quality": geometry_quality,
        "image_quality": (
            {
                "metric": "topiq_nr",
                "score": topiq_nr_score,
                "threshold_l": cfg.initialization.traversal.random.topiq_nr_threshold_l,
                "accepted": (
                    topiq_nr_score
                    > cfg.initialization.traversal.random.topiq_nr_threshold_l
                ),
            }
            if topiq_nr_score is not None
            else None
        ),
        "scene_root_transform": cfg.scene.root_transform.model_dump(mode="json"),
    }


def run_traversal(cfg: TraversalConfig) -> dict:
    scene_directory = cfg.output.root_directory / cfg.output.scene_name
    scene_directory.mkdir(parents=True, exist_ok=True)
    with (scene_directory / "resolved_traversal_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            cfg.model_dump(mode="json"), handle, sort_keys=False, allow_unicode=True
        )

    adapter = _renderer_adapter(cfg)
    print(f"Loading and analyzing scene: {cfg.input.ply_path}")
    scene = GaussianScene(cfg.input.ply_path, adapter)

    print(f"Planning random {cfg.scene.scene_type} traversal before GPU rendering...")
    positions, captures = plan_traversal(cfg, scene)
    sampling = cfg.initialization.traversal
    is_object = cfg.scene.scene_type == "object"
    is_random_quality = sampling.strategy == "random"
    is_object_random_quality = is_object and is_random_quality
    captures_to_render = captures
    score_candidate_budget = None
    if is_random_quality:
        if is_object_random_quality:
            score_candidate_budget = (
                sampling.random.target_count_k
                * cfg.initialization.object.random_candidate_multiplier
            )
            captures_to_render = sorted(
                captures,
                key=lambda capture: (
                    capture.lens_index,
                    capture.position.index,
                ),
            )
            print(
                f"Object quality strategy: score up to {score_candidate_budget} "
                f"coverage-qualified candidates from {len(captures_to_render)} "
                "center-facing precheck candidates; retain the best "
                f"{sampling.random.target_count_k} TOPIQ-NR scores and stop once all "
                f"are > {sampling.random.topiq_nr_threshold_l}"
            )
        else:
            order_rng = np.random.default_rng(
                np.random.SeedSequence([sampling.random_seed, 0x70A1])
            )
            order = order_rng.permutation(len(captures))
            captures_to_render = [captures[int(index)] for index in order]
            print(
                f"Random quality strategy: {len(positions)} sampled positions x "
                f"{sampling.images_per_position_l} images = {len(captures)} candidates; "
                f"stop after {sampling.random.target_count_k} captures with topiq_nr > "
                f"{sampling.random.topiq_nr_threshold_l}"
            )
    with (scene_directory / "resolved_traversal_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            cfg.model_dump(mode="json"), handle, sort_keys=False, allow_unicode=True
        )
    plan_json = {
        "scene_name": cfg.output.scene_name,
        "scene_type": cfg.scene.scene_type,
        "sampling_strategy": sampling.strategy,
        "random_quality": (
            sampling.random.model_dump(mode="json")
            if is_random_quality
            else None
        ),
        "position_count_k": len(positions),
        "images_per_position_l": cfg.initialization.traversal.images_per_position_l,
        "total_capture_count": len(captures),
        **(
            {"render_candidate_count": len(captures_to_render)}
            if is_object else {}
        ),
        "seed_pose": {
            "position": list(cfg.initialization.reference_match.seed_pose.position),
            "position_source": positions[0].seed_position_source,
            "yaw_deg": cfg.initialization.reference_match.seed_pose.yaw_deg,
            "pitch_deg": cfg.initialization.reference_match.seed_pose.pitch_deg,
            "roll_deg": cfg.initialization.reference_match.seed_pose.roll_deg,
            "fov_y_deg": cfg.initialization.reference_match.seed_pose.fov_y_deg,
        },
        "intrinsics_policy": {
            "fixed_across_traversal": True,
            "width": cfg.initialization.reference_match.preview_width,
            "height": cfg.initialization.reference_match.preview_height,
            "fov_y_deg": cfg.initialization.reference_match.seed_pose.fov_y_deg,
        },
        "object_sampling": (
            cfg.initialization.object.model_dump(mode="json")
            if is_object else None
        ),
        "positions": [
            {
                "index": item.index,
                "label": item.label,
                "position": item.position.tolist(),
                "offset_from_seed": item.offset_from_seed.tolist(),
                "distance_to_nearest_effective_gaussian": item.clearance,
                "look_at_target": (
                    item.look_at_target.tolist()
                    if item.look_at_target is not None else None
                ),
            }
            for item in positions
        ],
    }
    _write_json(scene_directory / "traversal_plan.json", plan_json)

    print(f"Loading {scene.analysis.effective_gaussian_count:,} effective Gaussians...")
    renderer = GaussianRenderer(scene.load_tensors(), adapter)
    topiq_nr_metric = None
    topiq_torch = None
    if is_random_quality:
        topiq_nr_metric, topiq_torch = _create_topiq_nr_metric(sampling.random.device)
    object_clearance_tree = None
    if is_object:
        clearance_points = scene.effective_position_sample()
        if not len(clearance_points):
            raise RuntimeError(
                "no effective points are available for object coverage fitting"
            )
        object_clearance_tree = cKDTree(clearance_points)

    attempted_count = 0
    scored_count = 0
    qualified_count = 0
    rejected_count = 0
    coverage_rejected_count = 0
    manifest_items = []
    top_candidates: list[tuple[float, int, dict]] = []
    width = cfg.initialization.reference_match.preview_width
    height = cfg.initialization.reference_match.preview_height
    for capture in tqdm(captures_to_render, desc="Rendering traversal", unit="image"):
        attempted_count += 1
        camera = _camera_frame(cfg, scene, capture)
        object_coverage = None
        if is_object:
            assert object_clearance_tree is not None
            camera, object_coverage = _fit_object_camera_coverage(
                cfg,
                scene,
                renderer,
                camera,
                object_clearance_tree,
            )
            if not object_coverage["constraints_met"]:
                coverage_rejected_count += 1
                if is_object_random_quality:
                    continue
                raise RuntimeError(
                    "object Gaussian coverage remained below "
                    f"{object_coverage['minimum_pixel_ratio']:.4f} after moving to "
                    "the configured exterior limit; lower "
                    "initialization.object.coverage.minimum_pixel_ratio or "
                    "exterior_margin_radius_ratio"
                )

        rgb, radii = renderer.render(
            camera,
            width,
            height,
            return_radii=True,
        )
        topiq_nr_score = None
        if is_random_quality:
            topiq_nr_score = _topiq_nr_score(
                rgb,
                cfg,
                topiq_nr_metric,
                topiq_torch,
                sampling.random.device,
            )
            scored_count += 1
            if topiq_nr_score > sampling.random.topiq_nr_threshold_l:
                qualified_count += 1
            else:
                rejected_count += 1

            if is_object_random_quality:
                rank_key = (topiq_nr_score, -attempted_count)
                should_retain = (
                    len(top_candidates) < sampling.random.target_count_k
                    or rank_key > top_candidates[0][:2]
                )
                if should_retain:
                    geometry_quality = screen_space_quality_metrics(
                        renderer.g,
                        camera,
                        radii,
                        width,
                        height,
                    )
                    image_path, metadata_path = capture_paths(
                        cfg, capture.position.label, capture.lens_label
                    )
                    metadata = _capture_metadata(
                        cfg,
                        scene,
                        capture,
                        camera,
                        image_path,
                        geometry_quality,
                        topiq_nr_score,
                        object_coverage,
                    )
                    buffered = {
                        "capture": capture,
                        "image_path": image_path,
                        "metadata_path": metadata_path,
                        "rgb": rgb,
                        "metadata": metadata,
                        "geometry_quality": geometry_quality,
                        "object_coverage": object_coverage,
                    }
                    entry = (rank_key[0], rank_key[1], buffered)
                    if len(top_candidates) < sampling.random.target_count_k:
                        heapq.heappush(top_candidates, entry)
                    else:
                        heapq.heapreplace(top_candidates, entry)
                del radii
                assert score_candidate_budget is not None
                if (
                    qualified_count >= sampling.random.target_count_k
                    or scored_count >= score_candidate_budget
                ):
                    break
                continue

            if topiq_nr_score <= sampling.random.topiq_nr_threshold_l:
                del radii
                continue

        geometry_quality = screen_space_quality_metrics(
            renderer.g,
            camera,
            radii,
            width,
            height,
        )
        del radii
        image_path, metadata_path = capture_paths(
            cfg, capture.position.label, capture.lens_label
        )
        if is_object:
            metadata = _capture_metadata(
                cfg,
                scene,
                capture,
                camera,
                image_path,
                geometry_quality,
                topiq_nr_score,
                object_coverage=object_coverage,
            )
        else:
            metadata = _capture_metadata(
                cfg,
                scene,
                capture,
                camera,
                image_path,
                geometry_quality,
                topiq_nr_score,
            )
        _save_capture_pair(image_path, metadata_path, rgb, metadata, cfg)
        manifest_items.append(
            {
                "position_label": capture.position.label,
                "lens_label": capture.lens_label,
                "image": str(image_path.relative_to(scene_directory)),
                "metadata": str(metadata_path.relative_to(scene_directory)),
                "geometry_quality": geometry_quality,
                "topiq_nr_score": topiq_nr_score,
                **(
                    {"object_coverage": object_coverage}
                    if object_coverage is not None else {}
                ),
            }
        )
        if is_random_quality and len(manifest_items) >= sampling.random.target_count_k:
            break

    fallback_used = False
    fallback_incomplete = False
    termination_reason = None
    search_exhaustive = False
    ranked_candidates: list[tuple[float, int, dict]] = []
    if is_object_random_quality:
        fallback_used = qualified_count < sampling.random.target_count_k
        fallback_incomplete = len(top_candidates) < sampling.random.target_count_k
        search_exhaustive = attempted_count >= len(captures_to_render)
        if qualified_count >= sampling.random.target_count_k:
            termination_reason = "threshold_complete"
        elif search_exhaustive:
            termination_reason = "candidate_pool_exhausted"
        else:
            termination_reason = "score_budget_exhausted"
        ranked_candidates = sorted(
            top_candidates,
            key=lambda item: (-item[0], -item[1]),
        )
        for rank, (score, _negative_attempt, buffered) in enumerate(
            ranked_candidates, start=1
        ):
            capture = buffered["capture"]
            image_path = buffered["image_path"]
            metadata_path = buffered["metadata_path"]
            metadata = buffered["metadata"]
            metadata["image_quality"].update(
                {
                    "selected": True,
                    "selection_rank": rank,
                    "selection_policy": (
                        "top_k_fallback" if fallback_used
                        else "threshold_complete"
                    ),
                }
            )
            _save_capture_pair(
                image_path,
                metadata_path,
                buffered["rgb"],
                metadata,
                cfg,
            )
            manifest_items.append(
                {
                    "position_label": capture.position.label,
                    "lens_label": capture.lens_label,
                    "image": str(image_path.relative_to(scene_directory)),
                    "metadata": str(metadata_path.relative_to(scene_directory)),
                    "geometry_quality": buffered["geometry_quality"],
                    "topiq_nr_score": score,
                    "selection_rank": rank,
                    "object_coverage": buffered["object_coverage"],
                }
            )
        if fallback_used:
            best_score = ranked_candidates[0][0] if ranked_candidates else None
            print(
                "Object quality fallback: "
                f"{qualified_count}/{sampling.random.target_count_k} scored captures "
                f"satisfied topiq_nr > {sampling.random.topiq_nr_threshold_l}; "
                f"saving {len(ranked_candidates)} highest-scoring captures "
                f"(best={best_score}, reason={termination_reason}, "
                f"search_exhaustive={search_exhaustive})."
            )

    if is_random_quality:
        if is_object_random_quality:
            random_quality_result = {
                "target_count_k": sampling.random.target_count_k,
                "topiq_nr_threshold_l": sampling.random.topiq_nr_threshold_l,
                "acceptance_condition": "topiq_nr > topiq_nr_threshold_l",
                "retention_policy": "top_k_by_topiq_nr",
                "candidate_budget": score_candidate_budget,
                "coverage_candidate_pool": len(captures_to_render),
                "attempted_count": attempted_count,
                "coverage_rejected_count": coverage_rejected_count,
                "scored_count": scored_count,
                "accepted_count": qualified_count,
                "rejected_count": rejected_count,
                "saved_count": len(manifest_items),
                "completed": qualified_count >= sampling.random.target_count_k,
                "fallback_used": fallback_used,
                "fallback_incomplete": fallback_incomplete,
                "termination_reason": termination_reason,
                "search_exhaustive": search_exhaustive,
            }
        else:
            random_quality_result = {
                "target_count_k": sampling.random.target_count_k,
                "topiq_nr_threshold_l": sampling.random.topiq_nr_threshold_l,
                "acceptance_condition": "topiq_nr > topiq_nr_threshold_l",
                "attempted_count": attempted_count,
                "accepted_count": len(manifest_items),
                "rejected_count": rejected_count,
                "completed": len(manifest_items) >= sampling.random.target_count_k,
            }
    else:
        random_quality_result = None

    summary = {
        **plan_json,
        "input_ply_path": str(cfg.input.ply_path),
        "output_scene_directory": str(scene_directory),
        "image_size": [
            cfg.initialization.reference_match.preview_width,
            cfg.initialization.reference_match.preview_height,
        ],
        "scene_analysis": scene.analysis.as_dict(),
        "captures": manifest_items,
        **(
            {
                "object_coverage_result": {
                    "minimum_pixel_ratio": (
                        cfg.initialization.object.coverage.minimum_pixel_ratio
                    ),
                    "attempted_count": attempted_count,
                    "constraints_met_count": (
                        attempted_count - coverage_rejected_count
                    ),
                    "rejected_count": coverage_rejected_count,
                }
            }
            if is_object else {}
        ),
        "random_quality_result": random_quality_result,
    }
    _write_json(scene_directory / "traversal_summary.json", summary)
    if (
        is_random_quality
        and not is_object_random_quality
        and len(manifest_items) < sampling.random.target_count_k
    ):
        raise RuntimeError(
            f"random traversal accepted {len(manifest_items)}/"
            f"{sampling.random.target_count_k} captures after {attempted_count} attempts; "
            "increase max_position_sampling_attempts or images_per_position_l, "
            "or lower topiq_nr_threshold_l"
        )
    print(
        f"Traversal complete: {len(manifest_items)} image/JSON pairs in {scene_directory}"
    )
    return summary
