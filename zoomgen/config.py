from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputConfig(StrictModel):
    ply_path: Path


class RootTransformConfig(StrictModel):
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_order: Literal["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"] = "xyz"
    scale: float = Field(default=1.0, gt=0.0)


class SceneConfig(StrictModel):
    root_transform: RootTransformConfig = RootTransformConfig()


class OutputConfig(StrictModel):
    directory: Path
    video_filename: str = "zoom_video.mp4"
    save_processed_frames: bool = True
    save_linear_frames: bool = False
    save_alpha_masks: bool = False
    save_metadata: bool = True
    frame_format: Literal["png", "jpg", "jpeg"] = "png"
    overwrite: bool = False


class RenderConfig(StrictModel):
    device: str = "cuda"
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sh_degree: int | Literal["auto"] = "auto"
    antialiasing: bool = True

    @model_validator(mode="after")
    def validate_renderer(self):
        if not self.device.startswith("cuda"):
            raise ValueError("diff_gaussian_rasterization requires a CUDA render.device")
        if self.sh_degree != "auto" and self.sh_degree < 0:
            raise ValueError("render.sh_degree must be non-negative or 'auto'")
        if any(x < 0.0 or x > 1.0 for x in self.background_rgb):
            raise ValueError("render.background_rgb values must be in [0, 1]")
        return self


class VideoConfig(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    total_frames: int = Field(gt=0)
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "medium"


class SceneAnalysisConfig(StrictModel):
    opacity_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    position_quantiles: tuple[float, float] = (0.01, 0.99)
    scale_extent_sigma: float = Field(default=3.0, ge=0.0)
    max_scale_quantile: float = Field(default=0.99, gt=0.0, le=1.0)
    analysis_max_samples: int = Field(default=1_000_000, gt=1000)
    io_chunk_size: int = Field(default=250_000, gt=0)

    @model_validator(mode="after")
    def validate_quantiles(self):
        lo, hi = self.position_quantiles
        if not 0 <= lo < hi <= 1:
            raise ValueError("position_quantiles must satisfy 0 <= low < high <= 1")
        return self


class InitialViewConfig(StrictModel):
    position: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    roll_deg: float = 0.0

    @model_validator(mode="after")
    def complete_pose(self):
        if (self.position is None) != (self.look_at is None):
            raise ValueError("initial_view.position and look_at must both be set or both be null")
        return self


class AutoFitConfig(StrictModel):
    enabled: bool = True
    preview_width: int = Field(default=320, gt=0)
    preview_height: int = Field(default=180, gt=0)
    alpha_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    target_coverage_ratio: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_coverage_ratio: float = Field(default=0.85, ge=0.0, le=1.0)
    max_largest_background_component_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    max_iterations: int = Field(default=20, gt=0)
    precheck_max_attempts: int = Field(default=3, ge=1, le=8)


class ClippingConfig(StrictModel):
    mode: Literal["auto", "auto_local"] = "auto"
    near_min: float = Field(default=0.001, gt=0.0)
    near_radius_ratio: float = Field(default=0.01, gt=0.0)
    far_radius_ratio: float = Field(default=10.0, gt=0.0)


class MotionConfig(StrictModel):
    enabled: bool = True
    type: Literal["smooth_random_spline", "low_frequency_sine"] = "smooth_random_spline"
    translation_amplitude_ratio: tuple[float, float, float] = (0.003, 0.002, 0.001)
    rotation_amplitude_deg: tuple[float, float, float] = (0.20, 0.25, 0.15)
    control_point_interval_seconds: float = Field(default=1.0, gt=0.0)
    seed: int = 3407


class ManualPoseConfig(StrictModel):
    position: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float = 0.0
    fov_y_deg: float | None = Field(default=None, gt=1.0, lt=179.0)

    @model_validator(mode="after")
    def validate_rotation(self):
        uses_look_at = self.look_at is not None
        uses_angles = self.yaw_deg is not None or self.pitch_deg is not None
        if uses_look_at and uses_angles:
            raise ValueError("manual_pose look_at and yaw/pitch cannot both be specified")
        if (self.yaw_deg is None) != (self.pitch_deg is None):
            raise ValueError("manual_pose yaw_deg and pitch_deg must be specified together")
        return self


class ReferenceConfig(StrictModel):
    image_path: Path | None = None
    image_paths: list[Path] = Field(default_factory=list)
    crop_xywh: tuple[int, int, int, int] | None = None
    reference_zoom_ratio: float = Field(default=1.0, gt=0.0)
    ignore_color_difference: bool = True
    relative_rotation_euler_deg: list[tuple[float, float, float]] = Field(default_factory=list)

    def paths(self) -> list[Path]:
        return ([self.image_path] if self.image_path is not None else []) + self.image_paths


class InteriorConfig(StrictModel):
    world_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    position_quantiles: tuple[float, float] = (0.02, 0.98)
    voxel_resolution: int = Field(default=64, ge=8, le=256)
    candidate_position_count: int = Field(default=128, ge=1)
    directions_per_position: int = Field(default=24, ge=1)
    minimum_clearance_radius_ratio: float = Field(default=0.005, gt=0.0)
    maximum_clearance_radius_ratio: float = Field(default=0.20, gt=0.0)
    minimum_coverage_ratio: float = Field(default=0.85, ge=0.0, le=1.0)
    maximum_largest_background_ratio: float = Field(default=0.08, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interior(self):
        lo, hi = self.position_quantiles
        if not 0 <= lo < hi <= 1:
            raise ValueError("interior.position_quantiles must satisfy 0 <= low < high <= 1")
        if self.minimum_clearance_radius_ratio >= self.maximum_clearance_radius_ratio:
            raise ValueError("interior minimum clearance must be smaller than maximum clearance")
        if sum(x * x for x in self.world_up) < 1e-18:
            raise ValueError("interior.world_up must be non-zero")
        return self


class SeedPoseConfig(StrictModel):
    position: tuple[float, float, float] | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float = 0.0
    fov_y_deg: float | None = Field(default=None, gt=1.0, lt=179.0)


class MatchLossConfig(StrictModel):
    grayscale_weight: float = Field(default=1.0, ge=0.0)
    gradient_weight: float = Field(default=1.0, ge=0.0)
    ssim_weight: float = Field(default=1.0, ge=0.0)
    lpips_weight: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_loss(self):
        if self.lpips_weight:
            raise ValueError("LPIPS is optional but unavailable in this lightweight implementation; use 0")
        if self.grayscale_weight + self.gradient_weight + self.ssim_weight <= 0:
            raise ValueError("at least one reference loss weight must be positive")
        return self


class ReferenceMatchConfig(StrictModel):
    preview_width: int = Field(default=320, gt=0)
    preview_height: int = Field(default=180, gt=0)
    optimize_position: bool = True
    optimize_rotation: bool = True
    optimize_fov: bool = True
    fov_y_range_deg: tuple[float, float] = (25.0, 100.0)
    top_k: int = Field(default=12, ge=1)
    seed_pose: SeedPoseConfig = SeedPoseConfig()
    search_radius_ratio: float = Field(default=0.15, gt=0.0)
    yaw_search_range_deg: float = Field(default=180.0, gt=0.0, le=360.0)
    pitch_search_range_deg: float = Field(default=45.0, gt=0.0, le=180.0)
    roll_search_range_deg: float = Field(default=5.0, ge=0.0, le=45.0)
    refine_iterations: int = Field(default=8, ge=0, le=100)
    minimum_similarity_score: float = Field(default=0.55, ge=0.0, le=1.0)
    ambiguity_score_margin: float = Field(default=0.03, ge=0.0)
    selected_candidate_index: int | None = Field(default=None, ge=0)
    loss: MatchLossConfig = MatchLossConfig()

    @model_validator(mode="after")
    def validate_fov_range(self):
        lo, hi = self.fov_y_range_deg
        if not 1.0 < lo < hi < 179.0:
            raise ValueError("reference_match.fov_y_range_deg must lie within (1, 179)")
        return self


class ResolvedPoseConfig(StrictModel):
    position: tuple[float, float, float]
    camera_to_world: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    fov_y_deg_at_1x: float
    source: str
    candidate_index: int | None = None


class InitializationConfig(StrictModel):
    scene_type: Literal["object", "interior", "manual", "reference_image", "auto"] = "object"
    mode: Literal["auto", "manual", "reference_image"] = "auto"
    reference: ReferenceConfig = ReferenceConfig()
    manual_pose: ManualPoseConfig = ManualPoseConfig()
    interior: InteriorConfig = InteriorConfig()
    reference_match: ReferenceMatchConfig = ReferenceMatchConfig()
    resolved_pose: ResolvedPoseConfig | None = None


class CameraConfig(StrictModel):
    fov_y_deg_at_1x: float = Field(default=55.0, gt=1.0, lt=179.0)
    principal_point_offset_px: tuple[float, float] = (0.0, 0.0)
    initial_view: InitialViewConfig = InitialViewConfig()
    auto_fit: AutoFitConfig = AutoFitConfig()
    clipping: ClippingConfig = ClippingConfig()
    initialization: InitializationConfig = InitializationConfig()
    motion: MotionConfig = MotionConfig()


class TemporalJumpConfig(StrictModel):
    strength: float = Field(ge=-1.0, le=1.0)
    position: float = Field(ge=0.0, le=1.0)


class ColorConfig(StrictModel):
    exposure_ev: float = 0.0
    white_balance_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_correction_matrix: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    black_level: float = 0.0
    contrast: float = Field(default=1.0, gt=0.0)
    saturation: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=2.2, gt=0.0)
    vignette_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    noise_std: float = Field(default=0.0, ge=0.0)
    tone_mapping: Literal["soft_shoulder", "clip"] = "soft_shoulder"


class LensConfig(StrictModel):
    name: str = Field(min_length=1)
    zoom_min: float = Field(gt=0.0)
    zoom_max: float = Field(gt=0.0)
    frame_count: int | None = Field(default=None, gt=0)
    camera_center_offset_ratio: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temporal_color_jump: TemporalJumpConfig
    color: ColorConfig

    @model_validator(mode="after")
    def valid_span(self):
        if self.zoom_max <= self.zoom_min:
            raise ValueError(f"lens {self.name}: zoom_max must be greater than zoom_min")
        return self


class FrameAllocationConfig(StrictModel):
    mode: Literal["proportional_to_zoom_span", "explicit"] = "proportional_to_zoom_span"


class LensSwitchConfig(StrictModel):
    color_mode: Literal["hard", "crossfade"] = "crossfade"
    color_transition_frames: int = Field(default=4, ge=0)
    camera_offset_mode: Literal["hard", "crossfade"] = "crossfade"
    camera_transition_frames: int = Field(default=4, ge=0)


class TemporalAdjustmentConfig(StrictModel):
    max_exposure_jump_ev: float = Field(default=1.0, ge=0.0)
    transition_frames: int = Field(default=0, ge=0)
    couple_noise_to_iso: bool = True
    noise_growth_per_ev: float = Field(default=0.5, ge=0.0)


class ZoomConfig(StrictModel):
    curve: Literal["linear", "smoothstep"] = "linear"
    frame_allocation: FrameAllocationConfig = FrameAllocationConfig()
    lens_switch: LensSwitchConfig = LensSwitchConfig()
    temporal_color_adjustment: TemporalAdjustmentConfig = TemporalAdjustmentConfig()
    lenses: list[LensConfig] = Field(min_length=1)


class GeneratorConfig(StrictModel):
    input: InputConfig
    scene: SceneConfig = SceneConfig()
    output: OutputConfig
    render: RenderConfig
    video: VideoConfig
    scene_analysis: SceneAnalysisConfig
    camera: CameraConfig
    zoom: ZoomConfig

    @model_validator(mode="after")
    def validate_global(self):
        if not self.input.ply_path.is_file():
            raise ValueError(f"input PLY does not exist: {self.input.ply_path}")
        init = self.camera.initialization
        if init.mode == "manual":
            pose = init.manual_pose
            if pose.position is None or pose.fov_y_deg is None:
                raise ValueError("manual mode requires manual_pose.position and fov_y_deg")
            if pose.look_at is None and pose.yaw_deg is None:
                raise ValueError("manual mode requires look_at or yaw/pitch")
        if init.mode == "reference_image":
            paths = init.reference.paths()
            if not paths:
                raise ValueError("reference_image mode requires reference.image_path or image_paths")
            missing_refs = [str(x) for x in paths if not x.is_file()]
            if missing_refs:
                raise ValueError(f"reference images do not exist: {missing_refs}")
            rel = init.reference.relative_rotation_euler_deg
            if rel and len(rel) != len(paths):
                raise ValueError("relative_rotation_euler_deg must match the reference image count")
        names = [x.name for x in self.zoom.lenses]
        if len(names) != len(set(names)):
            raise ValueError("lens names must be unique")
        for prev, cur in zip(self.zoom.lenses, self.zoom.lenses[1:]):
            if cur.zoom_min <= prev.zoom_max:
                raise ValueError("lens zoom intervals must be strictly ordered and non-overlapping")
        counts = [x.frame_count for x in self.zoom.lenses]
        mode = self.zoom.frame_allocation.mode
        if mode == "explicit":
            if any(x is None for x in counts):
                raise ValueError("explicit frame allocation requires frame_count for every lens")
            if sum(counts) != self.video.total_frames:
                raise ValueError("explicit lens frame_count sum must equal video.total_frames")
            allocated = [int(x) for x in counts]
        else:
            if any(x is not None for x in counts):
                raise ValueError("frame_count must be null in proportional_to_zoom_span mode")
            spans = [x.zoom_max - x.zoom_min for x in self.zoom.lenses]
            if not any(spans):
                spans = [1.0] * len(spans)
            raw = [x / sum(spans) * self.video.total_frames for x in spans]
            allocated = [int(x) for x in raw]
            remainder = self.video.total_frames - sum(allocated)
            order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - allocated[i]), i))
            for i in order[:remainder]:
                allocated[i] += 1
        if any(n < 1 for n in allocated):
            raise ValueError("total_frames must allocate at least one frame to every lens")
        if any(n < 2 and lens.zoom_max > lens.zoom_min
               for n, lens in zip(allocated, self.zoom.lenses)):
            raise ValueError("each non-zero zoom interval needs at least two frames for both endpoints")
        ta = self.zoom.temporal_color_adjustment.transition_frames
        for lens, count in zip(self.zoom.lenses, allocated):
            jump_index = round(lens.temporal_color_jump.position * (count - 1))
            if ta > count - jump_index:
                raise ValueError(
                    f"lens {lens.name}: temporal transition_frames exceeds remaining frames"
                )
        sw = self.zoom.lens_switch
        for count in allocated[1:]:
            for n in (sw.color_transition_frames, sw.camera_transition_frames):
                if n > count:
                    raise ValueError("lens switch transition exceeds the incoming lens frame count")
        return self


def _read_yaml_mapping(path: str | Path, label: str) -> dict:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} config must contain a YAML mapping: {resolved}")
    return raw


def _validate_fragment_keys(
    value: object,
    label: str,
    allowed: set[str],
    required: set[str],
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{label} contains cross-owned or unknown keys: {unexpected}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    return value


def _camera_json_section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"camera JSON field '{key}' must be an object")
    return value


def _validate_camera_json_root_transform(raw: dict, cfg: GeneratorConfig) -> None:
    actual = _camera_json_section(raw, "scene_root_transform")
    expected = cfg.scene.root_transform
    try:
        translation = np.asarray(actual["translation"], dtype=np.float64)
        rotation = np.asarray(actual["rotation_euler_deg"], dtype=np.float64)
        scale = float(actual["scale"])
        order = str(actual["rotation_order"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "camera JSON scene_root_transform is incomplete or non-numeric"
        ) from exc
    if translation.shape != (3,) or rotation.shape != (3,):
        raise ValueError(
            "camera JSON scene_root_transform translation/rotation must have 3 values"
        )
    if (
        not np.allclose(translation, expected.translation, atol=1e-7)
        or not np.allclose(rotation, expected.rotation_euler_deg, atol=1e-7)
        or not np.isclose(scale, expected.scale, atol=1e-9)
        or order != expected.rotation_order
    ):
        raise ValueError(
            "camera JSON scene_root_transform does not match the scene config"
        )


def apply_traversal_camera_json(
    cfg: GeneratorConfig, path: str | Path
) -> GeneratorConfig:
    resolved_path = Path(path).resolve()
    if not resolved_path.is_file():
        raise ValueError(f"camera JSON does not exist: {resolved_path}")
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid camera JSON: {resolved_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"camera JSON root must be an object: {resolved_path}")

    camera = _camera_json_section(raw, "camera")
    intrinsics = _camera_json_section(raw, "intrinsics")
    _validate_camera_json_root_transform(raw, cfg)
    try:
        c2w = np.asarray(camera["camera_to_world"], dtype=np.float64)
        position = np.asarray(camera["position"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "camera JSON camera.position/camera_to_world is incomplete or non-numeric"
        ) from exc
    if c2w.shape != (4, 4):
        raise ValueError("camera JSON camera_to_world must be a 4x4 matrix")
    if position.shape != (3,):
        raise ValueError("camera JSON camera.position must contain 3 values")
    if not np.all(np.isfinite(c2w)) or not np.all(np.isfinite(position)):
        raise ValueError("camera JSON pose values must be finite")
    if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("camera JSON camera_to_world has an invalid homogeneous row")
    rotation = c2w[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera JSON camera_to_world rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("camera JSON camera_to_world rotation must have determinant +1")
    if not np.allclose(c2w[:3, 3], position, atol=1e-7):
        raise ValueError(
            "camera JSON camera.position does not match camera_to_world translation"
        )

    try:
        width = float(intrinsics["width"])
        height = float(intrinsics["height"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        fov_y_deg = float(intrinsics["fov_y_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "camera JSON intrinsics width/height/cx/cy/fov_y_deg is incomplete "
            "or non-numeric"
        ) from exc
    intrinsic_values = np.asarray([width, height, cx, cy, fov_y_deg])
    if not np.all(np.isfinite(intrinsic_values)):
        raise ValueError("camera JSON intrinsics must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("camera JSON intrinsics width/height must be positive")
    if not 1.0 < fov_y_deg < 179.0:
        raise ValueError("camera JSON intrinsics.fov_y_deg must lie within (1, 179)")

    principal_offset = (
        (cx / width - 0.5) * cfg.video.width,
        (cy / height - 0.5) * cfg.video.height,
    )
    merged = cfg.model_dump(mode="python")
    merged["camera"]["fov_y_deg_at_1x"] = fov_y_deg
    merged["camera"]["principal_point_offset_px"] = principal_offset
    merged["camera"]["initialization"]["scene_type"] = "interior"
    merged["camera"]["initialization"]["resolved_pose"] = {
        "position": c2w[:3, 3].tolist(),
        "camera_to_world": c2w.tolist(),
        "fov_y_deg_at_1x": fov_y_deg,
        "source": f"traversal_camera_json:{resolved_path}",
        "candidate_index": None,
    }
    return GeneratorConfig.model_validate(merged)


def _check_output(cfg: GeneratorConfig, check_output: bool) -> GeneratorConfig:
    if check_output and cfg.output.directory.exists() and any(cfg.output.directory.iterdir()):
        if not cfg.output.overwrite:
            raise ValueError(
                f"output directory is not empty: {cfg.output.directory}; set output.overwrite=true"
            )
    return cfg


def load_config_parts(
    scene_path: str | Path,
    camera_path: str | Path,
    color_path: str | Path,
    *,
    camera_json_path: str | Path | None = None,
    check_output: bool = True,
) -> GeneratorConfig:
    scene_raw = _validate_fragment_keys(
        _read_yaml_mapping(scene_path, "scene"),
        "scene config",
        {"input", "scene", "output", "render", "video", "scene_analysis"},
        {"input", "scene", "output", "render", "video", "scene_analysis"},
    )
    camera_raw = _validate_fragment_keys(
        _read_yaml_mapping(camera_path, "camera"),
        "camera config",
        {"camera", "zoom"},
        {"camera", "zoom"},
    )
    color_raw = _validate_fragment_keys(
        _read_yaml_mapping(color_path, "color"),
        "color config",
        {"color"},
        {"color"},
    )

    scene_render = _validate_fragment_keys(
        scene_raw["render"],
        "scene render",
        {"device", "sh_degree", "antialiasing"},
        {"device", "sh_degree", "antialiasing"},
    )
    camera_zoom = _validate_fragment_keys(
        camera_raw["zoom"],
        "camera zoom",
        {"curve", "frame_allocation", "lens_switch", "lenses"},
        {"curve", "frame_allocation", "lens_switch", "lenses"},
    )
    camera_switch = _validate_fragment_keys(
        camera_zoom["lens_switch"],
        "camera lens_switch",
        {"camera_offset_mode", "camera_transition_frames"},
        {"camera_offset_mode", "camera_transition_frames"},
    )
    color_bundle = _validate_fragment_keys(
        color_raw["color"],
        "color bundle",
        {"background_rgb", "lens_switch", "temporal_color_adjustment", "lenses"},
        {"background_rgb", "lens_switch", "temporal_color_adjustment", "lenses"},
    )
    color_switch = _validate_fragment_keys(
        color_bundle["lens_switch"],
        "color lens_switch",
        {"color_mode", "color_transition_frames"},
        {"color_mode", "color_transition_frames"},
    )

    camera_lenses = camera_zoom["lenses"]
    color_lenses = color_bundle["lenses"]
    if not isinstance(camera_lenses, list) or not camera_lenses:
        raise ValueError("camera zoom.lenses must be a non-empty list")
    if not isinstance(color_lenses, list) or not color_lenses:
        raise ValueError("color lenses must be a non-empty list")

    camera_allowed = {
        "name", "zoom_min", "zoom_max", "frame_count", "camera_center_offset_ratio"
    }
    color_allowed = {"name", "temporal_color_jump", "color"}
    checked_camera_lenses = []
    for index, lens in enumerate(camera_lenses):
        checked_camera_lenses.append(
            _validate_fragment_keys(
                lens,
                f"camera lens[{index}]",
                camera_allowed,
                camera_allowed,
            )
        )
    color_by_name = {}
    for index, lens in enumerate(color_lenses):
        checked = _validate_fragment_keys(
            lens,
            f"color lens[{index}]",
            color_allowed,
            color_allowed,
        )
        name = checked["name"]
        if name in color_by_name:
            raise ValueError(f"duplicate color lens name: {name}")
        color_by_name[name] = checked

    camera_names = [lens["name"] for lens in checked_camera_lenses]
    if len(camera_names) != len(set(camera_names)):
        raise ValueError(f"duplicate camera lens names: {camera_names}")
    if set(camera_names) != set(color_by_name):
        raise ValueError(
            "camera/color lens names do not match: "
            f"camera={camera_names}, color={list(color_by_name)}"
        )

    merged_lenses = []
    for camera_lens in checked_camera_lenses:
        color_lens = color_by_name[camera_lens["name"]]
        merged_lenses.append(
            {
                **camera_lens,
                "temporal_color_jump": color_lens["temporal_color_jump"],
                "color": color_lens["color"],
            }
        )

    merged_zoom = {
        "curve": camera_zoom["curve"],
        "frame_allocation": camera_zoom["frame_allocation"],
        "lens_switch": {**camera_switch, **color_switch},
        "temporal_color_adjustment": color_bundle["temporal_color_adjustment"],
        "lenses": merged_lenses,
    }
    merged = {
        **scene_raw,
        "render": {**scene_render, "background_rgb": color_bundle["background_rgb"]},
        "camera": camera_raw["camera"],
        "zoom": merged_zoom,
    }
    cfg = GeneratorConfig.model_validate(merged)
    if camera_json_path is not None:
        cfg = apply_traversal_camera_json(cfg, camera_json_path)
    return _check_output(cfg, check_output)


def prepare_output(cfg: GeneratorConfig) -> None:
    out = cfg.output.directory
    if out.exists() and cfg.output.overwrite:
        # Delete only generated, known outputs; retain unrelated user files.
        for name in ("frames", "linear_frames", "alpha_masks", "camera_alignment"):
            p = out / name
            if p.exists():
                shutil.rmtree(p)
        for name in (
            cfg.output.video_filename,
            "camera_trajectory.json",
            "frame_metadata.json",
            "generation_summary.json",
            "resolved_config.yaml",
        ):
            p = out / name
            if p.is_file():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    if cfg.output.save_processed_frames:
        (out / "frames").mkdir(exist_ok=True)
    if cfg.output.save_linear_frames:
        (out / "linear_frames").mkdir(exist_ok=True)
    if cfg.output.save_alpha_masks:
        (out / "alpha_masks").mkdir(exist_ok=True)


def save_resolved_config(cfg: GeneratorConfig) -> None:
    data = cfg.model_dump(mode="json")
    with (cfg.output.directory / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
