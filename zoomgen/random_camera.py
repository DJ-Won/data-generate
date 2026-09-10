"""Small, repeatable variations of one phone camera rig per zoom video."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from pydantic import ValidationError

from .config import GeneratorConfig
from .zoom import allocate_frame_counts, build_schedule


RANDOM_CAMERA_VERSION = 1


def camera_random_seed(base_seed: int, scene_key: str, camera_key: str) -> int:
    payload = json.dumps([base_seed, scene_key, camera_key], ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _perturb_zoom_ranges(cfg: GeneratorConfig, rng: np.random.Generator) -> GeneratorConfig:
    """Move both sides of each switch together, preserving its zoom gap."""
    lenses = cfg.zoom.lenses
    displacements = [
        float(rng.uniform(-0.12, 0.12))
        * min(left.zoom_max - left.zoom_min, right.zoom_max - right.zoom_min)
        for left, right in zip(lenses, lenses[1:])
    ]
    # Proportional allocation can cross a frame-count threshold even for a tiny
    # change. Reduce the variation until all configured transitions still fit.
    for scale in [2.0 ** -i for i in range(12)] + [0.0]:
        candidate = cfg.model_copy(deep=True)
        for i, displacement in enumerate(displacements):
            candidate.zoom.lenses[i].zoom_max += displacement * scale
            candidate.zoom.lenses[i + 1].zoom_min += displacement * scale
        try:
            candidate = GeneratorConfig.model_validate(candidate.model_dump())
        except ValidationError:
            continue
        schedule = build_schedule(candidate)
        zooms = np.asarray([frame.zoom_ratio for frame in schedule])
        if (
            np.all(np.diff(zooms) > 0.0)
            and zooms[0] == lenses[0].zoom_min
            and zooms[-1] == lenses[-1].zoom_max
        ):
            return candidate
    raise ValueError("camera randomization requires a strictly increasing zoom schedule")


def randomize_camera_config(
    cfg: GeneratorConfig, seed: int
) -> tuple[GeneratorConfig, dict]:
    """Return a perturbed copy; keep rig parameters fixed throughout the video.

    Offsets use the existing scene-radius units and shared camera axes. The 1x
    focal baseline stays fixed so the two endpoint magnifications stay fixed.
    """
    rng = np.random.default_rng(seed)
    randomized = _perturb_zoom_ranges(cfg, rng)
    lenses = randomized.zoom.lenses

    # A shared shift plus small module-specific residuals keeps every lens near
    # the original viewpoint, without requiring any particular module layout.
    common_offset = rng.uniform(-1.0, 1.0, 3) * [0.001, 0.001, 0.0002]
    for lens in lenses:
        residual = rng.uniform(-1.0, 1.0, 3) * [0.00075, 0.00075, 0.00015]
        lens.camera_center_offset_ratio = tuple(
            np.asarray(lens.camera_center_offset_ratio) + common_offset + residual
        )
    motion = randomized.camera.motion
    if not motion.enabled:
        # The camera builder gates module offsets with motion.enabled. Activate
        # offsets without adding hand movement to a configured static capture.
        motion.enabled = True
        motion.translation_amplitude_ratio = (0.0, 0.0, 0.0)
        motion.rotation_amplitude_deg = (0.0, 0.0, 0.0)
    else:
        motion.translation_amplitude_ratio = tuple(
            np.asarray(motion.translation_amplitude_ratio) * rng.uniform(0.85, 1.15, 3)
        )
        motion.rotation_amplitude_deg = tuple(
            np.asarray(motion.rotation_amplitude_deg) * rng.uniform(0.85, 1.15, 3)
        )
    motion.seed = int(rng.integers(0, 2**32))
    randomized.camera.principal_point_offset_px = tuple(
        np.asarray(randomized.camera.principal_point_offset_px)
        + rng.uniform(-0.001, 0.001, 2) * [cfg.video.width, cfg.video.height]
    )

    common_ev = float(rng.uniform(-0.10, 0.10))
    common_wb = np.exp(rng.uniform(-0.015, 0.015, 3))
    common_gamma = float(rng.uniform(0.98, 1.02))
    original_ev = [lens.color.exposure_ev for lens in cfg.zoom.lenses]
    exposures = np.asarray(original_ev) + common_ev + rng.uniform(-0.06, 0.06, len(lenses))
    for i in range(1, len(lenses)):
        original_difference = original_ev[i] - original_ev[i - 1]
        difference = exposures[i] - exposures[i - 1]
        direction = np.sign(original_difference) or np.sign(difference) or 1.0
        minimum_difference = max(0.04, abs(original_difference) * 0.6)
        # Avoid accidentally canceling the configured sensor exposure contrast.
        if difference * direction < minimum_difference:
            exposures[i] = exposures[i - 1] + direction * minimum_difference

    counts = allocate_frame_counts(randomized)
    transition = randomized.zoom.temporal_color_adjustment.transition_frames
    for lens, exposure, count in zip(lenses, exposures, counts):
        color = lens.color
        color.exposure_ev = float(exposure)
        color.white_balance_gains = tuple(
            np.asarray(color.white_balance_gains) * common_wb
            * np.exp(rng.uniform(-0.015, 0.015, 3))
        )
        matrix_delta = rng.uniform(-0.004, 0.004, (3, 3))
        # Preserve each row sum so the CCM variation does not shift neutral gain.
        matrix_delta -= matrix_delta.mean(axis=1, keepdims=True)
        color.color_correction_matrix = tuple(
            tuple(row) for row in np.asarray(color.color_correction_matrix) + matrix_delta
        )
        color.contrast *= float(rng.uniform(0.97, 1.03))
        color.saturation *= float(rng.uniform(0.96, 1.04))
        color.gamma *= common_gamma * float(rng.uniform(0.99, 1.01))
        color.vignette_strength = float(np.clip(
            color.vignette_strength * rng.uniform(0.85, 1.15), 0.0, 1.0
        ))
        color.noise_std *= float(rng.uniform(0.85, 1.15))
        jump = lens.temporal_color_jump
        jump.strength = float(np.clip(jump.strength * rng.uniform(0.85, 1.15), -1.0, 1.0))
        # A disabled temporal jump remains disabled. Existing jumps retain their
        # sign and must finish inside the (possibly reallocated) lens segment.
        last_jump_index = min(count - 1, count - transition)
        jump.position = float(np.clip(
            jump.position + rng.uniform(-0.05, 0.05),
            0.0, last_jump_index / (count - 1),
        ))

    randomized = GeneratorConfig.model_validate(randomized.model_dump())
    signature = randomized.model_dump(mode="json", exclude={"output"})
    resolved = signature["camera"]["initialization"]["resolved_pose"]
    if resolved is not None:
        # Moving an accepted position from invaild to vaild must not change the
        # identity of the camera configuration on a subsequent resume.
        resolved.pop("source", None)
    digest = hashlib.sha256(json.dumps(
        signature, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")).hexdigest()
    return randomized, {
        "version": RANDOM_CAMERA_VERSION,
        "seed": seed,
        "config_sha256": digest,
    }
