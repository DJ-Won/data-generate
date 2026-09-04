from __future__ import annotations

import numpy as np

from .config import ColorConfig, GeneratorConfig, LensConfig
from .zoom import FrameSchedule, allocate_frame_counts, temporal_state


def _soft_shoulder(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.8, x, 0.8 + 0.2 * (1.0 - np.exp(-(x - 0.8) / 0.2)))


def linear_color_stage(rgb: np.ndarray, color: ColorConfig, exposure_gain: float) -> np.ndarray:
    x = rgb.astype(np.float32, copy=False) * (2.0 ** color.exposure_ev) * exposure_gain
    x = x * np.asarray(color.white_balance_gains, dtype=np.float32)[None, None, :]
    x = x @ np.asarray(color.color_correction_matrix, dtype=np.float32).T
    x = x + color.black_level
    x = (x - 0.18) * color.contrast + 0.18
    luma = np.sum(x * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1, keepdims=True)
    x = luma + color.saturation * (x - luma)
    return _soft_shoulder(x) if color.tone_mapping == "soft_shoulder" else np.clip(x, 0.0, 1.0)


def _encode(linear: np.ndarray, gamma: float, vignette: float, noise_std: float,
            seed: int, frame_index: int) -> np.ndarray:
    x = np.power(np.clip(linear, 0.0, 1.0), 1.0 / gamma)
    if vignette > 0:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
        falloff = np.clip(1.0 - vignette * (xx * xx + yy * yy) / 2.0, 0.0, 1.0)
        x *= falloff[..., None]
    if noise_std > 0:
        rng = np.random.default_rng(np.random.SeedSequence([seed, frame_index]))
        x += rng.normal(0.0, noise_std, x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def process_frame(rgb: np.ndarray, schedule: FrameSchedule, cfg: GeneratorConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    lens = cfg.zoom.lenses[schedule.lens_index]
    state = temporal_state(lens, schedule.local_index, schedule.lens_frame_count, cfg)
    current = linear_color_stage(rgb, lens.color, state["gain"])
    gamma = lens.color.gamma
    vignette = lens.color.vignette_strength
    noise = state["noise_std"]
    transition_meta = None
    if schedule.transition_from_lens is not None and schedule.color_blend < 1.0:
        prev = cfg.zoom.lenses[schedule.transition_from_lens]
        # Previous sensor remains in its final temporal state during hand-off.
        prev_count = allocate_frame_counts(cfg)[schedule.transition_from_lens]
        prev_state = temporal_state(prev, prev_count - 1, prev_count, cfg)
        previous = linear_color_stage(rgb, prev.color, prev_state["gain"])
        a = schedule.color_blend
        current = previous * (1.0 - a) + current * a
        gamma = prev.color.gamma * (1.0 - a) + gamma * a
        vignette = prev.color.vignette_strength * (1.0 - a) + vignette * a
        noise = prev_state["noise_std"] * (1.0 - a) + noise * a
        transition_meta = {"from_lens": prev.name, "blend": a}
    encoded = _encode(current, gamma, vignette, noise, cfg.camera.motion.seed, schedule.frame_index)
    meta = {
        "base_exposure_ev": lens.color.exposure_ev,
        "temporal_jump_strength": lens.temporal_color_jump.strength,
        "temporal_jump_position": lens.temporal_color_jump.position,
        "temporal_jump_local_index": state["jump_local_index"],
        "temporal_jump_applied": state["applied"],
        "temporal_jump_progress": state["progress"],
        "temporal_jump_ev": state["jump_ev"],
        "temporal_applied_ev": state["applied_ev"],
        "temporal_exposure_gain": state["gain"],
        "effective_exposure_ev": lens.color.exposure_ev + state["applied_ev"],
        "effective_noise_std": noise,
        "white_balance_gains": list(lens.color.white_balance_gains),
        "color_correction_matrix": [list(x) for x in lens.color.color_correction_matrix],
        "black_level": lens.color.black_level,
        "contrast": lens.color.contrast,
        "saturation": lens.color.saturation,
        "gamma": gamma,
        "vignette_strength": vignette,
        "lens_color_transition": transition_meta,
    }
    return encoded, current, meta
