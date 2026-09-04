from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeneratorConfig, LensConfig


@dataclass(frozen=True)
class FrameSchedule:
    frame_index: int
    lens_index: int
    lens_name: str
    local_index: int
    lens_frame_count: int
    zoom_ratio: float
    transition_from_lens: int | None
    color_blend: float
    camera_blend: float


def allocate_frame_counts(cfg: GeneratorConfig) -> list[int]:
    lenses = cfg.zoom.lenses
    if cfg.zoom.frame_allocation.mode == "explicit":
        return [int(x.frame_count) for x in lenses]
    spans = np.array([x.zoom_max - x.zoom_min for x in lenses], dtype=np.float64)
    if np.all(spans == 0):
        spans[:] = 1
    raw = spans / spans.sum() * cfg.video.total_frames
    counts = np.floor(raw).astype(int)
    remainder = cfg.video.total_frames - int(counts.sum())
    # Largest remainder, then lens order: deterministic.
    order = sorted(range(len(lenses)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    if np.any(counts < 1):
        raise ValueError("total_frames is too small to allocate at least one frame to every lens")
    return counts.tolist()


def _blend(local_index: int, frames: int, mode: str) -> float:
    if mode == "hard" or frames <= 0:
        return 1.0
    if frames == 1:
        return 1.0
    if local_index >= frames:
        return 1.0
    return local_index / (frames - 1)


def build_schedule(cfg: GeneratorConfig) -> list[FrameSchedule]:
    counts = allocate_frame_counts(cfg)
    result: list[FrameSchedule] = []
    global_index = 0
    for lens_i, (lens, count) in enumerate(zip(cfg.zoom.lenses, counts)):
        t = np.linspace(0.0, 1.0, count, dtype=np.float64) if count > 1 else np.array([0.0])
        if cfg.zoom.curve == "smoothstep":
            t = t * t * (3.0 - 2.0 * t)
        zooms = lens.zoom_min + (lens.zoom_max - lens.zoom_min) * t
        for local_i, zoom in enumerate(zooms):
            result.append(
                FrameSchedule(
                    frame_index=global_index,
                    lens_index=lens_i,
                    lens_name=lens.name,
                    local_index=local_i,
                    lens_frame_count=count,
                    zoom_ratio=float(zoom),
                    transition_from_lens=lens_i - 1 if lens_i > 0 else None,
                    color_blend=(
                        _blend(local_i, cfg.zoom.lens_switch.color_transition_frames,
                               cfg.zoom.lens_switch.color_mode)
                        if lens_i > 0 else 1.0
                    ),
                    camera_blend=(
                        _blend(local_i, cfg.zoom.lens_switch.camera_transition_frames,
                               cfg.zoom.lens_switch.camera_offset_mode)
                        if lens_i > 0 else 1.0
                    ),
                )
            )
            global_index += 1
    assert len(result) == cfg.video.total_frames
    zooms = np.array([x.zoom_ratio for x in result])
    if np.any(np.diff(zooms) <= 0):
        raise ValueError("configured zoom schedule is not strictly increasing")
    return result


def temporal_state(lens: LensConfig, local_index: int, count: int, cfg: GeneratorConfig) -> dict:
    jump = lens.temporal_color_jump
    jump_index = round(jump.position * (count - 1))
    transition = cfg.zoom.temporal_color_adjustment.transition_frames
    if jump.strength == 0.0 or local_index < jump_index:
        progress = 0.0
    elif transition == 0:
        progress = 1.0
    else:
        usable = min(transition, count - jump_index)
        progress = min(1.0, (local_index - jump_index + 1) / usable)
    jump_ev = jump.strength * cfg.zoom.temporal_color_adjustment.max_exposure_jump_ev
    applied_ev = jump_ev * progress
    gain = 2.0 ** applied_ev
    noise = lens.color.noise_std
    if cfg.zoom.temporal_color_adjustment.couple_noise_to_iso:
        noise *= max(0.0, 1.0 + applied_ev * cfg.zoom.temporal_color_adjustment.noise_growth_per_ev)
    return {
        "jump_local_index": jump_index,
        "applied": bool(progress > 0.0 and jump.strength != 0.0),
        "progress": float(progress),
        "jump_ev": float(jump_ev),
        "applied_ev": float(applied_ev),
        "gain": float(gain),
        "noise_std": float(noise),
    }
