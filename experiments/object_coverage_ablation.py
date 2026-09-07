#!/usr/bin/env python3
"""Compare legacy and current object-camera coverage fitting on one fixed plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scene_traversal import (  # noqa: E402
    CapturePlan,
    PositionPlan,
    TraversalConfig,
    _camera_frame,
    _create_topiq_nr_metric,
    _fit_object_camera_coverage,
    _object_camera_at_distance,
    _object_gaussian_coverage_ratio,
    _renderer_adapter,
    _save_capture_pair,
    _topiq_nr_score,
    load_traversal_config_parts,
    plan_traversal,
)
from zoomgen.camera import CameraFrame  # noqa: E402
from zoomgen.quality import screen_space_quality_metrics  # noqa: E402
from zoomgen.renderer import GaussianRenderer  # noqa: E402
from zoomgen.scene import GaussianScene  # noqa: E402


SCHEMA_VERSION = 1
EXPERIMENT_KIND = "object_coverage_ablation"
TOP_K = 10
LEGACY_MAX_ITERATIONS = 6
STRATEGY_LEGACY = "legacy_adaptive"
STRATEGY_CURRENT = "current_bracketed"
STRATEGIES = (STRATEGY_LEGACY, STRATEGY_CURRENT)
DEFAULT_CAMERA_CONFIG = (
    REPOSITORY_ROOT / "configs/travel/cameras/object_random.yaml"
)
DEFAULT_COLOR_CONFIG = REPOSITORY_ROOT / "configs/travel/colors/default.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _append_jsonl(path: Path, value: dict) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        view = memoryview(encoded)
        while view:
            written = handle.write(view)
            if written is None or written <= 0:
                raise OSError(f"failed to append experiment result: {path}")
            view = view[written:]
        os.fsync(handle.fileno())


def _load_and_repair_jsonl(path: Path) -> list[dict]:
    """Load records and discard only an incomplete final write."""
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r+b", buffering=0) as handle:
        file_size = os.fstat(handle.fileno()).st_size
        while handle.tell() < file_size:
            line_start = handle.tell()
            raw_line = handle.readline()
            at_end = handle.tell() == file_size
            has_newline = raw_line.endswith(b"\n")
            payload = raw_line.rstrip(b"\r\n")
            if not payload:
                if at_end:
                    handle.truncate(line_start)
                    break
                raise RuntimeError(
                    f"blank line inside experiment JSONL at byte {line_start}: {path}"
                )
            try:
                record = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if at_end and not has_newline:
                    handle.truncate(line_start)
                    break
                raise RuntimeError(
                    f"corrupt experiment JSONL at byte {line_start}: {path}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"experiment JSONL record is not an object at byte {line_start}"
                )
            records.append(record)
            if at_end and not has_newline:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                os.fsync(handle.fileno())
    return records


def _resolved_config(scene_config: Path, device: str) -> TraversalConfig:
    previous_directory = Path.cwd()
    try:
        # Repository configs intentionally resolve relative PLY paths from the repo root.
        os.chdir(REPOSITORY_ROOT)
        cfg = load_traversal_config_parts(
            scene_config,
            DEFAULT_CAMERA_CONFIG,
            DEFAULT_COLOR_CONFIG,
            check_output=False,
        )
        raw = cfg.model_dump(mode="python")
        ply_path = Path(raw["input"]["ply_path"])
        if not ply_path.is_absolute():
            raw["input"]["ply_path"] = str((REPOSITORY_ROOT / ply_path).resolve())
        raw["render"]["device"] = device
        raw["initialization"]["traversal"]["random"]["device"] = device
        cfg = TraversalConfig.model_validate(raw)
    finally:
        os.chdir(previous_directory)

    if cfg.scene.scene_type != "object":
        raise ValueError(
            f"coverage ablation requires scene.scene_type=object, got {cfg.scene.scene_type!r}"
        )
    if cfg.initialization.traversal.strategy != "random":
        raise ValueError("the fixed ablation camera config must use random traversal")
    return cfg


def _config_fingerprint(cfg: TraversalConfig, scene_config: Path) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_config": str(scene_config),
        "camera_config": str(DEFAULT_CAMERA_CONFIG),
        "color_config": str(DEFAULT_COLOR_CONFIG),
        "config": cfg.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ensure_experiment_directory(output_root: Path, identity: dict) -> None:
    marker_path = output_root / "experiment.json"
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"output root is not a directory: {output_root}")
        entries = list(output_root.iterdir())
        if entries and not marker_path.is_file():
            raise ValueError(
                f"refusing to use non-empty non-experiment output directory: {output_root}"
            )
    else:
        output_root.mkdir(parents=True)

    if marker_path.exists():
        with marker_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        required_matches = (
            "kind",
            "schema_version",
            "config_fingerprint",
            "input_ply_path",
            "source_files_sha256",
            "strategies",
            "top_k",
            "full_candidate_pool",
            "early_termination",
            "legacy_max_iterations",
            "current_max_iterations",
        )
        mismatches = [
            key for key in required_matches if existing.get(key) != identity.get(key)
        ]
        if mismatches:
            raise ValueError(
                "experiment output belongs to a different run; mismatched fields: "
                + ", ".join(mismatches)
            )
    else:
        _atomic_write_json(marker_path, identity)


def _serialize_position(position: PositionPlan) -> dict:
    return {
        "index": position.index,
        "label": position.label,
        "position": position.position.tolist(),
        "offset_from_seed": position.offset_from_seed.tolist(),
        "clearance": position.clearance,
        "seed_position_source": position.seed_position_source,
        "look_at_target": (
            position.look_at_target.tolist()
            if position.look_at_target is not None
            else None
        ),
    }


def _serialize_capture(capture: CapturePlan) -> dict:
    return {
        "position_index": capture.position.index,
        "lens_index": capture.lens_index,
        "lens_label": capture.lens_label,
        "yaw_deg": capture.yaw_deg,
        "pitch_deg": capture.pitch_deg,
        "roll_deg": capture.roll_deg,
        "fov_y_deg": capture.fov_y_deg,
    }


def _candidate_plan_sha256(captures: list[CapturePlan]) -> str:
    payload = [
        {
            "candidate_index": candidate_index,
            "position": _serialize_position(capture.position),
            "capture": _serialize_capture(capture),
        }
        for candidate_index, capture in enumerate(captures)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deserialize_plan(
    cfg: TraversalConfig, value: dict
) -> tuple[list[PositionPlan], list[CapturePlan]]:
    resolved_seed = value.get("resolved_seed_position")
    if resolved_seed is None:
        raise RuntimeError("saved plan has no resolved_seed_position")
    cfg.initialization.reference_match.seed_pose.position = tuple(
        float(component) for component in resolved_seed
    )

    positions = []
    by_index: dict[int, PositionPlan] = {}
    for item in value.get("positions", []):
        target = item.get("look_at_target")
        position = PositionPlan(
            int(item["index"]),
            str(item["label"]),
            np.asarray(item["position"], dtype=np.float64),
            np.asarray(item["offset_from_seed"], dtype=np.float64),
            float(item["clearance"]),
            str(item["seed_position_source"]),
            np.asarray(target, dtype=np.float64) if target is not None else None,
        )
        if position.index in by_index:
            raise RuntimeError(f"duplicate position index in saved plan: {position.index}")
        positions.append(position)
        by_index[position.index] = position

    captures = []
    for item in value.get("captures", []):
        position_index = int(item["position_index"])
        if position_index not in by_index:
            raise RuntimeError(
                f"capture refers to missing position index: {position_index}"
            )
        captures.append(
            CapturePlan(
                by_index[position_index],
                int(item["lens_index"]),
                str(item["lens_label"]),
                float(item["yaw_deg"]),
                float(item["pitch_deg"]),
                float(item["roll_deg"]),
                float(item["fov_y_deg"]),
            )
        )
    if not positions or not captures:
        raise RuntimeError("saved experiment plan is empty")
    return positions, captures


def _load_or_create_plan(
    path: Path,
    cfg: TraversalConfig,
    scene: GaussianScene,
    fingerprint: str,
) -> tuple[list[PositionPlan], list[CapturePlan]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("config_fingerprint") != fingerprint:
            raise ValueError("saved plan does not match the current experiment config")
        positions, captures = _deserialize_plan(cfg, value)
        candidate_plan_sha256 = _candidate_plan_sha256(captures)
        if value.get("candidate_plan_sha256") != candidate_plan_sha256:
            raise ValueError("saved candidate plan hash does not match its contents")
        return positions, captures

    positions, captures = plan_traversal(cfg, scene)
    if len(captures) < TOP_K:
        raise RuntimeError(
            f"fixed candidate plan has only {len(captures)} captures; at least {TOP_K} required"
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": fingerprint,
        "created_at": _utc_now(),
        "resolved_seed_position": list(
            cfg.initialization.reference_match.seed_pose.position
        ),
        "position_count": len(positions),
        "capture_count": len(captures),
        "candidate_plan_sha256": _candidate_plan_sha256(captures),
        "positions": [_serialize_position(item) for item in positions],
        "captures": [_serialize_capture(item) for item in captures],
    }
    _atomic_write_json(path, value)
    return positions, captures


def _serialize_camera(camera: CameraFrame) -> dict:
    return {
        "position": camera.position.tolist(),
        "target": camera.target.tolist(),
        "camera_to_world": camera.c2w.tolist(),
        "world_to_camera": camera.w2c.tolist(),
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "fov_x": camera.fov_x,
        "fov_y": camera.fov_y,
        "near": camera.near,
        "far": camera.far,
        "camera_center_offset": camera.camera_center_offset.tolist(),
    }


def _deserialize_camera(value: dict) -> CameraFrame:
    return CameraFrame(
        np.asarray(value["position"], dtype=np.float64),
        np.asarray(value["target"], dtype=np.float64),
        np.asarray(value["camera_to_world"], dtype=np.float64),
        np.asarray(value["world_to_camera"], dtype=np.float64),
        float(value["fx"]),
        float(value["fy"]),
        float(value["cx"]),
        float(value["cy"]),
        float(value["fov_x"]),
        float(value["fov_y"]),
        float(value["near"]),
        float(value["far"]),
        np.asarray(value["camera_center_offset"], dtype=np.float64),
    )


def _fit_legacy_adaptive(
    cfg: TraversalConfig,
    scene: GaussianScene,
    renderer: GaussianRenderer,
    camera: CameraFrame,
    clearance_tree: cKDTree,
) -> tuple[CameraFrame, dict]:
    """Local copy of the pre-bracket adaptive radial coverage search."""
    coverage_cfg = cfg.initialization.object.coverage
    center = np.asarray(scene.analysis.center, dtype=np.float64)
    outward = np.asarray(camera.position, dtype=np.float64) - center
    initial_distance = float(np.linalg.norm(outward))
    if initial_distance < 1e-12:
        raise ValueError("object camera position cannot equal the scene center")
    outward /= initial_distance
    lower = np.asarray(scene.analysis.aabb_min, dtype=np.float64)
    upper = np.asarray(scene.analysis.aabb_max, dtype=np.float64)
    boundary_distances = []
    for axis, component in enumerate(outward):
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
    legacy_exterior_distance = max(
        min(positive_distances)
        + coverage_cfg.exterior_margin_radius_ratio * float(scene.analysis.radius),
        np.finfo(np.float64).eps,
    )
    minimum_distance = min(
        initial_distance,
        legacy_exterior_distance,
    )
    preview_history: list[dict] = []
    output_history: list[dict] = []

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

    current_distance = initial_distance
    current_camera, current_ratio, current_clearance = evaluate(
        current_distance,
        preview_history,
    )
    initial_ratio = current_ratio
    target_ratio = coverage_cfg.minimum_pixel_ratio
    best_camera = current_camera
    best_ratio = current_ratio
    best_clearance = current_clearance
    best_distance = current_distance
    constraints_met = current_ratio >= target_ratio

    while (
        not constraints_met
        and len(preview_history) < LEGACY_MAX_ITERATIONS
        and current_distance > minimum_distance
    ):
        if len(preview_history) == LEGACY_MAX_ITERATIONS - 1:
            next_distance = minimum_distance
        else:
            projected_scale = math.sqrt(
                max(current_ratio, 1e-6) / max(target_ratio, 1e-6)
            )
            step_ratio = float(np.clip(projected_scale, 0.5, 0.9))
            next_distance = max(minimum_distance, current_distance * step_ratio)
        if next_distance >= current_distance:
            next_distance = minimum_distance
        if next_distance >= current_distance:
            break
        current_distance = next_distance
        current_camera, current_ratio, current_clearance = evaluate(
            current_distance,
            preview_history,
        )
        if (current_ratio, current_distance) > (best_ratio, best_distance):
            best_camera = current_camera
            best_ratio = current_ratio
            best_clearance = current_clearance
            best_distance = current_distance
        constraints_met = current_ratio >= target_ratio

    preview_constraints_met = constraints_met
    preview_selected_distance = best_distance
    preview_selected_ratio = best_ratio
    output_width = cfg.initialization.reference_match.preview_width
    output_height = cfg.initialization.reference_match.preview_height
    best_camera, best_ratio, best_clearance = evaluate(
        best_distance,
        output_history,
        output_width,
        output_height,
    )
    constraints_met = best_ratio >= target_ratio
    if not constraints_met and best_distance > minimum_distance:
        projected_scale = math.sqrt(
            max(best_ratio, 1e-6) / max(target_ratio, 1e-6)
        )
        correction_distance = max(
            minimum_distance,
            best_distance * float(np.clip(projected_scale, 0.5, 0.9)),
        )
        if correction_distance >= best_distance:
            correction_distance = minimum_distance
        correction_camera, correction_ratio, correction_clearance = evaluate(
            correction_distance,
            output_history,
            output_width,
            output_height,
        )
        if (correction_ratio, correction_distance) > (
            best_ratio,
            best_distance,
        ):
            best_camera = correction_camera
            best_ratio = correction_ratio
            best_clearance = correction_clearance
            best_distance = correction_distance
        constraints_met = best_ratio >= target_ratio
        if not constraints_met and correction_distance > minimum_distance:
            boundary_camera, boundary_ratio, boundary_clearance = evaluate(
                minimum_distance,
                output_history,
                output_width,
                output_height,
            )
            if (boundary_ratio, minimum_distance) > (
                best_ratio,
                best_distance,
            ):
                best_camera = boundary_camera
                best_ratio = boundary_ratio
                best_clearance = boundary_clearance
                best_distance = minimum_distance
            constraints_met = best_ratio >= target_ratio

    final_distance = float(np.linalg.norm(best_camera.position - center))
    metadata = {
        "method": "preview_adaptive_search_with_output_validation",
        "alpha_threshold": coverage_cfg.alpha_threshold,
        "minimum_pixel_ratio": target_ratio,
        "preview_size": [coverage_cfg.preview_width, coverage_cfg.preview_height],
        "output_validation_size": [output_width, output_height],
        "preview_constraints_met": preview_constraints_met,
        "output_constraints_met": constraints_met,
        "constraints_met": constraints_met,
        "camera_moved_closer": bool(final_distance < initial_distance),
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


def _candidate_record(
    candidate_index: int,
    capture: CapturePlan,
    strategy: str,
    camera: CameraFrame,
    coverage: dict,
    score: float | None,
    fit_seconds: float,
    render_score_seconds: float,
) -> dict:
    coverage_qualified = bool(
        coverage.get("output_constraints_met", coverage["constraints_met"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if coverage_qualified else "coverage_rejected",
        "completed_at": _utc_now(),
        "candidate_index": candidate_index,
        "candidate_id": (
            f"position_{capture.position.index}:lens_{capture.lens_index}"
        ),
        "strategy": strategy,
        "capture": _serialize_capture(capture),
        "planned_position": capture.position.position.tolist(),
        "camera": _serialize_camera(camera),
        "coverage": coverage,
        "coverage_qualified_at_output": coverage_qualified,
        "topiq_nr_score": score,
        "timing_seconds": {
            "coverage_fit": fit_seconds,
            "rgb_render_and_score": render_score_seconds,
            "total": fit_seconds + render_score_seconds,
        },
    }


def _terminal_records_by_key(records: list[dict]) -> dict[tuple[int, str], dict]:
    terminal: dict[tuple[int, str], dict] = {}
    for record in records:
        if record.get("status") not in {"ok", "coverage_rejected"}:
            continue
        strategy = record.get("strategy")
        if strategy not in STRATEGIES:
            continue
        key = (int(record["candidate_index"]), str(strategy))
        terminal[key] = record
    return terminal


def _run_candidates(
    cfg: TraversalConfig,
    scene: GaussianScene,
    renderer: GaussianRenderer,
    captures: list[CapturePlan],
    clearance_tree: cKDTree,
    metric,
    torch_module,
    results_path: Path,
    existing_records: list[dict],
) -> list[dict]:
    terminal = _terminal_records_by_key(existing_records)
    fitters: dict[
        str,
        Callable[
            [TraversalConfig, GaussianScene, GaussianRenderer, CameraFrame, cKDTree],
            tuple[CameraFrame, dict],
        ],
    ] = {
        STRATEGY_LEGACY: _fit_legacy_adaptive,
        STRATEGY_CURRENT: _fit_object_camera_coverage,
    }
    width = cfg.initialization.reference_match.preview_width
    height = cfg.initialization.reference_match.preview_height
    device = cfg.initialization.traversal.random.device

    for candidate_index, capture in enumerate(
        tqdm(captures, desc="Coverage ablation", unit="candidate")
    ):
        base_camera = _camera_frame(cfg, scene, capture)
        strategy_order = STRATEGIES if candidate_index % 2 == 0 else STRATEGIES[::-1]
        for strategy in strategy_order:
            key = (candidate_index, strategy)
            if key in terminal:
                continue
            try:
                fit_started = time.perf_counter()
                fitted_camera, coverage = fitters[strategy](
                    cfg, scene, renderer, base_camera, clearance_tree
                )
                fit_seconds = time.perf_counter() - fit_started

                coverage_qualified = bool(
                    coverage.get(
                        "output_constraints_met",
                        coverage["constraints_met"],
                    )
                )
                score = None
                render_score_seconds = 0.0
                if coverage_qualified:
                    render_started = time.perf_counter()
                    rgb = renderer.render(fitted_camera, width, height)
                    score = _topiq_nr_score(
                        rgb, cfg, metric, torch_module, device
                    )
                    render_score_seconds = time.perf_counter() - render_started
                record = _candidate_record(
                    candidate_index,
                    capture,
                    strategy,
                    fitted_camera,
                    coverage,
                    score,
                    fit_seconds,
                    render_score_seconds,
                )
                _append_jsonl(results_path, record)
                existing_records.append(record)
                terminal[key] = record
            except Exception as error:
                _append_jsonl(
                    results_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "error",
                        "failed_at": _utc_now(),
                        "candidate_index": candidate_index,
                        "candidate_id": (
                            f"position_{capture.position.index}:lens_{capture.lens_index}"
                        ),
                        "strategy": strategy,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise

    expected_keys = {
        (candidate_index, strategy)
        for candidate_index in range(len(captures))
        for strategy in STRATEGIES
    }
    actual_keys = set(terminal)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        raise RuntimeError(
            "experiment results are incomplete: "
            f"{len(actual_keys)}/{len(expected_keys)} terminal records "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return existing_records


def _save_strategy_top_k(
    cfg: TraversalConfig,
    renderer: GaussianRenderer,
    metric,
    torch_module,
    strategy: str,
    records: list[dict],
    output_root: Path,
) -> dict:
    ranked = sorted(
        (
            record
            for record in records
            if record.get("strategy") == strategy
            and record.get("status") == "ok"
            and record.get("coverage_qualified_at_output") is True
            and record.get("topiq_nr_score") is not None
            and math.isfinite(float(record["topiq_nr_score"]))
        ),
        key=lambda record: (
            -float(record["topiq_nr_score"]),
            int(record["candidate_index"]),
        ),
    )
    if len(ranked) < TOP_K:
        raise RuntimeError(
            f"strategy {strategy} produced only {len(ranked)} results; {TOP_K} required"
        )

    score_threshold = float(
        cfg.initialization.traversal.random.topiq_nr_threshold_l
    )
    threshold_pass_count = sum(
        float(record["topiq_nr_score"]) > score_threshold for record in ranked
    )
    fallback_used = threshold_pass_count < TOP_K
    strategy_root = output_root / "strategies" / strategy
    width = cfg.initialization.reference_match.preview_width
    height = cfg.initialization.reference_match.preview_height
    device = cfg.initialization.traversal.random.device
    manifest_items = []
    for rank, record in enumerate(ranked[:TOP_K], start=1):
        camera = _deserialize_camera(record["camera"])
        rgb, radii = renderer.render(
            camera, width, height, return_radii=True
        )
        geometry_quality = screen_space_quality_metrics(
            renderer.g, camera, radii, width, height
        )
        del radii
        rerender_score = _topiq_nr_score(
            rgb, cfg, metric, torch_module, device
        )
        score_delta = rerender_score - float(record["topiq_nr_score"])
        if abs(score_delta) > 1e-5:
            raise RuntimeError(
                f"strategy {strategy} rank {rank} rerender score drifted by "
                f"{score_delta:.8g}, exceeding tolerance 1e-5"
            )
        rank_root = strategy_root / f"rank_{rank:02d}"
        image_path = rank_root / "image.png"
        metadata_path = rank_root / "metadata.json"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "experiment_kind": EXPERIMENT_KIND,
            "strategy": strategy,
            "selection_policy": "top_k_by_topiq_nr_over_full_candidate_pool",
            "selection_rank": rank,
            "selection_score": record["topiq_nr_score"],
            "score_threshold": score_threshold,
            "score_threshold_operator": ">",
            "passes_score_threshold": (
                float(record["topiq_nr_score"]) > score_threshold
            ),
            "fallback_used": fallback_used,
            "rerender_score": rerender_score,
            "rerender_score_delta": score_delta,
            "rerender_score_tolerance": 1e-5,
            "candidate_index": record["candidate_index"],
            "candidate_id": record["candidate_id"],
            "capture": record["capture"],
            "planned_position": record["planned_position"],
            "camera": record["camera"],
            "coverage": record["coverage"],
            "coverage_qualified_at_output": record[
                "coverage_qualified_at_output"
            ],
            "geometry_quality": geometry_quality,
            "source_timing_seconds": record["timing_seconds"],
        }
        _save_capture_pair(
            image_path, metadata_path, rgb, metadata, cfg
        )
        manifest_items.append(
            {
                "rank": rank,
                "candidate_index": record["candidate_index"],
                "candidate_id": record["candidate_id"],
                "topiq_nr_score": record["topiq_nr_score"],
                "rerender_score": rerender_score,
                "coverage_qualified_at_output": record[
                    "coverage_qualified_at_output"
                ],
                "image": str(image_path.relative_to(output_root)),
                "metadata": str(metadata_path.relative_to(output_root)),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "candidate_result_count": len(ranked),
        "top_k": TOP_K,
        "score_threshold": score_threshold,
        "score_threshold_operator": ">",
        "threshold_pass_count": threshold_pass_count,
        "fallback_used": fallback_used,
        "items": manifest_items,
    }
    _atomic_write_json(strategy_root / "top10.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run legacy adaptive and current bracketed object coverage fitting "
            "on one persistent candidate plan."
        )
    )
    parser.add_argument(
        "--scene-config",
        required=True,
        help="object scene YAML from configs/travel/scenes",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="new or resumable experiment directory; non-experiment data is refused",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device used by the renderer and TOPIQ-NR (default: cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scene_config = Path(args.scene_config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not scene_config.is_file():
        raise ValueError(f"scene config does not exist: {scene_config}")
    if not args.device.startswith("cuda"):
        raise ValueError("diff_gaussian_rasterization requires a CUDA device")

    cfg = _resolved_config(scene_config, args.device)
    current_max_iterations = cfg.initialization.object.coverage.max_iterations
    if current_max_iterations != 8:
        raise ValueError(
            "the registered current strategy requires coverage.max_iterations=8, "
            f"got {current_max_iterations}"
        )
    configured_output = cfg.output.root_directory
    if not configured_output.is_absolute():
        configured_output = REPOSITORY_ROOT / configured_output
    configured_scene_directory = (
        configured_output / cfg.output.scene_name
    ).resolve()
    if output_root == configured_scene_directory:
        raise ValueError(
            "experiment output must differ from the scene's production output directory"
        )

    fingerprint = _config_fingerprint(cfg, scene_config)
    identity = {
        "kind": EXPERIMENT_KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "config_fingerprint": fingerprint,
        "scene_config": str(scene_config),
        "camera_config": str(DEFAULT_CAMERA_CONFIG),
        "color_config": str(DEFAULT_COLOR_CONFIG),
        "input_ply_path": str(cfg.input.ply_path),
        "device": args.device,
        "strategies": list(STRATEGIES),
        "top_k": TOP_K,
        "full_candidate_pool": True,
        "early_termination": False,
        "legacy_max_iterations": LEGACY_MAX_ITERATIONS,
        "current_max_iterations": current_max_iterations,
        "minimum_pixel_ratio": (
            cfg.initialization.object.coverage.minimum_pixel_ratio
        ),
        "topiq_nr_threshold_l": (
            cfg.initialization.traversal.random.topiq_nr_threshold_l
        ),
        "source_files_sha256": {
            "experiment_harness": _file_sha256(Path(__file__).resolve()),
            "scene_traversal": _file_sha256(
                REPOSITORY_ROOT / "scene_traversal.py"
            ),
        },
    }
    _ensure_experiment_directory(output_root, identity)

    lock_path = output_root / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another ablation process is using {output_root}"
            ) from error

        adapter = _renderer_adapter(cfg)
        print(f"Loading and analyzing object scene: {cfg.input.ply_path}")
        scene = GaussianScene(cfg.input.ply_path, adapter)
        positions, captures = _load_or_create_plan(
            output_root / "fixed_plan.json",
            cfg,
            scene,
            fingerprint,
        )
        expected_candidate_count = (
            cfg.initialization.traversal.max_position_sampling_attempts
            * cfg.initialization.traversal.images_per_position_l
        )
        if len(captures) != expected_candidate_count:
            raise RuntimeError(
                f"fixed candidate plan is not the full configured pool: "
                f"{len(captures)}/{expected_candidate_count}"
            )
        print(
            f"Fixed plan: {len(positions)} positions, {len(captures)} candidates, "
            f"{len(captures) * len(STRATEGIES)} strategy evaluations"
        )

        print(f"Loading {scene.analysis.effective_gaussian_count:,} Gaussians...")
        renderer = GaussianRenderer(scene.load_tensors(), adapter)
        clearance_points = scene.effective_position_sample()
        if not len(clearance_points):
            raise RuntimeError("no effective points are available for coverage fitting")
        clearance_tree = cKDTree(clearance_points)
        metric, torch_module = _create_topiq_nr_metric(
            cfg.initialization.traversal.random.device
        )

        results_path = output_root / "candidate_results.jsonl"
        records = _load_and_repair_jsonl(results_path)
        records = _run_candidates(
            cfg,
            scene,
            renderer,
            captures,
            clearance_tree,
            metric,
            torch_module,
            results_path,
            records,
        )
        terminal_records = list(_terminal_records_by_key(records).values())
        successful = [
            record for record in terminal_records if record.get("status") == "ok"
        ]
        threshold = float(
            cfg.initialization.traversal.random.topiq_nr_threshold_l
        )
        strategy_summaries = {}
        for strategy in STRATEGIES:
            strategy_terminal = [
                record
                for record in terminal_records
                if record.get("strategy") == strategy
            ]
            strategy_scored = [
                record
                for record in strategy_terminal
                if record.get("status") == "ok"
                and record.get("coverage_qualified_at_output") is True
                and record.get("topiq_nr_score") is not None
                and math.isfinite(float(record["topiq_nr_score"]))
            ]
            scores = [
                float(record["topiq_nr_score"]) for record in strategy_scored
            ]
            strategy_summaries[strategy] = {
                "terminal_count": len(strategy_terminal),
                "coverage_qualified_count": sum(
                    record.get("coverage_qualified_at_output") is True
                    for record in strategy_terminal
                ),
                "coverage_rejected_count": sum(
                    record.get("status") == "coverage_rejected"
                    for record in strategy_terminal
                ),
                "scored_count": len(scores),
                "threshold_pass_count": sum(
                    score > threshold for score in scores
                ),
                "total_seconds": sum(
                    float(record["timing_seconds"]["total"])
                    for record in strategy_terminal
                ),
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_median": float(np.median(scores)) if scores else None,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
            }

        by_candidate = {
            (int(record["candidate_index"]), str(record["strategy"])): record
            for record in terminal_records
        }
        paired_coverage = {
            "both_pass": 0,
            "legacy_only": 0,
            "current_only": 0,
            "neither_pass": 0,
        }
        for candidate_index in range(len(captures)):
            legacy_pass = (
                by_candidate[(candidate_index, STRATEGY_LEGACY)].get(
                    "coverage_qualified_at_output"
                )
                is True
            )
            current_pass = (
                by_candidate[(candidate_index, STRATEGY_CURRENT)].get(
                    "coverage_qualified_at_output"
                )
                is True
            )
            if legacy_pass and current_pass:
                paired_coverage["both_pass"] += 1
            elif legacy_pass:
                paired_coverage["legacy_only"] += 1
            elif current_pass:
                paired_coverage["current_only"] += 1
            else:
                paired_coverage["neither_pass"] += 1

        manifests = {
            strategy: _save_strategy_top_k(
                cfg,
                renderer,
                metric,
                torch_module,
                strategy,
                successful,
                output_root,
            )
            for strategy in STRATEGIES
        }
        summary = {
            **identity,
            "completed_at": _utc_now(),
            "position_count": len(positions),
            "candidate_count": len(captures),
            "candidate_plan_sha256": _candidate_plan_sha256(captures),
            "strategy_evaluation_count": len(terminal_records),
            "scored_strategy_evaluation_count": sum(
                item["scored_count"] for item in strategy_summaries.values()
            ),
            "strategy_summaries": strategy_summaries,
            "paired_output_coverage": paired_coverage,
            "results_jsonl": str(results_path.relative_to(output_root)),
            "manifests": {
                strategy: {
                    "path": f"strategies/{strategy}/top10.json",
                    "top_k": manifest["top_k"],
                }
                for strategy, manifest in manifests.items()
            },
        }
        _atomic_write_json(output_root / "summary.json", summary)
        print(f"Ablation complete: {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
