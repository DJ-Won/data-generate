#!/usr/bin/env python3
"""Render per-camera zoom videos after a persistent Qwen quality gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_TEMPLATE = (
    PROJECT_ROOT / "configs/zoom_video/scenes/dl3dv_001dccbc.yaml"
)
QWEN_PYTHON = Path("/home/wdj/miniconda3/envs/dg/bin/python")
QWEN_RESPONSE_PREFIX = "QWEN_PIPELINE_RESPONSE "
QWEN_TIMEOUT_SECONDS = 900.0
VALID_OUTPUT_DIRECTORY = "vaild"
INVALID_OUTPUT_DIRECTORY = "invaild"


class QwenWorkerUnavailable(RuntimeError):
    pass


class QwenEvaluationError(RuntimeError):
    pass


@dataclass
class ProcessedFrame:
    encoded: Any
    linear: Any
    color_metadata: dict[str, Any]


@dataclass
class PreparedTraversal:
    cfg: Any
    schedule: list[Any]
    cameras: list[Any]
    preview_metrics: list[tuple[float, float]]
    base: Any
    initialization_metadata: dict[str, Any]
    motion_scale: float
    warnings: list[str]
    random_camera: dict[str, Any] | None = None


@dataclass(frozen=True)
class PositionPaths:
    directory: Path
    lens_directory: Path
    validation_directory: Path
    capture_image: Path
    camera_json: Path
    validation_report: Path
    scene_config: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _position_paths(directory: Path) -> PositionPaths:
    lens_directory = directory / "lens_0000"
    validation_directory = directory / "validation"
    return PositionPaths(
        directory=directory,
        lens_directory=lens_directory,
        validation_directory=validation_directory,
        capture_image=lens_directory / "image.png",
        camera_json=lens_directory / "camera.json",
        validation_report=validation_directory / "validation_report.json",
        scene_config=validation_directory / "scene_config.yaml",
    )


def _rebase_path_values(value: Any, source: Path, destination: Path) -> Any:
    """Rebase absolute paths stored in nested JSON-compatible values."""
    if isinstance(value, dict):
        return {
            key: _rebase_path_values(item, source, destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebase_path_values(item, source, destination) for item in value]
    if isinstance(value, str):
        source_text = str(source)
        if value == source_text:
            return str(destination)
        prefix = source_text + "/"
        if value.startswith(prefix):
            return str(destination / value[len(prefix) :])
    return value


def _move_position_output(paths: PositionPaths, destination: Path) -> PositionPaths:
    if paths.directory == destination:
        return paths
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"position output target already exists: {destination}")
    paths.directory.replace(destination)
    moved = _position_paths(destination)
    for json_path in (
        moved.validation_report,
        moved.directory / "generation_summary.json",
    ):
        if not json_path.is_file():
            continue
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except json.JSONDecodeError:
            continue
        _write_json_atomic(
            json_path, _rebase_path_values(value, paths.directory, destination)
        )
    return moved


def _remove_invalid_cache(scene_directory: Path) -> bool:
    invalid_directory = scene_directory / INVALID_OUTPUT_DIRECTORY
    if invalid_directory.is_symlink():
        raise RuntimeError(
            f"refusing to remove symlinked invalid cache: {invalid_directory}"
        )
    if not invalid_directory.exists():
        return False
    if not invalid_directory.is_dir():
        raise RuntimeError(
            f"invalid cache path is not a directory: {invalid_directory}"
        )
    shutil.rmtree(invalid_directory)
    return True


def _save_image(path: Path, rgb: Any) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), 0, 255).astype(
        np.uint8
    )
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, 95]
        if path.suffix.lower() in (".jpg", ".jpeg")
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        if not cv2.imwrite(
            str(temporary), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), parameters
        ):
            raise RuntimeError(f"failed to save image: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class QwenClient:
    """Keep one Qwen process and model alive for all endpoint evaluations."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.selector: selectors.BaseSelector | None = None
        self.broken = False

    def start(self) -> None:
        if self.process is not None:
            return
        if not QWEN_PYTHON.is_file():
            self.broken = True
            raise QwenWorkerUnavailable(
                f"Qwen Python executable does not exist: {QWEN_PYTHON}"
            )
        self.process = subprocess.Popen(
            [
                str(QWEN_PYTHON),
                "-u",
                str(Path(__file__).resolve()),
                "--qwen-worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        response = self._receive("__ready__")
        if not response.get("ok"):
            self.broken = True
            raise QwenWorkerUnavailable(
                f"failed to load persistent Qwen model: {response.get('error', 'unknown error')}"
            )

    def _receive(self, request_id: str) -> dict[str, Any]:
        if self.process is None or self.selector is None:
            raise QwenWorkerUnavailable("Qwen worker has not been started")
        deadline = time.monotonic() + QWEN_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.broken = True
                raise QwenWorkerUnavailable(
                    f"Qwen worker timed out after {QWEN_TIMEOUT_SECONDS:.0f} seconds"
                )
            events = self.selector.select(timeout=min(remaining, 5.0))
            if not events:
                if self.process.poll() is not None:
                    self.broken = True
                    raise QwenWorkerUnavailable(
                        f"Qwen worker exited with status {self.process.returncode}"
                    )
                continue
            assert self.process.stdout is not None
            line = self.process.stdout.readline()
            if not line:
                self.broken = True
                raise QwenWorkerUnavailable(
                    f"Qwen worker closed its output with status {self.process.poll()}"
                )
            if not line.startswith(QWEN_RESPONSE_PREFIX):
                continue
            try:
                response = json.loads(line[len(QWEN_RESPONSE_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if response.get("id") == request_id:
                return response

    def evaluate(self, image_path: Path) -> dict[str, Any]:
        self.start()
        assert self.process is not None and self.process.stdin is not None
        request_id = uuid.uuid4().hex
        request = {
            "command": "evaluate",
            "id": request_id,
            "image": str(image_path.resolve()),
        }
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.broken = True
            raise QwenWorkerUnavailable("could not send a request to Qwen worker") from exc
        response = self._receive(request_id)
        if not response.get("ok"):
            raise QwenEvaluationError(str(response.get("error", "Qwen evaluation failed")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise QwenEvaluationError("Qwen worker returned a non-object result")
        return result

    def close(self) -> None:
        process = self.process
        selector = self.selector
        self.process = None
        self.selector = None
        if selector is not None:
            selector.close()
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"command": "shutdown", "id": "__shutdown__"}) + "\n"
                )
                process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=30)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()


def _qwen_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "你是严谨的3DGS渲染质量检测器。必须依据输入图像本身判断，"
                "不得套用固定答案或默认判定为正常。"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _qwen_worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=argparse.SUPPRESS)
    parser.add_argument("--model")
    args = parser.parse_args(arguments)
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        import qwen

        requested_model = str(qwen.DEFAULT_MODEL)
        requested_model = "/home/wdj/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"

        model_source, local_files_only = qwen.resolve_model_source(requested_model)
        processor = AutoProcessor.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            local_files_only=local_files_only,
            device_map=device,
            dtype="auto",
        )
        generation_config = deepcopy(model.generation_config)
        generation_config.max_new_tokens = 512
        generation_config.max_length = None
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50
        generation_config.return_dict_in_generate = False
    except Exception as exc:
        payload = {"id": "__ready__", "ok": False, "error": str(exc)}
        print(QWEN_RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)
        return 1

    print(
        QWEN_RESPONSE_PREFIX
        + json.dumps(
            {
                "id": "__ready__",
                "ok": True,
                "model": model_source,
                "device": device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            request_id = str(request.get("id", ""))
            if request.get("command") == "shutdown":
                return 0
            if request.get("command") != "evaluate":
                raise ValueError("unknown Qwen worker command")
            image_path = Path(request["image"]).expanduser().resolve()
            if not image_path.is_file():
                raise ValueError(f"image does not exist: {image_path}")
            inputs = processor.apply_chat_template(
                _qwen_messages(image_path, qwen.INSPECTION_PROMPT),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generated_text = processor.batch_decode(
                generated_ids[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            result = qwen.validate_result(qwen.json_object(generated_text))
            payload = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            payload = {
                "id": locals().get("request_id", ""),
                "ok": False,
                "error": str(exc),
            }
        print(QWEN_RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def _load_scene_template(ply_path: Path, output_directory: Path) -> dict[str, Any]:
    import yaml

    with SCENE_TEMPLATE.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"scene template is not a YAML mapping: {SCENE_TEMPLATE}")
    raw["input"]["ply_path"] = str(ply_path)
    raw["output"]["directory"] = str(output_directory)
    raw["output"]["overwrite"] = True
    return raw


def _write_scene_config(path: Path, template: dict[str, Any], output: Path) -> None:
    import yaml

    value = deepcopy(template)
    value["output"]["directory"] = str(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sync_position_artifacts(
    paths: PositionPaths,
    *,
    scene_directory: Path,
    scene_template: dict[str, Any],
    cfg: Any,
    metadata: dict[str, Any],
    result_entry: dict[str, Any],
) -> None:
    paths.lens_directory.mkdir(parents=True, exist_ok=True)
    paths.validation_directory.mkdir(parents=True, exist_ok=True)
    cfg.output.directory = paths.directory
    metadata["image_relative_path"] = str(
        paths.capture_image.relative_to(scene_directory)
    )
    _write_json_atomic(paths.camera_json, metadata)
    _write_scene_config(paths.scene_config, scene_template, paths.directory)
    if (paths.directory / "resolved_config.yaml").is_file():
        from zoomgen.config import save_resolved_config

        save_resolved_config(cfg)
    result_entry["camera_json"] = str(paths.camera_json)
    result_entry["validation_report"] = str(paths.validation_report)


def _prepare_traversal(
    cfg: Any, scene: Any, renderer: Any,
    *, random_camera: dict[str, Any] | None = None,
) -> PreparedTraversal:
    from zoomgen.alignment import resolved_pose_config
    from zoomgen.config import save_resolved_config
    from zoomgen.pipeline import precheck_trajectory, resolve_base_pose
    from zoomgen.zoom import build_schedule

    warnings_out: list[str] = []
    schedule = build_schedule(cfg)
    base, initialization_metadata = resolve_base_pose(
        cfg, scene, renderer, warnings_out
    )
    base, cameras, preview_metrics, motion_scale = precheck_trajectory(
        cfg, scene, renderer, schedule, base, warnings_out
    )
    cfg.camera.initialization.resolved_pose = resolved_pose_config(base)
    cfg.camera.fov_y_deg_at_1x = base.fov_y_deg_at_1x
    save_resolved_config(cfg)
    return PreparedTraversal(
        cfg=cfg,
        schedule=schedule,
        cameras=cameras,
        preview_metrics=preview_metrics,
        base=base,
        initialization_metadata=initialization_metadata,
        motion_scale=motion_scale,
        warnings=warnings_out,
        random_camera=random_camera,
    )


def _adapt_traversal_camera(
    frame: Any, scene: Any, scene_type: str, pullback: Any
) -> tuple[Any, dict[str, Any]]:
    """Move a traversal camera backward while respecting interior bounds."""
    import numpy as np

    position = np.asarray(frame.position, dtype=np.float64)
    look_direction = np.asarray(frame.c2w[:3, 2], dtype=np.float64)
    direction_norm = float(np.linalg.norm(look_direction))
    details: dict[str, Any] = {
        "enabled": bool(pullback.enabled),
        "requested_ratio": float(pullback.ratio),
        "max_distance_ratio": float(pullback.max_distance_ratio),
        "original_position": position.tolist(),
        "applied_distance": 0.0,
        "limited_by_boundary": False,
    }
    if not pullback.enabled or pullback.ratio <= 0.0 or direction_norm < 1e-12:
        details["position"] = position.tolist()
        return frame, details

    center = np.asarray(scene.analysis.center, dtype=np.float64)
    center_delta = position - center
    center_distance = float(np.linalg.norm(center_delta))
    reference_distance = max(
        center_distance, float(scene.analysis.radius) * 0.05, 1e-6
    )
    requested_distance = min(
        reference_distance * float(pullback.ratio),
        reference_distance * float(pullback.max_distance_ratio),
    )
    backward = -look_direction / direction_norm
    details["backward_direction"] = backward.tolist()
    if center_distance > 1e-12:
        away_from_center = float(np.dot(backward, center_delta / center_distance))
        details["away_from_center_cosine"] = away_from_center
        if away_from_center <= 0.0:
            details["skipped_toward_scene_center"] = True
            details.update(
                {
                    "reference_distance": reference_distance,
                    "requested_distance": requested_distance,
                    "position": position.tolist(),
                }
            )
            return frame, details
    max_distance = math.inf
    if scene_type == "interior":
        lower = np.asarray(scene.analysis.aabb_min, dtype=np.float64)
        upper = np.asarray(scene.analysis.aabb_max, dtype=np.float64)
        inside = bool(np.all(position >= lower) and np.all(position <= upper))
        if inside:
            for coordinate, velocity in enumerate(backward):
                if velocity < -1e-12:
                    max_distance = min(
                        max_distance, (lower[coordinate] - position[coordinate]) / velocity
                    )
                elif velocity > 1e-12:
                    max_distance = min(
                        max_distance, (upper[coordinate] - position[coordinate]) / velocity
                    )
            if math.isfinite(max_distance):
                max_distance = max(0.0, max_distance)
                max_distance *= 1.0 - float(pullback.boundary_margin_ratio)
        else:
            max_distance = 0.0
        if requested_distance > max_distance:
            details["limited_by_boundary"] = True
    applied_distance = max(0.0, min(requested_distance, max_distance))
    adjusted_position = position + backward * applied_distance
    adjusted_c2w = np.asarray(frame.c2w, dtype=np.float64).copy()
    adjusted_c2w[:3, 3] = adjusted_position
    adjusted = type(frame)(
        position=adjusted_position,
        target=np.asarray(frame.target, dtype=np.float64) + backward * applied_distance,
        c2w=adjusted_c2w,
        w2c=np.linalg.inv(adjusted_c2w),
        fx=frame.fx,
        fy=frame.fy,
        cx=frame.cx,
        cy=frame.cy,
        fov_x=frame.fov_x,
        fov_y=frame.fov_y,
        near=frame.near,
        far=frame.far,
        camera_center_offset=np.asarray(frame.camera_center_offset, dtype=np.float64),
    )
    details.update(
        {
            "reference_distance": reference_distance,
            "requested_distance": requested_distance,
            "applied_distance": applied_distance,
            "position": adjusted_position.tolist(),
        }
    )
    return adjusted, details


def _render_endpoints(
    prepared: PreparedTraversal,
    renderer: Any,
    validation_directory: Path,
    capture_image_path: Path,
) -> tuple[dict[int, ProcessedFrame], dict[str, Path]]:
    from zoomgen.color import process_frame

    last_index = len(prepared.schedule) - 1
    cached: dict[int, ProcessedFrame] = {}
    paths: dict[str, Path] = {}
    for label, index in (("first", 0), ("last", last_index)):
        rgb = renderer.render(prepared.cameras[index])
        encoded, linear, color_metadata = process_frame(
            rgb, prepared.schedule[index], prepared.cfg
        )
        cached[index] = ProcessedFrame(encoded, linear, color_metadata)
        path = validation_directory / f"frame_{index:06d}.png"
        _save_image(path, encoded)
        paths[label] = path
        if index == 0:
            _save_image(capture_image_path, encoded)
    return cached, paths


def _evaluate_endpoints(
    qwen_client: QwenClient,
    endpoint_paths: dict[str, Path],
    report_path: Path,
) -> tuple[bool, dict[str, Any]]:
    frames: dict[str, Any] = {}
    has_error = False
    for label in ("first", "last"):
        path = endpoint_paths[label]
        try:
            result = qwen_client.evaluate(path)
            frames[label] = {"image": str(path), "result": result}
        except (QwenEvaluationError, QwenWorkerUnavailable) as exc:
            has_error = True
            frames[label] = {"image": str(path), "error": str(exc)}

    accepted = not has_error and all(
        frame.get("result", {}).get("valid") is True for frame in frames.values()
    )
    report = {
        "evaluated_at": _utc_now(),
        "status": (
            "evaluation_error" if has_error else "accepted" if accepted else "rejected"
        ),
        "accepted": accepted,
        "rule": "first.valid is true and last.valid is true",
        "frames": frames,
    }
    _write_json_atomic(report_path, report)
    return accepted, report


def _frame_metadata(
    schedule: Any,
    camera: Any,
    preview: tuple[float, float],
    color_metadata: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    return {
        "frame_index": schedule.frame_index,
        "timestamp_seconds": schedule.frame_index / fps,
        "lens_name": schedule.lens_name,
        "lens_local_frame_index": schedule.local_index,
        "lens_frame_count": schedule.lens_frame_count,
        "zoom_ratio": schedule.zoom_ratio,
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
        "coverage_ratio": preview[0],
        "largest_background_component_ratio": preview[1],
        **color_metadata,
    }


def _generation_summary(
    prepared: PreparedTraversal,
    scene: Any,
    video_probe: dict[str, Any],
    validation_report_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from zoomgen.zoom import allocate_frame_counts, temporal_state

    cfg = prepared.cfg
    counts = allocate_frame_counts(cfg)
    lens_ranges = []
    cursor = 0
    for lens, count in zip(cfg.zoom.lenses, counts):
        state = temporal_state(lens, count - 1, count, cfg)
        jump_local = state["jump_local_index"]
        lens_ranges.append(
            {
                "name": lens.name,
                "global_frame_start": cursor,
                "global_frame_end": cursor + count - 1,
                "frame_count": count,
                "jump_local_frame": jump_local,
                "jump_global_frame": cursor + jump_local,
                "jump_ev": state["jump_ev"],
                "final_exposure_gain": 2.0 ** state["jump_ev"],
            }
        )
        cursor += count
    coverage = np.asarray([value[0] for value in prepared.preview_metrics])
    base = prepared.base
    return {
        "input_ply_path": str(cfg.input.ply_path),
        "scene": scene.analysis.as_dict(),
        "scene_root_transform": cfg.scene.root_transform.model_dump(mode="json"),
        "traversal_pullback": cfg.camera.traversal_pullback.model_dump(mode="json"),
        "random_camera": prepared.random_camera,
        "initialization": prepared.initialization_metadata,
        "auto_fit": prepared.initialization_metadata.get("object_auto_fit"),
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
        "trajectory_motion_scale": prepared.motion_scale,
        "coverage_ratio": {
            "min": float(coverage.min()),
            "mean": float(coverage.mean()),
            "max": float(coverage.max()),
        },
        "video": {
            **video_probe,
            "duration_seconds": cfg.video.total_frames / cfg.video.fps,
            "filename": cfg.output.video_filename,
        },
        "lenses": lens_ranges,
        "seed": cfg.camera.motion.seed,
        "warnings": prepared.warnings,
        "quality_gate": {
            "accepted": True,
            "validation_report": str(validation_report_path),
        },
    }


def _render_full_video(
    prepared: PreparedTraversal,
    scene: Any,
    renderer: Any,
    endpoint_cache: dict[int, ProcessedFrame],
    validation_report_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from tqdm import tqdm

    from zoomgen.color import process_frame
    from zoomgen.config import prepare_output, save_resolved_config
    from zoomgen.video import FFmpegWriter, probe_video

    cfg = prepared.cfg
    prepare_output(cfg)
    save_resolved_config(cfg)
    final_video = cfg.output.directory / cfg.output.video_filename
    partial_video = final_video.with_name(f".{final_video.stem}.partial.mp4")
    if partial_video.exists():
        partial_video.unlink()
    writer = FFmpegWriter(partial_video, cfg.video)
    frame_metadata: list[dict[str, Any]] = []
    camera_metadata: list[dict[str, Any]] = []
    extension = cfg.output.frame_format
    try:
        iterator = zip(
            prepared.schedule, prepared.cameras, prepared.preview_metrics
        )
        for schedule, camera, preview in tqdm(
            iterator,
            total=len(prepared.schedule),
            desc="Rendering approved video",
            unit="frame",
        ):
            processed = endpoint_cache.get(schedule.frame_index)
            if processed is None:
                rgb = renderer.render(camera)
                encoded, linear, color_metadata = process_frame(rgb, schedule, cfg)
                processed = ProcessedFrame(encoded, linear, color_metadata)
            encoded = processed.encoded
            u8 = np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)
            writer.append(u8)
            if cfg.output.save_processed_frames:
                _save_image(
                    cfg.output.directory
                    / "frames"
                    / f"frame_{schedule.frame_index:06d}.{extension}",
                    encoded,
                )
            if cfg.output.save_linear_frames:
                np.save(
                    cfg.output.directory
                    / "linear_frames"
                    / f"frame_{schedule.frame_index:06d}.npy",
                    processed.linear.astype(np.float16),
                )
            if cfg.output.save_alpha_masks:
                alpha = renderer.render(camera, alpha_only=True)
                _save_image(
                    cfg.output.directory
                    / "alpha_masks"
                    / f"frame_{schedule.frame_index:06d}.png",
                    np.repeat(alpha[..., None], 3, axis=2),
                )
            item = _frame_metadata(
                schedule,
                camera,
                preview,
                processed.color_metadata,
                cfg.video.fps,
            )
            frame_metadata.append(item)
            camera_metadata.append(
                {
                    key: item[key]
                    for key in (
                        "frame_index",
                        "timestamp_seconds",
                        "fx",
                        "fy",
                        "cx",
                        "cy",
                        "fov_x",
                        "fov_y",
                        "camera_to_world",
                        "world_to_camera",
                        "camera_position",
                        "look_at_target",
                        "near",
                        "far",
                    )
                }
            )
        writer.close()
        probe = probe_video(partial_video)
        expected = {
            "frame_count": cfg.video.total_frames,
            "width": cfg.video.width,
            "height": cfg.video.height,
        }
        for key, expected_value in expected.items():
            if probe[key] != expected_value:
                raise RuntimeError(
                    f"encoded video {key} mismatch: expected {expected_value}, got {probe[key]}"
                )
        if not math.isclose(
            probe["fps"], cfg.video.fps, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise RuntimeError(
                f"encoded video FPS mismatch: expected {cfg.video.fps}, got {probe['fps']}"
            )
        partial_video.replace(final_video)
    except BaseException:
        writer.abort()
        if partial_video.exists():
            partial_video.unlink()
        raise

    summary = _generation_summary(
        prepared, scene, probe, validation_report_path
    )
    if cfg.output.save_metadata:
        _write_json_atomic(
            cfg.output.directory / "camera_trajectory.json", camera_metadata
        )
        _write_json_atomic(
            cfg.output.directory / "frame_metadata.json", frame_metadata
        )
        _write_json_atomic(
            cfg.output.directory / "generation_summary.json", summary
        )
    print(f"Done: {final_video}")
    return summary


def _video_is_complete(
    cfg: Any, random_camera: dict[str, Any] | None = None
) -> bool:
    from zoomgen.video import probe_video

    video_path = cfg.output.directory / cfg.output.video_filename
    summary_path = cfg.output.directory / "generation_summary.json"
    if not video_path.is_file() or not summary_path.is_file():
        return False
    try:
        probe = probe_video(video_path)
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, RuntimeError, json.JSONDecodeError):
        return False
    current_pullback = cfg.camera.traversal_pullback.model_dump(mode="json")
    if summary.get("traversal_pullback") != current_pullback:
        return False
    if summary.get("random_camera") != random_camera:
        return False
    return (
        probe["frame_count"] == cfg.video.total_frames
        and probe["width"] == cfg.video.width
        and probe["height"] == cfg.video.height
        and math.isclose(probe["fps"], cfg.video.fps, rel_tol=1e-3, abs_tol=1e-3)
    )


def _summary_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _validation_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize pipeline outcomes and endpoint quality decisions."""
    summary = {
        "reject_count": 0,
        "generate_count": 0,
        "both_endpoints_failed_count": 0,
        "first_passed_last_failed_count": 0,
        "first_failed_last_passed_count": 0,
    }
    for result in results:
        status = result.get("status")
        if status == "rejected":
            summary["reject_count"] += 1
        elif status == "generated":
            summary["generate_count"] += 1

        qwen = result.get("qwen")
        if not isinstance(qwen, dict):
            continue
        frames = qwen.get("frames")
        if not isinstance(frames, dict):
            continue
        first = frames.get("first")
        last = frames.get("last")
        if not isinstance(first, dict) or not isinstance(last, dict):
            continue
        first_result = first.get("result")
        last_result = last.get("result")
        if not isinstance(first_result, dict) or not isinstance(last_result, dict):
            continue
        first_valid = first_result.get("valid")
        last_valid = last_result.get("valid")
        if first_valid is False and last_valid is False:
            summary["both_endpoints_failed_count"] += 1
        elif first_valid is True and last_valid is False:
            summary["first_passed_last_failed_count"] += 1
        elif first_valid is False and last_valid is True:
            summary["first_failed_last_passed_count"] += 1
    return summary


def _pipeline_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a zoom video for every camera whose first and last frames pass Qwen"
        )
    )
    parser.add_argument("data_path", type=Path, help="3DGS scene directory")
    parser.add_argument("camera_config", type=Path, help="zoom camera YAML")
    parser.add_argument("color_config", type=Path, help="zoom color YAML")
    parser.add_argument("output_path", type=Path, help="output root directory")
    parser.add_argument(
        "--randome_camera",
        action="store_true",
        help="perturb each video's camera rig and colors while preserving zoom endpoints and switch gaps",
    )
    parser.add_argument(
        "--random-camera-seed",
        type=int,
        help="base seed for --randome_camera (default: camera.motion.seed from the YAML)",
    )
    parser.add_argument(
        "--remove-cache",
        action="store_true",
        help="remove scene_name/invaild after the scene finishes",
    )
    args = parser.parse_args(arguments)
    if args.random_camera_seed is not None and not args.randome_camera:
        parser.error("--random-camera-seed requires --randome_camera")

    from render_dataset_cameras import (
        _fixed_intrinsics,
        _infer_scene_type,
        camera_frame,
        capture_metadata,
        discover_scene_inputs,
        load_camera_entries,
    )
    from zoomgen.config import load_config_parts
    from zoomgen.random_camera import camera_random_seed, randomize_camera_config
    from zoomgen.renderer import GaussianRenderer
    from zoomgen.scene import GaussianScene

    camera_config = args.camera_config.expanduser().resolve()
    color_config = args.color_config.expanduser().resolve()
    output_root = args.output_path.expanduser().resolve()
    if not camera_config.is_file():
        raise ValueError(f"camera config does not exist: {camera_config}")
    if not color_config.is_file():
        raise ValueError(f"color config does not exist: {color_config}")
    if not SCENE_TEMPLATE.is_file():
        raise ValueError(f"scene template does not exist: {SCENE_TEMPLATE}")

    inputs = discover_scene_inputs(args.data_path)
    source_cameras = load_camera_entries(inputs.cameras_path)
    scene_directory = output_root / inputs.data_path.name
    scene_directory.mkdir(parents=True, exist_ok=True)
    scene_template = _load_scene_template(inputs.ply_path, scene_directory)
    bootstrap_scene_path = scene_directory / "pipeline_scene_config.yaml"
    _write_scene_config(bootstrap_scene_path, scene_template, scene_directory)
    bootstrap_cfg = load_config_parts(
        bootstrap_scene_path,
        camera_config,
        color_config,
        check_output=False,
    )

    random_base_seed = (
        bootstrap_cfg.camera.motion.seed
        if args.random_camera_seed is None else args.random_camera_seed
    )

    print(f"Loading and analyzing scene once: {inputs.ply_path}")
    scene = GaussianScene(inputs.ply_path, bootstrap_cfg)
    print(
        f"Loading {scene.analysis.effective_gaussian_count:,} Gaussians on "
        f"{bootstrap_cfg.render.device}"
    )
    renderer = GaussianRenderer(scene.load_tensors(), bootstrap_cfg)
    scene_type, inside_ratio = _infer_scene_type(source_cameras, scene)
    fixed_intrinsics = _fixed_intrinsics(source_cameras)

    pipeline_summary_path = scene_directory / "pipeline_summary.json"
    pipeline_summary: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now(),
        "data_path": str(inputs.data_path),
        "input_ply_path": str(inputs.ply_path),
        "input_camera_json_path": str(inputs.cameras_path),
        "iteration": inputs.iteration,
        "color_config": str(color_config),
        "camera_config": str(camera_config),
        "output_scene_directory": str(scene_directory),
        "camera_count": len(source_cameras),
        "random_camera": {
            "enabled": args.randome_camera,
            "base_seed": random_base_seed if args.randome_camera else None,
        },
        "scene_type": scene_type,
        "camera_inside_robust_aabb_ratio": inside_ratio,
        "traversal_pullback": bootstrap_cfg.camera.traversal_pullback.model_dump(
            mode="json"
        ),
        "output_layout": {
            "generated": VALID_OUTPUT_DIRECTORY,
            "not_generated": INVALID_OUTPUT_DIRECTORY,
        },
        "remove_cache": args.remove_cache,
        "results": [],
        "counts": {},
    }
    _write_json_atomic(pipeline_summary_path, pipeline_summary)

    qwen_client = QwenClient()
    fatal_error: Exception | None = None
    try:
        for position_index, source_camera in enumerate(source_cameras):
            position_label = f"position_{position_index:04d}"
            print(
                f"\n[{position_index + 1}/{len(source_cameras)}] {position_label} "
                f"<- {source_camera.image_name}"
            )
            valid_position_directory = (
                scene_directory / VALID_OUTPUT_DIRECTORY / position_label
            )
            invalid_position_directory = (
                scene_directory / INVALID_OUTPUT_DIRECTORY / position_label
            )
            legacy_position_directory = scene_directory / position_label
            if valid_position_directory.exists():
                paths = _position_paths(valid_position_directory)
            elif invalid_position_directory.exists():
                paths = _position_paths(invalid_position_directory)
            elif legacy_position_directory.exists():
                paths = _position_paths(legacy_position_directory)
            else:
                paths = _position_paths(invalid_position_directory)
            paths.lens_directory.mkdir(parents=True, exist_ok=True)
            paths.validation_directory.mkdir(parents=True, exist_ok=True)

            source_frame = camera_frame(source_camera)
            source_frame, pullback_metadata = _adapt_traversal_camera(
                source_frame, scene, scene_type, bootstrap_cfg.camera.traversal_pullback
            )
            metadata = capture_metadata(
                source_camera,
                source_frame,
                scene_name=inputs.data_path.name,
                scene_type=scene_type,
                position_index=position_index,
                scene_directory=scene_directory,
                image_path=paths.capture_image,
                cameras_path=inputs.cameras_path,
                fixed_intrinsics=fixed_intrinsics,
            )
            metadata["camera"]["traversal_pullback"] = pullback_metadata
            metadata["sampling"]["position_source"] = (
                "input_camera_json_adaptive_pullback"
                if pullback_metadata["applied_distance"] > 0.0
                else "input_camera_json"
            )
            _write_json_atomic(paths.camera_json, metadata)

            _write_scene_config(paths.scene_config, scene_template, paths.directory)
            cfg = load_config_parts(
                paths.scene_config,
                camera_config,
                color_config,
                camera_json_path=paths.camera_json,
                check_output=False,
            )
            random_camera = None
            if args.randome_camera:
                random_seed = camera_random_seed(
                    random_base_seed,
                    str(inputs.data_path),
                    f"{source_camera.source_index}:{source_camera.image_name}",
                )
                cfg, random_camera = randomize_camera_config(cfg, random_seed)
                metadata["random_camera"] = random_camera
                # The camera JSON remains the common base pose; these offsets
                # and the resolved config describe the individual modules.
                metadata["camera"]["randomized_lens_offsets_ratio"] = {
                    lens.name: list(lens.camera_center_offset_ratio)
                    for lens in cfg.zoom.lenses
                }
            result_entry: dict[str, Any] = {
                "position_index": position_index,
                "position_label": position_label,
                "source_camera_index": source_camera.source_index,
                "source_camera_id": source_camera.source_id,
                "source_image_name": source_camera.image_name,
                "camera_json": str(paths.camera_json),
                "validation_report": str(paths.validation_report),
                "random_camera": random_camera,
            }

            try:
                if _video_is_complete(cfg, random_camera):
                    if paths.directory != valid_position_directory:
                        source_directory = paths.directory
                        paths = _move_position_output(
                            paths, valid_position_directory
                        )
                        metadata = _rebase_path_values(
                            metadata, source_directory, paths.directory
                        )
                    _sync_position_artifacts(
                        paths,
                        scene_directory=scene_directory,
                        scene_template=scene_template,
                        cfg=cfg,
                        metadata=metadata,
                        result_entry=result_entry,
                    )
                    result_entry["status"] = "skipped_existing"
                    result_entry["classification"] = VALID_OUTPUT_DIRECTORY
                    result_entry["video"] = str(
                        cfg.output.directory / cfg.output.video_filename
                    )
                    pipeline_summary["results"].append(result_entry)
                    pipeline_summary["counts"] = _summary_counts(
                        pipeline_summary["results"]
                    )
                    _write_json_atomic(pipeline_summary_path, pipeline_summary)
                    print(
                        "Existing complete video found; moved to vaild if needed "
                        "and continued to the next camera."
                    )
                    continue

                if paths.directory != invalid_position_directory:
                    source_directory = paths.directory
                    paths = _move_position_output(
                        paths, invalid_position_directory
                    )
                    metadata = _rebase_path_values(
                        metadata, source_directory, paths.directory
                    )
                _sync_position_artifacts(
                    paths,
                    scene_directory=scene_directory,
                    scene_template=scene_template,
                    cfg=cfg,
                    metadata=metadata,
                    result_entry=result_entry,
                )
                prepared = _prepare_traversal(
                    cfg, scene, renderer, random_camera=random_camera
                )
                endpoint_cache, endpoint_paths = _render_endpoints(
                    prepared,
                    renderer,
                    paths.validation_directory,
                    paths.capture_image,
                )
                accepted, validation_report = _evaluate_endpoints(
                    qwen_client, endpoint_paths, paths.validation_report
                )
                result_entry["qwen"] = validation_report
                if qwen_client.broken:
                    result_entry["status"] = "evaluation_error"
                    result_entry["classification"] = INVALID_OUTPUT_DIRECTORY
                    raise QwenWorkerUnavailable(
                        "persistent Qwen worker is unavailable; stopping the scene"
                    )
                if not accepted:
                    result_entry["status"] = validation_report["status"]
                    result_entry["classification"] = INVALID_OUTPUT_DIRECTORY
                    print(
                        "Rejected by endpoint quality gate; full video was not rendered."
                    )
                else:
                    _render_full_video(
                        prepared,
                        scene,
                        renderer,
                        endpoint_cache,
                        paths.validation_report,
                    )
                    validation_report["status"] = "generated"
                    validation_report["video"] = str(
                        cfg.output.directory / cfg.output.video_filename
                    )
                    _write_json_atomic(paths.validation_report, validation_report)

                    source_directory = paths.directory
                    paths = _move_position_output(paths, valid_position_directory)
                    metadata = _rebase_path_values(
                        metadata, source_directory, paths.directory
                    )
                    validation_report = _rebase_path_values(
                        validation_report, source_directory, paths.directory
                    )
                    result_entry["qwen"] = validation_report
                    _sync_position_artifacts(
                        paths,
                        scene_directory=scene_directory,
                        scene_template=scene_template,
                        cfg=cfg,
                        metadata=metadata,
                        result_entry=result_entry,
                    )
                    result_entry["status"] = "generated"
                    result_entry["classification"] = VALID_OUTPUT_DIRECTORY
                    result_entry["video"] = str(
                        cfg.output.directory / cfg.output.video_filename
                    )
                    validation_report["video"] = result_entry["video"]
                    _write_json_atomic(paths.validation_report, validation_report)
            except QwenWorkerUnavailable as exc:
                if "status" not in result_entry:
                    result_entry["status"] = "evaluation_error"
                result_entry["classification"] = INVALID_OUTPUT_DIRECTORY
                result_entry["error"] = str(exc)
                fatal_error = exc
            except Exception as exc:
                result_entry["status"] = "render_error"
                result_entry["classification"] = (
                    VALID_OUTPUT_DIRECTORY
                    if valid_position_directory.exists()
                    and not invalid_position_directory.exists()
                    else INVALID_OUTPUT_DIRECTORY
                )
                result_entry["error"] = str(exc)
                if result_entry.get("qwen", {}).get("accepted") is True:
                    result_entry["qwen"]["status"] = "render_error"
                    result_entry["qwen"]["render_error"] = str(exc)
                    report_path = (
                        _position_paths(valid_position_directory).validation_report
                        if result_entry["classification"] == VALID_OUTPUT_DIRECTORY
                        else paths.validation_report
                    )
                    _write_json_atomic(report_path, result_entry["qwen"])
                print(f"Camera failed: {exc}", file=sys.stderr)

            pipeline_summary["results"].append(result_entry)
            pipeline_summary["counts"] = _summary_counts(pipeline_summary["results"])
            _write_json_atomic(pipeline_summary_path, pipeline_summary)
            # Deliberately continue after generated/rejected/failed positions.
            if fatal_error is not None:
                break
    except Exception as exc:
        fatal_error = exc
        print(f"Fatal pipeline error: {exc}", file=sys.stderr)
    finally:
        qwen_client.close()

    invalid_cache_directory = scene_directory / INVALID_OUTPUT_DIRECTORY
    cleanup_status = "disabled"
    if args.remove_cache:
        if fatal_error is not None:
            cleanup_status = "skipped_incomplete_scene"
        else:
            try:
                removed = _remove_invalid_cache(scene_directory)
                cleanup_status = "removed" if removed else "not_present"
                if removed:
                    print(f"Removed invalid cache: {invalid_cache_directory}")
            except (OSError, RuntimeError) as exc:
                fatal_error = RuntimeError(
                    f"failed to remove invalid cache {invalid_cache_directory}: {exc}"
                )
                cleanup_status = "error"
                print(str(fatal_error), file=sys.stderr)
    pipeline_summary["invalid_cache_cleanup"] = {
        "requested": args.remove_cache,
        "path": str(invalid_cache_directory),
        "status": cleanup_status,
    }

    pipeline_summary["status"] = "failed" if fatal_error is not None else "completed"
    pipeline_summary["finished_at"] = _utc_now()
    pipeline_summary["counts"] = _summary_counts(pipeline_summary["results"])
    validation_summary = _validation_summary(pipeline_summary["results"])
    pipeline_summary["validation_summary"] = validation_summary
    if fatal_error is not None:
        pipeline_summary["fatal_error"] = str(fatal_error)
    _write_json_atomic(pipeline_summary_path, pipeline_summary)
    print(f"Pipeline summary: {pipeline_summary_path}")
    print(
        "Validation summary: "
        f"reject={validation_summary['reject_count']}, "
        f"generate={validation_summary['generate_count']}, "
        "first_and_last_failed="
        f"{validation_summary['both_endpoints_failed_count']}, "
        "first_passed_last_failed="
        f"{validation_summary['first_passed_last_failed_count']}, "
        "first_failed_last_passed="
        f"{validation_summary['first_failed_last_passed_count']}"
    )
    if fatal_error is not None:
        return 2
    if pipeline_summary["counts"].get("render_error", 0):
        return 1
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--qwen-worker":
        return _qwen_worker_main(sys.argv[2:])
    try:
        return _pipeline_main(sys.argv[1:])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
