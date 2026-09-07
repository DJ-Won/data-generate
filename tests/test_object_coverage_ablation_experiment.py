from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import object_coverage_ablation as ablation
from experiments import summarize_object_coverage_ablation as summary_module


def _camera_at(distance: float) -> ablation.CameraFrame:
    c2w = np.eye(4, dtype=np.float64)
    c2w[0, 3] = distance
    return ablation.CameraFrame(
        position=np.array([distance, 0.0, 0.0], dtype=np.float64),
        target=np.zeros(3, dtype=np.float64),
        c2w=c2w,
        w2c=np.linalg.inv(c2w),
        fx=10.0,
        fy=10.0,
        cx=2.0,
        cy=1.5,
        fov_x=1.0,
        fov_y=1.0,
        near=0.01,
        far=1000.0,
        camera_center_offset=np.zeros(3, dtype=np.float64),
    )


def _capture(index: int) -> ablation.CapturePlan:
    position = ablation.PositionPlan(
        index=index,
        label=f"position_{index:04d}",
        position=np.array([float(index), 0.0, 10.0]),
        offset_from_seed=np.zeros(3, dtype=np.float64),
        clearance=1.0,
        seed_position_source="test",
        look_at_target=np.zeros(3, dtype=np.float64),
    )
    return ablation.CapturePlan(
        position=position,
        lens_index=index,
        lens_label=f"lens_{index:04d}",
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        fov_y_deg=70.0,
    )


def test_legacy_adaptive_replays_six_step_distance_schedule(monkeypatch):
    target_ratio = 0.65
    expected_distances = np.array(
        [
            77.370495,
            56.839380,
            49.614435,
            44.652992,
            40.187693,
            11.033949,
        ],
        dtype=np.float64,
    )
    preview_ratios = [
        target_ratio * (expected_distances[index + 1] / distance) ** 2
        for index, distance in enumerate(expected_distances[:4])
    ] + [0.60, 0.64]
    returned_ratios = iter([*preview_ratios, 0.64])
    evaluation_sizes: list[tuple[int | None, int | None]] = []

    coverage = SimpleNamespace(
        minimum_pixel_ratio=target_ratio,
        alpha_threshold=0.02,
        preview_width=32,
        preview_height=24,
        exterior_margin_radius_ratio=0.0,
    )
    cfg = SimpleNamespace(
        initialization=SimpleNamespace(
            object=SimpleNamespace(coverage=coverage),
            reference_match=SimpleNamespace(preview_width=320, preview_height=240),
        )
    )
    minimum_distance = float(expected_distances[-1])
    scene = SimpleNamespace(
        analysis=SimpleNamespace(
            center=np.zeros(3, dtype=np.float64),
            aabb_min=np.array([-minimum_distance, -1.0, -1.0]),
            aabb_max=np.array([minimum_distance, 1.0, 1.0]),
            radius=1.0,
        )
    )

    def fake_move(_cfg, _scene, _camera, distance, _tree):
        return _camera_at(float(distance)), float(distance - minimum_distance)

    def fake_coverage(_renderer, _camera, _coverage, width=None, height=None):
        evaluation_sizes.append((width, height))
        return next(returned_ratios)

    monkeypatch.setattr(ablation, "_object_camera_at_distance", fake_move)
    monkeypatch.setattr(ablation, "_object_gaussian_coverage_ratio", fake_coverage)

    fitted, metadata = ablation._fit_legacy_adaptive(
        cfg,
        scene,
        object(),
        _camera_at(float(expected_distances[0])),
        object(),
    )

    actual_distances = [item["distance_to_center"] for item in metadata["history"]]
    np.testing.assert_allclose(actual_distances, expected_distances, rtol=0.0, atol=1e-6)
    assert metadata["method"] == "preview_adaptive_search_with_output_validation"
    assert metadata["preview_evaluation_count"] == ablation.LEGACY_MAX_ITERATIONS == 6
    assert metadata["output_validation_count"] == 1
    assert metadata["evaluation_count"] == 7
    assert metadata["preview_selected_distance_to_center"] == pytest.approx(
        minimum_distance
    )
    assert metadata["final_distance_to_center"] == pytest.approx(minimum_distance)
    assert metadata["preview_constraints_met"] is False
    assert metadata["output_constraints_met"] is False
    assert metadata["constraints_met"] is False
    np.testing.assert_allclose(fitted.position, [minimum_distance, 0.0, 0.0])
    assert evaluation_sizes == [(None, None)] * 6 + [(320, 240)]


def test_candidate_resume_treats_rejected_coverage_as_terminal(
    tmp_path, monkeypatch
):
    captures = [_capture(0), _capture(1)]
    results_path = tmp_path / "candidate_results.jsonl"
    initial_records = [
        {
            "status": "coverage_rejected",
            "candidate_index": 0,
            "strategy": ablation.STRATEGY_LEGACY,
        },
        {
            "status": "ok",
            "candidate_index": 0,
            "strategy": ablation.STRATEGY_CURRENT,
        },
    ]
    valid_prefix = b"".join(
        (json.dumps(record) + "\n").encode("utf-8") for record in initial_records
    )
    results_path.write_bytes(valid_prefix + b'{"status":"coverage_rejected"')

    loaded = ablation._load_and_repair_jsonl(results_path)
    assert loaded == initial_records
    assert results_path.read_bytes() == valid_prefix

    cfg = SimpleNamespace(
        initialization=SimpleNamespace(
            reference_match=SimpleNamespace(preview_width=4, preview_height=3),
            traversal=SimpleNamespace(random=SimpleNamespace(device="cpu")),
        )
    )
    fit_calls: list[tuple[str, int]] = []
    render_calls: list[int] = []
    score_calls: list[str] = []

    def fake_camera_frame(_cfg, _scene, capture):
        return _camera_at(float(capture.lens_index))

    def fit_current(_cfg, _scene, _renderer, camera, _tree):
        fit_calls.append((ablation.STRATEGY_CURRENT, int(camera.position[0])))
        return camera, {"constraints_met": True, "output_constraints_met": True}

    def fit_legacy(_cfg, _scene, _renderer, camera, _tree):
        fit_calls.append((ablation.STRATEGY_LEGACY, int(camera.position[0])))
        return camera, {"constraints_met": False, "output_constraints_met": False}

    class Renderer:
        def render(self, camera, width, height):
            assert (width, height) == (4, 3)
            render_calls.append(int(camera.position[0]))
            return np.zeros((height, width, 3), dtype=np.float32)

    def fake_score(_rgb, _cfg, _metric, _torch, device):
        score_calls.append(device)
        return 0.75

    monkeypatch.setattr(ablation, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(ablation, "_camera_frame", fake_camera_frame)
    monkeypatch.setattr(ablation, "_fit_object_camera_coverage", fit_current)
    monkeypatch.setattr(ablation, "_fit_legacy_adaptive", fit_legacy)
    monkeypatch.setattr(ablation, "_topiq_nr_score", fake_score)

    records = ablation._run_candidates(
        cfg,
        object(),
        Renderer(),
        captures,
        object(),
        object(),
        object(),
        results_path,
        loaded,
    )

    assert fit_calls == [
        (ablation.STRATEGY_CURRENT, 1),
        (ablation.STRATEGY_LEGACY, 1),
    ]
    assert render_calls == [1]
    assert score_calls == ["cpu"]
    terminal = ablation._terminal_records_by_key(records)
    assert set(terminal) == {
        (candidate_index, strategy)
        for candidate_index in range(2)
        for strategy in ablation.STRATEGIES
    }
    assert terminal[(1, ablation.STRATEGY_LEGACY)]["status"] == "coverage_rejected"
    assert terminal[(1, ablation.STRATEGY_LEGACY)]["topiq_nr_score"] is None
    assert terminal[(1, ablation.STRATEGY_CURRENT)]["status"] == "ok"
    assert terminal[(1, ablation.STRATEGY_CURRENT)]["topiq_nr_score"] == 0.75

    completed_bytes = results_path.read_bytes()
    reloaded = ablation._load_and_repair_jsonl(results_path)
    ablation._run_candidates(
        cfg,
        object(),
        Renderer(),
        captures,
        object(),
        object(),
        object(),
        results_path,
        reloaded,
    )
    assert results_path.read_bytes() == completed_bytes
    assert len(fit_calls) == 2
    assert render_calls == [1]
    assert score_calls == ["cpu"]


def _result_record(
    candidate_index: int,
    strategy: str,
    status: str,
    coverage_pass: bool,
    score,
    total_seconds: float,
) -> dict:
    return {
        "status": status,
        "candidate_index": candidate_index,
        "strategy": strategy,
        "coverage_qualified_at_output": coverage_pass,
        "coverage": {
            "output_constraints_met": coverage_pass,
            "constraints_met": coverage_pass,
        },
        "topiq_nr_score": score,
        "timing_seconds": {"total": total_seconds},
    }


def test_summary_keeps_rejected_pairs_and_renders_empty_scores_as_na(tmp_path):
    experiment_root = tmp_path / "fixture_scene"
    experiment_root.mkdir()
    summary_path = experiment_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "kind": "object_coverage_ablation",
                "scene_name": "fixture_scene",
                "input_ply_path": "/datasets/fixture_scene.ply",
                "candidate_count": 4,
                "candidate_plan_sha256": "fixture-plan-sha256",
                "score_threshold": 0.65,
                "results_jsonl": "candidate_results.jsonl",
            }
        ),
        encoding="utf-8",
    )
    records = [
        _result_record(0, ablation.STRATEGY_LEGACY, "ok", True, None, 1.0),
        _result_record(0, ablation.STRATEGY_CURRENT, "ok", True, 0.70, 2.0),
        _result_record(1, ablation.STRATEGY_LEGACY, "ok", True, "nan", 1.0),
        _result_record(
            1, ablation.STRATEGY_CURRENT, "coverage_rejected", False, None, 2.0
        ),
        _result_record(
            2, ablation.STRATEGY_LEGACY, "coverage_rejected", False, 0.99, 1.0
        ),
        _result_record(2, ablation.STRATEGY_CURRENT, "ok", True, 0.80, 2.0),
        _result_record(
            3, ablation.STRATEGY_LEGACY, "coverage_rejected", False, 0.95, 1.0
        ),
        _result_record(
            3, ablation.STRATEGY_CURRENT, "coverage_rejected", False, 0.96, 2.0
        ),
    ]
    (experiment_root / "candidate_results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = summary_module.summarize_experiment(summary_path, None)

    assert report.candidate_count == 4
    assert report.candidate_hash == "fixture-plan-sha256"
    assert report.candidate_hash_source == "candidate_plan_sha256"
    assert report.coverage_both_pass == 1
    assert report.coverage_legacy_only == 1
    assert report.coverage_current_only == 1
    assert report.coverage_neither_pass == 1

    legacy = report.strategies[ablation.STRATEGY_LEGACY]
    current = report.strategies[ablation.STRATEGY_CURRENT]
    assert legacy.scores == ()
    assert legacy.total_seconds == pytest.approx(4.0)
    assert legacy.threshold_pass_count == 0
    assert current.scores == (0.70, 0.80)
    assert current.total_seconds == pytest.approx(8.0)
    assert current.threshold_pass_count == 2

    markdown = summary_module._render_markdown([report])
    assert (
        "| fixture_scene | legacy_adaptive | 0 | n/a | n/a | n/a | n/a | 0 | 4.000 |"
        in markdown
    )

    csv_rows = list(csv.DictReader(io.StringIO(summary_module._render_csv([report]))))
    assert len(csv_rows) == 1
    csv_row = csv_rows[0]
    assert csv_row["legacy_adaptive_score_count"] == "0"
    assert csv_row["legacy_adaptive_score_mean"] == "n/a"
    assert csv_row["legacy_adaptive_score_median"] == "n/a"
    assert csv_row["legacy_adaptive_score_min"] == "n/a"
    assert csv_row["legacy_adaptive_score_max"] == "n/a"
