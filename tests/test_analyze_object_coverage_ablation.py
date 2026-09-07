from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

from experiments import analyze_object_coverage_ablation as analysis


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _coverage_record(
    index: int,
    strategy: str,
    coverage_pass: bool,
    score: float | None,
    capture: dict,
    planned_position: list[float],
) -> dict:
    is_current = strategy == analysis.STRATEGY_CURRENT
    fit_seconds = 0.8 if is_current else 1.0
    render_seconds = 0.2 if coverage_pass else 0.0
    initial_distance = 100.0 + index
    final_distance = initial_distance * (0.45 if is_current else 0.35)
    return {
        "schema_version": 1,
        "status": "ok" if coverage_pass else "coverage_rejected",
        "candidate_index": index,
        "candidate_id": f"position_{index}:lens_{index}",
        "strategy": strategy,
        "capture": capture,
        "planned_position": planned_position,
        "coverage_qualified_at_output": coverage_pass,
        "topiq_nr_score": score,
        "coverage": {
            "minimum_pixel_ratio": 0.85,
            "output_validation_size": [4, 3],
            "output_constraints_met": coverage_pass,
            "constraints_met": coverage_pass,
            "initial_distance_to_center": initial_distance,
            "minimum_exterior_distance_to_center": 10.0,
            "final_distance_to_center": final_distance,
            "initial_pixel_ratio": 0.20,
            "final_pixel_ratio": 0.86 if coverage_pass else 0.80,
            "final_distance_to_nearest_effective_gaussian": 2.0,
            "preview_evaluation_count": 5 if is_current else 6,
            "output_validation_count": 3 if is_current else 1,
            "evaluation_count": 8 if is_current else 7,
        },
        "timing_seconds": {
            "coverage_fit": fit_seconds,
            "rgb_render_and_score": render_seconds,
            "total": fit_seconds + render_seconds,
        },
    }


def _make_complete_fixture(root: Path) -> None:
    root.mkdir()
    positions = []
    captures = []
    for index in range(12):
        positions.append(
            {
                "index": index,
                "label": f"position_{index:04d}",
                "position": [float(index), 0.0, 100.0],
                "offset_from_seed": [0.0, 0.0, 0.0],
                "clearance": 1.0,
                "seed_position_source": "fixture",
                "look_at_target": [0.0, 0.0, 0.0],
            }
        )
        captures.append(
            {
                "position_index": index,
                "lens_index": index,
                "lens_label": f"lens_{index:04d}",
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fov_y_deg": 70.0,
            }
        )
    plan = {
        "schema_version": 1,
        "config_fingerprint": "fixture-config",
        "positions": positions,
        "captures": captures,
    }
    plan_hash, _ = analysis._candidate_plan_hash(plan, "fixture plan")
    plan["candidate_plan_sha256"] = plan_hash
    _write_json(root / "fixed_plan.json", plan)

    records = []
    records_by_strategy: dict[str, list[dict]] = {
        strategy: [] for strategy in analysis.STRATEGIES
    }
    for index, capture in enumerate(captures):
        legacy_pass = index <= 9
        current_pass = index <= 8 or index == 10
        legacy_score = (
            0.65
            if index == 9
            else (round(0.60 + 0.01 * index, 2) if legacy_pass else None)
        )
        current_score = (
            round(0.62 + 0.01 * index, 2)
            if index <= 8
            else (0.80 if index == 10 else None)
        )
        for strategy, passed, score in (
            (analysis.STRATEGY_LEGACY, legacy_pass, legacy_score),
            (analysis.STRATEGY_CURRENT, current_pass, current_score),
        ):
            record = _coverage_record(
                index,
                strategy,
                passed,
                score,
                capture,
                positions[index]["position"],
            )
            records.append(record)
            records_by_strategy[strategy].append(record)
    (root / "candidate_results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    marker = {
        "kind": analysis.EXPERIMENT_KIND,
        "schema_version": 1,
        "config_fingerprint": "fixture-config",
        "scene_config": "/configs/fixture_scene.yaml",
        "input_ply_path": "/data/fixture_scene.ply",
        "source_files_sha256": {
            "experiment_harness": "a" * 64,
            "scene_traversal": "b" * 64,
        },
        "strategies": list(analysis.STRATEGIES),
        "top_k": 10,
        "full_candidate_pool": True,
        "early_termination": False,
        "minimum_pixel_ratio": 0.85,
        "topiq_nr_threshold_l": 0.65,
    }
    _write_json(root / "experiment.json", marker)

    strategy_summaries = {}
    manifests = {}
    for strategy in analysis.STRATEGIES:
        strategy_records = records_by_strategy[strategy]
        scored = sorted(
            (record for record in strategy_records if record["status"] == "ok"),
            key=lambda record: (-record["topiq_nr_score"], record["candidate_index"]),
        )
        threshold_pass_count = sum(record["topiq_nr_score"] > 0.65 for record in scored)
        strategy_summaries[strategy] = {
            "terminal_count": 12,
            "coverage_qualified_count": len(scored),
            "coverage_rejected_count": 12 - len(scored),
            "scored_count": len(scored),
            "threshold_pass_count": threshold_pass_count,
        }
        items = []
        for rank, record in enumerate(scored[:10], start=1):
            rank_root = root / "strategies" / strategy / f"rank_{rank:02d}"
            rank_root.mkdir(parents=True)
            Image.new("RGB", (4, 3), color=(rank, index, 127)).save(
                rank_root / "image.png", format="PNG"
            )
            metadata = {
                "strategy": strategy,
                "selection_rank": rank,
                "candidate_index": record["candidate_index"],
                "selection_score": record["topiq_nr_score"],
            }
            _write_json(rank_root / "metadata.json", metadata)
            items.append(
                {
                    "rank": rank,
                    "candidate_index": record["candidate_index"],
                    "candidate_id": record["candidate_id"],
                    "topiq_nr_score": record["topiq_nr_score"],
                    "rerender_score": record["topiq_nr_score"] + 1e-7,
                    "image": str((rank_root / "image.png").relative_to(root)),
                    "metadata": str((rank_root / "metadata.json").relative_to(root)),
                }
            )
        manifest_path = root / "strategies" / strategy / "top10.json"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "strategy": strategy,
                "candidate_result_count": len(scored),
                "top_k": 10,
                "score_threshold": 0.65,
                "score_threshold_operator": ">",
                "threshold_pass_count": threshold_pass_count,
                "fallback_used": threshold_pass_count < 10,
                "items": items,
            },
        )
        manifests[strategy] = {
            "path": str(manifest_path.relative_to(root)),
            "top_k": 10,
        }

    summary = {
        **marker,
        "completed_at": "2026-09-06T00:00:00+00:00",
        "candidate_count": 12,
        "candidate_plan_sha256": plan_hash,
        "strategy_evaluation_count": 24,
        "strategy_summaries": strategy_summaries,
        "paired_output_coverage": {
            "both_pass": 9,
            "legacy_only": 1,
            "current_only": 1,
            "neither_pass": 1,
        },
        "results_jsonl": "candidate_results.jsonl",
        "manifests": manifests,
    }
    _write_json(root / "summary.json", summary)


def test_complete_fixture_reports_strict_paired_metrics_and_top10(tmp_path):
    root = tmp_path / "fixture_scene"
    _make_complete_fixture(root)

    report, scenes = analysis.build_report(
        [root],
        expected_candidates=12,
        expected_score_threshold=0.65,
        expected_scenes=("fixture_scene",),
    )

    assert len(scenes) == 1
    scene = report["scenes"][0]
    coverage = scene["metrics"]["coverage"]["paired_outcome"]
    assert coverage["both_pass"] == 9
    assert coverage["legacy_only"] == 1
    assert coverage["current_only"] == 1
    assert coverage["neither_pass"] == 1

    threshold = scene["metrics"]["topiq_nr"][
        "paired_threshold_outcome_over_all_candidates"
    ]
    assert threshold["both_pass"] == 3
    assert threshold["legacy_only"] == 0
    assert threshold["current_only"] == 3
    assert threshold["neither_pass"] == 6
    assert threshold["current_minus_legacy_pass_count"] == 3

    paired_score = scene["metrics"]["topiq_nr"]["paired_score_on_both_coverage_pass"]
    assert paired_score["delta"]["count"] == 9
    assert paired_score["delta"]["mean"] == pytest.approx(0.02)
    assert paired_score["current_higher_count"] == 9
    assert scene["metrics"]["strategies"][analysis.STRATEGY_LEGACY]["topiq_nr"][
        "threshold_pass_count"
    ] == 3
    assert scene["top10"]["strategies"][analysis.STRATEGY_CURRENT][
        "top10_threshold_pass_count"
    ] == 6
    assert scene["validation"]["top10_pairs_validated_per_strategy"] == 10

    markdown = analysis.render_markdown(report)
    assert "Coverage 配对四格" in markdown
    assert "双方 coverage 合格候选的配对分差" in markdown
    assert "fixture_scene Top-10 明细" in markdown
    assert scene["top10"]["items"][analysis.STRATEGY_LEGACY][0][
        "decoded_image"
    ] == {"format": "PNG", "mode": "RGB", "width": 4, "height": 3}

    report["overall"]["metrics"]["topiq_nr"]["threshold"] = 0.7
    markdown = analysis.render_markdown(report)
    assert "`TOPIQ-NR > 0.7`" in markdown
    assert "`TOPIQ-NR > 0.65`" not in markdown


def test_missing_atomic_summary_refuses_before_reading_broken_ledger(tmp_path):
    root = tmp_path / "still_running"
    root.mkdir()
    (root / "candidate_results.jsonl").write_text("{broken", encoding="utf-8")

    with pytest.raises(analysis.AnalysisError, match="ledger was not read"):
        analysis.analyze_scene(root, expected_candidates=12)


@pytest.mark.parametrize(
    ("legacy_only", "current_only"),
    ((2400, 2600), (12000, 13000)),
)
def test_exact_mcnemar_is_stable_for_full_experiment_sizes(
    legacy_only, current_only
):
    p_value = analysis._mcnemar_exact_two_sided(legacy_only, current_only)
    reverse = analysis._mcnemar_exact_two_sided(current_only, legacy_only)

    assert p_value is not None
    assert math.isfinite(p_value)
    assert 0.0 <= p_value <= 1.0
    assert p_value == pytest.approx(reverse, rel=0.0, abs=0.0)


def test_duplicate_terminal_key_is_rejected(tmp_path):
    root = tmp_path / "fixture_scene"
    _make_complete_fixture(root)
    ledger_path = root / "candidate_results.jsonl"
    first_line = ledger_path.read_text(encoding="utf-8").splitlines()[0]
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(first_line + "\n")

    with pytest.raises(analysis.AnalysisError, match="duplicate terminal key"):
        analysis.analyze_scene(root, expected_candidates=12)


@pytest.mark.parametrize("field", ("candidate_id", "capture", "planned_position"))
def test_ledger_candidate_identity_must_match_fixed_plan(tmp_path, field):
    root = tmp_path / "fixture_scene"
    _make_complete_fixture(root)
    ledger_path = root / "candidate_results.jsonl"
    records = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    if field == "candidate_id":
        records[0][field] = "position_999:lens_999"
    elif field == "capture":
        records[0][field]["yaw_deg"] = 12.0
    else:
        records[0][field] = [999.0, 0.0, 0.0]
    ledger_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(analysis.AnalysisError, match="vs fixed plan"):
        analysis.analyze_scene(root, expected_candidates=12)


def test_top10_png_must_decode(tmp_path):
    root = tmp_path / "fixture_scene"
    _make_complete_fixture(root)
    image_path = root / "strategies" / analysis.STRATEGY_LEGACY / "rank_01" / "image.png"
    image_path.write_bytes(b"not a PNG")

    with pytest.raises(analysis.AnalysisError, match="cannot decode"):
        analysis.analyze_scene(root, expected_candidates=12)


def test_build_report_defaults_to_exact_five_scene_set(tmp_path):
    root = tmp_path / "fixture_scene"
    _make_complete_fixture(root)

    with pytest.raises(analysis.AnalysisError, match="completed scene set mismatch"):
        analysis.build_report(
            [root], expected_candidates=12, expected_score_threshold=0.65
        )
