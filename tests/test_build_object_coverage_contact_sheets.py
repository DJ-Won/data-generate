from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from experiments import build_object_coverage_contact_sheets as contact_sheets


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_completed_scene(root: Path) -> None:
    root.mkdir()
    manifests = {}
    for strategy_index, strategy in enumerate(contact_sheets.STRATEGY_KEYS):
        items = []
        for rank in range(1, contact_sheets.TOP_K + 1):
            rank_root = root / "strategies" / strategy / f"rank_{rank:02d}"
            rank_root.mkdir(parents=True)
            image_path = rank_root / "image.png"
            Image.new(
                "RGB",
                (24, 16),
                color=(20 * rank, 60 * strategy_index, 127),
            ).save(image_path, format="PNG")
            items.append(
                {
                    "rank": rank,
                    "candidate_index": strategy_index * 100 + rank,
                    "candidate_id": f"candidate-{strategy_index}-{rank}",
                    "topiq_nr_score": 0.91 - 0.01 * rank,
                    "coverage_qualified_at_output": True,
                    "image": str(image_path.relative_to(root)),
                    "metadata": str((rank_root / "metadata.json").relative_to(root)),
                }
            )
        manifest_path = root / "strategies" / strategy / "top10.json"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "strategy": strategy,
                "candidate_result_count": 100,
                "top_k": 10,
                "score_threshold": 0.65,
                "score_threshold_operator": ">",
                "threshold_pass_count": 50,
                "fallback_used": False,
                "items": items,
            },
        )
        manifests[strategy] = {
            "path": f"strategies/{strategy}/top10.json",
            "top_k": 10,
        }

    _write_json(
        root / "summary.json",
        {
            "kind": "object_coverage_ablation",
            "schema_version": 1,
            "completed_at": "2026-09-06T00:00:00+00:00",
            "candidate_count": 5000,
            "strategy_evaluation_count": 10000,
            "full_candidate_pool": True,
            "early_termination": False,
            "strategies": list(contact_sheets.STRATEGY_KEYS),
            "top_k": 10,
            "topiq_nr_threshold_l": 0.65,
            "manifests": manifests,
        },
    )


def test_completed_scene_requires_and_renders_twenty_valid_pngs(tmp_path):
    root = tmp_path / "oxford"
    _make_completed_scene(root)

    scene = contact_sheets._load_scene(str(root))

    images = [item.image_path for strategy in scene.strategies for item in strategy.items]
    assert len(images) == len(set(images)) == 20
    assert all(path.is_file() for path in images)
    output_path = tmp_path / "report" / "oxford" / contact_sheets.OUTPUT_FILENAME
    contact_sheets.build_contact_sheet(scene, output_path)
    with Image.open(output_path) as output:
        output.load()
        assert output.format == "PNG"
        assert output.mode == "RGB"
        assert output.width > 0 and output.height > 0


def test_missing_summary_is_rejected_before_manifests(tmp_path):
    root = tmp_path / "still_running"
    root.mkdir()

    with pytest.raises(contact_sheets.ContactSheetError, match="summary.json"):
        contact_sheets._load_scene(str(root))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("candidate_count", 4999, "candidate_count must be 5000"),
        ("strategy_evaluation_count", 9999, "strategy_evaluation_count must be 10000"),
    ),
)
def test_full_experiment_counts_are_required(tmp_path, field, value, message):
    root = tmp_path / "oxford"
    _make_completed_scene(root)
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    summary[field] = value
    _write_json(summary_path, summary)

    with pytest.raises(contact_sheets.ContactSheetError, match=message):
        contact_sheets._load_scene(str(root))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("kind", "other", "kind must be"),
        ("schema_version", 2, "schema_version must be 1"),
        ("completed_at", None, "has no completed_at"),
        ("full_candidate_pool", False, "full_candidate_pool must be true"),
        ("early_termination", True, "early_termination must be false"),
        ("strategies", ["current_bracketed", "legacy_adaptive"], "strategies must be"),
        ("top_k", 9, "top_k must be 10"),
    ),
)
def test_summary_completion_contract_is_strict(tmp_path, field, value, message):
    root = tmp_path / "oxford"
    _make_completed_scene(root)
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    summary[field] = value
    _write_json(summary_path, summary)

    with pytest.raises(contact_sheets.ContactSheetError, match=message):
        contact_sheets._load_scene(str(root))


def test_greater_than_or_equal_threshold_operator_is_rejected(tmp_path):
    root = tmp_path / "oxford"
    _make_completed_scene(root)
    manifest_path = root / "strategies" / "legacy_adaptive" / "top10.json"
    manifest = _read_json(manifest_path)
    manifest["score_threshold_operator"] = ">="
    _write_json(manifest_path, manifest)

    with pytest.raises(contact_sheets.ContactSheetError, match="must be '>'"):
        contact_sheets._load_scene(str(root))


def test_nonstandard_manifest_reference_is_rejected(tmp_path):
    root = tmp_path / "oxford"
    _make_completed_scene(root)
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    summary["manifests"]["legacy_adaptive"]["path"] = "legacy_top10.json"
    _write_json(summary_path, summary)

    with pytest.raises(contact_sheets.ContactSheetError, match="manifest reference"):
        contact_sheets._load_scene(str(root))


def test_all_referenced_pngs_must_decode(tmp_path):
    root = tmp_path / "oxford"
    _make_completed_scene(root)
    image_path = root / "strategies" / "current_bracketed" / "rank_10" / "image.png"
    image_path.write_bytes(b"not a png")

    with pytest.raises(contact_sheets.ContactSheetError, match="cannot decode"):
        contact_sheets._load_scene(str(root))
