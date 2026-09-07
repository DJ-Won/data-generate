#!/usr/bin/env python3
"""Summarize object coverage ablation runs without loading the renderer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_LEGACY = "legacy_adaptive"
STRATEGY_CURRENT = "current_bracketed"
STRATEGIES = (STRATEGY_LEGACY, STRATEGY_CURRENT)
CSV_FILENAME = "object_coverage_ablation_summary.csv"
MARKDOWN_FILENAME = "object_coverage_ablation_summary.md"


class ReportError(ValueError):
    """An experiment cannot be summarized without ambiguous assumptions."""


@dataclass(frozen=True)
class StrategyReport:
    scores: tuple[float, ...]
    threshold_pass_count: int
    total_seconds: float

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.scores) if self.scores else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.scores) if self.scores else None

    @property
    def minimum(self) -> float | None:
        return min(self.scores) if self.scores else None

    @property
    def maximum(self) -> float | None:
        return max(self.scores) if self.scores else None


@dataclass(frozen=True)
class SceneReport:
    scene: str
    input_path: str
    experiment_root: Path | None
    candidate_hash: str
    candidate_hash_source: str
    candidate_count: int
    score_threshold: float | None
    score_thresholds: tuple[float, ...]
    strategies: Mapping[str, StrategyReport]
    coverage_both_pass: int
    coverage_legacy_only: int
    coverage_current_only: int
    coverage_neither_pass: int
    scene_count: int = 1
    row_type: str = "scene"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise ReportError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ReportError(
            f"invalid JSON in {label} {path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except OSError as error:
        raise ReportError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"{label} must contain a JSON object: {path}")
    return value


def _load_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise ReportError(f"{label} does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ReportError(f"invalid YAML in {label} {path}: {error}") from error
    except OSError as error:
        raise ReportError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"{label} must contain a YAML mapping: {path}")
    return value


def _nested(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_nested(
    sources: Iterable[Mapping[str, Any]], paths: Iterable[Sequence[str]]
) -> Any:
    for source in sources:
        for path in paths:
            value = _nested(source, path)
            if value is not None:
                return value
    return None


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ReportError(f"{context} must be a finite number, got boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} must be a finite number, got {value!r}") from error
    if not math.isfinite(result):
        raise ReportError(f"{context} must be finite, got {value!r}")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ReportError(f"{context} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} must be a non-negative integer") from error
    if result < 0 or result != value:
        raise ReportError(f"{context} must be a non-negative integer")
    return result


def _resolve_reference(reference: Any, experiment_root: Path) -> Path | None:
    if not isinstance(reference, (str, os.PathLike)) or not str(reference).strip():
        return None
    path = Path(reference).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        experiment_root / path,
        REPOSITORY_ROOT / path,
        Path.cwd() / path,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), None)


def _summary_path(argument: str) -> Path:
    path = Path(argument).expanduser().resolve()
    if path.is_dir():
        path = path / "summary.json"
    elif path.name != "summary.json":
        raise ReportError(
            f"experiment input must be a directory or summary.json file: {argument}"
        )
    if not path.is_file():
        raise ReportError(f"summary.json does not exist: {path}")
    return path


def _load_marker(experiment_root: Path) -> dict[str, Any]:
    marker_path = experiment_root / "experiment.json"
    if not marker_path.exists():
        return {}
    return _load_json_object(marker_path, "experiment marker")


def _load_config_if_present(
    reference: Any, experiment_root: Path, label: str
) -> tuple[dict[str, Any], Path | None]:
    path = _resolve_reference(reference, experiment_root)
    if path is None or not path.is_file():
        return {}, path
    return _load_yaml_object(path, label), path


def _load_terminal_records(
    results_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {
        strategy: {} for strategy in STRATEGIES
    }
    try:
        handle = results_path.open("r", encoding="utf-8")
    except FileNotFoundError as error:
        raise ReportError(f"candidate results do not exist: {results_path}") from error
    except OSError as error:
        raise ReportError(f"cannot read candidate results {results_path}: {error}") from error

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportError(
                    f"invalid JSON in {results_path} line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ReportError(
                    f"candidate record in {results_path} line {line_number} "
                    "must be a JSON object"
                )
            if record.get("status") not in {"ok", "coverage_rejected"}:
                continue
            strategy = record.get("strategy")
            if strategy not in STRATEGIES:
                continue
            if "candidate_index" in record:
                index = _nonnegative_int(
                    record["candidate_index"],
                    f"{results_path} line {line_number} candidate_index",
                )
                key = f"index:{index}"
            elif record.get("candidate_id") is not None:
                key = f"id:{record['candidate_id']}"
            else:
                raise ReportError(
                    f"terminal record in {results_path} line {line_number} has "
                    "neither candidate_index nor candidate_id"
                )
            # A resumed run may retain failed attempts. For terminal duplicates, the
            # last append is authoritative, matching the experiment loader.
            records[strategy][key] = record
    return records


def _record_score(record: Mapping[str, Any], context: str) -> float:
    value = _first_nested(
        (record,),
        (
            ("topiq_nr_score",),
            ("score", "topiq_nr"),
            ("metrics", "topiq_nr"),
        ),
    )
    if value is None:
        raise ReportError(f"{context} is missing topiq_nr_score")
    return _finite_float(value, f"{context} topiq_nr_score")


def _record_total_seconds(record: Mapping[str, Any], context: str) -> float:
    value = _first_nested(
        (record,),
        (
            ("timing_seconds", "total"),
            ("total_seconds",),
            ("timing", "total_seconds"),
        ),
    )
    if value is None:
        fit = _nested(record, ("timing_seconds", "coverage_fit"))
        render = _nested(record, ("timing_seconds", "rgb_render_and_score"))
        if fit is not None and render is not None:
            value = _finite_float(fit, f"{context} coverage_fit seconds") + _finite_float(
                render, f"{context} render/score seconds"
            )
    if value is None:
        raise ReportError(f"{context} is missing cumulative timing_seconds.total")
    seconds = _finite_float(value, f"{context} total seconds")
    if seconds < 0.0:
        raise ReportError(f"{context} total seconds must be non-negative")
    return seconds


def _record_coverage_pass(record: Mapping[str, Any], context: str) -> bool:
    value = _first_nested(
        (record,),
        (
            ("coverage_qualified_at_output",),
            ("coverage", "output_constraints_met"),
            ("coverage", "constraints_met"),
            ("coverage_qualified",),
        ),
    )
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if value is None:
        raise ReportError(f"{context} is missing output coverage qualification")
    raise ReportError(f"{context} coverage qualification must be boolean")


def _canonical_hash(value: Any, context: str) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReportError(f"cannot hash {context}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _candidate_identity(
    summary: Mapping[str, Any],
    marker: Mapping[str, Any],
    experiment_root: Path,
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str, int | None]:
    explicit = _first_nested(
        (summary, marker),
        (
            ("candidate_plan_sha256",),
            ("candidate_hash",),
            ("candidate_plan_hash",),
            ("plan_hash",),
        ),
    )
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ReportError(f"{experiment_root}: candidate hash must be a non-empty string")
        source = (
            "candidate_plan_sha256"
            if _first_nested((summary, marker), (("candidate_plan_sha256",),))
            is not None
            else "summary"
        )
        return explicit.strip(), source, None

    plan_reference = _first_nested(
        (summary, marker), (("fixed_plan",), ("plan_path",))
    )
    plan_path = _resolve_reference(plan_reference, experiment_root)
    if plan_path is None:
        default_plan = experiment_root / "fixed_plan.json"
        plan_path = default_plan if default_plan.is_file() else None
    if plan_path is not None and plan_path.is_file():
        plan = _load_json_object(plan_path, "fixed candidate plan")
        captures = plan.get("captures")
        positions = plan.get("positions")
        if not isinstance(captures, list) or not isinstance(positions, list):
            raise ReportError(
                f"fixed candidate plan lacks positions/captures arrays: {plan_path}"
            )
        payload = {
            "resolved_seed_position": plan.get("resolved_seed_position"),
            "positions": positions,
            "captures": captures,
        }
        return _canonical_hash(payload, str(plan_path)), "fixed_plan", len(captures)

    # Older runs may lack fixed_plan.json. Hash the strategy-independent candidate
    # description saved with each result instead of fitted cameras or scores.
    legacy_records = records[STRATEGY_LEGACY]
    if legacy_records:
        payload = []
        for key in sorted(legacy_records):
            record = legacy_records[key]
            payload.append(
                {
                    "key": key,
                    "candidate_id": record.get("candidate_id"),
                    "capture": record.get("capture"),
                    "planned_position": record.get("planned_position"),
                }
            )
        return (
            _canonical_hash(payload, f"candidate records in {experiment_root}"),
            "candidate_results",
            len(payload),
        )

    fingerprint = _first_nested(
        (summary, marker), (("config_fingerprint",),)
    )
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint, "config_fingerprint_fallback", None
    raise ReportError(
        f"{experiment_root}: cannot determine candidate hash; expected an explicit "
        "hash, fixed_plan.json, candidate records, or config_fingerprint"
    )


def _score_threshold(
    summary: Mapping[str, Any],
    marker: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    override: float | None,
    experiment_root: Path,
) -> float:
    if override is not None:
        return override
    value = _first_nested(
        (summary, marker, camera_config),
        (
            ("topiq_nr_threshold_l",),
            ("score_threshold",),
            ("random_quality", "topiq_nr_threshold_l"),
            ("initialization", "traversal", "random", "topiq_nr_threshold_l"),
            ("config", "initialization", "traversal", "random", "topiq_nr_threshold_l"),
        ),
    )
    if value is None:
        raise ReportError(
            f"{experiment_root}: cannot determine TOPIQ-NR threshold from summary.json, "
            "experiment.json, or camera_config; pass --score-threshold"
        )
    return _finite_float(value, f"{experiment_root} TOPIQ-NR threshold")


def _scene_and_input(
    summary: Mapping[str, Any],
    marker: Mapping[str, Any],
    scene_config: Mapping[str, Any],
    scene_config_path: Path | None,
    experiment_root: Path,
) -> tuple[str, str]:
    sources = (summary, marker, scene_config)
    scene = _first_nested(
        sources,
        (
            ("scene_name",),
            ("scene", "name"),
            ("output", "scene_name"),
        ),
    )
    if scene is None and scene_config_path is not None:
        scene = scene_config_path.stem
    if scene is None:
        scene = experiment_root.name

    input_path = _first_nested(
        sources,
        (("input_ply_path",), ("input", "ply_path"), ("input_path",)),
    )
    if input_path is None:
        raise ReportError(
            f"{experiment_root}: cannot determine input path from summary, marker, "
            "or scene config"
        )
    return str(scene), str(input_path)


def summarize_experiment(
    summary_path: Path, score_threshold_override: float | None
) -> SceneReport:
    experiment_root = summary_path.parent
    summary = _load_json_object(summary_path, "experiment summary")
    marker = _load_marker(experiment_root)

    kind = _first_nested((summary, marker), (("kind",), ("experiment_kind",)))
    if kind is not None and kind != "object_coverage_ablation":
        raise ReportError(
            f"{summary_path}: expected object_coverage_ablation, got {kind!r}"
        )

    results_reference = _first_nested(
        (summary, marker), (("results_jsonl",), ("candidate_results",))
    )
    if results_reference is None:
        results_path = experiment_root / "candidate_results.jsonl"
    else:
        results_path = _resolve_reference(results_reference, experiment_root)
        if results_path is None:
            results_path = experiment_root / str(results_reference)
    records = _load_terminal_records(results_path)

    legacy_keys = set(records[STRATEGY_LEGACY])
    current_keys = set(records[STRATEGY_CURRENT])
    if not legacy_keys or not current_keys:
        raise ReportError(
            f"{results_path}: both {STRATEGY_LEGACY} and {STRATEGY_CURRENT} "
            "must have terminal records"
        )
    if legacy_keys != current_keys:
        legacy_only = sorted(legacy_keys - current_keys)[:5]
        current_only = sorted(current_keys - legacy_keys)[:5]
        raise ReportError(
            f"{results_path}: strategies are not paired on identical candidates; "
            f"legacy-only={legacy_only}, current-only={current_only}"
        )

    summary_count_value = _first_nested(
        (summary, marker), (("candidate_count",), ("capture_count",))
    )
    summary_count = (
        _nonnegative_int(summary_count_value, f"{summary_path} candidate_count")
        if summary_count_value is not None
        else None
    )
    candidate_hash, hash_source, plan_count = _candidate_identity(
        summary, marker, experiment_root, records
    )
    record_count = len(legacy_keys)
    for label, count in (("summary", summary_count), ("fixed plan", plan_count)):
        if count is not None and count != record_count:
            raise ReportError(
                f"{experiment_root}: {label} candidate count is {count}, but paired "
                f"terminal results contain {record_count} candidates"
            )

    scene_config_reference = _first_nested(
        (summary, marker), (("scene_config",),)
    )
    scene_config, scene_config_path = _load_config_if_present(
        scene_config_reference, experiment_root, "scene config"
    )
    camera_config_reference = _first_nested(
        (summary, marker), (("camera_config",),)
    )
    camera_config, _ = _load_config_if_present(
        camera_config_reference, experiment_root, "camera config"
    )
    threshold = _score_threshold(
        summary, marker, camera_config, score_threshold_override, experiment_root
    )
    scene, input_path = _scene_and_input(
        summary, marker, scene_config, scene_config_path, experiment_root
    )

    strategy_reports: dict[str, StrategyReport] = {}
    coverage_by_strategy: dict[str, dict[str, bool]] = {}
    for strategy in STRATEGIES:
        scores = []
        total_seconds = 0.0
        coverage_by_strategy[strategy] = {}
        for key in sorted(records[strategy]):
            context = f"{results_path} {strategy} {key}"
            record = records[strategy][key]
            total_seconds += _record_total_seconds(record, context)
            coverage_pass = _record_coverage_pass(record, context)
            coverage_by_strategy[strategy][key] = coverage_pass
            if record.get("status") == "ok" and coverage_pass:
                score_value = _first_nested(
                    (record,),
                    (
                        ("topiq_nr_score",),
                        ("score", "topiq_nr"),
                        ("metrics", "topiq_nr"),
                    ),
                )
                if score_value is not None:
                    try:
                        score = float(score_value)
                    except (TypeError, ValueError):
                        score = math.nan
                    if math.isfinite(score):
                        scores.append(score)
        strategy_reports[strategy] = StrategyReport(
            scores=tuple(scores),
            threshold_pass_count=sum(score > threshold for score in scores),
            total_seconds=total_seconds,
        )

    both_pass = legacy_only = current_only = neither_pass = 0
    for key in sorted(legacy_keys):
        legacy_pass = coverage_by_strategy[STRATEGY_LEGACY][key]
        current_pass = coverage_by_strategy[STRATEGY_CURRENT][key]
        if legacy_pass and current_pass:
            both_pass += 1
        elif legacy_pass:
            legacy_only += 1
        elif current_pass:
            current_only += 1
        else:
            neither_pass += 1

    return SceneReport(
        scene=scene,
        input_path=input_path,
        experiment_root=experiment_root,
        candidate_hash=candidate_hash,
        candidate_hash_source=hash_source,
        candidate_count=record_count,
        score_threshold=threshold,
        score_thresholds=(threshold,),
        strategies=strategy_reports,
        coverage_both_pass=both_pass,
        coverage_legacy_only=legacy_only,
        coverage_current_only=current_only,
        coverage_neither_pass=neither_pass,
    )


def _aggregate(reports: Sequence[SceneReport]) -> SceneReport:
    thresholds = tuple(sorted({threshold for report in reports for threshold in report.score_thresholds}))
    strategy_reports = {}
    for strategy in STRATEGIES:
        scores = tuple(
            score
            for report in reports
            for score in report.strategies[strategy].scores
        )
        strategy_reports[strategy] = StrategyReport(
            scores=scores,
            threshold_pass_count=sum(
                report.strategies[strategy].threshold_pass_count for report in reports
            ),
            total_seconds=sum(
                report.strategies[strategy].total_seconds for report in reports
            ),
        )
    aggregate_hash = _canonical_hash(
        sorted(
            (report.scene, report.input_path, report.candidate_hash)
            for report in reports
        ),
        "aggregate candidate identities",
    )
    return SceneReport(
        scene="OVERALL",
        input_path=f"{len(reports)} scenes",
        experiment_root=None,
        candidate_hash=aggregate_hash,
        candidate_hash_source="aggregate",
        candidate_count=sum(report.candidate_count for report in reports),
        score_threshold=thresholds[0] if len(thresholds) == 1 else None,
        score_thresholds=thresholds,
        strategies=strategy_reports,
        coverage_both_pass=sum(report.coverage_both_pass for report in reports),
        coverage_legacy_only=sum(report.coverage_legacy_only for report in reports),
        coverage_current_only=sum(report.coverage_current_only for report in reports),
        coverage_neither_pass=sum(report.coverage_neither_pass for report in reports),
        scene_count=len(reports),
        row_type="overall",
    )


def _float_text(value: float) -> str:
    return format(value, ".10g")


def _optional_float_text(value: float | None, missing: str = "") -> str:
    return _float_text(value) if value is not None else missing


def _threshold_text(report: SceneReport) -> str:
    if report.score_threshold is not None:
        return _float_text(report.score_threshold)
    return "mixed: " + ", ".join(_float_text(value) for value in report.score_thresholds)


def _csv_row(report: SceneReport) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": report.row_type,
        "scene_count": report.scene_count,
        "scene": report.scene,
        "input": report.input_path,
        "experiment_root": str(report.experiment_root) if report.experiment_root else "",
        "candidate_hash": report.candidate_hash,
        "candidate_hash_source": report.candidate_hash_source,
        "candidate_count": report.candidate_count,
        "score_threshold_operator": ">",
        "score_threshold": (
            _float_text(report.score_threshold)
            if report.score_threshold is not None
            else ""
        ),
        "score_thresholds": ";".join(
            _float_text(value) for value in report.score_thresholds
        ),
        "paired_coverage_both_pass": report.coverage_both_pass,
        "paired_coverage_legacy_only": report.coverage_legacy_only,
        "paired_coverage_current_only": report.coverage_current_only,
        "paired_coverage_neither_pass": report.coverage_neither_pass,
    }
    for strategy in STRATEGIES:
        stats = report.strategies[strategy]
        prefix = strategy
        row[f"{prefix}_score_count"] = stats.count
        row[f"{prefix}_score_mean"] = _optional_float_text(stats.mean, "n/a")
        row[f"{prefix}_score_median"] = _optional_float_text(
            stats.median, "n/a"
        )
        row[f"{prefix}_score_min"] = _optional_float_text(stats.minimum, "n/a")
        row[f"{prefix}_score_max"] = _optional_float_text(stats.maximum, "n/a")
        row[f"{prefix}_threshold_pass_count"] = stats.threshold_pass_count
        row[f"{prefix}_total_seconds"] = _float_text(stats.total_seconds)
    return row


def _render_csv(reports: Sequence[SceneReport]) -> str:
    rows = [_csv_row(report) for report in reports]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(item) for item in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _render_markdown(reports: Sequence[SceneReport]) -> str:
    overview_rows = []
    score_rows = []
    coverage_rows = []
    for report in reports:
        overview_rows.append(
            (
                report.scene,
                report.input_path,
                report.candidate_count,
                report.candidate_hash[:12],
                report.candidate_hash_source,
                _threshold_text(report),
            )
        )
        for strategy in STRATEGIES:
            stats = report.strategies[strategy]
            score_rows.append(
                (
                    report.scene,
                    strategy,
                    stats.count,
                    _optional_float_text(stats.mean, "n/a"),
                    _optional_float_text(stats.median, "n/a"),
                    _optional_float_text(stats.minimum, "n/a"),
                    _optional_float_text(stats.maximum, "n/a"),
                    stats.threshold_pass_count,
                    f"{stats.total_seconds:.3f}",
                )
            )
        coverage_rows.append(
            (
                report.scene,
                report.coverage_both_pass,
                report.coverage_legacy_only,
                report.coverage_current_only,
                report.coverage_neither_pass,
                report.candidate_count,
            )
        )

    sections = [
        "# Object Coverage Ablation Summary",
        "",
        "Threshold pass counts use the strict comparison `TOPIQ-NR > threshold`. "
        "Times are cumulative candidate coverage-fit plus RGB-render/score seconds.",
        "",
        "## Experiments",
        "",
        _markdown_table(
            ("Scene", "Input", "Candidates", "Candidate hash", "Hash source", "Threshold"),
            overview_rows,
        ),
        "",
        "## TOPIQ-NR And Time",
        "",
        _markdown_table(
            (
                "Scene",
                "Strategy",
                "Scores",
                "Mean",
                "Median",
                "Min",
                "Max",
                "Threshold pass",
                "Total seconds",
            ),
            score_rows,
        ),
        "",
        "## Paired Output Coverage",
        "",
        _markdown_table(
            (
                "Scene",
                "Both pass",
                "Legacy only",
                "Current only",
                "Neither pass",
                "Pairs",
            ),
            coverage_rows,
        ),
        "",
    ]
    return "\n".join(sections)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed object coverage ablation directories into CSV "
            "and Markdown reports."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        metavar="EXPERIMENT_ROOT",
        help="experiment directory or its summary.json (typically about five scenes)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=f"directory for {CSV_FILENAME} and {MARKDOWN_FILENAME}",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="override TOPIQ-NR threshold for every run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.score_threshold is not None and not math.isfinite(args.score_threshold):
        raise ReportError("--score-threshold must be finite")

    summary_paths = [_summary_path(argument) for argument in args.roots]
    duplicates = sorted(
        str(path) for path in set(summary_paths) if summary_paths.count(path) > 1
    )
    if duplicates:
        raise ReportError("duplicate experiment inputs: " + ", ".join(duplicates))

    scene_reports = [
        summarize_experiment(path, args.score_threshold) for path in summary_paths
    ]
    all_reports = [*scene_reports, _aggregate(scene_reports)]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ReportError(f"output directory is not a directory: {output_dir}")
    csv_path = output_dir / CSV_FILENAME
    markdown_path = output_dir / MARKDOWN_FILENAME
    _atomic_write_text(csv_path, _render_csv(all_reports))
    _atomic_write_text(markdown_path, _render_markdown(all_reports))
    print(f"Wrote {len(scene_reports)}-scene CSV: {csv_path}")
    print(f"Wrote {len(scene_reports)}-scene Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReportError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
