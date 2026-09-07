#!/usr/bin/env python3
"""Validate completed object-coverage ablations and write a paired report.

This analyzer deliberately requires ``summary.json`` before opening the ledger.
The experiment harness writes that file atomically and only after candidate
evaluation and Top-10 materialization have completed, so an in-progress JSONL
is never treated as a finished experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


EXPERIMENT_KIND = "object_coverage_ablation"
SCHEMA_VERSION = 1
STRATEGY_LEGACY = "legacy_adaptive"
STRATEGY_CURRENT = "current_bracketed"
STRATEGIES = (STRATEGY_LEGACY, STRATEGY_CURRENT)
TOP_K = 10
DEFAULT_EXPECTED_CANDIDATES = 5000
DEFAULT_EXPECTED_SCORE_THRESHOLD = 0.65
DEFAULT_EXPECTED_SCENES = ("oxford", "canyon", "kelpies", "zwinger", "nightcity")
REPORT_JSON = "object_coverage_ablation_report.json"
REPORT_MARKDOWN = "object_coverage_ablation_report.md"


class AnalysisError(ValueError):
    """Raised when a run is incomplete or violates the experiment contract."""


@dataclass(frozen=True)
class Result:
    candidate_index: int
    candidate_id: str
    strategy: str
    coverage_pass: bool
    score: float | None
    initial_distance: float
    minimum_distance: float
    final_distance: float
    initial_pixel_ratio: float
    final_pixel_ratio: float
    nearest_gaussian_distance: float
    preview_evaluations: int
    output_evaluations: int
    evaluations: int
    fit_seconds: float
    render_score_seconds: float
    total_seconds: float
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class SceneData:
    name: str
    root: Path
    candidate_hash: str
    source_hashes: Mapping[str, str]
    score_threshold: float
    coverage_target: float
    pairs: tuple[Mapping[str, Result], ...]
    top10: Mapping[str, tuple[Mapping[str, Any], ...]]
    report: Mapping[str, Any]


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise AnalysisError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise AnalysisError(
            f"invalid JSON in {label} {path}:{error.lineno}:{error.colno}: "
            f"{error.msg}"
        ) from error
    except OSError as error:
        raise AnalysisError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must contain a JSON object: {path}")
    return value


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise AnalysisError(f"{context} is missing {key!r}")
    return mapping[key]


def _finite(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context} must be a finite number, got {value!r}") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{context} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise AnalysisError(f"{context} must be >= {minimum}, got {result}")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{context} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context} must be an integer, got {value!r}") from error
    if result != value or result < minimum:
        raise AnalysisError(f"{context} must be an integer >= {minimum}, got {value!r}")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError(f"{context} must be boolean, got {value!r}")
    return value


def _close(left: float, right: float, *, atol: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=atol)


def _assert_equal(left: Any, right: Any, context: str) -> None:
    if left != right:
        raise AnalysisError(f"{context} mismatch: {left!r} != {right!r}")


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"cannot hash fixed candidate plan: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    location = (len(sorted_values) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return sorted_values[lower]
    weight = location - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, Any]:
    data = sorted(float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev_population": None,
            "minimum": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "maximum": None,
            "mean_95ci_normal": [None, None],
        }
    mean = statistics.fmean(data)
    population_stddev = statistics.pstdev(data)
    if len(data) > 1:
        standard_error = statistics.stdev(data) / math.sqrt(len(data))
        mean_ci = [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
    else:
        mean_ci = [None, None]
    return {
        "count": len(data),
        "mean": mean,
        "median": statistics.median(data),
        "stddev_population": population_stddev,
        "minimum": data[0],
        "p10": _quantile(data, 0.10),
        "p25": _quantile(data, 0.25),
        "p75": _quantile(data, 0.75),
        "p90": _quantile(data, 0.90),
        "p95": _quantile(data, 0.95),
        "maximum": data[-1],
        "mean_95ci_normal": mean_ci,
    }


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.96
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def _mcnemar_exact_two_sided(legacy_only: int, current_only: int) -> float | None:
    discordant = legacy_only + current_only
    if discordant == 0:
        return None
    try:
        from scipy.stats import binomtest
    except ImportError as error:
        raise AnalysisError(
            "scipy is required for the numerically stable exact McNemar test"
        ) from error
    # For a 2x2 matched table, the exact McNemar test is a two-sided
    # Binomial(n=discordant, p=0.5) test on either off-diagonal count.
    return float(
        binomtest(
            legacy_only,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def _contingency(
    pairs: Sequence[Mapping[str, Result]],
    predicate: Callable[[Result], bool],
) -> dict[str, Any]:
    counts = {"both_pass": 0, "legacy_only": 0, "current_only": 0, "neither_pass": 0}
    for pair in pairs:
        legacy_pass = predicate(pair[STRATEGY_LEGACY])
        current_pass = predicate(pair[STRATEGY_CURRENT])
        if legacy_pass and current_pass:
            counts["both_pass"] += 1
        elif legacy_pass:
            counts["legacy_only"] += 1
        elif current_pass:
            counts["current_only"] += 1
        else:
            counts["neither_pass"] += 1
    total = len(pairs)
    legacy_pass_count = counts["both_pass"] + counts["legacy_only"]
    current_pass_count = counts["both_pass"] + counts["current_only"]
    return {
        **counts,
        "pair_count": total,
        "legacy_pass_count": legacy_pass_count,
        "current_pass_count": current_pass_count,
        "legacy_pass_rate": legacy_pass_count / total if total else None,
        "current_pass_rate": current_pass_count / total if total else None,
        "legacy_pass_rate_wilson_95ci": _wilson_interval(legacy_pass_count, total),
        "current_pass_rate_wilson_95ci": _wilson_interval(current_pass_count, total),
        "current_minus_legacy_pass_count": current_pass_count - legacy_pass_count,
        "current_minus_legacy_pass_rate": (
            (current_pass_count - legacy_pass_count) / total if total else None
        ),
        "discordant_pair_count": counts["legacy_only"] + counts["current_only"],
        "mcnemar_exact_two_sided_p": _mcnemar_exact_two_sided(
            counts["legacy_only"], counts["current_only"]
        ),
    }


def _paired_delta(
    pairs: Sequence[Mapping[str, Result]],
    value: Callable[[Result], float | None],
    include: Callable[[Mapping[str, Result]], bool] = lambda _pair: True,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    deltas: list[float] = []
    for pair in pairs:
        if not include(pair):
            continue
        legacy_value = value(pair[STRATEGY_LEGACY])
        current_value = value(pair[STRATEGY_CURRENT])
        if legacy_value is None or current_value is None:
            continue
        deltas.append(current_value - legacy_value)
    return {
        "definition": "current_bracketed - legacy_adaptive",
        "delta": _stats(deltas),
        "current_higher_count": sum(delta > tie_tolerance for delta in deltas),
        "tie_count": sum(abs(delta) <= tie_tolerance for delta in deltas),
        "current_lower_count": sum(delta < -tie_tolerance for delta in deltas),
    }


def _candidate_plan_payload(
    plan: Mapping[str, Any], context: str
) -> list[dict[str, Any]]:
    positions = _require(plan, "positions", context)
    captures = _require(plan, "captures", context)
    if not isinstance(positions, list) or not isinstance(captures, list):
        raise AnalysisError(f"{context} positions and captures must be arrays")
    by_index: dict[int, Mapping[str, Any]] = {}
    for offset, position in enumerate(positions):
        if not isinstance(position, dict):
            raise AnalysisError(f"{context} positions[{offset}] must be an object")
        index = _integer(_require(position, "index", context), f"{context} position index")
        if index in by_index:
            raise AnalysisError(f"{context} has duplicate position index {index}")
        by_index[index] = position
    payload = []
    for candidate_index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise AnalysisError(f"{context} captures[{candidate_index}] must be an object")
        position_index = _integer(
            _require(capture, "position_index", context),
            f"{context} captures[{candidate_index}].position_index",
        )
        if position_index not in by_index:
            raise AnalysisError(
                f"{context} capture {candidate_index} refers to missing position {position_index}"
            )
        payload.append(
            {
                "candidate_index": candidate_index,
                "position": by_index[position_index],
                "capture": capture,
            }
        )
    return payload


def _candidate_plan_hash(plan: Mapping[str, Any], context: str) -> tuple[str, int]:
    payload = _candidate_plan_payload(plan, context)
    return _canonical_sha256(payload), len(payload)


def _load_terminal_records(path: Path) -> tuple[dict[tuple[int, str], Mapping[str, Any]], int, int]:
    terminal: dict[tuple[int, str], Mapping[str, Any]] = {}
    error_attempt_count = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as error:
        raise AnalysisError(f"candidate ledger does not exist: {path}") from error
    except OSError as error:
        raise AnalysisError(f"cannot read candidate ledger {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AnalysisError(f"blank line in candidate ledger {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnalysisError(
                    f"invalid JSON in completed ledger {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise AnalysisError(f"ledger record {path}:{line_number} must be an object")
            status = record.get("status")
            if status == "error":
                error_attempt_count += 1
                continue
            if status not in {"ok", "coverage_rejected"}:
                raise AnalysisError(f"unknown ledger status {status!r} at {path}:{line_number}")
            candidate_index = _integer(
                _require(record, "candidate_index", f"{path}:{line_number}"),
                f"{path}:{line_number} candidate_index",
            )
            strategy = str(_require(record, "strategy", f"{path}:{line_number}"))
            if strategy not in STRATEGIES:
                raise AnalysisError(f"unknown strategy {strategy!r} at {path}:{line_number}")
            key = (candidate_index, strategy)
            if key in terminal:
                raise AnalysisError(
                    f"duplicate terminal key {key!r} in completed ledger at "
                    f"{path}:{line_number}"
                )
            terminal[key] = record
    return terminal, 0, error_attempt_count


def _normalise_result(record: Mapping[str, Any], context: str) -> Result:
    schema_version = _integer(_require(record, "schema_version", context), f"{context} schema")
    _assert_equal(schema_version, SCHEMA_VERSION, f"{context} schema_version")
    candidate_index = _integer(
        _require(record, "candidate_index", context), f"{context} candidate_index"
    )
    candidate_id = str(_require(record, "candidate_id", context))
    strategy = str(_require(record, "strategy", context))
    status = str(_require(record, "status", context))
    coverage_pass = _boolean(
        _require(record, "coverage_qualified_at_output", context),
        f"{context} coverage_qualified_at_output",
    )
    if (status == "ok") != coverage_pass:
        raise AnalysisError(
            f"{context} status={status!r} is inconsistent with coverage_pass={coverage_pass}"
        )
    score_value = record.get("topiq_nr_score")
    if coverage_pass:
        if score_value is None:
            raise AnalysisError(f"{context} passed coverage but has no TOPIQ-NR score")
        score: float | None = _finite(score_value, f"{context} topiq_nr_score")
    else:
        if score_value is not None:
            raise AnalysisError(f"{context} rejected coverage but unexpectedly has a score")
        score = None

    coverage = _require(record, "coverage", context)
    timing = _require(record, "timing_seconds", context)
    if not isinstance(coverage, dict) or not isinstance(timing, dict):
        raise AnalysisError(f"{context} coverage and timing_seconds must be objects")
    output_constraints = _boolean(
        _require(coverage, "output_constraints_met", context),
        f"{context} coverage.output_constraints_met",
    )
    constraints = _boolean(
        _require(coverage, "constraints_met", context),
        f"{context} coverage.constraints_met",
    )
    if output_constraints != coverage_pass or constraints != coverage_pass:
        raise AnalysisError(f"{context} has inconsistent coverage constraint flags")
    initial_distance = _finite(
        _require(coverage, "initial_distance_to_center", context),
        f"{context} initial distance",
        minimum=0.0,
    )
    minimum_distance = _finite(
        _require(coverage, "minimum_exterior_distance_to_center", context),
        f"{context} minimum exterior distance",
        minimum=0.0,
    )
    final_distance = _finite(
        _require(coverage, "final_distance_to_center", context),
        f"{context} final distance",
        minimum=0.0,
    )
    initial_pixel_ratio = _finite(
        _require(coverage, "initial_pixel_ratio", context),
        f"{context} initial pixel ratio",
        minimum=0.0,
    )
    final_pixel_ratio = _finite(
        _require(coverage, "final_pixel_ratio", context),
        f"{context} final pixel ratio",
        minimum=0.0,
    )
    nearest_gaussian_distance = _finite(
        _require(coverage, "final_distance_to_nearest_effective_gaussian", context),
        f"{context} nearest Gaussian distance",
        minimum=0.0,
    )
    preview_evaluations = _integer(
        _require(coverage, "preview_evaluation_count", context),
        f"{context} preview evaluation count",
        minimum=1,
    )
    output_evaluations = _integer(
        _require(coverage, "output_validation_count", context),
        f"{context} output evaluation count",
    )
    evaluations = _integer(
        _require(coverage, "evaluation_count", context),
        f"{context} evaluation count",
        minimum=1,
    )
    if evaluations != preview_evaluations + output_evaluations:
        raise AnalysisError(f"{context} evaluation counts do not add up")
    fit_seconds = _finite(
        _require(timing, "coverage_fit", context), f"{context} coverage fit time", minimum=0.0
    )
    render_score_seconds = _finite(
        _require(timing, "rgb_render_and_score", context),
        f"{context} RGB/TOPIQ time",
        minimum=0.0,
    )
    total_seconds = _finite(
        _require(timing, "total", context), f"{context} total time", minimum=0.0
    )
    if not _close(total_seconds, fit_seconds + render_score_seconds, atol=1e-7):
        raise AnalysisError(f"{context} timing components do not add up")
    if not coverage_pass and render_score_seconds != 0.0:
        raise AnalysisError(f"{context} rejected coverage but has non-zero RGB/TOPIQ time")
    return Result(
        candidate_index=candidate_index,
        candidate_id=candidate_id,
        strategy=strategy,
        coverage_pass=coverage_pass,
        score=score,
        initial_distance=initial_distance,
        minimum_distance=minimum_distance,
        final_distance=final_distance,
        initial_pixel_ratio=initial_pixel_ratio,
        final_pixel_ratio=final_pixel_ratio,
        nearest_gaussian_distance=nearest_gaussian_distance,
        preview_evaluations=preview_evaluations,
        output_evaluations=output_evaluations,
        evaluations=evaluations,
        fit_seconds=fit_seconds,
        render_score_seconds=render_score_seconds,
        total_seconds=total_seconds,
        raw=record,
    )


def _distance_fraction(result: Result) -> float | None:
    return result.final_distance / result.initial_distance if result.initial_distance > 0.0 else None


def _pull_in_fraction(result: Result) -> float | None:
    fraction = _distance_fraction(result)
    return 1.0 - fraction if fraction is not None else None


def _exterior_range_retained(result: Result) -> float | None:
    available = result.initial_distance - result.minimum_distance
    if available <= max(result.initial_distance, 1.0) * 1e-12:
        return None
    return (result.final_distance - result.minimum_distance) / available


def _strategy_metrics(
    pairs: Sequence[Mapping[str, Result]], strategy: str, threshold: float, target: float
) -> dict[str, Any]:
    results = [pair[strategy] for pair in pairs]
    passed = [result for result in results if result.coverage_pass]
    scores = [result.score for result in passed if result.score is not None]
    threshold_pass_count = sum(score > threshold for score in scores)
    total_seconds = sum(result.total_seconds for result in results)
    coverage_pass_count = len(passed)
    return {
        "candidate_count": len(results),
        "coverage": {
            "pass_count": coverage_pass_count,
            "pass_rate": coverage_pass_count / len(results) if results else None,
            "pass_rate_wilson_95ci": _wilson_interval(coverage_pass_count, len(results)),
            "final_pixel_ratio_all": _stats(result.final_pixel_ratio for result in results),
            "final_pixel_ratio_passed": _stats(result.final_pixel_ratio for result in passed),
            "pixel_ratio_overshoot_above_target_passed": _stats(
                result.final_pixel_ratio - target for result in passed
            ),
        },
        "topiq_nr": {
            "score_count": len(scores),
            "score": _stats(scores),
            "threshold": threshold,
            "threshold_operator": ">",
            "threshold_pass_count": threshold_pass_count,
            "threshold_pass_rate_over_all_candidates": (
                threshold_pass_count / len(results) if results else None
            ),
            "threshold_pass_rate_conditional_on_coverage": (
                threshold_pass_count / len(scores) if scores else None
            ),
        },
        "distance": {
            "initial_distance_to_center": _stats(result.initial_distance for result in results),
            "final_distance_to_center": _stats(result.final_distance for result in results),
            "pull_in_distance": _stats(
                result.initial_distance - result.final_distance for result in results
            ),
            "final_over_initial_distance": _stats(
                value for result in results if (value := _distance_fraction(result)) is not None
            ),
            "pull_in_fraction_of_initial": _stats(
                value for result in results if (value := _pull_in_fraction(result)) is not None
            ),
            "exterior_range_retained": _stats(
                value
                for result in results
                if (value := _exterior_range_retained(result)) is not None
            ),
            "nearest_effective_gaussian_distance": _stats(
                result.nearest_gaussian_distance for result in results
            ),
        },
        "coverage_evaluations": {
            "preview": _stats(result.preview_evaluations for result in results),
            "output_validation": _stats(result.output_evaluations for result in results),
            "total": _stats(result.evaluations for result in results),
            "total_evaluation_count": sum(result.evaluations for result in results),
        },
        "timing_seconds": {
            "coverage_fit": _stats(result.fit_seconds for result in results),
            "rgb_render_and_topiq": _stats(result.render_score_seconds for result in results),
            "rgb_render_and_topiq_when_scored": _stats(
                result.render_score_seconds for result in passed
            ),
            "total": _stats(result.total_seconds for result in results),
            "coverage_fit_sum": sum(result.fit_seconds for result in results),
            "rgb_render_and_topiq_sum": sum(result.render_score_seconds for result in results),
            "total_sum": total_seconds,
        },
        "sampling_efficiency": {
            "coverage_passes_per_1000_candidates": (
                1000.0 * coverage_pass_count / len(results) if results else None
            ),
            "threshold_passes_per_1000_candidates": (
                1000.0 * threshold_pass_count / len(results) if results else None
            ),
            "candidates_per_compute_hour": (
                3600.0 * len(results) / total_seconds if total_seconds > 0.0 else None
            ),
            "coverage_passes_per_compute_hour": (
                3600.0 * coverage_pass_count / total_seconds if total_seconds > 0.0 else None
            ),
            "threshold_passes_per_compute_hour": (
                3600.0 * threshold_pass_count / total_seconds if total_seconds > 0.0 else None
            ),
            "compute_seconds_per_threshold_pass": (
                total_seconds / threshold_pass_count if threshold_pass_count else None
            ),
        },
    }


def _core_metrics(
    pairs: Sequence[Mapping[str, Result]], threshold: float, target: float
) -> dict[str, Any]:
    both_coverage = lambda pair: (
        pair[STRATEGY_LEGACY].coverage_pass and pair[STRATEGY_CURRENT].coverage_pass
    )
    threshold_predicate = lambda result: (
        result.coverage_pass and result.score is not None and result.score > threshold
    )
    strategy_metrics = {
        strategy: _strategy_metrics(pairs, strategy, threshold, target)
        for strategy in STRATEGIES
    }
    legacy_fit_sum = strategy_metrics[STRATEGY_LEGACY]["timing_seconds"]["coverage_fit_sum"]
    current_fit_sum = strategy_metrics[STRATEGY_CURRENT]["timing_seconds"]["coverage_fit_sum"]
    legacy_total_sum = strategy_metrics[STRATEGY_LEGACY]["timing_seconds"]["total_sum"]
    current_total_sum = strategy_metrics[STRATEGY_CURRENT]["timing_seconds"]["total_sum"]
    return {
        "candidate_pair_count": len(pairs),
        "coverage": {
            "minimum_pixel_ratio": target,
            "paired_outcome": _contingency(pairs, lambda result: result.coverage_pass),
            "paired_final_pixel_ratio_current_minus_legacy": _paired_delta(
                pairs, lambda result: result.final_pixel_ratio
            ),
            "paired_passed_overshoot_current_minus_legacy": _paired_delta(
                pairs,
                lambda result: result.final_pixel_ratio - target,
                include=both_coverage,
            ),
        },
        "topiq_nr": {
            "threshold": threshold,
            "threshold_operator": ">",
            "paired_threshold_outcome_over_all_candidates": _contingency(
                pairs, threshold_predicate
            ),
            "paired_score_on_both_coverage_pass": _paired_delta(
                pairs, lambda result: result.score, include=both_coverage
            ),
        },
        "distance": {
            "note": (
                "Raw distances are scene units. Cross-scene interpretation should use "
                "the normalized fractions. Positive current-minus-legacy final-distance "
                "delta means the current method kept the camera farther from the object."
            ),
            "paired_final_distance_current_minus_legacy_all": _paired_delta(
                pairs, lambda result: result.final_distance
            ),
            "paired_final_distance_fraction_current_minus_legacy_all": _paired_delta(
                pairs, _distance_fraction
            ),
            "paired_final_distance_fraction_current_minus_legacy_both_pass": _paired_delta(
                pairs, _distance_fraction, include=both_coverage
            ),
            "paired_pull_in_fraction_current_minus_legacy_all": _paired_delta(
                pairs, _pull_in_fraction
            ),
            "paired_exterior_range_retained_current_minus_legacy_all": _paired_delta(
                pairs, _exterior_range_retained
            ),
        },
        "computation": {
            "paired_coverage_fit_seconds_current_minus_legacy": _paired_delta(
                pairs, lambda result: result.fit_seconds, tie_tolerance=1e-9
            ),
            "paired_coverage_evaluations_current_minus_legacy": _paired_delta(
                pairs, lambda result: float(result.evaluations)
            ),
            "aggregate_legacy_over_current_coverage_fit_speed_ratio": (
                legacy_fit_sum / current_fit_sum if current_fit_sum > 0.0 else None
            ),
            "aggregate_legacy_over_current_total_speed_ratio": (
                legacy_total_sum / current_total_sum if current_total_sum > 0.0 else None
            ),
        },
        "strategies": strategy_metrics,
    }


def _safe_relative(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{context} must be a non-empty relative path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise AnalysisError(f"{context} escapes experiment root: {value}") from error
    return candidate


def _decode_png(path: Path, context: str) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as error:
        raise AnalysisError("Pillow is required to decode and validate Top-10 PNGs") from error
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (OSError, SyntaxError, ValueError) as error:
        raise AnalysisError(f"cannot decode {context} {path}: {error}") from error
    if image_format != "PNG":
        raise AnalysisError(f"{context} is {image_format!r}, expected a PNG: {path}")
    if width <= 0 or height <= 0:
        raise AnalysisError(f"{context} has invalid dimensions {width}x{height}: {path}")
    if mode != "RGB":
        raise AnalysisError(f"{context} has mode {mode!r}, expected RGB: {path}")
    return {"format": image_format, "mode": mode, "width": width, "height": height}


def _load_top10(
    root: Path,
    summary: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Result]],
    strategy: str,
    threshold: float,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    manifests = _require(summary, "manifests", f"{root}/summary.json")
    if not isinstance(manifests, dict) or strategy not in manifests:
        raise AnalysisError(f"{root}/summary.json is missing manifest for {strategy}")
    manifest_reference = manifests[strategy]
    if not isinstance(manifest_reference, dict):
        raise AnalysisError(f"manifest reference for {strategy} must be an object")
    manifest_path = _safe_relative(
        root, _require(manifest_reference, "path", f"manifest reference {strategy}"), "manifest path"
    )
    manifest = _load_object(manifest_path, f"{strategy} Top-10 manifest")
    _assert_equal(
        _integer(_require(manifest, "schema_version", str(manifest_path)), "manifest schema"),
        SCHEMA_VERSION,
        f"{strategy} manifest schema",
    )
    _assert_equal(manifest.get("strategy"), strategy, f"{strategy} manifest strategy")
    manifest_top_k = _integer(_require(manifest, "top_k", str(manifest_path)), "manifest top_k")
    _assert_equal(manifest_top_k, TOP_K, f"{strategy} manifest top_k")
    _assert_equal(
        _integer(_require(manifest_reference, "top_k", "summary manifest reference"), "summary top_k"),
        TOP_K,
        f"{strategy} summary manifest top_k",
    )
    manifest_threshold = _finite(
        _require(manifest, "score_threshold", str(manifest_path)), "manifest threshold"
    )
    if not _close(manifest_threshold, threshold):
        raise AnalysisError(
            f"{strategy} manifest threshold {manifest_threshold} != experiment threshold {threshold}"
        )
    _assert_equal(manifest.get("score_threshold_operator"), ">", "manifest threshold operator")

    scored = sorted(
        (
            pair[strategy]
            for pair in pairs
            if pair[strategy].coverage_pass and pair[strategy].score is not None
        ),
        key=lambda result: (-float(result.score), result.candidate_index),
    )
    if len(scored) < TOP_K:
        raise AnalysisError(f"{strategy} has only {len(scored)} scored results; Top-10 is impossible")
    _assert_equal(
        _integer(
            _require(manifest, "candidate_result_count", str(manifest_path)),
            "manifest candidate_result_count",
        ),
        len(scored),
        f"{strategy} candidate_result_count",
    )
    all_threshold_passes = sum(float(result.score) > threshold for result in scored)
    _assert_equal(
        _integer(_require(manifest, "threshold_pass_count", str(manifest_path)), "manifest pass count"),
        all_threshold_passes,
        f"{strategy} manifest threshold pass count",
    )
    _assert_equal(
        _boolean(_require(manifest, "fallback_used", str(manifest_path)), "manifest fallback"),
        all_threshold_passes < TOP_K,
        f"{strategy} manifest fallback flag",
    )
    items = _require(manifest, "items", str(manifest_path))
    if not isinstance(items, list) or len(items) != TOP_K:
        raise AnalysisError(f"{strategy} manifest must contain exactly {TOP_K} items")

    report_items: list[Mapping[str, Any]] = []
    for rank, (item, expected) in enumerate(zip(items, scored[:TOP_K]), start=1):
        if not isinstance(item, dict):
            raise AnalysisError(f"{strategy} manifest rank {rank} must be an object")
        _assert_equal(_integer(item.get("rank"), "manifest rank", minimum=1), rank, "manifest rank")
        _assert_equal(
            _integer(item.get("candidate_index"), "manifest candidate index"),
            expected.candidate_index,
            f"{strategy} rank {rank} candidate",
        )
        _assert_equal(item.get("candidate_id"), expected.candidate_id, f"{strategy} rank {rank} id")
        selection_score = _finite(item.get("topiq_nr_score"), f"{strategy} rank {rank} score")
        if not _close(selection_score, float(expected.score), atol=1e-12):
            raise AnalysisError(f"{strategy} rank {rank} score does not match ledger")
        rerender_score = _finite(item.get("rerender_score"), f"{strategy} rank {rank} rerender")
        if abs(rerender_score - selection_score) > 1e-5:
            raise AnalysisError(f"{strategy} rank {rank} rerender drift exceeds 1e-5")
        image_path = _safe_relative(root, item.get("image"), f"{strategy} rank {rank} image")
        metadata_path = _safe_relative(
            root, item.get("metadata"), f"{strategy} rank {rank} metadata"
        )
        if not image_path.is_file() or not metadata_path.is_file():
            raise AnalysisError(f"{strategy} rank {rank} image/metadata pair is incomplete")
        decoded_image = _decode_png(image_path, f"{strategy} rank {rank} image")
        output_size = expected.raw["coverage"].get("output_validation_size")
        if (
            not isinstance(output_size, list)
            or len(output_size) != 2
            or any(isinstance(value, bool) for value in output_size)
        ):
            raise AnalysisError(
                f"{strategy} rank {rank} ledger coverage lacks output_validation_size"
            )
        expected_width = _integer(output_size[0], "output validation width", minimum=1)
        expected_height = _integer(output_size[1], "output validation height", minimum=1)
        if (decoded_image["width"], decoded_image["height"]) != (
            expected_width,
            expected_height,
        ):
            raise AnalysisError(
                f"{strategy} rank {rank} decoded image size "
                f"{decoded_image['width']}x{decoded_image['height']} != "
                f"ledger output size {expected_width}x{expected_height}"
            )
        metadata = _load_object(metadata_path, f"{strategy} rank {rank} metadata")
        _assert_equal(metadata.get("strategy"), strategy, f"{strategy} rank {rank} metadata strategy")
        _assert_equal(metadata.get("selection_rank"), rank, f"{strategy} rank {rank} metadata rank")
        _assert_equal(
            metadata.get("candidate_index"),
            expected.candidate_index,
            f"{strategy} rank {rank} metadata candidate",
        )
        metadata_score = _finite(
            metadata.get("selection_score"), f"{strategy} rank {rank} metadata score"
        )
        if not _close(metadata_score, selection_score, atol=1e-12):
            raise AnalysisError(f"{strategy} rank {rank} metadata score does not match")
        report_items.append(
            {
                "rank": rank,
                "candidate_index": expected.candidate_index,
                "candidate_id": expected.candidate_id,
                "topiq_nr_score": selection_score,
                "passes_score_threshold": selection_score > threshold,
                "rerender_score": rerender_score,
                "rerender_score_delta": rerender_score - selection_score,
                "final_pixel_ratio": expected.final_pixel_ratio,
                "final_distance_to_center": expected.final_distance,
                "coverage_evaluation_count": expected.evaluations,
                "decoded_image": decoded_image,
                "image": str(image_path.relative_to(root)),
                "metadata": str(metadata_path.relative_to(root)),
            }
        )
    top10_report = {
        "selection_policy": "top_k_by_topiq_nr_over_full_coverage_qualified_pool",
        "saved_count": TOP_K,
        "coverage_qualified_pool_count": len(scored),
        "full_pool_threshold_pass_count": all_threshold_passes,
        "fallback_used": all_threshold_passes < TOP_K,
        "top10_threshold_pass_count": sum(
            item["passes_score_threshold"] for item in report_items
        ),
        "top10_score": _stats(item["topiq_nr_score"] for item in report_items),
    }
    return tuple(report_items), top10_report


def _top10_comparison(
    top10: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    legacy_by_index = {
        int(item["candidate_index"]): item for item in top10[STRATEGY_LEGACY]
    }
    current_by_index = {
        int(item["candidate_index"]): item for item in top10[STRATEGY_CURRENT]
    }
    overlap = sorted(set(legacy_by_index) & set(current_by_index))
    union = set(legacy_by_index) | set(current_by_index)
    return {
        "candidate_overlap_count": len(overlap),
        "candidate_union_count": len(union),
        "jaccard_similarity": len(overlap) / len(union) if union else None,
        "overlap_candidate_indices": overlap,
        "score_current_minus_legacy_on_overlap": _stats(
            current_by_index[index]["topiq_nr_score"]
            - legacy_by_index[index]["topiq_nr_score"]
            for index in overlap
        ),
    }


def _summary_root(argument: str | os.PathLike[str]) -> tuple[Path, Path]:
    path = Path(argument).expanduser().resolve()
    if path.is_dir():
        root = path
        summary_path = root / "summary.json"
    elif path.name == "summary.json":
        summary_path = path
        root = path.parent
    else:
        raise AnalysisError(f"input must be an experiment directory or summary.json: {path}")
    # Do not inspect candidate_results.jsonl unless the atomic completion marker exists.
    if not summary_path.is_file():
        raise AnalysisError(
            f"experiment is not complete (summary.json is absent); ledger was not read: {root}"
        )
    return root, summary_path


def analyze_scene(
    argument: str | os.PathLike[str],
    *,
    expected_candidates: int = DEFAULT_EXPECTED_CANDIDATES,
    expected_score_threshold: float = DEFAULT_EXPECTED_SCORE_THRESHOLD,
) -> SceneData:
    root, summary_path = _summary_root(argument)
    summary = _load_object(summary_path, "experiment summary")
    marker = _load_object(root / "experiment.json", "experiment marker")
    for key in (
        "kind",
        "schema_version",
        "config_fingerprint",
        "input_ply_path",
        "source_files_sha256",
        "strategies",
        "top_k",
        "full_candidate_pool",
        "early_termination",
        "minimum_pixel_ratio",
        "topiq_nr_threshold_l",
    ):
        _assert_equal(summary.get(key), marker.get(key), f"{root} marker/summary {key}")
    _assert_equal(marker.get("kind"), EXPERIMENT_KIND, f"{root} experiment kind")
    _assert_equal(marker.get("schema_version"), SCHEMA_VERSION, f"{root} schema")
    _assert_equal(marker.get("strategies"), list(STRATEGIES), f"{root} strategies")
    _assert_equal(marker.get("top_k"), TOP_K, f"{root} top_k")
    _assert_equal(marker.get("full_candidate_pool"), True, f"{root} full candidate pool")
    _assert_equal(marker.get("early_termination"), False, f"{root} early termination")
    if not summary.get("completed_at"):
        raise AnalysisError(f"{summary_path} has no completed_at timestamp")
    candidate_count = _integer(summary.get("candidate_count"), f"{root} candidate_count")
    _assert_equal(candidate_count, expected_candidates, f"{root} full candidate count")
    _assert_equal(
        _integer(summary.get("strategy_evaluation_count"), f"{root} strategy count"),
        candidate_count * len(STRATEGIES),
        f"{root} strategy evaluation count",
    )
    threshold = _finite(marker.get("topiq_nr_threshold_l"), f"{root} TOPIQ threshold")
    if not _close(threshold, expected_score_threshold, atol=1e-12):
        raise AnalysisError(
            f"{root} TOPIQ threshold is {threshold}, expected {expected_score_threshold}"
        )
    target = _finite(
        marker.get("minimum_pixel_ratio"), f"{root} minimum pixel ratio", minimum=0.0
    )
    source_hashes = marker.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise AnalysisError(f"{root} source_files_sha256 must be a non-empty object")
    for label, digest in source_hashes.items():
        if not isinstance(label, str) or not isinstance(digest, str) or len(digest) != 64:
            raise AnalysisError(f"{root} has invalid source hash entry {label!r}: {digest!r}")

    plan = _load_object(root / "fixed_plan.json", "fixed candidate plan")
    _assert_equal(
        plan.get("config_fingerprint"), marker.get("config_fingerprint"), f"{root} plan fingerprint"
    )
    plan_candidates = _candidate_plan_payload(plan, f"{root}/fixed_plan.json")
    plan_hash = _canonical_sha256(plan_candidates)
    plan_count = len(plan_candidates)
    _assert_equal(plan_count, candidate_count, f"{root} fixed plan count")
    _assert_equal(plan.get("candidate_plan_sha256"), plan_hash, f"{root} embedded plan hash")
    _assert_equal(summary.get("candidate_plan_sha256"), plan_hash, f"{root} summary plan hash")

    results_reference = summary.get("results_jsonl", "candidate_results.jsonl")
    results_path = _safe_relative(root, results_reference, f"{root} results_jsonl")
    terminal, duplicate_terminal_count, error_attempt_count = _load_terminal_records(results_path)
    expected_keys = {
        (candidate_index, strategy)
        for candidate_index in range(candidate_count)
        for strategy in STRATEGIES
    }
    if set(terminal) != expected_keys:
        missing = sorted(expected_keys - set(terminal))[:10]
        unexpected = sorted(set(terminal) - expected_keys)[:10]
        raise AnalysisError(
            f"{root} terminal ledger is not the complete paired pool; "
            f"missing={missing}, unexpected={unexpected}"
        )

    pairs: list[Mapping[str, Result]] = []
    for candidate_index in range(candidate_count):
        pair = {
            strategy: _normalise_result(
                terminal[(candidate_index, strategy)],
                f"{root} candidate {candidate_index} {strategy}",
            )
            for strategy in STRATEGIES
        }
        legacy = pair[STRATEGY_LEGACY]
        current = pair[STRATEGY_CURRENT]
        fixed_candidate = plan_candidates[candidate_index]
        fixed_capture = fixed_candidate["capture"]
        fixed_position = fixed_candidate["position"]
        lens_index = _integer(
            _require(fixed_capture, "lens_index", "fixed-plan capture"),
            f"candidate {candidate_index} fixed-plan lens_index",
        )
        position_index = _integer(
            _require(fixed_position, "index", "fixed-plan position"),
            f"candidate {candidate_index} fixed-plan position index",
        )
        expected_candidate_id = f"position_{position_index}:lens_{lens_index}"
        expected_position = _require(
            fixed_position, "position", f"candidate {candidate_index} fixed-plan position"
        )
        for result in pair.values():
            _assert_equal(
                result.candidate_id,
                expected_candidate_id,
                f"candidate {candidate_index} candidate_id vs fixed plan",
            )
            _assert_equal(
                result.raw.get("capture"),
                fixed_capture,
                f"candidate {candidate_index} {result.strategy} capture vs fixed plan",
            )
            _assert_equal(
                result.raw.get("planned_position"),
                expected_position,
                f"candidate {candidate_index} {result.strategy} planned_position vs fixed plan",
            )
        if not _close(legacy.initial_distance, current.initial_distance, atol=1e-8):
            raise AnalysisError(f"candidate {candidate_index} strategies have different initial distance")
        for result in pair.values():
            record_target = _finite(
                result.raw["coverage"].get("minimum_pixel_ratio"),
                f"candidate {candidate_index} coverage target",
            )
            if not _close(record_target, target, atol=1e-12):
                raise AnalysisError(f"candidate {candidate_index} coverage target mismatch")
            if result.coverage_pass != (result.final_pixel_ratio >= target):
                raise AnalysisError(
                    f"candidate {candidate_index} {result.strategy} final ratio/qualification mismatch"
                )
        pairs.append(pair)

    core = _core_metrics(pairs, threshold, target)
    summary_paired = summary.get("paired_output_coverage")
    computed_paired = core["coverage"]["paired_outcome"]
    if isinstance(summary_paired, dict):
        for key in ("both_pass", "legacy_only", "current_only", "neither_pass"):
            _assert_equal(summary_paired.get(key), computed_paired[key], f"{root} summary coverage {key}")
    strategy_summaries = summary.get("strategy_summaries")
    if not isinstance(strategy_summaries, dict):
        raise AnalysisError(f"{root} summary is missing strategy_summaries")
    for strategy in STRATEGIES:
        item = strategy_summaries.get(strategy)
        if not isinstance(item, dict):
            raise AnalysisError(f"{root} summary is missing {strategy} statistics")
        metrics = core["strategies"][strategy]
        checks = {
            "terminal_count": candidate_count,
            "coverage_qualified_count": metrics["coverage"]["pass_count"],
            "coverage_rejected_count": candidate_count - metrics["coverage"]["pass_count"],
            "scored_count": metrics["topiq_nr"]["score_count"],
            "threshold_pass_count": metrics["topiq_nr"]["threshold_pass_count"],
        }
        for key, expected in checks.items():
            _assert_equal(item.get(key), expected, f"{root} {strategy} summary {key}")

    top10: dict[str, tuple[Mapping[str, Any], ...]] = {}
    top10_reports: dict[str, Mapping[str, Any]] = {}
    for strategy in STRATEGIES:
        top10[strategy], top10_reports[strategy] = _load_top10(
            root, summary, pairs, strategy, threshold
        )
    top10_report: dict[str, Any] = {
        "strategies": top10_reports,
        "comparison": _top10_comparison(top10),
        "items": top10,
    }

    scene_config = marker.get("scene_config")
    name = Path(scene_config).stem if isinstance(scene_config, str) else root.name
    report = {
        "scene": name,
        "experiment_root": str(root),
        "input_ply_path": marker.get("input_ply_path"),
        "completed_at": summary.get("completed_at"),
        "validation": {
            "complete": True,
            "candidate_count": candidate_count,
            "terminal_strategy_evaluation_count": len(terminal),
            "candidate_plan_sha256": plan_hash,
            "config_fingerprint": marker.get("config_fingerprint"),
            "source_files_sha256": source_hashes,
            "full_candidate_pool": True,
            "early_termination": False,
            "duplicate_terminal_record_count": duplicate_terminal_count,
            "prior_error_attempt_count": error_attempt_count,
            "top10_pairs_validated_per_strategy": TOP_K,
        },
        "metrics": core,
        "top10": top10_report,
    }
    return SceneData(
        name=name,
        root=root,
        candidate_hash=plan_hash,
        source_hashes=dict(source_hashes),
        score_threshold=threshold,
        coverage_target=target,
        pairs=tuple(pairs),
        top10=top10,
        report=report,
    )


def _aggregate(scenes: Sequence[SceneData]) -> dict[str, Any]:
    pairs = tuple(pair for scene in scenes for pair in scene.pairs)
    threshold = scenes[0].score_threshold
    target = scenes[0].coverage_target
    core = _core_metrics(pairs, threshold, target)
    top10_strategy: dict[str, Any] = {}
    for strategy in STRATEGIES:
        items = [item for scene in scenes for item in scene.top10[strategy]]
        top10_strategy[strategy] = {
            "saved_count": len(items),
            "threshold_pass_count": sum(item["passes_score_threshold"] for item in items),
            "score": _stats(item["topiq_nr_score"] for item in items),
        }
    overlaps = [scene.report["top10"]["comparison"] for scene in scenes]
    return {
        "aggregation": "micro over all fixed-plan candidate pairs",
        "scene_count": len(scenes),
        "candidate_pair_count": len(pairs),
        "metrics": core,
        "saved_top10": {
            "note": "These are the per-scene saved Top-10 sets, not a cross-scene reranking.",
            "strategies": top10_strategy,
            "overlap_count_sum": sum(item["candidate_overlap_count"] for item in overlaps),
            "overlap_count_per_scene": _stats(
                item["candidate_overlap_count"] for item in overlaps
            ),
        },
    }


def build_report(
    inputs: Sequence[str | os.PathLike[str]],
    *,
    expected_candidates: int = DEFAULT_EXPECTED_CANDIDATES,
    expected_score_threshold: float = DEFAULT_EXPECTED_SCORE_THRESHOLD,
    expected_scenes: Sequence[str] = DEFAULT_EXPECTED_SCENES,
    allow_mixed_source_hashes: bool = False,
) -> tuple[dict[str, Any], tuple[SceneData, ...]]:
    if not inputs:
        raise AnalysisError("at least one completed experiment is required")
    if expected_candidates < TOP_K:
        raise AnalysisError(f"expected candidate count must be at least {TOP_K}")
    if not math.isfinite(expected_score_threshold):
        raise AnalysisError("expected score threshold must be finite")
    expected_scene_names = tuple(str(name).strip() for name in expected_scenes)
    if not expected_scene_names or any(not name for name in expected_scene_names):
        raise AnalysisError("expected scene names must be non-empty")
    if len(expected_scene_names) != len(set(expected_scene_names)):
        raise AnalysisError(f"expected scene names contain duplicates: {expected_scene_names}")
    scenes = tuple(
        analyze_scene(
            item,
            expected_candidates=expected_candidates,
            expected_score_threshold=expected_score_threshold,
        )
        for item in inputs
    )
    names = [scene.name for scene in scenes]
    if len(names) != len(set(names)):
        raise AnalysisError(f"duplicate scene names in report inputs: {names}")
    actual_scene_names = set(names)
    expected_scene_set = set(expected_scene_names)
    if actual_scene_names != expected_scene_set:
        raise AnalysisError(
            "completed scene set mismatch; "
            f"missing={sorted(expected_scene_set - actual_scene_names)}, "
            f"unexpected={sorted(actual_scene_names - expected_scene_set)}"
        )
    thresholds = {scene.score_threshold for scene in scenes}
    targets = {scene.coverage_target for scene in scenes}
    if len(thresholds) != 1:
        raise AnalysisError(f"mixed TOPIQ thresholds cannot be pooled: {sorted(thresholds)}")
    if len(targets) != 1:
        raise AnalysisError(f"mixed coverage targets cannot be pooled: {sorted(targets)}")
    source_hash_sets = {
        json.dumps(scene.source_hashes, sort_keys=True, separators=(",", ":"))
        for scene in scenes
    }
    source_hashes_match = len(source_hash_sets) == 1
    if not source_hashes_match and not allow_mixed_source_hashes:
        raise AnalysisError(
            "scene runs use different experiment/source hashes; rerun consistently or "
            "pass --allow-mixed-source-hashes and disclose the mismatch"
        )
    report = {
        "schema_version": 1,
        "report_kind": "object_coverage_ablation_paired_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_language": "zh-CN",
        "methodology": {
            "strategies": {
                STRATEGY_LEGACY: "修改前：旧版自适应拉近搜索",
                STRATEGY_CURRENT: "修改后：有界搜索，选择满足覆盖率的最远相机",
            },
            "candidate_design": (
                "同一场景的两策略共享固定候选计划、姿态、内参、渲染配置和 "
                "TOPIQ-NR 模型；所有候选均完成，不按阈值提前终止。"
            ),
            "score_rule": f"TOPIQ-NR > {expected_score_threshold}（严格大于）",
            "coverage_rule": "final_pixel_ratio >= minimum_pixel_ratio",
            "paired_topiq_population": "仅双方都通过输出分辨率 coverage 的同候选",
            "timing_note": (
                "coverage_fit 是算法本体时间；total 还包含仅对 coverage 合格项执行的 "
                "RGB 渲染和 TOPIQ-NR，因此两者分开报告。"
            ),
            "distance_note": (
                "场景间原始距离尺度不可直接比较；总体结论优先使用 final/initial "
                "和 exterior-range-retained 等无量纲指标。"
            ),
            "statistical_tests": (
                "coverage 与端到端阈值通过使用配对四格表及双侧 exact McNemar；"
                "均值 95% 区间为正态近似描述性区间。"
            ),
        },
        "validation": {
            "scene_count": len(scenes),
            "expected_scenes": list(expected_scene_names),
            "actual_scenes": names,
            "scene_set_exact_match": True,
            "expected_candidates_per_scene": expected_candidates,
            "expected_strategy_evaluations_per_scene": expected_candidates * 2,
            "expected_top10_per_strategy_per_scene": TOP_K,
            "source_hashes_match_across_scenes": source_hashes_match,
            "mixed_source_hashes_explicitly_allowed": allow_mixed_source_hashes,
            "all_runs_complete": True,
        },
        "scenes": [scene.report for scene in scenes],
        "overall": _aggregate(scenes),
    }
    return report, scenes


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    scenes = list(report["scenes"])
    overall = report["overall"]
    score_threshold = float(overall["metrics"]["topiq_nr"]["threshold"])
    score_threshold_text = format(score_threshold, ".10g")
    rows_for = [
        (scene["scene"], scene["metrics"], scene["top10"], scene["validation"])
        for scene in scenes
    ] + [("总体（micro）", overall["metrics"], overall["saved_top10"], None)]

    coverage_rows = []
    threshold_rows = []
    score_rows = []
    distance_rows = []
    compute_rows = []
    efficiency_rows = []
    top10_rows = []
    for name, metrics, top10, _validation in rows_for:
        coverage = metrics["coverage"]["paired_outcome"]
        threshold = metrics["topiq_nr"]["paired_threshold_outcome_over_all_candidates"]
        coverage_rows.append(
            (
                name,
                coverage["both_pass"],
                coverage["legacy_only"],
                coverage["current_only"],
                coverage["neither_pass"],
                _pct(coverage["legacy_pass_rate"]),
                _pct(coverage["current_pass_rate"]),
                _pct(coverage["current_minus_legacy_pass_rate"]),
                _fmt(coverage["mcnemar_exact_two_sided_p"], 6),
            )
        )
        threshold_rows.append(
            (
                name,
                threshold["both_pass"],
                threshold["legacy_only"],
                threshold["current_only"],
                threshold["neither_pass"],
                _pct(threshold["legacy_pass_rate"]),
                _pct(threshold["current_pass_rate"]),
                _pct(threshold["current_minus_legacy_pass_rate"]),
                _fmt(threshold["mcnemar_exact_two_sided_p"], 6),
            )
        )
        paired_score = metrics["topiq_nr"]["paired_score_on_both_coverage_pass"]
        for strategy in STRATEGIES:
            item = metrics["strategies"][strategy]["topiq_nr"]
            score_rows.append(
                (
                    name,
                    strategy,
                    item["score_count"],
                    _fmt(item["score"]["mean"], 6),
                    _fmt(item["score"]["median"], 6),
                    _fmt(item["score"]["p10"], 6),
                    _fmt(item["score"]["p90"], 6),
                    item["threshold_pass_count"],
                    _pct(item["threshold_pass_rate_over_all_candidates"]),
                )
            )
        distance = metrics["distance"]
        distance_rows.append(
            (
                name,
                _fmt(
                    metrics["strategies"][STRATEGY_LEGACY]["distance"]
                    ["final_over_initial_distance"]["mean"],
                    6,
                ),
                _fmt(
                    metrics["strategies"][STRATEGY_CURRENT]["distance"]
                    ["final_over_initial_distance"]["mean"],
                    6,
                ),
                _fmt(
                    distance["paired_final_distance_fraction_current_minus_legacy_all"]
                    ["delta"]["mean"],
                    6,
                ),
                _fmt(
                    distance["paired_final_distance_fraction_current_minus_legacy_both_pass"]
                    ["delta"]["mean"],
                    6,
                ),
                _fmt(
                    metrics["coverage"]["paired_passed_overshoot_current_minus_legacy"]
                    ["delta"]["mean"],
                    6,
                ),
            )
        )
        compute = metrics["computation"]
        compute_rows.append(
            (
                name,
                _fmt(metrics["strategies"][STRATEGY_LEGACY]["coverage_evaluations"]["total"]["mean"], 3),
                _fmt(metrics["strategies"][STRATEGY_CURRENT]["coverage_evaluations"]["total"]["mean"], 3),
                _fmt(metrics["strategies"][STRATEGY_LEGACY]["timing_seconds"]["coverage_fit_sum"], 2),
                _fmt(metrics["strategies"][STRATEGY_CURRENT]["timing_seconds"]["coverage_fit_sum"], 2),
                _fmt(compute["aggregate_legacy_over_current_coverage_fit_speed_ratio"], 3),
                _fmt(compute["paired_coverage_fit_seconds_current_minus_legacy"]["delta"]["mean"], 6),
            )
        )
        for strategy in STRATEGIES:
            efficiency = metrics["strategies"][strategy]["sampling_efficiency"]
            efficiency_rows.append(
                (
                    name,
                    strategy,
                    _fmt(efficiency["coverage_passes_per_1000_candidates"], 2),
                    _fmt(efficiency["threshold_passes_per_1000_candidates"], 2),
                    _fmt(efficiency["candidates_per_compute_hour"], 2),
                    _fmt(efficiency["threshold_passes_per_compute_hour"], 2),
                    _fmt(efficiency["compute_seconds_per_threshold_pass"], 3),
                )
            )
        if name == "总体（micro）":
            for strategy in STRATEGIES:
                item = top10["strategies"][strategy]
                top10_rows.append(
                    (
                        name,
                        strategy,
                        item["saved_count"],
                        item["threshold_pass_count"],
                        _fmt(item["score"]["mean"], 6),
                        _fmt(item["score"]["minimum"], 6),
                        _fmt(item["score"]["maximum"], 6),
                        "n/a",
                    )
                )
        else:
            for strategy in STRATEGIES:
                item = top10["strategies"][strategy]
                top10_rows.append(
                    (
                        name,
                        strategy,
                        item["saved_count"],
                        item["top10_threshold_pass_count"],
                        _fmt(item["top10_score"]["mean"], 6),
                        _fmt(item["top10_score"]["minimum"], 6),
                        _fmt(item["top10_score"]["maximum"], 6),
                        _fmt(top10["comparison"]["candidate_overlap_count"]),
                    )
                )

    paired_score_rows = []
    for name, metrics, _top10, _validation in rows_for:
        paired = metrics["topiq_nr"]["paired_score_on_both_coverage_pass"]
        paired_score_rows.append(
            (
                name,
                paired["delta"]["count"],
                _fmt(paired["delta"]["mean"], 6),
                _fmt(paired["delta"]["median"], 6),
                paired["current_higher_count"],
                paired["tie_count"],
                paired["current_lower_count"],
            )
        )

    validation_rows = [
        (
            scene["scene"],
            scene["validation"]["candidate_count"],
            scene["validation"]["terminal_strategy_evaluation_count"],
            scene["validation"]["candidate_plan_sha256"][:12],
            scene["validation"]["duplicate_terminal_record_count"],
            scene["validation"]["prior_error_attempt_count"],
            scene["completed_at"],
        )
        for scene in scenes
    ]
    sections = [
        "# Object 场景相机覆盖率搜索全量配对实验报告",
        "",
        f"生成时间：`{report['generated_at']}`。所有场景均在原子 `summary.json` 存在后读取，"
        "并通过固定计划哈希、完整终态集合和 Top-10 反查校验。",
        "",
        "## 实验口径",
        "",
        report["methodology"]["candidate_design"],
        "",
        f"- Coverage：`{report['methodology']['coverage_rule']}`。",
        f"- 质量阈值：`{report['methodology']['score_rule']}`。",
        f"- Paired TOPIQ：{report['methodology']['paired_topiq_population']}。",
        f"- 时间：{report['methodology']['timing_note']}",
        f"- 距离：{report['methodology']['distance_note']}",
        "",
        "## 完整性验收",
        "",
        _table(
            ("场景", "候选对", "策略终态", "计划 SHA256", "重复终态", "历史错误", "完成时间"),
            validation_rows,
        ),
        "",
        "## Coverage 配对四格",
        "",
        _table(
            ("场景", "双方通过", "仅修改前", "仅修改后", "双方失败", "修改前率", "修改后率", "率差", "McNemar p"),
            coverage_rows,
        ),
        "",
        "## 端到端 TOPIQ 阈值配对四格",
        "",
        f"阈值通过同时要求输出 coverage 合格且 `TOPIQ-NR > {score_threshold_text}`。",
        "",
        _table(
            ("场景", "双方通过", "仅修改前", "仅修改后", "双方失败", "修改前率", "修改后率", "率差", "McNemar p"),
            threshold_rows,
        ),
        "",
        "## TOPIQ-NR 分布",
        "",
        _table(
            ("场景", "策略", "评分数", "均值", "中位数", "P10", "P90", ">阈值数", "全候选通过率"),
            score_rows,
        ),
        "",
        "### 双方 coverage 合格候选的配对分差",
        "",
        "差值定义为 `修改后 - 修改前`，正值表示修改后 TOPIQ-NR 更高。",
        "",
        _table(
            ("场景", "配对数", "平均差", "中位差", "修改后更高", "相同", "修改后更低"),
            paired_score_rows,
        ),
        "",
        "## 相机距离与覆盖率余量",
        "",
        "`final/initial` 越大表示拉近越少；双方合格时的距离率差为主要距离指标。"
        "Coverage overshoot 差越接近 0，表示两者对目标比例的过量覆盖越接近。",
        "",
        _table(
            ("场景", "修改前 final/initial", "修改后 final/initial", "全体距离率差", "双方合格距离率差", "双方合格 overshoot 差"),
            distance_rows,
        ),
        "",
        "## 迭代与耗时",
        "",
        "`旧/新 fit 加速比 > 1` 表示修改后 coverage 搜索更快。累计秒数不是并行任务的墙钟时间。",
        "",
        _table(
            ("场景", "修改前平均评估", "修改后平均评估", "修改前 fit 秒", "修改后 fit 秒", "旧/新 fit 加速比", "paired fit 秒差"),
            compute_rows,
        ),
        "",
        "## 采样效率",
        "",
        _table(
            ("场景", "策略", "每千候选 coverage", "每千候选阈值", "每计算小时候选", "每计算小时阈值", "每个阈值结果秒数"),
            efficiency_rows,
        ),
        "",
        "## 保存的 Top-10",
        "",
        _table(
            ("场景", "策略", "保存数", ">阈值数", "均值", "最低", "最高", "两策略候选重合数"),
            top10_rows,
        ),
        "",
    ]
    for scene in scenes:
        legacy_items = scene["top10"]["items"][STRATEGY_LEGACY]
        current_items = scene["top10"]["items"][STRATEGY_CURRENT]
        sections.extend(
            [
                f"### {scene['scene']} Top-10 明细",
                "",
                _table(
                    ("排名", "修改前候选", "修改前分数", "过阈值", "修改后候选", "修改后分数", "过阈值"),
                    (
                        (
                            rank,
                            legacy_items[rank - 1]["candidate_index"],
                            _fmt(legacy_items[rank - 1]["topiq_nr_score"], 6),
                            _fmt(legacy_items[rank - 1]["passes_score_threshold"]),
                            current_items[rank - 1]["candidate_index"],
                            _fmt(current_items[rank - 1]["topiq_nr_score"], 6),
                            _fmt(current_items[rank - 1]["passes_score_threshold"]),
                        )
                        for rank in range(1, TOP_K + 1)
                    ),
                ),
                "",
            ]
        )
    return "\n".join(sections)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
            "严格验收已完成的 object coverage ablation，并生成中文 paired JSON/Markdown 报告。"
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        metavar="COMPLETED_EXPERIMENT",
        help="已完成实验目录或其 summary.json；缺少 summary.json 时拒绝读取 ledger",
    )
    parser.add_argument("--output-dir", required=True, help="JSON 和 Markdown 报告输出目录")
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=DEFAULT_EXPECTED_CANDIDATES,
        help=f"每场景候选数（默认 {DEFAULT_EXPECTED_CANDIDATES}）",
    )
    parser.add_argument(
        "--expected-score-threshold",
        type=float,
        default=DEFAULT_EXPECTED_SCORE_THRESHOLD,
        help=f"要求实验使用的 TOPIQ-NR 阈值（默认 {DEFAULT_EXPECTED_SCORE_THRESHOLD}）",
    )
    parser.add_argument(
        "--expected-scenes",
        default=",".join(DEFAULT_EXPECTED_SCENES),
        help=(
            "要求且只允许出现的逗号分隔场景集合（默认 "
            + ",".join(DEFAULT_EXPECTED_SCENES)
            + "）"
        ),
    )
    parser.add_argument(
        "--allow-mixed-source-hashes",
        action="store_true",
        help="允许不同场景使用不同 harness/scene_traversal 哈希，并在报告中显式标记",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expected_scenes = tuple(
        name.strip() for name in args.expected_scenes.split(",") if name.strip()
    )
    report, _scenes = build_report(
        args.roots,
        expected_candidates=args.expected_candidates,
        expected_score_threshold=args.expected_score_threshold,
        expected_scenes=expected_scenes,
        allow_mixed_source_hashes=args.allow_mixed_source_hashes,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise AnalysisError(f"output path is not a directory: {output_dir}")
    json_path = output_dir / REPORT_JSON
    markdown_path = output_dir / REPORT_MARKDOWN
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(markdown_path, render_markdown(report))
    print(f"已验收 {len(report['scenes'])} 个完整场景")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, OSError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(2)
