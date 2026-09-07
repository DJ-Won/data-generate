#!/usr/bin/env python3
"""Build per-scene before/after contact sheets for coverage ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


TOP_K = 10
SCHEMA_VERSION = 1
EXPERIMENT_KIND = "object_coverage_ablation"
EXPECTED_CANDIDATE_COUNT = 5000
EXPECTED_STRATEGY_EVALUATION_COUNT = 10000
OUTPUT_FILENAME = "before_after_top10.png"
STRATEGIES = (
    ("legacy_adaptive", "BEFORE", "Legacy adaptive", "#D99043"),
    ("current_bracketed", "AFTER", "Current bracketed", "#4FA9C6"),
)
STRATEGY_KEYS = tuple(item[0] for item in STRATEGIES)

BACKGROUND = "#15191C"
PANEL_BACKGROUND = "#20262A"
IMAGE_BACKGROUND = "#080A0B"
LABEL_BACKGROUND = "#1B2024"
TEXT = "#F2F5F6"
MUTED_TEXT = "#AEB8BD"
PASS_COLOR = "#44BE7C"
BELOW_COLOR = "#E2A348"
BORDER_COLOR = "#465158"

MARGIN = 28
TITLE_HEIGHT = 104
SECTION_HEADER_HEIGHT = 72
SECTION_GAP = 30
GRID_COLUMNS = 5
GRID_GAP = 18
IMAGE_SIZE = 416
LABEL_HEIGHT = 78
CARD_HEIGHT = IMAGE_SIZE + LABEL_HEIGHT


class ContactSheetError(ValueError):
    """The ablation artifacts cannot produce an unambiguous contact sheet."""


@dataclass(frozen=True)
class RankedImage:
    rank: int
    score: float
    passes_threshold: bool
    image_path: Path


@dataclass(frozen=True)
class StrategyTopK:
    key: str
    stage: str
    display_name: str
    accent_color: str
    threshold: float
    threshold_operator: str
    items: tuple[RankedImage, ...]


@dataclass(frozen=True)
class SceneArtifacts:
    scene: str
    root: Path
    strategies: tuple[StrategyTopK, ...]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise ContactSheetError(f"manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ContactSheetError(
            f"invalid JSON in {path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except OSError as error:
        raise ContactSheetError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContactSheetError(f"manifest must contain a JSON object: {path}")
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ContactSheetError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContactSheetError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise ContactSheetError(f"{context} must be finite, got {value!r}")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ContactSheetError(f"{context} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ContactSheetError(f"{context} must be an integer") from error
    if result != value:
        raise ContactSheetError(f"{context} must be an integer")
    return result


def _passes_threshold(score: float, operator: str, threshold: float) -> bool:
    if operator == ">":
        return score > threshold
    raise ContactSheetError(
        f"unsupported score_threshold_operator {operator!r}; expected '>'"
    )


def _resolve_image_path(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContactSheetError(f"{context} image must be a non-empty path string")
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise ContactSheetError(f"{context} image path must be relative: {value}")
    root = root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ContactSheetError(
            f"{context} image path escapes experiment root: {value}"
        ) from error
    if not path.is_file():
        raise ContactSheetError(f"{context} image does not exist: {path}")
    return path


def _validate_png(path: Path, context: str) -> None:
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
        raise ContactSheetError(f"cannot decode {context} image {path}: {error}") from error
    if image_format != "PNG":
        raise ContactSheetError(f"{context} image must be PNG, got {image_format!r}: {path}")
    if width <= 0 or height <= 0:
        raise ContactSheetError(f"{context} image has invalid size {width}x{height}: {path}")
    if mode != "RGB":
        raise ContactSheetError(f"{context} image must be RGB, got {mode!r}: {path}")


def _load_completed_summary(root: Path) -> tuple[dict[str, Any], float, dict[str, Path]]:
    summary_path = root / "summary.json"
    summary = _load_json_object(summary_path)
    if summary.get("kind") != EXPERIMENT_KIND:
        raise ContactSheetError(
            f"{summary_path} kind must be {EXPERIMENT_KIND!r}, got {summary.get('kind')!r}"
        )
    schema_version = _integer(summary.get("schema_version"), f"{summary_path} schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ContactSheetError(
            f"{summary_path} schema_version must be {SCHEMA_VERSION}, got {schema_version}"
        )
    completed_at = summary.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at.strip():
        raise ContactSheetError(f"{summary_path} has no completed_at timestamp")
    candidate_count = _integer(summary.get("candidate_count"), f"{summary_path} candidate_count")
    if candidate_count != EXPECTED_CANDIDATE_COUNT:
        raise ContactSheetError(
            f"{summary_path} candidate_count must be {EXPECTED_CANDIDATE_COUNT}, "
            f"got {candidate_count}"
        )
    evaluation_count = _integer(
        summary.get("strategy_evaluation_count"),
        f"{summary_path} strategy_evaluation_count",
    )
    if evaluation_count != EXPECTED_STRATEGY_EVALUATION_COUNT:
        raise ContactSheetError(
            f"{summary_path} strategy_evaluation_count must be "
            f"{EXPECTED_STRATEGY_EVALUATION_COUNT}, got {evaluation_count}"
        )
    if summary.get("full_candidate_pool") is not True:
        raise ContactSheetError(f"{summary_path} full_candidate_pool must be true")
    if summary.get("early_termination") is not False:
        raise ContactSheetError(f"{summary_path} early_termination must be false")
    if summary.get("strategies") != list(STRATEGY_KEYS):
        raise ContactSheetError(
            f"{summary_path} strategies must be {list(STRATEGY_KEYS)!r}"
        )
    top_k = _integer(summary.get("top_k"), f"{summary_path} top_k")
    if top_k != TOP_K:
        raise ContactSheetError(f"{summary_path} top_k must be {TOP_K}, got {top_k}")
    threshold = _finite_float(
        summary.get("topiq_nr_threshold_l"), f"{summary_path} topiq_nr_threshold_l"
    )

    references = summary.get("manifests")
    if not isinstance(references, dict) or set(references) != set(STRATEGY_KEYS):
        raise ContactSheetError(
            f"{summary_path} manifests must contain exactly {list(STRATEGY_KEYS)!r}"
        )
    manifest_paths: dict[str, Path] = {}
    for key in STRATEGY_KEYS:
        reference = references[key]
        if not isinstance(reference, dict):
            raise ContactSheetError(f"{summary_path} manifest reference {key} must be an object")
        reference_top_k = _integer(
            reference.get("top_k"), f"{summary_path} manifest reference {key} top_k"
        )
        if reference_top_k != TOP_K:
            raise ContactSheetError(
                f"{summary_path} manifest reference {key} top_k must be {TOP_K}"
            )
        expected_reference = f"strategies/{key}/top10.json"
        if reference.get("path") != expected_reference:
            raise ContactSheetError(
                f"{summary_path} manifest reference {key} must be "
                f"{expected_reference!r}, got {reference.get('path')!r}"
            )
        manifest_paths[key] = root / expected_reference
    return summary, threshold, manifest_paths


def _load_strategy(
    root: Path,
    manifest_path: Path,
    expected_threshold: float,
    key: str,
    stage: str,
    display_name: str,
    accent_color: str,
) -> StrategyTopK:
    manifest = _load_json_object(manifest_path)
    schema_version = _integer(
        manifest.get("schema_version"), f"{manifest_path} schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise ContactSheetError(
            f"{manifest_path} schema_version must be {SCHEMA_VERSION}, got {schema_version}"
        )
    if manifest.get("strategy") != key:
        raise ContactSheetError(
            f"{manifest_path} strategy must be {key!r}, "
            f"got {manifest.get('strategy')!r}"
        )
    top_k = _integer(manifest.get("top_k"), f"{manifest_path} top_k")
    if top_k != TOP_K:
        raise ContactSheetError(
            f"{manifest_path} top_k must be {TOP_K}, got {top_k}"
        )
    threshold = _finite_float(
        manifest.get("score_threshold"), f"{manifest_path} score_threshold"
    )
    if not math.isclose(threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ContactSheetError(
            f"{manifest_path} threshold {threshold} differs from summary {expected_threshold}"
        )
    operator = manifest.get("score_threshold_operator")
    if operator != ">":
        raise ContactSheetError(
            f"{manifest_path} score_threshold_operator must be '>', got {operator!r}"
        )
    candidate_result_count = _integer(
        manifest.get("candidate_result_count"),
        f"{manifest_path} candidate_result_count",
    )
    if not TOP_K <= candidate_result_count <= EXPECTED_CANDIDATE_COUNT:
        raise ContactSheetError(
            f"{manifest_path} candidate_result_count must be in "
            f"[{TOP_K}, {EXPECTED_CANDIDATE_COUNT}], got {candidate_result_count}"
        )
    threshold_pass_count = _integer(
        manifest.get("threshold_pass_count"), f"{manifest_path} threshold_pass_count"
    )
    if not 0 <= threshold_pass_count <= candidate_result_count:
        raise ContactSheetError(f"{manifest_path} threshold_pass_count is out of range")
    fallback_used = manifest.get("fallback_used")
    if not isinstance(fallback_used, bool) or fallback_used != (threshold_pass_count < TOP_K):
        raise ContactSheetError(f"{manifest_path} fallback_used is inconsistent")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != TOP_K:
        actual = len(raw_items) if isinstance(raw_items, list) else "non-list"
        raise ContactSheetError(
            f"{manifest_path} must contain exactly {TOP_K} items, got {actual}"
        )

    items: list[RankedImage] = []
    seen_ranks: set[int] = set()
    for item_index, raw_item in enumerate(raw_items):
        context = f"{manifest_path} item {item_index}"
        if not isinstance(raw_item, dict):
            raise ContactSheetError(f"{context} must be a JSON object")
        rank = _integer(raw_item.get("rank"), f"{context} rank")
        if rank in seen_ranks:
            raise ContactSheetError(f"{manifest_path} has duplicate rank {rank}")
        seen_ranks.add(rank)
        score = _finite_float(
            raw_item.get("topiq_nr_score"), f"{context} topiq_nr_score"
        )
        if raw_item.get("coverage_qualified_at_output") is not True:
            raise ContactSheetError(
                f"{context} is not coverage-qualified at output"
            )
        image_path = _resolve_image_path(root, raw_item.get("image"), context)
        _validate_png(image_path, context)
        items.append(
            RankedImage(
                rank=rank,
                score=score,
                passes_threshold=_passes_threshold(score, operator, threshold),
                image_path=image_path,
            )
        )

    expected_ranks = set(range(1, TOP_K + 1))
    if seen_ranks != expected_ranks:
        raise ContactSheetError(
            f"{manifest_path} ranks must be exactly 1..{TOP_K}"
        )
    items.sort(key=lambda item: item.rank)
    if any(
        current.score > previous.score + 1e-12
        for previous, current in zip(items, items[1:])
    ):
        raise ContactSheetError(
            f"{manifest_path} scores are not non-increasing by rank"
        )
    return StrategyTopK(
        key=key,
        stage=stage,
        display_name=display_name,
        accent_color=accent_color,
        threshold=threshold,
        threshold_operator=operator,
        items=tuple(items),
    )


def _scene_name(root: Path) -> str:
    name = root.name.strip()
    if not name:
        raise ContactSheetError(f"cannot determine scene name from root: {root}")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe_name:
        raise ContactSheetError(f"scene name is not usable as a directory: {name!r}")
    return safe_name


def _load_scene(root_argument: str) -> SceneArtifacts:
    root = Path(root_argument).expanduser().resolve()
    if not root.is_dir():
        raise ContactSheetError(f"experiment root is not a directory: {root}")
    _summary, threshold, manifest_paths = _load_completed_summary(root)
    strategies = tuple(
        _load_strategy(
            root,
            manifest_paths[key],
            threshold,
            key,
            stage,
            display_name,
            accent_color,
        )
        for key, stage, display_name, accent_color in STRATEGIES
    )
    legacy, current = strategies
    if legacy.threshold_operator != current.threshold_operator or not math.isclose(
        legacy.threshold, current.threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContactSheetError(
            f"strategy thresholds differ under experiment root {root}: "
            f"{legacy.threshold_operator} {legacy.threshold} vs "
            f"{current.threshold_operator} {current.threshold}"
        )
    image_paths = [item.image_path for strategy in strategies for item in strategy.items]
    if len(image_paths) != 2 * TOP_K or len(set(image_paths)) != 2 * TOP_K:
        raise ContactSheetError(
            f"experiment root {root} must reference exactly {2 * TOP_K} distinct PNGs"
        )
    return SceneArtifacts(
        scene=_scene_name(root),
        root=root,
        strategies=strategies,
    )


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/truetype/liberation2")
        / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"),
    )
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def _right_aligned_x(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    right: int,
) -> int:
    left, _top, measured_right, _bottom = draw.textbbox((0, 0), text, font=font)
    return right - (measured_right - left)


def _paste_fitted_image(sheet: Image.Image, item: RankedImage, x: int, y: int) -> None:
    try:
        with Image.open(item.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail(
                (IMAGE_SIZE - 4, IMAGE_SIZE - 4), Image.Resampling.LANCZOS
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ContactSheetError(f"cannot decode image {item.image_path}: {error}") from error
    image_x = x + (IMAGE_SIZE - image.width) // 2
    image_y = y + (IMAGE_SIZE - image.height) // 2
    sheet.paste(image, (image_x, image_y))


def _draw_card(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    item: RankedImage,
    strategy: StrategyTopK,
    x: int,
    y: int,
    score_font: ImageFont.ImageFont,
    status_font: ImageFont.ImageFont,
) -> None:
    draw.rectangle(
        (x, y, x + IMAGE_SIZE - 1, y + CARD_HEIGHT - 1),
        fill=LABEL_BACKGROUND,
        outline=BORDER_COLOR,
        width=2,
    )
    draw.rectangle(
        (x + 2, y + 2, x + IMAGE_SIZE - 3, y + IMAGE_SIZE - 3),
        fill=IMAGE_BACKGROUND,
    )
    _paste_fitted_image(sheet, item, x, y)
    label_y = y + IMAGE_SIZE
    draw.rectangle(
        (x + 2, label_y, x + IMAGE_SIZE - 3, y + CARD_HEIGHT - 3),
        fill=LABEL_BACKGROUND,
    )
    draw.rectangle((x + 2, label_y, x + 8, y + CARD_HEIGHT - 3), fill=strategy.accent_color)
    draw.text(
        (x + 20, label_y + 8),
        f"#{item.rank:02d}   score {item.score:.5f}",
        fill=TEXT,
        font=score_font,
    )
    status = "PASS" if item.passes_threshold else "BELOW"
    status_color = PASS_COLOR if item.passes_threshold else BELOW_COLOR
    draw.text(
        (x + 20, label_y + 42), status, fill=status_color, font=status_font
    )
    threshold_text = f"{strategy.threshold_operator} {strategy.threshold:.5f}"
    threshold_x = _right_aligned_x(
        draw, threshold_text, status_font, x + IMAGE_SIZE - 18
    )
    draw.text(
        (threshold_x, label_y + 42),
        threshold_text,
        fill=MUTED_TEXT,
        font=status_font,
    )


def _draw_strategy_section(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    strategy: StrategyTopK,
    top: int,
    section_width: int,
    section_font: ImageFont.ImageFont,
    section_meta_font: ImageFont.ImageFont,
    score_font: ImageFont.ImageFont,
    status_font: ImageFont.ImageFont,
) -> None:
    grid_height = 2 * CARD_HEIGHT + GRID_GAP
    draw.rectangle(
        (
            MARGIN,
            top,
            MARGIN + section_width - 1,
            top + SECTION_HEADER_HEIGHT + grid_height - 1,
        ),
        fill=PANEL_BACKGROUND,
    )
    draw.rectangle(
        (MARGIN, top, MARGIN + 9, top + SECTION_HEADER_HEIGHT - 1),
        fill=strategy.accent_color,
    )
    draw.text(
        (MARGIN + 26, top + 16),
        f"{strategy.stage}  |  {strategy.display_name}",
        fill=TEXT,
        font=section_font,
    )
    pass_count = sum(item.passes_threshold for item in strategy.items)
    pass_text = f"Top-10 threshold pass: {pass_count}/{TOP_K}"
    pass_x = _right_aligned_x(
        draw, pass_text, section_meta_font, MARGIN + section_width - 22
    )
    draw.text(
        (pass_x, top + 23), fill=MUTED_TEXT, text=pass_text, font=section_meta_font
    )

    grid_top = top + SECTION_HEADER_HEIGHT
    for index, item in enumerate(strategy.items):
        row, column = divmod(index, GRID_COLUMNS)
        x = MARGIN + column * (IMAGE_SIZE + GRID_GAP)
        y = grid_top + row * (CARD_HEIGHT + GRID_GAP)
        _draw_card(
            sheet,
            draw,
            item,
            strategy,
            x,
            y,
            score_font,
            status_font,
        )


def _render_scene(scene: SceneArtifacts) -> Image.Image:
    section_width = GRID_COLUMNS * IMAGE_SIZE + (GRID_COLUMNS - 1) * GRID_GAP
    grid_height = 2 * CARD_HEIGHT + GRID_GAP
    section_height = SECTION_HEADER_HEIGHT + grid_height
    canvas_width = section_width + 2 * MARGIN
    canvas_height = (
        MARGIN
        + TITLE_HEIGHT
        + 2 * section_height
        + SECTION_GAP
        + MARGIN
    )
    sheet = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(42, bold=True)
    subtitle_font = _load_font(23)
    section_font = _load_font(30, bold=True)
    section_meta_font = _load_font(22, bold=True)
    score_font = _load_font(24, bold=True)
    status_font = _load_font(21, bold=True)

    legacy = scene.strategies[0]
    draw.text(
        (MARGIN, MARGIN),
        f"{scene.scene} | Object coverage Top 10",
        fill=TEXT,
        font=title_font,
    )
    draw.text(
        (MARGIN, MARGIN + 57),
        "Before vs after  |  TOPIQ-NR threshold: "
        f"score {legacy.threshold_operator} {legacy.threshold:.5f}",
        fill=MUTED_TEXT,
        font=subtitle_font,
    )

    first_top = MARGIN + TITLE_HEIGHT
    _draw_strategy_section(
        sheet,
        draw,
        scene.strategies[0],
        first_top,
        section_width,
        section_font,
        section_meta_font,
        score_font,
        status_font,
    )
    _draw_strategy_section(
        sheet,
        draw,
        scene.strategies[1],
        first_top + section_height + SECTION_GAP,
        section_width,
        section_font,
        section_meta_font,
        score_font,
        status_font,
    )
    return sheet


def _atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            image.save(handle, format="PNG", optimize=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_contact_sheet(scene: SceneArtifacts, output_path: Path) -> None:
    image = _render_scene(scene)
    try:
        _atomic_save_png(image, output_path)
    finally:
        image.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one legacy/current Top-10 contact sheet for each object "
            "coverage ablation output root."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        metavar="OUTPUT_ROOT",
        help="completed per-scene object coverage ablation output root",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "report directory; each image is written to "
            "<output-dir>/<scene>/before_after_top10.png"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    scenes = tuple(_load_scene(argument) for argument in args.roots)
    names = [scene.scene for scene in scenes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ContactSheetError(
            "scene output directory names collide: " + ", ".join(duplicates)
        )

    for scene in scenes:
        output_path = output_dir / scene.scene / OUTPUT_FILENAME
        build_contact_sheet(scene, output_path)
        print(f"Wrote {scene.scene}: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContactSheetError, OSError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        raise SystemExit(2)
