#!/usr/bin/env python3
"""Select the most visible lens per position, then rank its rendered image."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pyiqa
import torch


DEFAULT_ROOT = Path(
    "/home/wdj/projects/data-generate/outputs/garden_traversal/garden"
)
DEFAULT_OUTPUT_JSON = Path("best_lens_per_position_scores.json")
MAX_MAJOR_AXIS_RATIO = 10.0
METRIC_NAME = "topiq_nr"


def read_lens_record(lens_dir: Path) -> dict[str, Any] | None:
    """Read and validate the geometry fields needed to select a lens."""
    camera_json_path = lens_dir / "camera.json"
    image_path = lens_dir / "image.png"

    if not camera_json_path.is_file():
        print(f"Warning: missing {camera_json_path}", file=sys.stderr)
        return None
    if not image_path.is_file():
        print(f"Warning: missing {image_path}", file=sys.stderr)
        return None

    try:
        with camera_json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        geometry_quality = data["geometry_quality"]
        visible_count = geometry_quality["visible_gaussian_count"]
        major_axis_ratio = geometry_quality[
            "max_projected_major_axis_to_image_diagonal_ratio"
        ]

        if isinstance(visible_count, bool) or not isinstance(
            visible_count, (int, float)
        ):
            raise TypeError("visible_gaussian_count is not numeric")
        if isinstance(major_axis_ratio, bool) or not isinstance(
            major_axis_ratio, (int, float)
        ):
            raise TypeError(
                "max_projected_major_axis_to_image_diagonal_ratio is not numeric"
            )
        if not math.isfinite(float(visible_count)) or not math.isfinite(
            float(major_axis_ratio)
        ):
            raise ValueError("geometry metric is not finite")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"Warning: skipped {camera_json_path}: {error}", file=sys.stderr)
        return None

    if float(major_axis_ratio) > MAX_MAJOR_AXIS_RATIO:
        return None

    return {
        "position": lens_dir.parent.name,
        "lens": lens_dir.name,
        "visible_gaussian_count": int(visible_count),
        "max_projected_major_axis_to_image_diagonal_ratio": float(
            major_axis_ratio
        ),
        "combination_path": str(lens_dir.resolve()),
        "image_path": str(image_path.resolve()),
        "camera_json_path": str(camera_json_path.resolve()),
    }


def select_best_lens_per_position(root: Path) -> list[dict[str, Any]]:
    """Select one lens with the largest visible count for every position."""
    selected: list[dict[str, Any]] = []

    for position_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        candidates = [
            record
            for lens_dir in sorted(
                path for path in position_dir.iterdir() if path.is_dir()
            )
            if (record := read_lens_record(lens_dir)) is not None
        ]

        if not candidates:
            print(
                f"Warning: no valid lens found for {position_dir}", file=sys.stderr
            )
            continue

        # min(path) is used as a deterministic tie-breaker for equal counts.
        best = min(
            candidates,
            key=lambda item: (
                -item["visible_gaussian_count"],
                item["combination_path"],
            ),
        )
        selected.append(best)

    return selected


def score_selected_images(
    selected: list[dict[str, Any]], device: str
) -> list[dict[str, Any]]:
    """Run TOPIQ-NR on selected images and sort them by descending score."""
    metric = pyiqa.create_metric(METRIC_NAME, device=device)
    scored: list[dict[str, Any]] = []

    for index, record in enumerate(selected, start=1):
        print(
            f"Scoring {index}/{len(selected)}: {record['image_path']}",
            file=sys.stderr,
        )
        try:
            with torch.inference_mode():
                score = float(metric(record["image_path"]).item())
            if not math.isfinite(score):
                raise ValueError("TOPIQ-NR score is not finite")
        except Exception as error:
            print(
                f"Warning: scoring failed for {record['image_path']}: {error}",
                file=sys.stderr,
            )
            continue

        scored.append({**record, "score": score})

    scored.sort(key=lambda item: (-item["score"], item["combination_path"]))
    for rank, record in enumerate(scored, start=1):
        record["rank"] = rank
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For every position, select the lens with the largest visible "
            "Gaussian count, score its image, and rank all selected paths."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"input root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"ranked result file (default: {DEFAULT_OUTPUT_JSON})",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"input directory does not exist: {args.root}")

    selected = select_best_lens_per_position(args.root.resolve())
    if not selected:
        print("No valid position/lens combinations found.", file=sys.stderr)
        raise SystemExit(1)

    results = score_selected_images(selected, args.device)
    if not results:
        print("No selected images were scored successfully.", file=sys.stderr)
        raise SystemExit(1)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    for result in results:
        print(
            f"{result['rank']:2d}. score={result['score']:.8f}, "
            f"visible_gaussian_count={result['visible_gaussian_count']}\n"
            f"    position:    {result['position']}\n"
            f"    lens:        {result['lens']}\n"
            f"    combination: {result['combination_path']}\n"
            f"    image:       {result['image_path']}\n"
            f"    json:        {result['camera_json_path']}"
        )

    print(f"\nRanked results saved to: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
