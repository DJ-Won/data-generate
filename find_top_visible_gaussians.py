#!/usr/bin/env python3
"""Find second-level folders with the highest visible Gaussian counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Exclude combinations whose projected-major-axis ratio is greater than k.
k = 10

DEFAULT_ROOT = Path(
    "/home/wdj/projects/data-generate/outputs/garden_traversal/garden"
)


def find_top_combinations(root: Path, top_k: int) -> list[tuple[int, Path]]:
    """Return (visible_gaussian_count, camera.json path) sorted descending."""
    records: list[tuple[int, Path]] = []

    # The expected layout is: root/position_xxxx/lens_xxxx/camera.json
    for combination_dir in sorted(root.glob("*/*")):
        if not combination_dir.is_dir():
            continue

        json_path = combination_dir / "camera.json"
        if not json_path.is_file():
            print(f"Warning: missing {json_path}", file=sys.stderr)
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            geometry_quality = data["geometry_quality"]
            count = geometry_quality["visible_gaussian_count"]
            major_axis_ratio = geometry_quality[
                "max_projected_major_axis_to_image_diagonal_ratio"
            ]
            if isinstance(count, bool) or not isinstance(count, (int, float)):
                raise TypeError("visible_gaussian_count is not numeric")
            if isinstance(major_axis_ratio, bool) or not isinstance(
                major_axis_ratio, (int, float)
            ):
                raise TypeError(
                    "max_projected_major_axis_to_image_diagonal_ratio "
                    "is not numeric"
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            print(f"Warning: skipped {json_path}: {error}", file=sys.stderr)
            continue

        if major_axis_ratio > k:
            continue

        records.append((int(count), json_path))

    # The path provides deterministic ordering when counts are equal.
    records.sort(key=lambda item: (-item[0], str(item[1])))
    return records[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the second-level combinations whose camera.json files have "
            "the highest geometry_quality.visible_gaussian_count values."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"input root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10)
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"input directory does not exist: {args.root}")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    results = find_top_combinations(args.root.resolve(), args.top_k)
    if not results:
        print("No valid camera.json files found.", file=sys.stderr)
        raise SystemExit(1)

    for rank, (count, json_path) in enumerate(results, start=1):
        print(
            f"{rank:2d}. visible_gaussian_count={count}\n"
            f"    combination: {json_path.parent}\n"
            f"    json:        {json_path}"
        )


if __name__ == "__main__":
    main()
