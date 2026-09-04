#!/usr/bin/env python3
"""Score every second-level image and print the top-scoring combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyiqa
import torch


DEFAULT_ROOT = Path(
    "/home/wdj/projects/data-generate/outputs/apartment_traversal/apartment"
)
DEFAULT_OUTPUT_JSON = Path("topiq_nr_scores.json")
DEFAULT_TOP_K = 10
METRIC_NAME = "topiq_nr"


def score_combinations(root: Path, device: str) -> list[dict[str, object]]:
    """Calculate TOPIQ-NR for each root/position/lens/image.png."""
    metric = pyiqa.create_metric(METRIC_NAME, device=device)
    results: list[dict[str, object]] = []

    for combination_dir in sorted(root.glob("*/*")):
        if not combination_dir.is_dir():
            continue

        image_path = combination_dir / "image.png"
        if not image_path.is_file():
            print(f"Warning: missing {image_path}", file=sys.stderr)
            continue

        try:
            with torch.inference_mode():
                score = float(metric(str(image_path)).item())
        except Exception as error:
            print(f"Warning: skipped {image_path}: {error}", file=sys.stderr)
            continue

        results.append(
            {
                "score": score,
                "combination_path": str(combination_dir.resolve()),
                "image_path": str(image_path.resolve()),
                "camera_json_path": str((combination_dir / "camera.json").resolve()),
            }
        )

    # Use the path as a stable tie-breaker when two scores are equal.
    results.sort(key=lambda item: (-float(item["score"]), item["combination_path"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all second-level combinations and print the top results."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"input root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"score output file (default: {DEFAULT_OUTPUT_JSON})",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"input directory does not exist: {args.root}")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    results = score_combinations(args.root.resolve(), args.device)
    if not results:
        print("No images were scored successfully.", file=sys.stderr)
        raise SystemExit(1)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    for rank, result in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank:2d}. score={result['score']:.8f}\n"
            f"    combination: {result['combination_path']}\n"
            f"    image:       {result['image_path']}\n"
            f"    json:        {result['camera_json_path']}"
        )

    print(
        f"\nAll {len(results)} scores saved to: {args.output_json.resolve()}"
    )


if __name__ == "__main__":
    main()
