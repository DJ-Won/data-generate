#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from scene_traversal import (
    load_traversal_config,
    load_traversal_config_parts,
    run_traversal,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a random interior or object traversal dataset from a 3DGS PLY scene"
    )
    parser.add_argument(
        "--config",
        help="legacy combined traversal YAML; cannot be mixed with split configs",
    )
    parser.add_argument("--scene-config", help="scene/input/output YAML fragment")
    parser.add_argument(
        "--camera-config", help="camera/intrinsics/traversal YAML fragment"
    )
    parser.add_argument("--color-config", help="render/color encoding YAML fragment")
    parser.add_argument(
        "--validate-only", action="store_true", help="validate configuration and exit"
    )
    args = parser.parse_args()
    try:
        split_paths = (args.scene_config, args.camera_config, args.color_config)
        if args.config is not None:
            if any(path is not None for path in split_paths):
                raise ValueError("--config cannot be combined with split config arguments")
            cfg = load_traversal_config(
                args.config, check_output=not args.validate_only
            )
        else:
            split_names = ("--scene-config", "--camera-config", "--color-config")
            missing = [
                name for name, path in zip(split_names, split_paths) if path is None
            ]
            if missing:
                raise ValueError(
                    "provide --config or all three split configs; missing "
                    + ", ".join(missing)
                )
            cfg = load_traversal_config_parts(
                *split_paths, check_output=not args.validate_only
            )
        if args.validate_only:
            target = cfg.output.root_directory / cfg.output.scene_name
            sampling = cfg.initialization.traversal
            if sampling.strategy == "random":
                quality = sampling.random
                print(
                    f"Configuration is valid: random strategy targets {quality.target_count_k} "
                    f"captures with topiq_nr > {quality.topiq_nr_threshold_l}, using all "
                    f"valid positions from {sampling.max_position_sampling_attempts} "
                    f"position attempts x {sampling.images_per_position_l} images -> {target}"
                )
                return 0
            count = (
                sampling.position_count_k
                * sampling.images_per_position_l
            )
            print(
                f"Configuration is valid: {sampling.position_count_k} "
                f"positions x {sampling.images_per_position_l} images "
                f"= {count} image/JSON pairs -> {target}"
            )
            return 0
        run_traversal(cfg)
        return 0
    except (ValidationError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
