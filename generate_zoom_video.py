#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from zoomgen.alignment import AlignmentSelectionRequired
from zoomgen.config import load_config_parts
from zoomgen.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a multi-camera smooth zoom video from a 3DGS PLY"
    )
    parser.add_argument("--scene-config", required=True, help="scene/runtime YAML fragment")
    parser.add_argument(
        "--camera-config", required=True, help="camera/zoom geometry YAML fragment"
    )
    parser.add_argument(
        "--color-config", required=True, help="render color/lens appearance YAML fragment"
    )
    parser.add_argument(
        "--camera-json",
        help=(
            "optional traversal camera.json; overrides the initial pose and supplies "
            "the intrinsics at the camera config's minimum zoom"
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="validate configuration and exit"
    )
    args = parser.parse_args()
    try:
        cfg = load_config_parts(
            args.scene_config,
            args.camera_config,
            args.color_config,
            camera_json_path=args.camera_json,
            check_output=not args.validate_only,
        )
        if args.validate_only:
            print("Configuration is valid.")
            return 0
        run(cfg)
        return 0
    except AlignmentSelectionRequired as exc:
        print(f"alignment selection required: {exc}", file=sys.stderr)
        print(f"report: {exc.report_path}", file=sys.stderr)
        return 3
    except (ValidationError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
