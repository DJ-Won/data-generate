#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_dl3dv_validated_scene_range.sh" \
    1 50 "${GPU_ID:-0}" "$@"
