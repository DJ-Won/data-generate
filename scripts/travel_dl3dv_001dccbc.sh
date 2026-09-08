#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${PYTHON:-python}" generate_scene_traversal.py \
  --scene-config configs/travel/scenes/dl3dv_001dccbc.yaml \
  --camera-config configs/travel/cameras/object.yaml \
  --color-config configs/travel/colors/default.yaml \
  "$@"
