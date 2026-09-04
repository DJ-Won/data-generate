#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

exec "${PYTHON:-python}" generate_scene_traversal.py \
  --scene-config configs/travel/scenes/canyon_random.yaml \
  --camera-config configs/travel/cameras/random.yaml \
  --color-config configs/travel/colors/default.yaml \
  "$@"
