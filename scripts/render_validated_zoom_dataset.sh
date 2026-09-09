#!/usr/bin/env bash
set -euo pipefail

data_path="${1:-/data0/wdj/datasets/dl3dv-gs/3DGS/1K/0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac}"
if (( $# > 0 )); then
  shift
fi
camera_config="/data0/wdj/zooming/data-generate/configs/zoom_video/cameras/x3.yaml"
color_config="/data0/wdj/zooming/data-generate/configs/zoom_video/colors/multicamera_low.yaml"
output_path="/data0/wdj/zooming/data-generate/outputs/dl3dv_validated_far_cam"
render_python="${RENDER_PYTHON:-python}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

exec "${render_python}" "${project_root}/render_validated_zoom_dataset.py" \
  "${data_path}" \
  "${camera_config}" \
  "${color_config}" \
  "${output_path}" \
  "$@"
