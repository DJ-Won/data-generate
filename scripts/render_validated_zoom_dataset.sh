#!/usr/bin/env bash
set -euo pipefail

data_path="/data0/wdj/datasets/dl3dv-gs/3DGS/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
color_config="/data0/wdj/zooming/data-generate/configs/zoom_video/colors/multicamera_low.yaml"
output_path="/data0/wdj/zooming/data-generate/outputs/dl3dv_validated"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

exec python "${project_root}/render_validated_zoom_dataset.py" \
  "${data_path}" \
  "${color_config}" \
  "${output_path}"
