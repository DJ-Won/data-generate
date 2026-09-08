data_path="/data0/wdj/datasets/dl3dv-gs/3DGS/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"
output_path="/data0/wdj/zooming/data-generate/outputs/dl3dv_camera_renders"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-/home/wdj/miniconda3/envs/gs/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "${python_bin}" "${project_root}/render_dataset_cameras.py" \
  "${data_path}" "${output_path}" "$@"
