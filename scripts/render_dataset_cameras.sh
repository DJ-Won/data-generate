data_path="/data0/wdj/datasets/dl3dv-gs/3DGS/1K/0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac"
output_path="/data0/wdj/zooming/data-generate/outputs/dl3dv_032"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-/home/wdj/miniconda3/envs/gs/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "${python_bin}" "${project_root}/render_dataset_cameras.py" \
  "${data_path}" "${output_path}" "$@"
