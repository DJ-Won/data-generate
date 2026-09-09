#!/usr/bin/env bash
set -uo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 START_INDEX END_INDEX [GPU_ID] [--remove-cache]" >&2
    exit 2
fi

start_index="$1"
end_index="$2"
shift 2
gpu_id=0
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    gpu_id="$1"
    shift
fi
remove_cache=false
while (( $# > 0 )); do
    case "$1" in
        --remove-cache)
            remove_cache=true
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! "${start_index}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${end_index}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "START_INDEX and END_INDEX must be positive integers; GPU_ID must be non-negative." >&2
    exit 2
fi
if (( end_index < start_index )); then
    echo "END_INDEX must be greater than or equal to START_INDEX." >&2
    exit 2
fi

range_size=$((end_index - start_index + 1))
if (( range_size != 50 )); then
    echo "This runner requires exactly 50 scenes; requested ${range_size}." >&2
    exit 2
fi

dataset_root="${DATASET_ROOT:-/data0/wdj/datasets/dl3dv-gs/3DGS/1K}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
runner="${project_root}/scripts/render_validated_zoom_dataset.sh"
output_root="${project_root}/outputs/dl3dv_validated_far_cam"
render_python="${RENDER_PYTHON:-/home/wdj/miniconda3/envs/dg/bin/python}"
state_dir="${output_root}/_batch_state/range_${start_index}_${end_index}"
log_dir="${state_dir}/logs"

if [[ ! -d "${dataset_root}" ]]; then
    echo "Dataset root does not exist: ${dataset_root}" >&2
    exit 2
fi
if [[ ! -f "${runner}" ]]; then
    echo "Render script does not exist: ${runner}" >&2
    exit 2
fi
if [[ ! -x "${render_python}" ]]; then
    echo "Render Python does not exist or is not executable: ${render_python}" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate pipeline summaries." >&2
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to prevent duplicate range runners." >&2
    exit 2
fi

export PATH="$(dirname -- "${render_python}"):${PATH}"
mkdir -p "${log_dir}"
exec 9>"${state_dir}/run.lock"
if ! flock -n 9; then
    echo "Another runner already owns scene range ${start_index}-${end_index}." >&2
    exit 3
fi

mapfile -d '' -t all_scenes < <(
    find "${dataset_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
if (( ${#all_scenes[@]} < end_index )); then
    echo "Dataset has only ${#all_scenes[@]} scene directories; index ${end_index} is unavailable." >&2
    exit 2
fi
offset=$((start_index - 1))
scenes=("${all_scenes[@]:offset:range_size}")
if (( ${#scenes[@]} != 50 )); then
    echo "Internal range error: expected 50 scenes, selected ${#scenes[@]}." >&2
    exit 2
fi

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

scene_is_complete() {
    local scene_name="$1"
    local summary="${output_root}/${scene_name}/pipeline_summary.json"

    [[ -f "${summary}" ]] || return 1
    jq -e '
        .status == "completed"
        and (.camera_count | type == "number")
        and (.results | type == "array")
        and (.camera_count == (.results | length))
        and ((.counts.render_error // 0) == 0)
        and ((.counts.evaluation_error // 0) == 0)
    ' "${summary}" >/dev/null 2>&1
}

renderer_arguments=()
if [[ "${remove_cache}" == true ]]; then
    renderer_arguments+=(--remove-cache)
fi

run_id="$(date '+%Y%m%d_%H%M%S')-$$"
exec > >(tee -a "${state_dir}/batch.log") 2>&1
printf '[%s] run=%s range=%d-%d gpu=%s remove_cache=%s scenes=%d\n' \
    "$(timestamp)" "${run_id}" "${start_index}" "${end_index}" "${gpu_id}" \
    "${remove_cache}" "${#scenes[@]}"

completed=0
skipped=0
failed=0
for scene_offset in "${!scenes[@]}"; do
    scene_path="${scenes[${scene_offset}]}"
    scene_name="${scene_path##*/}"
    scene_number=$((start_index + scene_offset))
    scene_log="${log_dir}/${scene_name}.log"

    if scene_is_complete "${scene_name}"; then
        ((skipped += 1))
        printf '[%s] [%d/%d] skip complete scene #%d: %s\n' \
            "$(timestamp)" "$((scene_offset + 1))" "${#scenes[@]}" \
            "${scene_number}" "${scene_name}"
        continue
    fi

    printf '\n[%s] [%d/%d] start scene #%d on gpu=%s: %s\n' \
        "$(timestamp)" "$((scene_offset + 1))" "${#scenes[@]}" \
        "${scene_number}" "${gpu_id}" "${scene_name}" | tee -a "${scene_log}"

    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu_id}" \
        bash "${runner}" "${scene_path}" "${renderer_arguments[@]}" \
        2>&1 | tee -a "${scene_log}"
    renderer_status=${PIPESTATUS[0]}

    if (( renderer_status == 0 )) && scene_is_complete "${scene_name}"; then
        ((completed += 1))
        result="complete"
    else
        ((failed += 1))
        result="failed(renderer_status=${renderer_status})"
    fi
    printf '[%s] [%d/%d] %s scene #%d: %s\n' \
        "$(timestamp)" "$((scene_offset + 1))" "${#scenes[@]}" "${result}" \
        "${scene_number}" "${scene_name}" | tee -a "${scene_log}"
done

printf '[%s] run=%s finished range=%d-%d completed=%d skipped=%d failed=%d\n' \
    "$(timestamp)" "${run_id}" "${start_index}" "${end_index}" \
    "${completed}" "${skipped}" "${failed}"
(( failed == 0 ))
