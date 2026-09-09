#!/usr/bin/env bash
set -uo pipefail

dataset_root="${DATASET_ROOT:-/data0/wdj/datasets/dl3dv-gs/3DGS/1K}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${project_root}/scripts/render_validated_zoom_dataset.sh"
output_root="${project_root}/outputs/dl3dv_validated_far_cam"
state_dir="${output_root}/_batch_state"

read -r -a gpu_ids <<< "${GPU_IDS:-0 1}"
if (( ${#gpu_ids[@]} != 2 )); then
    echo "GPU_IDS must contain exactly two GPU indices (default: '0 1')." >&2
    exit 2
fi
if [[ ! -d "${dataset_root}" ]]; then
    echo "Dataset root does not exist: ${dataset_root}" >&2
    exit 2
fi
if [[ ! -f "${runner}" ]]; then
    echo "Render script does not exist: ${runner}" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate pipeline summaries." >&2
    exit 2
fi

mkdir -p "${state_dir}/logs"
exec 9>"${state_dir}/batch.lock"
if ! flock -n 9; then
    echo "Another DL3DV validated rendering batch already holds ${state_dir}/batch.lock." >&2
    exit 3
fi

exec > >(tee -a "${state_dir}/batch.log") 2>&1

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

declare -a tasks=()
skipped=0
while IFS= read -r -d '' scene_path; do
    scene_name="${scene_path##*/}"
    if scene_is_complete "${scene_name}"; then
        ((skipped += 1))
    else
        tasks+=("${scene_path}")
    fi
done < <(find "${dataset_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

run_id="$(date '+%Y%m%d_%H%M%S')-$$"
printf '%s\n' "${run_id}" > "${state_dir}/current_run"
printf '[%s] run=%s scenes=%d skipped_complete=%d pending=%d gpus=%s\n' \
    "$(timestamp)" "${run_id}" "$((skipped + ${#tasks[@]}))" "${skipped}" \
    "${#tasks[@]}" "${gpu_ids[*]}"

if (( ${#tasks[@]} == 0 )); then
    printf '[%s] all scenes are already complete\n' "$(timestamp)"
    printf '%s completed\n' "${run_id}" > "${state_dir}/current_run"
    exit 0
fi

declare -A pid_gpu=()
declare -A pid_scene=()
next_task=0
running=0
succeeded=0
failed=0

launch_task() {
    local gpu="$1"
    local scene_path="$2"
    local scene_name="${scene_path##*/}"
    local scene_log="${state_dir}/logs/${scene_name}.log"

    (
        if scene_is_complete "${scene_name}"; then
            exit 0
        fi
        {
            printf '\n[%s] run=%s gpu=%s start scene=%s\n' \
                "$(timestamp)" "${run_id}" "${gpu}" "${scene_name}"
            PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" \
                bash "${runner}" "${scene_path}"
            rc=$?
            printf '[%s] run=%s gpu=%s renderer_exit=%d scene=%s\n' \
                "$(timestamp)" "${run_id}" "${gpu}" "${rc}" "${scene_name}"
            if (( rc != 0 )); then
                exit "${rc}"
            fi
            if ! scene_is_complete "${scene_name}"; then
                echo "Renderer exited successfully, but the pipeline summary is incomplete." >&2
                exit 98
            fi
        } >> "${scene_log}" 2>&1
    ) &

    local pid=$!
    pid_gpu["${pid}"]="${gpu}"
    pid_scene["${pid}"]="${scene_name}"
    ((running += 1))
    ((next_task += 1))
    printf '[%s] gpu=%s pid=%d assigned=%d/%d scene=%s\n' \
        "$(timestamp)" "${gpu}" "${pid}" "${next_task}" "${#tasks[@]}" "${scene_name}"
}

stop_workers() {
    trap - INT TERM HUP
    printf '[%s] stopping %d active workers\n' "$(timestamp)" "${running}"
    local pid
    for pid in "${!pid_gpu[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait || true
    printf '%s interrupted\n' "${run_id}" > "${state_dir}/current_run"
    exit 130
}
trap stop_workers INT TERM HUP

for gpu in "${gpu_ids[@]}"; do
    if (( next_task < ${#tasks[@]} )); then
        launch_task "${gpu}" "${tasks[${next_task}]}"
    fi
done

while (( running > 0 )); do
    finished_pid=''
    if wait -n -p finished_pid; then
        rc=0
    else
        rc=$?
    fi

    gpu="${pid_gpu[${finished_pid}]}"
    scene_name="${pid_scene[${finished_pid}]}"
    unset 'pid_gpu['"${finished_pid}"']'
    unset 'pid_scene['"${finished_pid}"']'
    ((running -= 1))

    if (( rc == 0 )); then
        ((succeeded += 1))
        result='complete'
    else
        ((failed += 1))
        result="failed(rc=${rc})"
    fi
    printf '[%s] gpu=%s pid=%s result=%s progress=%d/%d failed=%d scene=%s\n' \
        "$(timestamp)" "${gpu}" "${finished_pid}" "${result}" \
        "$((succeeded + failed))" "${#tasks[@]}" "${failed}" "${scene_name}"

    if (( next_task < ${#tasks[@]} )); then
        launch_task "${gpu}" "${tasks[${next_task}]}"
    fi
done

printf '[%s] run=%s finished complete=%d failed=%d skipped_before_start=%d\n' \
    "$(timestamp)" "${run_id}" "${succeeded}" "${failed}" "${skipped}"
if (( failed == 0 )); then
    printf '%s completed\n' "${run_id}" > "${state_dir}/current_run"
    exit 0
fi
printf '%s completed_with_failures=%d\n' "${run_id}" "${failed}" > "${state_dir}/current_run"
exit 1
