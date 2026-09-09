#!/usr/bin/env bash
set -uo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
batch_runner="${project_root}/scripts/run_dl3dv_validated_two_gpu.sh"
dataset_root="${DATASET_ROOT:-/data0/wdj/datasets/dl3dv-gs/3DGS/1K}"
output_root="${project_root}/outputs/dl3dv_validated_far_cam"
render_python="${RENDER_PYTHON:-/home/wdj/miniconda3/envs/dg/bin/python}"
poll_seconds="${POLL_SECONDS:-60}"
max_idle_passes="${MAX_IDLE_PASSES:-3}"

if [[ ! -x "${render_python}" ]]; then
    echo "Render Python does not exist or is not executable: ${render_python}" >&2
    exit 2
fi

export PATH="$(dirname -- "${render_python}"):${PATH}"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

download_is_active() {
    pgrep -f '[d]ownload_DL3DV-GS-960P.py.*--subset 1K' >/dev/null 2>&1
}

ready_scene_count() {
    local scene_path
    local count=0
    while IFS= read -r -d '' scene_path; do
        if [[ -f "${scene_path}/cameras.json" || -f "${scene_path}/camera.json" ]] \
            && find "${scene_path}" -type f -name point_cloud.ply -print -quit \
                | grep -q .; then
            ((count += 1))
        fi
    done < <(find "${dataset_root}" -mindepth 1 -maxdepth 1 -type d -print0)
    printf '%d\n' "${count}"
}

complete_scene_count() {
    local summary
    local count=0
    while IFS= read -r -d '' summary; do
        if jq -e '
            .status == "completed"
            and (.camera_count | type == "number")
            and (.results | type == "array")
            and (.camera_count == (.results | length))
            and ((.counts.render_error // 0) == 0)
            and ((.counts.evaluation_error // 0) == 0)
        ' "${summary}" >/dev/null 2>&1; then
            ((count += 1))
        fi
    done < <(find "${output_root}" -mindepth 2 -maxdepth 2 \
        -type f -name pipeline_summary.json -print0)
    printf '%d\n' "${count}"
}

previous_complete=-1
idle_passes=0
while :; do
    printf '[%s] starting/resuming a two-GPU directory scan\n' "$(timestamp)"
    DATASET_ROOT="${dataset_root}" RENDER_PYTHON="${render_python}" \
        bash "${batch_runner}"
    rc=$?

    if (( rc == 3 )); then
        printf '[%s] another batch owns the lock; checking again in %ss\n' \
            "$(timestamp)" "${poll_seconds}"
        sleep "${poll_seconds}"
        continue
    fi

    ready="$(ready_scene_count)"
    complete="$(complete_scene_count)"
    printf '[%s] scan finished rc=%d ready=%d complete=%d\n' \
        "$(timestamp)" "${rc}" "${ready}" "${complete}"

    if download_is_active; then
        printf '[%s] downloader is active; rescanning in %ss\n' \
            "$(timestamp)" "${poll_seconds}"
        previous_complete="${complete}"
        idle_passes=0
        sleep "${poll_seconds}"
        continue
    fi

    if (( complete >= ready )); then
        printf '[%s] all %d ready scene directories are complete\n' \
            "$(timestamp)" "${ready}"
        exit 0
    fi

    if (( complete == previous_complete )); then
        ((idle_passes += 1))
    else
        idle_passes=0
    fi
    if (( idle_passes >= max_idle_passes )); then
        printf '[%s] no completion progress for %d passes; leaving failures for inspection\n' \
            "$(timestamp)" "${idle_passes}" >&2
        exit 1
    fi
    previous_complete="${complete}"
done
