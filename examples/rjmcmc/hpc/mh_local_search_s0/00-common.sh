#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

readonly HARNESS_ACCOUNT="chem007981"
readonly HARNESS_GIT_MODULE="git/2.45.1-pqk5"
export PIXI_EXE="${PIXI_EXE:-/user/work/bm13805/.pixi/bin/pixi}"

require_variable() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "required variable is unset: ${name}" >&2
        exit 2
    fi
}

load_harness_git() {
    module load "${HARNESS_GIT_MODULE}"
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

guard_repository() {
    require_variable REPO_ROOT
    require_variable FULL_SHA
    require_variable PIXI_LOCK_SHA256
    [[ "${PIXI_EXE}" = /* && -x "${PIXI_EXE}" ]]
    load_harness_git
    [[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${FULL_SHA}" ]]
    [[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]]
    [[ "$(sha256_file "${REPO_ROOT}/pixi.lock")" == "${PIXI_LOCK_SHA256}" ]]
}

guard_harness() {
    require_variable HARNESS_DIR
    require_variable HARNESS_SHA256
    pixi_python "${HARNESS_DIR}/24-final-audit.py" verify-harness \
        --harness-directory "${HARNESS_DIR}" \
        --expected-harness-sha256 "${HARNESS_SHA256}" \
        --expected-source-revision "${FULL_SHA}" \
        --expected-pixi-lock-sha256 "${PIXI_LOCK_SHA256}"
}

pixi_python() {
    (
        cd "${REPO_ROOT}"
        "${PIXI_EXE}" run --as-is -e dev python "$@"
    )
}

pixi_python_at() {
    local working_directory="$1"
    shift
    (
        cd "${working_directory}"
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PIXI_EXE}" run --manifest-path "${REPO_ROOT}/pyproject.toml" \
            --as-is -e dev python "$@"
    )
}

cell_row() {
    local task_id="$1"
    awk -F '\t' -v task="${task_id}" 'NR > 1 && $1 == task { print; found=1 } END { if (!found) exit 3 }' \
        "${HARNESS_DIR}/cell-map.tsv"
}

reference_row() {
    local task_id="$1"
    awk -F '\t' -v task="${task_id}" 'NR > 1 && $1 == task { print; found=1 } END { if (!found) exit 3 }' \
        "${HARNESS_DIR}/reference-map.tsv"
}

job_status_path() {
    local stage="$1"
    local state="$2"
    local task="${SLURM_ARRAY_TASK_ID:-single}"
    printf '%s/status/jobs/%s/%s.%s.json' "${RUN_ROOT}" "${stage}" "${task}" "${state}"
}

write_job_status() {
    local stage="$1"
    local state="$2"
    local artifact="${3:-}"
    local status_path
    local -a command
    status_path="$(job_status_path "${stage}" "${state}")"
    mkdir -p "$(dirname "${status_path}")"
    command=(
        "${HARNESS_DIR}/20-audit-artifact.py" write-status
        --path "${status_path}" \
        --stage "${stage}" \
        --state "${state}" \
        --task-id "${SLURM_ARRAY_TASK_ID:-single}" \
        --job-id "${SLURM_JOB_ID:-not-slurm}"
    )
    if [[ -n "${artifact}" ]]; then
        command+=(--artifact-completion "${artifact}")
    fi
    pixi_python "${command[@]}"
}

harness_job_init() {
    local stage="$1"
    require_variable RUN_ROOT
    guard_repository
    guard_harness
    umask 027
    write_job_status "${stage}" started
    HARNESS_CURRENT_STAGE="${stage}"
    export HARNESS_CURRENT_STAGE
    trap 'harness_job_failed "$?"' ERR
}

harness_job_complete() {
    local stage="$1"
    local artifact="$2"
    pixi_python "${HARNESS_DIR}/20-audit-artifact.py" audit --completion "${artifact}"
    trap - ERR
    write_job_status "${stage}" complete "${artifact}"
}

harness_job_failed() {
    local exit_code="$1"
    trap - ERR
    if [[ -n "${HARNESS_CURRENT_STAGE:-}" ]]; then
        write_job_status "${HARNESS_CURRENT_STAGE}" failed || true
    fi
    exit "${exit_code}"
}
