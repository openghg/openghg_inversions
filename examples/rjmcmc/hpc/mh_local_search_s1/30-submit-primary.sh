#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_PIXI_EXE="/user/work/bm13805/.pixi/bin/pixi"

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_clean_detached_repo() {
    local repo="$1"
    module load git/2.45.1-pqk5
    [[ -z "$(git -C "${repo}" status --porcelain)" ]]
    ! git -C "${repo}" symbolic-ref -q HEAD >/dev/null
    git -C "${repo}" rev-parse HEAD | grep -Eq '^[0-9a-f]{40}$'
}

repo_pixi() {
    (
        cd "${REPO_ROOT}"
        "${PIXI_EXE}" run --as-is -e dev "$@"
    )
}

stage_harness() {
    local base="$1"
    local repo="$2"
    local pixi_exe="${PIXI_EXE:-${DEFAULT_PIXI_EXE}}"
    [[ "${base}" = /* && "${repo}" = /* && "${pixi_exe}" = /* && -x "${pixi_exe}" ]]
    require_clean_detached_repo "${repo}"
    local full_sha lock_sha harness_sha run_parent run_root
    full_sha="$(git -C "${repo}" rev-parse HEAD)"
    lock_sha="$(sha256_file "${repo}/pixi.lock")"
    harness_sha="$(sha256_file "${SOURCE_HARNESS_DIR}/files.sha256")"
    run_parent="${base}/${full_sha}"
    run_root="${run_parent}/harness-${harness_sha}"
    mkdir -p "${run_parent}"
    mkdir "${run_root}"
    (
        cd "${repo}"
        PIXI_EXE="${pixi_exe}" "${pixi_exe}" run --as-is -e dev python \
            "${SOURCE_HARNESS_DIR}/24-final-audit.py" freeze \
            --source "${SOURCE_HARNESS_DIR}" \
            --destination "${run_root}/harness" \
            --source-revision "${full_sha}" \
            --pixi-lock-sha256 "${lock_sha}" \
            --expected-harness-sha256 "${harness_sha}"
    )
    mkdir "${run_root}/status" "${run_root}/jobs"
    (
        cd "${repo}"
        "${pixi_exe}" run --as-is -e dev python \
            "${run_root}/harness/00-check-s0-prerequisite.py" \
            --output "${run_root}/status/s0-prerequisite.complete.json"
    )
    printf '%s\n' "${run_root}"
}

load_run() {
    local run_root="$1"
    local repo="$2"
    local pixi_exe="${PIXI_EXE:-${DEFAULT_PIXI_EXE}}"
    [[ "${run_root}" = /* && "${repo}" = /* && "${pixi_exe}" = /* && -x "${pixi_exe}" ]]
    require_clean_detached_repo "${repo}"
    export RUN_ROOT="${run_root}"
    export REPO_ROOT="${repo}"
    export HARNESS_DIR="${run_root}/harness"
    export PIXI_EXE="${pixi_exe}"
    read -r FULL_SHA PIXI_LOCK_SHA256 HARNESS_SHA256 < <(
        repo_pixi python -c \
            'import json,sys; p=json.load(open(sys.argv[1])); print(p["source_revision"],p["pixi_lock_sha256"],p["harness_sha256"])' \
            "${HARNESS_DIR}/complete.json"
    )
    export FULL_SHA PIXI_LOCK_SHA256 HARNESS_SHA256
    [[ "$(git -C "${repo}" rev-parse HEAD)" == "${FULL_SHA}" ]]
    [[ "$(sha256_file "${repo}/pixi.lock")" == "${PIXI_LOCK_SHA256}" ]]
    repo_pixi python "${HARNESS_DIR}/24-final-audit.py" \
        verify-harness --harness-directory "${HARNESS_DIR}" \
        --expected-harness-sha256 "${HARNESS_SHA256}" \
        --expected-source-revision "${FULL_SHA}" \
        --expected-pixi-lock-sha256 "${PIXI_LOCK_SHA256}"
}

record_submission() {
    local name="$1"
    local job_id="$2"
    repo_pixi python "${HARNESS_DIR}/20-audit-artifact.py" \
        write-status --path "${RUN_ROOT}/status/submissions/${name}.complete.json" \
        --stage "submission-${name}" --state complete --task-id single --job-id "${job_id}"
}

submission_job_id() {
    local name="$1"
    local record="${RUN_ROOT}/status/submissions/${name}.complete.json"
    [[ -f "${record}" ]]
    repo_pixi python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' \
        "${record}"
}

submit_job() {
    local name="$1"
    local script="$2"
    local dependency="${3:-}"
    shift 3 || true
    local submission_record="${RUN_ROOT}/status/submissions/${name}.complete.json"
    if [[ -e "${submission_record}" ]]; then
        echo "refusing duplicate submission before sbatch: ${submission_record}" >&2
        return 20
    fi
    local -a command=(
        sbatch --parsable
        --chdir "${RUN_ROOT}/jobs"
        --output "${RUN_ROOT}/jobs/%x-%A_%a.out"
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE}"
    )
    if [[ -n "${dependency}" ]]; then
        command+=(--dependency "afterok:${dependency}")
    fi
    command+=("$@" "${HARNESS_DIR}/${script}")
    local job_id
    job_id="$("${command[@]}")"
    record_submission "${name}" "${job_id}"
    printf '%s' "${job_id}"
}

submit_primary() {
    local preflight flow materialize pair evaluation oracle nuts analysis local_p0 local_pstar
    preflight="$(submit_job preflight 01-preflight.sbatch "")"
    flow="$(submit_job flow 02-flow-oracle.sbatch "${preflight}")"
    materialize="$(submit_job materialize 03-materialize.sbatch "${flow}")"
    pair="$(submit_job pair-primary 10-pair-array.sbatch "${materialize}" --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=primary")"
    evaluation="$(submit_job materialize-evaluation 04-materialize-evaluation.sbatch "${pair}")"
    oracle="$(submit_job oracle-primary 11-oracle-array.sbatch "${evaluation}" --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=primary")"
    nuts="$(submit_job nuts-primary 12-nuts-array.sbatch "${evaluation}" --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},NUTS_PROFILE=primary")"
    analysis="$(submit_job analysis-primary 13-analyze-array.sbatch "${evaluation}" --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=primary")"
    local_p0="$(submit_job local-p0-primary 14-local-p0-array.sbatch "${evaluation}" --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=primary")"
    local_pstar="$(submit_job local-pstar-primary 15-local-pstar-array.sbatch "${oracle}" --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=primary")"
    printf '\nprimary stage submitted: pair=%s evaluation=%s oracle=%s nuts=%s analysis=%s local_p0=%s local_pstar=%s\n' \
        "${pair}" "${evaluation}" "${oracle}" "${nuts}" "${analysis}" "${local_p0}" "${local_pstar}"
}

submit_conditional() {
    local profile="$1"
    local job_id local_p0 local_pstar
    local_p0="$(submission_job_id "local-p0-${profile}")"
    local_pstar="$(submission_job_id "local-pstar-${profile}")"
    job_id="$(submit_job "conditional-${profile}" 16-conditional-array.sbatch "${local_p0}:${local_pstar}" \
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=${profile}")"
    printf '\nconditional %s submitted: %s\n' "${profile}" "${job_id}"
}

submit_index_aggregate() {
    local profile="$1"
    local index aggregate analysis
    analysis="$(submission_job_id "analysis-${profile}")"
    index="$(submit_job "index-${profile}" 17-build-index.sbatch "${analysis}" \
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=${profile}")"
    aggregate="$(submit_job "aggregate-${profile}" 18-aggregate.sbatch "${index}" \
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=${profile}")"
    printf '\nindex/aggregate %s submitted: %s/%s\n' "${profile}" "${index}" "${aggregate}"
}

continue_primary() {
    if [[ ! -f "${RUN_ROOT}/status/nuts-selection.json" ]]; then
        repo_pixi python "${HARNESS_DIR}/21-gate-nuts.py" \
            --run-root "${RUN_ROOT}" --harness-directory "${HARNESS_DIR}" --phase primary
        if [[ ! -f "${RUN_ROOT}/status/nuts-selection.json" ]]; then
            echo "sparse retry1 required; use 31-submit-nuts-retry1.sh" >&2
            return 10
        fi
        submit_conditional primary
        return
    fi
    if [[ ! -f "${RUN_ROOT}/status/conditional-primary-gate.json" ]]; then
        repo_pixi python "${HARNESS_DIR}/22-gate-conditional.py" \
            --run-root "${RUN_ROOT}" --harness-directory "${HARNESS_DIR}" \
            --repo-root "${REPO_ROOT}" --source-revision "${FULL_SHA}" --profile primary
    fi
    local action
    action="$(repo_pixi python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["action"])' \
        "${RUN_ROOT}/status/conditional-primary-gate.json")"
    if [[ "${action}" == "aggregate" ]]; then
        submit_index_aggregate primary
    elif [[ "${action}" == "factor4" ]]; then
        echo "authorized factor4 required; use 32-submit-factor4.sh" >&2
        return 11
    else
        echo "unrecognized conditional gate action: ${action}" >&2
        return 12
    fi
}

main() {
    case "${1:-}" in
        stage)
            [[ "$#" == 3 ]]
            stage_harness "$2" "$3"
            ;;
        submit)
            [[ "$#" == 3 ]]
            load_run "$2" "$3"
            submit_primary
            ;;
        continue)
            [[ "$#" == 3 ]]
            load_run "$2" "$3"
            continue_primary
            ;;
        *)
            echo "usage: $0 {stage BASE REPO|submit RUN_ROOT REPO|continue RUN_ROOT REPO}" >&2
            exit 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
