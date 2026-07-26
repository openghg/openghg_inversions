#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/30-submit-primary.sh"

submit_retry() {
    local map="${RUN_ROOT}/status/nuts-retry-map.tsv"
    [[ -f "${map}" && ! -f "${RUN_ROOT}/status/nuts-selection.json" ]]
    local array
    array="$(awk -F '\t' 'NR>1 {ids=(ids?ids",":"")$1} END {if(!ids) exit 2; print ids"%2"}' "${map}")"
    local job_id
    job_id="$(submit_job nuts-retry1 12-nuts-array.sbatch "" \
        --array "${array}" --time 08:00:00 \
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},NUTS_PROFILE=retry1")"
    printf '\nsparse retry1 submitted: %s (%s)\n' "${job_id}" "${array}"
}

continue_retry() {
    repo_pixi python "${HARNESS_DIR}/21-gate-nuts.py" \
        --run-root "${RUN_ROOT}" --harness-directory "${HARNESS_DIR}" --phase retry1
    submit_conditional primary
}

case "${1:-}" in
    submit)
        [[ "$#" == 3 ]]
        load_run "$2" "$3"
        submit_retry
        ;;
    continue)
        [[ "$#" == 3 ]]
        load_run "$2" "$3"
        continue_retry
        ;;
    *)
        echo "usage: $0 {submit|continue} RUN_ROOT REPO" >&2
        exit 2
        ;;
esac
