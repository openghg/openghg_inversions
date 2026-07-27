#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/30-submit-primary.sh"

authorization_token() {
    repo_pixi python -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(p["authorization"]["directory"]+"/token.json")' \
        "${RUN_ROOT}/status/conditional-primary-gate.json"
}

submit_factor4() {
    local token pair oracle analysis local_p0 local_pstar conditional
    token="$(authorization_token)"
    [[ -f "${token}" ]]
    local export_value="ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=factor4,RETRY_AUTHORIZATION_TOKEN=${token}"
    pair="$(submit_job pair-factor4 10-pair-array.sbatch "" --mem 12G --time 48:00:00 --export "${export_value}")"
    oracle="$(submit_job oracle-factor4 11-oracle-array.sbatch "" --mem 12G --time 48:00:00 --export "${export_value}")"
    analysis="$(submit_job analysis-factor4 13-analyze-array.sbatch "${pair}" --mem 8G --time 08:00:00 --export "${export_value}")"
    local_p0="$(submit_job local-p0-factor4 14-local-p0-array.sbatch "${pair}" --mem 16G --time 72:00:00 --export "${export_value}")"
    local_pstar="$(submit_job local-pstar-factor4 15-local-pstar-array.sbatch "${oracle}" --mem 16G --time 72:00:00 --export "${export_value}")"
    conditional="$(submit_job conditional-factor4 16-conditional-array.sbatch "${local_p0}:${local_pstar}:${analysis}" --mem 8G --time 08:00:00 --export "${export_value}")"
    printf '\nfactor4 submitted: pair=%s oracle=%s analysis=%s local=%s/%s conditional=%s\n' \
        "${pair}" "${oracle}" "${analysis}" "${local_p0}" "${local_pstar}" "${conditional}"
}

continue_factor4() {
    repo_pixi python "${HARNESS_DIR}/22-gate-conditional.py" \
        --run-root "${RUN_ROOT}" --harness-directory "${HARNESS_DIR}" \
        --repo-root "${REPO_ROOT}" --source-revision "${FULL_SHA}" --profile factor4
    local action
    action="$(repo_pixi python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["action"])' \
        "${RUN_ROOT}/status/conditional-factor4-gate.json")"
    [[ "${action}" == "aggregate" ]]
    local index aggregate
    index="$(submit_job index-factor4 17-build-index.sbatch "" --mem 8G --time 02:00:00 \
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=factor4")"
    aggregate="$(submit_job aggregate-factor4 18-aggregate.sbatch "${index}" --time 08:00:00 \
        --export "ALL,RUN_ROOT=${RUN_ROOT},REPO_ROOT=${REPO_ROOT},HARNESS_DIR=${HARNESS_DIR},FULL_SHA=${FULL_SHA},PIXI_LOCK_SHA256=${PIXI_LOCK_SHA256},HARNESS_SHA256=${HARNESS_SHA256},PIXI_EXE=${PIXI_EXE},BRANCH_PROFILE=factor4")"
    printf '\nfactor4 index/aggregate submitted: %s/%s\n' "${index}" "${aggregate}"
}

case "${1:-}" in
    submit)
        [[ "$#" == 3 ]]
        load_run "$2" "$3"
        submit_factor4
        ;;
    continue)
        [[ "$#" == 3 ]]
        load_run "$2" "$3"
        continue_factor4
        ;;
    *)
        echo "usage: $0 {submit|continue} RUN_ROOT REPO" >&2
        exit 2
        ;;
esac
