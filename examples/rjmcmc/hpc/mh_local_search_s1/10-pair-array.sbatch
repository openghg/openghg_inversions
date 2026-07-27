#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%6
#SBATCH --job-name=mh-s1-pair
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${BRANCH_PROFILE:-primary}"
harness_job_init "pair-${profile}"
IFS=$'\t' read -r _ _ _ cell_key <<<"$(cell_row "${SLURM_ARRAY_TASK_ID}")"
training="${RUN_ROOT}/03-materialize/training/${cell_key}.json"
output="${RUN_ROOT}/10-pair/${profile}/${cell_key}"
work="${RUN_ROOT}/jobs/practical/${profile}/${cell_key}"
mkdir -p "${work}" "$(dirname "${output}")"
command=(
    "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" run-pair
    --training "${training}" --output-directory "${output}"
    --source-revision "${FULL_SHA}" --profile "${profile}"
)
if [[ "${profile}" == "factor4" ]]; then
    require_variable RETRY_AUTHORIZATION_TOKEN
    command+=(--retry-authorization-token "${RETRY_AUTHORIZATION_TOKEN}")
fi
cd "${work}"
(
    unset RUN_ROOT HARNESS_DIR BRANCH_PROFILE RETRY_AUTHORIZATION_TOKEN
    pixi_python_at "${work}" "${command[@]}"
)
harness_job_complete "pair-${profile}" "${output}/complete.json"
