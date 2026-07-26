#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=mh-s0-local-pstar
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${BRANCH_PROFILE:-primary}"
harness_job_init "local-pstar-${profile}"
row="$(awk -F '\t' -v ordinal="$((SLURM_ARRAY_TASK_ID + 1))" 'NR>1 && $3=="pstar" {n++; if(n==ordinal){print; exit}}' "${HARNESS_DIR}/reference-map.tsv")"
[[ -n "${row}" ]]
IFS=$'\t' read -r _ scenario topology cell_task reference_key <<<"${row}"
IFS=$'\t' read -r _ _ _ cell_key <<<"$(cell_row "${cell_task}")"
output="${RUN_ROOT}/15-local-pstar/${profile}/${reference_key}"
mkdir -p "$(dirname "${output}")"
command=(
    "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" run-local-reference
    --training "${RUN_ROOT}/03-materialize/training/${cell_key}.json"
    --evaluation "${RUN_ROOT}/03-materialize/sealed-evaluation/${cell_key}.json"
    --topology pstar --branch-run-directory "${RUN_ROOT}/11-oracle/${profile}/${cell_key}"
    --output-directory "${output}" --source-revision "${FULL_SHA}" --profile "${profile}"
)
if [[ "${profile}" == "factor4" ]]; then
    require_variable RETRY_AUTHORIZATION_TOKEN
    command+=(--retry-authorization-token "${RETRY_AUTHORIZATION_TOKEN}")
fi
pixi_python "${command[@]}"
harness_job_complete "local-pstar-${profile}" "${output}/complete.json"
