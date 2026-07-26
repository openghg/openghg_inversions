#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%2
#SBATCH --job-name=mh-s0-conditional
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
branch_profile="${BRANCH_PROFILE:-primary}"
harness_job_init "conditional-${branch_profile}"
IFS=$'\t' read -r _ scenario topology cell_task reference_key <<<"$(reference_row "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r _ _ _ cell_key <<<"$(cell_row "${cell_task}")"
nuts_profile="$(pixi_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"][sys.argv[2]])' "${RUN_ROOT}/status/nuts-selection.json" "${reference_key}")"
if [[ "${topology}" == "p0" ]]; then
    local_directory="${RUN_ROOT}/14-local-p0/${branch_profile}/${reference_key}"
else
    local_directory="${RUN_ROOT}/15-local-pstar/${branch_profile}/${reference_key}"
fi
output="${RUN_ROOT}/16-conditional/local-${branch_profile}/nuts-${nuts_profile}/${reference_key}"
mkdir -p "$(dirname "${output}")"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_conditional_reference.py" \
    --training "${RUN_ROOT}/03-materialize/training/${cell_key}.json" \
    --evaluation "${RUN_ROOT}/03-materialize/sealed-evaluation/${cell_key}.json" \
    --nuts-directory "${RUN_ROOT}/12-nuts/${reference_key}/${nuts_profile}" \
    --local-directory "${local_directory}" --output-directory "${output}" \
    --source-revision "${FULL_SHA}"
harness_job_complete "conditional-${branch_profile}" "${output}/complete.json"
