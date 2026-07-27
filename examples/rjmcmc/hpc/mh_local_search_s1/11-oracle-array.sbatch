#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%6
#SBATCH --job-name=mh-s1-oracle
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${BRANCH_PROFILE:-primary}"
harness_job_init "oracle-${profile}"
IFS=$'\t' read -r _ scenario replicate cell_key <<<"$(cell_row "${SLURM_ARRAY_TASK_ID}")"
output="${RUN_ROOT}/11-oracle/${profile}/${cell_key}"
mkdir -p "$(dirname "${output}")"
command=(
    "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" run-oracle
    --training "${RUN_ROOT}/03-materialize/training/${cell_key}.json"
    --evaluation "${RUN_ROOT}/04-materialize-evaluation/sealed-evaluation/${cell_key}.json"
    --output-directory "${output}" --source-revision "${FULL_SHA}" --profile "${profile}"
)
if [[ "${profile}" == "factor4" ]]; then
    require_variable RETRY_AUTHORIZATION_TOKEN
    command+=(--retry-authorization-token "${RETRY_AUTHORIZATION_TOKEN}")
fi
pixi_python "${command[@]}"
harness_job_complete "oracle-${profile}" "${output}/complete.json"
