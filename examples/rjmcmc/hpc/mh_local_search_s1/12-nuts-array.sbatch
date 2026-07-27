#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%2
#SBATCH --job-name=mh-s1-nuts
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${NUTS_PROFILE:-primary}"
harness_job_init "nuts-${profile}"
IFS=$'\t' read -r _ scenario topology cell_task reference_key <<<"$(reference_row "${SLURM_ARRAY_TASK_ID}")"
IFS=$'\t' read -r _ _ _ cell_key <<<"$(cell_row "${cell_task}")"
training="${RUN_ROOT}/03-materialize/training/${cell_key}.json"
evaluation="${RUN_ROOT}/04-materialize-evaluation/sealed-evaluation/${cell_key}.json"
output="${RUN_ROOT}/12-nuts/${reference_key}/${profile}"
mkdir -p "$(dirname "${output}")"
read -r cell_id definition_sha256 < <(
    pixi_python -c 'import json,sys; p=json.load(open(sys.argv[1])); q=p["payload"]; print(q["cell_id"], q["definition_sha256"])' "${training}"
)
command=(
    "${REPO_ROOT}/examples/rjmcmc/mh_local_search_nuts_reference.py"
    --training "${training}" --evaluation "${evaluation}" --topology "${topology}"
    --profile "${profile}" --output-directory "${output}" --source-revision "${FULL_SHA}"
    --cell-id "${cell_id}" --definition-sha256 "${definition_sha256}"
)
if [[ "${profile}" == "retry1" ]]; then
    command+=(--primary-nuts-directory "${RUN_ROOT}/12-nuts/${reference_key}/primary")
fi
pixi_python "${command[@]}"
harness_job_complete "nuts-${profile}" "${output}/complete.json"
