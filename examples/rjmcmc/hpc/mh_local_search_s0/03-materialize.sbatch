#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=mh-s0-materialize
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
harness_job_init materialize
output="${RUN_ROOT}/03-materialize"
mkdir -p "${output}/training" "${output}/sealed-evaluation"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" define \
    --stage s0 --output "${output}/definition.json"
files=(definition.json)
while IFS=$'\t' read -r task scenario replicate cell_key; do
    [[ "${task}" == "task_id" ]] && continue
    pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" materialize \
        --definition "${output}/definition.json" \
        --scenario "${scenario}" --replicate "${replicate}" \
        --training-output "${output}/training/${cell_key}.json" \
        --evaluation-output "${output}/sealed-evaluation/${cell_key}.json"
    files+=("training/${cell_key}.json" "sealed-evaluation/${cell_key}.json")
done <"${HARNESS_DIR}/cell-map.tsv"
seal=( "${HARNESS_DIR}/20-audit-artifact.py" seal --directory "${output}" )
for name in "${files[@]}"; do seal+=(--file "${name}"); done
pixi_python "${seal[@]}"
harness_job_complete materialize "${output}/complete.json"
