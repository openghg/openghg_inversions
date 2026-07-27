#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=mh-s1-evaluation
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
harness_job_init materialize-evaluation
output="${RUN_ROOT}/04-materialize-evaluation"
mkdir -p "${output}/sealed-evaluation"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" define \
    --stage s1 --output "${output}/definition.json"
expected_definition_sha256="$(pixi_python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["frozen_definition"]["envelope_sha256"])' \
    "${HARNESS_DIR}/command-spec.json")"
[[ "$(sha256_file "${output}/definition.json")" == "${expected_definition_sha256}" ]]
files=(definition.json)
while IFS=$'\t' read -r task scenario replicate cell_key; do
    [[ "${task}" == "task_id" ]] && continue
    pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" materialize-evaluation \
        --definition "${output}/definition.json" \
        --training "${RUN_ROOT}/03-materialize/training/${cell_key}.json" \
        --scenario "${scenario}" --replicate "${replicate}" \
        --evaluation-output "${output}/sealed-evaluation/${cell_key}.json"
    files+=("sealed-evaluation/${cell_key}.json")
done <"${HARNESS_DIR}/cell-map.tsv"
seal=( "${HARNESS_DIR}/20-audit-artifact.py" seal --directory "${output}" )
for name in "${files[@]}"; do seal+=(--file "${name}"); done
pixi_python "${seal[@]}"
harness_job_complete materialize-evaluation "${output}/complete.json"
