#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=mh-s1-materialize
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
harness_job_init materialize
output="${RUN_ROOT}/03-materialize"
mkdir -p "${output}/training"
temporary="$(mktemp -d)"
trap 'rm -r -- "${temporary}"' EXIT
definition="${temporary}/definition.json"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" define \
    --stage s1 --output "${definition}"
pixi_python -c '
import hashlib,json,sys
definition_path,spec_path=sys.argv[1:]
raw=open(definition_path,"rb").read()
envelope=json.loads(raw)
spec=json.load(open(spec_path,encoding="utf-8"))["frozen_definition"]
if hashlib.sha256(raw).hexdigest()!=spec["envelope_sha256"]:
    raise SystemExit("S1 definition envelope digest drifted")
payload=envelope["payload"]
checks={
    "training_operator_sha256":payload["training_operator_sha256"],
    "heldout_operator_sha256":payload["heldout_operator_sha256"],
    "p0_sha256":payload["p0_sha256"],
    "edge_pstar_sha256":payload["scenarios"]["edge-one"]["pstar_sha256"],
    "relocation_pstar_sha256":payload["scenarios"]["relocation-one"]["pstar_sha256"],
}
if envelope["payload_sha256"]!=spec["payload_sha256"] or checks!={k:spec[k] for k in checks}:
    raise SystemExit("S1 frozen scientific identities drifted")
' "${definition}" "${HARNESS_DIR}/command-spec.json"
files=()
while IFS=$'\t' read -r task scenario replicate cell_key; do
    [[ "${task}" == "task_id" ]] && continue
    pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" materialize-training \
        --definition "${definition}" \
        --scenario "${scenario}" --replicate "${replicate}" \
        --training-output "${output}/training/${cell_key}.json"
    files+=("training/${cell_key}.json")
done <"${HARNESS_DIR}/cell-map.tsv"
seal=( "${HARNESS_DIR}/20-audit-artifact.py" seal --directory "${output}" )
for name in "${files[@]}"; do seal+=(--file "${name}"); done
pixi_python "${seal[@]}"
rm -r -- "${temporary}"
trap - EXIT
harness_job_complete materialize "${output}/complete.json"
