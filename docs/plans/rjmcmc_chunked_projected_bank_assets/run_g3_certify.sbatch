#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-cert
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"

stage="${BANK_RUN_ROOT}/g3"
report="${stage}/g3_decision.json"
complete="${stage}/G3_COMPLETE.txt"
for path in "${report}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G3 certification evidence: ${path}" >&2
    exit 2
  fi
done

mapfile -t manifests < <(
  find "${stage}/resource" -mindepth 3 -maxdepth 3 \
    -name bank_manifest.json -type f -print | sort
)
manifest_arguments=()
for manifest in "${manifests[@]}"; do
  manifest_arguments+=(--candidate-manifest "${manifest}")
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${BANK_SOURCE}"
pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  g3-certify \
  --source-revision "${BANK_REVISION}" \
  --prefix-manifest "${stage}/prefix/g3a_report.json" \
  --output "${report}" \
  --completion-marker "${complete}" \
  "${manifest_arguments[@]}"
