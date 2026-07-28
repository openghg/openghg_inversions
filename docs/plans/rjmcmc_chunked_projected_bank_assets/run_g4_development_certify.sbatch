#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g4-dev-cert
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"

stage="${BANK_RUN_ROOT}/g4/development"
report="${stage}/development_decision.json"
complete="${stage}/G4_DEVELOPMENT_COMPLETE.txt"
for path in "${report}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G4 development certification: ${path}" >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${BANK_SOURCE}"
pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_g4.py \
  development-certify \
  --seed-report "${stage}/seed731/seed_report.json" \
  --spectrum-manifest \
    "${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json" \
  --grid-manifest "${BANK_RUN_ROOT}/g4/grid/grid_manifest.json" \
  --source-revision "${BANK_REVISION}" \
  --output "${report}" \
  --completion-marker "${complete}"
