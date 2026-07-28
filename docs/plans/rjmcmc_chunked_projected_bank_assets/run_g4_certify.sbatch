#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g4-cert
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"

stage="${BANK_RUN_ROOT}/g4"
report="${stage}/g4_decision.json"
complete="${stage}/G4_SOURCE_LOCK.txt"
for path in "${report}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G4 certification evidence: ${path}" >&2
    exit 2
  fi
done
for seed in 1877 4099 8317; do
  if [[ ! -f "${stage}/confirmation/seed${seed}/G4_SEED_COMPLETE.txt" ]]; then
    echo "Missing complete G4 confirmation seed ${seed}." >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${BANK_SOURCE}"
pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_g4.py \
  all-seed-certify \
  --development-report \
    "${stage}/development/development_decision.json" \
  --seed-report "${stage}/development/seed731/seed_report.json" \
  --seed-report "${stage}/confirmation/seed1877/seed_report.json" \
  --seed-report "${stage}/confirmation/seed4099/seed_report.json" \
  --seed-report "${stage}/confirmation/seed8317/seed_report.json" \
  --spectrum-manifest \
    "${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json" \
  --grid-manifest "${stage}/grid/grid_manifest.json" \
  --source-revision "${BANK_REVISION}" \
  --output "${report}" \
  --lock-marker "${complete}"
