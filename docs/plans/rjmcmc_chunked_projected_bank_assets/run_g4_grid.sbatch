#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g4-grid
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"
: "${FROZEN_INPUT:?}"
: "${FROZEN_INPUT_SHA:?}"

stage="${BANK_RUN_ROOT}/g4/grid"
time_report="${stage}/resource.time"
for path in "${stage}/grid_manifest.json" "${stage}/G4_GRID_COMPLETE.txt" \
  "${time_report}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G4 grid evidence: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${BANK_RUN_ROOT}/g3/G3_COMPLETE.txt" ]]; then
  echo "A passing G3 certificate is required before G4." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-g4-grid-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-g4-grid-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${BANK_SOURCE}"
/usr/bin/time -v -o "${time_report}" \
  pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_g4.py \
  grid \
  --input "${FROZEN_INPUT}" \
  --expected-input-sha256 "${FROZEN_INPUT_SHA}" \
  --spectrum-manifest \
    "${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json" \
  --source-revision "${BANK_REVISION}" \
  --output-dir "${stage}"

printf 'G4 observation-blind grid complete for %s in Slurm job %s\n' \
  "${BANK_REVISION}" "${SLURM_JOB_ID}" >"${stage}/G4_GRID_COMPLETE.txt"
