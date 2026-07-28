#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g4-conf
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"
: "${FROZEN_INPUT:?}"
: "${FROZEN_INPUT_SHA:?}"
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "G4 confirmations must run as one Slurm array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 2 )); then
  echo "G4 confirmation array task must lie in 0..2." >&2
  exit 2
fi
seeds=(1877 4099 8317)
source_seed="${seeds[SLURM_ARRAY_TASK_ID]}"
stage="${BANK_RUN_ROOT}/g4/confirmation/seed${source_seed}"
mkdir -p "${stage}"
time_report="${stage}/resource.time"
for path in "${stage}/seed_report.json" "${stage}/projected_locations.npy" \
  "${stage}/log_likelihood.npy" "${stage}/G4_SEED_COMPLETE.txt" \
  "${time_report}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G4 confirmation evidence: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${BANK_RUN_ROOT}/g4/development/G4_DEVELOPMENT_COMPLETE.txt" ]]; then
  echo "A passing development certificate is required before confirmation." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-g4-conf-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-g4-conf-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${BANK_SOURCE}"
/usr/bin/time -v -o "${time_report}" \
  pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_g4.py \
  seed \
  --input "${FROZEN_INPUT}" \
  --expected-input-sha256 "${FROZEN_INPUT_SHA}" \
  --spectrum-manifest \
    "${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json" \
  --grid-manifest "${BANK_RUN_ROOT}/g4/grid/grid_manifest.json" \
  --g3-decision "${BANK_RUN_ROOT}/g3/g3_decision.json" \
  --source-revision "${BANK_REVISION}" \
  --source-seed "${source_seed}" \
  --output-dir "${stage}"
