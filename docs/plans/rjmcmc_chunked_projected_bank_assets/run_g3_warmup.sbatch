#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-warm
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"
: "${FROZEN_INPUT:?}"
: "${FROZEN_INPUT_SHA:?}"

stage="${BANK_RUN_ROOT}/g3/warmup"
report="${stage}/warmup_report.json"
complete="${stage}/WARMUP_COMPLETE.txt"
time_report="${stage}/resource.time"
for path in "${report}" "${complete}" "${time_report}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace warm-up evidence: ${path}" >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-warm-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-warm-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${BANK_SOURCE}"
/usr/bin/time -v -o "${time_report}" \
  pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  g3-warmup \
  --input "${FROZEN_INPUT}" \
  --expected-input-sha256 "${FROZEN_INPUT_SHA}" \
  --source-revision "${BANK_REVISION}" \
  --concentration 100 \
  --spectrum-manifest "${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json" \
  --g1-manifest "${BANK_RUN_ROOT}/g1/g1_report.json" \
  --output "${report}"

printf 'G3b excluded warm-up complete for %s in Slurm job %s\n' \
  "${BANK_REVISION}" "${SLURM_JOB_ID}" >"${complete}"
