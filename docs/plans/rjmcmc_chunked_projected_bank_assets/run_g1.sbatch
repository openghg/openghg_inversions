#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"

stage="${BANK_RUN_ROOT}/g1"
report="${stage}/g1_report.json"
complete="${stage}/G1_COMPLETE.txt"
time_report="${stage}/resource.time"
for path in "${report}" "${complete}" "${time_report}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G1 evidence: ${path}" >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-g1-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-g1-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${BANK_SOURCE}"
pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_chunked_projected_bank_hpc.py

/usr/bin/time -v -o "${time_report}" \
  pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  g1 \
  --output "${report}" \
  --source-revision "${BANK_REVISION}"

printf 'G1 complete for %s in Slurm job %s\n' \
  "${BANK_REVISION}" "${SLURM_JOB_ID}" >"${complete}"
