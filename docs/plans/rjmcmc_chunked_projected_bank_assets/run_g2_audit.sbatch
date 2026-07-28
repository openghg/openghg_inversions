#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g2-audit
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

authoritative="${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json"
stage="${BANK_RUN_ROOT}/g2/audit"
report="${stage}/g2_audit_report.json"
complete="${stage}/G2_AUDIT_COMPLETE.txt"
time_report="${stage}/resource.time"
for path in "${stage}/spectrum_manifest.json" "${report}" "${complete}" "${time_report}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G2 audit evidence: ${path}" >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-g2-audit-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-g2-audit-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${BANK_SOURCE}"
/usr/bin/time -v -o "${time_report}" \
  pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  g2 \
  --input "${FROZEN_INPUT}" \
  --expected-input-sha256 "${FROZEN_INPUT_SHA}" \
  --source-revision "${BANK_REVISION}" \
  --concentration 528.618161317525 \
  --output-dir "${stage}"

pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  g2-audit \
  --authoritative-manifest "${authoritative}" \
  --audit-manifest "${stage}/spectrum_manifest.json" \
  --source-revision "${BANK_REVISION}" \
  --output "${report}"

printf 'Second-node G2 audit complete for %s in Slurm job %s on %s\n' \
  "${BANK_REVISION}" "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-unknown}" >"${complete}"
