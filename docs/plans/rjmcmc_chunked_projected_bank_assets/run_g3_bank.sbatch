#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=chunk-bank-g3b
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
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "G3b candidates must run as one Slurm array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 11 )); then
  echo "G3b array task must lie in 0..11." >&2
  exit 2
fi
chunks=(1024 2048 4096 8192)
BANK_SAMPLE_CHUNK="${chunks[SLURM_ARRAY_TASK_ID / 3]}"
BANK_REPEAT="$((SLURM_ARRAY_TASK_ID % 3))"
BANK_CANDIDATE_DIR="${BANK_RUN_ROOT}/g3/resource/C${BANK_SAMPLE_CHUNK}/repeat${BANK_REPEAT}"
mkdir -p "${BANK_CANDIDATE_DIR}"

stage="${BANK_CANDIDATE_DIR}"
complete="${stage}/CANDIDATE_COMPLETE.txt"
time_report="${stage}/resource.time"
for path in "${stage}/projected_locations.npy" "${stage}/bank_manifest.json" \
  "${complete}" "${time_report}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G3b candidate evidence: ${path}" >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-g3b-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-g3b-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${BANK_SOURCE}"
/usr/bin/time -v -o "${time_report}" \
  pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  g3-bank \
  --input "${FROZEN_INPUT}" \
  --expected-input-sha256 "${FROZEN_INPUT_SHA}" \
  --source-revision "${BANK_REVISION}" \
  --concentration 528.618161317525 \
  --spectrum-manifest "${BANK_RUN_ROOT}/g2/authoritative/spectrum_manifest.json" \
  --g1-manifest "${BANK_RUN_ROOT}/g1/g1_report.json" \
  --sample-chunk-size "${BANK_SAMPLE_CHUNK}" \
  --repeat "${BANK_REPEAT}" \
  --output-dir "${stage}"

printf 'G3b C=%s repeat=%s complete in Slurm array %s task %s job %s\n' \
  "${BANK_SAMPLE_CHUNK}" "${BANK_REPEAT}" "${SLURM_ARRAY_JOB_ID}" \
  "${SLURM_ARRAY_TASK_ID}" "${SLURM_JOB_ID}" >"${complete}"
