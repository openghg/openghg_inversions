#!/usr/bin/env bash

#SBATCH --job-name=exact-mix-paris-probe
#SBATCH --account=chem007981
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

set -euo pipefail
set -o noclobber

: "${PROBE_SOURCE:?Set PROBE_SOURCE to the clean full-SHA worktree.}"
: "${PROBE_RUN_ROOT:?Set PROBE_RUN_ROOT to the fresh run root.}"
: "${PROBE_REVISION:?Set PROBE_REVISION to the complete candidate SHA.}"
: "${FROZEN_INPUT:?Set FROZEN_INPUT to the reviewed PARIS NetCDF.}"
: "${FROZEN_INPUT_SHA:?Set FROZEN_INPUT_SHA to its whole-file SHA-256.}"
: "${FROZEN_INPUT_ID:?Set FROZEN_INPUT_ID to its reviewed schema ID.}"

module load git/2.45.1-pqk5

concentrations=(100 500)
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#concentrations[@]}" ]]; then
  echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}." >&2
  exit 2
fi
eta="${concentrations[${SLURM_ARRAY_TASK_ID}]}"
probe_directory="${PROBE_RUN_ROOT}/probe"
marker_directory="${PROBE_RUN_ROOT}/markers"
log_directory="${PROBE_RUN_ROOT}/logs"
output="${probe_directory}/concentration-${eta}.json"
marker="${marker_directory}/concentration-${eta}.complete"
log="${log_directory}/concentration-${eta}.${SLURM_JOB_ID}.log"

for directory in "${PROBE_SOURCE}" "${PROBE_RUN_ROOT}" "${probe_directory}" \
  "${marker_directory}" "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
for path in "${output}" "${marker}" "${log}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace probe evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${PROBE_SOURCE}" rev-parse HEAD)" != "${PROBE_REVISION}" ]]; then
  echo "PROBE_SOURCE does not match PROBE_REVISION." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROBE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/exact-mixture-paris-probe-mpl-${SLURM_JOB_ID}-${eta}"
export NUMBA_CACHE_DIR="/tmp/exact-mixture-paris-probe-numba-${SLURM_JOB_ID}-${eta}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: >"${log}"
exec > >(tee -a "${log}") 2>&1

cd "${PROBE_SOURCE}"
/usr/bin/time -v pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_exact_mixture_paris_probe.py \
  --input "${FROZEN_INPUT}" \
  --expected-input-sha256 "${FROZEN_INPUT_SHA}" \
  --input-id "${FROZEN_INPUT_ID}" \
  --source-revision "${PROBE_REVISION}" \
  --concentration "${eta}" \
  --engine h5netcdf \
  --include-timings \
  --output "${output}"

printf 'complete revision=%s concentration=%s output_sha256=%s\n' \
  "${PROBE_REVISION}" "${eta}" "$(sha256sum "${output}" | awk '{print $1}')" \
  >"${marker}"
