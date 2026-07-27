#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=exact-mix-confirm
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${CONF_SOURCE:?Set CONF_SOURCE to the clean full-SHA worktree.}"
: "${CONF_RUN_ROOT:?Set CONF_RUN_ROOT to the fresh confirmation run root.}"
: "${CONF_REVISION:?Set CONF_REVISION to the complete candidate SHA.}"
: "${DEV_RUN_ROOT:?Set DEV_RUN_ROOT to the immutable d23 development run.}"

module load git/2.45.1-pqk5

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
expected_dev_root="/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_compression/d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea"
source_decision="${DEV_RUN_ROOT}/decision/common-source-decision.json"
compression_decision="${DEV_RUN_ROOT}/decision/common-compression-decision.json"
case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  skewed__two_cell__root
  skewed__four_cell__root
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
)
source_seeds=(1877 4099 8317)

if [[ "${DEV_RUN_ROOT}" != "${expected_dev_root}" ]]; then
  echo "DEV_RUN_ROOT is not the frozen d23 development run." >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 18 ]]; then
  echo "The confirmation array must contain exactly 18 tasks." >&2
  exit 2
fi
confirmation_directory="${CONF_RUN_ROOT}/confirmation"
marker_directory="${CONF_RUN_ROOT}/markers/confirmation"
log_directory="${CONF_RUN_ROOT}/logs/confirmation"
for directory in "${CONF_SOURCE}" "${CONF_RUN_ROOT}" "${confirmation_directory}" \
  "${marker_directory}" "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ "$(<"${CONF_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.txt")" != \
      "Exact-mixture confirmation preflight complete for ${CONF_REVISION}" ]]; then
  echo "The confirmation preflight marker is absent or invalid." >&2
  exit 2
fi
if [[ "$(git -C "${CONF_SOURCE}" rev-parse HEAD)" != "${CONF_REVISION}" ]]; then
  echo "CONF_SOURCE does not match CONF_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${CONF_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "CONF_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${CONF_SOURCE}/.pixi" ||
      "$(readlink -f "${CONF_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "CONF_SOURCE/.pixi must point to the canonical BP1 environment." >&2
  exit 2
fi

case_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
case_id="${case_ids[case_index]}"
source_seed="${source_seeds[seed_index]}"
stem="${case_id}__seed${source_seed}"
output="${confirmation_directory}/${stem}.json"
marker="${marker_directory}/${stem}.complete"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
for path in "${output}" "${marker}" "${log}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace confirmation evidence: ${path}" >&2
    exit 2
  fi
done
exec >"${log}" 2>&1

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${CONF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/exact-mix-confirm-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/exact-mix-confirm-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${CONF_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm.py \
  run \
  --case-id "${case_id}" \
  --source-seed "${source_seed}" \
  --development-source-decision "${source_decision}" \
  --development-compression-decision "${compression_decision}" \
  --source-revision "${CONF_REVISION}" \
  --output "${output}"

printf 'complete revision=%s case=%s source_seed=%s\n' \
  "${CONF_REVISION}" "${case_id}" "${source_seed}" >"${marker}"
