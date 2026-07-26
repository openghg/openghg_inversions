#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=gmm-c4b-confirm
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${GMM_SOURCE:?Set GMM_SOURCE to the clean full-SHA worktree.}"
: "${GMM_RUN_ROOT:?Set GMM_RUN_ROOT to the pre-created immutable run directory.}"
: "${GMM_REVISION:?Set GMM_REVISION to the complete 40-character source SHA.}"

module load git/2.45.1-pqk5

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  skewed__two_cell__root
  skewed__four_cell__root
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
)
confirmation_seeds=(1877 4099 8317)

if [[ "${#GMM_REVISION}" -ne 40 || ! "${GMM_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "GMM_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 18 ]]; then
  echo "The Slurm array must contain exactly 18 tasks." >&2
  exit 2
fi
if [[ "${GMM_SOURCE}" == *PARIS_inversions* || "${GMM_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "C4b source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
confirmation="${GMM_RUN_ROOT}/confirmation"
marker_directory="${GMM_RUN_ROOT}/markers/confirmation"
log_directory="${GMM_RUN_ROOT}/logs/confirmation"
lock="${GMM_RUN_ROOT}/lock/common-lock.json"
raw_sha_record="${GMM_RUN_ROOT}/lock/common-lock.raw.sha256"
for directory in "${GMM_SOURCE}" "${GMM_RUN_ROOT}" "${confirmation}" \
  "${marker_directory}" "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${lock}" || -L "${lock}" || ! -f "${raw_sha_record}" ||
      -L "${raw_sha_record}" ]]; then
  echo "The common lock and raw SHA record must be regular non-symlink files." >&2
  exit 2
fi
if [[ "$(git -C "${GMM_SOURCE}" rev-parse HEAD)" != "${GMM_REVISION}" ]]; then
  echo "GMM_SOURCE does not match GMM_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${GMM_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "GMM_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${GMM_SOURCE}/.pixi" ||
      "$(readlink -f "${GMM_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "GMM_SOURCE/.pixi must be a symlink to the canonical BP1 environment." >&2
  exit 2
fi
lock_raw_sha256="$(<"${raw_sha_record}")"
if [[ "${#lock_raw_sha256}" -ne 64 || ! "${lock_raw_sha256}" =~ ^[0-9a-f]+$ ||
      "$(sha256sum "${lock}" | awk '{print $1}')" != "${lock_raw_sha256}" ]]; then
  echo "The common lock does not match its immutable raw SHA record." >&2
  exit 2
fi
if [[ "$(cat "${GMM_RUN_ROOT}/lock/MERGE_DEVELOPMENT_COMPLETE.txt")" != \
      "GMM C4b development lock complete for ${GMM_REVISION} raw_sha256=${lock_raw_sha256}" ]]; then
  echo "The development-lock completion marker is absent or invalid." >&2
  exit 2
fi

case_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
case_id="${case_ids[case_index]}"
confirmation_seed="${confirmation_seeds[seed_index]}"
stem="${case_id}__seed${confirmation_seed}"
output="${confirmation}/${stem}.json"
marker="${marker_directory}/${stem}.complete"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ -e "${output}" || -L "${output}" || -e "${marker}" || -L "${marker}" ||
      -e "${log}" || -L "${log}" ]]; then
  echo "Refusing to replace existing confirmation evidence for ${stem}." >&2
  exit 2
fi
exec >"${log}" 2>&1

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${GMM_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gmm-c4b-confirm-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/gmm-c4b-confirm-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${GMM_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_gmm_tiny_screen.py \
  --profile development \
  --case-id "${case_id}" \
  --source-revision "${GMM_REVISION}" \
  --confirmation-lock "${lock}" \
  --expected-confirmation-lock-sha256 "${lock_raw_sha256}" \
  --confirmation-seed "${confirmation_seed}" \
  --output "${output}"

printf 'complete revision=%s case=%s confirmation_seed=%s lock_raw_sha256=%s\n' \
  "${GMM_REVISION}" "${case_id}" "${confirmation_seed}" "${lock_raw_sha256}" > "${marker}"
