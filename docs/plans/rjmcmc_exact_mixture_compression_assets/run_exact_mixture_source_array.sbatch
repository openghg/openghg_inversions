#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=exact-mix-source
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${MIX_SOURCE:?Set MIX_SOURCE to the clean full-SHA worktree.}"
: "${MIX_RUN_ROOT:?Set MIX_RUN_ROOT to the pre-created immutable run directory.}"
: "${MIX_REVISION:?Set MIX_REVISION to the complete 40-character source SHA.}"

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

if [[ "${#MIX_REVISION}" -ne 40 || ! "${MIX_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "MIX_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 6 ]]; then
  echo "The source array must contain exactly six tasks." >&2
  exit 2
fi
if [[ "${MIX_SOURCE}" == *PARIS_inversions* || "${MIX_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Exact-mixture source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
source_directory="${MIX_RUN_ROOT}/source"
marker_directory="${MIX_RUN_ROOT}/markers/source"
log_directory="${MIX_RUN_ROOT}/logs/source"
for directory in "${MIX_SOURCE}" "${MIX_RUN_ROOT}" "${source_directory}" \
  "${marker_directory}" "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${MIX_SOURCE}" rev-parse HEAD)" != "${MIX_REVISION}" ]]; then
  echo "MIX_SOURCE does not match MIX_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${MIX_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "MIX_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${MIX_SOURCE}/.pixi" ||
      "$(readlink -f "${MIX_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "MIX_SOURCE/.pixi must point to the canonical BP1 environment." >&2
  exit 2
fi
if [[ "$(<"${MIX_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.txt")" != \
      "Exact-mixture preflight complete for ${MIX_REVISION}" ]]; then
  echo "The exact preflight marker is absent or invalid." >&2
  exit 2
fi

case_id="${case_ids[SLURM_ARRAY_TASK_ID]}"
output="${source_directory}/${case_id}.json"
marker="${marker_directory}/${case_id}.complete"
log="${log_directory}/${case_id}.${SLURM_JOB_ID}.log"
for path in "${output}" "${marker}" "${log}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace source evidence: ${path}" >&2
    exit 2
  fi
done
exec >"${log}" 2>&1

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${MIX_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/exact-mix-source-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/exact-mix-source-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${MIX_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_compressed_mixture_tiny_screen.py \
  --profile development \
  --stage source \
  --case-id "${case_id}" \
  --source-revision "${MIX_REVISION}" \
  --output "${output}"

printf 'complete revision=%s stage=source case=%s\n' \
  "${MIX_REVISION}" "${case_id}" >"${marker}"
