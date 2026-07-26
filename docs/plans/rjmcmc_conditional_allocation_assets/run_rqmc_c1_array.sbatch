#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --partition=test
#SBATCH --array=0-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=rqmc-c1

set -euo pipefail

: "${RQMC_SOURCE:?Set RQMC_SOURCE to the clean full-SHA worktree.}"
: "${RQMC_RUN_ROOT:?Set RQMC_RUN_ROOT to the immutable run directory.}"
: "${RQMC_REVISION:?Set RQMC_REVISION to the complete 40-character source SHA.}"

case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  near_gaussian__four_cell__row
  skewed__two_cell__root
  skewed__four_cell__root
  skewed__four_cell__row
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
  boundary_heavy__four_cell__row
)

if [[ "${#RQMC_REVISION}" -ne 40 || ! "${RQMC_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "RQMC_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne "${#case_ids[@]}" ]]; then
  echo "The Slurm array size does not match the frozen C1-RQMC matrix." >&2
  exit 2
fi
if [[ ! -d "${RQMC_SOURCE}" ]]; then
  echo "RQMC_SOURCE must be an existing worktree." >&2
  exit 2
fi
if [[ "$(git -C "${RQMC_SOURCE}" rev-parse HEAD)" != "${RQMC_REVISION}" ]]; then
  echo "RQMC_SOURCE does not match RQMC_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${RQMC_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "RQMC_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ "$(readlink -f "${RQMC_SOURCE}/.pixi")" != \
  "/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi" ]]; then
  echo "RQMC_SOURCE/.pixi must resolve to the canonical BP1 environment." >&2
  exit 2
fi
if [[ "$(cat "${RQMC_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.txt")" != \
  "RQMC C1 preflight complete for ${RQMC_REVISION}" ]]; then
  echo "The pinned preflight completion marker is absent or invalid." >&2
  exit 2
fi

case_id="${case_ids[SLURM_ARRAY_TASK_ID]}"
output="${RQMC_RUN_ROOT}/cases/${case_id}.json"
if [[ -e "${output}" ]]; then
  echo "Refusing to replace existing case artifact: ${output}" >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${RQMC_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rqmc-c1-matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/rqmc-c1-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${RQMC_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_allocation_likelihood_rqmc_tiny_screen.py \
  --profile development \
  --case-id "${case_id}" \
  --source-revision "${RQMC_REVISION}" \
  --output "${output}"
