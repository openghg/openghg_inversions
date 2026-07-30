#!/usr/bin/env bash
#SBATCH --job-name=score-nle-promote-merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:45:00

set -euo pipefail

module load git/2.45.1-pqk5

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the fresh corrected run root.}"
: "${NLE_REVISION:?Set NLE_REVISION to the exact 40-character source SHA.}"
: "${NLE_MATRIX_ID:?Set NLE_MATRIX_ID to one frozen promotion matrix.}"
: "${NLE_ATTEMPT_TAG:?Set NLE_ATTEMPT_TAG to the immutable array tag.}"

source_root="$(realpath -e "${NLE_SOURCE}")"
run_root="$(realpath -e "${NLE_RUN_ROOT}")"
python_bin="${source_root}/.pixi/envs/nle-dev/bin/python"
log_directory="${run_root}/logs/promotion_merge"
log="${log_directory}/${NLE_MATRIX_ID}.${NLE_ATTEMPT_TAG}.${SLURM_JOB_ID:?}.log"

if [[ "${source_root}" == *PARIS_inversions* ||
      "${run_root}" == *PARIS_inversions* ]]; then
  echo "Refusing a PARIS_inversions path." >&2
  exit 64
fi
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${NLE_REVISION}" ||
      -n "$(git -C "${source_root}" symbolic-ref -q HEAD || true)" ||
      -n "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]]; then
  echo "NLE_SOURCE must be the exact clean detached revision." >&2
  exit 64
fi
mkdir -p "${log_directory}"
if [[ -e "${log}" || -L "${log}" ]]; then
  echo "Refusing to replace an existing merger log." >&2
  exit 73
fi
exec >"${log}" 2>&1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_parallel_codegen_split_count=1"
export PYTHONPATH="${source_root}${PYTHONPATH:+:${PYTHONPATH}}"
cache_root="${TMPDIR:-/tmp}/score-nle-promotion-merge-${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="${cache_root}/jax"
export XDG_CACHE_HOME="${cache_root}/xdg"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"
mkdir -p \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}"

cd "${source_root}"
exec "${python_bin}" \
  examples/rjmcmc/score_regularized_flow_corrected_promotion_merge.py \
  --matrix-id "${NLE_MATRIX_ID}" \
  --attempt-tag "${NLE_ATTEMPT_TAG}" \
  --source-git-revision "${NLE_REVISION}" \
  --output-root "${run_root}"
