#!/usr/bin/env bash
#SBATCH --job-name=score-nle-corrected
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00

set -euo pipefail

module load git/2.45.1-pqk5

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the fresh corrected run root.}"
: "${NLE_REVISION:?Set NLE_REVISION to the exact 40-character source SHA.}"
: "${NLE_MATRIX_ID:?Set NLE_MATRIX_ID to one committed array matrix.}"
: "${NLE_ATTEMPT_TAG:?Set NLE_ATTEMPT_TAG to a fresh immutable tag.}"
: "${NLE_MAX_EPOCHS:?Set NLE_MAX_EPOCHS to the measured epoch budget.}"
: "${NLE_PATIENCE:?Set NLE_PATIENCE to the exploratory patience.}"

source_root="$(realpath -e "${NLE_SOURCE}")"
run_root="$(realpath -e "${NLE_RUN_ROOT}")"
python_bin="${source_root}/.pixi/envs/nle-dev/bin/python"
task_id="${SLURM_ARRAY_TASK_ID:?Corrected NLE matrices must run as a SLURM array.}"
array_job_id="${SLURM_ARRAY_JOB_ID:?Corrected NLE matrices must run as a SLURM array.}"
log_directory="${run_root}/logs/${NLE_MATRIX_ID}"
log="${log_directory}/${NLE_ATTEMPT_TAG}.${array_job_id}_${task_id}.log"
oracle_bundle="${run_root}/oracle/oracle_bundle.json"
oracle_complete="${run_root}/oracle/COMPLETE.json"

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
if [[ ! -x "${python_bin}" ||
      ! -f "${oracle_bundle}" ||
      ! -f "${oracle_complete}" ]]; then
  echo "The nle-dev Python and completed oracle bundle are required." >&2
  exit 64
fi
mkdir -p "${log_directory}"
if [[ -e "${log}" || -L "${log}" ]]; then
  echo "Refusing to replace an existing array-task log." >&2
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
cache_root="${TMPDIR:-/tmp}/score-nle-e2-${array_job_id}-${task_id}"
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
  examples/rjmcmc/score_regularized_flow_corrected_array.py \
  --matrix-id "${NLE_MATRIX_ID}" \
  --attempt-tag "${NLE_ATTEMPT_TAG}" \
  --source-git-revision "${NLE_REVISION}" \
  --oracle-bundle "${oracle_bundle}" \
  --output-root "${run_root}" \
  --max-epochs "${NLE_MAX_EPOCHS}" \
  --patience "${NLE_PATIENCE}"
