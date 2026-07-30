#!/usr/bin/env bash
#SBATCH --job-name=score-nle-oracle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

module load git/2.45.1-pqk5

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the fresh corrected run root.}"
: "${NLE_REVISION:?Set NLE_REVISION to the exact 40-character source SHA.}"

source_root="$(realpath -e "${NLE_SOURCE}")"
run_root="$(realpath -e "${NLE_RUN_ROOT}")"
python_bin="${source_root}/.pixi/envs/nle-dev/bin/python"
log_directory="${run_root}/logs/oracle"
log="${log_directory}/oracle.${SLURM_JOB_ID:?oracle must run under Slurm}.log"

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
if [[ ! -x "${python_bin}" ]]; then
  echo "The frozen nle-dev Python is missing." >&2
  exit 64
fi
if [[ -e "${run_root}/oracle" || -L "${run_root}/oracle" ]]; then
  echo "Refusing to replace existing oracle evidence." >&2
  exit 73
fi
mkdir -p "${log_directory}"
if [[ -e "${log}" || -L "${log}" ]]; then
  echo "Refusing to replace an existing oracle log." >&2
  exit 73
fi
exec >"${log}" 2>&1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export PYTHONPATH="${source_root}${PYTHONPATH:+:${PYTHONPATH}}"
cache_root="${TMPDIR:-/tmp}/score-nle-oracle-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${cache_root}/xdg"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}"

cd "${source_root}"
exec "${python_bin}" \
  examples/rjmcmc/score_regularized_flow_corrected_oracle.py \
  --source-git-revision "${NLE_REVISION}" \
  --output-root "${run_root}"
