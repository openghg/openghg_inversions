#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=score-nle-n1-merge
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

module load git/2.45.1-pqk5

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the immutable run directory.}"
: "${NLE_REVISION:?Set NLE_REVISION to the complete 40-character source SHA.}"
: "${NLE_DRIVER_SHA256:?Set NLE_DRIVER_SHA256 to the committed driver digest.}"
: "${NLE_PROTOCOL_SHA256:?Set NLE_PROTOCOL_SHA256 to the frozen protocol digest.}"

driver="${NLE_SOURCE}/examples/rjmcmc/score_regularized_flow_tiny_screen.py"
certifier="${NLE_SOURCE}/examples/rjmcmc/score_regularized_flow_tiny_certify.py"
development="${NLE_RUN_ROOT}/development"
merge_directory="${NLE_RUN_ROOT}/development-merge"
log_directory="${NLE_RUN_ROOT}/logs/development-merge"
log="${log_directory}/N1_merge.${SLURM_JOB_ID:?N1 merge must run under Slurm.}.log"

validate_hex_digest() {
  local name="$1"
  local value="${!name}"
  local length="$2"
  if [[ "${#value}" -ne "${length}" || ! "${value}" =~ ^[0-9a-f]+$ ]]; then
    echo "${name} must be a ${length}-character lower-case hexadecimal digest." >&2
    exit 2
  fi
}

validate_real_directory() {
  local directory="$1"
  if [[ ! -d "${directory}" || -L "${directory}" ||
        "$(realpath -e "${directory}")" != "${directory}" ]]; then
    echo "Required canonical real directory is absent: ${directory}" >&2
    exit 2
  fi
}

validate_hex_digest NLE_REVISION 40
validate_hex_digest NLE_DRIVER_SHA256 64
validate_hex_digest NLE_PROTOCOL_SHA256 64
if [[ "${NLE_SOURCE}" == *PARIS_inversions* ||
      "${NLE_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NLE source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in \
  "${NLE_SOURCE}" \
  "${NLE_RUN_ROOT}" \
  "${development}" \
  "${merge_directory}" \
  "${log_directory}"
do
  validate_real_directory "${directory}"
done
if [[ ! -f "${driver}" || -L "${driver}" ||
      ! -f "${certifier}" || -L "${certifier}" ]]; then
  echo "The committed driver and certifier must be real files." >&2
  exit 2
fi
if [[ ! -d "${NLE_SOURCE}/.pixi/envs/nle-dev" ||
      -L "${NLE_SOURCE}/.pixi" ||
      "$(realpath -e "${NLE_SOURCE}/.pixi")" != "${NLE_SOURCE}/.pixi" ]]; then
  echo "The detached source must own its locked real nle-dev environment." >&2
  exit 2
fi
if [[ "$(git -C "${NLE_SOURCE}" rev-parse --verify HEAD)" != "${NLE_REVISION}" ||
      "$(git -C "${NLE_SOURCE}" rev-parse --verify "${NLE_REVISION}^{commit}")" != "${NLE_REVISION}" ]]; then
  echo "NLE_SOURCE does not match the full NLE_REVISION." >&2
  exit 2
fi
if git -C "${NLE_SOURCE}" symbolic-ref -q HEAD >/dev/null; then
  echo "NLE_SOURCE must be detached." >&2
  exit 2
fi
if [[ -n "$(git -C "${NLE_SOURCE}" status --porcelain --untracked-files=all)" ]]; then
  echo "NLE_SOURCE must be clean." >&2
  exit 2
fi
if [[ "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  echo "The committed driver digest does not match." >&2
  exit 2
fi
if [[ -n "$(find "${merge_directory}" -mindepth 1 -maxdepth 1 -print -quit)" ||
      -n "$(find "${log_directory}" -mindepth 1 -maxdepth 1 -print -quit)" ||
      -e "${log}" || -L "${log}" ]]; then
  echo "Refusing to replace existing N1 merge evidence." >&2
  exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-1}" -ne 1 ]]; then
  echo "N1 merger requires exactly one Slurm CPU." >&2
  exit 2
fi

exec >"${log}" 2>&1

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH="${NLE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
cache_root="${TMPDIR:-/tmp}/score-nle-n1-merge-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${cache_root}/xdg"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export JAX_COMPILATION_CACHE_DIR="${cache_root}/jax"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}"

cd "${NLE_SOURCE}"
pixi run --frozen --no-install -e nle-dev \
  python examples/rjmcmc/score_regularized_flow_tiny_certify.py \
  --input-directory "${development}" \
  --output-directory "${merge_directory}" \
  --expected-source-revision "${NLE_REVISION}" \
  --expected-driver-sha256 "${NLE_DRIVER_SHA256}" \
  --expected-protocol-sha256 "${NLE_PROTOCOL_SHA256}"

if [[ -n "$(git -C "${NLE_SOURCE}" status --porcelain --untracked-files=all)" ||
      "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  echo "NLE_SOURCE changed while the N1 merger was running." >&2
  exit 2
fi
if [[ ! -f "${merge_directory}/development-certificate.json" ||
      -L "${merge_directory}/development-certificate.json" ||
      ! -f "${merge_directory}/MERGE_COMPLETE.json" ||
      -L "${merge_directory}/MERGE_COMPLETE.json" ]]; then
  echo "N1 certifier did not publish its certificate and final marker." >&2
  exit 2
fi
# The certifier, not this wrapper, publishes MERGE_COMPLETE.json last.
