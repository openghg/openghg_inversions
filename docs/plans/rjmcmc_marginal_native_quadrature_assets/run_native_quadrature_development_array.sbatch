#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=native-quad-g1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${NQ_SOURCE:?Set NQ_SOURCE to the clean detached full-SHA worktree.}"
: "${NQ_RUN_ROOT:?Set NQ_RUN_ROOT to the fresh immutable run directory.}"
: "${NQ_REVISION:?Set NQ_REVISION to the complete source SHA.}"
: "${NQ_DRIVER_SHA256:?Set NQ_DRIVER_SHA256 to the committed driver digest.}"
: "${NQ_PROTOCOL_SHA256:?Set NQ_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

regimes=(near_gaussian near_gaussian skewed skewed boundary_heavy boundary_heavy)
families=(two_cell four_cell two_cell four_cell two_cell four_cell)
orders=(24 32 40 48)

if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 24 ]]; then
  echo "The Slurm G1 array must contain exactly 24 tasks." >&2
  exit 2
fi
if [[ "${NQ_SOURCE}" == *PARIS_inversions* ||
      "${NQ_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Native-quadrature paths must not be placed in PARIS_inversions." >&2
  exit 2
fi
development="${NQ_RUN_ROOT}/development"
log_directory="${NQ_RUN_ROOT}/logs/development"
nq_bin="${NQ_SOURCE}/.pixi/envs/nle-dev/bin"
for directory in "${NQ_SOURCE}" "${NQ_RUN_ROOT}" "${development}" \
  "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${NQ_SOURCE}" rev-parse HEAD)" != "${NQ_REVISION}" ||
      -n "$(git -C "${NQ_SOURCE}" status --porcelain)" ]]; then
  echo "NQ_SOURCE must be clean and match NQ_REVISION." >&2
  exit 2
fi
preflight="${NQ_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.json"
if [[ ! -f "${preflight}" || -L "${preflight}" ]]; then
  echo "The cross-node G0 completion marker is absent." >&2
  exit 2
fi

case_index=$((SLURM_ARRAY_TASK_ID / 4))
order_index=$((SLURM_ARRAY_TASK_ID % 4))
regime="${regimes[case_index]}"
family="${families[case_index]}"
order="${orders[order_index]}"
stem="${regime}__${family}__root__O${order}__base731"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ -e "${development}/${stem}.nq" ||
      -e "${development}/${stem}.json" ||
      -e "${development}/${stem}.complete.json" ||
      -e "${log}" ]]; then
  echo "Refusing to replace existing G1 evidence for ${stem}." >&2
  exit 2
fi
exec >"${log}" 2>&1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NQ_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/native-quad-g1-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/native-quad-g1-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${NQ_SOURCE}"
"${nq_bin}/python" \
  examples/rjmcmc/conditional_native_quadrature_tiny_screen.py \
  --profile development \
  --regime "${regime}" \
  --family "${family}" \
  --quadrature-order "${order}" \
  --base-seed 731 \
  --source-git-revision "${NQ_REVISION}" \
  --driver-sha256 "${NQ_DRIVER_SHA256}" \
  --output-directory "${development}"
