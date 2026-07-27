#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=native-quad-g2
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
confirmation_seeds=(1877 4099 8317)

if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 18 ]]; then
  echo "The Slurm G2 array must contain exactly 18 tasks." >&2
  exit 2
fi
lock="${NQ_RUN_ROOT}/lock/common-lock.json"
if [[ ! -f "${lock}" || -L "${lock}" ]]; then
  echo "An authenticated G1 common lock is required before G2." >&2
  exit 2
fi
if [[ "${NQ_SOURCE}" == *PARIS_inversions* ||
      "${NQ_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Native-quadrature paths must not be placed in PARIS_inversions." >&2
  exit 2
fi
confirmation="${NQ_RUN_ROOT}/confirmation"
development="${NQ_RUN_ROOT}/development"
log_directory="${NQ_RUN_ROOT}/logs/confirmation"
nq_bin="${NQ_SOURCE}/.pixi/envs/nle-dev/bin"
for directory in "${NQ_SOURCE}" "${NQ_RUN_ROOT}" "${confirmation}" \
  "${development}" "${log_directory}"; do
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NQ_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/native-quad-g2-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/native-quad-g2-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${NQ_SOURCE}"
"${nq_bin}/python" -c \
  'import sys; from pathlib import Path; from examples.rjmcmc import conditional_native_quadrature_confirmation as c; c._authenticate_lock(Path(sys.argv[1]), expected_source_revision=sys.argv[2], expected_driver_sha256=sys.argv[3], expected_protocol_sha256=sys.argv[4])' \
  "${lock}" "${NQ_REVISION}" "${NQ_DRIVER_SHA256}" "${NQ_PROTOCOL_SHA256}"
locked_order="$(
  "${nq_bin}/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["payload"]["locked_order"])' \
    "${lock}"
)"
case_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
regime="${regimes[case_index]}"
family="${families[case_index]}"
base_seed="${confirmation_seeds[seed_index]}"
stem="${regime}__${family}__root__O${locked_order}__base${base_seed}"
artifact="${development}/${regime}__${family}__root__O${locked_order}__base731.nq"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ ! -f "${artifact}" || -L "${artifact}" ]]; then
  echo "The G1 selected artifact is absent or unsafe: ${artifact}" >&2
  exit 2
fi
if [[ -e "${confirmation}/${stem}.json" ||
      -e "${confirmation}/${stem}.complete.json" ||
      -e "${log}" ]]; then
  echo "Refusing to replace existing G2 evidence for ${stem}." >&2
  exit 2
fi
exec >"${log}" 2>&1

"${nq_bin}/python" \
  examples/rjmcmc/conditional_native_quadrature_confirmation.py \
  --regime "${regime}" \
  --family "${family}" \
  --base-seed "${base_seed}" \
  --lock "${lock}" \
  --artifact "${artifact}" \
  --output-directory "${confirmation}" \
  --expected-source-revision "${NQ_REVISION}" \
  --expected-driver-sha256 "${NQ_DRIVER_SHA256}" \
  --expected-protocol-sha256 "${NQ_PROTOCOL_SHA256}"
