#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=flow-nle-g2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the pre-created immutable run directory.}"
: "${NLE_REVISION:?Set NLE_REVISION to the complete 40-character source SHA.}"
: "${NLE_DRIVER_SHA256:?Set NLE_DRIVER_SHA256 to the committed driver digest.}"
: "${NLE_PROTOCOL_SHA256:?Set NLE_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

regimes=(
  near_gaussian
  near_gaussian
  skewed
  skewed
  boundary_heavy
  boundary_heavy
)
families=(
  two_cell
  four_cell
  two_cell
  four_cell
  two_cell
  four_cell
)
confirmation_seeds=(1877 4099 8317)

if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 18 ]]; then
  echo "The Slurm G2 array must contain exactly 18 tasks." >&2
  exit 2
fi
if [[ "${NLE_SOURCE}" == *PARIS_inversions* || "${NLE_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NLE source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
lock="${NLE_RUN_ROOT}/lock/common-lock.json"
if [[ ! -f "${lock}" || -L "${lock}" ]]; then
  echo "An authenticated G1 common lock is required before G2." >&2
  exit 2
fi
confirmation="${NLE_RUN_ROOT}/confirmation"
log_directory="${NLE_RUN_ROOT}/logs/confirmation"
nle_bin="${NLE_SOURCE}/.pixi/envs/nle-dev/bin"
for directory in "${NLE_SOURCE}" "${NLE_RUN_ROOT}" "${confirmation}" "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${NLE_SOURCE}" rev-parse HEAD)" != "${NLE_REVISION}" ||
      -n "$(git -C "${NLE_SOURCE}" status --porcelain)" ]]; then
  echo "NLE_SOURCE must be clean and match NLE_REVISION." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH="${NLE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${NLE_SOURCE}"
"${nle_bin}/python" -c \
  'import sys; from pathlib import Path; from examples.rjmcmc import conditional_residual_image_flow_confirmation_certify as c; c._authenticate_lock(Path(sys.argv[1]), expected_source_revision=sys.argv[2], expected_driver_sha256=sys.argv[3], expected_protocol_sha256=sys.argv[4])' \
  "${lock}" "${NLE_REVISION}" "${NLE_DRIVER_SHA256}" "${NLE_PROTOCOL_SHA256}"
locked_sample_count="$(
  "${nle_bin}/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["payload"]["locked_sample_count"])' \
    "${lock}"
)"
case_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
regime="${regimes[case_index]}"
family="${families[case_index]}"
base_seed="${confirmation_seeds[seed_index]}"
case_id="${regime}__${family}__root"
stem="${case_id}__S${locked_sample_count}__base${base_seed}"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ -e "${confirmation}/${stem}.flow" ||
      -e "${confirmation}/${stem}.json" ||
      -e "${confirmation}/${stem}.complete.json" ||
      -e "${log}" ]]; then
  echo "Refusing to replace existing G2 evidence for ${stem}." >&2
  exit 2
fi
exec >"${log}" 2>&1

export MPLCONFIGDIR="${TMPDIR:-/tmp}/flow-nle-g2-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/flow-nle-g2-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

"${nle_bin}/python" \
  examples/rjmcmc/conditional_residual_image_flow_tiny_screen.py \
  --profile development \
  --regime "${regime}" \
  --family "${family}" \
  --training-sample-count "${locked_sample_count}" \
  --base-seed "${base_seed}" \
  --source-git-revision "${NLE_REVISION}" \
  --driver-sha256 "${NLE_DRIVER_SHA256}" \
  --output-directory "${confirmation}"
