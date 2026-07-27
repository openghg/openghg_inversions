#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=flow-nle-g1
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
sample_counts=(4096 16384 65536 262144)

if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 24 ]]; then
  echo "The Slurm G1 array must contain exactly 24 tasks." >&2
  exit 2
fi
if [[ "${NLE_SOURCE}" == *PARIS_inversions* || "${NLE_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NLE source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
development="${NLE_RUN_ROOT}/development"
log_directory="${NLE_RUN_ROOT}/logs/development"
for directory in "${NLE_SOURCE}" "${NLE_RUN_ROOT}" "${development}" "${log_directory}"; do
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
preflight="${NLE_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.json"
if [[ ! -f "${preflight}" || -L "${preflight}" ]]; then
  echo "The G0 preflight completion marker is absent." >&2
  exit 2
fi

case_index=$((SLURM_ARRAY_TASK_ID / 4))
size_index=$((SLURM_ARRAY_TASK_ID % 4))
regime="${regimes[case_index]}"
family="${families[case_index]}"
sample_count="${sample_counts[size_index]}"
case_id="${regime}__${family}__root"
stem="${case_id}__S${sample_count}__base731"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ -e "${development}/${stem}.flow" ||
      -e "${development}/${stem}.json" ||
      -e "${development}/${stem}.complete.json" ||
      -e "${log}" ]]; then
  echo "Refusing to replace existing G1 evidence for ${stem}." >&2
  exit 2
fi
exec >"${log}" 2>&1

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH="${NLE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/flow-nle-g1-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/flow-nle-g1-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${NLE_SOURCE}"
pixi run --frozen --no-install -e nle-dev \
  python examples/rjmcmc/conditional_residual_image_flow_tiny_screen.py \
  --profile development \
  --regime "${regime}" \
  --family "${family}" \
  --training-sample-count "${sample_count}" \
  --base-seed 731 \
  --source-git-revision "${NLE_REVISION}" \
  --driver-sha256 "${NLE_DRIVER_SHA256}" \
  --output-directory "${development}"
