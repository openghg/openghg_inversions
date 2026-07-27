#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=sbi-nsf-g1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${NSF_SOURCE:?Set NSF_SOURCE to the clean detached full-SHA worktree.}"
: "${NSF_RUN_ROOT:?Set NSF_RUN_ROOT to the fresh immutable run directory.}"
: "${NSF_REVISION:?Set NSF_REVISION to the complete source SHA.}"
: "${NSF_DRIVER_SHA256:?Set NSF_DRIVER_SHA256 to the committed driver digest.}"
: "${NSF_PROTOCOL_SHA256:?Set NSF_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

regimes=(near_gaussian near_gaussian skewed skewed boundary_heavy boundary_heavy)
families=(two_cell four_cell two_cell four_cell two_cell four_cell)
sample_counts=(4096 16384 65536 262144)

if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 24 ]]; then
  echo "The Slurm G1 array must contain exactly 24 tasks." >&2
  exit 2
fi
if [[ "${NSF_SOURCE}" == *PARIS_inversions* ||
      "${NSF_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NSF source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
development="${NSF_RUN_ROOT}/development"
log_directory="${NSF_RUN_ROOT}/logs/development"
nsf_bin="${NSF_SOURCE}/.pixi/envs/nle-dev/bin"
for directory in "${NSF_SOURCE}" "${NSF_RUN_ROOT}" "${development}" \
  "${log_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${NSF_SOURCE}" rev-parse HEAD)" != "${NSF_REVISION}" ||
      -n "$(git -C "${NSF_SOURCE}" status --porcelain)" ]]; then
  echo "NSF_SOURCE must be clean and match NSF_REVISION." >&2
  exit 2
fi
preflight="${NSF_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.json"
if [[ ! -f "${preflight}" || -L "${preflight}" ]]; then
  echo "The cross-node G0 completion marker is absent." >&2
  exit 2
fi

case_index=$((SLURM_ARRAY_TASK_ID / 4))
size_index=$((SLURM_ARRAY_TASK_ID % 4))
regime="${regimes[case_index]}"
family="${families[case_index]}"
sample_count="${sample_counts[size_index]}"
stem="${regime}__${family}__root__S${sample_count}__base731"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ -e "${development}/${stem}.nsf" ||
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
export PYTHONPATH="${NSF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sbi-nsf-g1-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/sbi-nsf-g1-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${NSF_SOURCE}"
"${nsf_bin}/python" \
  examples/rjmcmc/conditional_residual_image_sbi_nsf_tiny_screen.py \
  --profile development \
  --regime "${regime}" \
  --family "${family}" \
  --training-sample-count "${sample_count}" \
  --base-seed 731 \
  --source-git-revision "${NSF_REVISION}" \
  --driver-sha256 "${NSF_DRIVER_SHA256}" \
  --output-directory "${development}"
