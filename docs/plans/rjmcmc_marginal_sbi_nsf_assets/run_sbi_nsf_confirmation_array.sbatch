#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=sbi-nsf-g2
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
confirmation_seeds=(1877 4099 8317)

if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne 18 ]]; then
  echo "The Slurm G2 array must contain exactly 18 tasks." >&2
  exit 2
fi
lock="${NSF_RUN_ROOT}/lock/common-lock.json"
if [[ ! -f "${lock}" || -L "${lock}" ]]; then
  echo "An authenticated G1 common lock is required before G2." >&2
  exit 2
fi
if [[ "${NSF_SOURCE}" == *PARIS_inversions* ||
      "${NSF_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NSF source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
confirmation="${NSF_RUN_ROOT}/confirmation"
log_directory="${NSF_RUN_ROOT}/logs/confirmation"
nsf_bin="${NSF_SOURCE}/.pixi/envs/nle-dev/bin"
for directory in "${NSF_SOURCE}" "${NSF_RUN_ROOT}" "${confirmation}" \
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NSF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sbi-nsf-g2-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/sbi-nsf-g2-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${NSF_SOURCE}"
"${nsf_bin}/python" -c \
  'import sys; from pathlib import Path; from examples.rjmcmc import conditional_residual_image_sbi_nsf_confirmation_certify as c; c._authenticate_lock(Path(sys.argv[1]), expected_source_revision=sys.argv[2], expected_driver_sha256=sys.argv[3], expected_protocol_sha256=sys.argv[4])' \
  "${lock}" "${NSF_REVISION}" "${NSF_DRIVER_SHA256}" "${NSF_PROTOCOL_SHA256}"
locked_sample_count="$(
  "${nsf_bin}/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["payload"]["locked_sample_count"])' \
    "${lock}"
)"
case_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
regime="${regimes[case_index]}"
family="${families[case_index]}"
base_seed="${confirmation_seeds[seed_index]}"
stem="${regime}__${family}__root__S${locked_sample_count}__base${base_seed}"
log="${log_directory}/${stem}.${SLURM_JOB_ID}.log"
if [[ -e "${confirmation}/${stem}.nsf" ||
      -e "${confirmation}/${stem}.json" ||
      -e "${confirmation}/${stem}.complete.json" ||
      -e "${log}" ]]; then
  echo "Refusing to replace existing G2 evidence for ${stem}." >&2
  exit 2
fi
exec >"${log}" 2>&1

"${nsf_bin}/python" \
  examples/rjmcmc/conditional_residual_image_sbi_nsf_tiny_screen.py \
  --profile development \
  --regime "${regime}" \
  --family "${family}" \
  --training-sample-count "${locked_sample_count}" \
  --base-seed "${base_seed}" \
  --source-git-revision "${NSF_REVISION}" \
  --driver-sha256 "${NSF_DRIVER_SHA256}" \
  --output-directory "${confirmation}"
