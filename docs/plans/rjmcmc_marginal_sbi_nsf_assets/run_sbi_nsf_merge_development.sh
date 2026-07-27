#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${NSF_SOURCE:?Set NSF_SOURCE to the clean detached full-SHA worktree.}"
: "${NSF_RUN_ROOT:?Set NSF_RUN_ROOT to the fresh immutable run directory.}"
: "${NSF_REVISION:?Set NSF_REVISION to the complete source SHA.}"
: "${NSF_DRIVER_SHA256:?Set NSF_DRIVER_SHA256 to the committed driver digest.}"
: "${NSF_PROTOCOL_SHA256:?Set NSF_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

development="${NSF_RUN_ROOT}/development"
lock_directory="${NSF_RUN_ROOT}/lock"
log="${NSF_RUN_ROOT}/logs/merge-development.log"
nsf_bin="${NSF_SOURCE}/.pixi/envs/nle-dev/bin"
if [[ "${NSF_SOURCE}" == *PARIS_inversions* ||
      "${NSF_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NSF source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${NSF_SOURCE}" "${NSF_RUN_ROOT}" "${development}" \
  "${lock_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ -e "${log}" || -e "${lock_directory}/development-certificate.json" ||
      -e "${lock_directory}/common-lock.json" ||
      -e "${lock_directory}/MERGE_COMPLETE.json" ]]; then
  echo "Refusing to replace existing G1 merger evidence." >&2
  exit 2
fi
if [[ "$(git -C "${NSF_SOURCE}" rev-parse HEAD)" != "${NSF_REVISION}" ||
      -n "$(git -C "${NSF_SOURCE}" status --porcelain)" ]]; then
  echo "NSF_SOURCE must be clean and match NSF_REVISION." >&2
  exit 2
fi
if [[ ! -f "${NSF_RUN_ROOT}/preflight/PREFLIGHT_COMPLETE.json" ]]; then
  echo "The cross-node G0 completion marker is absent." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NSF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sbi-nsf-g1-merge-mpl-${SLURM_JOB_ID:-local}"

: > "${log}"
exec > >(tee -a "${log}") 2>&1
cd "${NSF_SOURCE}"
"${nsf_bin}/python" \
  examples/rjmcmc/conditional_residual_image_sbi_nsf_certify.py \
  --input-directory "${development}" \
  --output-directory "${lock_directory}" \
  --expected-source-revision "${NSF_REVISION}" \
  --expected-driver-sha256 "${NSF_DRIVER_SHA256}" \
  --expected-protocol-sha256 "${NSF_PROTOCOL_SHA256}"
