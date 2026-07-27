#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${NQ_SOURCE:?Set NQ_SOURCE to the clean detached full-SHA worktree.}"
: "${NQ_RUN_ROOT:?Set NQ_RUN_ROOT to the fresh immutable run directory.}"
: "${NQ_REVISION:?Set NQ_REVISION to the complete source SHA.}"
: "${NQ_DRIVER_SHA256:?Set NQ_DRIVER_SHA256 to the committed driver digest.}"
: "${NQ_PROTOCOL_SHA256:?Set NQ_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

confirmation="${NQ_RUN_ROOT}/confirmation"
lock="${NQ_RUN_ROOT}/lock/common-lock.json"
certificate_directory="${NQ_RUN_ROOT}/confirmation-certificate"
log="${NQ_RUN_ROOT}/logs/certify-confirmation.log"
nq_bin="${NQ_SOURCE}/.pixi/envs/nle-dev/bin"
if [[ "${NQ_SOURCE}" == *PARIS_inversions* ||
      "${NQ_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Native-quadrature paths must not be placed in PARIS_inversions." >&2
  exit 2
fi
if [[ ! -f "${lock}" || -L "${lock}" ]]; then
  echo "An authenticated G1 common lock is required before G2 certification." >&2
  exit 2
fi
for directory in "${NQ_SOURCE}" "${NQ_RUN_ROOT}" "${confirmation}" \
  "${certificate_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ -e "${log}" ||
      -e "${certificate_directory}/confirmation-certificate.json" ||
      -e "${certificate_directory}/G2_COMPLETE.json" ]]; then
  echo "Refusing to replace existing G2 certificate evidence." >&2
  exit 2
fi
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/native-quad-g2-cert-mpl-${SLURM_JOB_ID:-local}"

: > "${log}"
exec > >(tee -a "${log}") 2>&1
cd "${NQ_SOURCE}"
"${nq_bin}/python" \
  examples/rjmcmc/conditional_native_quadrature_confirmation_certify.py \
  --input-directory "${confirmation}" \
  --lock "${lock}" \
  --output-directory "${certificate_directory}" \
  --expected-source-revision "${NQ_REVISION}" \
  --expected-driver-sha256 "${NQ_DRIVER_SHA256}" \
  --expected-protocol-sha256 "${NQ_PROTOCOL_SHA256}"
