#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the pre-created immutable run directory.}"
: "${NLE_REVISION:?Set NLE_REVISION to the complete 40-character source SHA.}"
: "${NLE_DRIVER_SHA256:?Set NLE_DRIVER_SHA256 to the committed driver digest.}"
: "${NLE_PROTOCOL_SHA256:?Set NLE_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

confirmation="${NLE_RUN_ROOT}/confirmation"
lock="${NLE_RUN_ROOT}/lock/common-lock.json"
certificate_directory="${NLE_RUN_ROOT}/confirmation-certificate"
log="${NLE_RUN_ROOT}/logs/certify-confirmation.log"
nle_bin="${NLE_SOURCE}/.pixi/envs/nle-dev/bin"
if [[ "${NLE_SOURCE}" == *PARIS_inversions* || "${NLE_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NLE source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
if [[ ! -f "${lock}" || -L "${lock}" ]]; then
  echo "An authenticated G1 common lock is required before G2 certification." >&2
  exit 2
fi
for directory in "${NLE_SOURCE}" "${NLE_RUN_ROOT}" "${confirmation}" "${certificate_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ -e "${log}" ||
      -e "${certificate_directory}/confirmation-certificate.json" ||
      -e "${certificate_directory}/HOLDOUT_ELIGIBLE.json" ||
      -e "${certificate_directory}/CONFIRMATION_COMPLETE.json" ]]; then
  echo "Refusing to replace existing G2 certification evidence." >&2
  exit 2
fi
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

: > "${log}"
exec > >(tee -a "${log}") 2>&1
cd "${NLE_SOURCE}"
"${nle_bin}/python" \
  examples/rjmcmc/conditional_residual_image_flow_confirmation_certify.py \
  --input-directory "${confirmation}" \
  --lock-path "${lock}" \
  --output-directory "${certificate_directory}" \
  --expected-source-revision "${NLE_REVISION}" \
  --expected-driver-sha256 "${NLE_DRIVER_SHA256}" \
  --expected-protocol-sha256 "${NLE_PROTOCOL_SHA256}"
