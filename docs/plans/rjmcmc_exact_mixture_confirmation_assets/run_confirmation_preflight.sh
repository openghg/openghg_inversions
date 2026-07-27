#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${CONF_SOURCE:?Set CONF_SOURCE to the clean full-SHA worktree.}"
: "${CONF_RUN_ROOT:?Set CONF_RUN_ROOT to the fresh confirmation run root.}"
: "${CONF_REVISION:?Set CONF_REVISION to the complete candidate SHA.}"
: "${DEV_RUN_ROOT:?Set DEV_RUN_ROOT to the immutable d23 development run.}"

module load git/2.45.1-pqk5

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
expected_dev_root="/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_compression/d23e9d9b5b7d8c4e669ee940ab544fa8dc5148ea"
source_decision="${DEV_RUN_ROOT}/decision/common-source-decision.json"
compression_decision="${DEV_RUN_ROOT}/decision/common-compression-decision.json"
preflight="${CONF_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
validation="${preflight}/development-validation.json"
complete="${preflight}/PREFLIGHT_COMPLETE.txt"

if [[ "${DEV_RUN_ROOT}" != "${expected_dev_root}" ]]; then
  echo "DEV_RUN_ROOT is not the frozen d23 development run." >&2
  exit 2
fi
if [[ "${#CONF_REVISION}" -ne 40 || ! "${CONF_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "CONF_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${CONF_SOURCE}" == *PARIS_inversions* || "${CONF_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Confirmation source and output must not use PARIS_inversions." >&2
  exit 2
fi
for directory in "${CONF_SOURCE}" "${CONF_RUN_ROOT}" "${DEV_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
for path in "${log}" "${validation}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace confirmation preflight evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${CONF_SOURCE}" rev-parse HEAD)" != "${CONF_REVISION}" ]]; then
  echo "CONF_SOURCE does not match CONF_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${CONF_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "CONF_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${CONF_SOURCE}/.pixi" ||
      "$(readlink -f "${CONF_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "CONF_SOURCE/.pixi must point to the canonical BP1 environment." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${CONF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/exact-mixture-confirm-preflight-mpl-${CONF_REVISION}"
export NUMBA_CACHE_DIR="/tmp/exact-mixture-confirm-preflight-numba-${CONF_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: >"${log}"
exec > >(tee -a "${log}") 2>&1

cd "${CONF_SOURCE}"
echo "revision=${CONF_REVISION}"
pixi --version
pixi run --frozen --no-install -e dev \
  python -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'

pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture_confirm.py

pixi run --frozen --no-install -e dev ruff format --check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture_confirm.py
pixi run --frozen --no-install -e dev ruff check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture_confirm.py
pixi run --frozen --no-install -e dev pyright \
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture_confirm.py

pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm.py \
  validate-development \
  --development-source-decision "${source_decision}" \
  --development-compression-decision "${compression_decision}" \
  >"${validation}"

printf 'Exact-mixture confirmation preflight complete for %s\n' \
  "${CONF_REVISION}" >"${complete}"
