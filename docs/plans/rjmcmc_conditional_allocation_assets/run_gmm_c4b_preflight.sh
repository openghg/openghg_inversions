#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${GMM_SOURCE:?Set GMM_SOURCE to the clean full-SHA worktree.}"
: "${GMM_RUN_ROOT:?Set GMM_RUN_ROOT to the pre-created immutable run directory.}"
: "${GMM_REVISION:?Set GMM_REVISION to the complete 40-character source SHA.}"

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
preflight="${GMM_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
smoke="${preflight}/smoke.json"
complete="${preflight}/PREFLIGHT_COMPLETE.txt"

if [[ "${#GMM_REVISION}" -ne 40 || ! "${GMM_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "GMM_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${GMM_SOURCE}" == *PARIS_inversions* || "${GMM_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "C4b source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
if [[ ! -d "${GMM_SOURCE}" || -L "${GMM_SOURCE}" ]]; then
  echo "GMM_SOURCE must be a real worktree directory." >&2
  exit 2
fi
if [[ ! -d "${GMM_RUN_ROOT}" || -L "${GMM_RUN_ROOT}" ||
      ! -d "${preflight}" || -L "${preflight}" ]]; then
  echo "GMM_RUN_ROOT and its preflight directory must be pre-created real directories." >&2
  exit 2
fi
if [[ -e "${log}" || -L "${log}" || -e "${smoke}" || -L "${smoke}" ||
      -e "${complete}" || -L "${complete}" ]]; then
  echo "Refusing to replace existing preflight evidence." >&2
  exit 2
fi
if [[ "$(git -C "${GMM_SOURCE}" rev-parse HEAD)" != "${GMM_REVISION}" ]]; then
  echo "GMM_SOURCE does not match GMM_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${GMM_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "GMM_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${GMM_SOURCE}/.pixi" ||
      "$(readlink -f "${GMM_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "GMM_SOURCE/.pixi must be a symlink to the canonical BP1 environment." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${GMM_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/gmm-c4b-preflight-matplotlib-${GMM_REVISION}"
export NUMBA_CACHE_DIR="/tmp/gmm-c4b-preflight-numba-${GMM_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: > "${log}"
exec > >(tee -a "${log}") 2>&1

cd "${GMM_SOURCE}"
echo "revision=${GMM_REVISION}"
echo "head=$(git rev-parse HEAD)"
pixi --version
pixi run --frozen --no-install -e dev \
  python -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
echo "focused_pytest_begin"
pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mdn.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_gmm_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_gmm_certify.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_gmm_protected_certify.py
echo "focused_pytest_pass"
echo "focused_ruff_begin"
pixi run --frozen --no-install -e dev ruff check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mdn.py \
  examples/rjmcmc/conditional_residual_image_gmm_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_gmm_certify.py \
  examples/rjmcmc/conditional_residual_image_gmm_protected_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mdn.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_gmm_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_gmm_certify.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_gmm_protected_certify.py
echo "focused_ruff_pass"
echo "focused_pyright_begin"
pixi run --frozen --no-install -e dev pyright \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mdn.py \
  examples/rjmcmc/conditional_residual_image_gmm_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_gmm_certify.py \
  examples/rjmcmc/conditional_residual_image_gmm_protected_certify.py
echo "focused_pyright_pass"
echo "smoke_begin"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_gmm_tiny_screen.py \
  --profile smoke \
  --case-id near_gaussian__two_cell__root \
  --source-revision "${GMM_REVISION}" \
  --output "${smoke}"
echo "smoke_pass"

printf 'GMM C4b preflight complete for %s\n' "${GMM_REVISION}" > "${complete}"
