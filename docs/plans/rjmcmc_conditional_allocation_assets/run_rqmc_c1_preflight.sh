#!/usr/bin/env bash

set -euo pipefail

: "${RQMC_SOURCE:?Set RQMC_SOURCE to the clean full-SHA worktree.}"
: "${RQMC_RUN_ROOT:?Set RQMC_RUN_ROOT to the immutable run directory.}"
: "${RQMC_REVISION:?Set RQMC_REVISION to the complete 40-character source SHA.}"

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
preflight="${RQMC_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
complete="${preflight}/PREFLIGHT_COMPLETE.txt"
smoke="${preflight}/smoke.json"

if [[ "${#RQMC_REVISION}" -ne 40 || ! "${RQMC_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "RQMC_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ ! -d "${RQMC_SOURCE}" || ! -d "${preflight}" ]]; then
  echo "RQMC_SOURCE and the pre-created preflight directory must exist." >&2
  exit 2
fi
if [[ -e "${log}" || -e "${complete}" || -e "${smoke}" ]]; then
  echo "Refusing to replace existing preflight evidence." >&2
  exit 2
fi
if [[ "$(git -C "${RQMC_SOURCE}" rev-parse HEAD)" != "${RQMC_REVISION}" ]]; then
  echo "RQMC_SOURCE does not match RQMC_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${RQMC_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "RQMC_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ "$(readlink -f "${RQMC_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "RQMC_SOURCE/.pixi must resolve to the canonical BP1 environment." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/rqmc-c1-preflight-matplotlib-${RQMC_REVISION}"
export NUMBA_CACHE_DIR="/tmp/rqmc-c1-preflight-numba-${RQMC_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

exec > >(tee "${log}") 2>&1

cd "${RQMC_SOURCE}"
echo "revision=${RQMC_REVISION}"
echo "head=$(git rev-parse HEAD)"
echo "status_porcelain_begin"
git status --porcelain
echo "status_porcelain_end"
pixi --version
pixi run --frozen --no-install -e dev \
  python -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
echo "focused_pytest_begin"
pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_allocation_likelihood_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_allocation_likelihood_rqmc_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_allocation_likelihood_rqmc_certify.py
echo "focused_pytest_pass"
echo "focused_ruff_begin"
pixi run --frozen --no-install -e dev ruff check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mixture.py \
  examples/rjmcmc/conditional_allocation_likelihood_rqmc_tiny_screen.py \
  examples/rjmcmc/conditional_allocation_likelihood_rqmc_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_allocation_likelihood_rqmc_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_allocation_likelihood_rqmc_certify.py
echo "focused_ruff_pass"
echo "focused_pyright_begin"
pixi run --frozen --no-install -e dev pyright \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mixture.py \
  examples/rjmcmc/conditional_allocation_likelihood_rqmc_tiny_screen.py \
  examples/rjmcmc/conditional_allocation_likelihood_rqmc_certify.py
echo "focused_pyright_pass"
echo "smoke_begin"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_allocation_likelihood_rqmc_tiny_screen.py \
  --profile smoke \
  --case-id near_gaussian__two_cell__root \
  --sample-counts 64 \
  --repeat-seeds 731 \
  --source-revision "${RQMC_REVISION}" \
  --output "${smoke}"
echo "smoke_pass"

printf 'RQMC C1 preflight complete for %s\n' "${RQMC_REVISION}" > "${complete}"
