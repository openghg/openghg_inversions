#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${MIX_SOURCE:?Set MIX_SOURCE to the clean full-SHA worktree.}"
: "${MIX_RUN_ROOT:?Set MIX_RUN_ROOT to the pre-created immutable run directory.}"
: "${MIX_REVISION:?Set MIX_REVISION to the complete 40-character source SHA.}"

module load git/2.45.1-pqk5

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
preflight="${MIX_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
smoke_a="${preflight}/smoke-a.json"
smoke_b="${preflight}/smoke-b.json"
complete="${preflight}/PREFLIGHT_COMPLETE.txt"

if [[ "${#MIX_REVISION}" -ne 40 || ! "${MIX_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "MIX_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${MIX_SOURCE}" == *PARIS_inversions* || "${MIX_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Exact-mixture source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${MIX_SOURCE}" "${MIX_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
for path in "${log}" "${smoke_a}" "${smoke_b}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace existing preflight evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${MIX_SOURCE}" rev-parse HEAD)" != "${MIX_REVISION}" ]]; then
  echo "MIX_SOURCE does not match MIX_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${MIX_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "MIX_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${MIX_SOURCE}/.pixi" ||
      "$(readlink -f "${MIX_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "MIX_SOURCE/.pixi must point to the canonical BP1 environment." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${MIX_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/exact-mixture-preflight-mpl-${MIX_REVISION}"
export NUMBA_CACHE_DIR="/tmp/exact-mixture-preflight-numba-${MIX_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: >"${log}"
exec > >(tee -a "${log}") 2>&1

cd "${MIX_SOURCE}"
echo "revision=${MIX_REVISION}"
echo "head=$(git rev-parse HEAD)"
pixi --version
pixi run --frozen --no-install -e dev \
  python -c 'import platform, numpy, scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'

pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py

pixi run --frozen --no-install -e dev ruff format --check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py
pixi run --frozen --no-install -e dev ruff check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py
pixi run --frozen --no-install -e dev pyright \
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_compressed_mixture_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py

for output in "${smoke_a}" "${smoke_b}"; do
  pixi run --frozen --no-install -e dev \
    python examples/rjmcmc/conditional_residual_image_compressed_mixture_tiny_screen.py \
    --profile smoke \
    --stage source \
    --case-id near_gaussian__two_cell__root \
    --source-revision "${MIX_REVISION}" \
    --no-timings \
    --output "${output}"
done
if ! cmp -s "${smoke_a}" "${smoke_b}"; then
  echo "Timing-free smoke reports are not byte-identical." >&2
  exit 2
fi

printf 'Exact-mixture preflight complete for %s\n' "${MIX_REVISION}" >"${complete}"
