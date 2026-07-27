#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${PROBE_SOURCE:?Set PROBE_SOURCE to the clean full-SHA worktree.}"
: "${PROBE_RUN_ROOT:?Set PROBE_RUN_ROOT to the fresh run root.}"
: "${PROBE_REVISION:?Set PROBE_REVISION to the complete candidate SHA.}"
: "${FROZEN_INPUT:?Set FROZEN_INPUT to the reviewed PARIS NetCDF.}"
: "${FROZEN_INPUT_SHA:?Set FROZEN_INPUT_SHA to its whole-file SHA-256.}"

module load git/2.45.1-pqk5

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
preflight="${PROBE_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
complete="${preflight}/PREFLIGHT_COMPLETE.txt"

for directory in "${PROBE_SOURCE}" "${PROBE_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${FROZEN_INPUT}" || -L "${FROZEN_INPUT}" ]]; then
  echo "Frozen input must be a real regular file." >&2
  exit 2
fi
for path in "${log}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace preflight evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${PROBE_SOURCE}" rev-parse HEAD)" != "${PROBE_REVISION}" ]]; then
  echo "PROBE_SOURCE does not match PROBE_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${PROBE_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "PROBE_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${PROBE_SOURCE}/.pixi" ||
      "$(readlink -f "${PROBE_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "PROBE_SOURCE/.pixi must point to the canonical BP1 environment." >&2
  exit 2
fi
if [[ "$(sha256sum "${FROZEN_INPUT}" | awk '{print $1}')" != "${FROZEN_INPUT_SHA}" ]]; then
  echo "Frozen input SHA-256 mismatch." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROBE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/exact-mixture-paris-probe-mpl-${PROBE_REVISION}"
export NUMBA_CACHE_DIR="/tmp/exact-mixture-paris-probe-numba-${PROBE_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: >"${log}"
exec > >(tee -a "${log}") 2>&1

cd "${PROBE_SOURCE}"
echo "revision=${PROBE_REVISION}"
pixi --version
pixi run --frozen --no-install -e dev \
  python -c 'import platform,numpy,scipy,xarray; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}"); print(f"xarray={xarray.__version__}")'

pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture_confirm.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_exact_mixture_paris_probe.py

focused=(
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py
  examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm_certify.py
  examples/rjmcmc/conditional_residual_image_exact_mixture_paris_probe.py
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture_confirm.py
  tests/experimental/rjmcmc/test_conditional_residual_image_exact_mixture_paris_probe.py
)
pixi run --frozen --no-install -e dev ruff format --check "${focused[@]}"
pixi run --frozen --no-install -e dev ruff check "${focused[@]}"
pixi run --frozen --no-install -e dev pyright "${focused[@]}"

printf 'Exact-mixture PARIS resource-probe preflight complete for %s\n' \
  "${PROBE_REVISION}" >"${complete}"
