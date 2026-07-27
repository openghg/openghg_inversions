#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?Set BANK_SOURCE to the clean detached full-SHA worktree.}"
: "${BANK_RUN_ROOT:?Set BANK_RUN_ROOT to the fresh run root.}"
: "${BANK_REVISION:?Set BANK_REVISION to the complete candidate SHA.}"
: "${FROZEN_INPUT:?Set FROZEN_INPUT to the reviewed PARIS NetCDF.}"
: "${FROZEN_INPUT_SHA:?Set FROZEN_INPUT_SHA to its whole-file SHA-256.}"

module load git/2.45.1-pqk5

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
preflight="${BANK_RUN_ROOT}/g0"
log="${preflight}/preflight.log"
complete="${preflight}/G0_COMPLETE.txt"

for directory in "${BANK_SOURCE}" "${BANK_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${FROZEN_INPUT}" || -L "${FROZEN_INPUT}" ]]; then
  echo "Frozen input must be a real regular file." >&2
  exit 2
fi
for path in "${log}" "${complete}" \
  "${preflight}/tiny_bank.npy" "${preflight}/tiny_report.json"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace G0 evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${BANK_SOURCE}" rev-parse HEAD)" != "${BANK_REVISION}" ]]; then
  echo "BANK_SOURCE does not match BANK_REVISION." >&2
  exit 2
fi
source_status="$(git -C "${BANK_SOURCE}" status --porcelain)"
if [[ -n "${source_status}" && "${source_status}" != "?? .pixi" ]]; then
  echo "BANK_SOURCE may contain only the authenticated untracked .pixi link." >&2
  exit 2
fi
if [[ ! -L "${BANK_SOURCE}/.pixi" ||
      "$(readlink -f "${BANK_SOURCE}/.pixi")" != "${canonical_pixi}" ]]; then
  echo "BANK_SOURCE/.pixi must point to the canonical BP1 environment." >&2
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
export PYTHONPATH="${BANK_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/chunked-bank-g0-mpl-${BANK_REVISION}"
export NUMBA_CACHE_DIR="/tmp/chunked-bank-g0-numba-${BANK_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: >"${log}"
exec > >(tee -a "${log}") 2>&1

cd "${BANK_SOURCE}"
echo "revision=${BANK_REVISION}"
echo "input=${FROZEN_INPUT}"
echo "input_sha256=${FROZEN_INPUT_SHA}"
pixi --version
pixi run --frozen --no-install -e dev \
  python -c 'import platform,numpy,scipy,xarray; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}"); print(f"xarray={xarray.__version__}"); numpy.show_config()'

pixi run --frozen --no-install -e dev pytest -q \
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_exact_mixture_paris_probe.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_chunked_projected_bank_hpc.py

focused=(
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_mixture.py
  openghg_inversions/experimental/rjmcmc/aggregation_error_exact_mixture.py
  examples/rjmcmc/conditional_residual_image_exact_mixture_paris_probe.py
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py
  tests/experimental/rjmcmc/test_aggregation_error_exact_mixture.py
  tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py
  tests/experimental/rjmcmc/test_conditional_residual_image_compressed_mixture.py
  tests/experimental/rjmcmc/test_conditional_residual_image_exact_mixture_paris_probe.py
  tests/experimental/rjmcmc/test_conditional_residual_image_chunked_projected_bank_hpc.py
)
pixi run --frozen --no-install -e dev ruff format --check "${focused[@]}"
pixi run --frozen --no-install -e dev ruff check "${focused[@]}"
pixi run --frozen --no-install -e dev pyright "${focused[@]}"

pixi run --frozen --no-install -e dev python \
  examples/rjmcmc/conditional_residual_image_chunked_projected_bank_hpc.py \
  tiny \
  --output-dir "${preflight}" \
  --source-revision "${BANK_REVISION}"

printf 'G0 complete for %s\n' "${BANK_REVISION}" >"${complete}"
