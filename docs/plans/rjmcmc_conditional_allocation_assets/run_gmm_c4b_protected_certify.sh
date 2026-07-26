#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${GMM_SOURCE:?Set GMM_SOURCE to the clean full-SHA worktree.}"
: "${GMM_RUN_ROOT:?Set GMM_RUN_ROOT to the pre-created immutable run directory.}"
: "${GMM_REVISION:?Set GMM_REVISION to the complete 40-character source SHA.}"

# Deliberate G3 seal: this script remains disabled until the operator supplies
# the independent catalogue path after reviewing a passing G2 certificate.
if [[ -z "${GMM_PROTECTED_CATALOGUE:-}" ]]; then
  echo "Protected certification is disabled: explicitly set GMM_PROTECTED_CATALOGUE after G2 passes." >&2
  exit 2
fi

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
certificate="${GMM_RUN_ROOT}/certificate/development-certificate.json"
certificate_raw_sha_record="${GMM_RUN_ROOT}/certificate/development-certificate.raw.sha256"
development="${GMM_RUN_ROOT}/development"
protected_directory="${GMM_RUN_ROOT}/protected"
output="${protected_directory}/protected-certificate.json"
complete="${protected_directory}/PROTECTED_CERTIFICATION_COMPLETE.txt"

if [[ "${#GMM_REVISION}" -ne 40 || ! "${GMM_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "GMM_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${GMM_SOURCE}" == *PARIS_inversions* || "${GMM_RUN_ROOT}" == *PARIS_inversions* ||
      "${GMM_PROTECTED_CATALOGUE}" == *PARIS_inversions* ]]; then
  echo "C4b source, catalogue, and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${GMM_SOURCE}" "${GMM_RUN_ROOT}" "${development}" \
  "${protected_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${certificate}" || -L "${certificate}" ||
      ! -f "${certificate_raw_sha_record}" || -L "${certificate_raw_sha_record}" ]]; then
  echo "The development certificate and raw SHA record must be regular non-symlink files." >&2
  exit 2
fi
if [[ -e "${output}" || -L "${output}" || -e "${complete}" || -L "${complete}" ]]; then
  echo "Refusing to replace existing protected-certification evidence." >&2
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
# Check the canonical certificate decision in shell before invoking the
# protected Python executable. Do not print or otherwise inspect the catalogue.
certificate_raw_sha256="$(<"${certificate_raw_sha_record}")"
if [[ "${#certificate_raw_sha256}" -ne 64 ||
      ! "${certificate_raw_sha256}" =~ ^[0-9a-f]+$ ||
      "$(sha256sum "${certificate}" | awk '{print $1}')" != "${certificate_raw_sha256}" ]]; then
  echo "The development certificate does not match its immutable raw SHA record." >&2
  exit 2
fi
if [[ "$(cat "${GMM_RUN_ROOT}/certificate/CERTIFY_CONFIRMATION_COMPLETE.txt")" != \
      "GMM C4b confirmation certification complete for ${GMM_REVISION} raw_sha256=${certificate_raw_sha256}" ]]; then
  echo "The G2 completion marker is absent or invalid." >&2
  exit 2
fi
if ! grep -q '"decision":"pass"' "${certificate}" ||
   ! grep -q '"development_pass":true' "${certificate}" ||
   ! grep -q '"eligible_for_protected_holdout":true' "${certificate}" ||
   ! grep -q '"protected_holdout_pass":null' "${certificate}"; then
  echo "Protected certification is disabled because the development certificate did not pass." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${GMM_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/gmm-c4b-protected-matplotlib-${GMM_REVISION}"
export NUMBA_CACHE_DIR="/tmp/gmm-c4b-protected-numba-${GMM_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${GMM_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_gmm_protected_certify.py \
  --source-directory "${GMM_SOURCE}" \
  --expected-source-revision "${GMM_REVISION}" \
  --development-certificate "${certificate}" \
  --expected-development-certificate-sha256 "${certificate_raw_sha256}" \
  --development-shards-directory "${development}" \
  --protected-catalogue "${GMM_PROTECTED_CATALOGUE}" \
  --output "${output}"

printf 'GMM C4b protected certification complete for %s\n' \
  "${GMM_REVISION}" > "${complete}"
