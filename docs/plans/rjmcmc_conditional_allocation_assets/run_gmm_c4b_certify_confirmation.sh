#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${GMM_SOURCE:?Set GMM_SOURCE to the clean full-SHA worktree.}"
: "${GMM_RUN_ROOT:?Set GMM_RUN_ROOT to the pre-created immutable run directory.}"
: "${GMM_REVISION:?Set GMM_REVISION to the complete 40-character source SHA.}"

canonical_pixi="/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi"
development="${GMM_RUN_ROOT}/development"
confirmation="${GMM_RUN_ROOT}/confirmation"
marker_directory="${GMM_RUN_ROOT}/markers/confirmation"
lock="${GMM_RUN_ROOT}/lock/common-lock.json"
raw_sha_record="${GMM_RUN_ROOT}/lock/common-lock.raw.sha256"
certificate_directory="${GMM_RUN_ROOT}/certificate"
certificate="${certificate_directory}/development-certificate.json"
certificate_raw_sha_record="${certificate_directory}/development-certificate.raw.sha256"
complete="${certificate_directory}/CERTIFY_CONFIRMATION_COMPLETE.txt"
case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  skewed__two_cell__root
  skewed__four_cell__root
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
)
confirmation_seeds=(1877 4099 8317)

if [[ "${#GMM_REVISION}" -ne 40 || ! "${GMM_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "GMM_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
if [[ "${GMM_SOURCE}" == *PARIS_inversions* || "${GMM_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "C4b source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${GMM_SOURCE}" "${GMM_RUN_ROOT}" "${development}" \
  "${confirmation}" "${marker_directory}" "${certificate_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -f "${lock}" || -L "${lock}" || ! -f "${raw_sha_record}" ||
      -L "${raw_sha_record}" ]]; then
  echo "The common lock and raw SHA record must be regular non-symlink files." >&2
  exit 2
fi
if [[ -e "${certificate}" || -L "${certificate}" ||
      -e "${certificate_raw_sha_record}" || -L "${certificate_raw_sha_record}" ||
      -e "${complete}" || -L "${complete}" ]]; then
  echo "Refusing to replace existing development-certificate evidence." >&2
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
lock_raw_sha256="$(<"${raw_sha_record}")"
if [[ "${#lock_raw_sha256}" -ne 64 || ! "${lock_raw_sha256}" =~ ^[0-9a-f]+$ ||
      "$(sha256sum "${lock}" | awk '{print $1}')" != "${lock_raw_sha256}" ]]; then
  echo "The common lock does not match its immutable raw SHA record." >&2
  exit 2
fi
shopt -s nullglob dotglob
marker_entries=("${marker_directory}"/*)
if [[ "${#marker_entries[@]}" -ne 18 ]]; then
  echo "Confirmation marker directory must contain exactly 18 entries." >&2
  exit 2
fi
for case_id in "${case_ids[@]}"; do
  for confirmation_seed in "${confirmation_seeds[@]}"; do
    marker="${marker_directory}/${case_id}__seed${confirmation_seed}.complete"
    expected="complete revision=${GMM_REVISION} case=${case_id} confirmation_seed=${confirmation_seed} lock_raw_sha256=${lock_raw_sha256}"
    expected_size=$((${#expected} + 1))
    if [[ ! -f "${marker}" || -L "${marker}" ||
          "$(wc -c < "${marker}")" -ne "${expected_size}" ||
          "$(<"${marker}")" != "${expected}" ]]; then
      echo "Confirmation completion marker is absent or invalid: ${marker}" >&2
      exit 2
    fi
  done
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${GMM_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${GMM_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_gmm_certify.py \
  certify-confirmation \
  --source-dir "${GMM_SOURCE}" \
  --development-dir "${development}" \
  --confirmation-dir "${confirmation}" \
  --common-lock "${lock}" \
  --expected-lock-raw-sha256 "${lock_raw_sha256}" \
  --expected-source-revision "${GMM_REVISION}" \
  --output-certificate "${certificate}"

certificate_raw_sha256="$(sha256sum "${certificate}" | awk '{print $1}')"
if [[ "${#certificate_raw_sha256}" -ne 64 ||
      ! "${certificate_raw_sha256}" =~ ^[0-9a-f]+$ ]]; then
  echo "Failed to compute the development certificate raw SHA-256." >&2
  exit 2
fi
printf '%s\n' "${certificate_raw_sha256}" > "${certificate_raw_sha_record}"
printf 'GMM C4b confirmation certification complete for %s raw_sha256=%s\n' \
  "${GMM_REVISION}" "${certificate_raw_sha256}" > "${complete}"
