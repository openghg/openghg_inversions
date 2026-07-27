#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${CONF_SOURCE:?Set CONF_SOURCE to the clean full-SHA worktree.}"
: "${CONF_RUN_ROOT:?Set CONF_RUN_ROOT to the fresh confirmation run root.}"
: "${CONF_REVISION:?Set CONF_REVISION to the complete certifier candidate SHA.}"

# A recovery merge may authenticate immutable artifacts produced by an older
# sampler candidate after a reporting-only certifier fix.  In an ordinary run
# this defaults to the certifier revision, preserving the original protocol.
CONF_ARTIFACT_REVISION="${CONF_ARTIFACT_REVISION:-${CONF_REVISION}}"

module load git/2.45.1-pqk5

confirmation_directory="${CONF_RUN_ROOT}/confirmation"
marker_directory="${CONF_RUN_ROOT}/markers/confirmation"
decision_directory="${CONF_RUN_ROOT}/decision"
decision="${decision_directory}/confirmation-decision.json"
raw_sha_record="${decision_directory}/confirmation-decision.raw.sha256"
complete="${decision_directory}/CONFIRMATION_MERGE_COMPLETE.txt"
case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  skewed__two_cell__root
  skewed__four_cell__root
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
)
source_seeds=(1877 4099 8317)

for directory in "${CONF_SOURCE}" "${CONF_RUN_ROOT}" "${confirmation_directory}" \
  "${marker_directory}" "${decision_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
for path in "${decision}" "${raw_sha_record}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace confirmation decision evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${CONF_SOURCE}" rev-parse HEAD)" != "${CONF_REVISION}" ]]; then
  echo "CONF_SOURCE does not match CONF_REVISION." >&2
  exit 2
fi
shopt -s nullglob dotglob
marker_entries=("${marker_directory}"/*)
if [[ "${#marker_entries[@]}" -ne 18 ]]; then
  echo "Confirmation marker directory must contain exactly 18 entries." >&2
  exit 2
fi
for case_id in "${case_ids[@]}"; do
  for source_seed in "${source_seeds[@]}"; do
    stem="${case_id}__seed${source_seed}"
    marker="${marker_directory}/${stem}.complete"
    expected="complete revision=${CONF_ARTIFACT_REVISION} case=${case_id} source_seed=${source_seed}"
    if [[ ! -f "${marker}" || -L "${marker}" || "$(<"${marker}")" != "${expected}" ]]; then
      echo "Confirmation marker is absent or invalid: ${marker}" >&2
      exit 2
    fi
  done
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export PYTHONPATH="${CONF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CONF_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_compressed_mixture_confirm_certify.py \
  --report-directory "${confirmation_directory}" \
  --expected-source-revision "${CONF_ARTIFACT_REVISION}" \
  --output "${decision}"

decision_raw_sha256="$(sha256sum "${decision}" | awk '{print $1}')"
printf '%s\n' "${decision_raw_sha256}" >"${raw_sha_record}"
eligible="$(pixi run --frozen --no-install -e dev python -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["eligible"]).lower())' \
  "${decision}")"
printf 'Exact-mixture confirmation merge complete artifact_revision=%s certifier_revision=%s eligible=%s raw_sha256=%s\n' \
  "${CONF_ARTIFACT_REVISION}" "${CONF_REVISION}" "${eligible}" \
  "${decision_raw_sha256}" >"${complete}"
