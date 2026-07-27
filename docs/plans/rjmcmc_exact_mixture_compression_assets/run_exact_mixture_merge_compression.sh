#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${MIX_SOURCE:?Set MIX_SOURCE to the clean full-SHA worktree.}"
: "${MIX_RUN_ROOT:?Set MIX_RUN_ROOT to the pre-created immutable run directory.}"
: "${MIX_REVISION:?Set MIX_REVISION to the complete 40-character source SHA.}"

module load git/2.45.1-pqk5

compression_directory="${MIX_RUN_ROOT}/compression"
marker_directory="${MIX_RUN_ROOT}/markers/compression"
decision_directory="${MIX_RUN_ROOT}/decision"
source_lock="${decision_directory}/common-source-decision.json"
decision="${decision_directory}/common-compression-decision.json"
raw_sha_record="${decision_directory}/common-compression-decision.raw.sha256"
complete="${decision_directory}/COMPRESSION_MERGE_COMPLETE.txt"
case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  skewed__two_cell__root
  skewed__four_cell__root
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
)

for directory in "${MIX_SOURCE}" "${MIX_RUN_ROOT}" "${compression_directory}" \
  "${marker_directory}" "${decision_directory}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
for path in "${decision}" "${raw_sha_record}" "${complete}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "Refusing to replace compression-merge evidence: ${path}" >&2
    exit 2
  fi
done
if [[ "$(git -C "${MIX_SOURCE}" rev-parse HEAD)" != "${MIX_REVISION}" ]]; then
  echo "MIX_SOURCE does not match MIX_REVISION." >&2
  exit 2
fi
shopt -s nullglob dotglob
marker_entries=("${marker_directory}"/*)
if [[ "${#marker_entries[@]}" -ne 6 ]]; then
  echo "Compression marker directory must contain exactly six entries." >&2
  exit 2
fi
for case_id in "${case_ids[@]}"; do
  marker="${marker_directory}/${case_id}.complete"
  expected="complete revision=${MIX_REVISION} stage=compression case=${case_id}"
  if [[ ! -f "${marker}" || -L "${marker}" || "$(<"${marker}")" != "${expected}" ]]; then
    echo "Compression completion marker is absent or invalid: ${marker}" >&2
    exit 2
  fi
done

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export PYTHONPATH="${MIX_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${MIX_SOURCE}"
pixi run --frozen --no-install -e dev \
  python examples/rjmcmc/conditional_residual_image_compressed_mixture_certify.py \
  merge-compression \
  --report-directory "${compression_directory}" \
  --source-lock "${source_lock}" \
  --expected-source-revision "${MIX_REVISION}" \
  --output "${decision}"

decision_raw_sha256="$(sha256sum "${decision}" | awk '{print $1}')"
printf '%s\n' "${decision_raw_sha256}" >"${raw_sha_record}"
eligible="$(pixi run --frozen --no-install -e dev python -c \
  'import json,sys; print(str(json.load(open(sys.argv[1], encoding="ascii"))["eligible"]).lower())' \
  "${decision}")"
printf 'Exact-mixture compression merge complete for %s eligible=%s raw_sha256=%s\n' \
  "${MIX_REVISION}" "${eligible}" "${decision_raw_sha256}" >"${complete}"
