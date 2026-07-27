#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${BANK_SOURCE:?}"
: "${BANK_RUN_ROOT:?}"
: "${BANK_REVISION:?}"
: "${FROZEN_INPUT:?}"
: "${FROZEN_INPUT_SHA:?}"

assets="${BANK_SOURCE}/docs/plans/rjmcmc_chunked_projected_bank_assets"
stage="${BANK_RUN_ROOT}/g3"
submission="${stage}/G3_SUBMISSION.txt"
if [[ -e "${submission}" || -L "${submission}" ]]; then
  echo "Refusing to replace G3 submission evidence: ${submission}" >&2
  exit 2
fi
if [[ ! -f "${stage}/prefix/G3A_COMPLETE.txt" ]]; then
  echo "G3a must pass before submitting the resource matrix." >&2
  exit 2
fi

mkdir -p "${stage}/warmup" "${stage}/resource"
warmup_job="$(
  sbatch --parsable \
    --output="${stage}/warmup/slurm-%j.out" \
    --error="${stage}/warmup/slurm-%j.err" \
    "${assets}/run_g3_warmup.sbatch"
)"

previous="${warmup_job}"
candidate_jobs=()
for chunk in 1024 2048 4096 8192; do
  for repeat in 0 1 2; do
    candidate_dir="${stage}/resource/C${chunk}/repeat${repeat}"
    mkdir -p "${candidate_dir}"
    job_id="$(
      sbatch --parsable \
        --dependency="afterok:${previous}" \
        --export="ALL,BANK_SAMPLE_CHUNK=${chunk},BANK_REPEAT=${repeat},BANK_CANDIDATE_DIR=${candidate_dir}" \
        --output="${candidate_dir}/slurm-%j.out" \
        --error="${candidate_dir}/slurm-%j.err" \
        "${assets}/run_g3_bank.sbatch"
    )"
    candidate_jobs+=("${job_id}")
    previous="${job_id}"
  done
done

joined="$(
  IFS=:
  echo "${candidate_jobs[*]}"
)"
certifier_job="$(
  sbatch --parsable \
    --dependency="afterany:${joined}" \
    --output="${stage}/certify-slurm-%j.out" \
    --error="${stage}/certify-slurm-%j.err" \
    "${assets}/run_g3_certify.sbatch"
)"

{
  printf 'revision=%s\n' "${BANK_REVISION}"
  printf 'warmup_job=%s\n' "${warmup_job}"
  printf 'candidate_jobs=%s\n' "${candidate_jobs[*]}"
  printf 'certifier_job=%s\n' "${certifier_job}"
} >"${submission}"

printf 'CERTIFIER_JOB_ID=%s\n' "${certifier_job}"
