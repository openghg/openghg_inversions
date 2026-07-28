#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${BANK_RUN_ROOT:?}"

stage="${BANK_RUN_ROOT}/g3"
if [[ ! -f "${stage}/prefix/G3A_COMPLETE.txt" ]]; then
  echo "G3a must pass before preparing the resource array." >&2
  exit 2
fi
if [[ ! -f "${stage}/warmup/WARMUP_COMPLETE.txt" ]]; then
  echo "The excluded warm-up must pass before preparing the resource array." >&2
  exit 2
fi
mkdir -p "${stage}/resource" "${stage}/array-logs"
printf '%s\n' \
  "Prepared G3b directories. Submit run_g3_bank.sbatch as --array=0-11%1" \
  "through slurm-wakeup; submit the certifier separately after the array wakes."
