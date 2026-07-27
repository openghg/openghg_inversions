#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=mh-s1-index
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${BRANCH_PROFILE:-primary}"
harness_job_init "build-index-${profile}"
pixi_python "${HARNESS_DIR}/23-build-index.py" \
    --run-root "${RUN_ROOT}" --profile "${profile}" --source-revision "${FULL_SHA}"
harness_job_complete "build-index-${profile}" "${RUN_ROOT}/17-index/${profile}/complete.json"
