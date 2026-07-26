#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=mh-s0-aggregate
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${BRANCH_PROFILE:-primary}"
harness_job_init "aggregate-${profile}"
output="${RUN_ROOT}/18-aggregate/${profile}"
mkdir -p "$(dirname "${output}")"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" aggregate-s0 \
    --index "${RUN_ROOT}/17-index/${profile}/index.json" --output-directory "${output}"
harness_job_complete "aggregate-${profile}" "${output}/complete.json"
pixi_python "${HARNESS_DIR}/24-final-audit.py" finalize \
    --run-root "${RUN_ROOT}" --repo-root "${REPO_ROOT}" --profile "${profile}"
