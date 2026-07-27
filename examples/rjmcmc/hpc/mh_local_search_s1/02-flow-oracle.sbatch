#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=mh-s1-flow
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
harness_job_init flow
output="${RUN_ROOT}/02-flow"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_flow_oracle.py" \
    --output-directory "${output}" --source-revision "${FULL_SHA}"
harness_job_complete flow "${output}/complete.json"
