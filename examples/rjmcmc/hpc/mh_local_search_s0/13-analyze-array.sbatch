#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%6
#SBATCH --job-name=mh-s0-analysis
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
profile="${BRANCH_PROFILE:-primary}"
harness_job_init "analysis-${profile}"
IFS=$'\t' read -r _ scenario replicate cell_key <<<"$(cell_row "${SLURM_ARRAY_TASK_ID}")"
output="${RUN_ROOT}/13-analysis/${profile}/${cell_key}"
mkdir -p "$(dirname "${output}")"
pixi_python "${REPO_ROOT}/examples/rjmcmc/mh_local_search_synthetic.py" analyze \
    --training "${RUN_ROOT}/03-materialize/training/${cell_key}.json" \
    --evaluation "${RUN_ROOT}/03-materialize/sealed-evaluation/${cell_key}.json" \
    --run-directory "${RUN_ROOT}/10-pair/${profile}/${cell_key}" \
    --output-directory "${output}"
harness_job_complete "analysis-${profile}" "${output}/complete.json"
