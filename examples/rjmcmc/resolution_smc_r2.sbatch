#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=resolution-smc-r2
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: resolution_smc_r2.sbatch WORKTREE RUN_ROOT SOURCE_SHA" >&2
    exit 2
fi

experiment_worktree="$1"
experiment_run_root="$2"
experiment_source_sha="$3"
experiment_git_module="git/2.45.1-pqk5"

if [[ ! "${experiment_source_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "SOURCE_SHA must be a complete lowercase Git SHA" >&2
    exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required" >&2
    exit 2
fi

export PATH="/usr/bin:/bin:${PATH:-}"
source /etc/profile.d/modules.sh
module load "${experiment_git_module}"
experiment_git="$(command -v git || true)"
if [[ ! -x "${experiment_git}" ]]; then
    echo "required Git module did not provide an executable: ${experiment_git_module}" >&2
    exit 2
fi

cd "${experiment_worktree}"
actual_sha="$("${experiment_git}" rev-parse HEAD)"
if [[ "${actual_sha}" != "${experiment_source_sha}" ]]; then
    echo "detached worktree SHA mismatch: ${actual_sha}" >&2
    exit 3
fi
if [[ -n "$("${experiment_git}" status --porcelain)" ]]; then
    echo "detached worktree is not clean" >&2
    exit 3
fi
if "${experiment_git}" symbolic-ref -q HEAD >/dev/null; then
    echo "scientific worktree must be detached" >&2
    exit 3
fi

task_scratch="${TMPDIR:-/tmp}/resolution-smc-r2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${task_scratch}/matplotlib"
export MPLCONFIGDIR="${task_scratch}/matplotlib"
export PYTHONPATH="${experiment_worktree}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.pixi/envs/dev/bin/python \
    examples/rjmcmc/resolution_smc_r1_r2_experiment.py \
    --mode r2-task \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --source-sha "${experiment_source_sha}" \
    --output-root "${experiment_run_root}/r2"
