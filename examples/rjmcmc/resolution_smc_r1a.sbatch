#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --job-name=resolution-smc-r1a
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: resolution_smc_r1a.sbatch WORKTREE RUN_ROOT SOURCE_SHA" >&2
    exit 2
fi

experiment_worktree="$1"
experiment_run_root="$2"
experiment_source_sha="$3"
experiment_git="/usr/bin/git"

if [[ ! "${experiment_source_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "SOURCE_SHA must be a complete lowercase Git SHA" >&2
    exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required" >&2
    exit 2
fi
if [[ ! -x "${experiment_git}" ]]; then
    echo "required Git executable is unavailable: ${experiment_git}" >&2
    exit 2
fi

# BP1 compute jobs receive a deliberately minimal PATH.  Keep the Python
# provenance checks functional as well as using an authenticated absolute Git
# path in this launcher.
export PATH="/usr/bin:/bin:${PATH:-}"

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

task_scratch="${TMPDIR:-/tmp}/resolution-smc-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${task_scratch}/matplotlib"
export MPLCONFIGDIR="${task_scratch}/matplotlib"
export PYTHONPATH="${experiment_worktree}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.pixi/envs/dev/bin/python \
    examples/rjmcmc/resolution_smc_experiment.py \
    --mode r1a-task \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --source-sha "${experiment_source_sha}" \
    --output-root "${experiment_run_root}/r1a"
