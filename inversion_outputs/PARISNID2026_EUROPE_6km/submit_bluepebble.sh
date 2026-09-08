#!/usr/bin/env bash
#SBATCH --job-name=parisnid6km
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=chem007981
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --array=0-23
#SBATCH --output=/user/home/vq21425/openghg_inversions/inversion_outputs/PARISNID2026_EUROPE_6km/logs/%x_%A_%a.out
#SBATCH --error=/user/home/vq21425/openghg_inversions/inversion_outputs/PARISNID2026_EUROPE_6km/logs/%x_%A_%a.err

set -euo pipefail

workspace=/user/home/vq21425/openghg_inversions
run_dir=${workspace}/inversion_outputs/PARISNID2026_EUROPE_6km

module load git
cd "${workspace}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"

if [[ ${PYTENSOR_FLAGS:-} != *base_compiledir=* ]]; then
    export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=${TMPDIR:-/tmp}/pytensor-${USER}-${SLURM_JOB_ID}"
fi

mkdir -p "${MPLCONFIGDIR}"

start_dates=(
    2023-01-01 2023-02-01 2023-03-01 2023-04-01 2023-05-01 2023-06-01
    2023-07-01 2023-08-01 2023-09-01 2023-10-01 2023-11-01 2023-12-01
    2024-01-01 2024-02-01 2024-03-01 2024-04-01 2024-05-01 2024-06-01
    2024-07-01 2024-08-01 2024-09-01 2024-10-01 2024-11-01 2024-12-01
)
end_dates=(
    2023-02-01 2023-03-01 2023-04-01 2023-05-01 2023-06-01 2023-07-01
    2023-08-01 2023-09-01 2023-10-01 2023-11-01 2023-12-01 2024-01-01
    2024-02-01 2024-03-01 2024-04-01 2024-05-01 2024-06-01 2024-07-01
    2024-08-01 2024-09-01 2024-10-01 2024-11-01 2024-12-01 2025-01-01
)
start_date=${start_dates[SLURM_ARRAY_TASK_ID]}
end_date=${end_dates[SLURM_ARRAY_TASK_ID]}

echo "Workspace: ${workspace}"
echo "Commit: $(git rev-parse HEAD)"
echo "Config: ${run_dir}/hbmcmc.ini"
echo "Dates: ${start_date} to ${end_date}"
echo "Started: $(date --iso-8601=seconds)"

"${workspace}/.venv/bin/python" "${run_dir}/run_nested_local.py" \
    "${run_dir}/hbmcmc.ini" \
    --start-date "${start_date}" \
    --end-date "${end_date}"

echo "Finished: $(date --iso-8601=seconds)"
