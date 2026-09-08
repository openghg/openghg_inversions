#!/usr/bin/env bash
#SBATCH --job-name=parisnid6km
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=chem007981
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=/user/home/vq21425/openghg_inversions/inversion_outputs/PARISNID2026_EUROPE_6km/logs/%x_%j.out
#SBATCH --error=/user/home/vq21425/openghg_inversions/inversion_outputs/PARISNID2026_EUROPE_6km/logs/%x_%j.err

set -euo pipefail

workspace=/user/home/vq21425/openghg_inversions
run_dir=${workspace}/inversion_outputs/PARISNID2026_EUROPE_6km

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

echo "Workspace: ${workspace}"
echo "Commit: $(git rev-parse HEAD)"
echo "Config: ${run_dir}/hbmcmc.ini"
echo "Started: $(date --iso-8601=seconds)"

"${workspace}/.venv/bin/python" "${run_dir}/run_nested_local.py" "${run_dir}/hbmcmc.ini"

echo "Finished: $(date --iso-8601=seconds)"
