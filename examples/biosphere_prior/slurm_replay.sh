#!/usr/bin/env bash
#SBATCH --job-name=biosphere-replay
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
module load git || exit 1
output_root=${1:?New output root}
prepared=${2:?Original enhancement preparation}
receipt=${3:?Frozen preparation receipt}
prior_arrays=${4:?Frozen full prior arrays}
prior_sha256=${5:?Frozen full prior digest}
families=(lognormal_original lognormal_shrunk gaussian_original gaussian_shrunk)
family=${families[$SLURM_ARRAY_TASK_ID]}
cache_root=$(mktemp -d "${TMPDIR:-/tmp}/biosphere-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$cache_root"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DASK_NUM_WORKERS=2
export MPLCONFIGDIR="$cache_root/mpl" NUMBA_CACHE_DIR="$cache_root/numba"
IFS=',' read -r -a inherited_flags <<< "${PYTENSOR_FLAGS:-}"
export PYTENSOR_FLAGS="floatX=float64,base_compiledir=$cache_root/pytensor"
for flag in "${inherited_flags[@]}"; do
    case "${flag//[[:space:]]/}" in floatX=*|base_compiledir=*|'') continue ;; esac
    PYTENSOR_FLAGS+=",$flag"
done
.venv/bin/python -u -m examples.biosphere_prior.replay_sampling \
    --prepared "$prepared" --preparation-receipt "$receipt" \
    --prior-arrays "$prior_arrays" --prior-sha256 "$prior_sha256" \
    --family "$family" --output "$output_root/$family" \
    --chains 2 --tune 300 --draws 300
