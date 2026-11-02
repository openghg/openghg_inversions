#!/usr/bin/env bash
#SBATCH --job-name=ogi-linked-controls
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=inversion_outputs/linked-controls/slurm-%A_%a.out
set -euo pipefail
module load git || exit 1
repo_root=${OGI_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
cd "$repo_root"
export DASK_NUM_WORKERS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ogi-linked-mpl-${SLURM_JOB_ID}"
# These controls explicitly require double precision for centered O2 transport.
IFS=',' read -r -a inherited_flags <<< "${PYTENSOR_FLAGS:-floatX=float64}"
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/ogi-linked-pytensor-${SLURM_JOB_ID}"
for flag in "${inherited_flags[@]}"; do
    case "${flag//[[:space:]]/}" in base_compiledir=*|'') continue ;; esac
    PYTENSOR_FLAGS+=",$flag"
done
modes=(fixed multiplicative multiplicative_shrunk multiplicative multiplicative)
likelihoods=(fixed_independent fixed_independent fixed_independent fixed_ou inferred_ou)
mode=${modes[${SLURM_ARRAY_TASK_ID:-1}]}
likelihood=${likelihoods[${SLURM_ARRAY_TASK_ID:-1}]}
python_bin=${OGI_PYTHON:-$repo_root/.venv/bin/python}
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" examples/shared_baseline/linked_workflow.py \
    --output-dir "inversion_outputs/linked-controls/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/$mode-$likelihood" \
    --mode "$mode" --likelihood "$likelihood" --draws 250 --tune 250
