#!/usr/bin/env bash
#SBATCH --job-name=ogi-linked-return-replay
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
module load git || exit 1
repo=${1:?Candidate repository}
output_root=${2:?Unique replay result directory}
python_bin=/group/chem/acrg/verification_games_round_2/verification-games-work/sources/ope188-ogi-devel-20261028/.venv/bin/python
strategies=(full compact)
strategy=${strategies[${SLURM_ARRAY_TASK_ID:?}]}
task_cache=${TMPDIR:-/tmp}/ogi-linked-replay-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "$task_cache/numba" "$task_cache/mpl" "$task_cache/pytensor"
export PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$task_cache/numba" MPLCONFIGDIR="$task_cache/mpl"
export PYTENSOR_FLAGS=$("$python_bin" -c 'import os, sys; flags = [item for item in os.environ.get("PYTENSOR_FLAGS", "").split(",") if item and item.split("=", 1)[0].strip() not in {"floatX", "base_compiledir"}]; print(",".join(flags + ["floatX=float64", "base_compiledir=" + sys.argv[1]]))' "$task_cache/pytensor")
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
git -C "$repo" rev-parse HEAD
/usr/bin/time -v "$python_bin" -u "$repo/scripts/replay_linked_campaign_return.py" "$strategy" \
    --output-dir "$output_root/$strategy"
