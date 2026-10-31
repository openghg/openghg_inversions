#!/usr/bin/env bash
#SBATCH --job-name=ogi-bc-prior-replay
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00

set -euo pipefail
module load git || exit 1
replay_root=${1:?New replay directory prepared with scripts/replay_bc_priors.py}
arm=${2:-delta_A}
repo_root=${SLURM_SUBMIT_DIR:?Submit from the implementation worktree}
task_cache=${TMPDIR:-/tmp}/ogi-bc-priors-${SLURM_JOB_ID}
mkdir -p "$task_cache/mpl" "$task_cache/numba" "$task_cache/pytensor"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$task_cache/mpl" NUMBA_CACHE_DIR="$task_cache/numba"
export PYTENSOR_FLAGS=$("$repo_root/.venv/bin/python" - "$task_cache/pytensor" <<'PY'
import os
import sys
flags = [item for item in os.environ.get("PYTENSOR_FLAGS", "").split(",")
         if item and item.split("=", 1)[0].strip() != "base_compiledir"]
if not any(item.split("=", 1)[0].strip() == "floatX" for item in flags):
    flags.append("floatX=float64")
print(",".join([*flags, "base_compiledir=" + sys.argv[1]]))
PY
)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DASK_NUM_WORKERS=2
cd "$repo_root"
science_paths=(openghg_inversions pyproject.toml uv.lock scripts/replay_bc_priors.py scripts/slurm_bc_prior_replay.sh)
git status --porcelain --untracked-files=no -- "${science_paths[@]}"
if [[ -n $(git status --porcelain --untracked-files=no -- "${science_paths[@]}") ]]; then
    printf 'Commit the implementation before sampling replay.\n' >&2
    exit 1
fi
/usr/bin/time -v .venv/bin/python -u scripts/replay_bc_priors.py --root "$replay_root" --arm "$arm" --sample
