#!/usr/bin/env bash
#SBATCH --job-name=ogi-tox
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm-tox-%j.out

set -euo pipefail

repo_root=${OGI_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}
if [[ ! -f "$repo_root/tox.ini" || ! -f "$repo_root/pyproject.toml" ]]; then
    printf 'Not an openghg_inversions checkout: %s\n' "$repo_root" >&2
    exit 2
fi

scratch_parent=${SLURM_TMPDIR:-/tmp}
tox_work_dir=$(mktemp -d "$scratch_parent/ogi-tox-${SLURM_JOB_ID:-manual}.XXXXXX")
cleanup() {
    rm -rf -- "$tox_work_dir"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

export TOX_WORK_DIR="$tox_work_dir"
export UV_CACHE_DIR=${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}
# The shared cache and node-local tox directory are on different filesystems.
# Copies are expected here and disappear with TOX_WORK_DIR at job exit.
export UV_LINK_MODE=copy
export PYTEST_XDIST_WORKERS=${PYTEST_XDIST_WORKERS:-2}
export MPLCONFIGDIR="$tox_work_dir/matplotlib"
export PYTENSOR_FLAGS="base_compiledir=$tox_work_dir/pytensor${PYTENSOR_FLAGS:+,$PYTENSOR_FLAGS}"

cd "$repo_root"
uvx --from 'tox>=4.24' --with tox-uv tox --workdir "$TOX_WORK_DIR" -p --parallel-no-spinner "$@"
