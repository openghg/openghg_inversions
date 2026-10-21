#!/usr/bin/env bash
#SBATCH --job-name=ogi-tox
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-tox-%j.out

set -euo pipefail

# The OpenGHG devel tox factor installs a Git dependency, while compute nodes
# may not put Git on PATH by default.
if ! command -v git >/dev/null 2>&1; then
    module load git
fi

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
export TMPDIR="$tox_work_dir/tmp"
mkdir -p "$TMPDIR"
export UV_CACHE_DIR=${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}
# The shared cache and node-local tox directory are on different filesystems.
# Copies are expected here and disappear with TOX_WORK_DIR at job exit.
export UV_LINK_MODE=copy
export PYTEST_XDIST_WORKERS=${PYTEST_XDIST_WORKERS:-0}
# Dask can count host CPUs rather than this allocation's two CPUs.
export DASK_NUM_WORKERS=${DASK_NUM_WORKERS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Managed interpreters include the development headers required by PyTensor.
export UV_PYTHON_PREFERENCE=${UV_PYTHON_PREFERENCE:-managed}
export MPLCONFIGDIR="$tox_work_dir/matplotlib"
# Keep scientific settings while replacing inherited cache paths with local scratch.
IFS=',' read -r -a pytensor_flags <<< "${PYTENSOR_FLAGS:-}"
export PYTENSOR_FLAGS="base_compiledir=$tox_work_dir/pytensor"
for flag in "${pytensor_flags[@]}"; do
    case "${flag//[[:space:]]/}" in base_compiledir=*|'') continue ;; esac
    PYTENSOR_FLAGS+=",$flag"
done

cd "$repo_root"
uvx --from 'tox>=4.24' --with tox-uv tox --workdir "$TOX_WORK_DIR" "$@"
