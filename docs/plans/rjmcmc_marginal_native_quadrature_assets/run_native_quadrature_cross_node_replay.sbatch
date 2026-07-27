#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=native-quad-g0-replay
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${NQ_SOURCE:?Set NQ_SOURCE to the clean detached full-SHA worktree.}"
: "${NQ_RUN_ROOT:?Set NQ_RUN_ROOT to the fresh immutable run directory.}"
: "${NQ_REVISION:?Set NQ_REVISION to the complete source SHA.}"
: "${NQ_DRIVER_SHA256:?Set NQ_DRIVER_SHA256 to the committed driver digest.}"
: "${NQ_PROTOCOL_SHA256:?Set NQ_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

preflight="${NQ_RUN_ROOT}/preflight"
local_complete="${preflight}/LOCAL_PREFLIGHT_COMPLETE.json"
complete="${preflight}/PREFLIGHT_COMPLETE.json"
replay_record="${preflight}/cross-node-replay.json"
log="${preflight}/cross-node-replay.${SLURM_JOB_ID}.log"
artifact="${preflight}/smoke/near_gaussian__two_cell__root__O8__base731.nq"
nq_bin="${NQ_SOURCE}/.pixi/envs/nle-dev/bin"

for path in "${local_complete}" "${artifact}"; do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "Required local G0 evidence is absent or unsafe: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${complete}" || -e "${replay_record}" || -e "${log}" ]]; then
  echo "Refusing to replace existing cross-node evidence." >&2
  exit 2
fi
if [[ "${NQ_SOURCE}" == *PARIS_inversions* ||
      "${NQ_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Native-quadrature paths must not be placed in PARIS_inversions." >&2
  exit 2
fi
if [[ "$(git -C "${NQ_SOURCE}" rev-parse HEAD)" != "${NQ_REVISION}" ||
      -n "$(git -C "${NQ_SOURCE}" status --porcelain)" ]]; then
  echo "NQ_SOURCE must be clean and match NQ_REVISION." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NQ_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/native-quad-g0-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/native-quad-g0-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

exec >"${log}" 2>&1
cd "${NQ_SOURCE}"
artifact_sha256="$(
  "${nq_bin}/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["artifact_sha256"])' \
    "${local_complete}"
)"
local_replay_sha256="$(
  "${nq_bin}/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["local_replay_sha256"])' \
    "${local_complete}"
)"
replay="$(
  "${nq_bin}/python" \
    examples/rjmcmc/conditional_native_quadrature_replay.py \
    --artifact "${artifact}" \
    --expected-sha256 "${artifact_sha256}"
)"
"${nq_bin}/python" -c \
  'import json,sys; record=json.loads(sys.argv[1]); assert record["canonical_replay"] is True' \
  "${replay}"
cross_node_replay_sha256="$(
  "${nq_bin}/python" -c \
    'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())' \
    "${replay}"
)"
if [[ "${cross_node_replay_sha256}" != "${local_replay_sha256}" ]]; then
  echo "Cross-node likelihood, gradient, or simulator replay changed." >&2
  exit 2
fi
printf '{"artifact_sha256":"%s","local_replay_sha256":"%s","node":"%s","replay":%s,"schema":"rjmcmc-conditional-native-quadrature-cross-node-replay-v1","slurm_job_id":"%s"}\n' \
  "${artifact_sha256}" "${local_replay_sha256}" "${SLURMD_NODENAME}" \
  "${replay}" "${SLURM_JOB_ID}" \
  > "${replay_record}"
printf '{"cross_node_replay_sha256":"%s","driver_sha256":"%s","protocol_sha256":"%s","revision":"%s","schema":"rjmcmc-conditional-native-quadrature-preflight-complete-v1"}\n' \
  "$(sha256sum "${replay_record}" | awk '{print $1}')" \
  "${NQ_DRIVER_SHA256}" "${NQ_PROTOCOL_SHA256}" "${NQ_REVISION}" \
  > "${complete}"
