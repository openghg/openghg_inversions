#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=sbi-nsf-g0-replay
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

: "${NSF_SOURCE:?Set NSF_SOURCE to the clean detached full-SHA worktree.}"
: "${NSF_RUN_ROOT:?Set NSF_RUN_ROOT to the fresh immutable run directory.}"
: "${NSF_REVISION:?Set NSF_REVISION to the complete source SHA.}"
: "${NSF_DRIVER_SHA256:?Set NSF_DRIVER_SHA256 to the committed driver digest.}"
: "${NSF_PROTOCOL_SHA256:?Set NSF_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

preflight="${NSF_RUN_ROOT}/preflight"
local_complete="${preflight}/LOCAL_PREFLIGHT_COMPLETE.json"
complete="${preflight}/PREFLIGHT_COMPLETE.json"
replay_record="${preflight}/cross-node-replay.json"
log="${preflight}/cross-node-replay.${SLURM_JOB_ID}.log"
artifact="${preflight}/smoke/near_gaussian__two_cell__root__S4096__base731.nsf"
nsf_bin="${NSF_SOURCE}/.pixi/envs/nle-dev/bin"

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
if [[ "${NSF_SOURCE}" == *PARIS_inversions* ||
      "${NSF_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NSF source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
if [[ "$(git -C "${NSF_SOURCE}" rev-parse HEAD)" != "${NSF_REVISION}" ||
      -n "$(git -C "${NSF_SOURCE}" status --porcelain)" ]]; then
  echo "NSF_SOURCE must be clean and match NSF_REVISION." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NSF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sbi-nsf-g0-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/sbi-nsf-g0-numba-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

exec >"${log}" 2>&1
cd "${NSF_SOURCE}"
artifact_sha256="$(
  "${nsf_bin}/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["artifact_sha256"])' \
    "${local_complete}"
)"
replay="$(
  "${nsf_bin}/python" \
    examples/rjmcmc/conditional_residual_image_sbi_nsf_replay.py \
    --artifact "${artifact}" \
    --expected-sha256 "${artifact_sha256}"
)"
"${nsf_bin}/python" -c \
  'import json,sys; record=json.loads(sys.argv[1]); assert record["canonical_replay"] is True' \
  "${replay}"
printf '{"artifact_sha256":"%s","node":"%s","replay":%s,"schema":"rjmcmc-conditional-residual-image-sbi-nsf-cross-node-replay-v1","slurm_job_id":"%s"}\n' \
  "${artifact_sha256}" "${SLURMD_NODENAME}" "${replay}" "${SLURM_JOB_ID}" \
  > "${replay_record}"
printf '{"cross_node_replay_sha256":"%s","driver_sha256":"%s","protocol_sha256":"%s","revision":"%s","schema":"rjmcmc-conditional-residual-image-sbi-nsf-preflight-complete-v1"}\n' \
  "$(sha256sum "${replay_record}" | awk '{print $1}')" \
  "${NSF_DRIVER_SHA256}" "${NSF_PROTOCOL_SHA256}" "${NSF_REVISION}" \
  > "${complete}"
