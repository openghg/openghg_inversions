#!/usr/bin/env bash
#SBATCH --job-name=rjmcmc-nle-compile-canary
#SBATCH --array=0-1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${NLE_SOURCE:?Set NLE_SOURCE to the detached exact-SHA source.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the fresh score-NLE run root.}"
: "${NLE_REVISION:?Set NLE_REVISION to the exact full source SHA.}"
: "${NLE_DRIVER_SHA256:?Set NLE_DRIVER_SHA256 to the tiny-screen driver digest.}"
: "${NLE_PROTOCOL_SHA256:?Set NLE_PROTOCOL_SHA256 to the frozen protocol digest.}"

source_root="$(realpath -e "${NLE_SOURCE}")"
run_root="$(realpath -e "${NLE_RUN_ROOT}")"
case_index="${SLURM_ARRAY_TASK_ID:?Compile canary must run as an array.}"
revision="$(git -C "${source_root}" rev-parse HEAD)"
dimensions=(1 3)
if [[ "${SLURM_ARRAY_TASK_COUNT:?}" -ne 2 ||
      "${SLURM_ARRAY_TASK_MIN:?}" -ne 0 ||
      "${SLURM_ARRAY_TASK_MAX:?}" -ne 1 ||
      "${case_index}" -lt 0 ||
      "${case_index}" -gt 1 ]]; then
  echo "Compile canary requires the complete q=1,q=3 array." >&2
  exit 64
fi
dimension="${dimensions[case_index]}"
preflight="${run_root}/preflight"
n0_report="${preflight}/N0_report.json"
n0_complete="${preflight}/N0_COMPLETE.json"
output_directory="${run_root}/compile-canary"
log_directory="${run_root}/logs/compile-canary"
log="${log_directory}/q${dimension}.${SLURM_ARRAY_JOB_ID}_${case_index}.log"
cache_root="${TMPDIR:-/tmp}/score-nle-canary-${SLURM_ARRAY_JOB_ID}_${case_index}"
driver="${source_root}/examples/rjmcmc/score_regularized_flow_tiny_screen.py"
report="${output_directory}/q${dimension}.report.json"
complete="${output_directory}/q${dimension}.complete.json"

if [[ "${source_root}" == *PARIS_inversions* || "${run_root}" == *PARIS_inversions* ]]; then
  echo "Refusing a PARIS_inversions path." >&2
  exit 64
fi
if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ||
      "${revision}" != "${NLE_REVISION}" ]]; then
  echo "Source revision is not a full Git SHA." >&2
  exit 64
fi
if [[ "$(git -C "${source_root}" symbolic-ref -q HEAD || true)" != "" ]]; then
  echo "NLE_SOURCE must be detached." >&2
  exit 64
fi
if [[ -n "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]]; then
  echo "NLE_SOURCE must be clean." >&2
  exit 64
fi
if [[ ! -f "${driver}" || -L "${driver}" ||
      ! -f "${n0_report}" || -L "${n0_report}" ||
      ! -f "${n0_complete}" || -L "${n0_complete}" ]]; then
  echo "Authenticated N0 evidence is required." >&2
  exit 64
fi
if [[ "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  echo "The committed NLE driver digest does not match." >&2
  exit 64
fi

mkdir -p "${output_directory}" "${log_directory}" "${cache_root}"
for output in "${report}" "${complete}" "${log}"; do
  if [[ -e "${output}" || -L "${output}" ]]; then
    echo "Refusing to replace existing compile-canary evidence: ${output}" >&2
    exit 73
  fi
done
exec >"${log}" 2>&1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_parallel_codegen_split_count=1"
export JAX_COMPILATION_CACHE_DIR="${cache_root}/jax"
export XDG_CACHE_HOME="${cache_root}/xdg"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"
mkdir -p \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}"

python_bin="${source_root}/.pixi/envs/nle-dev/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "The frozen nle-dev Python is missing." >&2
  exit 64
fi
export PYTHONPATH="${source_root}${PYTHONPATH:+:${PYTHONPATH}}"

observed_protocol="$(
  "${python_bin}" -c \
    'import os; from examples.rjmcmc import score_regularized_flow_tiny_screen as m; assert os.environ["XLA_FLAGS"] == m.CPU_XLA_FLAGS; print(m._protocol_sha256())'
)"
if [[ "${observed_protocol}" != "${NLE_PROTOCOL_SHA256}" ]]; then
  echo "The imported NLE protocol digest does not match." >&2
  exit 64
fi
"${python_bin}" -c \
  'import hashlib,json,sys; from pathlib import Path; revision,driver_sha,protocol_sha,report_name,marker_name=sys.argv[1:]; canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True); report_raw=Path(report_name).read_bytes(); marker_raw=Path(marker_name).read_bytes(); report=json.loads(report_raw); marker=json.loads(marker_raw); assert report_raw==(canonical(report)+"\n").encode("ascii"); assert marker_raw==(canonical(marker)+"\n").encode("ascii"); assert set(report)=={"payload","sha256"}; assert set(marker)=={"payload","sha256"}; assert hashlib.sha256(canonical(report["payload"]).encode("ascii")).hexdigest()==report["sha256"]; assert hashlib.sha256(canonical(marker["payload"]).encode("ascii")).hexdigest()==marker["sha256"]; assert marker["payload"]=={"driver_sha256":driver_sha,"protocol_sha256":protocol_sha,"report_payload_sha256":report["sha256"],"report_sha256":hashlib.sha256(report_raw).hexdigest(),"revision":revision,"schema":"score-regularized-flow-n0-complete-v1"}; assert report["payload"]["source"]=={"driver_sha256":driver_sha,"protocol_sha256":protocol_sha,"revision":revision}; assert all(value=="passed" for value in report["payload"]["checks"].values())' \
  "${NLE_REVISION}" \
  "${NLE_DRIVER_SHA256}" \
  "${NLE_PROTOCOL_SHA256}" \
  "${n0_report}" \
  "${n0_complete}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${case_index}"
echo "dimension=${dimension}"
echo "revision=${revision}"

"${python_bin}" -m examples.rjmcmc.score_regularized_flow_compile_canary \
  --dimension "${dimension}" \
  --source-git-revision "${revision}" \
  --output-directory "${output_directory}"
