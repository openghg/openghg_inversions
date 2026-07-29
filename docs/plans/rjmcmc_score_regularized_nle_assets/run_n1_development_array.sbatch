#!/usr/bin/env bash
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=score-nle-n1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
set -o noclobber

module load git/2.45.1-pqk5

: "${NLE_SOURCE:?Set NLE_SOURCE to a fresh clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to a fresh real run directory.}"
: "${NLE_REVISION:?Set NLE_REVISION to the complete 40-character source SHA.}"
: "${NLE_DRIVER_SHA256:?Set NLE_DRIVER_SHA256 to the committed driver digest.}"
: "${NLE_PROTOCOL_SHA256:?Set NLE_PROTOCOL_SHA256 to the frozen protocol digest.}"
: "${NLE_SAMPLE_COUNT:?Set NLE_SAMPLE_COUNT to one frozen development size.}"

driver="${NLE_SOURCE}/examples/rjmcmc/score_regularized_flow_tiny_screen.py"
development="${NLE_RUN_ROOT}/development"
log_directory="${NLE_RUN_ROOT}/logs/development"
n0_report="${NLE_RUN_ROOT}/preflight/N0_report.json"
n0_complete="${NLE_RUN_ROOT}/preflight/N0_COMPLETE.json"
case_ids=(
  near_gaussian__two_cell__root
  near_gaussian__four_cell__root
  skewed__two_cell__root
  skewed__four_cell__root
  boundary_heavy__two_cell__root
  boundary_heavy__four_cell__root
)
sample_counts=(4096 16384 65536 262144)

validate_hex_digest() {
  local name="$1"
  local value="${!name}"
  if [[ "${#value}" -ne 64 || ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${name} must be a lower-case SHA-256 digest." >&2
    exit 2
  fi
}

validate_real_directory() {
  local directory="$1"
  if [[ ! -d "${directory}" || -L "${directory}" ||
        "$(realpath -e "${directory}")" != "${directory}" ]]; then
    echo "Required canonical real directory is absent: ${directory}" >&2
    exit 2
  fi
}

if [[ "${#NLE_REVISION}" -ne 40 || ! "${NLE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "NLE_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
validate_hex_digest NLE_DRIVER_SHA256
validate_hex_digest NLE_PROTOCOL_SHA256
if [[ "${NLE_SOURCE}" == *PARIS_inversions* ||
      "${NLE_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NLE source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in \
  "${NLE_SOURCE}" \
  "${NLE_RUN_ROOT}" \
  "${development}" \
  "${log_directory}"
do
  validate_real_directory "${directory}"
done
if [[ ! -f "${driver}" || -L "${driver}" ||
      ! -f "${n0_report}" || -L "${n0_report}" ||
      ! -f "${n0_complete}" || -L "${n0_complete}" ]]; then
  echo "The committed driver and authenticated N0 evidence must be real files." >&2
  exit 2
fi
if [[ ! -d "${NLE_SOURCE}/.pixi/envs/nle-dev" ||
      -L "${NLE_SOURCE}/.pixi" ||
      "$(realpath -e "${NLE_SOURCE}/.pixi")" != "${NLE_SOURCE}/.pixi" ]]; then
  echo "The detached source must own its locked real nle-dev environment." >&2
  exit 2
fi
if [[ "$(git -C "${NLE_SOURCE}" rev-parse --verify HEAD)" != "${NLE_REVISION}" ||
      "$(git -C "${NLE_SOURCE}" rev-parse --verify "${NLE_REVISION}^{commit}")" != "${NLE_REVISION}" ]]; then
  echo "NLE_SOURCE does not match the full NLE_REVISION." >&2
  exit 2
fi
if git -C "${NLE_SOURCE}" symbolic-ref -q HEAD >/dev/null; then
  echo "NLE_SOURCE must be a detached worktree." >&2
  exit 2
fi
if [[ -n "$(git -C "${NLE_SOURCE}" status --porcelain --untracked-files=all)" ]]; then
  echo "NLE_SOURCE must be clean." >&2
  exit 2
fi
if [[ "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  echo "The committed NLE driver digest does not match." >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:?N1 requires an explicit Slurm array.}" -ne 6 ||
      "${SLURM_ARRAY_TASK_MIN:?N1 requires an explicit Slurm array.}" -ne 0 ||
      "${SLURM_ARRAY_TASK_MAX:?N1 requires an explicit Slurm array.}" -ne 5 ||
      "${SLURM_ARRAY_TASK_ID:?N1 requires an explicit Slurm array.}" -lt 0 ||
      "${SLURM_ARRAY_TASK_ID}" -gt 5 ]]; then
  echo "N1 must be one complete homogeneous six-case Slurm array." >&2
  exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-1}" -ne 1 ]]; then
  echo "N1 requires exactly one Slurm CPU per task." >&2
  exit 2
fi
valid_sample_count=false
for frozen_sample_count in "${sample_counts[@]}"; do
  if [[ "${NLE_SAMPLE_COUNT}" == "${frozen_sample_count}" ]]; then
    valid_sample_count=true
  fi
done
if [[ "${valid_sample_count}" != true ]]; then
  echo "NLE_SAMPLE_COUNT is not one of the four frozen development sizes." >&2
  exit 2
fi

case_index="${SLURM_ARRAY_TASK_ID}"
case_id="${case_ids[case_index]}"
regime="${case_id%%__*}"
family="${case_id#*__}"
family="${family%%__*}"
sample_count="${NLE_SAMPLE_COUNT}"
stem="${case_id}__S${sample_count}__base731"
artifact="${development}/${stem}.score-flow"
task_report="${development}/${stem}.json"
task_complete="${development}/${stem}.complete.json"
log="${log_directory}/${stem}.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}.log"
for output in "${artifact}" "${task_report}" "${task_complete}" "${log}"; do
  if [[ -e "${output}" || -L "${output}" ]]; then
    echo "Refusing to replace existing N1 evidence: ${output}" >&2
    exit 2
  fi
done
exec >"${log}" 2>&1

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_parallel_codegen_split_count=1"
export PYTHONPATH="${NLE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
cache_root="${TMPDIR:-/tmp}/score-nle-n1-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${cache_root}/xdg"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export JAX_COMPILATION_CACHE_DIR="${cache_root}/jax"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${PYTHONPYCACHEPREFIX}"

cd "${NLE_SOURCE}"
observed_protocol="$(
  pixi run --frozen --no-install -e nle-dev python -c \
    'import os; from examples.rjmcmc import score_regularized_flow_tiny_screen as m; assert os.environ["XLA_FLAGS"] == m.CPU_XLA_FLAGS; print(m._protocol_sha256())'
)"
if [[ "${observed_protocol}" != "${NLE_PROTOCOL_SHA256}" ]]; then
  echo "The imported NLE protocol digest does not match." >&2
  exit 2
fi
pixi run --frozen --no-install -e nle-dev python -c \
  'import hashlib,json,sys; from pathlib import Path; revision,driver_sha,protocol_sha,report_name,marker_name,smoke_name,log_dir_name=sys.argv[1:]; canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True); digest=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(); report_raw=Path(report_name).read_bytes(); marker_raw=Path(marker_name).read_bytes(); report=json.loads(report_raw); marker=json.loads(marker_raw); assert report_raw==(canonical(report)+"\n").encode("ascii"); assert marker_raw==(canonical(marker)+"\n").encode("ascii"); assert set(report)=={"payload","sha256"}; assert set(marker)=={"payload","sha256"}; assert hashlib.sha256(canonical(report["payload"]).encode("ascii")).hexdigest()==report["sha256"]; assert hashlib.sha256(canonical(marker["payload"]).encode("ascii")).hexdigest()==marker["sha256"]; assert marker["payload"]=={"driver_sha256":driver_sha,"protocol_sha256":protocol_sha,"report_payload_sha256":report["sha256"],"report_sha256":hashlib.sha256(report_raw).hexdigest(),"revision":revision,"schema":"score-regularized-flow-n0-complete-v1"}; assert report["payload"]["source"]=={"driver_sha256":driver_sha,"protocol_sha256":protocol_sha,"revision":revision}; assert report["payload"]["scope"]=={"paris_inversions_write":False,"protected_catalogue_access":False,"realized_mf_access":False}; assert all(value=="passed" for value in report["payload"]["checks"].values()); smoke=Path(smoke_name); artifacts=list(smoke.glob("*.score-flow")); reports=[path for path in smoke.glob("*.json") if not path.name.endswith(".complete.json")]; markers=list(smoke.glob("*.complete.json")); logs=list(Path(log_dir_name).glob("*.log")); assert len(artifacts)==len(reports)==len(markers)==len(logs)==1; evidence=report["payload"]["evidence"]; assert evidence=={"log_sha256":digest(logs[0]),"smoke_artifact_sha256":digest(artifacts[0]),"smoke_marker_sha256":digest(markers[0]),"smoke_report_sha256":digest(reports[0])}' \
  "${NLE_REVISION}" \
  "${NLE_DRIVER_SHA256}" \
  "${NLE_PROTOCOL_SHA256}" \
  "${n0_report}" \
  "${n0_complete}" \
  "${NLE_RUN_ROOT}/preflight/smoke" \
  "${NLE_RUN_ROOT}/logs/preflight"

echo "array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "case_id=${case_id}"
echo "training_sample_count=${sample_count}"
echo "base_seed=731"
echo "revision=${NLE_REVISION}"
echo "driver_sha256=${NLE_DRIVER_SHA256}"
echo "protocol_sha256=${NLE_PROTOCOL_SHA256}"

pixi run --frozen --no-install -e nle-dev \
  python examples/rjmcmc/score_regularized_flow_tiny_screen.py \
  --profile development \
  --regime "${regime}" \
  --family "${family}" \
  --training-sample-count "${sample_count}" \
  --base-seed 731 \
  --source-git-revision "${NLE_REVISION}" \
  --driver-sha256 "${NLE_DRIVER_SHA256}" \
  --output-directory "${development}"

if [[ -n "$(git -C "${NLE_SOURCE}" status --porcelain --untracked-files=all)" ||
      "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  echo "NLE_SOURCE changed while the N1 task was running." >&2
  exit 2
fi
# The driver, not this array wrapper, publishes the strict task marker last.
for output in "${artifact}" "${task_report}" "${task_complete}"; do
  if [[ ! -f "${output}" || -L "${output}" ]]; then
    echo "N1 driver did not publish the required real output: ${output}" >&2
    exit 2
  fi
done
pixi run --frozen --no-install -e nle-dev python -c \
  'import hashlib,json,sys; from pathlib import Path; artifact_name,report_name,marker_name,case_id,sample_count,revision,driver_sha,protocol_sha=sys.argv[1:]; canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True); artifact=Path(artifact_name).read_bytes(); report_raw=Path(report_name).read_bytes(); marker_raw=Path(marker_name).read_bytes(); marker=json.loads(marker_raw); report=json.loads(report_raw); assert report_raw==(canonical(report)+"\n").encode("ascii"); assert marker_raw==(canonical(marker)+"\n").encode("ascii"); assert set(report)=={"payload","sha256"}; assert set(report["payload"])=={"artifact","result"}; assert hashlib.sha256(canonical(report["payload"]).encode("ascii")).hexdigest()==report["sha256"]; assert set(marker)=={"artifact_sha256","base_seed","case_id","report_sha256","schema","task_pass","training_sample_count"}; assert marker["schema"]=="rjmcmc-score-regularized-flow-task-complete-v1"; artifact_sha=hashlib.sha256(artifact).hexdigest(); result=report["payload"]["result"]; assert report["payload"]["artifact"]=={"path":Path(artifact_name).name,"sha256":artifact_sha}; assert result["schema"]=="rjmcmc-score-regularized-flow-tiny-screen-v1"; assert result["protocol"]["name"]=="score-regularized-projected-root-nle-v1"; assert result["protocol"]["sha256"]==protocol_sha; assert result["source"]=={"driver_sha256":driver_sha,"git_revision":revision}; assert len(result["attempts"])==2; assert result["selected_initialization"] in (0,1); assert marker["case_id"]==case_id; assert marker["training_sample_count"]==int(sample_count); assert marker["base_seed"]==731; assert marker["artifact_sha256"]==artifact_sha; assert marker["report_sha256"]==hashlib.sha256(report_raw).hexdigest()' \
  "${artifact}" \
  "${task_report}" \
  "${task_complete}" \
  "${case_id}" \
  "${sample_count}" \
  "${NLE_REVISION}" \
  "${NLE_DRIVER_SHA256}" \
  "${NLE_PROTOCOL_SHA256}"
