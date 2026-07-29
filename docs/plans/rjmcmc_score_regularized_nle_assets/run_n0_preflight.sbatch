#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=score-nle-n0
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

driver="${NLE_SOURCE}/examples/rjmcmc/score_regularized_flow_tiny_screen.py"
preflight="${NLE_RUN_ROOT}/preflight"
log_directory="${NLE_RUN_ROOT}/logs/preflight"
log="${log_directory}/N0.${SLURM_JOB_ID:?N0 must run under Slurm.}.log"
smoke="${preflight}/smoke"
report="${preflight}/N0_report.json"
complete="${preflight}/N0_COMPLETE.json"

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
  "${preflight}" \
  "${log_directory}"
do
  validate_real_directory "${directory}"
done
if [[ ! -d "${NLE_SOURCE}/.pixi/envs/nle-dev" ||
      -L "${NLE_SOURCE}/.pixi" ||
      "$(realpath -e "${NLE_SOURCE}/.pixi")" != "${NLE_SOURCE}/.pixi" ]]; then
  echo "The detached source must own its locked real nle-dev environment." >&2
  exit 2
fi
if [[ ! -f "${driver}" || -L "${driver}" ]]; then
  echo "The committed score-regularized NLE driver is absent." >&2
  exit 2
fi
if [[ -n "$(find "${preflight}" -mindepth 1 -maxdepth 1 -print -quit)" ||
      -n "$(find "${log_directory}" -mindepth 1 -maxdepth 1 -print -quit)" ||
      -n "$(find "${NLE_RUN_ROOT}" -mindepth 1 \( -type f -o -type l \) -print -quit)" ||
      -e "${log}" || -L "${log}" ||
      -e "${smoke}" || -L "${smoke}" ||
      -e "${report}" || -L "${report}" ||
      -e "${complete}" || -L "${complete}" ]]; then
  echo "Refusing to replace existing N0 evidence." >&2
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
if [[ "${SLURM_CPUS_PER_TASK:-1}" -ne 1 ]]; then
  echo "N0 requires exactly one Slurm CPU." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH="${NLE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
cache_root="${TMPDIR:-/tmp}/score-nle-n0-${SLURM_JOB_ID}"
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

run_preflight() {
  cd "${NLE_SOURCE}"
  echo "revision=${NLE_REVISION}"
  echo "head=$(git rev-parse HEAD)"
  echo "detached=true"
  echo "driver_sha256=${NLE_DRIVER_SHA256}"
  echo "protocol_sha256=${NLE_PROTOCOL_SHA256}"
  pixi --version
  pixi run --frozen --no-install -e nle-dev python -c \
    'import platform,numpy,scipy,jax,jaxlib,flowjax,equinox,optax,paramax; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}"); print(f"jax={jax.__version__}"); print(f"jaxlib={jaxlib.__version__}"); print(f"flowjax={flowjax.__version__}"); print(f"equinox={equinox.__version__}"); print(f"optax={optax.__version__}"); print(f"paramax={paramax.__version__}"); assert jax.config.jax_enable_x64'
  local observed_protocol
  observed_protocol="$(
    pixi run --frozen --no-install -e nle-dev python -c \
      'from examples.rjmcmc import score_regularized_flow_tiny_screen as m; print(m._protocol_sha256())'
  )"
  if [[ "${observed_protocol}" != "${NLE_PROTOCOL_SHA256}" ]]; then
    echo "The imported NLE protocol digest does not match." >&2
    return 2
  fi

  echo "focused_pytest_begin"
  pixi run --frozen --no-install -e nle-dev pytest -q \
    --confcutdir=tests/experimental/rjmcmc \
    tests/experimental/rjmcmc/test_aggregation_error_score_regularized_flow.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_flow_training.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_flow_optimizer.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_flow_artifact.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_tiny_domains.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_tiny_screen.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_tiny_certify.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_slurm_assets.py
  echo "focused_pytest_pass"

  echo "focused_ruff_begin"
  pixi run --frozen --no-install -e nle-dev ruff check \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_regularized_flow.py \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_flow_training.py \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_flow_optimizer.py \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_flow_artifact.py \
    examples/rjmcmc/score_regularized_flow_tiny_domains.py \
    examples/rjmcmc/score_regularized_flow_tiny_screen.py \
    examples/rjmcmc/score_regularized_flow_tiny_certify.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_regularized_flow.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_flow_training.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_flow_optimizer.py \
    tests/experimental/rjmcmc/test_aggregation_error_score_flow_artifact.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_tiny_domains.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_tiny_screen.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_tiny_certify.py \
    tests/experimental/rjmcmc/test_score_regularized_flow_slurm_assets.py
  echo "focused_ruff_pass"

  echo "focused_pyright_begin"
  pixi run --frozen --no-install -e nle-dev pyright \
    --project pyrightconfig.nle.json \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_regularized_flow.py \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_flow_training.py \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_flow_optimizer.py \
    openghg_inversions/experimental/rjmcmc/aggregation_error_score_flow_artifact.py \
    examples/rjmcmc/score_regularized_flow_tiny_domains.py \
    examples/rjmcmc/score_regularized_flow_tiny_screen.py \
    examples/rjmcmc/score_regularized_flow_tiny_certify.py
  echo "focused_pyright_pass"

  echo "driver_smoke_begin"
  pixi run --frozen --no-install -e nle-dev \
    python examples/rjmcmc/score_regularized_flow_tiny_screen.py \
    --profile smoke \
    --regime near_gaussian \
    --family two_cell \
    --training-sample-count 64 \
    --base-seed 731 \
    --source-git-revision "${NLE_REVISION}" \
    --driver-sha256 "${NLE_DRIVER_SHA256}" \
    --output-directory "${smoke}"

  mapfile -t smoke_artifacts < <(find "${smoke}" -mindepth 1 -maxdepth 1 -type f -name '*.score-flow' -print)
  mapfile -t smoke_reports < <(find "${smoke}" -mindepth 1 -maxdepth 1 -type f -name '*.json' ! -name '*.complete.json' -print)
  mapfile -t smoke_markers < <(find "${smoke}" -mindepth 1 -maxdepth 1 -type f -name '*.complete.json' -print)
  if [[ "${#smoke_artifacts[@]}" -ne 1 ||
        "${#smoke_reports[@]}" -ne 1 ||
        "${#smoke_markers[@]}" -ne 1 ]]; then
    echo "The smoke driver must publish exactly one artifact, report, and marker." >&2
    return 2
  fi
  if [[ -L "${smoke_artifacts[0]}" ||
        -L "${smoke_reports[0]}" ||
        -L "${smoke_markers[0]}" ]]; then
    echo "The smoke driver must publish real files, not symlinks." >&2
    return 2
  fi

  echo "separate_process_artifact_replay_begin"
  pixi run --frozen --no-install -e nle-dev python -c \
    'import hashlib,json,sys; from pathlib import Path; from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import ScoreRegularizedRootFlow; canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True); report_path=Path(sys.argv[1]); artifact_path=Path(sys.argv[2]); marker_path=Path(sys.argv[3]); revision,driver_sha,protocol_sha=sys.argv[4:]; report_raw=report_path.read_bytes(); marker_raw=marker_path.read_bytes(); report=json.loads(report_raw); marker=json.loads(marker_raw); assert report_raw==(canonical(report)+"\n").encode("ascii"); assert marker_raw==(canonical(marker)+"\n").encode("ascii"); assert set(report)=={"payload","sha256"}; assert set(report["payload"])=={"artifact","result"}; assert hashlib.sha256(canonical(report["payload"]).encode("ascii")).hexdigest()==report["sha256"]; assert set(marker)=={"artifact_sha256","base_seed","case_id","report_sha256","schema","task_pass","training_sample_count"}; assert marker["schema"]=="rjmcmc-score-regularized-flow-task-complete-v1"; result=report["payload"]["result"]; assert result["protocol"]["name"]=="score-regularized-projected-root-nle-v1"; assert result["protocol"]["sha256"]==protocol_sha; assert result["source"]=={"driver_sha256":driver_sha,"git_revision":revision}; record=report["payload"]["artifact"]; raw=artifact_path.read_bytes(); digest=hashlib.sha256(raw).hexdigest(); assert record=={"path":artifact_path.name,"sha256":digest}; assert marker["artifact_sha256"]==digest; assert marker["report_sha256"]==hashlib.sha256(report_raw).hexdigest(); assert marker["task_pass"] is True; replay=ScoreRegularizedRootFlow.from_bytes(raw,expected_sha256=digest); assert replay.to_bytes()==raw; assert replay.artifact_sha256==digest' \
    "${smoke_reports[0]}" \
    "${smoke_artifacts[0]}" \
    "${smoke_markers[0]}" \
    "${NLE_REVISION}" \
    "${NLE_DRIVER_SHA256}" \
    "${NLE_PROTOCOL_SHA256}"
  echo "separate_process_artifact_replay_pass"
  printf '%s\n' "${smoke_artifacts[0]}" "${smoke_reports[0]}" "${smoke_markers[0]}"
  echo "driver_smoke_pass"
}

run_preflight >"${log}" 2>&1

if [[ -n "$(git -C "${NLE_SOURCE}" status --porcelain --untracked-files=all)" ||
      "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  exit 2
fi
mapfile -t final_smoke_artifacts < <(find "${smoke}" -mindepth 1 -maxdepth 1 -type f -name '*.score-flow' -print)
mapfile -t final_smoke_reports < <(find "${smoke}" -mindepth 1 -maxdepth 1 -type f -name '*.json' ! -name '*.complete.json' -print)
mapfile -t final_smoke_markers < <(find "${smoke}" -mindepth 1 -maxdepth 1 -type f -name '*.complete.json' -print)
if [[ "${#final_smoke_artifacts[@]}" -ne 1 ||
      "${#final_smoke_reports[@]}" -ne 1 ||
      "${#final_smoke_markers[@]}" -ne 1 ]]; then
  exit 2
fi

# This process writes the canonical report and then N0_COMPLETE.json.  The
# completion marker is the final filesystem publication performed by N0.
pixi run --frozen --no-install -e nle-dev python -c \
  'import hashlib,json,sys; from pathlib import Path; revision,driver_sha,protocol_sha,log_name,artifact_name,task_report_name,task_marker_name,report_name,marker_name=sys.argv[1:]; canonical=lambda value: json.dumps(value,allow_nan=False,ensure_ascii=True,separators=(",",":"),sort_keys=True); digest=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(); payload={"checks":{"artifact_replay":"passed","driver_smoke":"passed","focused_pyright":"passed","focused_pytest":"passed","focused_ruff":"passed","protocol_identity":"passed","source_clean_after_checks":"passed"},"evidence":{"log_sha256":digest(log_name),"smoke_artifact_sha256":digest(artifact_name),"smoke_marker_sha256":digest(task_marker_name),"smoke_report_sha256":digest(task_report_name)},"schema":"score-regularized-flow-n0-report-v1","scope":{"paris_inversions_write":False,"protected_catalogue_access":False,"realized_mf_access":False},"source":{"driver_sha256":driver_sha,"protocol_sha256":protocol_sha,"revision":revision}}; envelope={"payload":payload,"sha256":hashlib.sha256(canonical(payload).encode("ascii")).hexdigest()}; report_bytes=(canonical(envelope)+"\n").encode("ascii"); Path(report_name).open("xb").write(report_bytes); marker_payload={"driver_sha256":driver_sha,"protocol_sha256":protocol_sha,"report_payload_sha256":envelope["sha256"],"report_sha256":hashlib.sha256(report_bytes).hexdigest(),"revision":revision,"schema":"score-regularized-flow-n0-complete-v1"}; marker={"payload":marker_payload,"sha256":hashlib.sha256(canonical(marker_payload).encode("ascii")).hexdigest()}; Path(marker_name).open("xb").write((canonical(marker)+"\n").encode("ascii"))' \
  "${NLE_REVISION}" \
  "${NLE_DRIVER_SHA256}" \
  "${NLE_PROTOCOL_SHA256}" \
  "${log}" \
  "${final_smoke_artifacts[0]}" \
  "${final_smoke_reports[0]}" \
  "${final_smoke_markers[0]}" \
  "${report}" \
  "${complete}"
