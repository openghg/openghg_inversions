#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the pre-created immutable run directory.}"
: "${NLE_REVISION:?Set NLE_REVISION to the complete 40-character source SHA.}"
: "${NLE_DRIVER_SHA256:?Set NLE_DRIVER_SHA256 to the committed driver digest.}"
: "${NLE_PROTOCOL_SHA256:?Set NLE_PROTOCOL_SHA256 to the frozen protocol digest.}"

preflight="${NLE_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
smoke="${preflight}/smoke"
complete="${preflight}/PREFLIGHT_COMPLETE.json"
driver="${NLE_SOURCE}/examples/rjmcmc/conditional_residual_image_flow_tiny_screen.py"
nle_bin="${NLE_SOURCE}/.pixi/envs/nle-dev/bin"

if [[ "${#NLE_REVISION}" -ne 40 || ! "${NLE_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "NLE_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
for digest_name in NLE_DRIVER_SHA256 NLE_PROTOCOL_SHA256; do
  digest="${!digest_name}"
  if [[ "${#digest}" -ne 64 || ! "${digest}" =~ ^[0-9a-f]+$ ]]; then
    echo "${digest_name} must be a lower-case SHA-256 digest." >&2
    exit 2
  fi
done
if [[ "${NLE_SOURCE}" == *PARIS_inversions* || "${NLE_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NLE source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${NLE_SOURCE}" "${NLE_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -d "${NLE_SOURCE}/.pixi/envs/nle-dev" ||
      -L "${NLE_SOURCE}/.pixi" ]]; then
  echo "The detached source must own its locked nle-dev environment." >&2
  exit 2
fi
if [[ -e "${log}" || -L "${log}" || -e "${smoke}" || -L "${smoke}" ||
      -e "${complete}" || -L "${complete}" ]]; then
  echo "Refusing to replace existing NLE preflight evidence." >&2
  exit 2
fi
if [[ "$(git -C "${NLE_SOURCE}" rev-parse HEAD)" != "${NLE_REVISION}" ]]; then
  echo "NLE_SOURCE does not match NLE_REVISION." >&2
  exit 2
fi
if [[ -n "$(git -C "${NLE_SOURCE}" status --porcelain)" ]]; then
  echo "NLE_SOURCE must be clean." >&2
  exit 2
fi
if [[ "$(sha256sum "${driver}" | awk '{print $1}')" != "${NLE_DRIVER_SHA256}" ]]; then
  echo "The committed NLE driver digest does not match." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH="${NLE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/flow-nle-preflight-matplotlib-${NLE_REVISION}"
export NUMBA_CACHE_DIR="/tmp/flow-nle-preflight-numba-${NLE_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: > "${log}"
exec > >(tee -a "${log}") 2>&1

cd "${NLE_SOURCE}"
echo "revision=${NLE_REVISION}"
echo "head=$(git rev-parse HEAD)"
echo "driver_sha256=${NLE_DRIVER_SHA256}"
echo "protocol_sha256=${NLE_PROTOCOL_SHA256}"
pixi --version
"${nle_bin}/python" -c \
  'import platform,numpy,scipy,jax,jaxlib,flowjax,equinox,optax,paramax; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}"); print(f"jax={jax.__version__}"); print(f"jaxlib={jaxlib.__version__}"); print(f"flowjax={flowjax.__version__}"); print(f"equinox={equinox.__version__}"); print(f"optax={optax.__version__}"); print(f"paramax={paramax.__version__}")'
observed_protocol="$(
  "${nle_bin}/python" -c \
    'from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as m; print(m._protocol_sha256())'
)"
if [[ "${observed_protocol}" != "${NLE_PROTOCOL_SHA256}" ]]; then
  echo "The imported NLE protocol digest does not match." >&2
  exit 2
fi

echo "focused_pytest_begin"
"${nle_bin}/pytest" -q \
  --confcutdir=tests/experimental/rjmcmc \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_flow.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_flow_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_flow_certify.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_flow_confirmation_certify.py
echo "focused_pytest_pass"

echo "focused_ruff_begin"
"${nle_bin}/ruff" check \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_flow.py \
  examples/rjmcmc/conditional_residual_image_flow_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_flow_certify.py \
  examples/rjmcmc/conditional_residual_image_flow_confirmation_certify.py \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_flow.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_flow_tiny_screen.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_flow_certify.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_flow_confirmation_certify.py
echo "focused_ruff_pass"

echo "focused_pyright_begin"
"${nle_bin}/pyright" \
  --project pyrightconfig.nle.json \
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_flow.py \
  examples/rjmcmc/conditional_residual_image_flow_tiny_screen.py \
  examples/rjmcmc/conditional_residual_image_flow_certify.py \
  examples/rjmcmc/conditional_residual_image_flow_confirmation_certify.py
echo "focused_pyright_pass"

echo "smoke_begin"
"${nle_bin}/python" \
  examples/rjmcmc/conditional_residual_image_flow_tiny_screen.py \
  --profile smoke \
  --regime near_gaussian \
  --family two_cell \
  --training-sample-count 4096 \
  --base-seed 731 \
  --source-git-revision "${NLE_REVISION}" \
  --driver-sha256 "${NLE_DRIVER_SHA256}" \
  --output-directory "${smoke}"
"${nle_bin}/python" -c \
  'import json,sys; marker=json.load(open(sys.argv[1],encoding="utf-8")); assert marker["task_pass"] is True' \
  "${smoke}/near_gaussian__two_cell__root__S4096__base731.complete.json"
"${nle_bin}/python" -c \
  'import hashlib,json,sys; from pathlib import Path; from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_flow import ConditionalResidualImageFlow; report=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); record=report["payload"]["artifact"]; raw=Path(sys.argv[2]).read_bytes(); assert hashlib.sha256(raw).hexdigest()==record["sha256"]; replay=ConditionalResidualImageFlow.from_bytes(raw,expected_sha256=record["sha256"]); assert replay.to_bytes()==raw' \
  "${smoke}/near_gaussian__two_cell__root__S4096__base731.json" \
  "${smoke}/near_gaussian__two_cell__root__S4096__base731.flow"
echo "smoke_pass"

printf '{"driver_sha256":"%s","protocol_sha256":"%s","revision":"%s","schema":"rjmcmc-conditional-residual-image-flow-preflight-complete-v1"}\n' \
  "${NLE_DRIVER_SHA256}" "${NLE_PROTOCOL_SHA256}" "${NLE_REVISION}" > "${complete}"
