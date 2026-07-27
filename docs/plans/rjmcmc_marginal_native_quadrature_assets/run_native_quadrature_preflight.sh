#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${NQ_SOURCE:?Set NQ_SOURCE to the clean detached full-SHA worktree.}"
: "${NQ_RUN_ROOT:?Set NQ_RUN_ROOT to the fresh immutable run directory.}"
: "${NQ_REVISION:?Set NQ_REVISION to the complete source SHA.}"
: "${NQ_DRIVER_SHA256:?Set NQ_DRIVER_SHA256 to the committed driver digest.}"
: "${NQ_PROTOCOL_SHA256:?Set NQ_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

preflight="${NQ_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
smoke="${preflight}/smoke"
local_complete="${preflight}/LOCAL_PREFLIGHT_COMPLETE.json"
driver="${NQ_SOURCE}/examples/rjmcmc/conditional_native_quadrature_tiny_screen.py"
nq_bin="${NQ_SOURCE}/.pixi/envs/nle-dev/bin"

if [[ "${#NQ_REVISION}" -ne 40 || ! "${NQ_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "NQ_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
for digest_name in NQ_DRIVER_SHA256 NQ_PROTOCOL_SHA256; do
  digest="${!digest_name}"
  if [[ "${#digest}" -ne 64 || ! "${digest}" =~ ^[0-9a-f]+$ ]]; then
    echo "${digest_name} must be a lower-case SHA-256 digest." >&2
    exit 2
  fi
done
if [[ "${NQ_SOURCE}" == *PARIS_inversions* ||
      "${NQ_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "Native-quadrature source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${NQ_SOURCE}" "${NQ_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -d "${NQ_SOURCE}/.pixi/envs/nle-dev" ||
      -L "${NQ_SOURCE}/.pixi" ]]; then
  echo "The detached source must own its locked nle-dev environment." >&2
  exit 2
fi
if [[ -e "${log}" || -L "${log}" || -e "${smoke}" || -L "${smoke}" ||
      -e "${local_complete}" || -L "${local_complete}" ||
      -e "${preflight}/PREFLIGHT_COMPLETE.json" ]]; then
  echo "Refusing to replace existing native-quadrature preflight evidence." >&2
  exit 2
fi
if [[ "$(git -C "${NQ_SOURCE}" rev-parse HEAD)" != "${NQ_REVISION}" ||
      -n "$(git -C "${NQ_SOURCE}" status --porcelain)" ]]; then
  echo "NQ_SOURCE must be clean and match NQ_REVISION." >&2
  exit 2
fi
observed_driver_sha256="$(sha256sum "${driver}" | awk '{print $1}')"
if [[ "${observed_driver_sha256}" != "${NQ_DRIVER_SHA256}" ]]; then
  echo "The committed native-quadrature driver digest does not match." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NQ_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/native-quadrature-preflight-mpl-${NQ_REVISION}"
export NUMBA_CACHE_DIR="/tmp/native-quadrature-preflight-numba-${NQ_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: > "${log}"
exec > >(tee -a "${log}") 2>&1

cd "${NQ_SOURCE}"
echo "revision=${NQ_REVISION}"
echo "head=$(git rev-parse HEAD)"
echo "driver_sha256=${NQ_DRIVER_SHA256}"
echo "protocol_sha256=${NQ_PROTOCOL_SHA256}"
/user/work/bm13805/.pixi/bin/pixi --version
"${nq_bin}/python" -c \
  'import platform,numpy,scipy; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
observed_protocol="$(
  "${nq_bin}/python" -c \
    'from examples.rjmcmc import conditional_native_quadrature_tiny_screen as m; print(m._protocol_sha256())'
)"
if [[ "${observed_protocol}" != "${NQ_PROTOCOL_SHA256}" ]]; then
  echo "The imported native-quadrature protocol digest does not match." >&2
  exit 2
fi

echo "focused_pytest_begin"
"${nq_bin}/pytest" -q \
  --confcutdir=tests/experimental/rjmcmc \
  tests/experimental/rjmcmc/test_aggregation_error_native_quadrature.py \
  tests/experimental/rjmcmc/test_conditional_native_quadrature_driver.py \
  tests/experimental/rjmcmc/test_conditional_native_quadrature_certify.py
echo "focused_pytest_pass"

changed_python=(
  openghg_inversions/experimental/rjmcmc/aggregation_error_native_quadrature.py
  examples/rjmcmc/conditional_native_quadrature_tiny_screen.py
  examples/rjmcmc/conditional_native_quadrature_replay.py
  examples/rjmcmc/conditional_native_quadrature_certify.py
  examples/rjmcmc/conditional_native_quadrature_confirmation.py
  examples/rjmcmc/conditional_native_quadrature_confirmation_certify.py
  tests/experimental/rjmcmc/test_aggregation_error_native_quadrature.py
  tests/experimental/rjmcmc/test_conditional_native_quadrature_driver.py
  tests/experimental/rjmcmc/test_conditional_native_quadrature_certify.py
)
echo "focused_ruff_begin"
"${nq_bin}/ruff" check "${changed_python[@]}"
echo "focused_ruff_pass"

echo "focused_pyright_begin"
"${nq_bin}/pyright" --project pyrightconfig.nle.json "${changed_python[@]}"
echo "focused_pyright_pass"

echo "shell_syntax_begin"
for script in docs/plans/rjmcmc_marginal_native_quadrature_assets/*; do
  bash -n "${script}"
done
echo "shell_syntax_pass"

echo "smoke_begin"
"${nq_bin}/python" \
  examples/rjmcmc/conditional_native_quadrature_tiny_screen.py \
  --profile smoke \
  --regime near_gaussian \
  --family two_cell \
  --quadrature-order 8 \
  --base-seed 731 \
  --source-git-revision "${NQ_REVISION}" \
  --driver-sha256 "${NQ_DRIVER_SHA256}" \
  --output-directory "${smoke}"
"${nq_bin}/python" -c \
  'import json,sys; marker=json.load(open(sys.argv[1],encoding="utf-8")); assert marker["task_pass"] is True' \
  "${smoke}/near_gaussian__two_cell__root__O8__base731.complete.json"
artifact="${smoke}/near_gaussian__two_cell__root__O8__base731.nq"
artifact_sha256="$(sha256sum "${artifact}" | awk '{print $1}')"
local_replay="$(
  "${nq_bin}/python" \
    examples/rjmcmc/conditional_native_quadrature_replay.py \
    --artifact "${artifact}" \
    --expected-sha256 "${artifact_sha256}"
)"
"${nq_bin}/python" -c \
  'import json,sys; record=json.loads(sys.argv[1]); assert record["canonical_replay"] is True' \
  "${local_replay}"
local_replay_sha256="$(
  "${nq_bin}/python" -c \
    'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest())' \
    "${local_replay}"
)"
printf '%s\n' "${local_replay}"
echo "smoke_and_separate_process_replay_pass"

printf '{"artifact_sha256":"%s","driver_sha256":"%s","local_replay_sha256":"%s","protocol_sha256":"%s","revision":"%s","schema":"rjmcmc-conditional-native-quadrature-local-preflight-complete-v1"}\n' \
  "${artifact_sha256}" "${NQ_DRIVER_SHA256}" "${local_replay_sha256}" \
  "${NQ_PROTOCOL_SHA256}" "${NQ_REVISION}" > "${local_complete}"
