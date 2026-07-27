#!/usr/bin/env bash

set -euo pipefail
set -o noclobber

: "${NSF_SOURCE:?Set NSF_SOURCE to the clean detached full-SHA worktree.}"
: "${NSF_RUN_ROOT:?Set NSF_RUN_ROOT to the fresh immutable run directory.}"
: "${NSF_REVISION:?Set NSF_REVISION to the complete source SHA.}"
: "${NSF_DRIVER_SHA256:?Set NSF_DRIVER_SHA256 to the committed driver digest.}"
: "${NSF_PROTOCOL_SHA256:?Set NSF_PROTOCOL_SHA256 to the frozen protocol digest.}"

module load git/2.45.1-pqk5

preflight="${NSF_RUN_ROOT}/preflight"
log="${preflight}/preflight.log"
smoke="${preflight}/smoke"
local_complete="${preflight}/LOCAL_PREFLIGHT_COMPLETE.json"
driver="${NSF_SOURCE}/examples/rjmcmc/conditional_residual_image_sbi_nsf_tiny_screen.py"
nsf_bin="${NSF_SOURCE}/.pixi/envs/nle-dev/bin"

if [[ "${#NSF_REVISION}" -ne 40 || ! "${NSF_REVISION}" =~ ^[0-9a-f]+$ ]]; then
  echo "NSF_REVISION must be a complete lower-case Git SHA." >&2
  exit 2
fi
for digest_name in NSF_DRIVER_SHA256 NSF_PROTOCOL_SHA256; do
  digest="${!digest_name}"
  if [[ "${#digest}" -ne 64 || ! "${digest}" =~ ^[0-9a-f]+$ ]]; then
    echo "${digest_name} must be a lower-case SHA-256 digest." >&2
    exit 2
  fi
done
if [[ "${NSF_SOURCE}" == *PARIS_inversions* ||
      "${NSF_RUN_ROOT}" == *PARIS_inversions* ]]; then
  echo "NSF source and outputs must not be placed in PARIS_inversions." >&2
  exit 2
fi
for directory in "${NSF_SOURCE}" "${NSF_RUN_ROOT}" "${preflight}"; do
  if [[ ! -d "${directory}" || -L "${directory}" ]]; then
    echo "Required real directory is absent: ${directory}" >&2
    exit 2
  fi
done
if [[ ! -d "${NSF_SOURCE}/.pixi/envs/nle-dev" ||
      -L "${NSF_SOURCE}/.pixi" ]]; then
  echo "The detached source must own its locked nle-dev environment." >&2
  exit 2
fi
if [[ -e "${log}" || -L "${log}" || -e "${smoke}" || -L "${smoke}" ||
      -e "${local_complete}" || -L "${local_complete}" ||
      -e "${preflight}/PREFLIGHT_COMPLETE.json" ]]; then
  echo "Refusing to replace existing NSF preflight evidence." >&2
  exit 2
fi
if [[ "$(git -C "${NSF_SOURCE}" rev-parse HEAD)" != "${NSF_REVISION}" ||
      -n "$(git -C "${NSF_SOURCE}" status --porcelain)" ]]; then
  echo "NSF_SOURCE must be clean and match NSF_REVISION." >&2
  exit 2
fi
observed_driver_sha256="$(sha256sum "${driver}" | awk '{print $1}')"
if [[ "${observed_driver_sha256}" != "${NSF_DRIVER_SHA256}" ]]; then
  echo "The committed NSF driver digest does not match." >&2
  exit 2
fi

export PATH="/user/work/bm13805/.pixi/bin:${PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NSF_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/sbi-nsf-preflight-mpl-${NSF_REVISION}"
export NUMBA_CACHE_DIR="/tmp/sbi-nsf-preflight-numba-${NSF_REVISION}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

: > "${log}"
exec > >(tee -a "${log}") 2>&1

cd "${NSF_SOURCE}"
echo "revision=${NSF_REVISION}"
echo "head=$(git rev-parse HEAD)"
echo "driver_sha256=${NSF_DRIVER_SHA256}"
echo "protocol_sha256=${NSF_PROTOCOL_SHA256}"
/user/work/bm13805/.pixi/bin/pixi --version
"${nsf_bin}/python" -c \
  'import importlib.metadata as m,platform,numpy,scipy,torch,sbi; print(f"python={platform.python_version()}"); print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}"); print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.cuda.is_available()}"); print(f"sbi={sbi.__version__}"); print("nflows="+m.version("nflows"))'
observed_protocol="$(
  "${nsf_bin}/python" -c \
    'from examples.rjmcmc import conditional_residual_image_sbi_nsf_tiny_screen as m; m._configure_torch(); print(m._protocol_sha256())'
)"
if [[ "${observed_protocol}" != "${NSF_PROTOCOL_SHA256}" ]]; then
  echo "The imported NSF protocol digest does not match." >&2
  exit 2
fi

echo "focused_pytest_begin"
"${nsf_bin}/pytest" -q \
  --confcutdir=tests/experimental/rjmcmc \
  tests/experimental/rjmcmc/test_aggregation_error_conditional_sbi_nsf.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_sbi_nsf_driver.py \
  tests/experimental/rjmcmc/test_conditional_residual_image_sbi_nsf_certify.py
echo "focused_pytest_pass"

changed_python=(
  openghg_inversions/experimental/rjmcmc/aggregation_error_conditional_sbi_nsf.py
  examples/rjmcmc/conditional_residual_image_sbi_nsf_tiny_screen.py
  examples/rjmcmc/conditional_residual_image_sbi_nsf_replay.py
  examples/rjmcmc/conditional_residual_image_sbi_nsf_certify.py
  examples/rjmcmc/conditional_residual_image_sbi_nsf_confirmation_certify.py
  tests/experimental/rjmcmc/test_aggregation_error_conditional_sbi_nsf.py
  tests/experimental/rjmcmc/test_conditional_residual_image_sbi_nsf_driver.py
  tests/experimental/rjmcmc/test_conditional_residual_image_sbi_nsf_certify.py
)
echo "focused_ruff_begin"
"${nsf_bin}/ruff" check "${changed_python[@]}"
echo "focused_ruff_pass"

echo "focused_pyright_begin"
"${nsf_bin}/pyright" --project pyrightconfig.nle.json "${changed_python[@]}"
echo "focused_pyright_pass"

echo "smoke_begin"
"${nsf_bin}/python" \
  examples/rjmcmc/conditional_residual_image_sbi_nsf_tiny_screen.py \
  --profile smoke \
  --regime near_gaussian \
  --family two_cell \
  --training-sample-count 4096 \
  --base-seed 731 \
  --source-git-revision "${NSF_REVISION}" \
  --driver-sha256 "${NSF_DRIVER_SHA256}" \
  --output-directory "${smoke}"
"${nsf_bin}/python" -c \
  'import json,sys; marker=json.load(open(sys.argv[1],encoding="utf-8")); assert marker["task_pass"] is True' \
  "${smoke}/near_gaussian__two_cell__root__S4096__base731.complete.json"
artifact="${smoke}/near_gaussian__two_cell__root__S4096__base731.nsf"
artifact_sha256="$(sha256sum "${artifact}" | awk '{print $1}')"
"${nsf_bin}/python" \
  examples/rjmcmc/conditional_residual_image_sbi_nsf_replay.py \
  --artifact "${artifact}" \
  --expected-sha256 "${artifact_sha256}"
echo "smoke_and_separate_process_replay_pass"

printf '{"artifact_sha256":"%s","driver_sha256":"%s","protocol_sha256":"%s","revision":"%s","schema":"rjmcmc-conditional-residual-image-sbi-nsf-local-preflight-complete-v1"}\n' \
  "${artifact_sha256}" "${NSF_DRIVER_SHA256}" "${NSF_PROTOCOL_SHA256}" \
  "${NSF_REVISION}" > "${local_complete}"
