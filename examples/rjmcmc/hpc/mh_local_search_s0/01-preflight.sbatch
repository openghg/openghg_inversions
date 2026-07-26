#!/usr/bin/env bash
#SBATCH --account=chem007981
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=mh-s0-preflight
#SBATCH --output=%x-%j.out

set -euo pipefail
module load git/2.45.1-pqk5
source "${HARNESS_DIR:?}/00-common.sh"
harness_job_init preflight
output="${RUN_ROOT}/01-preflight"
mkdir -p "${output}"
(
    cd "${REPO_ROOT}"
    "${PIXI_EXE}" --version
    "${PIXI_EXE}" run --as-is -e dev pytest -q \
        tests/experimental/rjmcmc/test_mh_local_search_synthetic.py \
        tests/experimental/rjmcmc/test_mh_local_search_flow_oracle.py \
        tests/experimental/rjmcmc/test_mh_local_search_nuts_reference.py \
        tests/experimental/rjmcmc/test_mh_local_search_conditional_reference.py \
        tests/experimental/rjmcmc/test_mh_local_search_retry_authorization.py \
        tests/experimental/rjmcmc/test_mh_local_search_hpc_harness.py
    "${PIXI_EXE}" run --as-is -e dev ruff check \
        examples/rjmcmc/mh_local_search_*.py \
        openghg_inversions/experimental/rjmcmc/mh_local_search_*.py \
        examples/rjmcmc/hpc/mh_local_search_s0/*.py
    "${PIXI_EXE}" run --as-is -e dev ruff format --check \
        examples/rjmcmc/mh_local_search_*.py \
        openghg_inversions/experimental/rjmcmc/mh_local_search_*.py \
        examples/rjmcmc/hpc/mh_local_search_s0/*.py
    "${PIXI_EXE}" run --as-is -e dev pyright \
        examples/rjmcmc/mh_local_search_*.py \
        openghg_inversions/experimental/rjmcmc/mh_local_search_*.py \
        examples/rjmcmc/hpc/mh_local_search_s0/*.py
) >"${output}/checks.log" 2>&1
pixi_version="$("${PIXI_EXE}" --version)"
pixi_sha256="$(sha256_file "${PIXI_EXE}")"
pixi_python -c '
import importlib.metadata
import json
import numpy
import platform
import pymc
import pytensor
import sys
from pathlib import Path

output, pixi_exe, pixi_version, pixi_sha256 = sys.argv[1:5]
payload = {
    "schema": "openghg_inversions.mh_local_search_s0_preflight_provenance.v1",
    "source_revision": sys.argv[5],
    "pixi_lock_sha256": sys.argv[6],
    "harness_sha256": sys.argv[7],
    "python": {"version": sys.version, "executable": sys.executable},
    "platform": platform.platform(),
    "numpy": {"version": numpy.__version__, "configuration": numpy.__config__.show(mode="dicts")},
    "pymc_version": pymc.__version__,
    "pytensor_version": pytensor.__version__,
    "numpyro_version": importlib.metadata.version("numpyro"),
    "pixi": {"path": pixi_exe, "version": pixi_version, "sha256": pixi_sha256},
}
with Path(output).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
' "${output}/provenance.json" "${PIXI_EXE}" "${pixi_version}" "${pixi_sha256}" \
    "${FULL_SHA}" "${PIXI_LOCK_SHA256}" "${HARNESS_SHA256}"
pixi_python "${HARNESS_DIR}/20-audit-artifact.py" seal \
    --directory "${output}" --file checks.log --file provenance.json
harness_job_complete preflight "${output}/complete.json"
