#!/usr/bin/env bash
#SBATCH --job-name=ope121-wur25-profile
#SBATCH --account=chem007981
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

VG_CHECKOUT="${VG_CHECKOUT:?VG_CHECKOUT must name the exact verification-games checkout}"
VG_REVISION="${VG_REVISION:?VG_REVISION must name the exact VG commit}"
OGI_CHECKOUT="${OGI_CHECKOUT:?OGI_CHECKOUT must name the exact OGI checkout}"
OGI_REVISION="${OGI_REVISION:?OGI_REVISION must name the exact OGI commit}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must name a new output directory}"
PROFILE_MODE="${PROFILE_MODE:?PROFILE_MODE must be single, single-native, or full}"
PYTHON="${PYTHON:-/group/chem/acrg/verification_games_round_2/verification-games/.venv/bin/python}"
OBS_PATH="${OBS_PATH:-/group/chem/acrg/verification_games_round_2/games_catalog/data/verification_games_obs/WUR/CTE_STILT_EUROPE_BASE_co2_concentrations_2021.nc}"

unset OPE121_STOP_AFTER_FIRST OPE121_COMPARE_PR651 OPE121_PROFILE_NATIVE
if [[ "${PROFILE_MODE}" == "single" ]]; then
  export OPE121_STOP_AFTER_FIRST=1 OPE121_COMPARE_PR651=1
elif [[ "${PROFILE_MODE}" == "single-native" ]]; then
  export OPE121_STOP_AFTER_FIRST=1 OPE121_COMPARE_PR651=1 OPE121_PROFILE_NATIVE=1
elif [[ "${PROFILE_MODE}" != "full" ]]; then
  echo "PROFILE_MODE must be single, single-native, or full" >&2
  exit 2
fi

task_tmp="${TMPDIR:-/tmp}/ope121-wur25-profile-${SLURM_JOB_ID}"
mkdir -p "${task_tmp}/matplotlib" "${task_tmp}/numba" "${task_tmp}/pytensor"
export MPLCONFIGDIR="${task_tmp}/matplotlib"
export NUMBA_CACHE_DIR="${task_tmp}/numba"
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=${task_tmp}/pytensor"
export PYTHONPATH="${OGI_CHECKOUT}:${VG_CHECKOUT}:${VG_CHECKOUT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export VG_REVISION OGI_REVISION

cd "${VG_CHECKOUT}"
"${PYTHON}" "${OGI_CHECKOUT}/scripts/profile_ope121_wur25_sensitivity.py" \
  --vg-path /group/chem/acrg/verification_games_round_2 \
  --output-root "${OUTPUT_ROOT}" \
  --obs-path "${OBS_PATH}" \
  --obs-games-scenario BASE \
  --truth-games-scenario BASE \
  --truth-domain inner-paris \
  --sites UTO JFJ OXK CMN HPB PAL RGL GAT SAC STE MHD TOH HTM NOR HEL BSD TAC KRE TRN HUN BIR WES CBW OPE LIN \
  --start-date 2021-01-01 \
  --end-date 2021-02-01 \
  --species co2 \
  --games-scenario BASE \
  --sectors GPP TER FF ocean \
  --basis-mode intem_mixed_outer \
  --basis-split-step axis-parallel \
  --basis-per-source \
  --nbasis 150 \
  --ocean-nbasis 80 \
  --basis-random-seed 20260611 \
  --min-greedy-split-parent-weight-share 0.01 \
  --prior-uncertainty-mode country-calibrated \
  --country-total-target-relative-sd 0.5 \
  --calibration-countries GBR DEU FRA ITA \
  --aggregation-error-mode none \
  --prune-zero-h-columns \
  --zero-h-tolerance 0 \
  --min-error 0 \
  --prepared-input-route ogi \
  --build-only
