#!/usr/bin/env bash
#SBATCH --job-name=score-nle-promote-cert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00

set -euo pipefail

module load git/2.45.1-pqk5

: "${NLE_SOURCE:?Set NLE_SOURCE to the clean detached full-SHA worktree.}"
: "${NLE_RUN_ROOT:?Set NLE_RUN_ROOT to the fresh corrected run root.}"
: "${NLE_REVISION:?Set NLE_REVISION to the exact evaluation-code SHA.}"
: "${NLE_ARTIFACT_REVISION:?Set NLE_ARTIFACT_REVISION to the exact attempt/oracle source SHA.}"
: "${NLE_CERTIFICATE_TAG:?Set NLE_CERTIFICATE_TAG to a fresh immutable tag.}"
: "${NLE_SUMMARY_ROOTS:?Set NLE_SUMMARY_ROOTS to colon-separated exact summary roots.}"

source_root="$(realpath -e "${NLE_SOURCE}")"
run_root="$(realpath -e "${NLE_RUN_ROOT}")"
python_bin="${source_root}/.pixi/envs/nle-dev/bin/python"
log_directory="${run_root}/logs/promotion_certify"
log="${log_directory}/${NLE_CERTIFICATE_TAG}.${SLURM_JOB_ID:?}.log"

if [[ "${source_root}" == *PARIS_inversions* ||
      "${run_root}" == *PARIS_inversions* ]]; then
  echo "Refusing a PARIS_inversions path." >&2
  exit 64
fi
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${NLE_REVISION}" ||
      -n "$(git -C "${source_root}" symbolic-ref -q HEAD || true)" ||
      -n "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]]; then
  echo "NLE_SOURCE must be the exact clean detached revision." >&2
  exit 64
fi
if [[ ! "${NLE_ARTIFACT_REVISION}" =~ ^[0-9a-f]{40}$ ]] ||
   ! git -C "${source_root}" cat-file -e "${NLE_ARTIFACT_REVISION}^{commit}"; then
  echo "NLE_ARTIFACT_REVISION must name one exact available commit." >&2
  exit 64
fi
while IFS= read -r changed_path; do
  case "${changed_path}" in
    docs/plans/rjmcmc_score_regularized_nle_corrected_assets/run_corrected_promotion_merge.sbatch | \
      docs/plans/rjmcmc_score_regularized_nle_corrected_assets/run_corrected_promotion_certify.sbatch | \
      docs/plans/rjmcmc_score_regularized_nle_corrected_exploration_log.md | \
      docs/plans/rjmcmc_score_regularized_nle_corrected_promotion_plan.md | \
      docs/plans/rjmcmc_score_regularized_nle_corrected_promotion_review.md | \
      examples/rjmcmc/score_regularized_flow_corrected_promotion_merge.py | \
      examples/rjmcmc/score_regularized_flow_corrected_promotion_certify.py | \
      tests/experimental/rjmcmc/test_score_regularized_flow_corrected_exploration.py | \
      tests/experimental/rjmcmc/test_score_regularized_flow_corrected_promotion.py | \
      tests/experimental/rjmcmc/test_score_regularized_flow_corrected_promotion_certify.py)
      ;;
    *)
      echo "Recovery revision changes a non-allowlisted path: ${changed_path}" >&2
      exit 64
      ;;
  esac
done < <(git -C "${source_root}" diff --name-only \
  "${NLE_ARTIFACT_REVISION}" "${NLE_REVISION}" --)
IFS=: read -r -a summary_roots <<<"${NLE_SUMMARY_ROOTS}"
if [[ "${#summary_roots[@]}" -ne 2 && "${#summary_roots[@]}" -ne 5 ]]; then
  echo "NLE_SUMMARY_ROOTS must contain the exact development pair or all five roots." >&2
  exit 64
fi
summary_arguments=()
for root in "${summary_roots[@]}"; do
  summary_arguments+=(--summary-root "$(realpath -e "${root}")")
done
mkdir -p "${log_directory}"
if [[ -e "${log}" || -L "${log}" ]]; then
  echo "Refusing to replace an existing certifier log." >&2
  exit 73
fi
exec >"${log}" 2>&1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_cpu_parallel_codegen_split_count=1"
export PYTHONPATH="${source_root}${PYTHONPATH:+:${PYTHONPATH}}"
cache_root="${TMPDIR:-/tmp}/score-nle-promotion-certify-${SLURM_JOB_ID}"
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

cd "${source_root}"
exec "${python_bin}" \
  examples/rjmcmc/score_regularized_flow_corrected_promotion_certify.py \
  "${summary_arguments[@]}" \
  --artifact-source-git-revision "${NLE_ARTIFACT_REVISION}" \
  --evaluation-source-git-revision "${NLE_REVISION}" \
  --output-root "${run_root}" \
  --certificate-tag "${NLE_CERTIFICATE_TAG}"
