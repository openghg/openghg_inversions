"""Static controls for the score-regularized N0/N1 Slurm assets."""

from __future__ import annotations

from pathlib import Path
import subprocess


_REPOSITORY = Path(__file__).resolve().parents[3]
_ASSETS = (
    _REPOSITORY
    / "docs"
    / "plans"
    / "rjmcmc_score_regularized_nle_assets"
)
_N0 = _ASSETS / "run_n0_preflight.sbatch"
_COMPILE_CANARY = _ASSETS / "run_n1_compile_canary.sbatch"
_N1 = _ASSETS / "run_n1_development_array.sbatch"
_N1_MERGE = _ASSETS / "run_n1_merge.sbatch"


def test_n0_and_n1_assets_are_valid_bash_with_shared_node_resources() -> None:
    for script in (_N0, _COMPILE_CANARY, _N1, _N1_MERGE):
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        text = script.read_text(encoding="utf-8")
        assert "#SBATCH --cpus-per-task=1" in text
        assert "#SBATCH --mem=8G" in text
        assert "--exclusive" not in text
        assert "#SBATCH --account" not in text
        assert "module load git/2.45.1-pqk5" in text
        assert "PARIS_inversions" in text
        assert "score_regularized_flow_tiny_screen.py" in text
        assert "symbolic-ref -q HEAD" in text
        assert "status --porcelain --untracked-files=all" in text
        assert text.count("--xla_cpu_parallel_codegen_split_count=1") == 1

    n0 = _N0.read_text(encoding="utf-8")
    assert "#SBATCH --time=00:30:00" in n0
    assert "N0_report.json" in n0
    assert "N0_COMPLETE.json" in n0
    assert n0.rfind('"${complete}"') > n0.rfind('"${report}"')
    assert "separate_process_artifact_replay" in n0
    assert "--training-sample-count 64" in n0
    assert ".score-flow" in n0
    assert "test_score_regularized_flow_compile_canary.py" in n0
    assert "score_regularized_flow_compile_canary.py" in n0
    assert "test_score_regularized_flow_tiny_certify.py" in n0


def test_compile_canary_is_a_short_complete_two_branch_array() -> None:
    text = _COMPILE_CANARY.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "dimensions=(1 3)" in text
    assert 'dimension="${dimensions[case_index]}"' in text
    assert "score_regularized_flow_compile_canary" in text
    assert "Authenticated N0 evidence is required." in text
    assert "Compile canary requires the complete q=1,q=3 array." in text
    assert "q${dimension}.report.json" in text
    assert "The frozen nle-dev Python is missing." in text


def test_n1_is_one_complete_frozen_size_tier_array() -> None:
    text = _N1.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-5" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "72:00:00" not in text
    assert 'case_index="${SLURM_ARRAY_TASK_ID}"' in text
    assert "size_index=" not in text
    assert "sample_counts=(4096 16384 65536 262144)" in text
    assert 'sample_count="${NLE_SAMPLE_COUNT}"' in text
    assert "NLE_SAMPLE_COUNT is not one of the four frozen development sizes." in text
    assert 'regime="${case_id%%__*}"' in text
    assert 'family="${family%%__*}"' in text
    for case_id in (
        "near_gaussian__two_cell__root",
        "near_gaussian__four_cell__root",
        "skewed__two_cell__root",
        "skewed__four_cell__root",
        "boundary_heavy__two_cell__root",
        "boundary_heavy__four_cell__root",
    ):
        assert text.count(case_id) == 1
    assert "score-regularized-flow-n0-complete-v1" in text
    assert "N0_COMPLETE.json" in text
    assert "--base-seed 731" in text
    assert "--profile development" in text
    assert "--regime" in text
    assert "--family" in text
    assert ".score-flow" in text
    assert "two frozen initializations" not in text
    assert "The driver, not this array wrapper, publishes the strict task marker last." in text


def test_n1_merge_is_create_only_and_uses_the_strict_certifier() -> None:
    text = _N1_MERGE.read_text(encoding="utf-8")
    assert "#SBATCH --time=00:30:00" in text
    assert "score_regularized_flow_tiny_certify.py" in text
    assert "development-certificate.json" in text
    assert "MERGE_COMPLETE.json" in text
    assert "common-lock.json" not in text
    assert "Refusing to replace existing N1 merge evidence." in text
    assert "The certifier, not this wrapper, publishes MERGE_COMPLETE.json last." in text
