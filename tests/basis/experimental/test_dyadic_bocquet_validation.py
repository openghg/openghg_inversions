"""Bounded end-to-end checks for the dyadic Bocquet validation example."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _load_example(script: Path, module_name: str) -> ModuleType:
    """Load the executable example as an isolated test module.

    Args:
        script: Path to the example source file.
        module_name: Temporary import name registered for dataclass support.

    Returns:
        Executed example module.

    Raises:
        RuntimeError: If Python cannot construct a loader for the source file.
    """
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load example module from {script}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_bounded_validation_writes_complete_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coarse K=3 run should preserve objective names, provenance, and outputs."""
    repository_root = Path(__file__).resolve().parents[3]
    output_directory = tmp_path / "output"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    module_name = "_test_dyadic_bocquet_validation_example"
    example = _load_example(
        repository_root / "examples/basis/dyadic_bocquet_validation.py",
        module_name,
    )
    assert example.build_parser().parse_args([]).coarsen_factor == 1

    try:
        status = example.main(
            [
                "--data-directory",
                str(repository_root / "tests/data"),
                "--output-directory",
                str(output_directory),
                "--coarsen-factor",
                "64",
                "--target-regions",
                "3",
            ]
        )
    finally:
        sys.modules.pop(module_name, None)

    assert status == 0
    expected_outputs = {
        "dyadic_bocquet_metrics.csv",
        "dyadic_bocquet_manifest.json",
        "dyadic_bocquet_report.md",
        "dyadic_bocquet_summary.png",
    }
    assert {path.name for path in output_directory.iterdir()} == expected_outputs

    with (output_directory / "dyadic_bocquet_metrics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["subset"] for row in rows} == {"training", "holdout"}
    assert {row["partition"] for row in rows} == {
        "root",
        "land_ocean",
        "rectangular_inner_outer",
        "dyadic_dfs",
        "dyadic_fisher",
        "dyadic_equation45",
    }
    objective_fields = {
        "dfs",
        "fisher",
        "aggregation_aware_fisher",
        "equation45",
        "bayesian_information_gain",
    }
    assert all(float(row[field]) >= 0.0 for row in rows for field in objective_fields)
    bound_fields = {
        "dfs": "native_dfs_bound",
        "fisher": "native_fisher_bound",
        "equation45": "native_equation45_bound",
    }
    assert all(
        float(row[metric]) <= float(row[bound]) + 1e-10
        for row in rows
        for metric, bound in bound_fields.items()
    )

    manifest = json.loads((output_directory / "dyadic_bocquet_manifest.json").read_text())
    assert manifest["experimental_only"]
    assert not manifest["centered_innovation"]["stored_real_mole_fraction_values_used"]
    assert not manifest["centered_innovation"]["stored_boundary_contribution_used"]
    assert manifest["centered_innovation"]["r_diag_formula"] == (
        "data.error**2 + explicit_model_error_ppb**2"
    )
    assert manifest["partition_selection"]["uses_training_rows_only"]
    assert manifest["search_resolution"]["coarsen_factor"] == 64
    assert manifest["search_resolution"]["mode"] == "explicit_coarsening_benchmark"
    assert manifest["search_resolution"]["default_is_native"]
    assert set(manifest["native_bounds"]) == {"training", "holdout", "all_rows"}
    assert "three additive DP selection objectives" in manifest["native_bounds_note"]
    assert "examples/basis/dyadic_bocquet_validation.py" in manifest["input_provenance"]["source_sha256"]

    report = (output_directory / "dyadic_bocquet_report.md").read_text()
    assert "Stored real mole-fraction values and stored boundary contributions are not used" in report
    assert "no coarsening is applied silently" in report
    assert "aggregation-aware Fisher" in report
    assert (output_directory / "dyadic_bocquet_summary.png").stat().st_size > 10_000
