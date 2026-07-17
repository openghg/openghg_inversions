"""End-to-end artifact checks for the bounded dyadic resolution sweep."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


_WEEK_FIXTURE_FILENAMES = {
    "obs_mhd_ch4_10m_2019-01-01_2019-01-07_data.nc",
    "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc",
    "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc",
    "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc",
    "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc",
}


def test_minimal_resolution_sweep_writes_consistent_artifacts(tmp_path: Path) -> None:
    """A one-fold, one-width sweep should preserve schemas and provenance."""
    repository_root = Path(__file__).resolve().parents[3]
    data_directory = repository_root / "tests/data"
    script = repository_root / "examples/basis/dyadic_resolution_sweep.py"
    output_directory = tmp_path / "output"
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-directory",
            str(data_directory),
            "--output-directory",
            str(output_directory),
            "--block-widths",
            "8",
            "--region-counts",
            "1",
            "--holdout-days",
            "2019-01-04",
            "--thinning-holdout-day",
            "2019-01-04",
        ],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote 2 candidate rows and 1 resolution rows" in result.stdout
    expected_outputs = {
        "emissions_holdout_sweep.csv",
        "coarsening_resolution.csv",
        "emissions_holdout_sweep.md",
        "emissions_holdout_sweep.png",
        "emissions_holdout_sweep_manifest.json",
    }
    assert {path.name for path in output_directory.iterdir()} == expected_outputs

    with (output_directory / "emissions_holdout_sweep.csv").open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    assert len(candidates) == 2
    assert {row["algorithm"] for row in candidates} == {"dyadic_greedy", "dyadic_exact_dp"}
    assert all(row["candidate_status"] == "completed" for row in candidates)
    assert all(row["holdout_full_weighted_trace"] for row in candidates)
    assert all(row["holdout_aggregation_weighted_trace"] for row in candidates)

    with (output_directory / "coarsening_resolution.csv").open(newline="", encoding="utf-8") as handle:
        resolution_rows = list(csv.DictReader(handle))
    assert len(resolution_rows) == 1
    assert resolution_rows[0]["ordinary_block_width"] == "8"
    assert resolution_rows[0]["partial_final_row_height"] == "5"
    assert resolution_rows[0]["partial_final_column_width"] == "7"

    manifest = json.loads((output_directory / "emissions_holdout_sweep_manifest.json").read_text())
    assert not manifest["observation_use"]["uses_mole_fraction_targets_or_residuals"]
    assert manifest["observation_use"]["error_weights_include_observed_within_hour_variability"]
    assert set(manifest["input_provenance"]["fixture_sha256"]) == _WEEK_FIXTURE_FILENAMES
    assert "examples/basis/dyadic_resolution_sweep.py" in manifest["source_provenance"]

    report = (output_directory / "emissions_holdout_sweep.md").read_text()
    assert "observed within-hour variability" in report
    assert "fixture/source hashes" in report
