"""Tests for the authenticated PARIS exact-mixture root resource probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from examples.rjmcmc import (
    conditional_residual_image_exact_mixture_paris_probe as probe,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
)


def _dataset(*, observation_shift: float = 0.0) -> xr.Dataset:
    """Return one small valid probe input with non-uniform native weights."""
    response = np.asarray(
        [
            [[1.0, 0.2, 0.5], [0.1, 0.7, 0.3]],
            [[0.2, 1.1, 0.4], [0.8, 0.1, 0.6]],
            [[0.6, 0.4, 1.2], [0.3, 0.9, 0.2]],
        ],
        dtype=np.float32,
    )
    return xr.Dataset(
        {
            "fp_x_flux": (("nmeasure", "lat", "lon"), response),
            "mf": (
                "nmeasure",
                np.asarray(
                    [2.0 + observation_shift, 3.0, 4.0],
                    dtype=np.float64,
                ),
            ),
            "mf_error": (
                "nmeasure",
                np.asarray([0.5, 0.75, 1.25], dtype=np.float64),
            ),
            "nominal_weight": (
                ("lat", "lon"),
                np.asarray(
                    [[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]],
                    dtype=np.float64,
                ),
            ),
            "outer_design": (
                ("nmeasure", "outer_region"),
                np.asarray(
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    dtype=np.float32,
                ),
            ),
            "YaprioriBC": (
                "nmeasure",
                np.asarray([0.2, 0.3, 0.4], dtype=np.float64),
            ),
        },
        coords={
            "nmeasure": np.asarray(["a", "b", "c"]),
            "lat": np.asarray([50.0, 51.0]),
            "lon": np.asarray([-2.0, -1.0, 0.0]),
            "outer_region": np.asarray(["outer_0", "outer_1"]),
        },
        attrs={"schema_id": "tiny-probe-v1"},
    )


def _write_dataset(path: Path, *, observation_shift: float = 0.0) -> str:
    """Write one fixture and return its whole-file SHA-256."""
    _dataset(observation_shift=observation_shift).to_netcdf(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(path: Path, digest: str, **overrides: Any) -> dict[str, Any]:
    """Run the small fixture through the public probe boundary."""
    arguments: dict[str, Any] = {
        "input_path": path,
        "expected_input_sha256": digest,
        "input_id": "tiny-probe",
        "source_revision": "1" * 40,
        "concentration": 10.0,
        "expected_shape": (3, 2, 3),
        "expected_outer_labels": ("outer_0", "outer_1"),
        "expected_schema": "tiny-probe-v1",
        "variance_fractions": (0.5, 0.9, 0.99),
        "diagnostic_ranks": (1, 2, 3),
        "source_sample_count": 8,
        "component_count": 2,
        "include_timings": False,
    }
    arguments.update(overrides)
    return probe.run_probe(**arguments)


def test_probe_matches_independent_physical_mass_covariance(
    tmp_path: Path,
) -> None:
    """The adapter conversion and spectrum match a direct dense oracle."""
    path = tmp_path / "input.nc"
    digest = _write_dataset(path)

    report = _run(path, digest)

    dataset = _dataset()
    weight = np.asarray(dataset["nominal_weight"].values, dtype=np.float64)
    weight /= float(np.sum(weight))
    scaling = np.asarray(dataset["fp_x_flux"].values, dtype=np.float64)
    design = scaling.reshape(3, -1) / weight.reshape(1, -1)
    noise = np.asarray(dataset["mf_error"].values, dtype=np.float64)
    mean = design @ weight.reshape(-1)
    centered = (design - mean[:, np.newaxis]) / noise[:, np.newaxis]
    factor = centered * np.sqrt(weight.reshape(-1) / 11.0)[np.newaxis, :]
    expected = np.linalg.eigvalsh(factor @ factor.T)[::-1]
    tolerance = float(report["spectrum"]["eigenvalue_tolerance"])
    expected = expected[expected > tolerance]

    np.testing.assert_allclose(
        report["spectrum"]["eigenvalues"],
        expected,
        rtol=5.0e-13,
        atol=5.0e-13,
    )
    np.testing.assert_allclose(mean, scaling.sum(axis=(1, 2)))
    assert report["closure"]["passed"] is True
    assert report["model"]["unit_scaling_to_mass_conversion"] == ("fp_x_flux / normalized_nominal_weight")


def test_probe_is_observation_blind_and_replayable(tmp_path: Path) -> None:
    """Changing only mf changes file identity but not the analytic spectrum."""
    first_path = tmp_path / "first.nc"
    second_path = tmp_path / "second.nc"
    first = _run(first_path, _write_dataset(first_path))
    second = _run(
        second_path,
        _write_dataset(second_path, observation_shift=100.0),
    )

    assert first == _run(first_path, probe._sha256_file(first_path))
    assert first["input"]["sha256"] != second["input"]["sha256"]
    assert first["input"]["variable_sha256"]["mf"] != (second["input"]["variable_sha256"]["mf"])
    assert first["spectrum"] == second["spectrum"]
    assert first["model"] == second["model"]
    assert len(first["protocol_sha256"]) == 64
    assert first["observed_residual_used_for_spectrum_selection"] is False
    assert first["structural_inference_licensed"] is False


def test_probe_rank_and_resource_accounting_is_deterministic(
    tmp_path: Path,
) -> None:
    """Rank tails and byte estimates obey their declared array formulas."""
    path = tmp_path / "input.nc"
    report = _run(path, _write_dataset(path))
    rank = int(report["spectrum"]["positive_numerical_rank"])
    resources = report["resources"]

    expected_share = 8 * 8 * 6
    expected_uniform = 8 * 8 * 5
    expected_source = 8 * (8 * rank + 2 * 3 + 3 * rank + 2 * 6 + 1)
    assert resources["current_sobol_share_array_bytes"] == expected_share
    assert resources["current_sobol_largest_uniform_block_bytes"] == (expected_uniform)
    assert resources["current_builder_full_rank_source_persistent_bytes"] == (expected_source)
    assert resources["current_sobol_known_simultaneous_lower_bound_bytes"] == (
        expected_share + expected_uniform + expected_source
    )
    assert resources["rank_scenarios"][-1]["mixture_rank"] == rank
    records = report["spectrum"]["diagnostic_ranks"]
    assert [record["rank"] for record in records] == sorted(record["rank"] for record in records)
    assert records[-1]["omitted_variance_per_squared_root_mass"] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert "timings" not in report


def test_spectrum_diagnostics_handles_tolerance_discarded_trace() -> None:
    """A positive all-discarded trace cannot overflow fraction-rank lookup."""
    spectrum = RootResidualSpectrum(
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.empty((1, 0), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        total_variance=1.0e-20,
        discarded_variance=1.0e-20,
        requested_retained_variance_fraction=1.0,
        eigenvalue_tolerance=1.0e-18,
        cell_alphas_sha256="a" * 64,
        design_sha256="b" * 64,
        noise_sd_sha256="c" * 64,
    )

    result = probe._spectrum_diagnostics(
        spectrum,
        native_cell_count=2,
        fractions=(0.9,),
        diagnostic_ranks=(1,),
    )

    assert result["positive_numerical_rank"] == 0
    assert result["variance_fraction_ranks"] == [
        {
            "requested_fraction": 0.9,
            "rank": 0,
            "actual_fraction": 0.0,
            "requested_fraction_met": False,
        }
    ]
    assert result["retained_spectrum_entropy_effective_rank"] == 0.0


def test_probe_hash_mismatch_fails_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad whole-file identity is rejected before xarray input access."""
    path = tmp_path / "input.nc"
    _write_dataset(path)

    def forbidden_loader(*args: object, **kwargs: object) -> xr.Dataset:
        raise AssertionError("dataset loader should not have been called")

    monkeypatch.setattr(probe, "_load_frozen_subset", forbidden_loader)
    with pytest.raises(ValueError, match="SHA-256"):
        _run(path, "0" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda dataset: dataset.assign(mf_error=("nmeasure", np.asarray([0.5, 0.0, 1.25]))),
            "mf_error",
        ),
        (
            lambda dataset: dataset.assign(
                nominal_weight=(
                    ("lat", "lon"),
                    np.asarray([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0]]),
                )
            ),
            "nominal_weight",
        ),
        (
            lambda dataset: dataset.drop_vars("YaprioriBC"),
            "missing required variables",
        ),
    ],
)
def test_probe_rejects_malformed_scientific_inputs(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    """Malformed required arrays fail closed without floors or discovery."""
    path = tmp_path / "input.nc"
    mutation(_dataset()).to_netcdf(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        _run(path, digest)


def test_atomic_report_is_canonical_create_only_and_nonproduction(
    tmp_path: Path,
) -> None:
    """Publication is canonical, refuses replacement, and bars PARIS paths."""
    output = tmp_path / "report.json"
    payload = {"z": 2, "a": [1, True]}
    probe._atomic_write(output, payload)

    assert output.read_bytes() == b'{"a":[1,true],"z":2}\n'
    assert json.loads(output.read_text(encoding="ascii")) == payload
    with pytest.raises(FileExistsError, match="refusing to replace"):
        probe._atomic_write(output, payload)
    prohibited = tmp_path / "PARIS_inversions"
    prohibited.mkdir()
    with pytest.raises(ValueError, match="PARIS_inversions"):
        probe._atomic_write(prohibited / "report.json", payload)

    alias = tmp_path / "safe" / "nested"
    alias.parent.mkdir()
    alias.symlink_to(prohibited, target_is_directory=True)
    with pytest.raises(ValueError, match="PARIS_inversions"):
        probe._atomic_write(alias / "report.json", payload)


def test_main_rejects_existing_output_before_expensive_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI publication policy is checked before any scientific construction."""
    output = tmp_path / "existing.json"
    output.write_text("sentinel", encoding="ascii")

    def forbidden_probe(**kwargs: object) -> dict[str, Any]:
        raise AssertionError("run_probe should not have been called")

    monkeypatch.setattr(probe, "run_probe", forbidden_probe)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        probe.main(
            [
                "--input",
                str(tmp_path / "absent.nc"),
                "--expected-input-sha256",
                "0" * 64,
                "--input-id",
                "input",
                "--source-revision",
                "1" * 40,
                "--concentration",
                "100",
                "--output",
                str(output),
            ]
        )


@pytest.mark.parametrize(
    ("source_count", "components", "message"),
    [
        (7, 2, "power of two"),
        (8, 9, "cannot exceed"),
    ],
)
def test_probe_rejects_incompatible_estimator_sizes_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_count: int,
    components: int,
    message: str,
) -> None:
    """Resource controls must describe a legal Sobol/compression construction."""
    path = tmp_path / "input.nc"
    digest = _write_dataset(path)

    def forbidden_loader(*args: object, **kwargs: object) -> xr.Dataset:
        raise AssertionError("dataset loader should not have been called")

    monkeypatch.setattr(probe, "_load_frozen_subset", forbidden_loader)
    with pytest.raises(ValueError, match=message):
        _run(
            path,
            digest,
            source_sample_count=source_count,
            component_count=components,
        )
