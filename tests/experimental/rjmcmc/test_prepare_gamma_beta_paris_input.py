"""Tests for the frozen Gamma--Beta PARIS native-input builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3] / "examples" / "rjmcmc" / "prepare_gamma_beta_paris_input.py"
)


@pytest.fixture(scope="module")
def builder_module() -> ModuleType:
    """Import the repository-root builder without making examples a package."""
    specification = importlib.util.spec_from_file_location(
        "prepare_gamma_beta_paris_input_example",
        EXAMPLE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load frozen Gamma-Beta input builder.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def small_contract(monkeypatch: pytest.MonkeyPatch, builder_module: ModuleType) -> None:
    """Replace only production dimensions so tests can exercise the full builder cheaply."""
    monkeypatch.setattr(builder_module, "PARIS_OBSERVATIONS", 3)
    monkeypatch.setattr(builder_module, "PARIS_GRID_SHAPE", (2, 2))


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_archive(path: Path, latitude: np.ndarray, longitude: np.ndarray) -> None:
    fractions = np.zeros((3, latitude.size, longitude.size), dtype=np.float64)
    fractions[1, 1:3, 2:4] = 0.75
    fractions[2, 1:3, 2:4] = 0.25
    xr.Dataset(
        {
            "country_fraction": (
                ("country", "latitude", "longitude"),
                fractions,
            )
        },
        coords={
            "country": np.asarray(["FRA", "GBR", "IRL"]),
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_netcdf(path, engine="h5netcdf")


def _write_snapshot(
    path: Path,
    archive: Path,
    *,
    duplicate_measurement: bool = False,
    rectangular_mask: bool = True,
) -> None:
    latitude = np.asarray([49.0, 50.0, 51.0, 52.0])
    longitude = np.asarray([-4.0, -3.0, -2.0, -1.0, 0.0])
    inner_mask = np.zeros((4, 5), dtype=np.bool_)
    inner_mask[1:3, 2:4] = True
    if not rectangular_mask:
        inner_mask[1, 2] = False
    site = np.asarray(["AAA", "AAA", "BBB"])
    times = np.asarray(
        [
            "2014-05-01T00:00:00",
            "2014-05-01T04:00:00",
            "2014-05-01T00:00:00",
        ],
        dtype="datetime64[ns]",
    ).astype(np.int64)
    if duplicate_measurement:
        site[1] = site[0]
        times[1] = times[0]
    metadata = {
        "schema": "paris-rjmcmc-native-snapshot-v1",
        "n_observations": 3,
        "n_inner_cells": 4,
        "n_outer_regions": 6,
        "archived_flux": str(archive),
        "archived_flux_sha256": _sha256_file(archive),
        "boundary_treatment": "fixed test boundary offset",
        "error_model": "fixed diagonal test error",
        "units": {"sensitivity": "ppb at unit flux scale"},
    }
    np.savez_compressed(
        path,
        observations=np.asarray([1900.0, 1901.0, 1902.0]),
        observation_sd=np.asarray([1.0, 2.0, 3.0]),
        sensitivity=np.arange(12, dtype=np.float32).reshape(3, 4),
        outer_design=np.arange(18, dtype=np.float32).reshape(3, 6),
        fixed_offset=np.asarray([1800.0, 1801.0, 1802.0]),
        archived_prior=np.asarray([1903.0, 1904.0, 1905.0]),
        archived_posterior=np.asarray([1901.0, 1902.0, 1903.0]),
        archived_uncertainty=np.asarray([4.0, 5.0, 6.0]),
        current_prior_flux=np.arange(20, dtype=np.float64).reshape(4, 5) * 1.0e-9,
        site=site,
        time=times,
        latitude=latitude,
        longitude=longitude,
        inner_mask=inner_mask,
        outer_region=np.zeros((4, 5), dtype=np.int8),
        metadata=np.frombuffer(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8"),
            dtype=np.uint8,
        ),
    )


def test_builds_exact_labelled_area_weighted_contract(
    tmp_path: Path,
    small_contract: None,
    builder_module: ModuleType,
) -> None:
    """The output should preserve mapping, labels, reviewed columns, and auxiliaries."""
    archive = tmp_path / "archive.nc"
    source = tmp_path / "source.npz"
    output = tmp_path / "frozen.nc"
    latitude = np.asarray([49.0, 50.0, 51.0, 52.0])
    longitude = np.asarray([-4.0, -3.0, -2.0, -1.0, 0.0])
    _write_archive(archive, latitude, longitude)
    _write_snapshot(source, archive)
    source_digest = _sha256_file(source)

    assert (
        builder_module.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--expected-source-sha256",
                source_digest,
            ]
        )
        == 0
    )

    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["schema_id"] == builder_module.SIDECAR_SCHEMA_ID
    assert sidecar["source"]["sha256"] == source_digest
    assert sidecar["output"]["sha256"] == _sha256_file(output)
    assert sidecar["construction"]["nominal_weight"]["policy_id"] == (
        "spherical-grid-cell-area-v1"
    )
    assert sidecar["construction"]["nominal_weight"]["epsilon_floor"] is None
    assert sidecar["construction"]["mapping"]["source_latitude_index_slice"] == [1, 3]
    assert sidecar["construction"]["mapping"]["source_longitude_index_slice"] == [2, 4]

    with xr.open_dataset(output, engine="h5netcdf") as dataset:
        assert dataset.sizes == {
            "nmeasure": 3,
            "lat": 2,
            "lon": 2,
            "outer_region": 6,
            "country": 2,
        }
        np.testing.assert_array_equal(
            dataset["fp_x_flux"],
            np.arange(12, dtype=np.float32).reshape(3, 2, 2),
        )
        np.testing.assert_array_equal(
            dataset["nmeasure"],
            [
                "AAA|2014-05-01T00:00:00.000000000",
                "AAA|2014-05-01T04:00:00.000000000",
                "BBB|2014-05-01T00:00:00.000000000",
            ],
        )
        np.testing.assert_array_equal(
            dataset["outer_region"],
            [f"intem_label_{index}" for index in range(6)],
        )
        np.testing.assert_array_equal(dataset["country"], ["GBR", "IRL"])
        np.testing.assert_array_equal(
            dataset["prior_flux"],
            np.arange(20, dtype=np.float64).reshape(4, 5)[1:3, 2:4] * 1.0e-9,
        )
        assert np.all(dataset["nominal_weight"].values > 0.0)
        assert float(dataset["nominal_weight"].sum()) == pytest.approx(1.0)
        assert dataset["nominal_weight"].attrs["policy_id"] == "spherical-grid-cell-area-v1"
        assert np.all(dataset["grid_cell_area"].values > 0.0)
        np.testing.assert_allclose(dataset["country_fraction"].sel(country="GBR"), 0.75)
        np.testing.assert_allclose(dataset["country_fraction"].sel(country="IRL"), 0.25)


@pytest.mark.parametrize(
    ("duplicate_measurement", "rectangular_mask", "message"),
    [
        (True, True, r"site\|time observation labels must be unique"),
        (False, False, "filled rectangular slice"),
    ],
)
def test_rejects_ambiguous_source_mapping(
    tmp_path: Path,
    small_contract: None,
    builder_module: ModuleType,
    duplicate_measurement: bool,
    rectangular_mask: bool,
    message: str,
) -> None:
    """Duplicate rows and nonrectangular inner masks cannot be frozen."""
    archive = tmp_path / "archive.nc"
    source = tmp_path / "source.npz"
    latitude = np.asarray([49.0, 50.0, 51.0, 52.0])
    longitude = np.asarray([-4.0, -3.0, -2.0, -1.0, 0.0])
    _write_archive(archive, latitude, longitude)
    _write_snapshot(
        source,
        archive,
        duplicate_measurement=duplicate_measurement,
        rectangular_mask=rectangular_mask,
    )
    with np.load(source, allow_pickle=False) as snapshot, pytest.raises(
        ValueError,
        match=message,
    ):
        builder_module._build_dataset(
            snapshot,
            source_path=source,
            source_digest=_sha256_file(source),
            include_country_fractions=True,
        )


def test_rejects_changed_source_before_creating_output(
    tmp_path: Path,
    small_contract: None,
    builder_module: ModuleType,
) -> None:
    """The reviewed source digest is a hard gate."""
    archive = tmp_path / "archive.nc"
    source = tmp_path / "source.npz"
    output = tmp_path / "frozen.nc"
    _write_archive(
        archive,
        np.asarray([49.0, 50.0, 51.0, 52.0]),
        np.asarray([-4.0, -3.0, -2.0, -1.0, 0.0]),
    )
    _write_snapshot(source, archive)
    with pytest.raises(ValueError, match="does not match"):
        builder_module.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--expected-source-sha256",
                "0" * 64,
            ]
        )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
