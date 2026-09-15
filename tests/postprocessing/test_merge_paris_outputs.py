"""Tests for merging sequential PARIS products."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.merge_paris_outputs import merge_paris_outputs


def _write_legacy_concentration(path: Path, time: str, sites: list[str], values: list[float]) -> None:
    xr.Dataset(
        {"Yobs": (("time", "nsite"), [values])},
        coords={"time": [np.datetime64(time)], "sitenames": ("nsite", sites)},
    ).to_netcdf(path)


def _write_latest_concentration(
    path: Path,
    times: list[str],
    platforms: list[str],
    identifiers: list[int],
    values: list[float],
) -> None:
    xr.Dataset(
        {
            "mf_observed": ("index", values),
            "number_of_identifier": ("index", np.asarray(identifiers, dtype="int16")),
        },
        coords={
            "time": ("index", np.asarray(times, dtype="datetime64[ns]")),
            "platform": ("platform", platforms),
        },
        attrs={"paris_concentration_template_version": "v04"},
    ).to_netcdf(path)


def test_merge_legacy_concentrations_aligns_sites_and_restores_sitenames(tmp_path: Path) -> None:
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    output = tmp_path / "merged.nc"
    _write_legacy_concentration(first, "2020-01-01", ["MHD", "TAC"], [1.0, 2.0])
    _write_legacy_concentration(second, "2021-01-01", ["TAC", "BSD"], [3.0, 4.0])

    merge_paris_outputs([second, first], output, output_type="concentration")

    with xr.open_dataset(output) as result:
        assert "nsite" not in result.coords
        assert result.sitenames.values.tolist() == ["BSD", "MHD", "TAC"]
        np.testing.assert_array_equal(
            result.Yobs.values,
            [[np.nan, 1.0, 2.0], [4.0, np.nan, 3.0]],
        )


def test_merge_latest_concentrations_rebuilds_platform_lookup(tmp_path: Path) -> None:
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    output = tmp_path / "merged.nc"
    _write_latest_concentration(
        first,
        ["2020-01-02", "2020-01-01"],
        ["MHD-10", "TAC-100"],
        [0, 1],
        [2.0, 1.0],
    )
    _write_latest_concentration(
        second,
        ["2021-01-01", "2021-01-02"],
        ["BSD-250", "MHD-10"],
        [0, 1],
        [3.0, 4.0],
    )

    merge_paris_outputs([second, first], output)

    with xr.open_dataset(output) as result:
        assert result.platform.values.tolist() == ["TAC-100", "MHD-10", "BSD-250"]
        assert result.number_of_identifier.values.tolist() == [0, 1, 2, 1]
        assert result.mf_observed.values.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert _platforms_for_observations(result) == ["TAC-100", "MHD-10", "BSD-250", "MHD-10"]
        assert "_platform_identifier" not in result


def _platforms_for_observations(ds: xr.Dataset) -> list[str]:
    return np.asarray(ds.platform.values)[ds.number_of_identifier.values].tolist()


def test_merge_fluxes_concatenates_time_but_not_static_country_fraction(tmp_path: Path) -> None:
    paths = []
    for year, value, country_fraction in ((2020, 1.0, 1.0), (2021, 2.0, 9.0)):
        path = tmp_path / f"flux-{year}.nc"
        xr.Dataset(
            {
                "flux_total_prior": (("time", "latitude", "longitude"), [[[value]]]),
                "country_fraction": (
                    ("country", "latitude", "longitude"),
                    [[[country_fraction]]],
                ),
            },
            coords={
                "time": [np.datetime64(f"{year}-01-01")],
                "latitude": [50.0],
                "longitude": [0.0],
                "country": ["GBR"],
            },
        ).to_netcdf(path)
        paths.append(path)

    output = tmp_path / "merged.nc"
    merge_paris_outputs(paths, output, output_type="flux")

    with xr.open_dataset(output) as result:
        assert result.flux_total_prior.values[:, 0, 0].tolist() == [1.0, 2.0]
        assert result.country_fraction.dims == ("country", "latitude", "longitude")
        assert result.country_fraction.item() == 1.0


def test_merge_repairs_repeated_latest_covariance_dimensions(tmp_path: Path) -> None:
    paths = []
    with pytest.warns(UserWarning, match="Duplicate dimension names"):
        for year in (2020, 2021):
            path = tmp_path / f"flux-{year}.nc"
            xr.Dataset(
                {
                    "flux_total_prior": (("time", "latitude", "longitude"), [[[1.0]]]),
                    "covariance_flux_total_posterior_country": (
                        ("time", "country", "country"),
                        np.ones((1, 2, 2)),
                    ),
                },
                coords={
                    "time": [np.datetime64(f"{year}-01-01")],
                    "latitude": [50.0],
                    "longitude": [0.0],
                    "country": ["GBR", "IRL"],
                },
            ).to_netcdf(path)
            paths.append(path)

        output = tmp_path / "merged.nc"
        merge_paris_outputs(paths, output)

    with xr.open_dataset(output) as result:
        assert result.covariance_flux_total_posterior_country.dims == (
            "country",
            "country_2",
            "time",
        )
        assert result.country_2.values.tolist() == ["GBR", "IRL"]


def test_merge_rejects_mixed_template_versions(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.nc"
    latest = tmp_path / "latest.nc"
    _write_legacy_concentration(legacy, "2020-01-01", ["MHD"], [1.0])
    _write_latest_concentration(latest, ["2021-01-01"], ["MHD-10"], [0], [2.0])

    with pytest.raises(ValueError, match="different template versions"):
        merge_paris_outputs([legacy, latest], tmp_path / "merged.nc")


def test_output_type_selects_concentrations_from_broad_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first-conc.nc"
    second = tmp_path / "second-conc.nc"
    flux = tmp_path / "flux.nc"
    _write_legacy_concentration(first, "2020-01-01", ["MHD"], [1.0])
    _write_legacy_concentration(second, "2021-01-01", ["MHD"], [2.0])
    xr.Dataset(
        {"flux_total_prior": (("time", "latitude", "longitude"), [[[3.0]]])},
        coords={"time": [0], "latitude": [50.0], "longitude": [0.0]},
    ).to_netcdf(flux)

    output = tmp_path / "merged.nc"
    merge_paris_outputs([first, flux, second], output, output_type="concentration")

    with xr.open_dataset(output) as result:
        assert result.Yobs.values[:, 0].tolist() == [1.0, 2.0]
