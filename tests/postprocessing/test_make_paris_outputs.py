"""Tests for multisector products written with the latest PARIS flux template."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest
import xarray as xr

from openghg_inversions import convert
from openghg_inversions.postprocessing import make_outputs, make_paris_outputs
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.flux_sanitization import FluxNonFiniteMetadata, NONFINITE_POLICY_ZERO_FILL
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_paris_outputs import (
    PARIS_LATEST_COUNTRIES,
    _latest_paris_countries,
    _paris_sector_name_by_suffix,
    paris_flux_output,
)


def _flux_nonfinite_metadata(data: xr.DataArray | xr.Dataset) -> FluxNonFiniteMetadata:
    """Return parsed non-finite flux metadata from an xarray object."""
    metadata = FluxNonFiniteMetadata.from_attrs(data.attrs)
    assert metadata is not None
    return metadata


def test_latest_paris_sector_names_are_template_safe(
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """PARIS sector names are lower-case variable-safe values derived from suffixes."""
    inv_out = multisector_postprocessing_inv_out()
    inv_out.model_metadata["sectors"] = [
        {"name": "FF", "flux_source": "ff-inventory", "variable_suffix": "FF-sector"},
        {"name": "Ocean", "flux_source": "ocean-inventory", "variable_suffix": "Ocean_sector"},
    ]
    assert _paris_sector_name_by_suffix(inv_out) == {
        "FF-sector": "ffsector",
        "Ocean_sector": "oceansector",
    }

    inv_out.model_metadata["sectors"] = [
        {"name": "One", "flux_source": "ff-inventory", "variable_suffix": "sector-2"},
        {"name": "Two", "flux_source": "ocean-inventory", "variable_suffix": "sector_2"},
    ]
    with pytest.raises(ValueError, match="duplicate sector name 'sector2'"):
        _paris_sector_name_by_suffix(inv_out)

    inv_out.model_metadata["sectors"] = [
        {"name": "Total", "flux_source": "ff-inventory", "variable_suffix": "total"},
    ]
    with pytest.raises(ValueError, match="reserved"):
        _paris_sector_name_by_suffix(inv_out)

    inv_out.model_metadata["sectors"] = [
        {"name": "Bad", "flux_source": "ff-inventory", "variable_suffix": "---"},
    ]
    with pytest.raises(ValueError, match="Could not derive"):
        _paris_sector_name_by_suffix(inv_out)


def test_latest_paris_flux_output_processes_multisector_sectors(
    europe_country_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_multisector_basis_functions_matching_country_grid: Callable[..., BasisFunctions],
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Latest multisector PARIS output projects countries before population covariance."""
    inv_out = multisector_postprocessing_inv_out(
        fake_multisector_basis_functions_matching_country_grid(
            europe_country_file,
            coord_offset=1e-10,
        )
    )
    reconstruction_calls: list[tuple[bool, bool]] = []
    sector_reconstruction_grids: list[bool] = []
    original_reconstruct = make_paris_outputs.make_multisector_flux_trace_outputs
    original_sector_reconstruct = make_outputs._sector_flux_trace_dataset

    def record_reconstruction(
        output: InversionOutput,
        report_flux_on_inversion_grid: bool = True,
        *,
        materialize: bool = True,
    ) -> xr.Dataset:
        """Record whether PARIS crosses the eager trace boundary before aggregation."""
        trace = original_reconstruct(
            output,
            report_flux_on_inversion_grid=report_flux_on_inversion_grid,
            materialize=materialize,
        )
        is_lazy_or_sparse = not isinstance(trace["flux_total_posterior"].data, np.ndarray)
        reconstruction_calls.append((materialize, is_lazy_or_sparse))
        return trace

    def record_sector_reconstruction(
        output: InversionOutput,
        sector: make_outputs.OutputSector,
        *,
        report_flux_on_inversion_grid: bool,
    ) -> xr.Dataset:
        """Record each sector reconstruction and the grid it targets."""
        sector_reconstruction_grids.append(report_flux_on_inversion_grid)
        return original_sector_reconstruct(
            output,
            sector,
            report_flux_on_inversion_grid=report_flux_on_inversion_grid,
        )

    monkeypatch.setattr(
        make_paris_outputs,
        "make_multisector_flux_trace_outputs",
        record_reconstruction,
    )
    monkeypatch.setattr(make_outputs, "_sector_flux_trace_dataset", record_sector_reconstruction)

    flux_outputs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
        country_selections=PARIS_LATEST_COUNTRIES,
        inversion_grid=True,
        template_version="latest",
    )

    assert flux_outputs.attrs["paris_flux_template_version"] == "v03"
    assert reconstruction_calls == [(False, True)]
    assert sector_reconstruction_grids.count(False) == 2
    assert sector_reconstruction_grids.count(True) == 2
    assert "flux_total_posterior" in flux_outputs
    assert "stdev_flux_total_posterior" in flux_outputs
    assert "flux_ff_posterior" in flux_outputs
    assert "flux_ocean_posterior" in flux_outputs
    assert "flux_ff_posterior_inversion_grid" in flux_outputs
    assert "flux_ocean_posterior_inversion_grid" in flux_outputs
    assert "flux_total_ff_posterior" not in flux_outputs
    assert "flux_total_posterior_country" in flux_outputs
    assert "flux_ff_posterior_country" in flux_outputs
    assert "flux_ocean_posterior_country" in flux_outputs
    assert "covariance_flux_total_posterior_country" in flux_outputs
    assert "covariance_flux_ff_posterior_country" in flux_outputs
    assert "covariance_flux_ocean_posterior_country" in flux_outputs
    assert "covariance_flux_sectors_posterior_country" in flux_outputs
    assert tuple(flux_outputs.sector.values) == ("ff", "ocean")
    assert flux_outputs.sector.attrs["long_name"] == "short name of flux sector"
    assert "ff" in flux_outputs["flux_ff_posterior"].attrs["long_name"]
    assert "sector_name" not in flux_outputs["flux_ff_posterior"].attrs["long_name"]
    assert flux_outputs["flux_ff_posterior_inversion_grid"].attrs == {
        "units": "mol m-2 s-1",
        "long_name": "posterior flux of  ch4 for ff on the inversion grid",
        "cell_methods": "time:mean area:mean",
    }
    assert tuple(flux_outputs.country.values) == PARIS_LATEST_COUNTRIES
    assert flux_outputs["flux_ff_posterior"].dtype == np.dtype("float32")
    assert flux_outputs["covariance_flux_ff_posterior_country"].dtype == np.dtype("float32")
    np.testing.assert_allclose(flux_outputs["flux_ff_posterior"].isel(time=0).values, 1.0)
    np.testing.assert_allclose(flux_outputs["flux_ocean_posterior"].isel(time=0).values, 1.0)
    country_fraction_data = flux_outputs["country_fraction"].data
    if hasattr(country_fraction_data, "todense"):
        country_fraction_data = country_fraction_data.todense()
    cell_area_data = flux_outputs["cell_area"].data
    if hasattr(cell_area_data, "todense"):
        cell_area_data = cell_area_data.todense()
    assert not np.isnan(np.asarray(country_fraction_data)).any()
    assert not np.isnan(np.asarray(cell_area_data)).any()
    countries = Countries.from_file(
        country_file=europe_country_file,
        country_code="alpha3",
        country_selections=list(PARIS_LATEST_COUNTRIES),
        domain="EUROPE",
    )
    expected_country = (
        2.0
        * (countries.matrix * countries.area_grid).sum(("lat", "lon"))
        * 365
        * 24
        * 3600
        * convert.molar_mass("ch4")
        * 1e-3
    ).reindex(country=list(PARIS_LATEST_COUNTRIES))
    np.testing.assert_allclose(
        flux_outputs["flux_total_posterior_country"].isel(time=0).values,
        expected_country.data.todense(),
        rtol=1e-6,
    )
    expected_sector_country = expected_country / 2.0
    np.testing.assert_allclose(
        flux_outputs["flux_ff_posterior_country"].isel(time=0).values,
        expected_sector_country.data.todense(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        flux_outputs["flux_ocean_posterior_country"].isel(time=0).values,
        expected_sector_country.data.todense(),
        rtol=1e-6,
    )
    largest_country = int(np.asarray(expected_sector_country.data.todense()).argmax())
    expected_sector_variance = np.asarray(expected_sector_country.data.todense())[largest_country] ** 2
    np.testing.assert_allclose(
        flux_outputs["covariance_flux_ff_posterior_country"].values[
            0,
            largest_country,
            largest_country,
        ],
        expected_sector_variance,
        rtol=1e-6,
    )
    for flux_label in ("total", "ff", "ocean"):
        covariance = flux_outputs[f"covariance_flux_{flux_label}_posterior_country"].values
        stdev = flux_outputs[f"stdev_flux_{flux_label}_posterior_country"].values
        np.testing.assert_allclose(
            np.diagonal(covariance, axis1=1, axis2=2),
            stdev**2,
            rtol=1e-6,
        )
        assert np.isfinite(covariance).all()

    sector_cross_covariance = flux_outputs["covariance_flux_sectors_posterior_country"].values
    for sector_index, sector_name in enumerate(("ff", "ocean")):
        np.testing.assert_allclose(
            sector_cross_covariance[:, :, sector_index, sector_index],
            flux_outputs[f"stdev_flux_{sector_name}_posterior_country"].values ** 2,
            rtol=1e-6,
        )
    assert np.isfinite(sector_cross_covariance).all()
    np.testing.assert_allclose(
        flux_outputs["covariance_flux_sectors_posterior_country"].values[
            0,
            largest_country,
            0,
            1,
        ],
        -expected_sector_variance,
        rtol=1e-6,
    )
    assert _flux_nonfinite_metadata(flux_outputs).policy == NONFINITE_POLICY_ZERO_FILL

    flux_file = tmp_path / "latest_multisector_flux.nc"
    flux_outputs.to_netcdf(flux_file)
    with xr.open_dataset(flux_file) as reloaded_flux:
        assert tuple(reloaded_flux.sector.values) == ("ff", "ocean")
        assert tuple(reloaded_flux.country.values) == PARIS_LATEST_COUNTRIES
        covariance_dims = {
            "covariance_flux_total_posterior_country": ("time", "country", "country"),
            "covariance_flux_ff_posterior_country": ("time", "country", "country"),
            "covariance_flux_ocean_posterior_country": ("time", "country", "country"),
            "covariance_flux_sectors_posterior_country": ("time", "country", "sector", "sector"),
        }
        for variable_name, expected_dims in covariance_dims.items():
            expected = flux_outputs[variable_name]
            actual = reloaded_flux[variable_name]
            assert actual.dims == expected_dims
            assert actual.shape == expected.shape
            assert actual.dtype == np.dtype("float32")
            assert actual.attrs == expected.attrs
            np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6)


def test_latest_paris_flux_output_renames_overlapping_sector_suffixes_exactly(
    europe_country_file: Path,
    fake_multisector_basis_functions_matching_country_grid: Callable[..., BasisFunctions],
    multisector_postprocessing_inv_out: Callable[..., InversionOutput],
) -> None:
    """Overlapping sector suffixes and a legitimate total prefix retain exact normalized names."""
    basis_functions = fake_multisector_basis_functions_matching_country_grid(europe_country_file)
    total_ff_flux = basis_functions.flux.sel(source="ff-inventory", drop=True).expand_dims(
        source=["total-ff-inventory"]
    )
    basis_functions = basis_functions.with_flux(
        xr.concat([basis_functions.flux, total_ff_flux], dim="source")
    )
    inv_out = multisector_postprocessing_inv_out(basis_functions)

    for group_name in ("prior", "posterior"):
        trace_group = getattr(inv_out.trace, group_name).rename(
            {"x_ff": "x_energy", "x_ocean": "x_energy_waste"}
        )
        trace_group["x_total_ff"] = trace_group["x_energy"]
        setattr(inv_out.trace, group_name, trace_group)
    inference_data = cast(Any, inv_out.trace)
    prior = inference_data.prior
    extra_prior_draw = prior.isel(draw=[0]).assign_coords(draw=[prior.sizes["draw"]])
    inference_data.prior = xr.concat([prior, extra_prior_draw], dim="draw")
    inv_out.model_metadata["sectors"] = [
        {"name": "Energy", "flux_source": "ff-inventory", "variable_suffix": "energy"},
        {
            "name": "Energy waste",
            "flux_source": "ocean-inventory",
            "variable_suffix": "energy_waste",
        },
        {
            "name": "Total fossil fuel",
            "flux_source": "total-ff-inventory",
            "variable_suffix": "total_ff",
        },
    ]

    flux_outputs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
        country_selections=PARIS_LATEST_COUNTRIES,
        inversion_grid=False,
        template_version="latest",
    )

    expected_names = ("energy", "energywaste", "totalff")
    assert tuple(flux_outputs.sector.values) == expected_names
    for sector_name in expected_names:
        variable_name = f"flux_{sector_name}_posterior"
        assert variable_name in flux_outputs
        assert f"flux_{sector_name}_posterior_country" in flux_outputs
        assert flux_outputs[variable_name].dtype == np.dtype("float32")
        assert flux_outputs[variable_name].attrs["long_name"] == (f"posterior ch4 fluxes from {sector_name}")
        assert flux_outputs[variable_name].attrs["cell_methods"] == "time:mean area:mean"

    assert "flux_energy_waste_posterior" not in flux_outputs
    assert "flux_total_ff_posterior" not in flux_outputs
    for flux_label in ("total", *expected_names):
        covariance = flux_outputs[f"covariance_flux_{flux_label}_posterior_country"].values
        stdev = flux_outputs[f"stdev_flux_{flux_label}_posterior_country"].values
        assert np.isfinite(covariance).all()
        np.testing.assert_allclose(
            np.diagonal(covariance, axis1=1, axis2=2),
            stdev**2,
            rtol=1e-6,
        )
    assert np.isfinite(flux_outputs["covariance_flux_sectors_posterior_country"].values).all()


def test_latest_paris_country_selection_defaults_to_domain_file(
    eastasia_country_file: Path,
) -> None:
    """Latest PARIS outputs do not inject the canonical European country list."""
    countries = _latest_paris_countries(
        country_file=eastasia_country_file,
        domain="EASTASIA",
        country_selections=None,
    )
    expected = Countries.from_file(
        country_file=eastasia_country_file,
        domain="EASTASIA",
        country_code="alpha3",
    )

    assert tuple(countries.matrix.country.values) == tuple(expected.matrix.country.values)
    assert set(countries.matrix.country.values) != set(PARIS_LATEST_COUNTRIES)


def test_latest_paris_country_selection_normalizes_names_to_alpha3(
    europe_country_file: Path,
) -> None:
    """Latest PARIS country names select finite masks with alpha-3 labels."""
    countries = _latest_paris_countries(
        country_file=europe_country_file,
        domain="EUROPE",
        country_selections=["France", "United Kingdom"],
    )
    expected = Countries.from_file(
        country_file=europe_country_file,
        domain="EUROPE",
        country_code="alpha3",
    ).matrix.sel(country=["FRA", "GBR"])

    assert tuple(countries.matrix.country.values) == ("FRA", "GBR")
    xr.testing.assert_identical(countries.matrix, expected)
