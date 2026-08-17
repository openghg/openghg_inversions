"""Integration coverage for the multisector RHIME-to-PARIS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.make_paris_outputs import (
    PARIS_LATEST_COUNTRIES,
    paris_template_files,
)
from openghg_inversions.rhime import run_rhime_multisector


_CDL_VARIABLE = re.compile(
    r"^\s*(byte|short|int|float|double|string)\s+([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)\s*;"
)
_CDL_ATTRIBUTE = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*):([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+)")
_NETCDF_DTYPES = {
    "byte": np.dtype("int8"),
    "short": np.dtype("int16"),
    "int": np.dtype("int32"),
    "float": np.dtype("float32"),
    "double": np.dtype("float64"),
}


@dataclass(frozen=True)
class _CdlVariable:
    """Describe one numeric variable declared by the PARIS CDL template."""

    dtype: np.dtype
    dims: tuple[str, ...]
    attrs: dict[str, str]


def _expanded_numeric_flux_schema(
    template: Path,
    *,
    species: str,
    sectors: tuple[str, ...],
) -> dict[str, _CdlVariable]:
    """Parse and expand emitted numeric variables from the latest flux CDL.

    Args:
        template: Latest PARIS flux CDL path.
        species: Species text substituted into template attributes.
        sectors: Concrete sector names substituted for ``sector_name``.

    Returns:
        Numeric output schema keyed by expanded variable name, excluding the
        optional inversion-grid variables disabled by this integration run.
    """
    parsed: dict[str, _CdlVariable] = {}
    in_variables = False
    for line in template.read_text(encoding="utf-8").splitlines():
        if line.startswith("variables:"):
            in_variables = True
            continue
        if not in_variables:
            continue
        variable_match = _CDL_VARIABLE.match(line)
        if variable_match is not None:
            netcdf_type, name, raw_dims = variable_match.groups()
            if netcdf_type in _NETCDF_DTYPES:
                parsed[name] = _CdlVariable(
                    dtype=_NETCDF_DTYPES[netcdf_type],
                    dims=tuple(dim.strip() for dim in raw_dims.split(",")),
                    attrs={},
                )
            continue
        attribute_match = _CDL_ATTRIBUTE.match(line)
        if attribute_match is None:
            continue
        name, attribute, raw_value = attribute_match.groups()
        if name in parsed and attribute != "_FillValue":
            value = raw_value.strip().strip('"').replace("<species>", species)
            parsed[name].attrs[attribute] = value

    expanded: dict[str, _CdlVariable] = {}
    for name, variable in parsed.items():
        if "_inversion_grid" in name:
            continue
        concrete_sectors = sectors if "sector_name" in name else (None,)
        for sector in concrete_sectors:
            output_name = name if sector is None else name.replace("sector_name", sector)
            attrs = {
                key: (
                    value
                    if sector is None
                    else value.replace("<sector_name>", sector).replace("sector_name", sector)
                )
                for key, value in variable.attrs.items()
            }
            expanded[output_name] = _CdlVariable(variable.dtype, variable.dims, attrs)
    return expanded


def _assert_raw_netcdf_matches_schema(
    dataset: xr.Dataset,
    expected: dict[str, _CdlVariable],
) -> str:
    """Validate raw NetCDF numeric variables and return Dataset.info output.

    Args:
        dataset: Runner output reopened with CF decoding disabled.
        expected: Expanded numeric CDL schema.

    Returns:
        The ncdump-like ``Dataset.info`` text used as schema diagnostics.
    """
    info_buffer = StringIO()
    dataset.info(buf=info_buffer)
    schema_info = info_buffer.getvalue()

    actual_numeric = {
        name for name, variable in dataset.variables.items() if np.issubdtype(variable.dtype, np.number)
    }
    assert actual_numeric == set(expected), schema_info
    for name, schema in expected.items():
        variable = dataset[name]
        assert variable.dims == schema.dims, schema_info
        assert variable.dtype == schema.dtype, schema_info
        if schema.dtype == np.dtype("float32"):
            assert np.isnan(variable.attrs["_FillValue"]), schema_info
        attrs_without_fill = {key: value for key, value in variable.attrs.items() if key != "_FillValue"}
        assert attrs_without_fill == schema.attrs, schema_info

    return schema_info


@pytest.mark.slow
def test_multisector_rhime_pipeline_writes_latest_paris_flux_schema(
    tac_ch4_data_args: dict[str, Any],
    default_bc_basis_directory: Path,
    europe_country_file: Path,
    tmp_path: Path,
) -> None:
    """Run real multisector RHIME sampling through latest PARIS NetCDF writes.

    The two store-backed flux products are numerically equal but have distinct
    source names and dimension order, exercising labelled alignment during
    preparation as well as PyMC sampling, multisector postprocessing, and the
    runner's on-disk NetCDF schema.
    """
    flux_sources = ("total-ukghg-edgar7", "total-ukghg-edgar7-shuffled")
    args = tac_ch4_data_args.copy()
    args.update(
        {
            "flux_sources": list(flux_sources),
            "sector_sources": {"FF": flux_sources[0], "ocean": flux_sources[1]},
            "output_name": "multisector_paris_pipeline",
            "output_path": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "bc_basis_directory": default_bc_basis_directory,
            "nbasis": 4,
            "draws": 2,
            "burn": 0,
            "tune": 0,
            "chains": 1,
            "reload_merged_data": False,
            "output_format": "paris",
            "save_inversion_output": False,
            "country_file": europe_country_file,
            "paris_postprocessing_kwargs": {
                "template_version": "latest",
                "inversion_grid": False,
                "flux_frequency": "yearly",
            },
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigma_prior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sample_kwargs": {"random_seed": 405, "compute_convergence_checks": False},
        }
    )
    args.pop("emissions_name")

    result = run_rhime_multisector(**args)

    assert result.basis_functions is not None
    xr.testing.assert_allclose(
        result.basis_functions.flux.sel(source=flux_sources[0], drop=True),
        result.basis_functions.flux.sel(source=flux_sources[1], drop=True),
    )
    flux = result.outputs["paris_flux"]
    for state in ("prior", "posterior"):
        np.testing.assert_allclose(
            flux[f"flux_total_{state}"],
            flux[f"flux_ff_{state}"] + flux[f"flux_ocean_{state}"],
            rtol=1e-6,
        )

    expected_flux_path = tmp_path / "multisector_paris_pipeline_flux_ch4_EUROPE_2019-01-01.nc"
    expected_diagnostics_path = tmp_path / "multisector_paris_pipeline2019-01-01_sector_flux_diagnostics.nc"
    assert result.output_metadata["paris_flux_path"] == str(expected_flux_path)
    assert result.output_metadata["sector_flux_diagnostics_path"] == str(expected_diagnostics_path)
    assert expected_flux_path.is_file()
    assert expected_diagnostics_path.is_file()
    assert "inversion_output_path" not in result.output_metadata
    assert "paris_concentration" not in result.outputs
    assert "paris_concentration_path" not in result.output_metadata
    assert not list(tmp_path.glob("*concentration*.nc"))

    expected_schema = _expanded_numeric_flux_schema(
        paris_template_files("latest").flux,
        species="ch4",
        sectors=("ff", "ocean"),
    )
    with xr.open_dataset(expected_flux_path, decode_cf=False) as reloaded:
        schema_info = _assert_raw_netcdf_matches_schema(reloaded, expected_schema)
        assert reloaded.encoding["unlimited_dims"] == {"time"}
        assert reloaded.sizes["country"] == len(PARIS_LATEST_COUNTRIES)
        assert reloaded.sizes["sector"] == 2
        assert reloaded.sizes["percentile"] == 2
        assert reloaded.sizes["nbnds"] == 2
        assert reloaded.sizes["longitude"] == 391
        assert reloaded.sizes["latitude"] == 293
        assert reloaded.sizes["time"] == 1
        assert tuple(reloaded.country.values) == PARIS_LATEST_COUNTRIES
        assert tuple(reloaded.country_2.values) == PARIS_LATEST_COUNTRIES
        assert tuple(reloaded.sector.values) == ("ff", "ocean")
        assert tuple(reloaded.sector_2.values) == ("ff", "ocean")
        assert reloaded.country.attrs["long_name"] == "country_ISO_3166_1_alpha3"
        assert reloaded.country_2.attrs == reloaded.country.attrs
        assert reloaded.sector.attrs["long_name"] == "short name of flux sector"
        assert reloaded.sector_2.attrs == reloaded.sector.attrs
        assert reloaded.attrs["paris_flux_template_version"] == "v03"
        assert {
            "title",
            "institution",
            "inversion_system",
            "inversion_system_version",
            "apriori_description",
            "transport_model",
            "domain",
            "species",
            "project",
            "references",
            "history",
            "conventions",
            "license",
        }.issubset(reloaded.attrs)
        assert reloaded["covariance_flux_total_posterior_country"].dims == (
            "country",
            "country_2",
            "time",
        )
        for sector in ("ff", "ocean"):
            assert reloaded[f"covariance_flux_{sector}_posterior_country"].dims == (
                "country",
                "country_2",
                "time",
            )
        assert reloaded["covariance_flux_sectors_posterior_country"].dims == (
            "sector_2",
            "sector",
            "country",
            "time",
        )
        assert "covariance_flux_total_posterior_country" in schema_info
        assert "covariance_flux_sectors_posterior_country" in schema_info
