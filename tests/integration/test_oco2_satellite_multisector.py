"""Real-data checks for the short OCO2 time-resolved multisector run.

These tests deliberately use the shared ACRG OCO2 stores. They are marked
``slow`` and skip when the external data are unavailable, so the ordinary CI
suite remains self-contained. Paths can be overridden for another cluster via
the ``OPENGHG_OCO2_*`` environment variables below.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data.preparation import RhimePreparedInputs, prepare_rhime_inputs


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_STORE = Path(
    os.environ.get("OPENGHG_OCO2_DATA_STORE", "/group/chem/acrg/object_stores/temp/OCO2_test")
)
_FOOTPRINT_STORE = Path(
    os.environ.get(
        "OPENGHG_OCO2_FOOTPRINT_STORE",
        "/group/chem/acrg/object_stores/OCO2/OCO2_HR_bk1day_2022",
    )
)
_BASIS_DIRECTORY = Path(
    os.environ.get("OPENGHG_OCO2_BASIS_DIRECTORY", "/group/chem/acrg/LPDM/basis_functions")
)
_BC_BASIS_DIRECTORY = Path(
    os.environ.get("OPENGHG_OCO2_BC_BASIS_DIRECTORY", "/group/chem/acrg/LPDM/bc_basis_functions")
)
_COUNTRY_DIRECTORY = Path(
    os.environ.get("OPENGHG_OCO2_COUNTRY_DIRECTORY", "/group/chem/acrg/LPDM/countries")
)
_PARIS_OUTPUT = Path(
    os.environ.get(
        "OPENGHG_OCO2_PARIS_OUTPUT",
        _PROJECT_ROOT
        / "run_artifacts/oco2_multisector_short/output/"
        "oco2_eastasia_6obs_3sector_flux_co2_EASTASIA_2022-03-31 04:00:00.nc",
    )
)
_SECTOR_DIAGNOSTICS = Path(
    os.environ.get(
        "OPENGHG_OCO2_SECTOR_DIAGNOSTICS",
        _PROJECT_ROOT
        / "run_artifacts/oco2_multisector_short/output/"
        "oco2_eastasia_6obs_3sector2022-03-31 04:00:00_sector_flux_diagnostics.nc",
    )
)

_FLUX_SOURCES = ("anth", "resp", "gpp_atm")
_SECTORS = ("anthropogenic", "respiration", "gpp")


def _skip_if_missing(*paths: Path) -> None:
    """Skip a real-data check with all unavailable inputs in the message."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"OCO2 integration data unavailable: {missing!r}")


@pytest.fixture(scope="module")
def real_oco2_prepared_inputs(tmp_path_factory: pytest.TempPathFactory) -> RhimePreparedInputs:
    """Prepare the six-sounding OCO2 sample with a fresh weighted basis."""
    _skip_if_missing(
        _DATA_STORE,
        _FOOTPRINT_STORE,
        _BASIS_DIRECTORY,
        _BC_BASIS_DIRECTORY,
        _COUNTRY_DIRECTORY,
    )
    basis_output_path = tmp_path_factory.mktemp("oco2_multisector_basis")
    return prepare_rhime_inputs(
        species="co2",
        sites=["OCO2-EASTASIA"],
        domain="EASTASIA",
        averaging_period=["1H"],
        # The first observation is 2022-04-01 04:07:34. Start one day
        # earlier so all 24 high-frequency footprint lags have flux data.
        start_date="2022-03-31 04:00:00",
        end_date="2022-04-01 04:08:10",
        output_name="oco2_eastasia_6obs_3sector_test",
        flux_sources=list(_FLUX_SOURCES),
        split_by_sectors=True,
        bc_store=str(_DATA_STORE),
        obs_store=str(_DATA_STORE),
        footprint_store=str(_FOOTPRINT_STORE),
        emissions_store=str(_DATA_STORE),
        fp_height=["column"],
        fp_species="co2",
        time_resolved=[True],
        inlet=["column"],
        instrument=[None],
        max_level=[3],
        platform=["satellite"],
        use_bc=True,
        basis_algorithm="weighted",
        nbasis=8,
        basis_directory=str(_BASIS_DIRECTORY),
        bc_basis_case="NESW",
        bc_basis_directory=_BC_BASIS_DIRECTORY,
        country_directory=str(_COUNTRY_DIRECTORY),
        bc_input="cams",
        basis_output_path=str(basis_output_path),
        averaging_error=True,
        min_error=0.0,
    )


@pytest.mark.slow
def test_real_oco2_store_prepares_time_resolved_multisector_columns(
    real_oco2_prepared_inputs: RhimePreparedInputs,
) -> None:
    """Check scientific invariants at the store-to-model input boundary."""
    prepared = real_oco2_prepared_inputs
    inputs = prepared.inv_inputs
    flux = prepared.basis_functions.flux

    assert inputs.sizes["nmeasure"] == 6
    assert inputs.sizes["H_back"] == 24
    assert inputs.sizes["source"] == 3
    assert inputs.sizes["region"] == 8
    assert tuple(inputs.source.values) == _FLUX_SOURCES
    assert {"fp", "fp_residual", "fp_time_resolved"}.issubset(inputs)
    assert inputs["H"].dims == ("region", "nmeasure", "source")
    assert np.isfinite(inputs["H"]).all().item()
    for source in _FLUX_SOURCES:
        assert np.any(np.abs(inputs["H"].sel(source=source).values) > 0.0)

    observation_times = pd.DatetimeIndex(inputs.indexes["nmeasure"].get_level_values("time"))
    assert observation_times[0].floor("s") == pd.Timestamp("2022-04-01 04:07:34")
    assert observation_times[-1] <= pd.Timestamp("2022-04-01 04:08:10")
    assert observation_times[-1] - observation_times[0] < pd.Timedelta(minutes=1)

    assert tuple(flux.source.values) == _FLUX_SOURCES
    assert flux.sizes["time"] == 25
    assert pd.Timestamp(flux.time.min().item()) <= observation_times[0] - pd.Timedelta(hours=24)
    assert float(flux.sel(source="anth").min()) >= 0.0
    assert float(flux.sel(source="anth").max()) > 0.0
    assert float(flux.sel(source="resp").min()) >= 0.0
    assert float(flux.sel(source="resp").max()) > 0.0
    assert float(flux.sel(source="gpp_atm").min()) < 0.0
    assert float(flux.sel(source="gpp_atm").max()) <= 0.0
    assert not {"all", "nep"}.intersection(str(source) for source in flux.source.values)

    # OCO2 stores the retrieved lower-column signal separately from prior
    # corrections. Their sum must recover a physically plausible XCO2 column.
    prior_corrections = inputs["mf_prior_factor"] + inputs["mf_prior_upper_level_factor"]
    full_column = inputs["mf"] + prior_corrections
    assert np.isfinite(prior_corrections).all().item()
    assert bool(((full_column > 400.0) & (full_column < 450.0)).all())
    assert inputs["mf"].attrs["units"] == "1e-06"
    assert inputs["H_bc"].attrs["satellite_column_bc_scale"].startswith(
        "Applied to satellite rows"
    )
    assert np.isfinite(inputs["H_bc"]).all().item()

    assert prepared.site_metadata.site.values.tolist() == ["OCO2-EASTASIA"]
    assert prepared.site_metadata.averaging_period.values.tolist() == ["1H"]


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Duplicate dimension names present:UserWarning")
def test_existing_oco2_paris_output_preserves_sector_totals() -> None:
    """Validate the generated real-data PARIS and diagnostic sector products."""
    _skip_if_missing(_PARIS_OUTPUT, _SECTOR_DIAGNOSTICS)

    with xr.open_dataset(_PARIS_OUTPUT, decode_cf=False) as paris:
        assert paris.attrs["domain"] == "EASTASIA"
        assert paris.attrs["species"] == "co2"
        assert paris.attrs["paris_flux_template_version"] == "v03"
        assert tuple(paris.sector.values) == _SECTORS
        assert paris.sizes["sector"] == 3
        assert paris.sizes["country"] == 31
        assert paris.sizes["time"] == 25
        assert paris.sizes["latitude"] == 340
        assert paris.sizes["longitude"] == 391

        for state in ("prior", "posterior"):
            sector_sum = sum(paris[f"flux_{sector}_{state}"] for sector in _SECTORS)
            np.testing.assert_allclose(
                paris[f"flux_total_{state}"],
                sector_sum,
                rtol=1e-6,
                # The PARIS schema stores float32 fields. Near cancellation,
                # summing the three independently encoded sectors differs by
                # a few 1e-12 to 1e-11 mol m-2 s-1.
                atol=2e-11,
            )
            country_sector_sum = sum(
                paris[f"flux_{sector}_{state}_country"] for sector in _SECTORS
            )
            np.testing.assert_allclose(
                paris[f"flux_total_{state}_country"],
                country_sector_sum,
                rtol=5e-5,
                atol=2e6,
            )
            assert np.isfinite(paris[f"flux_total_{state}"]).all().item()

        assert paris["flux_total_prior"].attrs["units"] == "mol m-2 s-1"
        assert paris["flux_total_prior_country"].attrs["units"] == "kg yr-1"
        assert float(paris["flux_anthropogenic_prior"].min()) >= 0.0
        assert float(paris["flux_respiration_prior"].min()) >= 0.0
        assert float(paris["flux_gpp_prior"].min()) < 0.0
        assert float(paris["flux_gpp_prior"].max()) <= 0.0
        for sector in _SECTORS:
            assert f"flux_{sector}_posterior_inversion_grid" in paris
        assert "covariance_flux_sectors_posterior_country" in paris

    with xr.open_dataset(_SECTOR_DIAGNOSTICS) as diagnostics:
        for state in ("prior", "posterior"):
            sector_sum = sum(
                diagnostics[f"flux_{sector}_{state}_mean"] for sector in _SECTORS
            )
            np.testing.assert_allclose(
                diagnostics[f"flux_total_{state}_mean"],
                sector_sum,
                rtol=1e-6,
                atol=2e-11,
            )
        for sector in _SECTORS:
            assert f"scaling_{sector}_prior_mean" in diagnostics
            assert f"scaling_{sector}_posterior_mean" in diagnostics
