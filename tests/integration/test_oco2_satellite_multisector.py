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

from openghg_inversions.inversion_data.preparation import RhimePreparedInputs, prepare_rhime_inputs
from openghg_inversions.rhime import (
    RhimeSampler,
    resolve_rhime_options,
    run_rhime_from_prepared_inputs,
)


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
_FLUX_SOURCES = ("anth", "resp", "gpp_atm")


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
def test_real_oco2_prepared_inputs_run_through_multisector_model(
    real_oco2_prepared_inputs: RhimePreparedInputs,
) -> None:
    """Sample the prepared satellite columns through the public RHIME model path."""
    setup = resolve_rhime_options(
        params={
            "species": "co2",
            "sites": ["OCO2-EASTASIA"],
            "averaging_period": ["1H"],
            "domain": "EASTASIA",
            "start_date": "2022-03-31 04:00:00",
            "end_date": "2022-04-01 04:08:10",
            "output_name": "oco2_eastasia_6obs_3sector_model_test",
            "output_format": "none",
            "flux_sources": list(_FLUX_SOURCES),
            "x_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
            "use_bc": True,
            "no_model_error": True,
        },
        multisector=True,
    )
    result = run_rhime_from_prepared_inputs(
        prepared_inputs=real_oco2_prepared_inputs,
        run_spec=setup.run_spec,
        sampler=RhimeSampler(draws=1, tune=0, chains=1, progressbar=False),
    )

    assert result.inv_inputs.sizes["nmeasure"] == 6
    assert tuple(result.inv_inputs.source.values) == _FLUX_SOURCES
    assert {"x_anth", "x_resp", "x_gpp_atm"}.issubset(result.idata.posterior)
    assert result.model_build_result is not None
    assert result.model_build_result.variable_roles["flux_scale:anth"] == "x_anth"
