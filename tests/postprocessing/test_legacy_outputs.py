from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import arviz as az
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions import _country_file as country_file_mod
from openghg_inversions import utils
from openghg_inversions.postprocessing import legacy_outputs
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.legacy_outputs import _compute_apriori_flux


def _minimal_legacy_inv_inputs(*, include_bc: bool = False) -> xr.Dataset:
    """Build observation and fallback sensitivity inputs for legacy output tests."""
    data_vars = {
        "H": (("region", "nmeasure"), np.array([[9.0, 9.0]])),
        "mf": (
            ("nmeasure",),
            np.array([1900.0, 1901.0]),
            {"units": "1e-09 mol/mol", "long_name": "observed_mole_fraction"},
        ),
        "mf_error": (("nmeasure",), np.array([2.0, 2.1]), {"units": "1e-09 mol/mol"}),
        "mf_repeatability": (("nmeasure",), np.array([1.0, 1.1]), {"units": "1e-09 mol/mol"}),
        "mf_variability": (("nmeasure",), np.array([1.5, 1.6]), {"units": "1e-09 mol/mol"}),
        "site_indicator": (("nmeasure",), np.array([0, 0])),
        "sigma_freq_index": (("nmeasure",), np.array([7, 8])),
        "min_error": (("nmeasure",), np.zeros(2)),
    }
    coords = {
        "region": np.array([0]),
        "nmeasure": np.arange(2),
        "site": ("nmeasure", np.array(["TAC", "TAC"])),
        "time": (
            "nmeasure",
            np.array(["2019-01-01T00:00:00", "2019-01-01T01:00:00"], dtype="datetime64[ns]"),
        ),
    }
    if include_bc:
        data_vars["H_bc"] = (("bc_region", "nmeasure"), np.array([[1.0, 2.0]]))
        coords["bc_region"] = np.array([0])

    return xr.Dataset(data_vars=data_vars, coords=coords).set_index(nmeasure=["site", "time"])


def _basis_functions_stub() -> SimpleNamespace:
    """Return the minimal basis-functions interface consumed by the legacy adapter."""
    flux = xr.DataArray(
        np.array([[1.0]]),
        dims=("lat", "lon"),
        coords={"lat": [52.0], "lon": [1.0]},
        name="flux",
    )
    return SimpleNamespace(
        flux=flux,
        operator=SimpleNamespace(meta=SimpleNamespace(state_dim="region")),
    )


def _write_minimal_country_file(path: str | Path) -> None:
    """Write a minimal country file for fallback loader tests."""
    with h5py.File(path, "w") as h5:
        h5.create_dataset("lat", data=np.array([52.0], dtype="float32"))
        h5.create_dataset("lon", data=np.array([1.0], dtype="float32"))
        h5.create_dataset("country", data=np.array([[1.0]]))
        h5.create_dataset("name", data=np.array([b"UNITED KINGDOM"]))


def _legacy_inv_out(*, model_data: bool, include_bc: bool = False, chains: int = 1) -> InversionOutput:
    """Build a minimal InversionOutput for legacy adapter tests."""
    draw_count = 3
    posterior_vars = {
        "x": (("chain", "draw", "region"), np.ones((chains, draw_count, 1))),
        "sigma": (
            ("chain", "draw", "nsigma_site", "nsigma_time"),
            np.ones((chains, draw_count, 1, 2)),
        ),
    }
    coords: dict[str, object] = {
        "chain": np.arange(chains),
        "draw": np.arange(draw_count),
        "region": np.array([0]),
        "nsigma_site": np.array([0]),
        "nsigma_time": np.array([0, 1]),
    }
    if include_bc:
        posterior_vars["bc"] = (("chain", "draw", "bc_region"), np.full((chains, draw_count, 1), 0.5))
        coords["bc_region"] = np.array([0])

    groups: dict[str, xr.Dataset] = {"posterior": xr.Dataset(posterior_vars, coords=coords)}
    if model_data:
        constant_vars = {
            "hx": (("nmeasure", "region"), np.array([[0.25], [0.75]])),
            "sigma_freq_index": (("nmeasure",), np.array([0, 1])),
        }
        constant_coords: dict[str, object] = {"nmeasure": np.arange(2), "region": np.array([0])}
        if include_bc:
            constant_vars["hbc"] = (("nmeasure", "bc_region"), np.array([[0.5], [1.5]]))
            constant_coords["bc_region"] = np.array([0])
        groups["constant_data"] = xr.Dataset(constant_vars, coords=constant_coords)

    return InversionOutput(
        trace=cast(Any, az.InferenceData)(**groups),
        inv_inputs=_minimal_legacy_inv_inputs(include_bc=include_bc),
        basis_functions=cast(Any, _basis_functions_stub()),
        run_metadata={
            "start_date": "2019-01-01",
            "end_date": "2019-01-02",
            "sites": ["TAC"],
            "site_lats": [52.5],
            "site_lons": [1.25],
            "split_by_sectors": False,
        },
        model_metadata={"species": "ch4", "domain": "EUROPE"},
    )


@pytest.fixture
def stub_legacy_product_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub heavy postprocessing builders so adapter tests focus on input derivation."""
    nmeasure = np.arange(2)
    hdi = np.array(["lower", "upper"])
    lat = np.array([52.0])
    lon = np.array([1.0])
    country = np.array(["United Kingdom"])

    def fake_concentration_outputs(inv_out: InversionOutput, **kwargs: object) -> xr.Dataset:
        data_vars = {
            "y_posterior_predictive_mean": (("nmeasure",), np.array([1.0, 2.0])),
            "y_posterior_predictive_median": (("nmeasure",), np.array([1.0, 2.0])),
            "y_posterior_predictive_mode": (("nmeasure",), np.array([1.0, 2.0])),
            "y_posterior_predictive_hdi_68": (("nmeasure", "hdi"), np.ones((2, 2))),
            "y_posterior_predictive_hdi_95": (("nmeasure", "hdi"), np.ones((2, 2))),
        }
        if "bc" in cast(Any, inv_out.trace).posterior:
            data_vars.update(
                {
                    "mu_bc_posterior_mean": (("nmeasure",), np.array([0.5, 1.5])),
                    "mu_bc_posterior_median": (("nmeasure",), np.array([0.5, 1.5])),
                    "mu_bc_posterior_mode": (("nmeasure",), np.array([0.5, 1.5])),
                    "mu_bc_posterior_hdi_68": (("nmeasure", "hdi"), np.ones((2, 2))),
                    "mu_bc_posterior_hdi_95": (("nmeasure", "hdi"), np.ones((2, 2))),
                }
            )
        return xr.Dataset(data_vars, coords={"nmeasure": nmeasure, "hdi": hdi})

    def fake_flux_outputs(inv_out: InversionOutput, **kwargs: object) -> xr.Dataset:
        return xr.Dataset(
            {
                "flux_posterior_mode": (("lat", "lon"), np.array([[1.0]])),
                "scaling_posterior_mean": (("lat", "lon"), np.array([[1.0]])),
                "scaling_posterior_mode": (("lat", "lon"), np.array([[1.0]])),
            },
            coords={"lat": lat, "lon": lon},
        )

    def fake_country_outputs(inv_out: InversionOutput, **kwargs: object) -> xr.Dataset:
        return xr.Dataset(
            {
                "country_posterior_mean": (("country",), np.array([1.0])),
                "country_posterior_median": (("country",), np.array([1.0])),
                "country_posterior_mode": (("country",), np.array([1.0])),
                "country_posterior_stdev": (("country",), np.array([0.1])),
                "country_posterior_hdi_68": (("country", "hdi"), np.ones((1, 2))),
                "country_posterior_hdi_95": (("country", "hdi"), np.ones((1, 2))),
                "country_prior_mean": (("country",), np.array([1.0])),
            },
            coords={"country": country, "hdi": hdi},
        )

    monkeypatch.setattr(legacy_outputs, "make_concentration_outputs", fake_concentration_outputs)
    monkeypatch.setattr(legacy_outputs, "make_flux_outputs", fake_flux_outputs)
    monkeypatch.setattr(legacy_outputs, "make_country_outputs", fake_country_outputs)
    monkeypatch.setattr(
        legacy_outputs,
        "flat_basis_for_output",
        lambda inv_out: xr.DataArray(
            np.array([[1]]), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="basis"
        ),
    )
    monkeypatch.setattr(
        legacy_outputs, "_legacy_country_index", lambda domain, country_file=None: np.array([[1]])
    )


def test_compute_apriori_flux_handles_missing_month():
    """Apriori flux weighting handles skipped monthly flux periods."""
    flux = xr.DataArray(
        np.array([[[1.0, 3.0]]]),
        dims=["lat", "lon", "flux_time"],
        coords={
            "lat": [0.0],
            "lon": [0.0],
            "flux_time": pd.to_datetime(["2019-01-01", "2019-03-01"]),
        },
    )
    times = xr.DataArray(
        pd.to_datetime(["2019-01-15", "2019-01-20", "2019-03-10", "2019-03-20"]),
        dims=["nmeasure"],
    )

    apriori_flux = _compute_apriori_flux(flux, "2019-01-01", "2019-04-01", times)

    xr.testing.assert_allclose(
        apriori_flux, xr.DataArray([[2.0]], dims=["lat", "lon"], coords={"lat": [0.0], "lon": [0.0]})
    )


def test_map_times_to_available_period_positions_handles_gappy_flux_months():
    """Period mapping uses the nearest available monthly flux positions."""
    times = pd.to_datetime(["2019-01-15", "2019-01-20", "2019-03-10", "2019-04-20"])
    flux_times = pd.to_datetime(["2019-01-01", "2019-03-01", "2019-04-01"])

    positions = utils._map_times_to_available_period_positions(
        times.to_numpy(), flux_times.to_numpy(), "monthly"
    )

    np.testing.assert_array_equal(positions, np.array([0, 0, 1, 2]))


def test_legacy_country_index_falls_back_when_h5netcdf_open_fails(monkeypatch, tmp_path):
    """Legacy country index loading uses the same direct HDF5 fallback as modern country outputs."""
    country_file = tmp_path / "country_TEST.nc"
    _write_minimal_country_file(country_file)

    def fail_open_dataset(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Unspecified error in H5DSget_num_scales")

    monkeypatch.setattr(country_file_mod.xr, "open_dataset", fail_open_dataset)

    country_index = legacy_outputs._legacy_country_index("TEST", country_file=country_file)

    np.testing.assert_array_equal(country_index, np.array([[1.0]]))


def test_compute_apriori_flux_handles_multi_year_flux_time():
    """Apriori flux weighting handles yearly flux periods across multiple years."""
    flux = xr.DataArray(
        np.array([[[1.0, 2.0, 3.0]]]),
        dims=["lat", "lon", "flux_time"],
        coords={
            "lat": [0.0],
            "lon": [0.0],
            "flux_time": pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
        },
    )
    times = xr.DataArray(
        pd.to_datetime(["2023-03-15", "2023-11-20", "2024-07-10", "2025-04-20"]),
        dims=["nmeasure"],
    )

    apriori_flux = _compute_apriori_flux(flux, "2023-01-01", "2025-05-01", times)

    xr.testing.assert_allclose(
        apriori_flux,
        xr.DataArray([[1.75]], dims=["lat", "lon"], coords={"lat": [0.0], "lon": [0.0]}),
    )


def test_make_legacy_hbmcmc_output_derives_model_data_inputs(
    stub_legacy_product_builders: None,
) -> None:
    """Legacy HBMCMC output no longer requires inferpymc result or side-channel arrays."""
    inv_out = _legacy_inv_out(model_data=True)
    inv_out.output_metadata["legacy_hbmcmc_attrs"] = {
        "Burn in": "1",
        "Tuning steps": "2",
        "Number of chains": "1",
        "Error for each site": "True",
        "Emissions Prior": "pdf,normal,mu,1.0,sigma,0.2",
        "Model error Prior": "pdf,uniform,lower,0.1,upper,10.0",
    }

    output = legacy_outputs.make_legacy_hbmcmc_output(inv_out)

    np.testing.assert_allclose(output["xsensitivity"].values, np.array([[0.25], [0.75]]))
    np.testing.assert_array_equal(output["sigmafreqindex"].values, np.array([0, 1]))
    np.testing.assert_allclose(output["min_model_error"].values, np.zeros(2))
    assert "min_model_error" not in output.attrs
    np.testing.assert_allclose(output["sitelats"].values, np.array([52.5]))
    np.testing.assert_allclose(output["sitelons"].values, np.array([1.25]))
    np.testing.assert_array_equal(output["basisfunctions"].values, np.array([[0]]))
    for name in output.data_vars:
        if np.issubdtype(output[name].dtype, np.floating):
            assert output[name].dtype == np.dtype("float32")
    assert np.isfinite(output["Ymod68"].values).sum() == output["Ymod68"].size
    assert np.isfinite(output["Ymod95"].values).sum() == output["Ymod95"].size
    assert np.isfinite(output["country68"].values).sum() == output["country68"].size
    assert np.isfinite(output["country95"].values).sum() == output["country95"].size
    np.testing.assert_allclose(output["xtrace"].values, np.ones((3, 1)))
    np.testing.assert_allclose(output["sigtrace"].values, np.ones((3, 1, 2)))
    assert output.attrs["Convergence"] == "Unavailable"
    assert output.attrs["Emissions Prior"] == "pdf,normal,mu,1.0,sigma,0.2"


def test_make_legacy_hbmcmc_output_falls_back_to_inv_inputs_for_sensitivities(
    stub_legacy_product_builders: None,
) -> None:
    """Legacy HBMCMC output derives sensitivities from inv_inputs when model data is absent."""
    inv_out = _legacy_inv_out(model_data=False, include_bc=True)

    output = legacy_outputs.make_legacy_hbmcmc_output(inv_out, use_bc=True)

    np.testing.assert_allclose(output["xsensitivity"].values, np.array([[9.0], [9.0]]))
    np.testing.assert_allclose(output["bcsensitivity"].values, np.array([[1.0], [2.0]]))
    np.testing.assert_array_equal(output["sigmafreqindex"].values, np.array([7, 8]))
    np.testing.assert_allclose(output["bctrace"].values, np.full((3, 1), 0.5))
    assert output["bcsensitivity"].dtype == np.dtype("float32")
    assert output["bctrace"].dtype == np.dtype("float32")
    assert output["YaprioriBC"].dtype == np.dtype("float32")
    assert "YaprioriBC" in output
