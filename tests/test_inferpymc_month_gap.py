"""Synthetic regression scaffolding for monthly sigma/bc with missing month."""

from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions import utils
import openghg_inversions.hbmcmc.inversion_pymc as inversion_pymc_module
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import (
    _weighted_apriori_flux_for_months,
    inferpymc,
)
from openghg_inversions.sigma import SigmaAlignment


def _synthetic_fp_data_one_site_with_missing_month() -> dict[str, xr.Dataset]:
    """Create one-site daily data over 3 months with the middle month missing."""
    all_days = pd.date_range("2019-01-01", "2019-04-01", freq="1D", inclusive="left")
    times = all_days[all_days.month != 2]  # keep Jan + Mar, remove Feb entirely

    ntime = len(times)
    nregion = 3
    bc_regions = ["n", "e", "s", "w"]

    h = np.vstack(
        [
            np.linspace(0.2, 0.8, ntime),
            np.linspace(0.6, 0.1, ntime),
            np.linspace(0.1, 0.4, ntime),
        ]
    ).astype(np.float32)
    h_bc = (np.arange(1, len(bc_regions) + 1)[:, None] * np.linspace(0.01, 0.1, ntime)[None, :]).astype(
        np.float32
    )

    ds = xr.Dataset(
        data_vars={
            "H": (("region", "time"), h),
            "H_bc": (("bc_region", "time"), h_bc),
            "mf": (("time",), np.linspace(1800.0, 1810.0, ntime).astype(np.float32)),
            "mf_error": (("time",), np.full(ntime, 0.2, dtype=np.float32)),
            "mf_repeatability": (("time",), np.full(ntime, 0.05, dtype=np.float32)),
            "mf_variability": (("time",), np.full(ntime, 0.05, dtype=np.float32)),
        },
        coords={
            "time": times,
            "region": np.arange(nregion),
            "bc_region": bc_regions,
        },
    )

    return {"AAA": ds}


def _mock_month_gap_trace(model: pm.Model) -> az.InferenceData:
    """Create deterministic posterior and predictive groups for month-gap routing."""
    coords = {
        "chain": [0],
        "draw": [0],
        **{dim: np.asarray(values) for dim, values in model.coords.items() if values is not None},
    }
    posterior_values = {
        "x": 1.0,
        "bc": 0.25,
        "sigma": 0.2,
        "mu": 2.0,
        "mu_bc": 0.5,
    }
    posterior_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name, value in posterior_values.items():
        dims = cast(tuple[str, ...], ("chain", "draw", *model.named_vars_to_dims[name]))
        shape = tuple(len(coords[dim]) for dim in dims)
        posterior_vars[name] = (dims, np.full(shape, value))

    prediction_dims = ("chain", "draw", "nmeasure")
    prediction_shape = tuple(len(coords[dim]) for dim in prediction_dims)
    return az.InferenceData(
        posterior=xr.Dataset(posterior_vars, coords=coords),
        prior=xr.Dataset(
            {
                "x": (
                    ("chain", "draw", "region"),
                    np.ones((1, 1, len(coords["region"]))),
                )
            },
            coords={dim: coords[dim] for dim in ("chain", "draw", "region")},
        ),
        prior_predictive=xr.Dataset(
            {"y": (prediction_dims, np.ones(prediction_shape))},
            coords={dim: coords[dim] for dim in prediction_dims},
        ),
        posterior_predictive=xr.Dataset(
            {"y": (prediction_dims, np.full(prediction_shape, 2.5))},
            coords={dim: coords[dim] for dim in prediction_dims},
        ),
    )


def test_sigma_alignment_month_gap_monthly_indices_are_contiguous():
    """Frequency-derived sigma indices stay contiguous when a month is missing."""
    fp_data = _synthetic_fp_data_one_site_with_missing_month()

    inv_inputs = make_inv_inputs(
        fp_data,
        sites=["AAA"],
        bc_freq="monthly",
        min_error=0.0,
        min_error_per_site=False,
        start_date="2019-01-01",
    )
    alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency="monthly",
        anchor_time="2019-01-01",
    )

    uniq = np.unique(alignment.period_index.values)
    np.testing.assert_array_equal(uniq, np.array([0, 1]))


def test_inferpymc_routes_month_gap_through_legacy_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy wrapper routes compact monthly inputs without running MCMC."""
    fp_data = _synthetic_fp_data_one_site_with_missing_month()

    inv_inputs = make_inv_inputs(
        fp_data,
        sites=["AAA"],
        bc_freq="monthly",
        min_error=0.0,
        min_error_per_site=False,
        start_date="2019-01-01",
    )
    alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency="monthly",
        anchor_time="2019-01-01",
    )
    inv_inputs = inv_inputs.assign(sigma_freq_index=alignment.period_index.rename("sigma_freq_index"))

    captured: dict[str, Any] = {}

    def fake_sample(model: pm.Model, **kwargs: Any) -> az.InferenceData:
        """Capture legacy routing and return deterministic posterior products."""
        captured["model"] = model
        captured["sample_kwargs"] = kwargs
        return _mock_month_gap_trace(model)

    monkeypatch.setattr(inversion_pymc_module, "sample", fake_sample)
    monkeypatch.setattr(inversion_pymc_module.pm, "NUTS", lambda variables: "mock-nuts-step")
    monkeypatch.setattr(inversion_pymc_module.pm, "Slice", lambda variables: "mock-slice-step")
    monkeypatch.setattr(
        inversion_pymc_module.pm,
        "rhat",
        lambda trace: xr.Dataset({"x": xr.DataArray(1.0)}),
    )

    result = inferpymc(
        inv_inputs=inv_inputs,
        xprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        bcprior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        sigprior={"pdf": "uniform", "lower": 0.1, "upper": 0.4},
        nuts_sampler="pymc",
        nit=1,
        burn=0,
        tune=0,
        nchain=1,
        sigma_per_site=True,
        verbose=False,
        use_bc=True,
        sampler_kwargs={"compute_convergence_checks": False},
    )

    sample_kwargs = captured["sample_kwargs"]
    assert sample_kwargs["draws"] == 1
    assert sample_kwargs["burn"] == 0
    assert sample_kwargs["tune"] == 0
    assert sample_kwargs["chains"] == 1
    assert sample_kwargs["sample_prior_predictive"] is True
    assert sample_kwargs["sample_posterior_predictive"] == ["y"]
    assert sample_kwargs["nuts_sampler"] == "pymc"
    assert sample_kwargs["step"] == ["mock-nuts-step", "mock-slice-step"]

    assert result["sigouts"].sizes["nsigma_time"] == 2
    assert result["sigouts"].sizes["nsigma_site"] == 1
    assert result["xouts"].sizes["nx"] == inv_inputs.sizes["region"]
    assert result["bcouts"].sizes["nbc"] == inv_inputs.sizes["bc_region"]
    assert result["Ytrace"].shape == (inv_inputs.sizes["nmeasure"], 1)
    assert result["YBCtrace"].shape == result["Ytrace"].shape
    assert {"prior", "prior_predictive", "posterior_predictive"}.issubset(result["trace"].groups())


def test_weighted_apriori_flux_handles_missing_month():
    """Weighted prior fluxes use compacted month positions for missing months."""
    flux_array_all = np.array([[[1.0, 3.0]]], dtype=np.float32)
    month_index = np.array([0, 0, 2, 2], dtype=int)

    apriori_flux = _weighted_apriori_flux_for_months(flux_array_all, month_index)

    np.testing.assert_allclose(apriori_flux, np.array([[2.0]], dtype=np.float32))


def test_map_times_to_available_period_positions_handles_gappy_flux_months():
    """Monthly period mapping uses available periods even when months are skipped."""
    times = pd.to_datetime(["2019-01-15", "2019-01-20", "2019-03-10", "2019-04-20"])
    flux_times = pd.to_datetime(["2019-01-01", "2019-03-01", "2019-04-01"])

    positions = utils._map_times_to_available_period_positions(times, flux_times, "monthly")

    np.testing.assert_array_equal(positions, np.array([0, 0, 1, 2]))


def test_map_times_to_available_period_positions_handles_multi_year_flux_time():
    """Yearly period mapping stays stable across multiple available years."""
    times = pd.to_datetime(["2023-03-15", "2023-11-20", "2024-07-10", "2025-04-20"])
    flux_times = pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"])

    positions = utils._map_times_to_available_period_positions(times, flux_times, "yearly")

    np.testing.assert_array_equal(positions, np.array([0, 0, 1, 2]))
