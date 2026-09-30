"""Public unsampled graphs and exact fixed-OU prior prediction."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import (
    build_rhime_co2,
    build_rhime_co2_cached_sigma,
    sample_co2_cached_prior_predictive,
)
from openghg_inversions.rhime.co2 import co2_cached_sigma_runner, co2_runner
from openghg_inversions.serialization import load_trace, save_trace


def test_unsampled_builders_and_correlated_cached_prior(monkeypatch: Any, tmp_path: Path) -> None:
    inputs = xr.Dataset(
        {
            "H": (("nmeasure", "region"), [[0.5], [0.8], [1.0]]),
            "alpha_prior_mean": ("region", [1.0]),
            "alpha_prior_covariance": (("region", "region_cov"), [[0.1]]),
            "fixed_prior_contribution": ("nmeasure", [0.1, 0.2, 0.3]),
            "mf": ("nmeasure", [1.0, 1.2, 1.4]),
            "mf_error": ("nmeasure", [0.1, 0.1, 0.1]),
            "aggregation_error_covariance": (
                ("nmeasure", "nmeasure_cov"),
                np.eye(3) * 0.02 + 0.005,
            ),
        },
        coords={
            "nmeasure": [0, 1, 2],
            "region": ["flux"],
            "site": ("nmeasure", ["AAA"] * 3),
            "time": (
                "nmeasure",
                np.asarray(["2021-01-01T00", "2021-01-01T01", "2021-01-01T03"], dtype="datetime64[h]"),
            ),
        },
    )
    inputs["mf"].attrs["units"] = "ppm"

    class Prepared:
        inv_inputs = inputs
        rhime_inputs = None
        aggregation_error_mode = "dense"

        def validated(self) -> "Prepared":
            return self

    def materialize(_prepared: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        return inputs[list(variable_names)]

    def unexpected_sampling(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Building a staged graph must not sample the posterior")

    for module in (co2_runner, co2_cached_sigma_runner):
        monkeypatch.setattr(module, "materialize_pymc_inputs", materialize)
        monkeypatch.setattr(module, "sample_rhime_model", unexpected_sampling)
    prepared = cast(Any, Prepared())
    ordinary = build_rhime_co2(prepared_inputs=prepared, no_model_error=True)
    assert ordinary.variable_roles["model_mean"] == "modelled_concentration"
    cached = build_rhime_co2_cached_sigma(
        prepared_inputs=prepared,
        tau_hours=4.0,
        site_amplitude_prior_scale=0.5,
    )
    with pytest.warns(UserWarning, match="Potentials"):
        trace = sample_co2_cached_prior_predictive(
            cached, prepared, draws=2000, random_seed=917,
        )
    assert "posterior" not in trace.children
    assert "log_likelihood" not in trace.children
    assert trace.prior_predictive["y"].attrs["units"] == "ppm"
    xr.testing.assert_equal(trace.prior.coords["time"], inputs.coords["time"])
    xr.testing.assert_equal(trace.prior.coords["site"], inputs.coords["site"])
    mean = trace.prior["modelled_concentration"].values[0]
    amplitudes = trace.prior["ou_site_amplitude"].values[0, :, 0]
    residual = trace.prior_predictive["y"].values[0] - mean
    hours = np.asarray([0.0, 1.0, 3.0])
    correlation = np.exp(-np.abs(hours[:, None] - hours[None, :]) / 4.0)
    base = np.eye(3) * 0.03 + 0.005
    # An independent dense Cholesky oracle checks the full correlated noise,
    # rather than just the marginal standard deviations.
    whitened = np.stack([
        np.linalg.solve(np.linalg.cholesky(base + amplitude**2 * correlation), noise)
        for amplitude, noise in zip(amplitudes, residual, strict=True)
    ])
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(3), atol=0.08)
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=0.08)

    # The cached normalized likelihood is stored as one value per joint
    # observation vector. Its provenance must survive canonical NetCDF replay.
    posterior = xr.DataTree.from_dict({"posterior": trace.prior.to_dataset().isel(draw=slice(0, 2))})
    posterior = co2_cached_sigma_runner._append_joint_outputs(
        posterior,
        cached_model=cached,
        observations=inputs["mf"],
        posterior_predictive=True,
        random_seed=918,
    )
    path = tmp_path / "cached-posterior.nc"
    save_trace(posterior, path)
    replay = load_trace(path)
    xr.testing.assert_identical(replay.log_likelihood.to_dataset(), posterior.log_likelihood.to_dataset())
    assert replay.log_likelihood["y"].attrs["rhime_normalized_log_likelihood"] == 1
