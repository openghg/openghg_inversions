"""Public unsampled graphs and exact fixed-OU prior prediction."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.rhime.co2 import (
    materialize_co2_inputs,
    build_co2_from_materialized,
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

    def materialize(_prepared: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        return inputs[list(variable_names)]

    def unexpected_sampling(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Building a staged graph must not sample the posterior")

    for module in (co2_runner, co2_cached_sigma_runner):
        if module is co2_runner:
            monkeypatch.setattr(module, "_materialize_pymc_inputs_and_priors",
                                lambda prepared, *, variable_names, prior_parameters: (
                                    materialize(prepared, variable_names=variable_names), prior_parameters,
                                ))
        else:
            monkeypatch.setattr(module, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(pm, "sample", unexpected_sampling)
    prepared = cast(Any, Prepared())
    inputs, priors = materialize_co2_inputs(prepared_inputs=prepared, no_model_error=True)
    ordinary = build_co2_from_materialized(inputs, prepared_inputs=prepared, priors=priors, no_model_error=True)
    assert ordinary.output_contract.variable_roles["model_mean"] == "modelled_concentration"
    cached = build_rhime_co2_cached_sigma(
        prepared_inputs=prepared,
        tau_hours=4.0,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.5},
    )
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
    with cached.model:
        pm.compute_log_likelihood(posterior)
        pm.sample_posterior_predictive(posterior, extend_inferencedata=True, random_seed=918)
    posterior = co2_cached_sigma_runner._annotate_cached_co2_trace(
        posterior,
        co2_cached_sigma_runner.co2_cached_sigma_build_result(cached, prepared),
        concentration_units="ppm",
    )
    path = tmp_path / "cached-posterior.nc"
    save_trace(posterior, path)
    replay = load_trace(path)
    xr.testing.assert_identical(replay.log_likelihood.to_dataset(), posterior.log_likelihood.to_dataset())
    assert replay.log_likelihood["y"].attrs["rhime_normalized_log_likelihood"] == 1
