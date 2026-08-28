"""RHIME integration tests for the fixed within-site OU likelihood."""

from __future__ import annotations

import json

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from openghg_inversions.models.coords import get_coord_registry, registered_model
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.likelihoods import fixed_ou_likelihood_builder
from openghg_inversions.rhime.multisector import build_multisector_rhime_model
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.specs import SectorSpec
from openghg_inversions.rhime.standard import build_standard_rhime_model


def _ou_inputs(*, low_rank: bool = True, zero_base: bool = False) -> xr.Dataset:
    nmeasure = 4
    data = xr.Dataset(
        {
            "mf": ("nmeasure", [1.0, 2.0, 3.0, 4.0]),
            "mf_error": (
                "nmeasure",
                np.zeros(nmeasure) if zero_base else np.full(nmeasure, 0.2),
            ),
            "min_error": ("nmeasure", [0.5, 0.0, 0.0, 0.0]),
        },
        coords={
            "nmeasure": np.arange(nmeasure),
            "site": ("nmeasure", ["MHD", "TAC", "MHD", "TAC"]),
            "time": (
                "nmeasure",
                np.array(
                    [
                        "2020-01-01T00",
                        "2020-01-01T00",
                        "2020-01-01T01",
                        "2020-01-01T02",
                    ],
                    dtype="datetime64[h]",
                ),
            ),
        },
    )
    data["mf"].attrs["units"] = "ppm"
    if low_rank:
        data["low_rank_factor"] = (
            ("nmeasure", "aggregation_rank"),
            [[0.1], [0.2], [0.1], [0.2]],
        )
        data["diagonal_residual_variance"] = (
            "nmeasure",
            np.zeros(nmeasure) if zero_base else [0.03, 0.04, 0.05, 0.06],
        )
    return data


def _build_fixed_ou_model(
    data: xr.Dataset,
    *,
    aggregation_mode: str | None = None,
    fixed_site_amplitudes: float | dict[str, float] | None = None,
    site_amplitude_prior: dict[str, float | str] | None = None,
) -> pm.Model:
    mode = aggregation_mode or (
        "low_rank" if "low_rank_factor" in data else "none"
    )
    with registered_model(coords={"nmeasure": data.nmeasure.values}) as model:
        mean = pm.Data("completed_mean", np.ones(data.sizes["nmeasure"]), dims="nmeasure")
        fixed_ou_likelihood_builder(
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=resolve_aggregation_error(data, mode),
            mean=mean,
            pollution_mean=mean,
            pollution_event_baseline=None,
            output_dim="nmeasure",
            tau_hours={"TAC": 7.0, "MHD": 5.0},
            fixed_site_amplitudes=fixed_site_amplitudes,
            site_amplitude_prior=site_amplitude_prior,
        )
    return model


def test_low_rank_builder_keeps_the_dynamic_cholesky_rank_sized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-rank OU evaluation does not register or factor an observation square."""
    model = _build_fixed_ou_model(
        _ou_inputs(),
        fixed_site_amplitudes={"TAC": 0.7, "MHD": 0.5},
    )

    assert "MvNormal" not in type(model["y"].owner.op).__name__
    assert all(
        dims != ("nmeasure", "nmeasure_cov")
        for dims in model.named_vars_to_dims.values()
    )
    assert "ou_correlation_template" not in model.named_vars
    original_cholesky = np.linalg.cholesky
    cholesky_shapes: list[tuple[int, ...]] = []

    def record_cholesky(value: np.ndarray) -> np.ndarray:
        cholesky_shapes.append(value.shape)
        return original_cholesky(value)

    monkeypatch.setattr(np.linalg, "cholesky", record_cholesky)
    assert np.isfinite(float(model.compile_logp()(model.initial_point())))
    assert cholesky_shapes == [(1, 1)]


def test_fixed_ou_builder_preserves_labels_tau_and_static_base_floor() -> None:
    """Site mappings follow first occurrence and OU is added above the fixed floor."""
    model = _build_fixed_ou_model(
        _ou_inputs(),
        fixed_site_amplitudes={"TAC": 0.7, "MHD": 0.5},
    )

    registry = get_coord_registry(model)
    assert registry is not None
    assert tuple(registry.original_coords["ou_site"]) == ("MHD", "TAC")
    np.testing.assert_allclose(model["ou_tau_hours"].eval(), [5.0, 7.0])
    np.testing.assert_allclose(model["ou_site_amplitude"].eval(), [0.5, 0.7])
    epsilon = np.asarray(model["epsilon"].eval())
    np.testing.assert_allclose(epsilon[0], np.sqrt(0.5**2 + 0.5**2))
    assert epsilon[0] > 0.5


def test_fixed_ou_builder_rejects_a_singular_fixed_complete_covariance() -> None:
    """A zero base and zero OU amplitude fail rather than gaining hidden jitter."""
    data = _ou_inputs(low_rank=False, zero_base=True)
    data["min_error"][:] = 0.0
    with pytest.raises(ValueError, match="positive-definite complete observation covariance"):
        _build_fixed_ou_model(data, fixed_site_amplitudes=0.0)


def test_fixed_ou_builder_creates_an_inferred_labelled_amplitude() -> None:
    """Inferred OU amplitudes use their own site-labelled variable."""
    model = _build_fixed_ou_model(
        _ou_inputs(),
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
    )

    assert model["ou_site_amplitude"] in model.free_RVs
    assert model.named_vars_to_dims["ou_site_amplitude"] == ("ou_site",)
    assert "sigma" not in model.named_vars


def test_fixed_ou_builder_samples_state_and_site_amplitudes_together() -> None:
    data = _ou_inputs(low_rank=True)
    with registered_model(coords={"nmeasure": data.nmeasure.values}):
        state = pm.Normal("state", mu=1.0, sigma=0.2)
        mean = state * pm.Data(
            "state_design",
            np.array([0.8, 1.8, 2.8, 3.8]),
            dims="nmeasure",
        )
        fixed_ou_likelihood_builder(
            observations=data["mf"],
            observation_error=data["mf_error"],
            minimum_error=data["min_error"],
            aggregation_error=resolve_aggregation_error(data, "low_rank"),
            mean=mean,
            pollution_mean=mean,
            pollution_event_baseline=None,
            output_dim="nmeasure",
            tau_hours=5.0,
        )
        trace = pm.sample(
            draws=10,
            tune=10,
            chains=1,
            cores=1,
            random_seed=20260825,
            progressbar=False,
            compute_convergence_checks=False,
        )

    assert trace.posterior["state"].shape == (1, 10)
    assert trace.posterior["ou_site_amplitude"].shape == (1, 10, 2)
    assert np.isfinite(trace.posterior["ou_site_amplitude"]).all()


@pytest.mark.parametrize("mode", ["none", "diagonal", "low_rank", "dense"])
def test_builder_logp_matches_dense_covariance_for_every_aggregation_mode(
    mode: str,
) -> None:
    data = _ou_inputs(low_rank=False)
    factor = np.array([[0.1], [0.2], [0.1], [0.2]])
    residual_variance = np.array([0.03, 0.04, 0.05, 0.06])
    if mode == "none":
        aggregation = np.zeros((4, 4))
    elif mode == "diagonal":
        aggregation = np.diag(residual_variance)
        data["aggregation_error_sd"] = (
            "nmeasure",
            np.sqrt(residual_variance),
        )
    else:
        aggregation = factor @ factor.T + np.diag(residual_variance)
        if mode == "low_rank":
            data["low_rank_factor"] = (
                ("nmeasure", "aggregation_rank"),
                factor,
            )
            data["diagonal_residual_variance"] = (
                "nmeasure",
                residual_variance,
            )
        else:
            data["aggregation_error_covariance"] = (
                ("nmeasure", "nmeasure_cov"),
                aggregation,
            )

    model = _build_fixed_ou_model(
        data,
        aggregation_mode=mode,
        fixed_site_amplitudes={"TAC": 0.7, "MHD": 0.5},
    )
    site_index = np.array([0, 1, 0, 1])
    # Fixed model data follows the active PyMC graph precision before the
    # float64 OU evaluation boundary.
    amplitude = np.asarray(model["ou_site_amplitude"].eval(), dtype=np.float64)
    tau = np.array([5.0, 7.0])
    time_hours = np.array([0.0, 0.0, 1.0, 2.0])
    lag = np.abs(time_hours[:, None] - time_hours[None, :])
    correlation = np.exp(-lag / tau[site_index, None])
    correlation[site_index[:, None] != site_index[None, :]] = 0.0
    observation_variance = np.square(data["mf_error"].values)
    floor_variance = np.maximum(
        np.square(data["min_error"].values)
        - observation_variance
        - np.diag(aggregation),
        0.0,
    )
    covariance = (
        aggregation
        + np.diag(observation_variance + floor_variance)
        + amplitude[site_index, None]
        * amplitude[site_index[None, :]]
        * correlation
    )
    expected = multivariate_normal.logpdf(
        data["mf"].values,
        mean=np.ones(4),
        cov=covariance,
    )

    assert float(model.compile_logp()(model.initial_point())) == pytest.approx(
        expected,
        rel=1.0e-10,
    )


@pytest.mark.parametrize("multisector", [False, True])
def test_standard_recipes_accept_installed_fixed_ou_likelihood(
    multisector: bool,
) -> None:
    """Both ordinary recipes select the installed OU component explicitly."""
    data = _ou_inputs(low_rank=False)
    design = xr.DataArray(
        np.ones((data.sizes["nmeasure"], 1)),
        dims=("nmeasure", "region"),
        coords={"nmeasure": data.nmeasure, "region": ["r1"]},
    )
    common = {
        "observations": data["mf"],
        "observation_error": data["mf_error"],
        "minimum_error": data["min_error"],
        "aggregation_error": resolve_aggregation_error(data, "none"),
        "use_bc": False,
        "likelihood_builder": fixed_ou_likelihood_builder,
        "likelihood_kwargs": {"tau_hours": 5.0, "fixed_site_amplitudes": 0.5},
    }
    if multisector:
        source_design = xr.concat(
            [design, 2.0 * design],
            dim=xr.IndexVariable("source", ["ff", "ocean"]),
        )
        model = build_multisector_rhime_model(
            source_design,
            sectors=(
                SectorSpec(
                    name="fossil fuel",
                    flux_source="ff",
                    variable_suffix="ff",
                    x_prior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
                ),
                SectorSpec(
                    name="ocean",
                    flux_source="ocean",
                    variable_suffix="ocean",
                    x_prior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
                ),
            ),
            **common,
        )
    else:
        model = build_standard_rhime_model(
            design,
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            **common,
        )

    assert {"y", "epsilon", "ou_site_amplitude", "ou_tau_hours"} <= set(
        model.named_vars
    )
    assert "sigma" not in model.named_vars


def test_likelihood_trace_annotation_round_trips_ou_identity_and_units(tmp_path) -> None:
    """Raw trace metadata identifies fixed OU configuration and concentration units."""
    idata = az.InferenceData(
        posterior=xr.Dataset(
            {
                "ou_site_amplitude": (
                    ("chain", "draw", "ou_site"),
                    np.ones((1, 1, 2)),
                ),
                "epsilon": (("chain", "draw", "nmeasure"), np.ones((1, 1, 4))),
            },
            coords={"ou_site": ["MHD", "TAC"]},
        ),
        constant_data=xr.Dataset(
            {"ou_tau_hours": ("ou_site", [5.0, 7.0])},
            coords={"ou_site": ["MHD", "TAC"]},
        ),
    )
    identity = {
        "module": "openghg_inversions.rhime.likelihoods",
        "qualname": "fixed_ou_likelihood_builder",
    }
    options = {"tau_hours": {"MHD": 5.0, "TAC": 7.0}}
    annotate_likelihood_trace(
        idata,
        builder_identity=identity,
        likelihood_kwargs=options,
        concentration_units="ppm",
    )
    path = tmp_path / "ou-trace.nc"
    idata.to_netcdf(path)
    loaded = az.from_netcdf(path)

    assert json.loads(loaded.attrs["rhime_likelihood_builder"]) == identity
    assert json.loads(loaded.attrs["rhime_likelihood_kwargs"]) == options
    assert loaded.attrs["rhime_mismatch_component"] == "fixed_within_site_ou"
    assert tuple(loaded.posterior.ou_site.values) == ("MHD", "TAC")
    assert loaded.posterior["ou_site_amplitude"].attrs["units"] == "ppm"
    assert loaded.posterior["epsilon"].attrs["units"] == "ppm"
    assert loaded.constant_data["ou_tau_hours"].attrs["units"] == "h"
