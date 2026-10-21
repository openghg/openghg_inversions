import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.models.priors import (
    lognormal_mu_sigma,
    parse_prior,
    positive_prior_args,
)


def test_lognormal_mu_sigma_matches_requested_moments() -> None:
    """Check lognormal moment conversion reproduces the requested moments."""
    mu, sigma = lognormal_mu_sigma(2.0, 0.5)
    expected_mean = np.exp(mu + 0.5 * sigma**2)
    expected_stdev = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
    assert np.isclose(expected_mean, 2.0)
    assert np.isclose(expected_stdev, 0.5)


def test_lognormal_mu_sigma_supports_array_parameters() -> None:
    """Check lognormal moment conversion works elementwise for state arrays."""
    mean = np.array([1.0, 2.0])
    stdev = np.array([0.2, 0.5])
    mu, sigma = lognormal_mu_sigma(mean, stdev)

    np.testing.assert_allclose(np.exp(mu + 0.5 * sigma**2), mean)


def test_parse_prior_reparameterised_lognormal_uses_latent_name() -> None:
    """Check reparameterized lognormal priors use the ``_latent`` naming convention."""
    with pm.Model(coords={"nx": np.arange(3)}) as model:
        prior = parse_prior(
            "x",
            {"pdf": "lognormal", "mean": 1.5, "stdev": 0.2, "reparameterise": True},
            dims="nx",
        )

    assert prior.name == "x"
    assert "x_latent" in model.named_vars
    assert "x" in model.named_vars


def test_parse_prior_rejects_unknown_distribution() -> None:
    """Check parse_prior rejects unsupported continuous distributions."""
    with pm.Model():
        with pytest.raises(ValueError, match="continuous distribution"):
            parse_prior("bad", {"pdf": "definitely_not_real"})


@pytest.mark.parametrize(
    "prior",
    [
        {"pdf": "normal", "mu": 0.0, "sigma": 1.0},
        {"pdf": "uniform", "lower": -1.0, "upper": 1.0},
    ],
)
def test_positive_prior_args_rejects_negative_support(prior: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive|lower bound"):
        positive_prior_args(prior)


def test_positive_prior_args_normalises_distribution_name() -> None:
    assert positive_prior_args({"pdf": "Half-Normal", "sigma": 1.0}) == {
        "pdf": "halfnormal",
        "sigma": 1.0,
    }


def test_parse_prior_registers_labels_without_mutating_inputs() -> None:
    from openghg_inversions.models.coords import registered_model
    from openghg_inversions.models.priors import prior_records

    labelled_mu = xr.DataArray([1.0, 2.0], dims="region", coords={"region": ["a", "b"]})
    with registered_model() as model:
        parse_prior("x", {"pdf": "normal", "mu": labelled_mu, "sigma": 0.2}, dims="region")
    np.testing.assert_array_equal(model["x_prior_mu"].get_value(), [1.0, 2.0])
    assert prior_records(model)["x"] == {"pdf": "normal", "mu": {"data": "x_prior_mu"}, "sigma": 0.2}
    assert "x_prior_sigma" not in model
    assert labelled_mu.region.values.tolist() == ["a", "b"]


@pytest.mark.parametrize("bounds", [
    {"lower": 1.0}, {"upper": 0.0}, {"lower": 2.0, "upper": 1.0},
    {"lower": float("nan")}, {"upper": float("inf")},
])
def test_uniform_rejects_invalid_bound_pairs_including_effective_defaults(bounds):
    with pm.Model(), pytest.raises(ValueError, match="Uniform prior.*bound"):
        parse_prior("bounded", {"pdf": "uniform", **bounds})


@pytest.mark.parametrize("bounds", [{}, {"lower": 0.5}, {"upper": 0.5}])
def test_uniform_accepts_valid_effective_default_bounds(bounds):
    with pm.Model() as model:
        parse_prior("bounded", {"pdf": "uniform", **bounds})
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize("positive", [False, True])
def test_uniform_aligns_lazy_bound_pairs_once_without_mutating_inputs(positive):
    import dask.array as da
    from dask import delayed
    from openghg_inversions.models.coords import registered_model

    executions = []

    @delayed
    def bounds():
        executions.append(1)
        return np.array([[0.6, 0.1], [0.3, 1.2]])

    payload = da.from_delayed(bounds(), shape=(2, 2), dtype=float)
    lower = xr.DataArray(payload[0], dims="region", coords={"region": ["b", "a"]})
    upper = xr.DataArray(payload[1], dims="region", coords={"region": ["a", "b"]})
    prior = {"pdf": "uniform", "lower": lower, "upper": upper}
    with registered_model(coords={"region": ["a", "b"]}) as model:
        parse_prior("bounded", prior, positive=positive, dims="region")
    assert executions == [1]
    np.testing.assert_array_equal(model["bounded_prior_lower"].get_value(), [0.1, 0.6])
    assert prior["lower"] is lower and isinstance(lower.data, da.Array)
    assert lower.region.values.tolist() == ["b", "a"]


def test_uniform_preserves_symbolic_bound_dependencies():
    with pm.Model() as model:
        lower = pm.Normal("lower", initval=-1.0)
        parse_prior("bounded", {"pdf": "uniform", "lower": lower, "upper": lower + 2.0})
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize("dims,lower,upper,active", [
    (None, [0.0, 2.0], [1.0, 3.0], None),
    ("region", [0.0], [1.0, 3.0], None),
    ("region", [0.0, 4.0], [1.0, 3.0], np.array([0])),
    ("region", xr.DataArray(0.0), [1.0, 3.0], np.array([0])),
])
def test_uniform_bound_validation_matches_positional_tensor_broadcasting(dims, lower, upper, active):
    with pm.Model(coords={"region": ["a", "b"]}) as model:
        parse_prior("bounded", {"pdf": "uniform", "lower": lower, "upper": upper},
                    dims=dims, shape=1 if active is not None else 2, active_indices=active)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize("labelled", [False, True])
def test_positive_uniform_accepts_sparse_bounds_without_mutating_inputs(labelled):
    import sparse

    lower = sparse.COO.from_numpy(np.array([0.0, 0.2]))
    upper = sparse.COO.from_numpy(np.array([1.0, 1.2]))
    if labelled:
        lower = xr.DataArray(lower, dims="region", coords={"region": ["a", "b"]})
        upper = xr.DataArray(upper, dims="region", coords={"region": ["a", "b"]})
    with pm.Model(coords={"region": ["a", "b"]}) as model:
        parse_prior("bounded", {"pdf": "uniform", "lower": lower, "upper": upper},
                    positive=True, dims="region")
    np.testing.assert_array_equal(model["bounded_prior_lower"].get_value(), [0.0, 0.2])
    assert isinstance(lower.data if labelled else lower, sparse.COO)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_positive_uniform_rejects_negative_array_bounds():
    with pm.Model(coords={"region": ["a", "b"]}), pytest.raises(ValueError, match="lower bound >= 0"):
        parse_prior("bounded", {"pdf": "uniform", "lower": [0.0, -0.2], "upper": 1.0},
                    positive=True, dims="region")


def test_positive_uniform_rejects_symbolic_lower_bounds_without_evaluating_them():
    with pm.Model(), pytest.raises(ValueError, match="numeric lower bound"):
        lower = pm.Normal("lower", initval=-1.0)
        parse_prior("bounded", {"pdf": "uniform", "lower": lower, "upper": lower + 2.0}, positive=True)
