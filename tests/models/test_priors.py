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
