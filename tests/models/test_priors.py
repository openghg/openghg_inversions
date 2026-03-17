import numpy as np
import pymc as pm
import pytest

from openghg_inversions.models.priors import lognormal_mu_sigma, parse_prior


def test_lognormal_mu_sigma_matches_requested_moments() -> None:
    """Check lognormal moment conversion reproduces the requested moments."""
    mu, sigma = lognormal_mu_sigma(2.0, 0.5)
    expected_mean = np.exp(mu + 0.5 * sigma**2)
    expected_stdev = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
    assert np.isclose(expected_mean, 2.0)
    assert np.isclose(expected_stdev, 0.5)


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
