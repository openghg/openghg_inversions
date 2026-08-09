from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.correlated_state import (
    CorrelatedLognormalPrior,
    MarginalCorrelatedLognormalPrior,
)
from openghg_inversions.models import (
    CoordRegistry,
    add_correlated_lognormal_state,
    attach_coord_registry,
    restore_inferencedata_coords,
)
from openghg_inversions.models.priors import lognormal_mu_sigma
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
)


def _gathered_mean() -> xr.DataArray:
    """Return a non-lexically ordered two-source state coordinate."""
    index = pd.MultiIndex.from_tuples(
        [("ocean", "atlantic"), ("ff", "north"), ("ff", "south")],
        names=("source", "region_in_source"),
    )
    coords = xr.Coordinates.from_pandas_multiindex(index, "state")
    return xr.DataArray(
        [1.0, 1.0, 1.0],
        dims=("state",),
        coords=coords,
        name="mean",
        attrs={"units": "1"},
    )


def _arithmetic_covariance() -> np.ndarray:
    """Return an SPD covariance with within- and cross-source terms."""
    return np.array(
        [
            [0.16, 0.03, 0.01],
            [0.03, 0.09, 0.02],
            [0.01, 0.02, 0.25],
        ]
    )


def _prior() -> CorrelatedLognormalPrior:
    """Return the common gathered correlated prior fixture."""
    return CorrelatedLognormalPrior.from_moments(_gathered_mean(), _arithmetic_covariance())


def test_correlated_prior_requires_validated_factory_construction() -> None:
    """Prevent callers from bypassing moment and coordinate validation."""
    with pytest.raises(TypeError, match="from_moments"):
        CorrelatedLognormalPrior()  # type: ignore[call-arg]


def test_correlated_lognormal_matches_requested_arithmetic_moments() -> None:
    """Convert arithmetic moments exactly and retain cross-source covariance."""
    prior = _prior()
    expected_latent_covariance = np.log1p(_arithmetic_covariance())
    expected_latent_mean = -0.5 * np.diag(expected_latent_covariance)

    np.testing.assert_allclose(prior.latent_covariance, expected_latent_covariance)
    np.testing.assert_allclose(prior.latent_mean, expected_latent_mean)

    latent_covariance = np.asarray(prior.latent_covariance)
    latent_mean = np.asarray(prior.latent_mean)
    reconstructed_mean = np.exp(latent_mean + 0.5 * np.diag(latent_covariance))
    reconstructed_covariance = np.outer(reconstructed_mean, reconstructed_mean) * (
        np.exp(latent_covariance) - 1.0
    )
    np.testing.assert_allclose(reconstructed_mean, _gathered_mean())
    np.testing.assert_allclose(reconstructed_covariance, _arithmetic_covariance())
    assert float(reconstructed_covariance[0, 1]) > 0.0


def test_correlated_lognormal_supports_positive_nonunit_means() -> None:
    """Use the general arithmetic moment map when state means are not one."""
    mean = _gathered_mean().copy(data=[0.5, 2.0, 1.5])
    covariance = 0.25 * _arithmetic_covariance()
    prior = CorrelatedLognormalPrior.from_moments(mean, covariance)
    latent_covariance = np.asarray(prior.latent_covariance)
    latent_mean = np.asarray(prior.latent_mean)
    reconstructed_mean = np.exp(latent_mean + 0.5 * np.diag(latent_covariance))
    reconstructed_covariance = np.outer(reconstructed_mean, reconstructed_mean) * (
        np.exp(latent_covariance) - 1.0
    )

    np.testing.assert_allclose(reconstructed_mean, mean)
    np.testing.assert_allclose(reconstructed_covariance, covariance)


def test_correlated_lognormal_monte_carlo_preserves_cross_source_moments() -> None:
    """Whitened draws reproduce the requested non-diagonal arithmetic moments."""
    prior = _prior()
    rng = np.random.default_rng(9274)
    whitened = rng.standard_normal((100_000, 3))
    draws = np.exp(
        np.asarray(prior.latent_mean)
        + whitened @ np.asarray(prior.latent_cholesky).T
    )

    np.testing.assert_allclose(draws.mean(axis=0), np.ones(3), atol=0.01)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), _arithmetic_covariance(), atol=0.01)


def test_diagonal_correlated_prior_matches_existing_lognormal_conversion() -> None:
    """The joint component reduces to existing independent marginal moments."""
    stdev = np.array([0.2, 0.3, 0.5])
    prior = CorrelatedLognormalPrior.from_moments(_gathered_mean(), np.diag(stdev**2))
    expected_mu, expected_sigma = lognormal_mu_sigma(np.ones(3), stdev)

    np.testing.assert_allclose(prior.latent_mean, expected_mu)
    np.testing.assert_allclose(np.diag(prior.latent_cholesky), expected_sigma)
    np.testing.assert_allclose(
        prior.latent_covariance - np.diag(np.diag(prior.latent_covariance)),
        0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("mean_values", "covariance", "match"),
    [
        ([1.0, 0.0, 1.0], _arithmetic_covariance(), "positive"),
        ([1.0, np.nan, 1.0], _arithmetic_covariance(), "finite"),
        (
            [1.0, 1.0, 1.0],
            np.array([[0.16, 0.1, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.25]]),
            "symmetric",
        ),
        (
            [1.0, 1.0, 1.0],
            np.array([[1.0, -0.49, -0.49], [-0.49, 1.0, -0.49], [-0.49, -0.49, 1.0]]),
            "latent covariance must be positive definite",
        ),
    ],
)
def test_correlated_lognormal_rejects_invalid_moments(
    mean_values: list[float],
    covariance: np.ndarray,
    match: str,
) -> None:
    """Reject invalid arithmetic inputs and an invalid transformed covariance."""
    mean = _gathered_mean().copy(data=mean_values)
    with pytest.raises(ValueError, match=match):
        CorrelatedLognormalPrior.from_moments(mean, covariance)


def test_correlated_lognormal_rejects_reordered_row_or_column_labels() -> None:
    """Never reinterpret an equal-sized covariance positionally after reordering."""
    mean = _gathered_mean()
    covariance_dim = "state_covariance"
    column_labels = np.empty(3, dtype=object)
    column_labels[:] = mean.indexes["state"].tolist()
    covariance = xr.DataArray(
        _arithmetic_covariance(),
        dims=("state", covariance_dim),
        coords={
            **xr.Coordinates.from_pandas_multiindex(mean.indexes["state"], "state"),
            covariance_dim: column_labels,
        },
    )

    with pytest.raises(ValueError, match="row labels"):
        CorrelatedLognormalPrior.from_moments(
            mean,
            covariance.isel(state=[1, 0, 2]),
        )
    with pytest.raises(ValueError, match="column labels"):
        CorrelatedLognormalPrior.from_moments(
            mean,
            covariance.isel(state_covariance=[1, 0, 2]),
        )


def test_correlated_lognormal_rejects_covariance_dimension_coord_collision() -> None:
    """Do not let the second matrix axis claim a MultiIndex level coordinate."""
    with pytest.raises(ValueError, match="must not collide"):
        CorrelatedLognormalPrior.from_moments(
            _gathered_mean(),
            _arithmetic_covariance(),
            covariance_dim="source",
        )


def test_correlated_lognormal_canonicalizes_tolerance_level_asymmetry() -> None:
    """Store the symmetric matrix actually used by the Cholesky factorization."""
    covariance = _arithmetic_covariance()
    covariance[0, 1] += 5.0e-11
    prior = CorrelatedLognormalPrior.from_moments(_gathered_mean(), covariance)

    np.testing.assert_array_equal(
        prior.arithmetic_covariance,
        np.asarray(prior.arithmetic_covariance).T,
    )


def test_select_marginal_takes_principal_submatrix_without_fixed_state() -> None:
    """Marginalization keeps arithmetic moments and never restores a constant slot."""
    prior = _prior()
    retained = xr.DataArray(
        [True, False, True],
        dims=("state",),
        coords={"state": prior.mean.coords["state"]},
    )

    selection = prior.select_marginal(retained)

    np.testing.assert_allclose(selection.prior.mean, [1.0, 1.0])
    np.testing.assert_allclose(
        selection.prior.arithmetic_covariance,
        _arithmetic_covariance()[np.ix_([0, 2], [0, 2])],
    )
    np.testing.assert_array_equal(selection.retained, [True, False, True])
    np.testing.assert_array_equal(selection.omitted, [False, True, False])
    assert selection.prior.mean.sizes["state"] == 2
    assert not hasattr(selection, "fixed_value")


def test_add_correlated_lognormal_state_builds_whitened_public_graph() -> None:
    """Expose a standard-normal sampler state and positive effective state."""
    prior = _prior()
    registry = CoordRegistry()
    with pm.Model() as model:
        attach_coord_registry(model, registry)
        result = add_correlated_lognormal_state(prior, var_name="x")

    assert set(model.named_vars) == {"x_latent", "x"}
    assert [rv.name for rv in model.free_RVs] == ["x_latent"]
    assert result.latent is model.named_vars["x_latent"]
    assert result.state is model.named_vars["x"]
    assert result.prior is prior
    assert registry.original_coords["state"].equals(prior.mean.indexes["state"])
    assert list(registry.auxiliary_coords["source"].values) == ["ocean", "ff", "ff"]
    draw = pm.draw(result.state, random_seed=42)
    assert draw.shape == (3,)
    assert (draw > 0).all()


def test_add_correlated_lognormal_state_rejects_float32_cholesky_underflow() -> None:
    """Fail rather than silently make a validated state deterministic in PyTensor."""
    mean = _gathered_mean().isel(state=[0])
    prior = CorrelatedLognormalPrior.from_moments(mean, np.array([[1.0e-100]]))

    with pm.Model():
        with pytest.raises(ValueError, match="remain positive in the model float dtype"):
            add_correlated_lognormal_state(prior, var_name="x")


def test_correlated_state_prior_predictive_restores_gathered_coordinate() -> None:
    """Restore the exact non-lexical gathered coordinate on sampled state draws."""
    prior = _prior()
    registry = CoordRegistry()
    with pm.Model() as model:
        attach_coord_registry(model, registry)
        add_correlated_lognormal_state(prior, var_name="x")
        idata = pm.sample_prior_predictive(draws=4, random_seed=42)

    restored = restore_inferencedata_coords(idata, registry)
    assert restored.prior.indexes["state"].equals(prior.mean.indexes["state"])
    assert list(restored.prior["source"].values) == ["ocean", "ff", "ff"]


def test_correlated_prior_dataset_roundtrip_preserves_multiindex(tmp_path) -> None:
    """Round-trip both moment spaces through the forgiving MultiIndex boundary."""
    prior = _prior()
    encoded = encode_multiindexes_for_storage(prior.to_dataset())
    path = tmp_path / "correlated-prior.nc"
    encoded.to_netcdf(path)
    restored = restore_serialisation_multiindexes(xr.load_dataset(path), strict=True)
    loaded = CorrelatedLognormalPrior.from_dataset(restored)

    assert loaded.mean.indexes["state"].equals(prior.mean.indexes["state"])
    xr.testing.assert_identical(loaded.mean, prior.mean)
    xr.testing.assert_allclose(loaded.arithmetic_covariance, prior.arithmetic_covariance)
    xr.testing.assert_allclose(loaded.latent_covariance, prior.latent_covariance)


def test_marginal_prior_dataset_roundtrip_preserves_omitted_identity(tmp_path) -> None:
    """Persist marginalization semantics and full-state retained labels."""
    prior = _prior()
    retained = xr.DataArray(
        [True, False, True],
        dims=("state",),
        coords={"state": prior.mean.coords["state"]},
    )
    selection = prior.select_marginal(retained)
    encoded = encode_multiindexes_for_storage(selection.to_dataset())
    path = tmp_path / "marginal-correlated-prior.nc"
    encoded.to_netcdf(path)
    restored = restore_serialisation_multiindexes(xr.load_dataset(path), strict=True)

    loaded = MarginalCorrelatedLognormalPrior.from_dataset(restored)

    assert loaded.full_prior.mean.indexes["state"].equals(prior.mean.indexes["state"])
    np.testing.assert_array_equal(loaded.retained, [True, False, True])
    np.testing.assert_array_equal(loaded.omitted, [False, True, False])
    np.testing.assert_allclose(
        loaded.prior.arithmetic_covariance,
        _arithmetic_covariance()[np.ix_([0, 2], [0, 2])],
    )


def test_correlated_prior_dataset_rejects_inconsistent_derived_moments() -> None:
    """Do not trust serialized latent matrices that disagree with arithmetic moments."""
    dataset = _prior().to_dataset()
    dataset["latent_mean"].values[...] += 0.1

    with pytest.raises(ValueError, match="latent_mean.*inconsistent"):
        CorrelatedLognormalPrior.from_dataset(dataset)


def test_correlated_prior_dataset_rejects_derived_dimension_tampering() -> None:
    """Require semantic dimensions as well as equal positional values."""
    dataset = _prior().to_dataset()
    dataset["latent_mean"] = xr.DataArray(
        dataset["latent_mean"].values,
        dims=("unrelated_state",),
    )

    with pytest.raises(ValueError, match="latent_mean.*dimensions"):
        CorrelatedLognormalPrior.from_dataset(dataset)


def test_correlated_prior_dataset_rejects_column_identity_tampering() -> None:
    """Bind the positional matrix axis to serialized scientific state labels."""
    dataset = _prior().to_dataset()
    dataset["state_covariance_label"].values[1] = '"tampered"'

    with pytest.raises(ValueError, match="serialized column identity"):
        CorrelatedLognormalPrior.from_dataset(dataset)
