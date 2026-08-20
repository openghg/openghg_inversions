from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import (
    CoordRegistry,
    add_correlated_lognormal_state,
    add_correlated_lognormal_state_with_activity,
    attach_coord_registry,
    detect_zero_sensitivity,
    get_coord_registry,
    registered_model,
    resolve_state_activity,
    restore_inferencedata_coords,
    StateActivity,
)
from openghg_inversions.models.priors import lognormal_mu_sigma
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage,
    load_inferencedata,
    restore_serialisation_multiindexes,
    save_inferencedata,
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
    return CorrelatedLognormalPrior(_gathered_mean(), _arithmetic_covariance())


def test_correlated_prior_constructor_exposes_derived_arrays() -> None:
    """Direct construction exposes labelled latent moments and their factor."""
    mean = _gathered_mean()
    covariance = _arithmetic_covariance()

    prior = CorrelatedLognormalPrior(mean, covariance)

    assert prior.latent_mean.dims == ("state",)
    assert prior.latent_covariance.dims == ("state", "state_covariance")
    assert prior.latent_cholesky.dims == ("state", "state_covariance")
    cholesky = np.asarray(prior.latent_cholesky)
    np.testing.assert_allclose(
        cholesky @ cholesky.T,
        prior.latent_covariance,
    )


def test_correlated_prior_warns_before_large_covariance_validation() -> None:
    """Warn above the operational state-size threshold before covariance work."""
    state_size = 1001
    mean = xr.DataArray(
        np.ones(state_size),
        dims="state",
        coords={"state": np.arange(state_size)},
    )

    with pytest.warns(
        UserWarning,
        match=r"already-reduced dense covariance.*operational threshold of 1000",
    ):
        with pytest.raises(ValueError, match="square and match the state length"):
            CorrelatedLognormalPrior(mean, np.empty((0, 0)))


def test_correlated_prior_does_not_warn_at_large_state_threshold() -> None:
    """Do not warn when state size equals the documented operational threshold."""
    state_size = 1000
    mean = xr.DataArray(
        np.ones(state_size),
        dims="state",
        coords={"state": np.arange(state_size)},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match="square and match the state length"):
            CorrelatedLognormalPrior(mean, np.empty((0, 0)))


def test_correlated_prior_owns_inputs_and_returns_independent_arrays() -> None:
    """Keep the validated prior stable when caller-visible arrays are mutated."""
    mean = _gathered_mean()
    covariance = _arithmetic_covariance()
    prior = CorrelatedLognormalPrior(mean, covariance)

    mean.values[0] = 7.0
    covariance[0, 0] = 8.0
    exposed_mean = prior.mean
    exposed_mean.values[1] = 9.0
    dataset = prior.to_dataset()
    dataset["arithmetic_mean"].values[2] = 10.0
    dataset["latent_mean"].values[0] = 11.0

    np.testing.assert_allclose(prior.mean, 1.0)
    np.testing.assert_allclose(prior.arithmetic_covariance, _arithmetic_covariance())
    np.testing.assert_allclose(
        prior.latent_mean,
        -0.5 * np.diag(np.log1p(_arithmetic_covariance())),
    )


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
    prior = CorrelatedLognormalPrior(mean, covariance)
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
    draws = np.exp(np.asarray(prior.latent_mean) + whitened @ np.asarray(prior.latent_cholesky).T)

    np.testing.assert_allclose(draws.mean(axis=0), np.ones(3), atol=0.01)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), _arithmetic_covariance(), atol=0.01)


def test_diagonal_correlated_prior_matches_existing_lognormal_conversion() -> None:
    """The joint component reduces to existing independent marginal moments."""
    stdev = np.array([0.2, 0.3, 0.5])
    prior = CorrelatedLognormalPrior(_gathered_mean(), np.diag(stdev**2))
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
        CorrelatedLognormalPrior(mean, covariance)


@pytest.mark.parametrize("target", ["mean", "covariance"])
def test_correlated_lognormal_rejects_complex_moments(target: str) -> None:
    """Reject imaginary components instead of silently discarding them."""
    mean = _gathered_mean().astype(np.complex128)
    covariance = _arithmetic_covariance().astype(np.complex128)
    if target == "mean":
        mean.values[0] += 0.5j
    else:
        mean = _gathered_mean()
        covariance[0, 0] += 0.5j

    with pytest.raises(ValueError, match="real numeric"):
        CorrelatedLognormalPrior(mean, covariance)


def test_correlated_lognormal_rejects_nonfinite_relative_covariance() -> None:
    """Fail clearly when finite inputs overflow during moment conversion."""
    mean = xr.DataArray([1.0e-200], dims="state", coords={"state": ["tiny"]})

    with pytest.raises(ValueError, match="finite relative covariance"):
        CorrelatedLognormalPrior(mean, np.array([[1.0]]))


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
        CorrelatedLognormalPrior(
            mean,
            covariance.isel(state=[1, 0, 2]),
        )
    with pytest.raises(ValueError, match="column labels"):
        CorrelatedLognormalPrior(
            mean,
            covariance.isel(state_covariance=[1, 0, 2]),
        )


def test_correlated_lognormal_rejects_covariance_dimension_coord_collision() -> None:
    """Do not let the second matrix axis claim a MultiIndex level coordinate."""
    with pytest.raises(ValueError, match="must not collide"):
        CorrelatedLognormalPrior(
            _gathered_mean(),
            _arithmetic_covariance(),
            covariance_dim="source",
        )


def test_correlated_lognormal_canonicalizes_tolerance_level_asymmetry() -> None:
    """Store the symmetric matrix actually used by the Cholesky factorization."""
    covariance = _arithmetic_covariance()
    covariance[0, 1] += 5.0e-11
    prior = CorrelatedLognormalPrior(_gathered_mean(), covariance)

    np.testing.assert_array_equal(
        prior.arithmetic_covariance,
        np.asarray(prior.arithmetic_covariance).T,
    )


@pytest.mark.parametrize(
    "reserved_name",
    [
        "arithmetic_mean",
        "arithmetic_covariance",
        "latent_mean",
        "latent_covariance",
        "latent_cholesky",
    ],
)
def test_correlated_prior_rejects_reserved_serialization_coordinate_names(
    reserved_name: str,
) -> None:
    """Reject auxiliary names that collide with serialized state variables."""
    mean = _gathered_mean().assign_coords(
        {reserved_name: ("state", np.arange(3))},
    )

    with pytest.raises(ValueError, match="reserved correlated-state"):
        CorrelatedLognormalPrior(mean, _arithmetic_covariance())


def test_add_correlated_lognormal_state_builds_whitened_public_graph() -> None:
    """Expose a standard-normal sampler state and positive effective state."""
    prior = _prior()
    registry = CoordRegistry()
    with registered_model() as model:
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


def test_correlated_lognormal_activity_restores_fixed_gathered_state_and_covariance(
    tmp_path: Path,
) -> None:
    """Active draws retain their selected arithmetic covariance and full labels."""
    mean = _gathered_mean()
    prior = CorrelatedLognormalPrior(mean, _arithmetic_covariance())
    sensitivity = xr.DataArray(
        np.ones((2, 3)),
        dims=("nmeasure", "state"),
        coords={"nmeasure": [0, 1], **mean.coords},
    )
    activity = resolve_state_activity(
        detect_zero_sensitivity(sensitivity),
        StateActivity(
            active=xr.DataArray(
                [True, False, True],
                dims="state",
                coords={"state": mean.coords["state"]},
            ),
            fixed_value=xr.DataArray(
                [10.0, 7.5, 12.0],
                dims="state",
                coords={"state": mean.coords["state"]},
            ),
        ),
    )

    with registered_model() as model:
        result = add_correlated_lognormal_state_with_activity(activity, prior, var_name="x")
        idata = pm.sample_prior_predictive(draws=8000, random_seed=42)
    registry = get_coord_registry(model)
    assert registry is not None
    restored = restore_inferencedata_coords(idata, registry)
    draws = restored.prior["x"].stack(sample=("chain", "draw")).transpose("sample", "state")

    np.testing.assert_array_equal(draws.isel(state=1), 7.5)
    active_draws = draws.isel(state=[0, 2]).values
    expected_covariance = prior.arithmetic_covariance.values[np.ix_([0, 2], [0, 2])]
    np.testing.assert_allclose(
        np.cov(active_draws, rowvar=False),
        expected_covariance,
        rtol=0.15,
        atol=0.005,
    )
    assert restored.prior.indexes["state"].equals(mean.indexes["state"])
    assert restored.prior["source"].values.tolist() == ["ocean", "ff", "ff"]
    active_index = restored.prior.indexes["state_x_active"]
    assert isinstance(active_index, pd.MultiIndex)
    assert active_index.names == ["source_x_active", "region_in_source_x_active"]
    path = tmp_path / "correlated-gathered-trace.nc"
    save_inferencedata(restored.isel(draw=slice(2)), path)
    reloaded = load_inferencedata(path)
    assert reloaded.prior.indexes["state"].equals(restored.prior.indexes["state"])
    assert reloaded.prior.indexes["state_x_active"].equals(active_index)
    assert result.state is model["x"]


def test_correlated_lognormal_activity_rejects_reordered_prior_before_model_mutation() -> None:
    """The validated prior and activity must describe one identical state order."""
    mean = _gathered_mean()
    prior = CorrelatedLognormalPrior(
        mean.isel(state=[1, 0, 2]),
        _arithmetic_covariance()[np.ix_([1, 0, 2], [1, 0, 2])],
    )
    sensitivity = xr.DataArray(
        np.ones((2, 3)),
        dims=("nmeasure", "state"),
        coords={"nmeasure": [0, 1], **mean.coords},
    )
    activity = resolve_state_activity(detect_zero_sensitivity(sensitivity), None)

    with registered_model() as model:
        with pytest.raises(ValueError, match="labels must exactly match"):
            add_correlated_lognormal_state_with_activity(activity, prior, var_name="x")

    assert model.named_vars == {}


def test_add_correlated_lognormal_state_rejects_float32_cholesky_underflow() -> None:
    """Fail atomically rather than leaving a deterministic or partial state."""
    mean = _gathered_mean().isel(state=[0])
    prior = CorrelatedLognormalPrior(mean, np.array([[1.0e-100]]))

    with pytensor.config.change_flags(floatX="float32"):
        with registered_model() as model:
            with pytest.raises(ValueError, match="remain positive in the model float dtype"):
                add_correlated_lognormal_state(prior, var_name="x")
            assert model.named_vars == {}
            valid_prior = CorrelatedLognormalPrior(mean, np.array([[0.1]]))
            add_correlated_lognormal_state(valid_prior, var_name="x")

    assert set(model.named_vars) == {"x_latent", "x"}


@pytest.mark.parametrize("mean_value", [1.0e-100, 1.0e100])
def test_add_correlated_lognormal_state_rejects_unrepresentable_float32_mean(
    mean_value: float,
) -> None:
    """Reject obviously unusable backend scales before changing the model."""
    mean = xr.DataArray([mean_value], dims="state", coords={"state": ["scale"]})
    covariance = np.array([[(0.1 * mean_value) ** 2]])
    prior = CorrelatedLognormalPrior(mean, covariance)

    with pytensor.config.change_flags(floatX="float32"):
        with registered_model() as model:
            with pytest.raises(ValueError, match="arithmetic means.*model float dtype"):
                add_correlated_lognormal_state(prior, var_name="x")

    assert model.named_vars == {}


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


def test_correlated_prior_preserves_ordinary_state_auxiliary_coordinates(tmp_path) -> None:
    """Retain non-index scientific state metadata through graph and storage boundaries."""
    mean = xr.DataArray(
        np.ones(3),
        dims="state",
        coords={
            "state": ["ocean-2", "ff-7", "ff-3"],
            "source": ("state", ["ocean", "ff", "ff"]),
            "region_in_source": ("state", [2, 7, 3]),
            "latitude": ("state", [50.0, 51.0, 52.0]),
        },
    )
    prior = CorrelatedLognormalPrior(mean, _arithmetic_covariance())
    registry = CoordRegistry()
    with pm.Model() as model:
        attach_coord_registry(model, registry)
        add_correlated_lognormal_state(prior, var_name="x")
        idata = pm.sample_prior_predictive(draws=2, random_seed=42)

    restored_trace = restore_inferencedata_coords(idata, registry)
    np.testing.assert_array_equal(restored_trace.prior["source"], mean["source"])
    np.testing.assert_array_equal(restored_trace.prior["region_in_source"], mean["region_in_source"])
    np.testing.assert_array_equal(restored_trace.prior["latitude"], mean["latitude"])

    path = tmp_path / "ordinary-aux-coords.nc"
    prior.to_dataset().to_netcdf(path)
    loaded = CorrelatedLognormalPrior.from_dataset(xr.load_dataset(path))
    for name in ("source", "region_in_source", "latitude"):
        xr.testing.assert_identical(loaded.mean[name], mean[name])


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
