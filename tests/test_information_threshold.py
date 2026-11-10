"""Tests for marginalising low-information retained states into ``A``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.covariance_products import (
    RetainedProjection,
    project_native_covariance,
)
from openghg_inversions.coherent_reduction import (
    CoherentGaussianReduction,
    marginalise_low_information_states,
    marginalise_retained_states,
    reduce_native_gaussian,
    retained_state_information,
)
from openghg_inversions.models import (
    StateActivity,
    information_thresholds,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.native_covariance import SeparableExponentialCovariance


LATITUDE = xr.DataArray([-1.0, 0.0], dims="lat", coords={"lat": [-1.0, 0.0]}, attrs={"units": "degrees_north"})
LONGITUDE = xr.DataArray(
    [10.0, 11.0, 12.0, 13.0],
    dims="lon",
    coords={"lon": [10.0, 11.0, 12.0, 13.0]},
    attrs={"units": "degrees_east"},
)
SOURCES = ["land", "co2_ocean", "o2_ocean", "o2_ocean"]


def _grid(values: np.ndarray, **attrs: str) -> xr.DataArray:
    """Return a labelled native lat/lon field."""
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": LATITUDE, "lon": LONGITUDE}, attrs=attrs)


def _problem(*, correlated: bool) -> dict:
    """Build four one-column regions; regions 3 and 4 have tiny sensitivity.

    ``correlated=False`` gives each region its own covariance class, so the
    retained prior covariance is diagonal and regions are prior independent.
    """
    labels = _grid(np.tile([1, 2, 3, 4], (2, 1)))
    covariance = SeparableExponentialCovariance(
        LATITUDE,
        LONGITUDE,
        sigma=1.3,
        correlation_length=1.5,
        class_labels=None if correlated else labels,
    )
    regions = np.array([1, 2, 3, 4])
    prolongation = (
        (labels == xr.DataArray(regions, dims="state", coords={"state": regions}))
        .astype(float)
        .transpose("lat", "lon", "state")
        .assign_coords(source=("state", SOURCES))
        .assign_attrs(units="1")
    )
    rng = np.random.default_rng(1049)
    sensitivity_values = rng.normal(size=(6, 2, 4))
    sensitivity_values[:, :, 2:] *= 1e-3
    sensitivity = xr.DataArray(
        sensitivity_values,
        dims=("observation", "lat", "lon"),
        coords={"observation": [f"obs-{i}" for i in range(6)], "lat": LATITUDE, "lon": LONGITUDE},
        attrs={"units": "ppm"},
    )
    native_mean = _grid(np.array([[0.8, 1.1, 1.3, 0.9], [1.2, 0.7, 1.0, 1.4]]), units="1")
    reduction = reduce_native_gaussian(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_mean=native_mean,
        native_sensitivity=sensitivity,
        observation_dim="observation",
    )
    variance = xr.DataArray(
        np.full(6, 0.25), dims="observation", coords={"observation": sensitivity.observation}, attrs={"units": "ppm**2"}
    )
    return {
        "covariance": covariance,
        "prolongation": prolongation,
        "sensitivity": sensitivity,
        "native_mean": native_mean,
        "reduction": reduction,
        "variance": variance,
    }


@dataclass(frozen=True)
class _FixedRestriction:
    """Return a supplied restriction, used to re-reduce with ``Pi_K``."""

    restriction: xr.DataArray

    def projection(self, covariance, basis_prolongation, *, native_dims, state_dim) -> RetainedProjection:
        """Return the stored rows of the full restriction."""
        return RetainedProjection(restriction=self.restriction, strategy="rows-of-full-restriction")


def test_information_matches_definition() -> None:
    """I_j is the prior-SD-scaled squared operator column over sigma_o**2."""
    problem = _problem(correlated=True)
    reduction: CoherentGaussianReduction = problem["reduction"]
    information = retained_state_information(reduction, problem["variance"])

    operator = reduction.effective_observation_operator.values
    sd = np.sqrt(np.diag(reduction.retained_covariance.values))
    expected = (((operator * sd) ** 2) / 0.25).sum(axis=0)
    np.testing.assert_allclose(information.values, expected)
    assert list(information.source.values) == SOURCES
    assert information.values[2:].max() < 1e-3 < information.values[:2].min()


@pytest.mark.parametrize("policy", [None, StateActivity(), StateActivity(min_information=0.0)])
def test_none_or_zero_threshold_returns_reduction_unchanged(policy: StateActivity | None) -> None:
    """``None`` and ``min_information=0`` keep the identical reduction object and arrays."""
    problem = _problem(correlated=True)
    reduction = problem["reduction"]

    selection = marginalise_low_information_states(
        reduction, observation_variance=problem["variance"], policy=policy
    )

    assert selection.reduction is reduction
    assert not selection.marginalised.any()
    np.testing.assert_array_equal(selection.unresolved_variance_increase.values, 0.0)
    assert selection.provenance["n_marginalised"] == 0


def test_resolved_activity_is_unchanged_by_default_policy_field() -> None:
    """Model-time activity is bit-for-bit unchanged for ``min_information=None``."""
    reduction = _problem(correlated=True)["reduction"]
    prepared = prepare_linear_sensitivity(
        reduction.effective_observation_operator.rename(observation="nmeasure")
    )
    explicit = resolve_state_activity(prepared.removed, StateActivity(min_information=None))
    default = resolve_state_activity(prepared.removed, None)
    xr.testing.assert_identical(explicit.active, default.active)
    xr.testing.assert_identical(explicit.fixed_value, default.fixed_value)


@pytest.mark.parametrize("correlated", [True, False])
def test_marginalised_state_variance_moves_into_unresolved_covariance(correlated: bool) -> None:
    """Dropping states equals re-reducing with ``Pi_K`` and adds ``H_D S H_D.T`` to ``A``."""
    problem = _problem(correlated=correlated)
    reduction = problem["reduction"]
    selection = marginalise_low_information_states(
        reduction, observation_variance=problem["variance"], policy=StateActivity(min_information=0.1)
    )
    dropped = selection.marginalised.values
    np.testing.assert_array_equal(dropped, [False, False, True, True])
    marginalised = selection.reduction

    # Oracle: the coherent reduction restricted to the kept rows of Pi.
    full = project_native_covariance(
        covariance=problem["covariance"],
        basis_prolongation=problem["prolongation"],
        state_dim="state",
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
    )
    keep = ~dropped
    oracle = reduce_native_gaussian(
        covariance=problem["covariance"],
        basis_prolongation=problem["prolongation"].isel(state=keep),
        state_dim="state",
        native_mean=problem["native_mean"],
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
        strategy=_FixedRestriction(full.restriction.isel(state=keep)),
    )
    np.testing.assert_allclose(
        marginalised.unresolved_observation_covariance.values,
        oracle.unresolved_observation_covariance.values,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        marginalised.effective_observation_operator.values[:, keep],
        oracle.effective_observation_operator.values,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_array_equal(marginalised.effective_observation_operator.values[:, dropped], 0.0)
    np.testing.assert_allclose(
        marginalised.observation_intercept.values, oracle.observation_intercept.values, rtol=1e-9, atol=1e-12
    )

    # The added covariance is exactly H_D S H_D.T and is positive semidefinite.
    covariance = reduction.retained_covariance.values
    h_d = reduction.effective_observation_operator.values[:, dropped]
    schur = covariance[np.ix_(dropped, dropped)] - covariance[np.ix_(dropped, keep)] @ np.linalg.solve(
        covariance[np.ix_(keep, keep)], covariance[np.ix_(keep, dropped)]
    )
    increase = (
        marginalised.unresolved_observation_covariance.values - reduction.unresolved_observation_covariance.values
    )
    np.testing.assert_allclose(increase, h_d @ schur @ h_d.T, rtol=1e-9, atol=1e-15)
    assert np.linalg.eigvalsh(increase).min() > -1e-15
    np.testing.assert_allclose(selection.unresolved_variance_increase.values, np.diag(increase))
    assert (selection.unresolved_variance_increase.values > 0).all()

    # Prior predictive covariance of y is unchanged by exact marginalisation.
    def predictive(item: CoherentGaussianReduction, columns: np.ndarray) -> np.ndarray:
        operator = item.effective_observation_operator.values[:, columns]
        return (
            operator @ item.retained_covariance.values[np.ix_(columns, columns)] @ operator.T
            + item.unresolved_observation_covariance.values
        )

    np.testing.assert_allclose(
        predictive(marginalised, keep), predictive(reduction, np.ones(4, dtype=bool)), rtol=1e-9, atol=1e-12
    )

    # Model construction sees the marginalised states as zero-sensitivity states.
    prepared = prepare_linear_sensitivity(
        marginalised.effective_observation_operator.rename(observation="nmeasure")
    )
    np.testing.assert_array_equal(prepared.removed.values, dropped)
    provenance = selection.provenance
    assert provenance["n_marginalised"] == 2
    assert [item["state"] for item in provenance["marginalised_states"]] == [3, 4]


def test_grouped_threshold_applies_only_to_named_source_groups() -> None:
    """Thresholds keyed by source act on those groups, including separate oceans."""
    problem = _problem(correlated=True)
    reduction = problem["reduction"]

    o2_only = marginalise_low_information_states(
        reduction,
        observation_variance=problem["variance"],
        policy=StateActivity(group_coord="source", min_information={"o2_ocean": 1e6}),
    )
    np.testing.assert_array_equal(o2_only.marginalised.values, [False, False, True, True])
    np.testing.assert_array_equal(np.isnan(o2_only.threshold.values), [True, True, False, False])

    co2_only = marginalise_low_information_states(
        reduction,
        observation_variance=problem["variance"],
        policy=StateActivity(group_coord="source", min_information={"co2_ocean": 1e6}),
    )
    np.testing.assert_array_equal(co2_only.marginalised.values, [False, True, False, False])

    with pytest.raises(ValueError, match="absent"):
        information_thresholds(
            reduction.retained_mean, StateActivity(group_coord="source", min_information={"ff": 0.1})
        )
    with pytest.raises(ValueError, match="'basis_group' state coordinate"):
        information_thresholds(reduction.retained_mean, StateActivity(min_information={"o2_ocean": 0.1}))


def test_country_totals_hold_marginalised_regions_at_prior_mean() -> None:
    """Retained-state reconstructions leave marginalised regions at the prior flux."""
    problem = _problem(correlated=False)
    reduction = problem["reduction"]
    selection = marginalise_low_information_states(
        reduction, observation_variance=problem["variance"], policy=StateActivity(min_information=0.1)
    )
    dropped = selection.marginalised

    flux = _grid(np.full((2, 4), 2.0), units="mol m-2 s-1")
    flux_map = AffineFluxMap(
        native_mean=problem["native_mean"],
        flux=flux,
        prolongation=problem["prolongation"],
        state_dim="state",
    )
    # Any posterior draw for kept states; marginalised states take their
    # fixed value, the retained prior mean, as the CO2 handoff records.
    rng = np.random.default_rng(7)
    draws = reduction.retained_mean + xr.DataArray(rng.normal(size=(5, 4)), dims=("draw", "state"))
    draws = xr.where(dropped, reduction.retained_mean, draws).assign_attrs(units="1")
    reconstructed = flux_map.state_to_flux(draws, reference_state=reduction.retained_mean)

    labels = _grid(np.tile([1, 2, 3, 4], (2, 1)))
    country = labels.isin([3, 4])  # a "country" made of the marginalised regions
    prior_total = float((flux * problem["native_mean"]).where(country).sum())
    totals = reconstructed.where(country).sum(("lat", "lon"))
    np.testing.assert_allclose(totals.values, prior_total)
    kept_total = reconstructed.where(~country).sum(("lat", "lon"))
    assert np.ptp(kept_total.values) > 0


def test_state_activity_validates_threshold() -> None:
    """Thresholds must be finite and non-negative and are rejected at model time."""
    with pytest.raises(ValueError, match="non-negative"):
        StateActivity(min_information=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        StateActivity(min_information={"o2_ocean": float("nan")})
    with pytest.raises(ValueError, match="at least one"):
        StateActivity(min_information={})
    zero = xr.DataArray([False, False], dims="state", coords={"state": [1, 2]})
    with pytest.raises(ValueError, match="marginalise_low_information_states"):
        resolve_state_activity(zero, StateActivity(min_information=0.1))


def test_marginalise_rejects_misaligned_mask() -> None:
    """The marginalised mask must use the retained-state labels exactly."""
    reduction = _problem(correlated=True)["reduction"]
    mask = xr.DataArray([True, False, False, False], dims="state", coords={"state": [9, 2, 3, 4]})
    with pytest.raises(ValueError):
        marginalise_retained_states(reduction, mask)
