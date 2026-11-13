"""Tests for the pre-sampling prior diagnostics."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.covariance_products import RetainedProjection
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.presample_diagnostics import (
    PresampleThresholds,
    assess_country_budget,
    assess_noise_calibration,
    assess_state_signal,
    country_prior_budget,
    noise_calibration,
    prior_predictive_component_sd,
    state_signal,
    state_values_on_grid,
)


def _calibration_inputs(mismatch_scale: dict[str, float], n: int = 4000, draws: int = 400):
    """Return observations whose de-meaned mismatch is ``scale`` times the unit noise per site."""
    rng = np.random.default_rng(1)
    sites = np.repeat(list(mismatch_scale), n)
    scale = np.repeat(list(mismatch_scale.values()), n)
    index = {"index": np.arange(sites.size)}
    offset = np.where(sites == sites[0], 5.0, -3.0)
    observed = xr.DataArray(offset + scale * rng.standard_normal(sites.size), dims="index", coords=index)
    predicted = xr.zeros_like(observed)
    noise = xr.DataArray(
        rng.standard_normal((1, draws, sites.size)), dims=("chain", "draw", "index"), coords=index
    )
    groups = xr.DataArray(sites, dims="index", coords=index)
    return observed, predicted, noise, groups


def test_calibrated_noise_passes_and_offsets_are_removed() -> None:
    """Unit mismatch against unit noise gives ratio and coverage near nominal."""
    observed, predicted, noise, groups = _calibration_inputs({"MHD": 1.0, "TAC": 1.0})
    result = noise_calibration(observed, predicted, noise, observation_dim="index", groups=groups)

    assert list(result.group.values) == ["MHD", "TAC", "network"]
    np.testing.assert_allclose(result.mismatch_mean.sel(group=["MHD", "TAC"]), [5.0, -3.0], atol=0.1)
    np.testing.assert_allclose(result.noise_over_mismatch, 1.0, atol=0.05)
    np.testing.assert_allclose(result.coverage, 0.95, atol=0.02)
    assert (assess_noise_calibration(result).status == "pass").all()


def test_narrow_and_wide_noise_are_graded() -> None:
    """Three-times mismatch fails, half mismatch warns as too wide, and the network inherits failures."""
    observed, predicted, noise, groups = _calibration_inputs({"MHD": 3.0, "TAC": 0.5, "JFJ": 1.0})
    result = assess_noise_calibration(
        noise_calibration(observed, predicted, noise, observation_dim="index", groups=groups),
        PresampleThresholds(ratio_wide=1.5),
    )

    status = dict(zip(result.group.values, result.status.values, strict=True))
    assert status["MHD"] == "fail"
    assert status["TAC"] == "warn"
    assert status["JFJ"] == "pass"
    assert status["network"] in {"warn", "fail"}


def test_network_warns_when_enough_groups_fail() -> None:
    """A passing pooled statistic is raised to warn when a quarter of the groups fail."""
    calibration = xr.Dataset(
        {
            "noise_over_mismatch": ("group", [0.4, 1.0, 1.0, 1.0, 1.0]),
            "coverage": ("group", [0.6, 0.95, 0.95, 0.95, 0.95]),
        },
        coords={"group": ["A", "B", "C", "D", "network"]},
        attrs={"network_label": "network"},
    )

    assert assess_noise_calibration(calibration).status.sel(group="network") == "warn"


def test_calibration_rejects_misaligned_rows() -> None:
    """Observation rows must match exactly; xarray's default inner join is not used."""
    observed, predicted, noise, groups = _calibration_inputs({"MHD": 1.0}, n=10, draws=5)

    with pytest.raises(ValueError):
        noise_calibration(
            observed, predicted.isel(index=slice(1, None)), noise, observation_dim="index", groups=groups
        )


def test_component_sd_pools_rms_and_shares_sum_to_one() -> None:
    """Component SDs are RMS over rows of the per-row SD."""
    rng = np.random.default_rng(2)
    shape = (1, 20000, 4)
    scale = np.array([1.0, 1.0, 3.0, 3.0])
    coords = {"index": np.arange(4)}
    flux = xr.DataArray(rng.standard_normal(shape) * scale, dims=("chain", "draw", "index"), coords=coords)
    noise = xr.DataArray(rng.standard_normal(shape), dims=("chain", "draw", "index"), coords=coords)
    groups = xr.DataArray(["A", "A", "B", "B"], dims="index", coords=coords)

    result = prior_predictive_component_sd(
        {"flux": flux, "noise": noise}, observation_dim="index", groups=groups
    )

    np.testing.assert_allclose(result.sd.sel(component="flux"), [1.0, 3.0, np.sqrt(5.0)], rtol=0.02)
    np.testing.assert_allclose(result.variance_share.sum("component"), 1.0)


def test_state_signal_matches_hand_calculation_by_group() -> None:
    """Signal is the RMS over rows of |H s| within each observation group."""
    sensitivity = xr.DataArray(
        [[1.0, 0.0], [3.0, 0.1], [0.0, 2.0]],
        dims=("index", "state"),
        coords={"index": [0, 1, 2], "state": ["a", "b"]},
    )
    state_sd = xr.DataArray([2.0, 0.5], dims="state", coords={"state": ["a", "b"]})
    species = xr.DataArray(["co2", "co2", "o2"], dims="index", coords={"index": [0, 1, 2]})

    pooled = state_signal(sensitivity, state_sd, observation_dim="index", state_dim="state")
    grouped = state_signal(
        sensitivity, state_sd, observation_dim="index", state_dim="state", observation_groups=species
    )

    np.testing.assert_allclose(pooled.signal_rms, [np.sqrt((4 + 36) / 3), np.sqrt((0.0025 + 1) / 3)])
    np.testing.assert_allclose(grouped.signal_rms.sel(group="co2"), [np.sqrt(20.0), np.sqrt(0.0025 / 2)])
    np.testing.assert_allclose(grouped.signal_max.sel(group="o2"), [0.0, 1.0])


def test_state_signal_classes_and_grades() -> None:
    """Weak states with a wide prior are counted and graded; dead states are not."""
    states = {"state": list("abcdef")}
    signal = xr.DataArray([0.001, 0.01, 0.02, 0.03, 0.5, 1.0], dims="state", coords=states)
    scale = xr.DataArray([5.0, 3.0, 2.5, 0.5, 4.0, 1.0], dims="state", coords=states)

    result = assess_state_signal(signal, scale, negligible=0.04, dead=0.004, state_dim="state")

    assert list(result.signal_class.values) == ["dead", "weak", "weak", "weak", "active", "active"]
    assert list(result.weak_heavy_tail.values) == [False, True, True, False, False, False]
    assert int(result.n_weak_heavy_tail) == 2
    assert result.weak_heavy_tail_status.item() == "warn"
    assert float(result.negligible_share) == pytest.approx(4 / 6)
    assert result.negligible_share_status.item() == "fail"
    with pytest.raises(ValueError, match="dead"):
        assess_state_signal(signal, scale, negligible=0.004, dead=0.04, state_dim="state")


def _budget_problem():
    """Return a 3 x 4 grid with two bucket states, its dense covariance and country weights."""
    lat = xr.DataArray([0.0, 1.0, 2.0], dims="lat", coords={"lat": [0.0, 1.0, 2.0]})
    lon = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims="lon", coords={"lon": [0.0, 1.0, 2.0, 3.0]})
    covariance = SeparableExponentialCovariance(lat, lon, sigma=0.8, correlation_length=1.5)
    labels = xr.DataArray([[0, 0, 1, 1]] * 3, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    prolongation = (labels == xr.DataArray([0, 1], dims="state", coords={"state": [0, 1]})).astype(float)
    flux = xr.DataArray(np.linspace(1.0, 2.0, 12).reshape(3, 4), dims=("lat", "lon"), coords=labels.coords)
    # Country "X" sits inside state 0; "Y" straddles both states; "XY" is their union.
    x = xr.zeros_like(flux).where(labels != 0, 1.0)
    y = xr.zeros_like(flux)
    y[1, 1:3] = 1.0
    weights = xr.concat([x, y, x + y], dim=xr.DataArray(["X", "Y", "XY"], dims="country", name="country"))
    lat_factor = np.exp(-np.abs(lat.values[:, None] - lat.values[None, :]) / 1.5)
    lon_factor = np.exp(-np.abs(lon.values[:, None] - lon.values[None, :]) / 1.5)
    dense = 0.8**2 * np.kron(lat_factor, lon_factor)
    return covariance, prolongation, flux, weights, dense


def test_country_budget_matches_dense_native_variance() -> None:
    """Resolved plus unresolved variance equals the dense native variance of each country total."""
    covariance, prolongation, flux, weights, dense = _budget_problem()

    budget = country_prior_budget(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_flux=flux,
        country_weights=weights,
    )

    functional = (weights * flux).transpose("country", "lat", "lon").values.reshape(3, -1)
    np.testing.assert_allclose(
        budget.variance_native, np.einsum("cg,gh,ch->c", functional, dense, functional)
    )
    np.testing.assert_allclose(budget.prior_total, functional.sum(axis=1))
    assert (budget.variance_unresolved >= -1e-12).all()
    assert ((budget.resolved_fraction > 0) & (budget.resolved_fraction <= 1 + 1e-12)).all()
    assert list(assess_country_budget(budget).status.dims) == ["country"]


class _FluxWeightedRestriction:
    """Restriction whose states are flux-weighted means of their native cells."""

    def __init__(self, flux: xr.DataArray) -> None:
        """Store the native flux used as weights."""
        self.flux = flux

    def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
        """Return ``Pi[j, g] = U[g, j] F[g] / sum_g U[g, j] F[g]``."""
        weighted = basis_prolongation * self.flux
        restriction = (weighted / weighted.sum(native_dims)).transpose(state_dim, *native_dims)
        return RetainedProjection(restriction, "test_flux_weighted")


def test_country_budget_resolves_flux_weighted_state_totals() -> None:
    """A country made of whole states is fully resolved under flux-weighted restriction."""
    covariance, prolongation, flux, weights, _ = _budget_problem()
    budget = country_prior_budget(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim="state",
        native_flux=flux,
        country_weights=weights.sel(country=["X"]),
        strategy=_FluxWeightedRestriction(flux),
    )

    np.testing.assert_allclose(budget.resolved_fraction, 1.0, rtol=1e-8)


def test_state_values_on_grid_is_piecewise_constant() -> None:
    """Each cell takes its state's value and uncovered cells are NaN."""
    labels = xr.DataArray([[0, 1, 2]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0]})
    prolongation = (labels == xr.DataArray([0, 1], dims="state", coords={"state": [0, 1]})).astype(float)
    values = xr.DataArray([3.0, 4.0], dims="state", coords={"state": [0, 1]})

    grid = state_values_on_grid(values, prolongation, "state")

    np.testing.assert_array_equal(grid.values, [[3.0, 4.0, np.nan]])
