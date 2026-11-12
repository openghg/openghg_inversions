"""Low-rank joint aggregation error, error floor and KL diagnostics for CO2/O2 inputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_kl_per_observation,
    block_ou_correlation,
    dense_aggregation_covariance,
    prepare_low_rank_aggregation_error,
)
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs, build_co2_o2_model, prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_preparation import _validate_prepared_inputs
from test_rhime_co2_o2 import _independent_error
from test_rhime_co2_o2 import _two_site_week_inputs as _dense_two_site_week_inputs
from test_rhime_co2_o2_preparation import _inputs as _dense_small_inputs


def _two_site_week_inputs() -> dict[str, object]:
    """Fourteen rows per channel, two sites, equal units, using the default rank."""
    inputs = _dense_two_site_week_inputs()
    del inputs["aggregation_error_rank"]
    return inputs


def _small_inputs() -> dict[str, object]:
    """Two CO2 and three O2 rows in mixed units, using the default rank."""
    inputs = _dense_small_inputs()
    del inputs["aggregation_error_rank"]
    return inputs


def _smooth_two_site_week_inputs(
    *, cross_correlation: float = 0.6, o2_scale: float = 2.0, length: float = 4.0
) -> dict[str, object]:
    """Fourteen rows per channel with a smooth, non-diagonal joint aggregation covariance.

    Each channel block is an exponential kernel over observation index and the
    cross block is ``cross_correlation`` times the same kernel, so the joint
    matrix is ``[[1, rho], [rho, 1]] kron K`` scaled per channel, plus a small
    nugget. The O2 block is ``o2_scale`` times larger in standard deviation.
    """
    inputs = _two_site_week_inputs()
    co2 = inputs["co2_aggregation_covariance"]
    n = co2.shape[0]
    index = np.arange(n)
    kernel = np.exp(-np.abs(index[:, None] - index[None, :]) / length)
    co2_sd, o2_sd = 0.5, 0.5 * o2_scale
    inputs["co2_aggregation_covariance"] = co2.copy(data=co2_sd**2 * kernel + 1e-4 * np.eye(n))
    inputs["o2_aggregation_covariance"] = inputs["o2_aggregation_covariance"].copy(
        data=o2_sd**2 * kernel + 1e-4 * o2_scale**2 * np.eye(n)
    )
    inputs["co2_o2_aggregation_covariance"] = inputs["co2_o2_aggregation_covariance"].copy(
        data=cross_correlation * co2_sd * o2_sd * kernel
    )
    return inputs


def _rescale_o2(inputs: dict[str, object], factor: float) -> dict[str, object]:
    """Express the O2 channel in units ``factor`` times smaller (values ``factor`` times larger)."""
    inputs = dict(inputs)
    for name in ("o2_observations", "o2_prior_forward_mean", "o2_sensitivity"):
        inputs[name] = inputs[name] * factor
    inputs["o2_aggregation_covariance"] = inputs["o2_aggregation_covariance"] * factor**2
    inputs["co2_o2_aggregation_covariance"] = inputs["co2_o2_aggregation_covariance"] * factor
    return inputs


def _covariance(n: int, seed: int = 0) -> xr.DataArray:
    """Smooth, full-rank covariance with a decaying spectrum."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(n, n))
    values = (basis * np.exp(-np.arange(n) / 4.0)) @ (basis * np.exp(-np.arange(n) / 4.0)).T
    labels = np.arange(n)
    return xr.DataArray(
        values + 1e-3 * np.eye(n),
        dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": labels, "nmeasure_cov": labels},
    )


def _dense(covariance: xr.DataArray) -> AggregationError:
    return AggregationError(
        mode="dense", marginal_variance=np.diag(covariance.values).copy(), covariance=covariance
    )


def _kl_oracle(reference: np.ndarray, approximate: np.ndarray, noise: np.ndarray) -> float:
    """Textbook Gaussian KL(N(0, R + D) || N(0, M + D)) per observation."""
    total_reference = reference + noise
    total_approximate = approximate + noise
    n = reference.shape[0]
    trace = np.trace(np.linalg.solve(total_approximate, total_reference))
    logdet = np.linalg.slogdet(total_approximate)[1] - np.linalg.slogdet(total_reference)[1]
    return 0.5 * (trace - n + logdet) / n


def test_kl_is_exactly_zero_for_the_dense_reference_and_small_at_full_rank() -> None:
    covariance = _covariance(24)
    exact = aggregation_error_kl_per_observation(
        covariance, _dense(covariance), sigma=[0.0, 0.1, 1.0], independent_error_sd=np.full(24, 0.01)
    )
    np.testing.assert_array_equal(exact, 0.0)
    full = prepare_low_rank_aggregation_error(covariance, rank=24).aggregation_error
    np.testing.assert_allclose(
        aggregation_error_kl_per_observation(covariance, full, sigma=[0.1, 1.0]), 0.0, atol=1e-10
    )


def test_kl_grows_as_rank_falls_and_shrinks_as_sigma_grows() -> None:
    covariance = _covariance(30)
    sigma = [0.05, 0.3, 1.0]
    kl = np.array([
        aggregation_error_kl_per_observation(
            covariance,
            prepare_low_rank_aggregation_error(covariance, rank=rank).aggregation_error,
            sigma=sigma,
        )
        for rank in (25, 15, 8, 3, 1)
    ])
    assert np.all(kl > 0)
    assert np.all(np.diff(kl, axis=0) > 0), kl
    assert np.all(np.diff(kl, axis=1) < 0), kl


def test_kl_matches_textbook_formula_with_ou_correlation() -> None:
    covariance = _covariance(12, seed=3)
    approximation = prepare_low_rank_aggregation_error(covariance, rank=4).aggregation_error
    times = np.tile(np.arange("2021-07-01T00", "2021-07-01T06", dtype="datetime64[h]"), 2)
    groups = np.repeat([0, 1], 6)
    correlation = block_ou_correlation(times, groups, 5.0)
    assert correlation[0, 1] == pytest.approx(np.exp(-1 / 5))
    assert correlation[0, 6] == 0.0
    sigma = np.repeat([0.2, 0.5], 6)
    error = np.full(12, 0.05)
    [value] = aggregation_error_kl_per_observation(
        covariance, approximation, sigma=[sigma], independent_error_sd=error, sigma_correlation=correlation
    )
    noise = np.diag(error**2) + sigma[:, None] * correlation * sigma[None, :]
    expected = _kl_oracle(covariance.values, dense_aggregation_covariance(approximation), noise)
    assert value == pytest.approx(expected, rel=1e-9)


def test_kl_rejects_singular_totals() -> None:
    covariance = _covariance(6)
    factor = xr.DataArray(np.ones((6, 1)), dims=("nmeasure", "agg_rank"))
    approximation = AggregationError(
        mode="low_rank",
        marginal_variance=np.ones(6),
        factor=factor,
        diagonal_variance=xr.DataArray(np.zeros(6), dims="nmeasure"),
    )
    with pytest.raises(ValueError, match="not positive definite"):
        aggregation_error_kl_per_observation(covariance, approximation, sigma=[0.0])


def _logp(prepared, *, tau_hours=None) -> tuple[float, float]:
    options = {} if tau_hours is None else {"tau_hours": tau_hours, "fixed_site_amplitudes": 0.4}
    model = build_co2_o2_model(
        observations=prepared.observations,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        co2_sensitivity=prepared.co2_sensitivity,
        o2_sensitivity=prepared.o2_sensitivity,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=_independent_error(prepared, value=0.3),
        **options,
    )
    logp = model.compile_logp()
    point = model.initial_point()
    shifted = {name: value + 0.05 for name, value in point.items()}
    return float(logp(point)), float(logp(shifted))


@pytest.mark.parametrize("tau_hours", [None, 30.0])
@pytest.mark.parametrize("factorization", ["joint", "own_species"])
def test_full_rank_low_rank_likelihood_equals_dense(tau_hours, factorization) -> None:
    inputs = _smooth_two_site_week_inputs()
    if factorization == "own_species":
        # Own-species factors are exact only when the cross block is zero.
        inputs["co2_o2_aggregation_covariance"] = inputs["co2_o2_aggregation_covariance"] * 0.0
    dense = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=None)
    low_rank = prepare_co2_o2_inputs(**inputs, aggregation_error_factorization=factorization)
    assert dense.aggregation_error.mode == "dense"
    assert low_rank.aggregation_error.mode == "low_rank"
    assert low_rank.provenance["aggregation_error"]["rank"] == 512
    assert low_rank.provenance["aggregation_error"]["actual_rank"] == 28
    np.testing.assert_allclose(
        dense_aggregation_covariance(low_rank.aggregation_error), dense.aggregation_error.covariance.values, atol=1e-12
    )
    # The fixed-independent graph runs in floatX (float32 here); fixed OU evaluates in float64.
    rtol = 1e-5 if tau_hours is None else 1e-9
    np.testing.assert_allclose(_logp(low_rank, tau_hours=tau_hours), _logp(dense, tau_hours=tau_hours), rtol=rtol)


def test_truncated_rank_changes_likelihood_and_records_kl() -> None:
    inputs = _two_site_week_inputs()
    dense = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=None)
    sigma = {"co2": [0.05, 0.5], "o2": [0.05, 0.5]}
    joint = prepare_co2_o2_inputs(
        **inputs,
        aggregation_error_rank=2,
        independent_error_floor={"co2": 0.1, "o2": 0.1},
        aggregation_error_kl_sigma=sigma,
        aggregation_error_kl_tau_hours=5.0,
    )
    own = prepare_co2_o2_inputs(
        **inputs,
        aggregation_error_rank=28,
        aggregation_error_factorization="own_species",
        independent_error_floor={"co2": 0.1, "o2": 0.1},
        aggregation_error_kl_sigma=sigma,
    )
    assert _logp(joint) != pytest.approx(_logp(dense))
    record = joint.provenance["aggregation_error"]
    # The rank is per species: the joint factor keeps 2 * rank modes.
    assert record["factorization"] == "joint" and record["rank_budget"] == 4 and record["actual_rank"] == 4
    assert record["relative_frobenius_reconstruction_error"] > 0
    kl = joint.provenance["aggregation_error_kl"]
    assert kl["kernel"] == "fixed OU by species and site" and kl["tau_hours"] == 5.0
    assert kl["independent_error"] == "prepared independent_error_sd" and kl["bound"] == "exact for this D"
    assert kl["sigma_rule"] == "supplied"
    assert kl["kl_per_observation"][0] > kl["kl_per_observation"][1] > 0
    expected = aggregation_error_kl_per_observation(
        dense.aggregation_error.covariance,
        joint.aggregation_error,
        sigma=[0.05, 0.5],
        independent_error_sd=joint.independent_error_sd.values,
        sigma_correlation=block_ou_correlation(
            joint.observations.time.values,
            np.char.add(joint.observations.species.values.astype(str), joint.observations.site.values.astype(str)),
            5.0,
        ),
    )
    np.testing.assert_allclose(kl["kl_per_observation"], expected)
    # Own-species factors at full block rank lose only the dropped cross-covariance.
    own_record = own.provenance["aggregation_error"]
    assert own_record["actual_rank"] == 28
    assert own_record["dropped_cross_covariance_relative_frobenius"] > 0
    np.testing.assert_allclose(
        own_record["relative_frobenius_reconstruction_error"],
        own_record["dropped_cross_covariance_relative_frobenius"],
        rtol=1e-9,
    )
    factor = own.aggregation_error.factor
    species = own.observations.species.values
    assert not np.any(factor.values[species == "co2"][:, 14:]) and not np.any(factor.values[species == "o2"][:, :14])
    assert own.provenance["aggregation_error_kl"]["kl_per_observation"][0] > 0


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("factorization", ["joint", "own_species"])
def test_low_rank_linked_inputs_round_trip(tmp_path: Path, suffix: str, factorization: str) -> None:
    prepared = prepare_co2_o2_inputs(
        **_small_inputs(),
        aggregation_error_rank=2,
        aggregation_error_factorization=factorization,
        independent_error_floor={"co2": 0.2, "o2": 3.0},
    )
    path = tmp_path / f"linked{suffix}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    assert restored.aggregation_error.mode == "low_rank"
    assert restored.aggregation_error.covariance is None
    for name in ("factor", "diagonal_variance"):
        xr.testing.assert_identical(getattr(restored.aggregation_error, name), getattr(prepared.aggregation_error, name))
    np.testing.assert_allclose(
        restored.aggregation_error.marginal_variance, prepared.aggregation_error.marginal_variance
    )
    assert restored.aggregation_error.factor.attrs["units"] == "observation_units"
    assert restored.provenance == prepared.provenance
    assert restored.provenance["aggregation_error"]["factorization"] == factorization
    xr.testing.assert_identical(restored.independent_error_sd, prepared.independent_error_sd)


def test_validation_rejects_low_rank_rows_with_wrong_units() -> None:
    prepared = prepare_co2_o2_inputs(**_small_inputs(), aggregation_error_rank=2)
    factor = prepared.aggregation_error.factor
    broken = replace(
        prepared,
        aggregation_error=replace(
            prepared.aggregation_error,
            factor=factor.assign_coords(observation_units=("observation", ["ppm"] * factor.sizes["observation"])),
        ),
    )
    with pytest.raises(ValueError, match="low_rank_factor observation_units must match"):
        _validate_prepared_inputs(broken)


def test_independent_error_floor_is_explicit_and_recorded() -> None:
    inputs = _small_inputs()
    none = prepare_co2_o2_inputs(**inputs)
    assert none.independent_error_sd is None
    assert none.provenance["independent_error_floor"] == {"floor": None, "source": "none; supply at run time"}

    floor_only = prepare_co2_o2_inputs(**inputs, independent_error_floor={"co2": 0.2, "o2": 3.0})
    np.testing.assert_array_equal(floor_only.independent_error_sd.values, [0.2, 0.2, 3.0, 3.0, 3.0])
    xr.testing.assert_identical(
        floor_only.independent_error_sd.observation_units, floor_only.observations.observation_units
    )
    assert floor_only.provenance["independent_error_floor"] == {
        "floor": {"co2": 0.2, "o2": 3.0},
        "rule": "per-row max(independent_error_sd, floor)",
        "source": "floor only",
    }

    supplied = floor_only.independent_error_sd.copy(data=[0.1, 0.5, 1.0, 4.0, 2.0])
    floored = prepare_co2_o2_inputs(
        **inputs, independent_error_sd=supplied, independent_error_floor={"co2": 0.2, "o2": 3.0}
    )
    np.testing.assert_array_equal(floored.independent_error_sd.values, [0.2, 0.5, 3.0, 4.0, 3.0])
    assert floored.provenance["independent_error_floor"]["source"] == "supplied independent_error_sd with floor"
    unfloored = prepare_co2_o2_inputs(**inputs, independent_error_sd=supplied)
    assert unfloored.independent_error_sd is supplied
    assert unfloored.provenance["independent_error_floor"] == {
        "floor": None,
        "source": "supplied independent_error_sd",
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"independent_error_floor": {"co2": 0.1}}, "exactly 'co2' and 'o2'"),
        ({"independent_error_floor": {"co2": 0.0, "o2": 1.0}}, "finite and positive"),
        ({"independent_error_floor": {"co2": True, "o2": 1.0}}, "real number"),
        ({"aggregation_error_rank": 0}, "positive integer"),
        ({"aggregation_error_factorization": "per_site"}, "'joint' or 'own_species'"),
        ({"aggregation_error_kl_sigma": None, "aggregation_error_kl_tau_hours": 5.0}, "requires aggregation_error_kl_sigma"),
        ({"aggregation_error_rank": None, "aggregation_error_kl_sigma": {"co2": [1.0], "o2": [1.0]}}, "dense A is exact"),
        ({"aggregation_error_kl_sigma": {"co2": [1.0], "o2": []}}, "equal-length"),
        ({"aggregation_error_kl_sigma": {"co2": [1.0], "o2": [1.0]}, "aggregation_error_kl_tau_hours": 5.0}, "'site'"),
        ({"provenance": {"aggregation_error": "hand-built rank 256"}}, "recorded by preparation"),
    ],
)
def test_rejects_invalid_low_rank_options(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_co2_o2_inputs(**{**_small_inputs(), **options})


@pytest.mark.parametrize("rank", [3, 512])
def test_own_species_factors_preserve_the_smooth_diagonal_and_drop_only_cross(rank) -> None:
    inputs = _smooth_two_site_week_inputs()
    dense = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=None).aggregation_error.covariance.values
    own = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=rank, aggregation_error_factorization="own_species")
    record = own.provenance["aggregation_error"]
    assert record["rank_budget"] == {"co2": rank, "o2": rank}
    assert record["actual_rank_by_block"] == {"co2": min(rank, 14), "o2": min(rank, 14)}
    approximate = dense_aggregation_covariance(own.aggregation_error)
    np.testing.assert_allclose(np.diag(approximate), np.diag(dense), rtol=1e-12)
    np.testing.assert_allclose(own.aggregation_error.marginal_variance, np.diag(dense), rtol=1e-12)
    assert record["diagonal_preservation_error"] < 1e-12
    species = own.observations.species.values
    cross = np.ix_(species == "co2", species == "o2")
    assert np.abs(dense[cross]).max() > 0.1
    np.testing.assert_array_equal(approximate[cross], 0.0)
    # Relative errors are in channel-scaled units: divide each channel by its scale.
    scale = np.where(species == "co2", record["channel_scale"]["co2"], record["channel_scale"]["o2"])
    scaled = dense / np.outer(scale, scale)
    scaled_error = (dense - approximate) / np.outer(scale, scale)
    np.testing.assert_allclose(
        record["relative_frobenius_reconstruction_error"],
        np.linalg.norm(scaled_error) / np.linalg.norm(scaled),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        record["dropped_cross_covariance_relative_frobenius"],
        np.sqrt(2.0) * np.linalg.norm(scaled[cross]) / np.linalg.norm(scaled),
        rtol=1e-9,
    )
    if rank == 512:
        np.testing.assert_allclose(
            record["relative_frobenius_reconstruction_error"],
            record["dropped_cross_covariance_relative_frobenius"],
            rtol=1e-9,
        )


def test_joint_rank_is_per_species_and_preserves_the_smooth_diagonal() -> None:
    inputs = _smooth_two_site_week_inputs()
    dense = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=None).aggregation_error.covariance.values
    joint = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=3)
    record = joint.provenance["aggregation_error"]
    assert record["rank"] == 3 and record["rank_meaning"] == "per species"
    assert record["rank_budget"] == 6 and record["actual_rank"] == 6
    assert record["channel_scale"]["o2"] == pytest.approx(2.0 * record["channel_scale"]["co2"])
    approximate = dense_aggregation_covariance(joint.aggregation_error)
    np.testing.assert_allclose(np.diag(approximate), np.diag(dense), rtol=1e-12)
    species = joint.observations.species.values
    # The joint factor keeps cross-covariance that own-species factors drop.
    assert np.abs(approximate[np.ix_(species == "co2", species == "o2")]).max() > 0.1
    scale = np.where(species == "co2", record["channel_scale"]["co2"], record["channel_scale"]["o2"])
    np.testing.assert_allclose(
        record["relative_frobenius_reconstruction_error"],
        np.linalg.norm((dense - approximate) / np.outer(scale, scale)) / np.linalg.norm(dense / np.outer(scale, scale)),
        rtol=1e-9,
    )


@pytest.mark.parametrize("factorization", ["joint", "own_species"])
def test_factorization_and_default_kl_do_not_depend_on_o2_units(factorization) -> None:
    inputs = _smooth_two_site_week_inputs(o2_scale=1.0)
    options = {"aggregation_error_rank": 2, "aggregation_error_factorization": factorization}
    base = prepare_co2_o2_inputs(**inputs, **options)
    rescaled = prepare_co2_o2_inputs(**_rescale_o2(inputs, 5.0), **options)
    species = base.observations.species.values
    row_factor = np.where(species == "o2", 5.0, 1.0)
    np.testing.assert_allclose(
        dense_aggregation_covariance(rescaled.aggregation_error),
        dense_aggregation_covariance(base.aggregation_error) * np.outer(row_factor, row_factor),
        rtol=1e-9,
        atol=1e-12,
    )
    base_kl, rescaled_kl = (prepared.provenance["aggregation_error_kl"] for prepared in (base, rescaled))
    np.testing.assert_allclose(rescaled_kl["sigma"]["o2"], 5.0 * np.asarray(base_kl["sigma"]["o2"]), rtol=1e-12)
    np.testing.assert_allclose(rescaled_kl["kl_per_observation"], base_kl["kl_per_observation"], rtol=1e-8)


def test_default_kl_is_recorded_on_the_1053_grid_as_an_upper_bound() -> None:
    inputs = _smooth_two_site_week_inputs()
    dense = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=None)
    assert "aggregation_error_kl" not in dense.provenance
    low_rank = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=2)
    kl = low_rank.provenance["aggregation_error_kl"]
    assert kl["sigma"]["co2"] == [0.1, 0.25, 0.5, 1.0]
    np.testing.assert_allclose(kl["sigma"]["o2"], [0.2, 0.5, 1.0, 2.0], rtol=1e-6)
    assert kl["sigma_rule"].startswith("default")
    assert kl["kernel"] == "diagonal"
    assert kl["independent_error"].startswith("none") and kl["bound"].startswith("upper bound")
    species = low_rank.observations.species.values
    expected = aggregation_error_kl_per_observation(
        dense.aggregation_error.covariance,
        low_rank.aggregation_error,
        sigma=[np.where(species == "co2", co2, o2) for co2, o2 in zip(kl["sigma"]["co2"], kl["sigma"]["o2"])],
    )
    np.testing.assert_allclose(kl["kl_per_observation"], expected)
    assert np.all(np.diff(kl["kl_per_observation"]) < 0)
    # Adding run-time independent error to both sides can only reduce the KL.
    floored = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=2, independent_error_floor={"co2": 0.1, "o2": 0.2})
    floored_kl = floored.provenance["aggregation_error_kl"]
    assert floored_kl["bound"] == "exact for this D"
    assert np.all(np.asarray(floored_kl["kl_per_observation"]) < np.asarray(kl["kl_per_observation"]))


def test_missing_kl_record_warns() -> None:
    inputs = _smooth_two_site_week_inputs()
    with pytest.warns(UserWarning, match="No aggregation-error KL diagnostic"):
        disabled = prepare_co2_o2_inputs(**inputs, aggregation_error_rank=2, aggregation_error_kl_sigma=None)
    assert disabled.provenance["aggregation_error_kl"] == {"recorded": False, "reason": "disabled by caller"}
    with pytest.warns(UserWarning, match="co2_units is 'ppb'"):
        other_units = prepare_co2_o2_inputs(**{**inputs, "co2_units": "ppb", "o2_units": "ppb"}, aggregation_error_rank=2)
    assert other_units.provenance["aggregation_error_kl"]["recorded"] is False
