"""Tests for the functional coverage helper, using small synthetic oracles."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.functional_coverage import functional_coverage, wilson_interval


def _draws(values, dim="country", labels=None):
    """Wrap draws of shape (chain, draw, functional) in a DataArray."""
    values = np.asarray(values, dtype=float)
    labels = labels if labels is not None else list(range(values.shape[-1]))
    return xr.DataArray(values, dims=("chain", "draw", dim), coords={dim: labels})


def _truth(values, dim="country", labels=None):
    values = np.asarray(values, dtype=float)
    labels = labels if labels is not None else list(range(values.shape[-1]))
    return xr.DataArray(values, dims=(dim,), coords={dim: labels})


def test_gaussian_toy_is_calibrated_and_overconfidence_is_caught():
    rng = np.random.default_rng(20261112)
    n_functionals, n_chains, n_draws, noise_sd = 2000, 2, 200, 0.7
    theta = rng.normal(size=n_functionals)
    estimate = theta + noise_sd * rng.normal(size=n_functionals)  # flat-prior posterior mean
    posterior = estimate + noise_sd * rng.normal(size=(n_chains, n_draws, n_functionals))
    truth = _truth(theta)

    result = functional_coverage(_draws(posterior), truth, intervals=(0.68, 0.95))

    coverage = result["coverage"].sel(subset="all")
    lower = result["coverage_lower"].sel(subset="all")
    upper = result["coverage_upper"].sel(subset="all")
    for interval in (0.68, 0.95):
        assert lower.sel(interval=interval) <= interval <= upper.sel(interval=interval)
        assert abs(coverage.sel(interval=interval) - interval) < 0.03
    assert result["rms_z"].sel(subset="unflagged") == pytest.approx(1.0, abs=0.06)
    assert result["n"].sel(subset="all") == n_functionals
    # PIT is uniform for a calibrated posterior.
    assert result["pit"].mean() == pytest.approx(0.5, abs=0.02)

    overconfident = functional_coverage(_draws(estimate + 0.4 * (posterior - estimate)), truth)
    assert overconfident["coverage_upper"].sel(subset="all", interval=0.95) < 0.95
    assert overconfident["rms_z"].sel(subset="all") > 1.5


def test_grouped_total_uses_joint_draws_not_summed_moments():
    rng = np.random.default_rng(3)
    shared = rng.normal(size=(2, 500))
    # Two members that trade off against each other: their sum is well constrained.
    member_a = 5.0 + shared
    member_b = 3.0 - shared + 0.05 * rng.normal(size=(2, 500))
    draws = _draws(np.stack([member_a, member_b], axis=-1), labels=["A", "B"])
    truth = _truth([5.0, 3.0], labels=["A", "B"])

    result = functional_coverage(draws, truth, groups={"AB": ["A", "B"]})

    joint_sd = (member_a + member_b).std(ddof=1)
    naive_sd = np.sqrt(member_a.var(ddof=1) + member_b.var(ddof=1))
    assert result["sd"].sel(country="AB") == pytest.approx(joint_sd)
    assert joint_sd < 0.1 * naive_sd
    assert result["mean"].sel(country="AB") == pytest.approx((member_a + member_b).mean())
    assert result["truth"].sel(country="AB") == 8.0
    assert list(result["country"].values) == ["AB"]
    # Treating the members as independent would have given a much smaller z.
    naive_z = (result["mean"].sel(country="AB") - 8.0) / naive_sd
    assert abs(result["z"].sel(country="AB")) > 5 * abs(naive_z)


def test_sum_over_sectors_then_groups_and_overlapping_groups():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(2, 100, 3, 2))
    draws = xr.DataArray(
        values,
        dims=("chain", "draw", "country", "sector"),
        coords={"country": ["FRA", "DEU", "GBR"], "sector": ["energy", "waste"]},
    )
    truth = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dims=("country", "sector"),
        coords=draws.coords["country"].coords | draws.coords["sector"].coords,
    )

    result = functional_coverage(
        draws,
        truth,
        groups={"FRA": ["FRA"], "EU": ["FRA", "DEU"], "ALL": ["FRA", "DEU", "GBR"]},
        sum_dims=("sector",),
    )

    expected = values.sum(-1)
    assert result["mean"].sel(country="EU") == pytest.approx(expected[..., :2].sum(-1).mean())
    assert result["sd"].sel(country="ALL") == pytest.approx(expected.sum(-1).std(ddof=1))
    assert result["truth"].sel(country=["FRA", "EU", "ALL"]).values.tolist() == [3.0, 10.0, 21.0]
    assert draws.dims == ("chain", "draw", "country", "sector")  # input untouched


def test_tiny_sd_is_flagged_and_excluded_from_rms_z():
    rng = np.random.default_rng(5)
    labels = ["FRA", "DEU", "LUX"]
    sds = np.array([1.0, 1.0, 1e-5])
    draws = _draws(rng.normal(size=(2, 400, 3)) * sds, labels=labels)
    prior = _draws(rng.normal(size=(2, 400, 3)) * np.array([1.0, 1.0, 0.5]), labels=labels)
    truth = _truth([0.0, 0.0, 0.2], labels=labels)  # LUX is wrong by 0.2, enormous in z

    absolute = functional_coverage(draws, truth, min_sd=1e-3)
    relative = functional_coverage(draws, truth, min_sd_fraction_of_prior=0.01, prior_draws=prior)

    for result in (absolute, relative):
        assert result["sd_below_floor"].values.tolist() == [False, False, True]
        assert result["excluded"].values.tolist() == [False, False, True]
        assert abs(result["z"].sel(country="LUX")) > 1e3
        assert result["rms_z"].sel(subset="unflagged") < 1.5
        assert result["rms_z"].sel(subset="all") > 1e3
        assert result["n"].sel(subset="unflagged") == 2
        assert result["coverage"].sel(subset="all", interval=0.95) == pytest.approx(2 / 3, abs=0.2)
        assert result["coverage_upper"].sel(subset="unflagged", interval=0.95) <= 1.0
    assert relative["prior_sd"].sel(country="LUX") == pytest.approx(0.5, rel=0.1)

    unflagged = functional_coverage(draws, truth)
    assert not unflagged["sd_below_floor"].any()


@pytest.mark.parametrize(
    ("truth_value", "pit"),
    [(-100.0, 0.0), (100.0, 1.0), (0.0, 0.5), (0.5, 0.75)],
)
def test_pit_edge_cases(truth_value, pit):
    draws = _draws(np.linspace(-1.0, 1.0, 4).reshape(2, 2, 1))
    result = functional_coverage(draws, _truth([truth_value]))
    assert result["pit"].item() == pit


def test_pit_ties_and_degenerate_draws():
    constant = _draws(np.full((2, 5, 1), 2.0))
    result = functional_coverage(constant, _truth([2.0]))
    assert result["pit"].item() == 0.5
    assert np.isnan(result["z"].item())  # zero SD gives no z
    assert bool(result["excluded"].item())
    assert result["in_interval"].all()  # the truth sits on the degenerate interval
    assert np.isnan(result["rms_z"].sel(subset="unflagged").item())

    ties = _draws([[[0.0], [1.0]], [[1.0], [2.0]]])
    assert functional_coverage(ties, _truth([1.0]))["pit"].item() == 0.5


def test_missing_truth_is_not_counted():
    draws = _draws(np.random.default_rng(6).normal(size=(2, 50, 3)))
    result = functional_coverage(draws, _truth([0.0, np.nan, 0.0]))
    assert np.isnan(result["pit"].sel(country=1).item())
    assert result["n"].sel(subset="all") == 3
    trials = result["coverage"].sel(subset="all", interval=0.95)
    assert trials.item() in {0.0, 0.5, 1.0}  # two scored functionals


def test_retained_only_from_attrs_and_coordinate():
    rng = np.random.default_rng(7)
    draws = _draws(rng.normal(size=(2, 100, 3)), labels=["A", "B", "C"])
    truth = _truth([0.0, 0.0, 0.0], labels=["A", "B", "C"])

    assert not functional_coverage(draws, truth)["retained_only"].any()

    flagged = draws.copy()
    flagged.attrs["uncertainty_scope"] = "retained_only"
    result = functional_coverage(flagged, truth)
    assert result["retained_only"].all()
    assert result["n"].sel(subset="unflagged") == 0
    assert np.isnan(result["rms_z"].sel(subset="unflagged").item())

    conditional = draws.copy()
    conditional.attrs["uncertainty_scope"] = "retained_state_conditional"
    assert functional_coverage(conditional, truth)["retained_only"].all()

    per_functional = draws.assign_coords(uncertainty_scope=("country", ["full", "retained_only", "full"]))
    result = functional_coverage(
        per_functional, truth, groups={"B": ["B"], "AC": ["A", "C"], "ABC": ["A", "B", "C"]}
    )
    assert result["retained_only"].values.tolist() == [True, False, True]


def test_stacked_sample_dimension():
    rng = np.random.default_rng(8)
    draws = xr.DataArray(
        rng.normal(size=(300, 2)), dims=("sample", "country"), coords={"country": ["A", "B"]}
    )
    result = functional_coverage(draws, _truth([0.0, 0.0], labels=["A", "B"]), sample_dims=("sample",))
    assert result["sd"].sel(country="A") == pytest.approx(draws.sel(country="A").std(ddof=1).item())


def test_wilson_interval_known_values():
    result = wilson_interval(xr.DataArray(5), xr.DataArray(10), confidence=0.95)
    assert result["proportion"].item() == 0.5
    assert result["lower"].item() == pytest.approx(0.2366, abs=1e-3)
    assert result["upper"].item() == pytest.approx(0.7634, abs=1e-3)
    assert np.isnan(wilson_interval(xr.DataArray(0), xr.DataArray(0))["lower"].item())


def test_input_validation():
    draws = _draws(np.random.default_rng(9).normal(size=(2, 20, 2)), labels=["A", "B"])
    truth = _truth([0.0, 0.0], labels=["A", "B"])

    with pytest.raises(ValueError, match="requires prior_draws"):
        functional_coverage(draws, truth, min_sd_fraction_of_prior=0.1)
    with pytest.raises(ValueError, match="intervals"):
        functional_coverage(draws, truth, intervals=(1.0,))
    with pytest.raises(ValueError, match="not in 'country'"):
        functional_coverage(draws, truth, groups={"X": ["A", "Z"]})
    with pytest.raises(ValueError, match="no members"):
        functional_coverage(draws, truth, groups={"X": []})
    with pytest.raises(ValueError, match="sample dimensions"):
        functional_coverage(draws.rename(chain="a", draw="b"), truth)
    with pytest.raises(ValueError, match="two draws"):
        functional_coverage(draws.isel(chain=[0], draw=[0]), truth)
    with pytest.raises(ValueError, match="no 'region' dimension"):
        functional_coverage(draws, truth, functional_dim="region")
    with pytest.raises(ValueError):  # truth on different countries
        functional_coverage(draws, _truth([0.0, 0.0], labels=["A", "C"]))
