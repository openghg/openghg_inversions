"""Tests for choosing the LRPD aggregation-error rank from the whitened spectrum."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh
import xarray as xr

from openghg_inversions.observation_error import (
    WhitenedRankRule,
    _prepare_whitened_low_rank_aggregation_error,
    _reference_cholesky,
    _whitened_spectrum,
    prepare_low_rank_aggregation_error,
    select_whitened_rank,
    whitened_eigenvalue_kl,
)
from openghg_inversions.rhime.co2 import Co2PreparedInputs, Co2ReferenceCovariance, handoffs, prepare_co2_inputs
from openghg_inversions.rhime.co2 import co2_preparation, execution

from test_co2_handoffs import FACTS, _model
from test_co2_preparation import _canonical_inputs, _reduction


def _smooth_case(n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth non-diagonal A, two interleaved sites, hours and errors."""
    rng = np.random.default_rng(7)
    hours = np.repeat(np.arange(n // 2, dtype=float), 2)
    sites = np.tile(["MHD", "TAC"], n // 2)
    position = np.arange(n, dtype=float)
    kernel = np.exp(-np.abs(position[:, None] - position[None, :]) / 4.0)
    loading = rng.normal(size=(n, 3))
    covariance = 2.0 * kernel + loading @ loading.T
    error = rng.uniform(0.1, 0.3, size=n)
    return covariance, sites, hours, error


def _observations(sites: np.ndarray, hours: np.ndarray, error: np.ndarray) -> tuple[xr.DataArray, xr.DataArray]:
    """Label observations in ppm with site and numeric-hour coordinates, and their errors."""
    coords = {"nmeasure": np.arange(sites.size), "site": ("nmeasure", sites), "time": ("nmeasure", hours)}
    observations = xr.DataArray(np.zeros(sites.size), dims="nmeasure", coords=coords, attrs={"units": "ppm"})
    return observations, xr.DataArray(error, dims="nmeasure", coords=coords, attrs={"units": "ppm"})


def _reference(settings: Co2ReferenceCovariance, sites, hours, error) -> np.ndarray:
    """Return dense R0 values built by the CO2 settings."""
    return np.asarray(settings.covariance(*_observations(sites, hours, error)).values)


def _dense_ou_reference(sites, hours, error, amplitude: dict[str, float], tau: dict[str, float]) -> np.ndarray:
    """Build a dense within-site OU reference independently of the settings."""
    same_site = sites[:, None] == sites[None, :]
    sd = np.asarray([amplitude[site] for site in sites])
    scale = np.asarray([tau[site] for site in sites])
    correlation = np.where(same_site, np.exp(-np.abs(hours[:, None] - hours[None, :]) / scale[:, None]), 0.0)
    return np.diag(error**2) + sd[:, None] * correlation * sd[None, :]


def _gaussian_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Return KL(N(0, p) || N(0, q)) from the textbook trace/log-det formula."""
    n = p.shape[0]
    _, logdet_p = np.linalg.slogdet(p)
    _, logdet_q = np.linalg.slogdet(q)
    return 0.5 * float(np.trace(np.linalg.solve(q, p)) - n + logdet_q - logdet_p)


def _labelled(values: np.ndarray) -> xr.DataArray:
    """Label a dense covariance with integer observation coordinates."""
    n = values.shape[0]
    return xr.DataArray(
        values, dims=("nmeasure", "nmeasure_cov"), coords={"nmeasure": np.arange(n), "nmeasure_cov": np.arange(n)}
    )


def _spectrum(covariance: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the private whitened spectrum of A against dense R0."""
    return _whitened_spectrum(covariance, _reference_cholesky(reference, covariance.shape[0]))


def test_dropped_mode_kl_matches_dense_gaussian_kl_for_ou_reference() -> None:
    """The spectral dropped-mode KL equals the dense KL of the whitened truncation."""
    covariance, sites, hours, error = _smooth_case()
    amplitude = {"MHD": 0.8, "TAC": 1.3}
    dense = _dense_ou_reference(sites, hours, error, amplitude, {"MHD": 5.0, "TAC": 5.0})
    built = _reference(Co2ReferenceCovariance(reference_sd=amplitude, tau_hours=5.0), sites, hours, error)
    np.testing.assert_allclose(built, dense, rtol=1e-12)

    mu = _spectrum(covariance, built)
    values, vectors = eigh(covariance, dense)
    np.testing.assert_allclose(mu, values[::-1], rtol=1e-9, atol=1e-12)

    k = 4
    leading = vectors[:, ::-1][:, :k]
    truncated = dense @ leading @ np.diag(values[::-1][:k]) @ leading.T @ dense
    expected = _gaussian_kl(dense + covariance, dense + truncated)
    rank, record = select_whitened_rank(mu, WhitenedRankRule(rank=k, mu_threshold=None, kl_budget_per_observation=1.0))

    assert rank == k
    np.testing.assert_allclose(record["dropped_mode_kl"], expected, rtol=1e-8)
    np.testing.assert_allclose(whitened_eigenvalue_kl(mu[k:]), expected, rtol=1e-8)


def test_independent_reference_spectrum_matches_scaled_covariance() -> None:
    """Without OU the reference is diagonal and the spectrum matches the scaled covariance."""
    covariance, sites, hours, error = _smooth_case()

    built = _reference(Co2ReferenceCovariance(), sites, hours, error)

    np.testing.assert_array_equal(built, np.diag(error**2 + 1.0))
    expected = eigh(covariance, np.diag(error**2 + 1.0), eigvals_only=True)[::-1]
    np.testing.assert_allclose(_spectrum(covariance, built), expected, rtol=1e-9, atol=1e-12)


def test_factorization_owns_the_roundoff_check_on_a() -> None:
    """Negative roundoff accepted by the LRPD factorization is clipped for scoring; material negativity fails there."""
    covariance = np.diag([100.0, 0.0, -9e-9])
    reference = np.diag([10001.0, 1.0, 1.0])

    np.testing.assert_allclose(_spectrum(covariance, reference), [100.0 / 10001.0, 0.0, 0.0], atol=1e-15)
    prepare_low_rank_aggregation_error(_labelled(covariance), rank=1)
    _prepare_whitened_low_rank_aggregation_error(
        _labelled(covariance), reference, WhitenedRankRule(), reference_description={}
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        _prepare_whitened_low_rank_aggregation_error(
            _labelled(np.diag([1.0, -1e-3, 0.5])), reference, WhitenedRankRule(), reference_description={}
        )


@pytest.mark.parametrize(
    ("covariance", "reference", "match"),
    [
        (np.diag([1.0, np.nan, 1.0]), np.eye(3), "finite"),
        (np.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.eye(3), "symmetric"),
        (np.eye(3), np.eye(4), "3 x 3"),
        (np.eye(3), np.diag([1.0, np.inf, 1.0]), "reference covariance must be finite"),
        (np.eye(3), np.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (np.eye(3), np.diag([1.0, -1.0, 1.0]), "positive definite"),
        (np.eye(3), np.ones((3, 3)), "positive definite|numerically singular"),
        (np.eye(3), np.diag([1.0, 1e-17, 1.0]), "numerically singular"),
    ],
)
def test_integrated_path_validates_a_and_the_reference(
    covariance: np.ndarray, reference: np.ndarray, match: str
) -> None:
    """A is checked once for finite symmetric values; R0 must be finite, symmetric and numerically positive definite."""
    with pytest.raises(ValueError, match=match):
        _prepare_whitened_low_rank_aggregation_error(
            _labelled(covariance), reference, WhitenedRankRule(), reference_description={}
        )


def test_reference_rank_check_uses_the_condition_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A singular R0 that a LAPACK build factors with a tiny pivot is still rejected."""
    from openghg_inversions import _linalg

    real_cho_factor = _linalg.cho_factor

    def accepting_cholesky(matrix, *args, **kwargs):
        """Factor a singular matrix with a tiny positive last pivot, as Accelerate can."""
        shift = 4 * np.finfo(np.float64).eps * np.abs(np.diag(matrix)).max()
        return real_cho_factor(matrix + shift * np.eye(matrix.shape[0]), *args, **kwargs)

    monkeypatch.setattr(_linalg, "cho_factor", accepting_cholesky)
    with pytest.raises(ValueError, match="numerically singular"):
        _reference_cholesky(np.ones((3, 3)), 3)


def test_ou_reference_is_independent_of_row_order() -> None:
    """Per-site OU settings are resolved by label, so permuting rows only permutes the reference."""
    covariance, sites, hours, error = _smooth_case()
    settings = Co2ReferenceCovariance(reference_sd={"MHD": 0.8, "TAC": 1.3}, tau_hours={"MHD": 0.1, "TAC": 10.0})
    order = np.random.default_rng(3).permutation(sites.size)

    forward = _reference(settings, sites, hours, error)
    shuffled = _reference(settings, sites[order], hours[order], error[order])

    np.testing.assert_allclose(shuffled, forward[np.ix_(order, order)], rtol=1e-12)
    dense = _dense_ou_reference(sites, hours, error, {"MHD": 0.8, "TAC": 1.3}, {"MHD": 0.1, "TAC": 10.0})
    np.testing.assert_allclose(forward, dense, rtol=1e-12)
    rule = WhitenedRankRule(mu_threshold=1.0)
    rank_forward, _ = select_whitened_rank(_spectrum(covariance, forward), rule)
    rank_shuffled, _ = select_whitened_rank(_spectrum(covariance[np.ix_(order, order)], shuffled), rule)
    assert rank_forward == rank_shuffled


def test_co2_reference_returns_the_labelled_covariance_contract() -> None:
    """The built reference is labelled like a supplied one, with squared units and a JSON description."""
    _, sites, hours, error = _smooth_case()
    observations, observation_error = _observations(sites, hours, error)
    settings = Co2ReferenceCovariance(reference_sd={"MHD": 800.0, "TAC": 1300.0}, units="ppb", tau_hours=5.0)

    reference = settings.covariance(observations, observation_error)

    assert reference.dims == ("nmeasure", "nmeasure_cov")
    assert reference.indexes["nmeasure"].equals(observations.indexes["nmeasure"])
    assert reference.indexes["nmeasure_cov"].equals(observations.indexes["nmeasure"])
    assert reference.attrs["units"] == "(ppm)**2"
    description = json.loads(reference.attrs["description"])
    assert description["reference_sd"] == {"MHD": 800.0, "TAC": 1300.0}
    assert description["units"] == "ppb"
    assert description["reference_sd_observation_units_by_site"] == pytest.approx({"MHD": 0.8, "TAC": 1.3})
    in_ppm = Co2ReferenceCovariance(reference_sd={"MHD": 0.8, "TAC": 1.3}, tau_hours=5.0)
    np.testing.assert_allclose(reference.values, _reference(in_ppm, sites, hours, error), rtol=1e-12)


def test_co2_reference_names_missing_sites_and_coordinates() -> None:
    """Incomplete site mappings and missing coordinates raise ValueError naming the problem."""
    _, sites, hours, error = _smooth_case()
    observations, observation_error = _observations(sites, hours, error)
    with pytest.raises(ValueError, match=r"reference_sd site mapping is missing observation site\(s\): \['TAC'\]"):
        Co2ReferenceCovariance(reference_sd={"MHD": 1.0}).covariance(observations, observation_error)
    with pytest.raises(ValueError, match=r"tau_hours site mapping is missing observation site\(s\): \['TAC'\]"):
        Co2ReferenceCovariance(tau_hours={"MHD": 5.0}).covariance(observations, observation_error)
    with pytest.raises(ValueError, match="'time'"):
        Co2ReferenceCovariance(tau_hours=5.0).covariance(observations.drop_vars("time"), observation_error)


def test_rank_selection_on_synthetic_spectrum() -> None:
    """Count, KL-budget and combined criteria select the documented ranks."""
    mu = np.asarray([5.0, 1.0, 0.5, 0.2, 0.11, 0.1, 0.05, 0.01])
    terms = 0.5 * (mu - np.log1p(mu))

    count_rank, count_record = select_whitened_rank(mu, WhitenedRankRule())
    assert count_rank == 5
    assert count_record["count_mu_above_threshold"] == 5
    np.testing.assert_allclose(count_record["dropped_mode_kl"], terms[5:].sum())
    assert count_record["first_dropped_mu"] == 0.1

    budget = (terms[3:].sum() + 1e-12) / mu.size
    kl_rank, kl_record = select_whitened_rank(mu, WhitenedRankRule(mu_threshold=None, kl_budget_per_observation=budget))
    assert kl_rank == 3
    assert kl_record["kl_budget_rank"] == 3
    assert kl_record["dropped_mode_kl_per_observation"] <= budget

    both_rank, _ = select_whitened_rank(mu, WhitenedRankRule(kl_budget_per_observation=budget))
    assert both_rank == 5

    none_rank, none_record = select_whitened_rank(np.asarray([0.05, 0.0]), WhitenedRankRule())
    assert none_rank == 1
    assert none_record["count_mu_above_threshold"] == 0


@pytest.mark.parametrize(
    ("eigenvalues", "match"),
    [
        (np.asarray([]), "non-empty"),
        (np.asarray([[1.0, 0.5]]), "non-empty"),
        (np.asarray([1.0, -0.1]), "non-negative"),
        (np.asarray([1.0, np.inf]), "finite"),
        (np.asarray([0.5, 1.0]), "descending"),
    ],
)
def test_rank_selection_validates_eigenvalues(eigenvalues: np.ndarray, match: str) -> None:
    """Rank selection rejects spectra it cannot interpret."""
    with pytest.raises(ValueError, match=match):
        select_whitened_rank(eigenvalues, WhitenedRankRule())


def test_explicit_rank_below_rule_rank_warns_and_is_kept() -> None:
    """A benchmarking override is honoured, with a warning only below the rule rank."""
    mu = np.asarray([5.0, 1.0, 0.5, 0.2, 0.01])

    with pytest.warns(UserWarning, match="below the whitened-spectrum rule rank 4"):
        rank, record = select_whitened_rank(mu, WhitenedRankRule(rank=2))
    assert (rank, record["rule_rank"], record["override_rank"]) == (2, 4, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank, _ = select_whitened_rank(mu, WhitenedRankRule(rank=50))
    assert rank == mu.size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu_threshold": 0.0},
        {"kl_budget_per_observation": -1.0},
        {"mu_threshold": None, "kl_budget_per_observation": None},
        {"rank": 0},
        {"rank": True},
    ],
)
def test_rule_rejects_invalid_settings(kwargs: dict[str, object]) -> None:
    """Non-positive criteria, empty criteria and invalid ranks are rejected."""
    with pytest.raises(ValueError, match="WhitenedRankRule"):
        WhitenedRankRule(**kwargs)  # type: ignore[arg-type]


def test_rule_carries_only_rank_criteria() -> None:
    """The reference covariance settings live with the CO2 reference, not the rule."""
    assert set(WhitenedRankRule.__dataclass_fields__) == {"mu_threshold", "kl_budget_per_observation", "rank"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_sd": 0.0},
        {"reference_sd": {"MHD": -1.0}},
        {"reference_sd": {}},
        {"tau_hours": 0.0},
        {"units": ""},
    ],
)
def test_co2_reference_rejects_invalid_settings(kwargs: dict[str, object]) -> None:
    """Zero amplitudes, empty mappings, non-positive time scales and empty units are rejected."""
    with pytest.raises(ValueError, match="Co2ReferenceCovariance"):
        Co2ReferenceCovariance(**kwargs)  # type: ignore[arg-type]


def test_integrated_path_matches_standalone_selection_and_factorization() -> None:
    """The integrated path chooses the spectrum's rank and stores the standalone factorization."""
    covariance, sites, hours, error = _smooth_case()
    reference = _reference(Co2ReferenceCovariance(reference_sd=0.5, tau_hours=5.0), sites, hours, error)
    rule = WhitenedRankRule(mu_threshold=0.5)

    rank, selection = select_whitened_rank(_spectrum(covariance, reference), rule)
    approximation, record = _prepare_whitened_low_rank_aggregation_error(
        _labelled(covariance), reference, rule, reference_description={"source": "test"}
    )
    standalone = prepare_low_rank_aggregation_error(_labelled(covariance), rank=rank)

    assert record == {"reference_covariance": {"source": "test"}, **selection}
    assert approximation.diagnostics == standalone.diagnostics
    xr.testing.assert_identical(approximation.aggregation_error.factor, standalone.aggregation_error.factor)
    with pytest.raises(ValueError, match="JSON-safe"):
        _prepare_whitened_low_rank_aggregation_error(
            _labelled(covariance), reference, rule, reference_description={"bad": float("nan")}
        )


def test_default_preparation_records_rank_selection_through_serialization() -> None:
    """CO2 preparation records the reference description with the criteria and survives a round trip."""
    canonical = _canonical_inputs()
    reference = Co2ReferenceCovariance(reference_sd={"MHD": 1.5, "TAC": 2.0}, tau_hours=24.0)

    prepared = prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=reference)
    restored = Co2PreparedInputs.from_datatree(prepared.to_datatree())

    selection = restored.provenance["aggregation_error_rank_selection"]
    assert restored.provenance["aggregation_error_rank"] == "whitened_spectrum"
    assert selection["chosen_rank"] == prepared.inv_inputs.sizes["agg_rank"]
    assert selection["chosen_rank"] == restored.provenance["aggregation_error"]["requested_rank"]
    assert selection["mu_threshold"] == 0.1
    assert selection["dropped_mode_kl"] >= 0.0
    recorded = selection["reference_covariance"]
    assert recorded["name"] == "reference_covariance"
    assert recorded["units"] == "(ppm)**2"
    assert recorded["unit_scale_to_observation_units_squared"] == 1.0
    assert recorded["representation"] == "dense"
    settings = recorded["description"]
    assert settings["source"] == "Co2ReferenceCovariance"
    assert settings["reference_sd"] == {"MHD": 1.5, "TAC": 2.0}
    assert settings["tau_hours"] == 24.0
    assert settings["correlation"] == "within-site OU"


def test_explicit_preparation_rank_below_count_warns() -> None:
    """An integer rank is an override checked against the default reference."""
    canonical = _canonical_inputs()

    with pytest.warns(UserWarning, match="below the whitened-spectrum rule rank"):
        prepared = prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=1)

    selection = prepared.provenance["aggregation_error_rank_selection"]
    assert prepared.provenance["aggregation_error_rank"] == 1
    assert (selection["override_rank"], selection["chosen_rank"]) == (1, 1)
    assert selection["rule_rank"] == 3
    assert selection["reference_covariance"]["description"]["source"] == "Co2ReferenceCovariance"
    assert prepared.inv_inputs.sizes["agg_rank"] == 1


def _labelled_reference(canonical, values: np.ndarray, units: str) -> xr.DataArray:
    """Return R0 values on the reduction's observation-by-observation labels."""
    return _reduction(canonical).unresolved_observation_covariance.copy(data=values).assign_attrs(units=units)


def test_built_and_supplied_references_are_equivalent() -> None:
    """An equivalent supplied R0 in squared ppb selects the same rank and stores the same factor."""
    canonical = _canonical_inputs()
    settings = Co2ReferenceCovariance(reference_sd={"MHD": 1.5, "TAC": 2.0}, tau_hours=24.0)
    built = settings.covariance(canonical.inv_inputs["mf"], canonical.inv_inputs["mf_error"])

    from_settings = prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=settings)
    supplied = prepare_co2_inputs(
        canonical,
        _reduction(canonical),
        reference_covariance=_labelled_reference(canonical, built.values * 1e6, "(ppb)^2").rename("R0"),
    )

    expected = from_settings.provenance["aggregation_error_rank_selection"]
    actual = supplied.provenance["aggregation_error_rank_selection"]
    for key in ("chosen_rank", "count_mu_above_threshold", "rule_rank"):
        assert actual[key] == expected[key]
    np.testing.assert_allclose(actual["dropped_mode_kl"], expected["dropped_mode_kl"], rtol=1e-10)
    xr.testing.assert_allclose(supplied.inv_inputs["low_rank_factor"], from_settings.inv_inputs["low_rank_factor"])

    assert expected["reference_covariance"]["unit_scale_to_observation_units_squared"] == 1.0
    assert expected["reference_covariance"]["description"]["source"] == "Co2ReferenceCovariance"
    recorded = actual["reference_covariance"]
    assert recorded["name"] == "R0"
    assert recorded["units"] == "(ppb)^2"
    assert recorded["description"] is None
    assert recorded["unit_scale_to_observation_units_squared"] == pytest.approx(1e-6)


def test_supplied_reference_is_checked_while_labelled() -> None:
    """Row and column order and covariance units are checked before materializing."""
    canonical = _canonical_inputs()
    good = _labelled_reference(canonical, np.eye(3), "(ppm)^2")
    reversed_columns = good.isel({good.dims[1]: [2, 1, 0]})
    reversed_rows = good.isel({good.dims[0]: [2, 1, 0]})

    for supplied, match in (
        (reversed_columns, "columns must match"),
        (reversed_rows, "rows must match"),
        (good.assign_attrs(units="ppm"), "squared mole-fraction unit"),
        (good.assign_attrs(units="(m)^2"), "squared mole-fraction unit"),
        (good.assign_attrs(units=""), "non-empty units"),
        (xr.DataArray(np.ones(3), dims="nmeasure"), "observation-by-observation"),
        (good.copy(data=np.diag([1.0, -1.0, 1.0])), "positive definite"),
    ):
        with pytest.raises(ValueError, match=match):
            prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=supplied)
    with pytest.raises(TypeError, match="reference_covariance"):
        prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=np.eye(3))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only to low-rank storage"):
        prepare_co2_inputs(
            canonical, _reduction(canonical), aggregation_error_rank=None, reference_covariance=Co2ReferenceCovariance()
        )


def test_observation_error_units_are_checked_before_whitening(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default route checks mf_error before the rank calculation uses it."""
    canonical = _canonical_inputs()
    canonical.inv_inputs["mf_error"].attrs["units"] = "ppb"

    def unreachable(*args, **kwargs):
        raise AssertionError("the rank calculation ran before mf_error was checked")

    monkeypatch.setattr(co2_preparation, "_prepare_whitened_low_rank_aggregation_error", unreachable)
    with pytest.raises(ValueError, match="CO2 mf_error units"):
        prepare_co2_inputs(canonical, _reduction(canonical))


def test_dense_re_preparation_drops_inherited_rank_diagnostics() -> None:
    """Aggregation-error diagnostics belong to the preparation that wrote them."""
    canonical = _canonical_inputs()
    low_rank = prepare_co2_inputs(canonical, _reduction(canonical))
    assert "aggregation_error_rank_selection" in low_rank.provenance

    dense = prepare_co2_inputs(
        canonical, _reduction(canonical), aggregation_error_rank=None, provenance=low_rank.provenance
    )

    assert dense.aggregation_error_mode == "dense"
    assert dense.provenance["aggregation_error_rank"] is None
    assert "aggregation_error_rank_selection" not in dense.provenance
    assert "aggregation_error" not in dense.provenance


def test_staged_preparation_uses_the_supplied_reference(tmp_path) -> None:
    """The staged prepare handoff passes the reference to a new preparation and saves its description."""
    canonical = _canonical_inputs()
    reference = Co2ReferenceCovariance(reference_sd={"MHD": 1.5, "TAC": 2.0}, tau_hours=24.0)

    handoffs.prepare(
        canonical_inputs=canonical,
        reduction=_reduction(canonical),
        reference_covariance=reference,
        output_dir=tmp_path / "stage",
        **FACTS,
    )
    saved = Co2PreparedInputs.load(tmp_path / "stage" / "prepared-inputs.nc")
    direct = prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=reference)

    assert (
        saved.provenance["aggregation_error_rank_selection"]
        == direct.provenance["aggregation_error_rank_selection"]
    )
    assert saved.provenance["aggregation_error_rank_selection"]["reference_covariance"]["description"]["tau_hours"] == 24.0
    with pytest.raises(ValueError, match="reference_covariance applies only to a new preparation"):
        handoffs.prepare(
            prepared_path=tmp_path / "stage" / "prepared-inputs.nc",
            output_dir=tmp_path / "import",
            reference_covariance=reference,
        )


def test_full_run_passes_the_supplied_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_co2 forwards the reference to a new preparation and rejects it otherwise."""
    canonical = _canonical_inputs()
    reference = Co2ReferenceCovariance(reference_sd=2.0)
    seen = {}

    class Reached(Exception):
        pass

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise Reached

    params = {
        "format_version": 1,
        "recipe": "co2",
        "prepared_inputs": dict(FACTS),
        **_model(),
        "outputs": {"output_format": "none"},
    }
    prepared = prepare_co2_inputs(canonical, _reduction(canonical))
    monkeypatch.setattr(execution, "prepare_co2_inputs", capture)
    with pytest.raises(Reached):
        execution.run_co2(
            params=params, canonical_inputs=canonical, reduction=_reduction(canonical), reference_covariance=reference
        )
    assert seen["reference_covariance"] is reference
    with pytest.raises(ValueError, match="reference_covariance applies only to a new preparation"):
        execution.run_co2(params=params, prepared=prepared, reference_covariance=reference)
