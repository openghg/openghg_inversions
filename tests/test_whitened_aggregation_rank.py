"""Tests for choosing the LRPD aggregation-error rank from the whitened spectrum."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh
import xarray as xr

from openghg_inversions.observation_error import (
    ReferenceMismatchCovariance,
    WhitenedRankRule,
    choose_whitened_aggregation_rank,
    prepare_low_rank_aggregation_error,
    prepare_whitened_low_rank_aggregation_error,
    select_whitened_rank,
    whitened_aggregation_spectrum,
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


def _dense_ou_reference(
    sites: np.ndarray,
    hours: np.ndarray,
    error: np.ndarray,
    amplitude: dict[str, float],
    tau: dict[str, float],
) -> np.ndarray:
    """Build a dense within-site OU reference with one time scale per site."""
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


def test_dropped_mode_kl_matches_dense_gaussian_kl_for_ou_reference() -> None:
    """The spectral dropped-mode KL equals the dense KL of the whitened truncation."""
    covariance, sites, hours, error = _smooth_case()
    amplitude = {"MHD": 0.8, "TAC": 1.3}
    tau = {"MHD": 5.0, "TAC": 5.0}
    dense = _dense_ou_reference(sites, hours, error, amplitude, tau)
    reference = Co2ReferenceCovariance(reference_sd=amplitude, tau_hours=5.0).build(
        error, sites, observation_units="ppm", observation_times=hours
    )

    mu = whitened_aggregation_spectrum(covariance, reference)
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
    """Without OU the reference is diagonal and whitening is a row/column scaling."""
    covariance, sites, _, error = _smooth_case()

    reference = Co2ReferenceCovariance().build(error, sites, observation_units="ppm")
    mu = whitened_aggregation_spectrum(covariance, reference)

    assert reference.site_rows == ()
    expected = eigh(covariance, np.diag(error**2 + 1.0), eigvals_only=True)[::-1]
    np.testing.assert_allclose(mu, expected, rtol=1e-9, atol=1e-12)


def test_spectrum_keeps_covariance_scale_roundoff_tolerance() -> None:
    """Negative roundoff accepted by the LRPD factorization is clipped, not rejected, after whitening."""
    covariance = np.diag([100.0, 0.0, -9e-9])
    reference = ReferenceMismatchCovariance(variance=np.asarray([10001.0, 1.0, 1.0]))

    mu = whitened_aggregation_spectrum(covariance, reference)

    assert np.all(mu >= 0.0)
    np.testing.assert_allclose(mu, [100.0 / 10001.0, 0.0, 0.0], atol=1e-15)
    prepare_low_rank_aggregation_error(_labelled(covariance), rank=1)
    prepare_whitened_low_rank_aggregation_error(_labelled(covariance), reference, WhitenedRankRule())
    with pytest.raises(ValueError, match="positive semidefinite"):
        whitened_aggregation_spectrum(np.diag([1.0, -1e-3, 0.5]), reference)


@pytest.mark.parametrize(
    ("covariance", "match"),
    [
        (np.asarray(1.0), "square matrix"),
        (np.ones((3, 2)), "square matrix"),
        (np.eye(4), "4 rows"),
        (np.diag([1.0, np.nan, 1.0]), "finite"),
        (np.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
    ],
)
def test_spectrum_validates_the_aggregation_covariance(covariance: np.ndarray, match: str) -> None:
    """Scalars, wrong shapes, non-finite and asymmetric inputs are rejected before the eigenvalues."""
    reference = ReferenceMismatchCovariance(variance=np.ones(3))
    with pytest.raises(ValueError, match=match):
        whitened_aggregation_spectrum(covariance, reference)
    with pytest.raises(ValueError, match=match):
        choose_whitened_aggregation_rank(covariance, reference, WhitenedRankRule())


def test_ou_reference_is_independent_of_row_order() -> None:
    """Per-site OU settings are resolved by label, so permuting rows only permutes the reference."""
    covariance, sites, hours, error = _smooth_case()
    settings = Co2ReferenceCovariance(reference_sd={"MHD": 0.8, "TAC": 1.3}, tau_hours={"MHD": 0.1, "TAC": 10.0})
    order = np.random.default_rng(3).permutation(sites.size)

    forward = settings.build(error, sites, observation_units="ppm", observation_times=hours)
    shuffled = settings.build(error[order], sites[order], observation_units="ppm", observation_times=hours[order])
    mu_forward = whitened_aggregation_spectrum(covariance, forward)
    mu_shuffled = whitened_aggregation_spectrum(covariance[np.ix_(order, order)], shuffled)

    np.testing.assert_allclose(mu_shuffled, mu_forward, rtol=1e-9, atol=1e-12)
    dense = _dense_ou_reference(sites, hours, error, {"MHD": 0.8, "TAC": 1.3}, {"MHD": 0.1, "TAC": 10.0})
    np.testing.assert_allclose(mu_forward, eigh(covariance, dense, eigvals_only=True)[::-1], rtol=1e-9, atol=1e-12)
    rule = WhitenedRankRule(mu_threshold=1.0)
    rank_forward, _ = choose_whitened_aggregation_rank(covariance, forward, rule)
    rank_shuffled, _ = choose_whitened_aggregation_rank(covariance[np.ix_(order, order)], shuffled, rule)
    assert rank_forward == rank_shuffled


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"variance": np.asarray([1.0, 0.0])}, "positive finite"),
        ({"variance": np.ones(3), "site_rows": (np.asarray([0, 1]),), "blocks": (np.eye(2),)}, "partition"),
        ({"variance": np.ones(2), "site_rows": (np.asarray([0, 1]),), "blocks": (2.0 * np.eye(2),)}, "diagonal"),
        (
            {"variance": np.ones(2), "site_rows": (np.asarray([0, 1]),), "blocks": (np.ones((2, 2)),)},
            "positive definite",
        ),
        (
            {
                "variance": np.ones(2),
                "site_rows": (np.asarray([0, 1]),),
                "blocks": (np.asarray([[1.0, 0.5], [0.0, 1.0]]),),
            },
            "symmetric",
        ),
        ({"variance": np.ones(2), "site_rows": (np.asarray([0.4, 1.4]),), "blocks": (np.eye(2),)}, "integer array"),
        (
            {
                "variance": np.ones(2),
                "site_rows": (np.asarray([0, 1]), np.asarray([], dtype=int)),
                "blocks": (np.eye(2), np.empty((0, 0))),
            },
            "non-empty one-dimensional",
        ),
        (
            {"variance": np.ones(2), "site_rows": (np.asarray([[0, 1]]),), "blocks": (np.eye(2),)},
            "non-empty one-dimensional",
        ),
        ({"variance": np.ones(2), "description": {"bad": object()}}, "JSON-safe"),
        ({"variance": np.ones(2), "description": {"bad": float("nan")}}, "JSON-safe"),
    ],
)
def test_explicit_reference_covariance_is_validated(kwargs: dict[str, object], match: str) -> None:
    """A supplied R0 must be positive, partition the rows with integer indices, and be described in JSON."""
    with pytest.raises(ValueError, match=match):
        ReferenceMismatchCovariance(**kwargs)  # type: ignore[arg-type]


def test_explicit_dense_reference_matches_generalized_eigenvalues() -> None:
    """One block over every row is a general dense reference."""
    block = np.asarray([[2.0, 1.0], [1.0, 2.0]])
    reference = ReferenceMismatchCovariance(
        variance=np.asarray([2.0, 2.0]), site_rows=(np.asarray([0, 1]),), blocks=(block,), description={"source": "test"}
    )
    assert reference.minimum_eigenvalue == pytest.approx(1.0)
    np.testing.assert_allclose(
        whitened_aggregation_spectrum(np.eye(2), reference), eigh(np.eye(2), block, eigvals_only=True)[::-1]
    )
    _, record = choose_whitened_aggregation_rank(np.eye(2), reference, WhitenedRankRule())
    assert record["reference_covariance"] == {"source": "test"}


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


def test_co2_reference_site_mapping_and_units_are_resolved_per_row() -> None:
    """Site mappings expand by label, reference units convert, and missing sites are named."""
    covariance, sites, hours, error = _smooth_case()
    ppm = Co2ReferenceCovariance(reference_sd={"MHD": 0.8, "TAC": 1.3}, tau_hours=5.0)
    ppb = Co2ReferenceCovariance(reference_sd={"MHD": 800.0, "TAC": 1300.0}, units="ppb", tau_hours=5.0)

    reference_ppm = ppm.build(error, sites, observation_units="ppm", observation_times=hours)
    reference_ppb = ppb.build(error, sites, observation_units="ppm", observation_times=hours)

    np.testing.assert_allclose(
        whitened_aggregation_spectrum(covariance, reference_ppm), whitened_aggregation_spectrum(covariance, reference_ppb)
    )
    description = reference_ppb.description
    assert description["reference_sd"] == {"MHD": 800.0, "TAC": 1300.0}
    assert description["units"] == "ppb"
    assert description["reference_sd_observation_units_by_site"] == pytest.approx({"MHD": 0.8, "TAC": 1.3})
    assert json.loads(json.dumps(description)) == description
    with pytest.raises(ValueError, match=r"reference_sd site mapping is missing observation site\(s\): \['TAC'\]"):
        Co2ReferenceCovariance(reference_sd={"MHD": 1.0}).build(error, sites, observation_units="ppm")
    with pytest.raises(ValueError, match=r"tau_hours site mapping is missing observation site\(s\): \['TAC'\]"):
        Co2ReferenceCovariance(tau_hours={"MHD": 5.0}).build(
            error, sites, observation_units="ppm", observation_times=hours
        )


def test_integrated_preparation_matches_standalone_rank_and_factorization() -> None:
    """The integrated path chooses the standalone rank and stores the same factorization."""
    covariance, sites, hours, error = _smooth_case()
    reference = Co2ReferenceCovariance(reference_sd=0.5, tau_hours=5.0).build(
        error, sites, observation_units="ppm", observation_times=hours
    )
    rule = WhitenedRankRule(mu_threshold=0.5)

    rank, record = choose_whitened_aggregation_rank(covariance, reference, rule)
    approximation, integrated = prepare_whitened_low_rank_aggregation_error(_labelled(covariance), reference, rule)
    standalone = prepare_low_rank_aggregation_error(_labelled(covariance), rank=rank)

    assert integrated == record
    assert approximation.diagnostics == standalone.diagnostics
    xr.testing.assert_identical(approximation.aggregation_error.factor, standalone.aggregation_error.factor)


def test_default_preparation_records_rank_selection_through_serialization() -> None:
    """CO2 preparation builds the reference it owns and records one description with the criteria."""
    canonical = _canonical_inputs()
    reference = Co2ReferenceCovariance(reference_sd={"MHD": 1.5, "TAC": 2.0}, tau_hours=24.0)

    prepared = prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=reference)
    restored = Co2PreparedInputs.from_datatree(prepared.to_datatree())

    selection = restored.provenance["aggregation_error_rank_selection"]
    assert restored.provenance["aggregation_error_rank"] == "whitened_spectrum"
    assert selection["chosen_rank"] == prepared.inv_inputs.sizes["agg_rank"]
    assert selection["chosen_rank"] == restored.provenance["aggregation_error"]["requested_rank"]
    assert selection["mu_threshold"] == 0.1
    assert selection["count_mu_above_threshold"] is not None
    assert selection["dropped_mode_kl"] >= 0.0
    description = selection["reference_covariance"]
    assert description["source"] == "Co2ReferenceCovariance"
    assert description["reference_sd"] == {"MHD": 1.5, "TAC": 2.0}
    assert description["tau_hours"] == 24.0
    assert description["correlation"] == "within-site OU"
    assert description["observation_units"] == "ppm"


def test_explicit_preparation_rank_below_count_warns() -> None:
    """An integer rank is an override checked against the default reference."""
    canonical = _canonical_inputs()

    with pytest.warns(UserWarning, match="below the whitened-spectrum rule rank"):
        prepared = prepare_co2_inputs(canonical, _reduction(canonical), aggregation_error_rank=1)

    selection = prepared.provenance["aggregation_error_rank_selection"]
    assert prepared.provenance["aggregation_error_rank"] == 1
    assert (selection["override_rank"], selection["chosen_rank"]) == (1, 1)
    assert selection["rule_rank"] == 3
    assert selection["reference_covariance"]["source"] == "Co2ReferenceCovariance"
    assert prepared.inv_inputs.sizes["agg_rank"] == 1


def _labelled_reference(canonical, values: np.ndarray, units: str) -> xr.DataArray:
    """Return R0 values on the reduction's observation-by-observation labels."""
    return _reduction(canonical).unresolved_observation_covariance.copy(data=values).assign_attrs(units=units)


def test_supplied_labelled_reference_is_converted_and_matches_the_built_reference() -> None:
    """A labelled R0 in squared ppb gives the same rank selection as the equivalent built reference."""
    canonical = _canonical_inputs()
    settings = Co2ReferenceCovariance(reference_sd={"MHD": 1.5, "TAC": 2.0}, tau_hours=24.0)
    observations = canonical.inv_inputs["mf"]
    built = settings.build(
        canonical.inv_inputs["mf_error"].values,
        observations["site"].values,
        observation_units="ppm",
        observation_times=observations["time"].values,
    )
    dense = np.diag(built.variance)
    for rows, block in zip(built.site_rows, built.blocks, strict=True):
        dense[np.ix_(rows, rows)] = block

    from_settings = prepare_co2_inputs(canonical, _reduction(canonical), reference_covariance=settings)
    supplied = prepare_co2_inputs(
        canonical,
        _reduction(canonical),
        reference_covariance=_labelled_reference(canonical, dense * 1e6, "(ppb)^2").rename("R0"),
    )

    expected = from_settings.provenance["aggregation_error_rank_selection"]
    actual = supplied.provenance["aggregation_error_rank_selection"]
    for key in ("chosen_rank", "count_mu_above_threshold"):
        assert actual[key] == expected[key]
    np.testing.assert_allclose(actual["dropped_mode_kl"], expected["dropped_mode_kl"], rtol=1e-10)
    description = actual["reference_covariance"]
    assert description["source"] == "supplied labelled covariance"
    assert description["name"] == "R0"
    assert description["unit_scale_to_observation_units_squared"] == pytest.approx(1e-6)


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

    monkeypatch.setattr(co2_preparation, "prepare_whitened_low_rank_aggregation_error", unreachable)
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
    assert saved.provenance["aggregation_error_rank_selection"]["reference_covariance"]["tau_hours"] == 24.0
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
