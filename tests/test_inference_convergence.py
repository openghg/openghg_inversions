"""ArviZ assessment and project-specific trace adaptation."""

import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inference import convergence
from tests.helpers import make_trace


def _trace(chains: int = 4, draws: int = 512) -> xr.DataTree:
    samples = np.random.default_rng(686).normal(size=(chains, draws, 2))
    return make_trace(
        posterior=xr.Dataset(
            {"x": (("chain", "draw", "region"), samples)},
            coords={"region": ["north", "south"]},
        ),
        sample_stats=xr.Dataset({"diverging": (("chain", "draw"), np.zeros((chains, draws), bool))}),
    )


def test_arviz_defaults_produce_plain_json_assessment_without_mutating_trace():
    trace = _trace()
    original = trace.copy(deep=True)

    summary, report = convergence.posterior_convergence_check(trace)

    assert report["status"] == "pass"
    assert report["measured_values"]["arviz_has_errors"] is False
    assert report["measured_values"]["divergences"] == 0
    assert report["thresholds"] == {"policy": "arviz.diagnose defaults"}
    assert set(summary.metric.values) >= {"r_hat", "ess_bulk", "ess_tail"}
    assert set(report) == {"status", "measured_values", "thresholds", "message"}
    json.dumps(report, allow_nan=False)
    xr.testing.assert_identical(trace, original)


@pytest.mark.parametrize("saved_scope", [False, True])
def test_saved_variable_scope_excludes_deterministics_and_legacy_scope_reports_nonfinite(saved_scope):
    trace = _trace()
    trace["posterior"]["epsilon"] = xr.ones_like(trace["posterior"]["x"])
    if saved_scope:
        trace.attrs["sampler_convergence_variables"] = json.dumps(["x"])

    summary, report = convergence.posterior_convergence_check(trace)

    assert report["status"] == ("pass" if saved_scope else "unknown")
    assert report["measured_values"]["assessed_variables"] == (["x"] if saved_scope else ["x", "epsilon"])
    assert set(summary.data_vars) == ({"x"} if saved_scope else {"x", "epsilon"})
    assert bool(report["measured_values"]["unavailable"]) is not saved_scope


def test_one_chain_is_unknown_even_when_arviz_reports_no_error():
    _, report = convergence.posterior_convergence_check(_trace(chains=1))

    assert report["status"] == "unknown"
    assert (
        "between-chain convergence requires at least two chains" in report["measured_values"]["unavailable"]
    )


@pytest.mark.parametrize("has_errors", [False, True])
def test_nonfinite_diagnostics_cannot_pass_but_do_not_hide_arviz_failures(monkeypatch, has_errors):
    monkeypatch.setattr(convergence.az, "diagnose", lambda *args, **kwargs: (has_errors, {}))
    monkeypatch.setattr(
        convergence,
        "posterior_summary",
        lambda *args, **kwargs: xr.Dataset(
            {"x": ("metric", [np.nan, 800.0, 700.0])},
            coords={"metric": ["r_hat", "ess_bulk", "ess_tail"]},
        ),
    )

    _, report = convergence.posterior_convergence_check(_trace())

    assert report["status"] == ("fail" if has_errors else "unknown")
    assert report["measured_values"]["unavailable"] == ["non-finite diagnostics for x"]


@pytest.mark.parametrize(
    "ordinary,cached,expected",
    [
        ([False, True], None, 1),
        (None, [True, False], 1),
        ([False, True], [True, False], 2),
        ([True, False], [True, False], 1),
        ([False, False], [False, False], 0),
        (None, None, None),
    ],
)
def test_arviz_receives_combined_divergence_flags_without_mutating_trace(ordinary, cached, expected):
    trace = _trace(chains=2)
    stats = {}
    for name, flags in (("diverging", ordinary), ("sigma_nuts_diverging", cached)):
        if flags is not None:
            values = np.zeros((2, 512), dtype=bool)
            values[0, :2] = flags
            stats[name] = (("chain", "draw"), values)
    trace["sample_stats"] = xr.DataTree(xr.Dataset(stats))
    original = trace.copy(deep=True)

    _, report = convergence.posterior_convergence_check(trace)

    assert report["measured_values"]["divergences"] == expected
    assert report["status"] == ("unknown" if expected is None else "fail" if expected else "pass")
    xr.testing.assert_identical(trace, original)


def test_arviz_ess_override_changes_assessment():
    _, report = convergence.posterior_convergence_check(_trace(), ess_threshold=100_000)

    assert report["status"] == "fail"
    assert report["thresholds"]["ess_threshold"] == 100_000


@pytest.mark.parametrize("variable_names", [None, []])
def test_empty_trace_is_unassessable(variable_names):
    summary, report = convergence.posterior_convergence_check(xr.DataTree(), variable_names=variable_names)

    assert not summary.data_vars
    assert report["status"] == "unknown"


@pytest.mark.parametrize("statistic", ["reached_max_treedepth", "energy"])
def test_arviz_checks_tree_depth_and_energy_when_available(statistic):
    trace = _trace()
    values = (
        np.ones((4, 512), bool) if statistic == "reached_max_treedepth" else np.tile(np.arange(512), (4, 1))
    )
    trace["sample_stats"][statistic] = xr.DataArray(values, dims=("chain", "draw"))

    _, report = convergence.posterior_convergence_check(trace)

    assert report["status"] == "fail"
    assert report["measured_values"]["divergences"] == 0
