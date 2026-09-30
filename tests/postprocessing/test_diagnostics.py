import pytest
import numpy as np
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.postprocessing.diagnostics import _r2_by_site, summary
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from tests.helpers import make_trace


@pytest.fixture
def inv_out():
    basis = xr.DataArray([[1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]})
    basis_functions = BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=xr.ones_like(basis, dtype=float),
        operator_kwargs={"state_dim": "region"},
    )
    return InversionOutput(
        trace=make_trace(
            posterior=xr.Dataset(
                {"x": (("chain", "draw", "region"), [[[1.0], [1.1]], [[0.9], [1.2]]])},
                coords={"chain": [0, 1], "draw": [0, 1], "region": [0]},
            ),
        ),
        inv_inputs=xr.Dataset(coords={"region": [0], "nmeasure": [0]}),
        basis_functions=basis_functions,
    )


def test_summary(inv_out):
    """ArviZ diagnostics consume the exact posterior Dataset group."""
    summ = summary(inv_out)

    assert [f"{dv}_trace" for dv in inv_out.trace_group("posterior").data_vars] == list(summ.data_vars)

    assert list(summ.metric) == ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]


def test_summary_preserves_unrounded_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    inv_out: InversionOutput,
) -> None:
    """Diagnostic summaries retain values on convergence-threshold edges."""
    seen: dict[str, object] = {}

    def fake_summary(*args: object, **kwargs: object) -> xr.Dataset:
        """Record ArviZ options and return a minimal diagnostics summary."""
        seen.update(kwargs)
        return xr.Dataset(
            {"x": ("summary", [1.0101, 399.99, 399.99, 0.1, 0.1])},
            coords={"summary": ["r_hat", "ess_bulk", "ess_tail", "mcse_mean", "mcse_sd"]},
        )

    monkeypatch.setattr("openghg_inversions.inference.diagnostics.az.summary", fake_summary)

    result = summary(inv_out)

    assert seen["round_to"] == "none"
    assert result.sel(metric="r_hat")["x_trace"].item() == 1.0101


def test_bayesian_r2_preserves_removed_arviz_score_semantics() -> None:
    """Local Bayesian R² retains the former ArviZ mean and standard deviation."""
    observed = xr.DataArray([1.0, 2.0, 3.0], dims="time")
    predicted = xr.DataArray(
        [[1.0, 2.0, 4.0], [0.0, 2.0, 3.0]],
        dims=("draw", "time"),
    )
    data = xr.Dataset({"y_obs": observed, "y_posterior_predictive": predicted}).expand_dims(site=["MHD"])

    result = _r2_by_site(data)
    variance_estimate = np.var(predicted.values, axis=1)
    variance_residual = np.var(observed.values - predicted.values, axis=1)
    samples = variance_estimate / (variance_estimate + variance_residual)

    np.testing.assert_allclose(result["r2_bayes"], [samples.mean()])
    np.testing.assert_allclose(result["r2_bayes_std"], [samples.std()])


def test_scientific_metrics_preserve_diagnostic_registry() -> None:
    """Canonical metrics and compatibility entry points use the same functions."""
    from openghg_inversions.postprocessing import diagnostics, metrics

    assert list(diagnostics.diagnostics) == ["summary", "bayes_r2_by_site", "bayes_r2_by_site_resample"]
    assert diagnostics._r2_by_site is metrics._r2_by_site
    for name, params in [
        ("bayes_r2_by_site", ["inv_out", "report_prior"]),
        ("bayes_r2_by_site_resample", ["inv_out", "freq", "report_prior"]),
    ]:
        entry = diagnostics.diagnostics[name]
        assert entry.func is getattr(metrics, name)
        assert getattr(diagnostics, name) is entry.func
        assert entry.params == params
