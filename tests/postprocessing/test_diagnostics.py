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
