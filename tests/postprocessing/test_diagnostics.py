import pytest
import arviz as az
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.postprocessing.diagnostics import summary
from openghg_inversions.postprocessing.inversion_output import InversionOutput


@pytest.fixture
def inv_out():
    basis = xr.DataArray([[1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]})
    basis_functions = BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=xr.ones_like(basis, dtype=float),
        operator_kwargs={"state_dim": "region"},
    )
    return InversionOutput(
        trace=az.from_dict(
            posterior={"x": [[[1.0], [1.1]], [[0.9], [1.2]]]},
            coords={"region": [0]},
            dims={"x": ["region"]},
        ),
        inv_inputs=xr.Dataset(coords={"region": [0], "nmeasure": [0]}),
        basis_functions=basis_functions,
    )


def test_summary(inv_out):
    summ = summary(inv_out)
    print(summ.metric)

    assert [f"{dv}_trace" for dv in inv_out.trace.posterior.data_vars] == list(summ.data_vars)

    assert list(summ.metric) == ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]
