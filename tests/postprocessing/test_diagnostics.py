import pytest

from openghg_inversions.postprocessing.diagnostics import summary
from openghg_inversions.postprocessing.inversion_output import InversionOutput


@pytest.fixture
def inv_out(raw_data_path):
    return InversionOutput.load(raw_data_path / "inversion_output.nc")


def test_summary(inv_out):
    summ = summary(inv_out)
    print(summ.metric)

    assert [f"{dv}_trace" for dv in inv_out.trace.posterior.data_vars] == list(summ.data_vars)

    assert list(summ.metric) == ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]
