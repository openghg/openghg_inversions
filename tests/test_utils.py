import pytest
import xarray as xr
from openghg.retrieve import get_footprint, get_flux

from openghg_inversions.utils import combine_datasets


def test_combine_datasets():
    fp = get_footprint(site="tac", domain="europe").data
    flux = get_flux(species="ch4", source="total-ukghg-edgar7", domain="europe").data

    comb = combine_datasets(fp, flux)

    with pytest.raises(AssertionError) as exc_info:
        xr.testing.assert_allclose(flux.flux.squeeze("time").drop_vars("time"), comb.flux.isel(time=0))

    # coordinates should be different because we aligned the flux to the footprint
    assert exc_info.match("Differing coordinates")

    # values should not be different
    with pytest.raises(AssertionError):
        # the match fails, so this raises an assertion error; if the match is found
        # no error is raised and pytest complains that it did not see an AssertionError
        assert exc_info.match("Differing values")
