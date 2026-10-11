import pytest
import xarray as xr
from openghg.retrieve import get_footprint, get_flux

from openghg_inversions.utils import combine_datasets

def test_combine_datasets(openghg_test_store):
    fp = get_footprint(site="tac", domain="europe").data
    flux = get_flux(species="ch4", source="total-ukghg-edgar7", domain="europe").data

    comb = combine_datasets(fp, flux, method="nearest")

    if not (fp.lon == flux.lon).all() or not (fp.lat == flux.lat).all():
        # check that comb.flux has different coordinates from the original flux
        with pytest.raises(AssertionError) as exc_info:
            xr.testing.assert_allclose(flux.flux.squeeze("time").drop_vars("time"), comb.flux.isel(time=0, drop=True))

        # coordinates should be different because we aligned the flux to the footprint
        assert exc_info.match("Differing coordinates")

        # values should not be different
        with pytest.raises(AssertionError):
            # the match fails, so this raises an assertion error; if the match is found
            # no error is raised and pytest complains that it did not see an AssertionError
            assert exc_info.match("Differing values")


@pytest.mark.parametrize("lazy", [False, True])
def test_netcdf_writer_preserves_standalone_coordinates_and_bounds(tmp_path, lazy):
    """Densify sparse data without losing PARIS platform coordinates or bounds attrs."""
    import dask.array as da
    import numpy as np
    import sparse

    from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs

    payload = sparse.COO.from_numpy(np.array([1.0, 2.0]))
    if lazy:
        payload = da.from_array(payload, chunks=1)
    time_attrs = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}
    source = xr.Dataset(
        {
            "mf_observed": ("index", payload),
            "time_bnds": (("index", "nbnds"), [[0.0, 1.0], [1.0, 2.0]], time_attrs),
        },
        coords={
            "time": ("index", [0.5, 1.5], {**time_attrs, "bounds": "time_bnds"}),
            "platform": ("platform", ["TAC-185m"], {"long_name": "observing platform"}),
        },
        attrs={"product": "concentration"},
    )
    path = tmp_path / "concentration.nc"
    write_netcdf_preserving_bounds_attrs(source, path, unlimited_dims=["index"])

    with xr.open_dataset(path, decode_cf=False) as restored:
        assert restored.sizes["platform"] == 1
        assert restored.platform.dims == ("platform",)
        np.testing.assert_array_equal(restored.platform, ["TAC-185m"])
        assert restored.platform.attrs == source.platform.attrs
        assert restored.attrs == source.attrs
        assert restored.encoding["unlimited_dims"] == {"index"}
        np.testing.assert_array_equal(restored.mf_observed, [1.0, 2.0])
        for name, value in time_attrs.items():
            assert restored.time_bnds.attrs[name] == value
    assert source.mf_observed.data is payload
