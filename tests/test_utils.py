import pytest
import xarray as xr

from openghg_inversions.utils import combine_datasets


def test_combine_datasets():
    """Align offset flux coordinates to the footprint without changing either input."""
    import numpy as np

    fp = xr.Dataset(
        {"fp": (("lat", "lon", "time"), np.arange(1.0, 13.0).reshape(2, 3, 2))},
        coords={
            "lat": [50.0, 51.0],
            "lon": [-2.0, -1.0, 0.0],
            "time": np.array(["2019-01-01T00:00", "2019-01-01T01:00"], dtype="datetime64[ns]"),
        },
    )
    flux = xr.Dataset(
        {
            "flux": (
                ("time", "lat", "lon"),
                [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]],
                {"units": "mol m-2 s-1"},
            )
        },
        coords={
            "lat": [51.1, 50.1],
            "lon": [-1.9, -0.9, 0.1],
            "time": np.array(["2019-01-01T00:15", "2019-01-01T01:15"], dtype="datetime64[ns]"),
        },
    )
    original_fp = fp.copy(deep=True)
    original_flux = flux.copy(deep=True)

    combined = combine_datasets(fp, flux, method="nearest")

    expected_flux = xr.DataArray(
        [[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], [[14.0, 15.0, 16.0], [11.0, 12.0, 13.0]]],
        dims=("time", "lat", "lon"),
        coords={name: fp[name] for name in ("time", "lat", "lon")},
        attrs={"units": "mol m-2 s-1"},
        name="flux",
    )
    xr.testing.assert_identical(combined.flux, expected_flux)
    xr.testing.assert_identical(combined.fp, fp.fp)
    xr.testing.assert_identical(fp, original_fp)
    xr.testing.assert_identical(flux, original_flux)


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
