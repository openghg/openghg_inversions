"""The twelve lateral regions preserve the native curtain partition and transport."""

from functools import partial

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import twelve_region_basis
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.preparation import resolve_bc_basis
from tests.test_staged_workflow import _params


def _curtains(*, centered=False):
    coordinates = {
        "height": xr.IndexVariable("height", [0., 2999., 3000., 5000.], attrs={"units": "m", "datum": "native"}),
        "lat": xr.IndexVariable("lat", [40., 50., 60.], attrs={"units": "degrees_north"}),
        "lon": xr.IndexVariable("lon", [-20., 0., 20.], attrs={"units": "degrees_east"}),
        "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
    }
    site = xr.Dataset(coords=coordinates)
    # Four equal-size curtains, globally normalized at each release time.
    for number, direction in enumerate("nesw", start=1):
        axis = "lon" if direction in "ns" else "lat"
        mass = xr.DataArray(da.full((4, 3, 2), 1 / 48, chunks=(2, 3, 1)),
                            dims=("height", axis, "time"),
                            coords={dim: site[dim] for dim in ("height", axis, "time")})
        if centered:
            site[f"bc_mass_{direction}"] = mass.assign_attrs(units="1")
            site[f"bc_anomaly_{direction}"] = (mass * number).assign_attrs(units="ppm")
        else:
            site[f"bc_{direction}"] = (mass * (400 + number)).assign_attrs(units="ppm")
    if centered:
        site["bc_reference"] = xr.DataArray(400., attrs={"units": "ppm"})
        site.attrs.update(boundary_representation="centered-v1", boundary_normalization="all-curtains")
    return site


def test_partition_edges_labels_coordinate_units_and_orientation():
    site = _curtains()
    basis = twelve_region_basis(site)
    assert basis.region.values.tolist() == [f"{d}_{part}" for d in "nesw"
                                            for part in ("lower_first", "lower_second", "upper")]
    for axis in ("height", "lat", "lon"):
        xr.testing.assert_identical(basis[axis], site[axis])
    for direction in "nesw":
        mask = basis[f"bc_basis_{direction}"]
        axis = "lon" if direction in "ns" else "lat"
        np.testing.assert_array_equal(mask.sum("region"), np.ones((4, 3)))
        assert mask.attrs["units"] == "1"
        assert mask.sel(height=2999, **{axis: site[axis].values[0]}, region=f"{direction}_lower_first") == 1
        assert mask.sel(height=2999, **{axis: site[axis].values[1]}, region=f"{direction}_lower_second") == 1
        assert mask.sel(height=3000, region=f"{direction}_upper").sum() == 3
        assert mask.sel(height=5000, region=f"{direction}_upper").sum() == 3
        assert mask.sel(region=[r for r in basis.region.values if not r.startswith(direction)]).sum() == 0
    reversed_site = site.isel(lat=slice(None, None, -1), lon=slice(None, None, -1))
    xr.testing.assert_identical(twelve_region_basis(reversed_site).sortby(["lat", "lon"]), basis)
    custom = twelve_region_basis(site, split_height=2999)
    assert custom.bc_basis_n.sel(height=2999, region="n_upper").sum() == 3


def test_real_shaped_partition_does_not_read_payloads():
    grid = xr.Dataset(coords={"height": np.arange(20) * 500., "lat": np.linspace(10, 80, 340),
                              "lon": np.linspace(-100, 60, 390)})
    with Callback(pretask=lambda *args: pytest.fail("read a borrowed payload")):
        basis = twelve_region_basis(grid)
    assert basis.attrs["bc_basis_height_units"] == "unspecified"
    assert basis.attrs["bc_basis_height_datum"] == "unspecified"
    for direction in "nesw":
        np.testing.assert_array_equal(basis[f"bc_basis_{direction}"].sum("region"), 1)
    assert all(sum(int(basis[f"bc_basis_{d}"].sel(region=label).sum()) for d in "nesw") > 0
               for label in basis.region.values)


@pytest.mark.parametrize("centered", [False, True])
def test_projection_is_lazy_preserves_reference_and_collapses_to_nesw(centered):
    site = _curtains(centered=centered)
    original = site.copy(deep=True)
    config = RhimeConfig.from_params(_params(use_bc=True, bc_basis_algorithm="twelve-region"), multisector=False)
    builder = resolve_bc_basis(config.data)
    with Callback(pretask=lambda *args: pytest.fail("computed borrowed curtains")):
        projected = bc_sensitivity({"AAA": site}, builder)["AAA"]
        nesw = bc_sensitivity({"AAA": site})["AAA"]
    names = ["H_bc_anomaly", "H_bc_mass"] if centered else ["H_bc"]
    for name in names:
        assert isinstance(projected[name].data, da.Array)
        assert projected[name].attrs["bc_basis_algorithm"] == "twelve-region"
        assert projected[name].attrs["bc_basis_split_height"] == 3000.
        for direction in "nesw":
            collapsed = projected[name].sel(bc_region=[f"{direction}_{part}"
                                                     for part in ("lower_first", "lower_second", "upper")]).sum("bc_region")
            xr.testing.assert_allclose(collapsed, nesw[name].sel(bc_region=direction, drop=True))
    if centered:
        xr.testing.assert_identical(projected.bc_reference, site.bc_reference)
        np.testing.assert_allclose(projected.H_bc_mass.sum("bc_region"), 1., atol=1e-15)
        absolute = bc_sensitivity({"AAA": _curtains()}, builder)["AAA"].H_bc
        xr.testing.assert_allclose(projected.H_bc_anomaly + projected.bc_reference * projected.H_bc_mass, absolute)
    xr.testing.assert_identical(site, original)


def test_basis_netcdf_roundtrip_preserves_geometry_and_provenance(tmp_path):
    basis = twelve_region_basis(_curtains(), split_height=2999)
    target = tmp_path / "basis.nc"
    basis.to_netcdf(target)
    with xr.open_dataset(target) as restored:
        xr.testing.assert_identical(restored, basis)
        actual = bc_sensitivity({"AAA": _curtains()}, restored)["AAA"].H_bc.compute()
    expected = bc_sensitivity({"AAA": _curtains()}, partial(twelve_region_basis, split_height=2999))["AAA"].H_bc.compute()
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("split", [None, True, "3000", 3000 + 1j, np.inf, np.nan])
def test_bad_split_is_rejected_at_configuration_boundary(split):
    with pytest.raises(ValueError, match="bc_basis_split_height"):
        RhimeConfig.from_params(_params(use_bc=True, bc_basis_algorithm="twelve-region",
                                       bc_basis_split_height=split), multisector=False)


@pytest.mark.parametrize("axis", ["height", "lat", "lon"])
def test_nonfinite_or_missing_coordinates_are_rejected(axis):
    site = _curtains()
    with pytest.raises(ValueError, match=axis):
        twelve_region_basis(site.drop_indexes(axis))
    with pytest.raises(ValueError, match=axis):
        twelve_region_basis(site.assign_coords({axis: np.full(site.sizes[axis], np.nan)}))


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("basis_kind", ["builder", "dataset"])
def test_prepared_sensitivity_roundtrip_retains_geometry_provenance(tmp_path, centered, suffix, basis_kind):
    from openghg_inversions.inversion_data import PreparedInputs, prepare_rhime_inputs_from_xarray
    from tests.test_centered_boundary_preparation import _site
    from tests.test_xarray_input_adapter import _basis_functions

    curtains = _curtains(centered=centered)
    basis = twelve_region_basis if basis_kind == "builder" else twelve_region_basis(curtains)
    projected = bc_sensitivity({"AAA": curtains}, basis)["AAA"]
    site = _site().isel(time=slice(0, 2)).drop_dims("bc_region").assign_coords(time=projected.time)
    names = ["H_bc_anomaly", "H_bc_mass"] if centered else ["H_bc"]
    if not centered:
        site = site.drop_vars("bc_reference")
        site.attrs = {}
    for name in names:
        site[name] = projected[name]
    prepared = prepare_rhime_inputs_from_xarray({"AAA": site}, basis_functions=_basis_functions())
    path = tmp_path / f"prepared{suffix}"
    prepared.save(path)
    loaded = PreparedInputs.load(path)
    xr.testing.assert_identical(loaded.inv_inputs, prepared.inv_inputs)
    for name in names:
        assert loaded.inv_inputs[name].attrs["bc_basis_algorithm"] == "twelve-region"
        assert loaded.inv_inputs[name].attrs["bc_basis_split_height"] == 3000.
        assert loaded.inv_inputs[name].attrs["bc_basis_height_units"] == "m"
        assert loaded.inv_inputs[name].attrs["bc_basis_height_datum"] == "native"
        assert loaded.inv_inputs[name].attrs["bc_basis_lat_units"] == "degrees_north"
        assert loaded.inv_inputs[name].attrs["bc_basis_lon_units"] == "degrees_east"
        for direction in "nesw":
            assert loaded.inv_inputs[name].attrs[f"bc_basis_midpoint_{direction}"] == (0. if direction in "ns" else 50.)


@pytest.mark.parametrize("split", [None, True, "3000", 3000 + 1j, np.inf, np.nan])
def test_public_basis_constructor_rejects_invalid_split(split):
    with pytest.raises(ValueError, match="split_height"):
        twelve_region_basis(_curtains(), split_height=split)


@pytest.mark.parametrize("builder", [None, twelve_region_basis])
def test_nonfinite_curtain_cell_stays_in_its_own_regions(builder):
    site = _curtains(centered=True).compute()
    site["bc_anomaly_n"][0, 0, 0] = np.nan
    basis = {} if builder is None else {"bc_basis": builder}
    anomaly = bc_sensitivity({"AAA": site}, **basis)["AAA"].H_bc_anomaly.isel(time=0)
    poisoned = [str(r) for r in anomaly.bc_region.values if np.isnan(anomaly.sel(bc_region=r))]
    assert poisoned == (["n"] if builder is None else ["n_lower_first"])


@pytest.mark.parametrize("basis_kind", ["NESW", "twelve-region", "file"])
def test_absolute_sensitivity_metadata_does_not_depend_on_basis_kind(tmp_path, basis_kind):
    from openghg_inversions.basis import boundary_basis

    site = _curtains()
    if basis_kind == "file":
        twelve_region_basis(site).to_netcdf(tmp_path / "basis.nc")
        basis = xr.open_dataset(tmp_path / "basis.nc").load()
    else:
        basis = boundary_basis(basis_kind)
    sensitivity = bc_sensitivity({"AAA": site}, basis)["AAA"].H_bc
    assert sensitivity.dims == ("bc_region", "time")
    assert sensitivity.attrs["units"] == "ppm"
    assert sensitivity.attrs["bc_basis_algorithm"] == ("NESW" if basis_kind == "NESW" else "twelve-region")


def test_unknown_builtin_boundary_basis_is_rejected():
    from openghg_inversions.basis import boundary_basis

    with pytest.raises(ValueError, match="NESW"):
        boundary_basis("eight-region")
