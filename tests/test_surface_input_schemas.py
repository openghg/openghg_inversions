"""Raw surface schemas inspect metadata while borrowing scientific arrays."""

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data._surface_schema import validate_surface_inputs


def _inputs():
    time = pd.date_range("2020-01-01", periods=2, freq="h")
    grid = {"lat": [50, 51], "lon": [-2, -1], "time": time}
    obs = xr.Dataset({"mf": ("time", [400., 401.], {"units": "ppm"})}, coords={"time": time})
    fp = xr.Dataset({"fp": (("lat", "lon", "time"), np.ones((2, 2, 2)))}, coords=grid)
    flux = fp.rename(fp="flux")
    return {"observations": obs, "site_footprint_data": fp, "footprint_data": {"bio": fp},
            "flux_data": {"bio": flux}, "transport": {"bio": "integrated"}, "species": "co2",
            "boundary_data": None}


def _boundary(inputs):
    fp = inputs["site_footprint_data"].assign_coords(height=[100, 200])
    bc = xr.Dataset(coords=fp.coords)
    for direction in "nesw":
        edge = "lon" if direction in "ns" else "lat"
        dims = (edge, "height", "time")
        fp[f"particle_locations_{direction}"] = (dims, np.full((2, 2, 2), .25))
        bc[f"vmr_{direction}"] = (dims, np.full((2, 2, 2), 400.))
    inputs["site_footprint_data"] = fp
    inputs["boundary_data"] = bc


@pytest.mark.parametrize("boundary", [False, True])
def test_validation_preserves_integer_coordinates_graphs_chunks_and_metadata(boundary):
    inputs = _inputs()
    if boundary:
        _boundary(inputs)

    @delayed
    def forbidden():
        pytest.fail("Schema executed a borrowed array")

    datasets = list({id(ds): ds for ds in [inputs["observations"], inputs["site_footprint_data"],
                *inputs["footprint_data"].values(), *inputs["flux_data"].values()]}.values())
    if boundary:
        datasets.append(inputs["boundary_data"])
    originals = []
    for ds in datasets:
        for name in ds.data_vars:
            ds[name] = ds[name].copy(data=da.from_delayed(forbidden(), shape=ds[name].shape, dtype=float))
        ds.coords["custom_auxiliary"] = ("time", da.from_delayed(forbidden(), shape=(2,), dtype=float))
        ds.attrs["extension"] = "retained"
        originals.append(ds.copy(deep=False))
    with Callback(pretask=lambda *args: pytest.fail("Validation executed a task")):
        validate_surface_inputs(**inputs)
    for ds, original in zip(datasets, originals, strict=True):
        assert ds.attrs == original.attrs
        for name in ds.variables:
            assert ds[name].attrs == original[name].attrs
            assert ds[name].dtype == original[name].dtype
            if name not in ds.indexes:
                assert ds[name].data is original[name].data
                assert ds[name].chunks == original[name].chunks
    assert inputs["site_footprint_data"].lat.dtype.kind == "i"


@pytest.mark.parametrize("location,field", [
    ("observations", "mf"), ("site_footprint_data", "fp"),
    ("footprint_data", "fp"), ("flux_data", "flux"), ("boundary_data", "vmr_n"),
])
@pytest.mark.parametrize("defect", ["missing", "dtype", "dimensions"])
def test_invalid_input_reports_context_without_computation(location, field, defect):
    inputs = _inputs()
    if location == "boundary_data":
        _boundary(inputs)
    mapped = location in {"footprint_data", "flux_data"}
    ds = inputs[location]["bio"] if mapped else inputs[location]
    ds = ds.copy(deep=False)
    if defect == "missing":
        ds = ds.drop_vars(field)
    elif defect == "dtype":
        ds[field] = ds[field].astype(str)
    else:
        ds[field] = xr.DataArray(np.ones(2), dims="wrong_axis")
    if mapped:
        inputs[location] = {"bio": ds}
    else:
        inputs[location] = ds
    context = location + (r"\['bio'\]" if mapped else "")
    expected_field = "dtype" if defect == "dtype" else ("co2" if field == "mf" else field)
    with Callback(pretask=lambda *args: pytest.fail("Validation executed a task")):
        with pytest.raises(ValueError, match=rf"(?s){context}.*{expected_field}"):
            validate_surface_inputs(**inputs)


@pytest.mark.parametrize("legacy", [False, True])
def test_resolved_schema_tracks_consumed_footprint_fields(legacy):
    inputs = _inputs()
    fp = inputs["site_footprint_data"]
    resolved = fp.fp.expand_dims(H_back=[0, 1])
    if legacy:
        fp = resolved.to_dataset(name="fp_HiTRes")
    else:
        fp = xr.Dataset({"fp_time_resolved": resolved, "fp_residual": fp.fp})
    inputs["footprint_data"] = {"bio": fp}
    inputs["transport"] = {"bio": "time_resolved"}
    validate_surface_inputs(**inputs)
    required = "fp_HiTRes" if legacy else "fp_residual"
    inputs["footprint_data"] = {"bio": fp.drop_vars(required)}
    with pytest.raises(ValueError, match="footprint_data"):
        validate_surface_inputs(**inputs)


def test_public_preparation_validates_before_provider_operations(monkeypatch):
    from openghg_inversions.inversion_data import prepare_surface_site, surface

    inputs = _inputs()
    inputs["flux_data"]["bio"] = inputs["flux_data"]["bio"].drop_vars("flux")
    monkeypatch.setattr(surface, "_surface_operations", lambda: pytest.fail("Called provider before validation"))
    with pytest.raises(ValueError, match=r"flux_data\['bio'\].*flux"):
        prepare_surface_site(
            inputs["observations"], inputs["site_footprint_data"], inputs["flux_data"],
            species="co2", transport=inputs["transport"],
        )


@pytest.mark.parametrize("name", ["co2", "custom_concentration"])
def test_selected_primary_schema_view_validates_original_name_without_mutation(name):
    inputs = _inputs()
    obs = inputs["observations"].rename(mf=name)
    inputs["observations"] = obs
    inputs["observation_variable"] = name
    before = obs.copy(deep=True)
    validate_surface_inputs(**inputs)
    xr.testing.assert_identical(obs, before)
    inputs["observations"] = obs.assign({name: obs[name].astype(str)})
    with pytest.raises(ValueError, match="observations.*dtype"):
        validate_surface_inputs(**inputs)
