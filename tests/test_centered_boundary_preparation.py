"""Centered boundary operands survive preparation without changing their identity."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.boundary_sensitivity import BoundaryAlignment, align_boundary_operands
from openghg_inversions.filters import daily_median, six_hr_mean
from openghg_inversions.inversion_data import RhimePreparedInputs, prepare_rhime_inputs_from_xarray
from openghg_inversions.inversion_inputs import make_inv_inputs
from tests.test_xarray_input_adapter import _basis_functions, _site_dataset


def _site():
    ds = _site_dataset("AAA", ["2020-01-01", "2020-01-01T03:00", "2020-01-01T12:00"])
    ds["H"] = xr.DataArray(np.ones((2, 3)), dims=("region", "time"),
                           coords={"region": [0, 1], "time": ds.time}, attrs={"units": "ppm"})
    ds["H_bc_anomaly"] = xr.DataArray(
        [[1., 3., 2.], [-2., 0., 4.]], dims=("bc_region", "time"),
        coords={"bc_region": ["north", "south"], "time": ds.time}, attrs={"units": "ppm"})
    ds["H_bc_mass"] = xr.ones_like(ds.H_bc_anomaly) * .25
    ds.H_bc_mass.attrs["units"] = "1"
    ds["bc_reference"] = xr.DataArray(400., attrs={"units": "ppm"})
    ds.attrs.update(boundary_representation="centered-v1", boundary_normalization="all-curtains")
    return ds.drop_vars(["release_lat", "release_lon"])


def test_arbitrary_basis_projects_both_operands_lazily():
    coords = {"lat": [0], "lon": [0], "height": [0], "time": pd.date_range("2020", periods=2)}
    ds = xr.Dataset(attrs={"boundary_representation": "centered-v1"})
    basis = xr.Dataset()
    for i, direction in enumerate("nesw"):
        for operand, units in [("anomaly", "ppm"), ("mass", "1")]:
            ds[f"bc_{operand}_{direction}"] = xr.DataArray(
                da.full((1, 1, 1, 2), i + 1., chunks=(1, 1, 1, 1)),
                dims=("lat", "lon", "height", "time"), coords=coords, attrs={"units": units})
        basis[f"bc_basis_{direction}"] = xr.DataArray(
            [[[[i + 2., -(i + 1.)]]]], dims=("lat", "lon", "height", "region"),
            coords={"lat": [0], "lon": [0], "height": [0], "region": ["a", "b"]})
    ds["bc_reference"] = xr.DataArray(400., attrs={"units": "ppm"})
    original = ds.copy(deep=True)
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args)):
        result = bc_sensitivity({"AAA": ds}, basis)["AAA"]
    assert not tasks
    for name in ("H_bc_anomaly", "H_bc_mass"):
        np.testing.assert_allclose(result[name].transpose("bc_region", "time"), [[40., 40.], [-30., -30.]])
    xr.testing.assert_identical(result.bc_reference, ds.bc_reference)
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("frequency", [None, "6h", "monthly"])
def test_reusable_alignment_identity(frequency):
    ds = _site().chunk({"time": 1})
    original = ds.copy(deep=True)
    anchor = None if frequency in (None, "monthly") else "2020-01-01"
    with Callback(posttask=lambda *args: pytest.fail("operand materialized during alignment")):
        result = align_boundary_operands(ds, frequency=frequency, anchor_time=anchor)
    absolute = ds.H_bc_anomaly + ds.bc_reference * ds.H_bc_mass
    expected = BoundaryAlignment.prepare(absolute, frequency=frequency, anchor_time=anchor).data
    xr.testing.assert_allclose(result.H_bc_anomaly + result.bc_reference * result.H_bc_mass, expected.rename(None))
    assert result.bc_reference.ndim == 0
    assert result.H_bc_mass.attrs["units"] == "1"
    xr.testing.assert_identical(ds, original)


def test_gather_keeps_raw_regions_and_scalar_reference():
    ds = _site()
    original = ds.copy(deep=True)
    result = make_inv_inputs({"AAA": ds, "CCC": ds}, bc_freq="6h")
    assert result.sizes["bc_region"] == 2
    assert result.bc_reference.ndim == 0
    assert result.attrs["boundary_normalization"] == "all-curtains"
    assert not isinstance(result.indexes["bc_region"], pd.MultiIndex)
    xr.testing.assert_identical(ds, original)


@pytest.mark.parametrize("failure", ["mixed", "reference", "partial", "units"])
def test_gather_rejects_inconsistent_sites(failure):
    first, second = _site(), _site()
    if failure == "mixed":
        second = second.drop_vars(["H_bc_anomaly", "H_bc_mass", "bc_reference"])
        second.attrs = {}
    elif failure == "reference":
        second["bc_reference"] = second.bc_reference + 1
        second.bc_reference.attrs["units"] = "ppm"
    elif failure == "partial":
        second = second.drop_vars("H_bc_mass")
    else:
        second.H_bc_mass.attrs["units"] = "ppm"
    with pytest.raises(ValueError):
        make_inv_inputs({"AAA": first, "BBB": second})


def test_core_drops_common_nonfinite_rows():
    ds = _site()
    ds.H_bc_mass[0, 1] = np.inf
    result = make_inv_inputs({"AAA": ds})
    assert result.sizes["nmeasure"] == 2
    ds["bc_reference"] = xr.DataArray(np.nan, attrs={"units": "ppm"})
    with pytest.raises(ValueError, match="finite"):
        make_inv_inputs({"AAA": ds})


def test_filter_rejects_median_and_means_use_common_rows():
    ds = _site()
    with pytest.raises(ValueError, match="daily_median"):
        daily_median(ds)
    ds.H_bc_anomaly[0, 1] = np.nan
    original = ds.copy(deep=True)
    result = six_hr_mean(ds)
    assert result.bc_reference.ndim == 0
    np.testing.assert_allclose(result.H_bc_mass.isel(time=0), ds.H_bc_mass.isel(time=0))
    np.testing.assert_allclose(result.mf[0], ds.mf[0])
    np.testing.assert_allclose(result.H_bc_anomaly.isel(time=0), ds.H_bc_anomaly.isel(time=0))
    xr.testing.assert_identical(ds, original)


def test_adapter_and_centered_codec_round_trip(tmp_path):
    prepared = prepare_rhime_inputs_from_xarray({"AAA": _site()}, basis_functions=_basis_functions(), bc_freq="6h")
    assert prepared.inv_inputs.bc_reference.ndim == 0
    tree = prepared.to_datatree()
    assert tree.attrs["schema_version"] == 2
    restored = RhimePreparedInputs.from_datatree(tree)
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)
    prepared.save(tmp_path / "centered.nc")
    restored = RhimePreparedInputs.load(tmp_path / "centered.nc")
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)
    tree.attrs["schema_version"] = 1
    with pytest.raises(ValueError, match="version"):
        RhimePreparedInputs.from_datatree(tree)


def test_absolute_codec_still_writes_version_one():
    prepared = prepare_rhime_inputs_from_xarray(
        {"AAA": _site_dataset("AAA", ["2020-01-01"], sampled_baseline=True)}, basis_functions=_basis_functions())
    assert prepared.to_datatree().attrs["schema_version"] == 1
    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)


def test_gathered_alignment_uses_requested_labels_and_keeps_zero_anomaly_states():
    ds = _site()
    ds["H_bc_anomaly"] = xr.zeros_like(ds.H_bc_anomaly)
    ds.H_bc_anomaly.attrs["units"] = "ppm"
    gathered = make_inv_inputs({"AAA": ds}, bc_freq="6h")
    assert gathered.sizes["bc_region"] == 2
    labels = gathered.nmeasure[[2, 0]]
    aligned = align_boundary_operands(gathered, frequency="6h", observation_labels=labels)
    np.testing.assert_array_equal(aligned.nmeasure, labels)
    expected = BoundaryAlignment.prepare(
        gathered.H_bc_anomaly + gathered.bc_reference * gathered.H_bc_mass,
        frequency="6h", observation_labels=labels,
    ).data
    xr.testing.assert_allclose(aligned.H_bc_anomaly + aligned.bc_reference * aligned.H_bc_mass, expected)


def test_common_mean_preserves_absolute_identity_with_missing_mass():
    ds = _site()
    ds.H_bc_mass[1, 1] = np.nan
    ds["bc_mod"] = (ds.H_bc_anomaly + ds.bc_reference * ds.H_bc_mass).sum("bc_region", skipna=False)
    result = six_hr_mean(ds)
    reconstructed = (result.H_bc_anomaly + result.bc_reference * result.H_bc_mass).sum("bc_region")
    xr.testing.assert_allclose(reconstructed, result.bc_mod)
    assert result.mf_error[0] == ds.mf_error[0]


def test_core_materializes_shared_operands_once_without_mutating_inputs():
    from dask import delayed

    executions = []

    @delayed
    def shared_value():
        executions.append(True)
        return np.array(1.)

    scalar = da.from_delayed(shared_value(), shape=(), dtype=float)
    ds = _site()
    for name in ("H", "H_bc_anomaly", "H_bc_mass", "bc_reference", "mf", "mf_error"):
        ds[name] = (ds[name] * scalar).assign_attrs(ds[name].attrs)
    original = ds.copy(deep=True)
    gathered = make_inv_inputs({"AAA": ds}, bc_freq="6h")
    assert executions == [True]
    assert not hasattr(gathered.H_bc_anomaly.data, "__dask_graph__")
    assert hasattr(ds.H_bc_anomaly.data, "__dask_graph__")
    xr.testing.assert_identical(ds, original)


def test_prepared_boundary_rejects_already_expanded_operands():
    from openghg_inversions.boundary_sensitivity import validate_boundary_operands

    expanded = align_boundary_operands(_site(), frequency="6h")
    with pytest.raises(ValueError, match="unexpanded spatial operands"):
        validate_boundary_operands(expanded)
    with pytest.raises(ValueError, match="unexpanded spatial operands"):
        align_boundary_operands(expanded, frequency="1D")


def test_optional_diagnostic_missing_value_does_not_choose_science_rows():
    ds = _site()
    ds["pblh"] = ("time", [100., np.nan, 200.])
    averaged = six_hr_mean(ds)
    assert averaged.mf[0] == pytest.approx(float(ds.mf[:2].mean()))
    np.testing.assert_allclose(averaged.H_bc_anomaly.isel(time=0), ds.H_bc_anomaly.isel(time=slice(0, 2)).mean("time"))


def test_unknown_representation_does_not_load_as_legacy_absolute():
    from openghg_inversions.boundary_sensitivity import validate_boundary_operands

    legacy = _site_dataset("AAA", ["2020-01-01"], sampled_baseline=True)
    prepared = prepare_rhime_inputs_from_xarray({"AAA": legacy}, basis_functions=_basis_functions())
    tree = prepared.to_datatree()
    tree["inv_inputs"].attrs["boundary_representation"] = "centered-v2"
    with pytest.raises(ValueError, match="Unsupported boundary_representation"):
        RhimePreparedInputs.from_datatree(tree)
    with pytest.raises(ValueError, match="Unsupported boundary_representation"):
        validate_boundary_operands(xr.Dataset(attrs={"boundary_representation": "centered-v2"}))


def test_gather_preserves_site_specific_transport_and_normalization_provenance():
    import json

    first, second = _site(), _site()
    for number, data in enumerate((first, second)):
        data.attrs.update(surface_transport='{"inventory": "integrated"}', surface_provider_revision="abc123",
                          surface_transport_provenance=f"site-{number}", boundary_mass_tolerance=number * 0.001)
    prepared = prepare_rhime_inputs_from_xarray({"AAA": first, "BBB": second}, basis_functions=_basis_functions())
    restored = RhimePreparedInputs.from_datatree(prepared.to_datatree())
    facts = json.loads(restored.inv_inputs.attrs["surface_preparation_provenance"])
    assert facts["AAA"]["surface_transport_provenance"] == "site-0"
    assert facts["BBB"]["boundary_mass_tolerance"] == 0.001
    assert "surface_transport_provenance" not in restored.inv_inputs.attrs


@pytest.mark.parametrize("frequency,anchor", [("1D", "NaT"), (None, "invalid"), ("monthly", "2020-01-01")])
def test_alignment_rejects_invalid_or_ignored_anchors(frequency, anchor):
    with pytest.raises(ValueError, match="anchor_time"):
        align_boundary_operands(_site(), frequency=frequency, anchor_time=anchor)
