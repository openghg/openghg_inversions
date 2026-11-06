"""Native temporal contracts, independent field oracles, and borrowed lazy ownership."""
from types import SimpleNamespace
from unittest.mock import Mock

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import BoundaryTemporalPolicy
from openghg_inversions.inversion_data import acquisition
from openghg_inversions.inversion_data._boundary import baseline_reference_sensitivities, select_boundary_fields


def field(hours=(0, 3, 9), constant=False):
    times = pd.Timestamp("2024-01-31 21:00") + pd.to_timedelta(hours, unit="h")
    return xr.Dataset({f"vmr_{d}": ("time", da.from_array(np.full(len(hours), 400.) if constant else 400. + np.array(hours), chunks=1), {"units": "ppm"}) for d in "nesw"}, coords={"time": times}, attrs={"time_period": "6h"})


def targets(hours):
    return xr.DataArray(pd.Timestamp("2024-01-31 21:00") + pd.to_timedelta(hours, unit="h"), dims="time")


def policy(method, **kwargs):
    return BoundaryTemporalPolicy(method=method, input_semantics="point", max_gap="6h", **kwargs)


@pytest.mark.parametrize("method,expected", [("nearest", [400, 403, 403, 409]), ("linear", [400, 401.5, 404, 409])])
def test_irregular_fields_ties_and_month_seam_remain_lazy(method, expected):
    bc = field()
    original = bc.copy(deep=True)
    with Callback(pretask=lambda *args: pytest.fail("temporal selection computed borrowed data")):
        selected = select_boundary_fields(bc, targets([0, 1.5, 4, 9]), policy=policy(method))
    assert isinstance(selected.vmr_n.data, da.Array)
    np.testing.assert_allclose(selected.vmr_n.compute(), expected)
    assert selected.vmr_n.attrs["units"] == "ppm"
    xr.testing.assert_identical(bc, original)


@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_constant_native_oracle(method):
    selected = select_boundary_fields(field(constant=True), targets([1, 4, 7]), policy=policy(method))
    np.testing.assert_array_equal(selected.compute().to_array(), 400.)


@pytest.mark.parametrize("method", ["nearest", "linear"])
@pytest.mark.parametrize("hours", [[-1], [10]])
def test_point_samples_do_not_extrapolate(method, hours):
    with pytest.raises(ValueError, match="extrapolate"):
        select_boundary_fields(field(), targets(hours), policy=policy(method))


def test_guard_is_explicit_and_not_cadence():
    with pytest.raises(ValueError, match="interval exceeds"):
        select_boundary_fields(field((0, 9)), targets([1]), policy=policy("linear"))
    with pytest.raises(ValueError, match="sample exceeds"):
        select_boundary_fields(field((0, 15)), targets([7]), policy=policy("nearest"))
    # Exact native samples need no interpolating bracket, including a singleton.
    np.testing.assert_allclose(select_boundary_fields(field((0,)), targets([0]), policy=policy("linear")).vmr_n, [400])


def interval_field():
    bc = field((1, 4, 8))
    bc["time_bounds"] = (("time", "bounds"), np.stack([targets([0, 3, 7]).values, targets([3, 6, 10]).values], axis=1))
    return bc


def test_interval_containment_is_left_closed_right_open_with_gaps():
    bc = interval_field()
    contract = BoundaryTemporalPolicy(method="interval", input_semantics="interval_mean", bounds="time_bounds")
    np.testing.assert_allclose(select_boundary_fields(bc, targets([0, 3, 5.9, 7, 9.9]), policy=contract).vmr_n, [401, 404, 404, 408, 408])
    for hours in ([6], [10], [-1]):
        with pytest.raises(ValueError, match="do not cover"):
            select_boundary_fields(bc, targets(hours), policy=contract)
    for bad in (xr.DataArray(0), bc.time_bounds.chunk(time=1)):
        with pytest.raises(ValueError, match="shape|eager"):
            select_boundary_fields(bc.assign(time_bounds=bad), targets([1]), policy=contract)


@pytest.mark.parametrize("kwargs", [dict(method="nearest"), dict(method="linear", input_semantics="interval_mean", max_gap="6h"), dict(method="interval", input_semantics="interval_mean"), dict(method="nearest", input_semantics="point", max_gap="0h"), dict(method="hold", input_semantics="point"), dict(method="hold", retrieval_padding="6h")])
def test_contract_must_be_declared(kwargs):
    with pytest.raises(ValueError):
        BoundaryTemporalPolicy(**kwargs)


def test_archived_hold_has_existing_period_coverage():
    bc = field()
    np.testing.assert_allclose(select_boundary_fields(bc, targets([0, 2, 8, 14]), policy=BoundaryTemporalPolicy()).vmr_n, [400, 400, 403, 409])
    with pytest.raises(ValueError, match="coverage"):
        select_boundary_fields(bc, targets([15]), policy=BoundaryTemporalPolicy())


@pytest.mark.parametrize("species,reference", [("co2", 400.), ("o2", 209300.)])
def test_native_linear_field_contracted_before_changing_particles(species, reference):
    bc = field((0, 3))
    for i, d in enumerate("nesw"):
        bc[f"vmr_{d}"] = (bc[f"vmr_{d}"] - 400. + reference + i).assign_attrs(units="ppm")
    fp = xr.Dataset({f"particle_locations_{d}": ("time", [1. if d == "n" else 0., 1. if d == "e" else 0.], {"units": "1"}) for d in "nesw"}, coords={"time": targets([1, 2]).values})
    operands = baseline_reference_sensitivities(bc, fp, species=species, reference=reference * 1e-6, reference_units="mol/mol", mass_tolerance=0., temporal_policy=policy("linear"))
    anomaly = sum(operands[f"bc_anomaly_{d}"] for d in "nesw")
    np.testing.assert_allclose(anomaly, [1., 3.], atol=1e-9)
    np.testing.assert_allclose(sum(operands[f"bc_mass_{d}"] for d in "nesw"), 1.)


def test_retrieval_search_includes_explicit_adjacent_file_support(monkeypatch):
    search = Mock(return_value=SimpleNamespace(results=pd.DataFrame({"uuid": ["tm5", "tm5"]})))
    get = Mock(return_value=SimpleNamespace(data=field(), metadata={"uuid": "tm5"}))
    monkeypatch.setattr(acquisition, "search_bc", search)
    monkeypatch.setattr(acquisition, "get_bc", get)
    kwargs = dict(use_bc=True, species="co2", domain="EUROPE", bc_input="TM5", start_date="2024-02-01", end_date="2024-02-02", store="user")
    with pytest.raises(ValueError, match="retrieval_padding"):
        acquisition._get_boundary_data(**kwargs, temporal_policy=policy("linear"))
    acquisition._get_boundary_data(**kwargs, temporal_policy=policy("linear", retrieval_padding="6h"))
    assert pd.Timestamp(search.call_args.kwargs["start_date"]) == pd.Timestamp("2024-01-31 18:00")
    assert pd.Timestamp(get.call_args.kwargs["end_date"]) == pd.Timestamp("2024-02-02 06:00")
    assert get.call_args.kwargs["uuid"] == ["tm5"]


@pytest.mark.parametrize("hours", [(0, 0, 3), (0, 3, 1)])
def test_invalid_native_time_rejected(hours):
    with pytest.raises(ValueError, match="unique and increasing"):
        select_boundary_fields(field(hours), targets([1]), policy=policy("linear"))


def test_interval_bounds_overlap_missing_and_invalid_rejected():
    bc = interval_field()
    contract = BoundaryTemporalPolicy(method="interval", input_semantics="interval_mean", bounds="time_bounds")
    with pytest.raises(ValueError, match="missing"):
        select_boundary_fields(bc.drop_vars("time_bounds"), targets([1]), policy=contract)
    for values in ([[-1, 5], [3, 6], [7, 10]], [[0, .5], [3, 6], [7, 10]], [[0, 3], [6, 3], [7, 10]]):
        bounds = np.stack([targets([v[0] for v in values]).values, targets([v[1] for v in values]).values], axis=1)
        with pytest.raises(ValueError, match="intervals must"):
            select_boundary_fields(bc.assign(time_bounds=(("time", "bounds"), bounds)), targets([1]), policy=contract)


def test_linear_promotes_before_subtraction():
    bc = field((0, 3))
    for d in "nesw":
        bc[f"vmr_{d}"] = xr.DataArray(np.array([-3e38, 3e38], dtype=np.float32), dims="time", attrs={"units": "ppm"})
    selected = select_boundary_fields(bc, targets([1.5]), policy=policy("linear"))
    np.testing.assert_allclose(selected.vmr_n, 0.)


@pytest.mark.parametrize("representation", ["ordinary", "centered"])
def test_surface_policy_wiring_and_provenance(representation):
    from tests.test_surface_acquisition import inputs, boundary_inputs
    from openghg_inversions.inversion_data import prepare_surface_site

    obs, fp, flux = inputs.__wrapped__()
    fp, bc = boundary_inputs(fp)
    bc = bc.isel(time=[0, 2])
    for d in "nesw":
        bc[f"vmr_{d}"] = (bc[f"vmr_{d}"] + xr.DataArray([0., 2.], dims="time")).assign_attrs(units="ppm")
    result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"}, boundary_data=bc,
                                  boundary_representation=representation, boundary_reference=400., boundary_reference_units="ppm",
                                  boundary_mass_tolerance=0., boundary_temporal_policy=policy("linear"))
    np.testing.assert_allclose(result.bc_mod, [400., 401., 402.])
    assert result.attrs["boundary_temporal_policy"] == "linear"
    assert result.attrs["boundary_input_semantics"] == "point"
    assert result.attrs["boundary_temporal_policy_revision"] == "native-fields-v1"
    assert result.attrs["boundary_temporal_max_gap"] == "0 days 06:00:00"


def test_linear_mixed_datetime_resolutions():
    bc = field((0, 3)).assign_coords(time=lambda ds: ds.time.astype("datetime64[s]"))
    selected = select_boundary_fields(bc, targets([1]).astype("datetime64[ns]"), policy=policy("linear"))
    np.testing.assert_allclose(selected.vmr_n, [401.])


def test_sparse_dask_interpolation_remains_lazy():
    import sparse

    bc = field((0, 3))
    for d in "nesw":
        bc[f"vmr_{d}"] = xr.DataArray(da.from_array(sparse.COO.from_numpy(np.array([400., 403.])), chunks=1), dims="time", attrs={"units": "ppm"})
    with Callback(pretask=lambda *args: pytest.fail("interpolation computed sparse source")):
        selected = select_boundary_fields(bc, targets([1, 2]), policy=policy("linear"))
    assert isinstance(selected.vmr_n.data, da.Array)
    computed = selected.vmr_n.data.compute()
    assert isinstance(computed, sparse.COO)
    np.testing.assert_allclose(computed.todense(), [401., 402.])


def test_retrieval_rejects_distinct_products_in_padding(monkeypatch):
    from openghg.types import SearchError

    monkeypatch.setattr(acquisition, "search_bc", Mock(return_value=SimpleNamespace(results=pd.DataFrame({"uuid": ["a", "b"]}))))
    retrieve = Mock(side_effect=AssertionError("must not combine distinct products"))
    monkeypatch.setattr(acquisition, "get_bc", retrieve)
    with pytest.raises(SearchError, match="distinct boundary products"):
        acquisition._get_boundary_data(use_bc=True, species="co2", domain="EUROPE", bc_input="TM5", start_date="2024-02-01", end_date="2024-02-02", store="user", temporal_policy=policy("linear", retrieval_padding="6h"))


def test_fresh_interval_acquisition_resolves_only_bounds_metadata(monkeypatch):
    native = interval_field()
    native["time_bounds"] = native.time_bounds.chunk(time=1)
    for d in "nesw":
        native[f"vmr_{d}"] = native[f"vmr_{d}"].copy(data=da.from_array(np.array([401., 404., 408.]), chunks=1, name=f"borrowed-curtain-{d}"))
    wrapper = SimpleNamespace(data=native, metadata={"uuid": "tm5"})
    monkeypatch.setattr(acquisition, "search_bc", Mock(return_value=SimpleNamespace(results=pd.DataFrame({"uuid": ["tm5"]}))))
    monkeypatch.setattr(acquisition, "get_bc", Mock(return_value=wrapper))
    contract = BoundaryTemporalPolicy(method="interval", input_semantics="interval_mean", bounds="time_bounds", retrieval_padding="6h")
    with Callback(pretask=lambda key, *args: pytest.fail("acquisition computed curtain data") if "borrowed-curtain" in str(key) else None):
        result, _ = acquisition._get_boundary_data(use_bc=True, species="co2", domain="EUROPE", bc_input="TM5", start_date="2024-02-01", end_date="2024-02-02", store="user", temporal_policy=contract)
    assert isinstance(result.data.time_bounds.data, np.ndarray)
    assert isinstance(native.time_bounds.data, da.Array)
    assert result is not wrapper
    for d in "nesw":
        assert result.data[f"vmr_{d}"].data is native[f"vmr_{d}"].data
    np.testing.assert_allclose(select_boundary_fields(result.data, targets([1]), policy=contract).vmr_n, [401.])


def test_interval_source_labels_can_mark_interval_ends():
    bc = interval_field().assign_coords(time=targets([3, 6, 10]).values)
    contract = BoundaryTemporalPolicy(method="interval", input_semantics="interval_mean", bounds="time_bounds")
    np.testing.assert_allclose(select_boundary_fields(bc, targets([1, 4, 9]), policy=contract).vmr_n, [401., 404., 408.])
    with pytest.raises(ValueError, match="do not cover"):
        select_boundary_fields(bc, targets([10]), policy=contract)
