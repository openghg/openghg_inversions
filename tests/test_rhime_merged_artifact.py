"""Acquisition artifacts preserve selectors and datasets without wrapper codecs."""

from dataclasses import fields
from types import SimpleNamespace
import json

import dask
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
from openghg_inversions.inversion_data._merged_artifact import selected_provenance


def merged_data():
    options = SiteOptions.from_inputs(
        sites=["TAC", "GOSAT"],
        averaging_period=["4h", "1h"],
        inlet=[slice("20m", "100m", 2), "column"],
        fp_height=["100m", None],
        instrument=["picarro", "satellite"],
        platform=["surface", "satellite"],
        obs_data_level=["L2", "L3"],
        met_model=["UKV", "UM"],
        max_level=[None, 8],
        time_resolved=[False, True],
    )
    site = xr.Dataset(
        {
            "mf": ("time", da.from_array([1.0, 2.0], chunks=1)),
            "fp": (("source", "time"), da.ones((2, 2), chunks=(1, 1))),
        },
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "source": ["a/b", "b"],
        },
        attrs={"scale": "WMO", "scientific_flag": True, "absent": None},
    )
    site.mf.attrs = {"units": "nmol mol-1", "description": "measured concentration"}
    flux = xr.Dataset({"flux": ("time", da.from_array([3.0, 4.0], chunks=1))}, coords={"time": site.time})
    flux.flux.attrs = {"units": "mol m-2 s-1", "time_period": "monthly"}
    bc = xr.Dataset(
        {f"vmr_{direction}": ("time", [1.0, 2.0]) for direction in "nesw"}, coords={"time": site.time}
    )
    return RhimeMergedData(
        site_data={"TAC": site, "GOSAT": site.assign_attrs(platform="satellite")},
        flux_data={"a/b": flux, "b": flux * 2},
        boundary_data=bc,
        site_options=options,
        split_by_sectors=True,
        acquisition={"stage": "acquired", "species": "ch4", "domain": "EUROPE", "use_bc": True},
    )


@pytest.mark.parametrize("suffix", ["zarr", "zarr.zip", "nc"])
def test_round_trip_all_selectors_scientific_attrs_and_lazy_arrays(tmp_path, suffix):
    original = merged_data()
    original.provenance["inputs"]["flux:a/b"] = {"store": "archive", "uuid": "abc", "dataversion": "v3"}
    original.provenance["openghg"] = {"version": "0.10", "commit": "abc123"}
    path = f"acquired.{suffix}"
    original.save(tmp_path, merged_data_name=path)
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        restored = RhimeMergedData.load(tmp_path, merged_data_name=path)
    try:
        assert not calls
        assert restored.site_options == original.site_options
        assert {field.name for field in fields(SiteOptions)} == set(restored.site_options.__dict__)
        assert restored.provenance == original.provenance
        assert restored.acquisition == original.acquisition
        assert restored.split_by_sectors
        assert tuple(restored.flux_data) == ("a/b", "b")
        assert not hasattr(restored, "fp_all")
        for site in original.sites:
            assert isinstance(restored.site_data[site].mf.data, da.Array)
            xr.testing.assert_identical(restored.site_data[site], original.site_data[site])
        for source in original.flux_data:
            xr.testing.assert_identical(restored.flux_data[source], original.flux_data[source])
        xr.testing.assert_identical(restored.boundary_data, original.boundary_data)
        assert original.site_data["TAC"].attrs["scientific_flag"] is True
        assert original.site_data["TAC"].mf.attrs["units"] == "nmol mol-1"
    finally:
        restored.close()


def test_dataset_boundary_borrows_arrays_and_explicit_adapter_isolated():
    original = merged_data()
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        legacy = original.to_legacy_fp_all()
        restored = RhimeMergedData.from_legacy_fp_all(legacy, original.site_options)
    assert not calls
    assert restored.site_data["TAC"] is original.site_data["TAC"]
    assert set(original.flux_data["a/b"].flux.data.dask) <= set(legacy[".flux"]["a/b"].data.flux.data.dask)
    legacy[".flux"]["a/b"].data["flux"] = legacy[".flux"]["a/b"].data.flux * 5
    np.testing.assert_array_equal(original.flux_data["a/b"].flux.compute(), [3.0, 4.0])


def test_selected_input_provenance_discards_arbitrary_metadata():
    wrapped = SimpleNamespace(
        metadata={"UUID": "u1", "dataversion": "v7", "secret": object()}, data=xr.Dataset()
    )
    assert selected_provenance(wrapped, "archive") == {"uuid": "u1", "dataversion": "v7", "store": "archive"}
    assert selected_provenance(xr.Dataset()) == {
        "uuid": "unknown",
        "dataversion": "unknown",
        "store": "unknown",
    }


def test_invalid_modern_inputs_do_not_reacquire(tmp_path, monkeypatch):
    monkeypatch.setattr(RhimeMergedData, "from_options", lambda **kw: pytest.fail("Unexpected reacquisition"))
    with pytest.raises((ValueError, FileNotFoundError)):
        RhimeMergedData.load(tmp_path, merged_data_name="missing.zarr")
    (tmp_path / "corrupt.nc").write_bytes(b"not an artifact")
    with pytest.raises((ValueError, OSError)):
        RhimeMergedData.load(tmp_path, merged_data_name="corrupt.nc")
    with pytest.raises(ValueError, match="Unsupported"):
        RhimeMergedData.load(tmp_path, merged_data_name="missing", output_format="pickle")
    # An alternate suffix must not rescue an explicitly requested missing file.
    merged_data().save(tmp_path, merged_data_name="existing.zarr")
    with pytest.raises((ValueError, FileNotFoundError)):
        RhimeMergedData.load(tmp_path, merged_data_name="existing.nc")


@pytest.mark.parametrize("change", ["version", "selector", "inventory", "legacy", "selector_scalar"])
def test_reject_malformed_schema(tmp_path, change):
    merged_data().save(tmp_path, merged_data_name="data.zarr")
    path = tmp_path / "data.zarr" / ".zattrs"
    attrs = json.loads(path.read_text())
    manifest = json.loads(attrs["manifest"])
    if change == "version":
        manifest["version"] = 99
    elif change == "selector":
        del manifest["site_options"]["inlet"]
    elif change == "selector_scalar":
        manifest["site_options"]["sites"] = "AB"
        manifest["site_options"]["averaging_period"] = "12"
    elif change == "inventory":
        manifest["sources"].append("absent")
    else:
        del attrs["manifest"]
    if change != "legacy":
        attrs["manifest"] = json.dumps(manifest)
    path.write_text(json.dumps(attrs))
    # Readers use consolidated Zarr metadata; refresh after deliberate corruption.
    import zarr

    zarr.consolidate_metadata(tmp_path / "data.zarr")
    with pytest.raises((ValueError, KeyError)):
        RhimeMergedData.load(tmp_path, merged_data_name="data.zarr")


def test_legacy_import_is_explicit_and_migrates(tmp_path):
    from openghg_inversions.inversion_data.serialise import _save_merged_data

    original = merged_data()
    # Old format uses source names as paths and requires slash-free labels.
    original.flux_data["a"] = original.flux_data.pop("a/b")
    legacy = original.to_legacy_fp_all()
    _save_merged_data(legacy, tmp_path, merged_data_name="old.zarr")
    with pytest.raises((KeyError, ValueError)):
        RhimeMergedData.load(tmp_path, merged_data_name="old.zarr")
    with pytest.raises(TypeError):
        RhimeMergedData.load_legacy(tmp_path, merged_data_name="old.zarr")
    selectors = SiteOptions.from_inputs(sites=original.sites, averaging_period="4h", inlet="column")
    with pytest.warns(DeprecationWarning, match="removed in 0.9"):
        converted = RhimeMergedData.load_legacy(
            tmp_path,
            merged_data_name="old.zarr",
            site_options=selectors,
            split_by_sectors=True,
            acquisition_stage="acquired",
        )
    converted.save(tmp_path, merged_data_name="new.zarr")
    restored = RhimeMergedData.load(tmp_path, merged_data_name="new.zarr")
    try:
        assert restored.site_options == selectors
        assert restored.acquisition["legacy_import"]
        assert restored.provenance["openghg"]["version"] == "unknown"
        xr.testing.assert_identical(restored.site_data["GOSAT"], original.site_data["GOSAT"])
    finally:
        restored.close()


def test_filtered_snapshot_cannot_be_saved_as_acquisition(tmp_path):
    original = merged_data()
    original.acquisition["stage"] = "filtered"
    with pytest.raises(ValueError, match="unfiltered"):
        original.save(tmp_path, merged_data_name="filtered.zarr")


def test_legacy_unknown_and_filtered_stage_are_not_relabelled(tmp_path):
    from openghg_inversions.inversion_data.serialise import _save_merged_data

    data = merged_data()
    legacy = data.to_legacy_fp_all()
    legacy.pop(".artifact_stage", None)
    legacy[".flux"] = {"a": legacy[".flux"]["a/b"]}
    selectors = SiteOptions.from_inputs(sites=data.sites, averaging_period="1h")
    _save_merged_data(legacy, tmp_path, merged_data_name="unknown.zarr")
    with pytest.warns(DeprecationWarning):
        unknown = RhimeMergedData.load_legacy(
            tmp_path, site_options=selectors, merged_data_name="unknown.zarr", split_by_sectors=True
        )
    assert unknown.acquisition["stage"] == "unknown"
    with pytest.raises(ValueError, match="acquisition_stage"):
        unknown.save(tmp_path, merged_data_name="not-acquired.zarr")
    legacy[".artifact_stage"] = "filtered"
    _save_merged_data(legacy, tmp_path, merged_data_name="filtered.zarr")
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="conflicts"):
        RhimeMergedData.load_legacy(
            tmp_path,
            site_options=selectors,
            merged_data_name="filtered.zarr",
            split_by_sectors=True,
            acquisition_stage="acquired",
        )


def test_fresh_save_load_scientific_preparation_equivalence(tmp_path, monkeypatch):
    """Real sensitivity/assembly equations consume the same fresh and saved data."""
    from openghg_inversions.basis.basis_functions import BasisFunctions
    from openghg_inversions.inversion_data import acquisition
    from openghg_inversions.rhime.preparation import (
        assemble_rhime_inputs,
        build_rhime_sensitivities,
        filter_rhime_observations,
    )

    times = np.array(["2020-01-01T02:00", "2020-01-01T14:00"], dtype="datetime64[ns]")
    coords = {"time": times, "lat": [0.0], "lon": [0.0]}
    site = xr.Dataset(
        {
            "release_lon": ("time", [0.0, 0.0]),
            "mf": ("time", [10.0, 12.0], {"units": "1e-9"}),
            "mf_error": ("time", [1.0, 2.0], {"units": "1e-9"}),
            "mf_repeatability": ("time", [0.5, 0.5], {"units": "1e-9"}),
            "mf_variability": ("time", [0.25, 0.25], {"units": "1e-9"}),
            "fp_x_flux": (("time", "lat", "lon"), np.array([2.0, 4.0]).reshape(2, 1, 1), {"units": "1e-9"}),
        },
        coords=coords,
    ).chunk(time=1)
    options = SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h")
    initial = RhimeMergedData(site_data={"TAC": site}, flux_data={}, site_options=options)
    monkeypatch.setattr(
        acquisition,
        "_retrieve_inversion_data_from_options",
        lambda **kw: (initial.to_legacy_fp_all(), ["TAC"]),
    )
    fresh = RhimeMergedData.from_options(
        site_options=options,
        species="ch4",
        domain="EUROPE",
        start_date="2020-01-01",
        end_date="2020-01-02",
        output_name="equivalence",
        flux_sources=["inventory"],
        use_bc=False,
        save_merged_data=True,
        merged_data_dir=tmp_path,
        merged_data_name="acquired.zarr",
    )
    loaded = RhimeMergedData.load(tmp_path, merged_data_name="acquired.zarr")
    basis_array = xr.DataArray([[1]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]}, name="basis")
    basis = BasisFunctions.from_flat_basis(
        basis_flat=basis_array,
        flux=xr.ones_like(basis_array, dtype=float).rename("flux"),
        operator_kwargs={"state_dim": "region"},
    )
    results = []
    try:
        for merged in (fresh, loaded):
            filtered = filter_rhime_observations(merged, filters=["daytime"])
            site_data = build_rhime_sensitivities(
                filtered, basis, domain="EUROPE", flux_sources=["inventory"], use_bc=False, multisector=False
            )
            results.append(
                assemble_rhime_inputs(
                    filtered, basis, site_data, domain="EUROPE", start_date="2020-01-01", use_bc=False
                )
            )
        xr.testing.assert_identical(results[0].inv_inputs, results[1].inv_inputs)
        xr.testing.assert_identical(results[0].site_metadata, results[1].site_metadata)
        assert loaded.site_data["TAC"].sizes["time"] == 2
        assert results[0].inv_inputs.mf.size == 1
    finally:
        loaded.close()


def test_available_openghg_revision_is_recorded(monkeypatch):
    import openghg
    from openghg_inversions.inversion_data._merged_artifact import software_provenance

    monkeypatch.setattr(openghg, "__version__", "test-version")
    monkeypatch.setattr(openghg, "__revisionid__", "test-commit")
    assert software_provenance() == {"version": "test-version", "commit": "test-commit"}


def test_fresh_acquisition_records_each_input_identity_and_saves_only_on_request(monkeypatch):
    from openghg_inversions.inversion_data import get_data

    options = SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h", platform="surface")
    time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    scenario = xr.Dataset(
        {
            "mf": ("time", [1.0], {"units": "1e-9"}),
            "mf_repeatability": ("time", [0.1]),
            "mf_variability": ("time", [0.2]),
        },
        coords={"time": time},
        attrs={"scale": "WMO"},
    )

    def wrapper(kind):
        return SimpleNamespace(
            data=xr.Dataset({"flux": ("time", [1.0])}, coords={"time": time}),
            metadata={
                "uuid": kind + "-uuid",
                "dataversion": "v2",
                "object_store": kind + "-store",
                "private": object(),
            },
        )

    monkeypatch.setattr(get_data, "get_obs_data", lambda **kw: wrapper("observations"))
    monkeypatch.setattr(get_data, "get_footprint_data", lambda **kw: wrapper("footprints"))
    monkeypatch.setattr(get_data, "get_flux_data", lambda **kw: {"inventory": wrapper("flux")})
    monkeypatch.setattr(get_data, "get_bc", lambda **kw: wrapper("boundary"))
    monkeypatch.setattr(get_data, "merged_scenario_data", lambda *args, **kw: scenario)
    monkeypatch.setattr(RhimeMergedData, "save", lambda *args, **kw: pytest.fail("Saving was not requested"))
    acquired = RhimeMergedData.from_options(
        site_options=options,
        species="ch4",
        domain="EUROPE",
        start_date="2020-01-01",
        end_date="2020-01-02",
        output_name="provenance",
        flux_sources=["inventory"],
    )
    for label, kind in (
        ("observations:TAC", "observations"),
        ("footprints:TAC", "footprints"),
        ("flux:inventory", "flux"),
        ("boundary", "boundary"),
    ):
        assert acquired.provenance["inputs"][label] == {
            "uuid": kind + "-uuid",
            "dataversion": "v2",
            "store": kind + "-store",
        }


@pytest.mark.parametrize("suffix", ["zarr", "zarr.zip", "nc"])
def test_sparse_lazy_payload_densifies_only_at_serialization(tmp_path, suffix):
    import sparse

    original = merged_data()
    payload = da.from_array(sparse.COO.from_numpy(np.array([1.0, 0.0])), chunks=1)
    original.site_data["TAC"]["mf"] = xr.DataArray(
        payload, dims="time", coords={"time": original.site_data["TAC"].time}
    )
    original.site_data["TAC"] = original.site_data["TAC"].assign_coords(release_lon=("time", payload))
    original.site_data["TAC"].release_lon.attrs["units"] = "degrees_east"
    original.save(tmp_path, merged_data_name="sparse." + suffix)
    assert isinstance(original.site_data["TAC"].mf.data._meta, sparse.COO)
    assert original.site_data["TAC"].release_lon.data is payload
    assert isinstance(payload._meta, sparse.COO)
    restored = RhimeMergedData.load(tmp_path, merged_data_name="sparse." + suffix)
    try:
        np.testing.assert_array_equal(restored.site_data["TAC"].mf.compute(), [1.0, 0.0])
        assert "release_lon" in restored.site_data["TAC"].coords
        assert restored.site_data["TAC"].release_lon.attrs == {"units": "degrees_east"}
        np.testing.assert_array_equal(restored.site_data["TAC"].release_lon.compute(), [1.0, 0.0])
    finally:
        restored.close()


def test_multiple_footprint_inlets_retain_each_available_identity(monkeypatch, tmp_path):
    import pandas as pd
    from openghg_inversions.inversion_data import getters

    times = pd.date_range("2020-01-01", periods=2, freq="h")
    obs = SimpleNamespace(
        data=xr.Dataset({"inlet": ("time", [10.0, 100.0])}, coords={"time": times}), metadata={"site": "TAC"}
    )
    footprints = {
        inlet: SimpleNamespace(
            data=xr.Dataset({"fp": ("time", [1.0, 2.0])}, coords={"time": times}),
            metadata={"uuid": inlet, "dataversion": "v1", "object_store": "archive"},
        )
        for inlet in ("10m", "100m")
    }
    monkeypatch.setattr(
        getters,
        "search_footprints",
        lambda **kw: SimpleNamespace(results=pd.DataFrame({"inlet": ["10m", "100m"]})),
    )
    monkeypatch.setattr(getters, "get_footprint", lambda **kw: footprints[kw["inlet"]])
    merged = getters.get_footprint_to_match(
        obs, domain="EUROPE", start_date="2020-01-01", end_date="2020-01-02", averaging_period="1h"
    )
    identifiers = selected_provenance(merged)
    assert identifiers == {
        "uuid": ["10m", "100m"],
        "dataversion": ["v1", "v1"],
        "store": ["archive", "archive"],
    }
    assert footprints["10m"].metadata["uuid"] == "10m"
    original = merged_data()
    original.provenance["inputs"]["footprints:TAC"] = identifiers
    original.save(tmp_path, merged_data_name="multiple.zarr")
    restored = RhimeMergedData.load(tmp_path, merged_data_name="multiple.zarr")
    try:
        assert restored.provenance == original.provenance
    finally:
        restored.close()


@pytest.mark.parametrize("stage", [None, "filtered"])
def test_explicit_legacy_adapter_cannot_certify_unknown_or_filtered_data(tmp_path, stage):
    original = merged_data()
    legacy = original.to_legacy_fp_all()
    legacy.pop(".artifact_stage", None)
    if stage is not None:
        legacy[".artifact_stage"] = stage
    record = RhimeMergedData.from_legacy_fp_all(legacy, original.site_options)
    assert record.acquisition["stage"] == (stage or "unknown")
    with pytest.raises(ValueError, match="acquired"):
        record.save(tmp_path, merged_data_name="invalid.zarr")
    if stage == "filtered":
        with pytest.raises(ValueError, match="conflicts"):
            RhimeMergedData.from_legacy_fp_all(
                legacy, original.site_options, acquisition={"stage": "acquired"}
            )


def test_direct_record_does_not_implicitly_certify_acquisition(tmp_path):
    original = merged_data()
    original.acquisition = {}
    with pytest.raises(ValueError, match="acquired"):
        original.save(tmp_path, merged_data_name="unknown.zarr")


@pytest.mark.parametrize("kwargs", [{"acquisition_stage": "invalid"}, {"output_format": "pickle"}])
def test_legacy_enum_validation_precedes_io(monkeypatch, kwargs):
    from openghg_inversions.inversion_data import acquisition

    monkeypatch.setattr(
        acquisition, "load_merged_data", lambda *a, **kw: pytest.fail("Invalid enum reached I/O")
    )
    with pytest.raises(ValueError):
        RhimeMergedData.load_legacy("unused", site_options=merged_data().site_options, **kwargs)


def test_filtered_adapter_round_trip_preserves_phase():
    record = merged_data()
    record.acquisition["stage"] = "filtered"
    legacy = record.to_legacy_fp_all()
    with pytest.raises(ValueError, match="conflicts"):
        RhimeMergedData.from_legacy_fp_all(legacy, record.site_options, acquisition={"stage": "acquired"})


@pytest.mark.parametrize("suffix", ["nc", "zarr", "zarr.zip"])
def test_save_computes_shared_site_and_flux_source_once(tmp_path, suffix):
    """One serialization boundary computes shared graph dependencies together."""
    calls = []

    @dask.delayed
    def source():
        calls.append("computed")
        return np.array([1.0, 2.0])

    shared = da.from_delayed(source(), shape=(2,), dtype=float)
    original = RhimeMergedData(
        site_data={
            "TAC": xr.Dataset({"mf": ("time", shared)}),
            "BSD": xr.Dataset({"mf": ("time", shared + 10)}),
        },
        flux_data={"total": xr.Dataset({"flux": ("time", shared * 2)})},
        site_options=SiteOptions.from_inputs(sites=["TAC", "BSD"], averaging_period="1h"),
        acquisition={"stage": "acquired"},
    )
    with dask.config.set(scheduler="synchronous"):
        original.save(tmp_path, merged_data_name=f"shared.{suffix}")
    assert calls == ["computed"]
    assert original.site_data["TAC"].mf.data is shared
    restored = RhimeMergedData.load(tmp_path, merged_data_name=f"shared.{suffix}")
    try:
        np.testing.assert_array_equal(restored.site_data["TAC"].mf.values, [1.0, 2.0])
        np.testing.assert_array_equal(restored.site_data["BSD"].mf.values, [11.0, 12.0])
        np.testing.assert_array_equal(restored.flux_data["total"].flux.values, [2.0, 4.0])
    finally:
        restored.close()
    assert calls == ["computed"]


@pytest.mark.parametrize("selected_version", [None, "v2"])
def test_catalog_latest_version_requires_retrieval_context(selected_version):
    wrapped = SimpleNamespace(metadata={"latest_version": "v7", "versions": ["v2", "v7"]})
    if selected_version is not None:
        wrapped._version = selected_version
    assert selected_provenance(wrapped)["dataversion"] == (selected_version or "unknown")
    assert selected_provenance(wrapped, requested_version="latest")["dataversion"] == (
        selected_version or "v7"
    )


@pytest.mark.parametrize("kind", ["flux", "footprints-single", "footprints-multiple", "older"])
def test_native_openghg_public_retrieval_preserves_selected_version(monkeypatch, kind):
    """Exercise real search-result selection and public typed-wrapper reconstruction."""
    import pandas as pd
    import openghg.retrieve
    from openghg.dataobjects import SearchResults
    from openghg.dataobjects import _basedata
    from openghg_inversions.inversion_data import getters

    times = pd.date_range("2020-01-01", periods=2, freq="h")
    loaded_versions = []
    metadata = {
        "uuid": "native-uuid",
        "object_store": "archive",
        "latest_version": "v7",
        "versions": ["v2", "v7"],
        "data_type": "flux" if kind == "flux" else "footprints",
    }

    def get_dataset(*, version):
        loaded_versions.append(version)
        variable = "flux" if kind == "flux" else "fp"
        dataset = xr.Dataset({variable: ("time", [1.0, 2.0])}, coords={"time": times})
        dataset[variable].attrs["units"] = "mol m-2 s-1" if kind == "flux" else "mol mol-1 / (mol m-2 s-1)"
        return dataset

    monkeypatch.setattr(_basedata, "get_datasource", lambda **kw: SimpleNamespace(get_data=get_dataset))
    def search(**kwargs):
        selected = dict(metadata)
        if kind == "footprints-multiple":
            height = kwargs["inlet"].removesuffix("m")
            selected["uuid"] = f"native-{height}"
            selected["latest_version"] = "v2" if height == "10" else "v7"
        return SearchResults(metadata={selected["uuid"]: selected})

    monkeypatch.setattr(openghg.retrieve, "search", search)
    if kind == "older":
        result = SearchResults(metadata={"native-uuid": dict(metadata)}).retrieve_all(version="v2")
        expected = "v2"
        assert selected_provenance(result, requested_version="latest")["dataversion"] == "v2"
    elif kind == "flux":
        monkeypatch.setattr(getters, "adjust_flux_start_date", lambda *args: "2020-01-01")
        result = getters.get_flux_data(
            sources=["inventory"],
            species="ch4",
            domain="EUROPE",
            store="archive",
            start_date="2020-01-01",
            end_date="2020-01-02",
        )["inventory"]
        expected = "v7"
    else:
        inlets = [10.0, 100.0] if kind.endswith("multiple") else [10.0, 10.0]
        obs = SimpleNamespace(
            data=xr.Dataset({"inlet": ("time", inlets)}, coords={"time": times}),
            metadata={"site": "TAC"},
        )
        monkeypatch.setattr(
            getters,
            "search_footprints",
            lambda **kw: SimpleNamespace(results=pd.DataFrame({"inlet": ["10m", "100m"]})),
        )
        result = getters.get_footprint_to_match(
            obs,
            domain="EUROPE",
            store="archive",
            start_date="2020-01-01",
            end_date="2020-01-02",
            averaging_period="1h",
        )
        expected = ["v2", "v7"] if kind.endswith("multiple") else "v7"
    assert loaded_versions == (expected if kind.endswith("multiple") else [expected])
    provenance = selected_provenance(result)
    assert provenance["dataversion"] == expected
    if kind == "footprints-multiple":
        assert list(zip(provenance["uuid"], provenance["dataversion"], strict=True)) == [
            ("native-10", "v2"), ("native-100", "v7")
        ]
    assert "dataversion" not in metadata


def test_preparation_binding_preserves_unknown_historical_facts():
    original = merged_data()
    original.acquisition = {"stage": "unknown", "species": "unknown", "domain": None}
    before = dict(original.acquisition)
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        original.validate_for_preparation(
            species="co2", domain="USA", start_date="2021-01-01",
            end_date="2021-02-01", split_by_sectors=True,
        )
    assert original.acquisition == before
    assert original.site_data["TAC"].sizes["time"] == 2
    assert not calls


def test_owned_site_selection_aligns_and_isolates_metadata_without_computing():
    from copy import deepcopy

    original = merged_data()
    original.provenance["inputs"]["observations:GOSAT"]["uuid"] = ["one", "two"]
    provenance = deepcopy(original.provenance)
    acquisition = dict(original.acquisition)
    retained = original.site_data["GOSAT"].isel(time=slice(1, None))
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        selected = original.with_site_data({"GOSAT": retained}, stage="filtered")
    assert not calls
    assert selected.sites == ("GOSAT",)
    assert selected.site_options == original.site_options.select_indices([1])
    assert selected.site_data["GOSAT"] is retained
    assert selected.flux_data["a/b"] is original.flux_data["a/b"]
    assert selected.boundary_data is original.boundary_data
    assert set(selected.provenance["inputs"]) == {
        "observations:GOSAT", "footprints:GOSAT", "flux:a/b", "flux:b", "boundary",
    }
    assert selected.acquisition == {**acquisition, "stage": "filtered"}
    selected.provenance["inputs"]["observations:GOSAT"]["uuid"].append("new")
    selected.provenance["openghg"]["version"] = "new"
    selected.acquisition["domain"] = "different"
    assert original.provenance == provenance
    assert original.acquisition == acquisition
    assert original.sites == ("TAC", "GOSAT")
    assert original.site_data["GOSAT"].sizes["time"] == 2


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_compatibility_preparation_binds_actual_reopened_acquisition(monkeypatch, tmp_path, suffix):
    from openghg_inversions.inversion_data import prepare_rhime_inputs
    from openghg_inversions.rhime import preparation

    original = merged_data()
    original.acquisition.update(start_date="2020-01-01", end_date="2020-02-01")
    original.save(tmp_path, merged_data_name=f"reused.{suffix}")
    reached = []

    class ReachedFiltering(Exception):
        pass

    def filtering(merged, **kwargs):
        reached.append(merged)
        assert merged.site_options == original.site_options
        assert merged.acquisition == original.acquisition
        raise ReachedFiltering

    monkeypatch.setattr(preparation, "filter_rhime_observations", filtering)
    request = dict(
        species="ch4", domain="EUROPE", sites=["TAC"], averaging_period="1h",
        start_date="2020-01-01", end_date="2020-02-01", output_name="reuse",
        flux_sources=["a/b", "b"], split_by_sectors=True,
        reload_merged_data=True, merged_data_dir=tmp_path, merged_data_name=f"reused.{suffix}",
    )
    try:
        with pytest.warns(DeprecationWarning), pytest.raises(ReachedFiltering):
            prepare_rhime_inputs(**request)
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="end_date"):
            prepare_rhime_inputs(**{**request, "end_date": "2020-01-02"})
        assert len(reached) == 1
    finally:
        for merged in reached:
            merged.close()


@pytest.mark.parametrize("fact", ["species", "domain", "start_date", "end_date"])
def test_preparation_binding_rejects_malformed_known_facts(fact):
    original = merged_data()
    original.acquisition[fact] = True
    with pytest.raises(ValueError, match=fact):
        original.validate_for_preparation(
            species="ch4", domain="EUROPE", start_date="2020-01-01",
            end_date="2020-02-01", split_by_sectors=True,
        )


@pytest.mark.parametrize("fact", ["start_date", "end_date"])
@pytest.mark.parametrize("recorded,requested", [
    ("2020-01-01", "2020-01-01T00:00:00Z"),
    ("2020-01-01T00:00:00Z", "2020-01-01"),
    ("2020-01-01", "2020-01-01T01:00:00+01:00"),
])
def test_preparation_binding_compares_utc_instants(fact, recorded, requested):
    original = merged_data()
    original.acquisition[fact] = recorded
    request = dict(species="ch4", domain="EUROPE", start_date="2020-01-01",
                   end_date="2020-02-01", split_by_sectors=True)
    request[fact] = requested
    original.validate_for_preparation(**request)
    assert original.acquisition[fact] == recorded
    request[fact] = "2020-01-01T00:00:00+01:00"
    with pytest.raises(ValueError, match=fact):
        original.validate_for_preparation(**request)


def test_site_selection_cannot_relabel_filtered_data_as_acquired():
    original = merged_data()
    filtered = original.with_site_data(original.site_data, stage="filtered")
    with pytest.raises(ValueError, match="acquired"):
        filtered.with_site_data(filtered.site_data, stage="acquired")
    assert filtered.acquisition["stage"] == "filtered"
