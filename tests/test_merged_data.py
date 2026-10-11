"""Borrowed acquisition datasets preserve selectors and selected provenance."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inversion_data import AcquisitionFacts, InputProvenance, MergedDataProvenance, RhimeMergedData, SiteOptions
from openghg_inversions.inversion_data._provenance import selected_provenance


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
        acquisition=AcquisitionFacts(species="ch4", domain="EUROPE", use_bc=True),
    )


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
    assert selected_provenance(wrapped, "archive") == InputProvenance("archive", "u1", "v7")
    assert selected_provenance(xr.Dataset()) == InputProvenance()


def test_selected_input_provenance_does_not_interpret_dataset_metadata_variables():
    dataset = xr.Dataset({"metadata": 1}, attrs={"uuid": "selected", "dataversion": "v2"})
    assert selected_provenance(dataset) == InputProvenance(uuid="selected", dataversion="v2")


def test_provenance_owns_input_mappings():
    inputs = {"TAC": InputProvenance(uuid="selected")}
    provenance = MergedDataProvenance(observations=inputs)
    inputs["TAC"] = InputProvenance(uuid="changed")
    assert provenance.observations["TAC"].uuid == "selected"


def test_legacy_projection_retains_selected_flux_and_boundary_identities():
    original = merged_data()
    original.provenance = replace(
        original.provenance,
        flux={"a/b": InputProvenance("archive", "flux-one", "v3"), "b": InputProvenance()},
        boundary=InputProvenance("boundary-store", "boundary-one", "v2"),
    )
    legacy = original.to_legacy_fp_all()
    assert legacy[".flux"]["a/b"].metadata == {
        "data_type": "flux", "store": "archive", "uuid": "flux-one", "dataversion": "v3",
    }
    restored = RhimeMergedData.from_legacy_fp_all(legacy, original.site_options)
    assert restored.provenance.flux == original.provenance.flux
    assert restored.provenance.boundary == original.provenance.boundary
    assert restored.provenance.openghg_version == "unknown"


def test_available_openghg_revision_is_recorded(monkeypatch):
    import openghg

    monkeypatch.setattr(openghg, "__version__", "test-version")
    monkeypatch.setattr(openghg, "__revisionid__", "test-commit")
    provenance = MergedDataProvenance.from_retrieval(observations={}, footprints={}, flux={})
    assert provenance.openghg_version == "test-version"
    assert provenance.openghg_commit == "test-commit"


def test_fresh_acquisition_records_each_input_identity(monkeypatch):
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
    acquired = RhimeMergedData.from_options(
        site_options=options,
        species="ch4",
        domain="EUROPE",
        start_date="2020-01-01",
        end_date="2020-01-02",

        flux_sources=["inventory"],
    )
    for identity, kind in (
        (acquired.provenance.observations["TAC"], "observations"),
        (acquired.provenance.footprints["TAC"], "footprints"),
        (acquired.provenance.flux["inventory"], "flux"),
        (acquired.provenance.boundary, "boundary"),
    ):
        assert identity == InputProvenance(kind + "-store", kind + "-uuid", "v2")


@pytest.mark.parametrize("selected_version", [None, "v2"])
def test_catalog_latest_version_requires_retrieval_context(selected_version):
    wrapped = SimpleNamespace(metadata={"latest_version": "v7", "versions": ["v2", "v7"]})
    if selected_version is not None:
        wrapped._version = selected_version
    assert selected_provenance(wrapped).dataversion == (selected_version or "unknown")
    assert selected_provenance(wrapped, requested_version="latest").dataversion == (
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
        assert selected_provenance(result, requested_version="latest").dataversion == "v2"
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
    assert provenance.dataversion == (tuple(expected) if isinstance(expected, list) else expected)
    if kind == "footprints-multiple":
        assert list(zip(provenance.uuid, provenance.dataversion, strict=True)) == [
            ("native-10", "v2"), ("native-100", "v7")
        ]
    assert "dataversion" not in metadata


def test_preparation_binding_preserves_unknown_historical_facts():
    original = merged_data()
    original.acquisition = AcquisitionFacts(species="unknown")
    before = original.acquisition
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
    original.provenance = replace(
        original.provenance,
        observations={**original.provenance.observations, "GOSAT": InputProvenance(uuid=("one", "two"))},
    )
    provenance = deepcopy(original.provenance)
    acquisition = original.acquisition
    retained = original.site_data["GOSAT"].isel(time=slice(1, None))
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        selected = original.with_site_data({"GOSAT": retained})
    assert not calls
    assert selected.sites == ("GOSAT",)
    assert selected.site_options == original.site_options.select_indices([1])
    assert selected.site_data["GOSAT"] is retained
    assert selected.flux_data["a/b"] is original.flux_data["a/b"]
    assert selected.boundary_data is original.boundary_data
    assert tuple(selected.provenance.observations) == ("GOSAT",)
    assert tuple(selected.provenance.footprints) == ("GOSAT",)
    assert selected.provenance.flux == original.provenance.flux
    assert selected.provenance.boundary == original.provenance.boundary
    assert selected.provenance.flux is not original.provenance.flux
    assert selected.acquisition == acquisition
    with pytest.raises(FrozenInstanceError):
        selected.provenance.observations["GOSAT"].uuid = ("new",)
    selected.provenance = replace(
        selected.provenance, openghg_version="new", observations={"GOSAT": InputProvenance(uuid="new")},
    )
    selected.acquisition = replace(selected.acquisition, domain="different")
    assert original.provenance == provenance
    assert original.acquisition == acquisition
    assert original.sites == ("TAC", "GOSAT")
    assert original.site_data["GOSAT"].sizes["time"] == 2


@pytest.mark.parametrize("fact", ["start_date", "end_date"])
@pytest.mark.parametrize("recorded,requested", [
    ("2020-01-01", "2020-01-01T00:00:00Z"),
    ("2020-01-01T00:00:00Z", "2020-01-01"),
    ("2020-01-01", "2020-01-01T01:00:00+01:00"),
])
def test_preparation_binding_compares_utc_instants(fact, recorded, requested):
    original = merged_data()
    original.acquisition = replace(original.acquisition, **{fact: recorded})
    request = dict(species="ch4", domain="EUROPE", start_date="2020-01-01",
                   end_date="2020-02-01", split_by_sectors=True)
    request[fact] = requested
    original.validate_for_preparation(**request)
    assert getattr(original.acquisition, fact) == recorded
    request[fact] = "2020-01-01T00:00:00+01:00"
    with pytest.raises(ValueError, match=fact):
        original.validate_for_preparation(**request)


def test_default_provenance_is_false_and_selected_identities_are_true():
    assert not MergedDataProvenance()
    assert not merged_data().provenance
    assert not InputProvenance()
    assert MergedDataProvenance(openghg_version="0.10")
    assert MergedDataProvenance(observations={"TAC": InputProvenance(uuid="selected")})


@pytest.mark.parametrize("version", ["unknown", "known"])
@pytest.mark.parametrize("boundary", [None, InputProvenance(uuid="selected-bc")])
def test_partial_provenance_defaults_missing_inputs_and_preserves_supplied_identities(version, boundary):
    original = merged_data()
    observation = InputProvenance(uuid="selected-observation")
    flux = InputProvenance(store="archive")
    supplied = MergedDataProvenance(
        openghg_version=version,
        observations={"TAC": observation},
        flux={"a/b": flux},
        boundary=boundary,
    )
    with Callback(pretask=lambda *args: pytest.fail("provenance defaulting computed borrowed arrays")):
        merged = replace(original, provenance=supplied)
        selected = merged.with_site_data({"TAC": original.site_data["TAC"]})
        legacy = selected.to_legacy_fp_all()
    assert merged.provenance.openghg_version == version
    assert merged.provenance.observations == {"TAC": observation, "GOSAT": InputProvenance()}
    assert merged.provenance.footprints == {site: InputProvenance() for site in original.sites}
    assert merged.provenance.flux == {"a/b": flux, "b": InputProvenance()}
    assert merged.provenance.boundary == (boundary if boundary is not None else InputProvenance())
    assert merged.provenance.observations["TAC"] is observation
    assert merged.provenance.flux["a/b"] is flux
    assert merged.site_data["TAC"] is original.site_data["TAC"]
    assert selected.provenance.observations == {"TAC": observation}
    assert legacy[".flux"]["a/b"].metadata["store"] == "archive"
    assert supplied.footprints == {}
    assert supplied.observations == {"TAC": observation}


@pytest.mark.parametrize("supplied", [
    MergedDataProvenance(),
    MergedDataProvenance(openghg_version="known"),
    MergedDataProvenance(observations={"TAC": InputProvenance()}),
])
def test_missing_provenance_defaults_consistently_without_inferring_dataset_identities(supplied):
    original = merged_data()
    original.flux_data["a/b"] = original.flux_data["a/b"].assign_attrs(uuid="unconfirmed")
    original.boundary_data = original.boundary_data.assign_attrs(uuid="unconfirmed-bc")
    merged = replace(original, provenance=supplied)
    assert merged.provenance.observations == {site: InputProvenance() for site in original.sites}
    assert merged.provenance.footprints == {site: InputProvenance() for site in original.sites}
    assert merged.provenance.flux == {source: InputProvenance() for source in original.flux_data}
    assert merged.provenance.boundary == InputProvenance()
    assert merged.provenance.openghg_version == supplied.openghg_version
    assert bool(merged.provenance) == bool(supplied)


@pytest.mark.parametrize("kind", ["observations", "footprints", "flux", "boundary"])
@pytest.mark.parametrize("identity", [InputProvenance(), InputProvenance(uuid="selected")])
def test_provenance_rejects_unexpected_inputs_even_when_identities_are_unknown(kind, identity):
    original = merged_data()
    if kind == "boundary":
        original.boundary_data = None
        supplied = MergedDataProvenance(boundary=identity)
    else:
        supplied = MergedDataProvenance(**{kind: {"unexpected": identity}})
    with pytest.raises(ValueError, match="inputs absent from the merged data"):
        replace(original, provenance=supplied)


@pytest.mark.parametrize("uuid,known", [
    ((), False), (("unknown", "unknown"), False), (("unknown", "selected"), True),
])
def test_input_provenance_truthiness_uses_known_identifiers(uuid, known):
    assert bool(InputProvenance(uuid=uuid)) is known
