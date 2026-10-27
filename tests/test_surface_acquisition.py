"""Explicit surface convolution and the bounded acquisition opt-in."""

import json
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import MergedData, SiteOptions, prepare_surface_site
from openghg_inversions.inversion_data import acquisition, surface


@pytest.fixture
def inputs():
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    obs = xr.Dataset(
        {
            "co2": ("time", [400.0, 401.0, 402.0], {"units": "ppm"}),
            "co2_repeatability": ("time", [0.1, 0.2, 0.3], {"units": "ppm"}),
            "co2_variability": ("time", [0.3, 0.4, 0.5], {"units": "ppm"}),
            "co2_number_of_observations": ("time", [2, 3, 4]),
        },
        coords={"time": times},
        attrs={"scale": "WMO", "platform": "surface", "sampling_period": 3600},
    )
    fp = xr.Dataset(
        {
            "fp": (("lat", "lon", "time"), np.ones((1, 1, 3)), {"units": "s m2 / mol"}),
            "pblh": ("time", [100.0, 200.0, 300.0]),
        },
        coords={"lat": [0.0], "lon": [0.0], "time": times},
        attrs={"model": "NAME", "met_model": "UKV"},
    )
    flux = xr.Dataset(
        {"flux": (("lat", "lon", "time"), np.array([[[1.0, 2.0, 3.0]]]) * 1.0e-6, {"units": "mol / m2 / s"})},
        coords=fp.coords,
    )
    return obs, fp, flux


def boundary_inputs(fp):
    """Supply schema-valid directional inputs for boundary preparation tests."""
    fp = fp.assign_coords(height=[10.0])
    bc = xr.Dataset(coords=fp.coords, attrs={"time_period": "1 hour"})
    for direction in "nesw":
        edge = "lon" if direction in "ns" else "lat"
        values = xr.DataArray(
            np.full((1, fp.sizes[edge], fp.sizes["time"]), .25),
            dims=("height", edge, "time"),
        )
        fp[f"particle_locations_{direction}"] = values.assign_attrs(units="1")
        bc[f"vmr_{direction}"] = (values * 1600.).assign_attrs(units="1e-06 mol/mol")
    return fp, bc


def test_independent_integrated_sources_units_filters_and_borrowing(inputs):
    obs, fp, flux = inputs
    before = [v.copy(deep=True) for v in inputs]
    coarse = flux.isel(time=[0]).copy(deep=True)
    coarse.flux.attrs["time_period"] = "monthly"
    coarse["flux"] = coarse.flux * -2
    coarse.flux.attrs = dict(flux.flux.attrs, time_period="monthly")
    result = prepare_surface_site(
        obs,
        fp,
        {"hourly": flux, "monthly": coarse},
        species="co2",
        transport={"hourly": "integrated", "monthly": "integrated"},
        output_units="ppb",
    )
    np.testing.assert_allclose(result.fp_x_flux.squeeze(), [-1000.0, 0.0, 1000.0], atol=1e-10)
    xr.testing.assert_allclose(result.fp_x_flux, result.fp_x_flux_sectoral.sum("source", keep_attrs=True))
    np.testing.assert_allclose(result.mf, [400000.0, 401000.0, 402000.0])
    np.testing.assert_allclose(result.mf_repeatability, [100.0, 200.0, 300.0])
    np.testing.assert_equal(result.mf_number_of_observations.values, obs.co2_number_of_observations.values)
    xr.testing.assert_identical(result.pblh, fp.pblh)
    assert result.attrs["footprint_transport_model"] == "NAME"
    for original, saved in zip(inputs, before, strict=True):
        xr.testing.assert_identical(original, saved)
    reversed_result = prepare_surface_site(
        obs,
        fp,
        {"monthly": coarse, "hourly": flux},
        species="co2",
        transport={"monthly": "integrated", "hourly": "integrated"},
        output_units="ppb",
    )
    xr.testing.assert_identical(result, reversed_result)


def test_integrated_stays_lazy(inputs):
    obs, fp, flux = inputs
    flux = flux.copy()
    flux["flux"] = flux.flux.copy(data=da.from_array(flux.flux.values, chunks=(1, 1, 2)))
    with Callback(pretask=lambda *args: pytest.fail("producer computed borrowed payload")):
        result = prepare_surface_site(
            obs, fp.chunk({"time": 2}), {"a": flux}, species="co2", transport={"a": "integrated"}
        )
    assert isinstance(result.fp_x_flux.data, da.Array)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("entry_point", ["surface", "acquisition"])
def test_exact_alignment_reuses_payloads(inputs, monkeypatch, lazy, entry_point):
    obs, fp, flux = inputs
    flux = flux.assign(extra=flux.flux * 2).assign_attrs(identity="native flux")
    if lazy:
        fp = fp.chunk({"time": 2})
        flux = flux.chunk({"time": 2})
    align = xr.align
    aligned_payloads = []

    def observe_alignment(*arrays, **kwargs):
        result = align(*arrays, **kwargs)
        if kwargs.get("join") == "exact" and (
            (len(arrays) == 2 and isinstance(arrays[0], xr.Dataset) and "mf" in arrays[0])
            or all(isinstance(array, xr.DataArray) and array.name == "flux" for array in arrays)
        ):
            for before, after in zip(arrays, result, strict=True):
                names = before.data_vars if isinstance(before, xr.Dataset) else [None]
                for name in names:
                    original = before[name].data if name is not None else before.data
                    aligned = after[name].data if name is not None else after.data
                    if isinstance(original, da.Array):
                        assert aligned.chunks == original.chunks
                        assert aligned.__dask_keys__() == original.__dask_keys__()
                    else:
                        assert np.shares_memory(original, aligned)
                    aligned_payloads.append(name)
        return result

    monkeypatch.setattr(xr, "align", observe_alignment)
    with Callback(pretask=lambda *args: pytest.fail("alignment computed borrowed payload")):
        if entry_point == "surface":
            result = prepare_surface_site(
                obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"}
            )
        else:
            wrappers = {
                source: SimpleNamespace(
                    data=flux.assign(diagnostic=xr.DataArray([1.], dims="diagnostic_axis", coords={"diagnostic_axis": [i]})),
                    metadata={"uuid": source},
                )
                for i, source in enumerate(("a", "b"))
            }
            monkeypatch.setattr(acquisition, "get_flux_data", lambda **kw: wrappers)
            monkeypatch.setattr(
                acquisition, "get_obs_data", lambda **kw: SimpleNamespace(data=obs, metadata={})
            )
            monkeypatch.setattr(
                acquisition, "get_footprint_data", lambda **kw: SimpleNamespace(data=fp, metadata={})
            )
            merged = MergedData.from_options(
                species="co2",
                site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
                domain="EUROPE", start_date="2020-01-01", end_date="2020-01-02",
                flux_sources=["a", "b"], use_bc=False,
                surface_transport={"a": "integrated", "b": "integrated"},
            )
            result = merged.site_data["MHD"]
    if entry_point == "acquisition":
        for source in wrappers:
            xr.testing.assert_identical(merged.flux_data[source], wrappers[source].data)
            retained = merged.flux_data[source].flux.data
            if lazy:
                assert retained.chunks == flux.flux.data.chunks
                assert retained.__dask_keys__() == flux.flux.data.__dask_keys__()
            else:
                assert np.shares_memory(retained, flux.flux.data)
            assert wrappers[source].data.flux.data is flux.flux.data
            assert wrappers[source].metadata == {"uuid": source}
            assert merged.provenance.flux[source].uuid == source
    assert aligned_payloads
    np.testing.assert_allclose(result.mf_mod, np.array([1., 2., 3.]) * (2 if entry_point == "acquisition" else 1))


def test_provider_response_spatial_mismatch_remains_an_error(inputs, monkeypatch):
    obs, fp, flux = inputs
    align, integrated, resolved = surface._surface_operations()
    monkeypatch.setattr(
        surface, "_surface_operations",
        lambda: (align, lambda *args: integrated(*args).assign_coords(lat=[1.]), resolved),
    )
    with pytest.raises(ValueError, match="matching spatial and observed timestamps"):
        prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"})


def test_per_source_footprints_represent_distinct_physical_contributions(inputs):
    obs, fp, flux = inputs
    other = fp.copy()
    other["fp"] = 3 * fp.fp
    other.fp.attrs = fp.fp.attrs
    result = prepare_surface_site(
        obs,
        {"a": fp, "b": other},
        {"a": flux, "b": flux},
        species="co2",
        transport={"a": "integrated", "b": "integrated"},
        site_footprint_data=fp.assign(pblh=fp.pblh + 500),
    )
    np.testing.assert_allclose(result.mf_mod, [4.0, 8.0, 12.0])
    np.testing.assert_allclose(result.pblh, [600.0, 700.0, 800.0])
    reversed_result = prepare_surface_site(
        obs,
        {"b": other, "a": fp},
        {"b": flux, "a": flux},
        species="co2",
        transport={"b": "integrated", "a": "integrated"},
        site_footprint_data=fp.assign(pblh=fp.pblh + 500),
    )
    xr.testing.assert_identical(result, reversed_result)


def resolved_inputs(inputs, representation="split"):
    obs, fp, flux = inputs
    fp = fp.assign_coords(H_back=[0, 1])
    fp["fp_time_resolved"] = xr.DataArray(
        np.ones((1, 1, 3, 2)), dims=("lat", "lon", "time", "H_back"), attrs={"units": fp.fp.attrs["units"]}
    )
    fp["fp_residual"] = fp.fp * 5
    fp.fp_residual.attrs = fp.fp.attrs
    times = pd.date_range("2019-12-01", "2020-02-01", freq="h", inclusive="left")
    flux = xr.Dataset(
        {"flux": (("lat", "lon", "time"), np.full((1, 1, len(times)), 2.0e-6), {"units": "mol / m2 / s"})},
        coords={"lat": [0.0], "lon": [0.0], "time": times},
    )
    if representation == "legacy":
        high_res = xr.concat(
            [fp.fp_time_resolved, fp.fp_residual.expand_dims(H_back=[24])], dim="H_back"
        )
        fp = fp.drop_dims("H_back").drop_vars("fp_residual")
        fp["fp_HiTRes"] = high_res
    return obs, fp, flux


@pytest.mark.parametrize("species", ["co2", "o2"])
def test_actual_resolved_kernel_counts_residual_once(inputs, species, compatible_provider):
    obs, fp, flux = resolved_inputs(inputs)
    obs = obs.rename(
        {
            "co2": "mf",
            "co2_repeatability": "mf_repeatability",
            "co2_variability": "mf_variability",
            "co2_number_of_observations": "mf_number_of_observations",
        }
    )
    result = prepare_surface_site(
        obs,
        fp,
        {"resolved": flux, "integrated": flux},
        species=species,
        transport={"resolved": "time_resolved", "integrated": "integrated"},
    )
    np.testing.assert_allclose(result.mf_mod_sectoral.sel(source="resolved"), 14.0)
    np.testing.assert_allclose(result.mf_mod, 16.0)
    assert result.attrs["surface_provider_policy_revision"] == "openghg-time-resolved-v1"



def test_resolved_provider_borrows_lazy_flux(inputs, compatible_provider):
    obs, fp, flux = resolved_inputs(inputs)
    flux = flux.chunk({"time": 744})
    borrowed = flux.flux.data
    before = flux.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        result = prepare_surface_site(
            obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"}
        )
    assert not tasks
    assert isinstance(result.fp_x_flux.data, da.Array)
    assert flux.flux.data is borrowed
    np.testing.assert_allclose(result.mf_mod.compute(scheduler="synchronous"), 14.)
    xr.testing.assert_identical(flux, before)

@pytest.mark.parametrize("legacy_units", [None, "", "explicit"])
@pytest.mark.parametrize("lazy", [False, True])
def test_legacy_resolved_units_inherit_without_mutating_inputs(inputs, compatible_provider, legacy_units, lazy):
    obs, fp, flux = resolved_inputs(inputs, "legacy")
    if legacy_units == "explicit":
        # The resolved field's own units take precedence over integrated metadata.
        fp.fp.attrs["units"] = "m"
    elif legacy_units is None:
        fp.fp_HiTRes.attrs.pop("units")
    else:
        fp.fp_HiTRes.attrs["units"] = legacy_units
    if lazy:
        fp, flux = fp.chunk({"time": 2}), flux.chunk({"time": 744})
    borrowed = fp.fp_HiTRes.data
    metadata = dict(fp.fp_HiTRes.attrs)
    with Callback(pretask=lambda *args: pytest.fail("preparation computed borrowed payload")):
        result = prepare_surface_site(
            obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"}
        )
    assert fp.fp_HiTRes.data is borrowed
    assert fp.fp_HiTRes.attrs == metadata
    if lazy:
        assert isinstance(result.mf_mod.data, da.Array)
    np.testing.assert_allclose(result.mf_mod.compute(scheduler="synchronous"), [14., 14., 14.])


def test_resolved_rejects_missing_history_and_incomplete_residual_month(inputs, compatible_provider):
    obs, fp, flux = resolved_inputs(inputs)
    for sliced in [flux.sel(time=slice("2020-01-01", None)), flux.sel(time=slice(None, "2020-01-02"))]:
        with pytest.raises(ValueError, match="Inadequate flux coverage"):
            prepare_surface_site(obs, fp, {"a": sliced}, species="co2", transport={"a": "time_resolved"})


@pytest.mark.parametrize(
    "change, message",
    [
        ("platform", "continuous surface"),
        ("species", "only CO2 and O2"),
        ("mode", "transport modes"),
        ("labels", "one mode per flux source"),
        ("coverage", "Inadequate flux coverage"),
    ],
)
def test_invalid_inputs_fail_clearly(inputs, change, message):
    obs, fp, flux = inputs
    species, mode, transport = "co2", "integrated", None
    if change == "platform":
        obs.attrs["platform"] = "flask"
    elif change == "species":
        species = "ch4"
    elif change == "mode":
        mode = "auto"
    elif change == "labels":
        transport = {"other": mode}
    elif change == "coverage":
        flux = flux.isel(time=slice(1, None))
    with pytest.raises(ValueError, match=message):
        prepare_surface_site(obs, fp, {"a": flux}, species=species, transport=transport or {"a": mode})




def test_opt_in_reuses_getters_retention_and_provenance(inputs, monkeypatch):
    obs, fp, flux = inputs
    options = SiteOptions.from_inputs(
        sites=["MHD", "TAC", "BSD"], averaging_period="1h", inlet="10m", fp_height="10m"
    )
    calls = []

    def wrap(ds, uuid):
        return SimpleNamespace(data=ds, metadata={"uuid": uuid, "dataversion": "v1"})

    monkeypatch.setattr(acquisition, "get_flux_data", lambda **kw: {"a": wrap(flux, "flux")})
    monkeypatch.setattr(
        acquisition, "get_obs_data", lambda **kw: None if kw["site"] == "TAC" else wrap(obs, "obs")
    )
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: calls.append(kw) or wrap(fp, "fp"))
    monkeypatch.setattr(
        acquisition, "merged_scenario_data", lambda *a, **kw: pytest.fail("used ModelScenario path")
    )
    result = MergedData.from_options(
        species="co2",
        site_options=options,
        domain="EUROPE",
        start_date="2020-01-01",
        end_date="2020-01-02",
        flux_sources=["a"],
        use_bc=False,
        surface_transport={"a": "integrated"},
    )
    assert result.sites == ("MHD", "BSD")
    assert result.site_options == options.select_indices([0, 2])
    assert result.flux_data["a"] is flux
    assert result.provenance.footprints["BSD"].uuid == "fp"
    assert len(calls) == 2
    provenance = json.loads(result.site_data["BSD"].attrs["surface_transport_provenance"])
    assert provenance["site_footprint"]["uuid"] == "fp"
    assert provenance["sources"]["a"]["footprint"]["dataversion"] == "v1"
    assert provenance["sources"]["a"]["flux"]["uuid"] == "flux"


def test_fresh_rejects_mixed_native_axes(inputs, monkeypatch):
    obs, fp, flux = inputs
    monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: SimpleNamespace(data=obs, metadata={}))
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: SimpleNamespace(data=fp, metadata={}))
    monkeypatch.setattr(
        acquisition,
        "get_flux_data",
        lambda **kw: {"a": SimpleNamespace(data=flux, metadata={}), "b": SimpleNamespace(data=flux.isel(time=[0]), metadata={})},
    )
    with pytest.raises(ValueError, match="mixed native source axes"):
        MergedData.from_options(
            species="co2",
            site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
            domain="EUROPE",
            start_date="2020-01-01",
            end_date="2020-01-02",
            flux_sources=["a", "b"],
            use_bc=False,
            surface_transport={"a": "integrated", "b": "integrated"},
        )




@pytest.mark.parametrize("lazy", [False, True])
def test_sparse_boundary_weights_through_surface_preparation(inputs, lazy):
    obs, fp, flux = inputs
    fp = fp.assign_coords(height=[10., 20.])
    bc = xr.Dataset(coords={"height": fp.height, "lat": fp.lat, "lon": fp.lon, "time": fp.time},
                    attrs={"time_period": "1 hour"})
    for direction in "nesw":
        side = "lon" if direction in "ns" else "lat"
        weights = np.zeros((2, 1, 3))
        weights[0] = 0.25
        fp[f"particle_locations_{direction}"] = xr.DataArray(
            sparse.COO.from_numpy(weights), dims=("height", side, "time"), attrs={"units": "1"}
        )
        bc[f"vmr_{direction}"] = xr.DataArray(
            np.full((2, 1, 3), 400.05), dims=("height", side, "time"), attrs={"units": "ppm"}
        )
    if lazy:
        fp, bc = fp.chunk({"time": 1}), bc.chunk({"time": 1})
    borrowed = {name: value.data for name, value in fp.data_vars.items()}
    with Callback(pretask=lambda *args: pytest.fail("preparation computed borrowed payload")):
        result = prepare_surface_site(
            obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"},
            boundary_data=bc,
        )
    if lazy:
        assert isinstance(result.bc_mod.data, da.Array)
        result = result.compute(scheduler="synchronous")
    np.testing.assert_allclose(result.bc_mod.data.todense(), 400.05, rtol=0,
                               atol=2e-5)
    for name, value in fp.data_vars.items():
        assert value.data is borrowed[name]




def test_resolved_acquisition_requests_history_and_complete_months(inputs, monkeypatch, compatible_provider):
    obs, fp, flux = resolved_inputs(inputs)
    obs = obs.rename({"co2": "o2"})
    requests = []

    def get_flux(**kwargs):
        requests.append(kwargs)
        return {"a": SimpleNamespace(data=flux, metadata={})}

    monkeypatch.setattr(acquisition, "get_flux_data", get_flux)
    monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: SimpleNamespace(data=obs, metadata={}))
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: SimpleNamespace(data=fp, metadata={}))
    result = MergedData.from_options(
        species="o2",
        site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h", time_resolved=True),
        domain="EUROPE",
        start_date="2020-01-01",
        end_date="2020-01-02",
        flux_sources=["a"],
        use_bc=False,
        surface_transport={"a": "time_resolved"},
    )
    assert requests[0]["allow_older_flux"] is False
    assert pd.Timestamp(requests[0]["start_date"]) == pd.Timestamp("2019-12-31 22:00")
    assert pd.Timestamp(requests[0]["end_date"]) == pd.Timestamp("2020-02-01")
    assert result.flux_data["a"] is flux


@pytest.mark.parametrize("representation", ["split", "legacy"])
def test_actual_midmonth_kernel_matches_independent_history_and_residual(inputs, compatible_provider, representation):
    obs, fp, flux = resolved_inputs(inputs, representation)
    obs = obs.assign_coords(time=obs.time + np.timedelta64(15, "D"))
    fp = fp.assign_coords(time=fp.time + np.timedelta64(15, "D"))
    values = 1 + np.arange(flux.sizes["time"]) * 0.01
    flux["flux"] = flux.flux.copy(data=values.reshape(1, 1, -1) * 1.0e-6)
    result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
    indices = flux.indexes["time"].get_indexer(obs.time.values)
    january = flux.time.dt.month.values == 1
    expected = values[indices] + values[indices - 1] + 5 * values[january].mean()
    np.testing.assert_allclose(result.mf_mod, expected, rtol=1e-14, atol=0)


def test_mapped_footprints_require_identical_observation_timestamps(inputs):
    obs, fp, flux = inputs
    with pytest.raises(ValueError, match="matching spatial and observed timestamps"):
        prepare_surface_site(
            obs,
            {"a": fp, "b": fp.isel(time=slice(1, None))},
            {"a": flux, "b": flux},
            species="co2",
            site_footprint_data=fp,
            transport={"a": "integrated", "b": "integrated"},
        )


@pytest.mark.parametrize("representation", ["split", "legacy"])
def test_year_boundary_varying_flux_matches_independent_history_and_residual(inputs, compatible_provider, representation):
    obs, fp, flux = resolved_inputs(inputs, representation)
    january = flux.time.values >= np.datetime64("2020-01-01")
    values = np.ones(flux.sizes["time"])
    values[january] = 2.0 + 0.1 * np.arange(january.sum())
    flux["flux"] = flux.flux.copy(data=values.reshape(1, 1, -1) * 1.0e-6)
    result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
    # H_back 0/1 contract the current/previous hour; residual uses all January.
    expected = np.array([2.0 + 1.0, 2.1 + 2.0, 2.2 + 2.1]) + 5 * (2 + 0.1 * 743 / 2)
    np.testing.assert_allclose(result.mf_mod, expected, rtol=1.0e-14)


def test_error_components_without_units_use_original_observation_units(inputs):
    obs, fp, flux = inputs
    obs.co2_repeatability.attrs = {}
    obs["mf_error"] = xr.DataArray([0.4, 0.5, 0.6], dims="time")
    result = prepare_surface_site(
        obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"}, output_units="ppb"
    )
    np.testing.assert_allclose(result.mf_repeatability, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(result.mf_error, [400.0, 500.0, 600.0])
    assert obs.co2_repeatability.attrs == {}


def test_fresh_surface_retains_native_domain_axes(inputs, monkeypatch):
    monkeypatch.setattr(
        acquisition, "get_flux_data", lambda **kw: pytest.fail("retrieved unsupported domain")
    )
    with pytest.raises(ValueError, match="matching flux and footprint domains"):
        MergedData.from_options(
            species="co2",
            site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
            domain="EUROPE",
            flux_domain="GLOBAL",
            start_date="2020-01-01",
            end_date="2020-01-02",
            flux_sources=["a"],
            use_bc=False,
            surface_transport={"a": "integrated"},
        )


def test_mapped_footprints_require_explicit_site_transport(inputs):
    obs, fp, flux = inputs
    with pytest.raises(ValueError, match="require explicit site_footprint_data"):
        prepare_surface_site(obs, {"a": fp}, {"a": flux}, species="co2", transport={"a": "integrated"})


def test_mapped_boundaries_use_explicit_site_transport(inputs, monkeypatch):
    obs, fp, flux = inputs
    site_fp, bc = boundary_inputs(fp.assign(pblh=fp.pblh + 500))
    site_fp.attrs = {**fp.attrs, "uuid": "explicit-site-transport", "model": "site-model"}
    contribution = fp.assign(pblh=fp.pblh * 0)

    def provider(boundary, footprint, **kwargs):
        xr.testing.assert_identical(footprint.pblh, site_fp.pblh)
        raise RuntimeError("explicit site transport reached boundary provider")

    import openghg.analyse._modelled_baseline as boundary_provider
    monkeypatch.setattr(boundary_provider, "baseline_sensitivities", provider)
    with pytest.raises(RuntimeError, match="explicit site transport reached"):
        prepare_surface_site(
            obs,
            {"a": contribution},
            {"a": flux},
            species="co2",
            transport={"a": "integrated"},
            site_footprint_data=site_fp,
            boundary_data=bc,
        )
    result = prepare_surface_site(
        obs,
        {"a": contribution},
        {"a": flux},
        species="co2",
        transport={"a": "integrated"},
        site_footprint_data=site_fp,
    )
    assert result.attrs["footprint_transport_model"] == "site-model"
    provenance = json.loads(result.attrs["surface_transport_provenance"])
    assert provenance["site_footprint_selection"] == "explicit"
    assert provenance["site_footprint"]["uuid"] == "explicit-site-transport"
    assert provenance["sources"]["a"]["mode"] == "integrated"


def test_resolved_requires_provider_convolution_operation(inputs, monkeypatch):
    import openghg.analyse._modelled_obs as provider

    monkeypatch.delattr(provider, "fp_x_flux_time_resolved")
    obs, fp, flux = resolved_inputs(inputs)
    with pytest.raises(RuntimeError, match="convolution helpers"):
        prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})


@pytest.mark.parametrize("representation", ["split", "legacy"])
@pytest.mark.parametrize("phase", [0, 30, 90])
@pytest.mark.parametrize("flux_phase", [0, 30])
@pytest.mark.parametrize("cadence_hours", [1, 2])
def test_resolved_coarse_flux_and_shifted_releases_match_interval_history(
    inputs, compatible_provider, cadence_hours, representation, phase, flux_phase
):
    obs, fp, flux = resolved_inputs(inputs, representation)
    obs = obs.assign_coords(time=obs.time + np.timedelta64(phase, "m"))
    fp = fp.assign_coords(time=obs.time)
    flux = flux.isel(time=slice(None, None, cadence_hours))
    flux = flux.assign_coords(time=flux.time + np.timedelta64(flux_phase, "m"))
    values = 1 + np.arange(flux.sizes["time"]) * .01
    flux["flux"] = flux.flux.copy(data=values.reshape(1, 1, -1) * 1e-6)
    result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
    times = flux.indexes["time"]
    releases = pd.DatetimeIndex(obs.time.values)
    current = times.get_indexer(releases, method="ffill")
    previous = times.get_indexer(releases - pd.Timedelta(hours=1), method="ffill")
    january = times.month == 1
    expected = values[current] + values[previous] + 5 * values[january].mean()
    np.testing.assert_allclose(result.mf_mod, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("change", ["fractional_history", "irregular_history", "grid_window", "history_spacing", "monthly_fallback"])
def test_resolved_rejects_unsupported_history_and_provider_fallback(inputs, compatible_provider, change):
    obs, fp, flux = resolved_inputs(inputs)
    message = "whole-hour"
    if change == "fractional_history":
        fp = fp.assign_coords(H_back=[0, 0.5])
    elif change == "irregular_history":
        fp = fp.reindex(H_back=[0, 1, 3], fill_value=1)
    elif change == "grid_window":
        obs = obs.assign_coords(time=pd.date_range("2020-01-01", periods=3, freq="2h"))
        fp = fp.assign_coords(time=obs.time, H_back=[0, 2])
        message = "rolling window and transport grid"
    elif change == "history_spacing":
        fp = fp.assign_coords(H_back=[0, 5])
        message = "divide 24 hours"
    else:
        flux = flux.isel(time=slice(None, None, 24))
        message = "provider would substitute monthly flux"
    with pytest.raises(ValueError, match=message):
        prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})


def test_explicit_site_alignment_sets_observation_cadence_once(inputs):
    obs, fp, flux = inputs
    times = pd.date_range("2020-01-01", periods=6, freq="h")
    obs = obs.reindex(time=times, method="ffill")
    fp = fp.reindex(time=times, method="ffill")
    flux = flux.reindex(time=times, method="ffill")
    fp["fp"] = fp.fp.copy(data=flux.flux.values * 1e6)
    flux["flux"] = flux.flux.copy(data=np.full(flux.flux.shape, 1e-6))
    site_fp = fp.isel(time=[0, 2, 4])
    result = prepare_surface_site(
        obs,
        {"a": fp},
        {"a": flux},
        species="co2",
        transport={"a": "integrated"},
        site_footprint_data=site_fp,
    )
    # Site transport chooses two-hour observation means. Hourly source transport
    # must be averaged to those same intervals, rather than sampled every two hours.
    np.testing.assert_allclose(result.mf, [400.5, 402.0, 402.0])
    np.testing.assert_allclose(result.mf_mod, [1.5, 3.0, 3.0])


def test_resolved_grid_uses_release_and_history_gcd(inputs, compatible_provider):
    obs, fp, flux = resolved_inputs(inputs)
    releases = pd.date_range("2020-01-01", periods=3, freq="2h")
    obs = obs.assign_coords(time=releases)
    fp = fp.assign_coords(time=releases, H_back=[0, 3])
    values = 1 + np.arange(flux.sizes["time"]) * .01
    flux["flux"] = flux.flux.copy(data=values.reshape(1, 1, -1) * 1e-6)
    # gcd(2h releases, 3h history) is 1h; flux may be coarser than that grid.
    for cadence in (1, 3):
        sampled = flux.isel(time=slice(None, None, cadence))
        sampled_values = values[::cadence]
        result = prepare_surface_site(obs, fp, {"a": sampled}, species="co2", transport={"a": "time_resolved"})
        index = sampled.indexes["time"]
        current = index.get_indexer(releases, method="ffill")
        previous = index.get_indexer(releases - pd.Timedelta(hours=3), method="ffill")
        expected = sampled_values[current] + sampled_values[previous] + 5 * sampled_values[index.month == 1].mean()
        np.testing.assert_allclose(result.mf_mod, expected, rtol=1e-14, atol=0)


def test_integrated_declared_year_validity_never_reuses_climatological_year(inputs):
    obs, fp, flux = inputs
    obs = obs.assign_coords(time=obs.time + pd.Timedelta(days=366))
    fp = fp.assign_coords(time=obs.time)
    annual = flux.isel(time=[0]).assign_coords(time=[pd.Timestamp("2020-12-01")])
    annual.flux.attrs["time_period"] = "yearly"
    result = prepare_surface_site(obs, fp, {"annual": annual}, species="co2", transport={"annual": "integrated"})
    np.testing.assert_allclose(result.mf_mod, 1.0)






@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("mode", ["integrated", "time_resolved"])
@pytest.mark.parametrize("extra_releases", [False, True])
def test_native_products_precede_observation_averaging(inputs, mode, lazy, extra_releases):
    if mode == "time_resolved":
        obs, fp, flux = resolved_inputs(inputs)
    else:
        obs, fp, flux = inputs
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    obs = obs.reindex(time=times, method="ffill")
    obs = obs.isel(time=[0, 2])
    obs.attrs["sampling_period"] = 7200
    fp = fp.reindex(time=times, method="ffill")
    fp["fp"] = fp.fp.copy(data=np.array([[[1., 3., 1., 3.]]]))
    if mode == "integrated":
        flux = flux.reindex(time=times, method="ffill")
    flux["flux"] = flux.flux.copy(data=np.where(flux.time.dt.hour.values % 2, 3e-6, 1e-6)[None, None, :])
    if mode == "time_resolved":
        fp["fp_time_resolved"] = fp.fp.expand_dims(H_back=[0, 1]) / 2
        fp.fp_time_resolved.attrs["units"] = fp.fp.attrs["units"]
        fp["fp_residual"] = fp.fp * 0
        fp.fp_residual.attrs["units"] = fp.fp.attrs["units"]
    fp = fp.assign_coords(height=[10.])
    bc = xr.Dataset(coords={"height": [10.], "lat": fp.lat, "lon": fp.lon, "time": times})
    for direction in "nesw":
        horizontal = "lon" if direction in "ns" else "lat"
        fp[f"particle_locations_{direction}"] = xr.DataArray(
            np.array([.1, .4, .1, .4] if direction == "n" else [.3, .2, .3, .2])[None, None, :],
            dims=("height", horizontal, "time"), attrs={"units": "dimensionless"},
        )
        bc[f"vmr_{direction}"] = xr.DataArray(
            np.array([400., 404., 400., 404.])[None, None, :],
            dims=("height", horizontal, "time"), attrs={"units": "ppm"},
        )
    bc.attrs["time_period"] = "hourly"
    if extra_releases:
        # These unused releases must not require flux or boundary coverage.
        fp = fp.reindex(time=pd.date_range(times[0], periods=6, freq="h"), method="ffill")
    if lazy:
        fp, flux, bc = (value.chunk({"time": 2}) for value in (fp, flux, bc))
    before = [value.copy(deep=True) for value in (obs, fp, flux, bc)]
    result = prepare_surface_site(
        obs, fp, {"a": flux}, species="co2", transport={"a": mode}, boundary_data=bc,
    )
    # Integrated native products: mean(1*1, 3*3)=5; resolved uses the two-hour history mean=2.
    np.testing.assert_allclose(result.mf_mod, 5. if mode == "integrated" else 4.)
    np.testing.assert_allclose(result.bc_mod, 402.)
    np.testing.assert_allclose(result.bc_n, (400. * .1 + 404. * .4) / 2)
    for value, saved in zip((obs, fp, flux, bc), before, strict=True):
        xr.testing.assert_identical(value, saved)


def test_transport_metadata_falls_back_to_wrapper_and_serializes_numpy(inputs):
    obs, fp, flux = inputs
    fp.attrs.update(model="not_set", transport_model_version=None, met_model="", max_level=np.int64(40))
    wrapper = SimpleNamespace(data=fp, metadata={"model": "NAME", "model_version": "v2", "metmodel": "UKV", "uuid": "selected", "dataversion": np.int64(3)})
    result = prepare_surface_site(obs, wrapper, {"a": flux}, species="co2", transport={"a": "integrated"})
    assert result.attrs["footprint_transport_model"] == "NAME"
    assert result.attrs["footprint_transport_model_version"] == "v2"
    assert result.attrs["footprint_met_model"] == "UKV"
    metadata = json.loads(result.attrs["surface_transport_provenance"])["site_footprint"]
    assert metadata["max_level"] == 40
    assert metadata["dataversion"] == 3
    assert metadata["transport_model"] == "NAME"


def test_mapped_source_cannot_coarsen_explicit_observation_axis(inputs):
    obs, fp, flux = inputs
    times = pd.date_range("2020-01-01", periods=6, freq="h")
    obs, fp, flux = (value.reindex(time=times, method="ffill") for value in (obs, fp, flux))
    source_fp = fp.isel(time=[0, 2, 4])
    with pytest.raises(ValueError, match="source 'a'.*cadence must not be coarser"):
        prepare_surface_site(
            obs, {"a": source_fp}, {"a": flux}, site_footprint_data=fp,
            species="co2", transport={"a": "integrated"},
        )


def test_gather_and_serialization_preserve_each_site_transport_provenance():
    from openghg_inversions.inversion_data import PreparedInputs, prepare_rhime_inputs_from_xarray
    from tests.test_xarray_input_adapter import _basis_functions, _site_dataset

    sites = {
        name: _site_dataset(name, ["2020-01-01", "2020-01-01T03:00"], sampled_baseline=True)
        for name in ("AAA", "BBB")
    }
    for number, dataset in enumerate(sites.values()):
        dataset.attrs.update(
            surface_transport='{"inventory": "integrated"}',
            surface_transport_provenance=f"site-{number}",
            surface_provider_revision=f"provider-{number}",
        )
    prepared = prepare_rhime_inputs_from_xarray(sites, basis_functions=_basis_functions())
    restored = PreparedInputs.from_datatree(prepared.to_datatree())
    facts = json.loads(restored.inv_inputs.attrs["surface_preparation_provenance"])
    assert facts["AAA"]["surface_transport_provenance"] == "site-0"
    assert facts["BBB"]["surface_transport_provenance"] == "site-1"
    assert facts["AAA"]["surface_provider_revision"] == "provider-0"
    assert facts["BBB"]["surface_provider_revision"] == "provider-1"
    assert "surface_transport_provenance" not in restored.inv_inputs.attrs


@pytest.mark.parametrize("declared", [False, True])
def test_single_observation_bin_keeps_chosen_cadence_without_sampling_attr(inputs, declared):
    obs, fp, flux = inputs
    obs = obs.isel(time=[0, 2])
    obs.attrs.pop("sampling_period")
    if declared:
        obs.attrs["averaged_period"] = 7200
    fp = fp.assign_coords(time=pd.date_range("2020-01-01", periods=3, freq="4h"))
    flux.flux.attrs["time_period"] = "daily"
    flux = flux.isel(time=[0])
    result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"})
    assert result.sizes["time"] == 1
    assert result.attrs["sampling_period"] == 14400
    np.testing.assert_allclose(result.mf_mod, 1.)
