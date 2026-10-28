"""Fresh acquisition derives support from the transport it will actually use."""

import json
from copy import deepcopy

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback
from openghg.dataobjects import FluxData, FootprintData, ObsData

from openghg_inversions.inversion_data import MergedData, SiteOptions, prepare_surface_site
from openghg_inversions.inversion_data import acquisition


def native_inputs(release, history, *, legacy=False, flux_phase=0, cadence=1):
    releases = pd.date_range(release, periods=3, freq="h")
    obs = xr.Dataset(
        {"co2": ("time", [400., 401., 402.], {"units": "ppm"}),
         "mf_error": ("time", [.1, .2, .3], {"units": "ppm"})},
        coords={"time": releases}, attrs={"scale": "WMO", "sampling_period": 3600},
    )
    fp = xr.Dataset(
        {"fp": (("lat", "lon", "time"), np.ones((1, 1, 3)), {"units": "s m2/mol"})},
        coords={"lat": [0.], "lon": [0.], "time": releases, "H_back": np.arange(history)},
    )
    fp["fp_time_resolved"] = fp.fp.expand_dims(H_back=fp.H_back)
    fp["fp_residual"] = fp.fp * 3
    fp.fp_residual.attrs = fp.fp.attrs
    if legacy:
        resolved = xr.concat([fp.fp_time_resolved, fp.fp_residual.expand_dims(H_back=[history])], dim="H_back")
        fp = fp.drop_dims("H_back").drop_vars("fp_residual")
        fp["fp_HiTRes"] = resolved
    first = releases[0] - pd.Timedelta(days=40)
    last = (releases[-1].to_period("M") + 1).start_time
    times = pd.date_range(first, last, freq=f"{cadence}h", inclusive="left") + pd.Timedelta(minutes=flux_phase)
    values = 1 + np.arange(len(times)) * .01
    flux = xr.Dataset(
        {"flux": (("lat", "lon", "time"), values.reshape(1, 1, -1) * 1e-6, {"units": "mol/m2/s"})},
        coords={"lat": [0.], "lon": [0.], "time": times},
    )
    return obs, fp, flux


def install_store(monkeypatch, observations, footprints, flux):
    requests = []
    observation_requests = []

    def get_obs(**kwargs):
        observation_requests.append(kwargs)
        return observations.get(kwargs["site"])

    def get_flux(**kwargs):
        requests.append(kwargs)
        assert kwargs["allow_older_flux"] is False
        assert kwargs["start_date"] is not None
        start, end = pd.Timestamp(kwargs["start_date"]), pd.Timestamp(kwargs["end_date"])
        sliced = flux.sel(time=(flux.time >= start - pd.Timedelta(seconds=1)) &
                              (flux.time <= end - pd.Timedelta(seconds=1)))
        return {source: FluxData(data=sliced, metadata={"uuid": "flux-id", "dataversion": "v3"}, sort=False)
                for source in kwargs["sources"]}

    monkeypatch.setattr(acquisition, "get_obs_data", get_obs)
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: footprints.get(kw["site"]))
    monkeypatch.setattr(acquisition, "get_flux_data", get_flux)
    return requests, observation_requests


def acquire(sites, *, cls=MergedData):
    return cls.from_options(
        species="co2", site_options=SiteOptions.from_inputs(
            sites=sites, averaging_period="1h", time_resolved=True),
        domain="EUROPE", start_date="2020-01-01", end_date="2021-04-01",
        flux_sources=["a"], surface_transport={"a": "time_resolved"}, use_bc=False,
    )


@pytest.mark.parametrize("release", ["2020-03-01", "2021-01-01"])
@pytest.mark.parametrize("history", [24, 120, 720])
@pytest.mark.parametrize("legacy", [False, True])
def test_fresh_native_history_and_monthly_residual_match_independent_result(
    monkeypatch, compatible_provider, release, history, legacy
):
    obs, fp, flux = native_inputs(release, history, legacy=legacy)
    requests, obs_requests = install_store(
        monkeypatch, {"MHD": ObsData(data=obs, metadata={}, sort=False)},
        {"MHD": FootprintData(data=fp, metadata={}, sort=False)}, flux,
    )
    result = acquire(["MHD"])
    first = pd.Timestamp(release)
    assert pd.Timestamp(requests[0]["start_date"]) == first - pd.Timedelta(hours=history)
    assert pd.Timestamp(requests[0]["end_date"]) - pd.Timedelta(seconds=1) + pd.Timedelta("1ns") == (first.to_period("M") + 1).start_time
    assert len(requests) == 1
    index = flux.indexes["time"]
    values = flux.flux.values.ravel() * 1e6
    monthly = values[index.to_period("M") == first.to_period("M")].mean()
    expected = [sum(values[index.get_indexer([release_time - pd.Timedelta(hours=h)])[0]]
                    for h in range(history)) + 3 * monthly for release_time in obs.indexes["time"]]
    np.testing.assert_allclose(result.site_data["MHD"].mf_mod, expected, rtol=1e-14, atol=0)
    # Explicit total error survives retrieval's shared variable selection policy.
    assert "mf_error" in obs_requests[0]["keep_variables"]
    assert "co2_error" in obs_requests[0]["keep_variables"]
    supplied = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
    xr.testing.assert_identical(result.site_data["MHD"].mf_error, supplied.mf_error)



def test_retained_sites_union_excludes_missing_and_unused_releases(monkeypatch, compatible_provider):
    early_obs, early_fp, flux = native_inputs("2020-03-01", 24)
    later_obs, later_fp, _ = native_inputs("2020-03-03", 120)
    # Extra regular native transport outside the observed bins contributes no support.
    extra = pd.date_range("2020-02-28", "2020-03-06", freq="h")
    later_fp = later_fp.reindex(time=extra, method="nearest")
    observations = {"MHD": ObsData(data=early_obs, metadata={}, sort=False),
                    "TAC": ObsData(data=later_obs, metadata={}, sort=False),
                    "RGL": ObsData(data=early_obs, metadata={}, sort=False)}
    footprints = {"MHD": FootprintData(data=early_fp, metadata={}, sort=False),
                  "TAC": FootprintData(data=later_fp, metadata={}, sort=False)}
    requests, _ = install_store(monkeypatch, observations, footprints, flux)
    result = acquire(["MHD", "MISSING", "TAC", "RGL"])
    assert result.sites == ("MHD", "TAC")
    assert result.site_options.averaging_period == ("1h", "1h")
    assert pd.Timestamp(requests[0]["start_date"]) == pd.Timestamp("2020-02-27")
    assert pd.Timestamp(requests[0]["end_date"]) - pd.Timedelta(seconds=1) + pd.Timedelta("1ns") == pd.Timestamp("2020-04-01")


def test_selected_identities_own_metadata_borrow_payloads_and_preserve_subclass(monkeypatch, compatible_provider):
    obs, fp, flux = native_inputs("2020-03-01", 24)
    observations = ObsData(data=obs.chunk({"time": 2}), metadata={"uuid": ["obs-a", "obs-b"], "latest_version": "v99"}, sort=False)
    footprints = FootprintData(data=fp.chunk({"time": 2}), metadata={"uuid": ["fp-b", "fp-a"], "latest_version": ["v2", "v1"]}, sort=False)
    before = [deepcopy(value.metadata) for value in (observations, footprints)]
    copied = acquisition._with_selected_provenance(observations, acquisition.selected_provenance(observations, "obs-store"))
    assert copied.data is observations.data
    assert copied.metadata is not observations.metadata
    assert copied.metadata["dataversion"] == "unknown"
    install_store(monkeypatch, {"MHD": observations}, {"MHD": footprints}, flux.chunk({"time": 24}))

    class CustomMergedData(MergedData):
        pass

    with Callback(pretask=lambda *args: pytest.fail("Acquisition computed payload")):
        result = acquire(["MHD"], cls=CustomMergedData)
    assert isinstance(result, CustomMergedData)
    assert isinstance(result.site_data["MHD"].mf.data, da.Array)
    assert [value.metadata for value in (observations, footprints)] == before
    assert result.provenance.observations["MHD"].dataversion == "unknown"
    details = json.loads(result.site_data["MHD"].attrs["surface_transport_provenance"])
    assert details["site_footprint"]["uuid"] == ["fp-b", "fp-a"]
    assert details["site_footprint"]["dataversion"] == ["v2", "v1"]
    assert details["sources"]["a"]["footprint"] == details["site_footprint"]
    assert details["sources"]["a"]["flux"]["dataversion"] == "v3"


def test_mixed_modes_retrieve_common_native_support(monkeypatch, compatible_provider):
    obs, fp, flux = native_inputs("2020-03-01", 24)
    requests, _ = install_store(
        monkeypatch, {"MHD": ObsData(data=obs, metadata={}, sort=False)},
        {"MHD": FootprintData(data=fp, metadata={}, sort=False)}, flux,
    )
    result = MergedData.from_options(
        species="co2", site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h", time_resolved=True),
        domain="EUROPE", start_date="2020-03-01", end_date="2020-03-02",
        flux_sources=["resolved", "integrated"],
        surface_transport={"resolved": "time_resolved", "integrated": "integrated"}, use_bc=False,
    )
    assert requests[0]["sources"] == ["resolved", "integrated"]
    xr.testing.assert_identical(result.flux_data["resolved"], result.flux_data["integrated"])
    expected_integrated = flux.flux.sel(time=obs.time).sum(["lat", "lon"]) * 1e6
    np.testing.assert_allclose(result.site_data["MHD"].mf_mod_sectoral.sel(source="integrated"),
                               expected_integrated, rtol=1e-14, atol=0)


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize("release_shift_ms", [0, 500])
def test_fresh_real_getter_preserves_final_release_and_shifted_declared_intervals(
    monkeypatch, compatible_provider, resolved, release_shift_ms
):
    from types import SimpleNamespace
    from openghg_inversions.inversion_data import getters

    cadence = 2 if resolved else 1
    obs, fp, flux = native_inputs(pd.Timestamp("2020-03-01") + pd.Timedelta(milliseconds=release_shift_ms),
                                  24, flux_phase=30 if resolved else 0, cadence=cadence)
    if not resolved:
        fp = fp.drop_dims("H_back").drop_vars("fp_residual")
    monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: ObsData(data=obs, metadata={}, sort=False))
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: FootprintData(data=fp, metadata={}, sort=False))
    monkeypatch.setattr(getters, "search_flux", lambda **kw: SimpleNamespace(results=pd.DataFrame({"time_period": [f"{cadence}h"]})))

    def provider_flux(**kwargs):
        # Match OpenGHG _BaseData's exclusive-end conversion at actual retrieval.
        sliced = flux.sel(time=slice(pd.Timestamp(kwargs["start_date"]) - pd.Timedelta(seconds=1),
                                     pd.Timestamp(kwargs["end_date"]) - pd.Timedelta(seconds=1)))
        return FluxData(data=sliced, metadata={"time_period": f"{cadence}h"}, sort=False)

    monkeypatch.setattr(getters, "get_flux", provider_flux)
    monkeypatch.setattr(acquisition, "get_flux_data", getters.get_flux_data)
    result = MergedData.from_options(
        species="co2", site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h", time_resolved=resolved),
        domain="EUROPE", start_date="2020-03-01", end_date="2020-03-02",
        flux_sources=["a"], surface_transport={"a": "time_resolved" if resolved else "integrated"}, use_bc=False,
    )
    retained = result.flux_data["a"]
    assert retained.flux.dtype == np.float32
    if resolved:
        index = retained.indexes["time"]
        raw_values = retained.flux.values.ravel()
        values = raw_values.astype(float) * 1e6
        # The provider's residual mean retains the sanitized source float32 dtype.
        monthly = float(raw_values[index.month == 3].mean(dtype=np.float32)) * 1e6
        expected = [values[index.get_indexer(
            pd.date_range(t - pd.Timedelta(hours=23), t, freq="h"), method="ffill"
        )].sum() + 3 * monthly for t in obs.indexes["time"]]
        assert index[0] < pd.Timestamp("2020-02-29")
    else:
        assert retained.indexes["time"][-1] == obs.indexes["time"][-1]
        expected = retained.flux.sel(time=obs.time).values.ravel().astype(float) * 1e6
    np.testing.assert_allclose(result.site_data["MHD"].mf_mod, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("source_order", [("declared", "inferred"), ("inferred", "declared")])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cadence,phase,declared_period", [(1, 0, "1h"), (2, 30, None)])
def test_real_getter_sources_share_effective_support_after_padding_or_retry(
    monkeypatch, compatible_provider, source_order, cadence, phase, declared_period, legacy
):
    from types import SimpleNamespace
    from openghg_inversions.inversion_data import getters

    obs, fp, flux = native_inputs("2020-03-01", 24, cadence=cadence, flux_phase=phase, legacy=legacy)
    # Binary-exact varying inputs keep the interval oracle independent of sum-order roundoff.
    values = np.ldexp((1 + np.arange(flux.sizes["time"]) % 8).astype(float), -20)
    flux["flux"] = flux.flux.copy(data=values.reshape(1, 1, -1))
    monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: ObsData(data=obs, metadata={}, sort=False))
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: FootprintData(data=fp, metadata={}, sort=False))
    monkeypatch.setattr(getters, "search_flux", lambda **kw: SimpleNamespace(results=pd.DataFrame({
        "time_period": [declared_period if kw["source"] == "declared" else None]
    })))

    def provider_flux(**kwargs):
        sliced = flux.sel(time=slice(pd.Timestamp(kwargs["start_date"]) - pd.Timedelta(seconds=1),
                                     pd.Timestamp(kwargs["end_date"]) - pd.Timedelta(seconds=1)))
        metadata = {"uuid": kwargs["source"]}
        if kwargs["source"] == "declared" and declared_period is not None:
            metadata["time_period"] = declared_period
        return FluxData(data=sliced, metadata=metadata, sort=False)

    monkeypatch.setattr(getters, "get_flux", provider_flux)
    monkeypatch.setattr(acquisition, "get_flux_data", getters.get_flux_data)
    result = MergedData.from_options(
        species="co2", site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h", time_resolved=True),
        domain="EUROPE", start_date="2020-03-01", end_date="2020-03-02",
        flux_sources=source_order, surface_transport={source: "time_resolved" for source in source_order}, use_bc=False,
    )
    first, second = (result.flux_data[source] for source in source_order)
    assert first.indexes["time"].equals(second.indexes["time"])
    assert first.indexes["time"][0] <= pd.Timestamp("2020-02-29")
    assert first.indexes["time"][-1] < pd.Timestamp("2020-04-01")
    raw = first.flux.values.ravel()
    times = first.indexes["time"]
    monthly = float(raw[times.month == 3].mean(dtype=np.float32)) * 1e6
    expected = [raw[times.get_indexer(pd.date_range(t - pd.Timedelta(hours=23), t, freq="h"),
                                     method="ffill")].astype(float).sum() * 1e6 + 3 * monthly
                for t in obs.indexes["time"]]
    for source in source_order:
        np.testing.assert_allclose(result.site_data["MHD"].mf_mod_sectoral.sel(source=source), expected,
                                   rtol=1e-14, atol=0)
    np.testing.assert_allclose(result.site_data["MHD"].mf_mod, 2 * np.asarray(expected), rtol=1e-14, atol=0)


@pytest.mark.parametrize("surface_mode", [False, True])
def test_calendar_flux_lookup_keeps_governing_off_anchor_label_for_both_routes(monkeypatch, surface_mode):
    from types import SimpleNamespace
    from openghg_inversions.inversion_data import getters

    obs, fp, _ = native_inputs("2021-01-01", 24)
    fp = fp.drop_dims("H_back").drop_vars("fp_residual")
    times = pd.to_datetime(["2019-12-01", "2020-12-01", "2021-12-01"])
    flux = xr.Dataset({"flux": (("lat", "lon", "time"), np.array([1., 2., 3.]).reshape(1, 1, -1) * 1e-6,
                                {"units": "mol/m2/s", "time_period": "yearly"})},
                      coords={"lat": [0.], "lon": [0.], "time": times})
    monkeypatch.setattr(acquisition, "get_obs_data", lambda **kw: ObsData(data=obs, metadata={}, sort=False))
    monkeypatch.setattr(acquisition, "get_footprint_data", lambda **kw: FootprintData(data=fp, metadata={}, sort=False))
    monkeypatch.setattr(getters, "search_flux", lambda **kw: SimpleNamespace(results=pd.DataFrame({"time_period": ["yearly"]})))

    def provider_flux(**kwargs):
        assert kwargs["start_date"] is not None, "calendar lookup must be bounded"
        sliced = flux.sel(time=slice(pd.Timestamp(kwargs["start_date"]) - pd.Timedelta(seconds=1),
                                     pd.Timestamp(kwargs["end_date"]) - pd.Timedelta(seconds=1)))
        return FluxData(data=sliced, metadata={"time_period": "yearly"}, sort=False)

    monkeypatch.setattr(getters, "get_flux", provider_flux)
    monkeypatch.setattr(acquisition, "get_flux_data", getters.get_flux_data)
    monkeypatch.setattr(acquisition, "merged_scenario_data", lambda obs, fp, flux, bc, **kw: obs.data.rename(co2="mf"))
    result = MergedData.from_options(
        species="co2", site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
        domain="EUROPE", start_date="2021-01-01", end_date="2021-01-02",
        flux_sources=["a"], surface_transport={"a": "integrated"} if surface_mode else None, use_bc=False,
    )
    assert result.flux_data["a"].indexes["time"].equals(pd.DatetimeIndex(["2020-12-01"]))
    np.testing.assert_allclose(result.flux_data["a"].flux, np.float32(2e-6))


def test_fresh_sites_resolve_coarsest_alignment_once_and_warn_on_native_scales(
    monkeypatch, compatible_provider, caplog,
):
    """Support planning reuses the converted bins used by the physical response."""
    from openghg_inversions.inversion_data import surface

    obs, fp, flux = native_inputs("2020-03-01", 24)
    native_times = obs.time
    obs = obs.isel(time=[0, 2]).assign_attrs(sampling_period=7200)
    other_obs = obs.copy(deep=False).assign_attrs(scale="OTHER")
    install_store(
        monkeypatch,
        {"MHD": ObsData(data=obs, metadata={}, sort=False),
         "TAC": ObsData(data=other_obs, metadata={}, sort=False)},
        {site: FootprintData(data=fp, metadata={}, sort=False) for site in ("MHD", "TAC")}, flux,
    )
    align, integrated, resolved = surface._surface_operations()
    coarsest_calls = []

    def record_alignment(*args, **kwargs):
        if kwargs.get("resample_to") == "coarsest":
            coarsest_calls.append(args[0].attrs["scale"])
        return align(*args, **kwargs)

    monkeypatch.setattr(surface, "_surface_operations", lambda: (record_alignment, integrated, resolved))
    result = acquire(["MHD", "TAC"])
    assert coarsest_calls == ["WMO", "OTHER"]
    assert "Not all sites using the same calibration scale: 2 scales found" in caplog.text
    values = flux.flux * 1e6
    monthly = values.sel(time=slice("2020-03-01", "2020-03-31")).mean("time")
    native_response = xr.concat([
        values.sel(time=native_times - pd.Timedelta(hours=h)).assign_coords(time=native_times)
        for h in range(24)
    ], dim="lag").sum("lag") + 3 * monthly
    expected = native_response.resample(time="2h").mean().squeeze(["lat", "lon"])
    np.testing.assert_allclose(result.site_data["MHD"].mf_mod, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(result.site_data["TAC"].mf_mod, expected, rtol=1e-14, atol=0)
