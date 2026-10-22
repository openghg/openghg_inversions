"""Explicit surface convolution and the bounded acquisition opt-in."""

import json
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data import MergedData, SiteOptions, prepare_surface_site
from openghg_inversions.inversion_data import acquisition, surface


@pytest.fixture
def compatible_provider():
    import openghg.analyse._modelled_baseline as provider

    if not hasattr(provider, "baseline_reference_sensitivities"):
        pytest.skip("Requires issue #957 OpenGHG development provider with midmonth fix")


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


def test_native_boundaries_forwarded_and_reconstructed_before_rounding(inputs, monkeypatch):
    obs, fp, flux = inputs
    bc = xr.Dataset()
    seen = {}

    def provider(boundary, footprint, **kwargs):
        seen.update(kwargs)
        assert footprint.time.identical(obs.time)
        return xr.Dataset(
            {
                **{
                    f"bc_anomaly_{d}": (("height", "time"), np.full((1, 3), 0.001), {"units": "ppm"})
                    for d in "nesw"
                },
                **{
                    f"bc_mass_{d}": (("height", "time"), np.full((1, 3), 0.25), {"units": "dimensionless"})
                    for d in "nesw"
                },
                "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
            },
            coords={"time": obs.time, "height": [10.0]},
        )

    monkeypatch.setattr(surface, "_reference_sensitivities", provider)
    result = prepare_surface_site(
        obs,
        fp,
        {"a": flux},
        species="co2",
        transport={"a": "integrated"},
        boundary_data=bc,
        boundary_reference=400.0,
        boundary_reference_units="ppm",
        boundary_mass_tolerance=0.01,
        output_units="ppb",
    )
    assert seen == {"species": "co2", "reference": 400.0, "reference_units": "ppm", "mass_tolerance": 0.01}
    assert result.bc_reference.item() == pytest.approx(400000.0)
    assert result.bc_reference.dims == ()
    np.testing.assert_allclose(result.bc_anomaly_n, 1.0)
    np.testing.assert_allclose(result.bc_mass_n, 0.25)
    assert not any(f"bc_{direction}" in result for direction in "nesw")
    np.testing.assert_allclose(result.bc_mod, 400004.0)


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
    _, _, flux = inputs
    monkeypatch.setattr(
        acquisition,
        "get_flux_data",
        lambda **kw: {"a": SimpleNamespace(data=flux), "b": SimpleNamespace(data=flux.isel(time=[0]))},
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


@pytest.mark.parametrize("species", ["co2", "o2"])
def test_actual_provider_boundary_operands(inputs, species):
    import openghg.analyse._modelled_baseline as provider

    if not hasattr(provider, "baseline_reference_sensitivities"):
        pytest.skip("Requires issue #957 OpenGHG development provider")
    obs, fp, flux = inputs
    obs = obs.rename({"co2": "mf"})
    fp = fp.assign_coords(height=[10.0])
    bc = xr.Dataset(
        coords={"height": [10.0], "lat": fp.lat, "lon": fp.lon, "time": fp.time[:1]},
        attrs={"time_period": "1 month"},
    )
    for d in "nesw":
        side = "lon" if d in "ns" else "lat"
        fp[f"particle_locations_{d}"] = xr.DataArray(
            np.full((1, 1, 3), 0.25001), dims=("height", side, "time"), attrs={"units": "1"}
        )
        bc[f"vmr_{d}"] = xr.DataArray(
            np.full((1, 1, 1), 400.123456), dims=("height", side, "time"), attrs={"units": "ppm"}
        )
    saved_bc, saved_fp = bc.copy(deep=True), fp.copy(deep=True)
    result = prepare_surface_site(
        obs,
        fp,
        {"a": flux},
        species=species,
        transport={"a": "integrated"},
        boundary_data=bc,
        boundary_reference=400.0,
        boundary_reference_units="ppm",
        boundary_mass_tolerance=0.001,
    )
    np.testing.assert_allclose(result.bc_mod, 400.123456, rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(result.bc_anomaly_n, 0.123456 / 4, rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(result.bc_mass_n, 0.25)
    assert result.bc_anomaly_n.dtype == np.float64
    assert result.attrs["boundary_representation"] == "centered-v1"
    assert result.attrs["boundary_mass_tolerance"] == 0.001
    assert result.attrs["boundary_normalization_policy"] == "joint-curtain-unit-mass-v1"
    assert result.attrs["boundary_provider_policy_revision"] == "957-reference-sensitivities-v1"
    import openghg

    assert result.attrs["surface_provider_revision"] == str(openghg.__revisionid__ or "unknown")
    assert result.attrs["surface_policy_revision"] == "957-source-wise-v1"
    xr.testing.assert_identical(bc, saved_bc)
    xr.testing.assert_identical(fp, saved_fp)


def test_missing_provider_capability_fails_without_legacy_fallback(inputs, monkeypatch):
    import openghg.analyse._modelled_baseline as provider

    monkeypatch.delattr(provider, "baseline_reference_sensitivities", raising=False)
    obs, fp, flux = inputs
    with pytest.raises(RuntimeError, match="Legacy rounded boundaries"):
        prepare_surface_site(
            obs,
            fp,
            {"a": flux},
            species="co2",
            transport={"a": "integrated"},
            boundary_data=xr.Dataset(),
            boundary_reference=400.0,
            boundary_reference_units="ppm",
            boundary_mass_tolerance=0.001,
        )


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
    assert pd.Timestamp(requests[0]["start_date"]) == pd.Timestamp("2019-12-01")
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
    np.testing.assert_allclose(result.mf_mod, expected, rtol=1e-14)


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
    site_fp = fp.assign(pblh=fp.pblh + 500)
    site_fp.attrs = {**fp.attrs, "uuid": "explicit-site-transport", "model": "site-model"}
    contribution = fp.assign(pblh=fp.pblh * 0)

    def provider(boundary, footprint, **kwargs):
        xr.testing.assert_identical(footprint.pblh, site_fp.pblh)
        raise RuntimeError("explicit site transport reached boundary provider")

    monkeypatch.setattr(surface, "_reference_sensitivities", provider)
    with pytest.raises(RuntimeError, match="explicit site transport reached"):
        prepare_surface_site(
            obs,
            {"a": contribution},
            {"a": flux},
            species="co2",
            transport={"a": "integrated"},
            site_footprint_data=site_fp,
            boundary_data=xr.Dataset(),
            boundary_reference=400.0,
            boundary_reference_units="ppm",
            boundary_mass_tolerance=0.001,
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


def test_resolved_requires_coordinated_provider_revision(inputs, monkeypatch):
    import openghg.analyse._modelled_baseline as provider

    monkeypatch.delattr(provider, "baseline_reference_sensitivities", raising=False)
    obs, fp, flux = resolved_inputs(inputs)
    with pytest.raises(RuntimeError, match="midmonth residual-flux fix"):
        prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})


@pytest.mark.parametrize("representation", ["split", "legacy"])
@pytest.mark.parametrize("cadence_hours", [1, 2, 3, 24])
def test_resolved_cadence_bounded_by_transport_grid(
    inputs, compatible_provider, cadence_hours, representation
):
    obs, fp, flux = resolved_inputs(inputs, representation)
    flux = flux.isel(time=slice(None, None, cadence_hours))
    if cadence_hours > 1:
        with pytest.raises(ValueError, match="coarser than the resolved transport grid GCD"):
            prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
    else:
        result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
        # The legacy H_back=24 sentinel is residual mass, never a 25th history hour.
        np.testing.assert_allclose(result.mf_mod, 14.0)


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_three_hour_flux_rejected_before_provider_contraction(inputs, compatible_provider, monkeypatch, phase):
    obs, fp, flux = resolved_inputs(inputs)
    obs = obs.assign_coords(time=obs.time + np.timedelta64(phase, "h"))
    fp = fp.drop_dims("H_back").assign_coords(time=obs.time, H_back=[0, 1, 2, 3])
    fp["fp_time_resolved"] = xr.DataArray(
        np.ones((1, 1, 3, 4)), dims=("lat", "lon", "time", "H_back"), attrs=fp.fp.attrs
    )
    supported = prepare_surface_site(
        obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"}
    )
    np.testing.assert_allclose(supported.mf_mod, 18.)
    operations = surface._surface_operations()
    monkeypatch.setattr(
        surface, "_surface_operations",
        lambda: (*operations[:2], lambda *args: pytest.fail("unsupported cadence reached provider")),
    )
    # Constant flux 2 times (four resolved weights + residual 5) must be 18.
    # At phase 2 the inherited provider returns [16, 18, 18]; reject all phases.
    with pytest.raises(ValueError, match="coarser than the resolved transport grid GCD"):
        prepare_surface_site(
            obs, fp, {"a": flux.isel(time=slice(None, None, 3))},
            species="co2", transport={"a": "time_resolved"},
        )


@pytest.mark.parametrize("change", ["fractional_history", "irregular_history", "release_phase", "flux_phase", "grid_window"])
def test_resolved_rejects_unsupported_history_and_phases(inputs, compatible_provider, change):
    obs, fp, flux = resolved_inputs(inputs)
    message = "whole-hour"
    if change == "fractional_history":
        fp = fp.assign_coords(H_back=[0, 0.5])
    elif change == "irregular_history":
        fp = fp.reindex(H_back=[0, 1, 3], fill_value=1)
    elif change == "release_phase":
        obs = obs.assign_coords(time=obs.time + np.timedelta64(30, "m"))
        fp = fp.assign_coords(time=obs.time)
        message = "phases must align exactly"
    elif change == "flux_phase":
        flux = flux.assign_coords(time=flux.time + np.timedelta64(30, "m"))
        message = "phases must align exactly"
    else:
        obs = obs.assign_coords(time=pd.date_range("2020-01-01", periods=3, freq="2h"))
        fp = fp.assign_coords(time=obs.time, H_back=[0, 2])
        message = "phases must align exactly"
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
    # gcd(2h releases, 3h history) is 1h: hourly flux is supported even though
    # the applicable histories are not contiguous. Three-hour flux is rejected.
    result = prepare_surface_site(obs, fp, {"a": flux}, species="co2", transport={"a": "time_resolved"})
    indices = flux.indexes["time"].get_indexer(releases)
    january = flux.time.dt.month.values == 1
    expected = values[indices] + values[indices - 3] + 5 * values[january].mean()
    np.testing.assert_allclose(result.mf_mod, expected, rtol=1e-14)
    with pytest.raises(ValueError, match=r"transport grid GCD \(1h\)"):
        prepare_surface_site(
            obs, fp, {"a": flux.isel(time=slice(None, None, 3))},
            species="co2", transport={"a": "time_resolved"},
        )


def test_integrated_declared_year_validity_never_reuses_climatological_year(inputs):
    obs, fp, flux = inputs
    obs = obs.assign_coords(time=obs.time + pd.Timedelta(days=366))
    fp = fp.assign_coords(time=obs.time)
    annual = flux.isel(time=[0]).assign_coords(time=[pd.Timestamp("2020-12-01")])
    annual.flux.attrs["time_period"] = "yearly"
    result = prepare_surface_site(obs, fp, {"annual": annual}, species="co2", transport={"annual": "integrated"})
    np.testing.assert_allclose(result.mf_mod, 1.0)
