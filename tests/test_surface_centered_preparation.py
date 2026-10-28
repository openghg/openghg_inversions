"""Centered surface operands and native observation averaging."""

import json

from dask.callbacks import Callback
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from openghg_inversions.inversion_data import MergedData, SiteOptions, prepare_surface_site
from openghg_inversions.inversion_data import acquisition, surface
from tests.test_surface_acquisition import boundary_inputs, inputs as inputs, resolved_inputs
from tests.test_boundary_getter_policy import boundary_provider as boundary_provider, acquire_monthly_boundary as acquire_monthly_boundary


@pytest.mark.parametrize("start,end,expected", [
    ("2020-01-16", "2020-01-17", [401.]),
    ("2020-02-01", "2020-02-02", [402.]),
    ("2020-01-31", "2020-02-02", [401., 402.]),
])
def test_fresh_centered_acquisition_retains_monthly_boundary(acquire_monthly_boundary, start, end, expected):
    result = acquire_monthly_boundary(start, end, centered=True)
    site = result.site_data["MHD"]
    np.testing.assert_allclose(site.bc_mod.squeeze(), np.repeat(expected, 2))
    np.testing.assert_allclose(site.bc_reference, 400.)
    for direction in "nesw":
        np.testing.assert_allclose(site[f"bc_mass_{direction}"], .25)
    assert result.provenance.boundary.uuid == "boundary-id"
    assert result.provenance.boundary.dataversion == "v3"

def test_native_boundaries_forwarded_and_reconstructed_before_rounding(inputs, monkeypatch):
    obs, fp, flux = inputs
    fp, bc = boundary_inputs(fp)
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
        boundary_representation="centered", boundary_data=bc,
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


@pytest.mark.parametrize("species", ["co2", "o2"])
@pytest.mark.parametrize("north_units,reference_units,output_units", [
    ("ppm", "ppm", "ppm"),
    ("ppm", "1e-06 mol/mol", "ppb"),
    ("1e-06 mol/mol", "ppm", "ppb"),
    ("1e-06 mol/mol", "1e-06 mol/mol", "ppm"),
])
def test_local_boundary_operands_without_upstream_kernel(
    inputs, species, monkeypatch, north_units, reference_units, output_units
):
    import openghg.analyse._modelled_baseline as provider

    monkeypatch.delattr(provider, "baseline_reference_sensitivities", raising=False)
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
    bc.vmr_n.attrs["units"] = north_units
    bc["vmr_e"] = (bc.vmr_e * 1000).assign_attrs(units="1e-09 mol/mol")
    saved_bc, saved_fp = bc.copy(deep=True), fp.copy(deep=True)
    result = prepare_surface_site(
        obs,
        fp,
        {"a": flux},
        species=species,
        transport={"a": "integrated"},
        boundary_representation="centered", boundary_data=bc,
        boundary_reference=400.0,
        boundary_reference_units=reference_units,
        boundary_mass_tolerance=0.001,
        output_units=output_units,
    )
    scale = 1000 if output_units == "ppb" else 1
    np.testing.assert_allclose(result.bc_mod, 400.123456 * scale, rtol=0, atol=1.0e-12 * scale)
    np.testing.assert_allclose(result.bc_anomaly_n, 0.123456 / 4 * scale, rtol=0, atol=1.0e-12 * scale)
    assert result.bc_reference.item() == pytest.approx(400 * scale)
    assert result.bc_anomaly_n.attrs["units"] == result.mf.attrs["units"]
    np.testing.assert_allclose(result.bc_mass_n, 0.25)
    assert result.bc_anomaly_n.dtype == np.float64
    assert result.attrs["boundary_representation"] == "centered-v1"
    assert result.attrs["boundary_mass_tolerance"] == 0.001
    assert result.attrs["boundary_normalization_policy"] == "joint-curtain-unit-mass-v1"
    assert result.attrs["boundary_provider_policy_revision"] == "957-reference-sensitivities-v1"
    assert result.attrs["boundary_operator_owner"] == "openghg_inversions"
    import openghg

    assert result.attrs["surface_provider_revision"] == str(openghg.__revisionid__ or "unknown")
    assert result.attrs["surface_policy_revision"] == "957-source-wise-v1"
    xr.testing.assert_identical(bc, saved_bc)
    xr.testing.assert_identical(fp, saved_fp)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("representation", ["ordinary", "centered"])
def test_sparse_boundary_weights_through_surface_preparation(inputs, lazy, representation):
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
            boundary_representation=representation, boundary_data=bc,
            boundary_reference=400., boundary_reference_units="ppm", boundary_mass_tolerance=0.001,
        )
    if lazy:
        assert isinstance(result.bc_mod.data, da.Array)
        result = result.compute(scheduler="synchronous")
    np.testing.assert_allclose(result.bc_mod.data.todense(), 400.05, rtol=0,
                               atol=1e-12 if representation == "centered" else 2e-5)
    for name, value in fp.data_vars.items():
        assert value.data is borrowed[name]


@pytest.mark.parametrize("target,bad_units", [
    ("reference", "m"), ("reference", "nonsense"),
    ("n", "0 mol/mol"), ("e", "m"), ("e", "nonsense"),
])
def test_surface_rejects_invalid_boundary_units(inputs, target, bad_units):
    obs, fp, flux = inputs
    fp, bc = boundary_inputs(fp)
    reference_units = bad_units if target == "reference" else "1e-06 mol/mol"
    if target != "reference":
        bc[f"vmr_{target}"].attrs["units"] = bad_units
    with pytest.raises(ValueError, match="Invalid boundary units"):
        prepare_surface_site(
            obs, fp, {"a": flux}, species="co2", transport={"a": "integrated"},
            boundary_representation="centered", boundary_data=bc, boundary_reference=400, boundary_reference_units=reference_units,
            boundary_mass_tolerance=0.001,
        )


def test_mapped_boundaries_use_explicit_site_transport(inputs, monkeypatch):
    obs, fp, flux = inputs
    site_fp, bc = boundary_inputs(fp.assign(pblh=fp.pblh + 500))
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
            boundary_representation="centered", boundary_data=bc,
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


def test_boundary_representation_is_explicit_and_ordinary_needs_no_reference(inputs):
    from openghg_inversions.basis._helpers import bc_sensitivity

    obs, fp, flux = inputs
    fluxes = {"a": flux}
    fp = fp.assign_coords(height=[10.])
    bc = xr.Dataset(coords={"height": fp.height, "lat": fp.lat, "lon": fp.lon, "time": fp.time},
                    attrs={"time_period": "1 hour"})
    for direction in "nesw":
        side = "lon" if direction in "ns" else "lat"
        fp[f"particle_locations_{direction}"] = xr.DataArray(
            np.full((1, 1, 3), .25), dims=("height", side, "time"), attrs={"units": "1"}
        )
        bc[f"vmr_{direction}"] = xr.DataArray(
            np.full((1, 1, 3), 400.05), dims=("height", side, "time"), attrs={"units": "ppm"}
        )
    options = dict(species="co2", transport={source: "integrated" for source in fluxes},
                   boundary_data=bc, boundary_mass_tolerance=1e-6)
    ordinary = prepare_surface_site(obs, fp, fluxes, **options)
    referenced = prepare_surface_site(obs, fp, fluxes, **options,
                                      boundary_reference=400., boundary_reference_units="ppm")
    centered = prepare_surface_site(obs, fp, fluxes, **options, boundary_representation="centered",
                                    boundary_reference=400., boundary_reference_units="ppm")
    for result in (ordinary, referenced):
        assert "bc_reference" not in result and "bc_anomaly_n" not in result
        assert "boundary_representation" not in result.attrs
        np.testing.assert_allclose(result.bc_mod, 400.05, atol=2e-5, rtol=0)
    raw = bc_sensitivity({"TOY": ordinary})["TOY"].H_bc
    centered_raw = bc_sensitivity({"TOY": centered})["TOY"]
    xr.testing.assert_allclose(raw, centered_raw.H_bc_anomaly + centered_raw.bc_reference * centered_raw.H_bc_mass)
    with pytest.raises(ValueError, match="require reference"):
        prepare_surface_site(obs, fp, fluxes, **options, boundary_representation="centered")


@pytest.mark.parametrize("transport,use_bc", [(None, True), ({"a": "integrated"}, False)])
def test_centered_acquisition_rejects_ignored_choice(monkeypatch, transport, use_bc):
    monkeypatch.setattr(acquisition, "get_flux_data", lambda **kw: pytest.fail("retrieved ignored centering choice"))
    with pytest.raises(ValueError, match="requires surface_transport and use_bc"):
        MergedData.from_options(
            species="co2", site_options=SiteOptions.from_inputs(sites=["MHD"], averaging_period="1h"),
            domain="EUROPE", start_date="2020-01-01", end_date="2020-02-01", flux_sources=["a"],
            surface_transport=transport, use_bc=use_bc, boundary_representation="centered",
        )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("mode", ["integrated", "time_resolved"])
@pytest.mark.parametrize("representation", ["ordinary", "centered"])
@pytest.mark.parametrize("extra_releases", [False, True])
def test_native_products_precede_observation_averaging(inputs, mode, lazy, representation, extra_releases):
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
        boundary_representation=representation, boundary_reference=400.,
        boundary_reference_units="ppm", boundary_mass_tolerance=1e-6,
    )
    # Integrated native products: mean(1*1, 3*3)=5; resolved uses the two-hour history mean=2.
    np.testing.assert_allclose(result.mf_mod, 5. if mode == "integrated" else 4.)
    np.testing.assert_allclose(result.bc_mod, 402.)
    if representation == "ordinary":
        np.testing.assert_allclose(result.bc_n, (400. * .1 + 404. * .4) / 2)
    else:
        np.testing.assert_allclose(result.bc_anomaly_n, .8)
        np.testing.assert_allclose(result.bc_mass_n, .25)
        assert result.bc_reference.dims == ()
    for value, saved in zip((obs, fp, flux, bc), before, strict=True):
        xr.testing.assert_identical(value, saved)
