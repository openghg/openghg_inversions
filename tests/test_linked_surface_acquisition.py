"""Independent native tracer acquisition before any linked-model assembly."""

from types import SimpleNamespace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import MergedData, SiteOptions, prepare_surface_site
from openghg_inversions.inversion_data import acquisition
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from tests.test_surface_acquisition import boundary_inputs, inputs as inputs


@pytest.fixture
def native_channel(inputs):
    """Tiny labelled channels with distinct references and a missing leading row."""
    def build(species):
        obs, footprint, flux = (dataset.copy(deep=True) for dataset in inputs)
        footprint, boundary = boundary_inputs(footprint)
        reference = 400.0 if species == "co2" else 209300.0
        perturbation = np.array([0.0, 1.0e-4, 2.0e-4], dtype=np.float64)
        obs = obs.rename({name: name.replace("co2", species) for name in obs.data_vars})
        obs[species] = xr.DataArray(
            reference + perturbation, dims="time", coords={"time": obs.time}, attrs={"units": "ppm"},
        )
        obs = obs.isel(time=slice(1, None)).assign_attrs(species=species, site="TEST", inlet="10m")
        for direction in "nesw":
            values = np.broadcast_to((reference + perturbation) * 1e-6, (1, 1, 3)).copy()
            boundary[f"vmr_{direction}"] = xr.DataArray(
                values, dims=("height", "lon" if direction in "ns" else "lat", "time"),
                attrs={"units": "mol/mol"},
            )
        other = flux.copy(deep=True)
        other["flux"] = (-2 * other.flux).assign_attrs(units="mol / m2 / s")
        return obs, footprint.chunk({"time": 2}), {
            "biosphere": flux.chunk({"time": 2}), "ocean": other.chunk({"time": 2}),
        }, boundary.chunk({"time": 2}), reference
    return build


@pytest.mark.parametrize("species", ["co2", "o2"])
def test_independent_native_acquisition_matches_supplied_surface(monkeypatch, native_channel, species):
    obs, footprint, fluxes, boundary, reference = native_channel(species)
    calls = {}

    def wrapper(dataset, name):
        return SimpleNamespace(data=dataset, metadata={"uuid": f"{species}-{name}", "dataversion": "v2"})

    def get_obs(**kwargs):
        calls["obs"] = kwargs
        return wrapper(obs, "obs")

    def get_footprint(**kwargs):
        calls["footprint"] = kwargs
        return wrapper(footprint, "footprint")

    def get_flux(**kwargs):
        calls["flux"] = kwargs
        return {name: wrapper(dataset, name) for name, dataset in fluxes.items()}

    def search_boundary(**kwargs):
        calls["boundary_search"] = kwargs
        return SimpleNamespace(results=pd.DataFrame({
            "uuid": [f"{species}-boundary"], "time_period": ["1 hour"],
        }))

    def get_boundary(**kwargs):
        calls["boundary"] = kwargs
        return wrapper(boundary, "boundary")

    monkeypatch.setattr(acquisition, "get_obs_data", get_obs)
    monkeypatch.setattr(acquisition, "get_footprint_data", get_footprint)
    monkeypatch.setattr(acquisition, "get_flux_data", get_flux)
    monkeypatch.setattr(acquisition, "search_bc", search_boundary)
    monkeypatch.setattr(acquisition, "get_bc", get_boundary)
    transport = dict.fromkeys(fluxes, "integrated")
    before = footprint.copy(deep=False)
    with Callback(pretask=lambda *args: pytest.fail("acquisition computed borrowed payload")):
        merged = MergedData.from_options(
            species=species,
            site_options=SiteOptions.from_inputs(
                sites=["TEST"], inlet="10m", fp_height="10m", averaging_period="1h", platform="surface",
            ),
            domain="EUROPE", start_date="2020-01-01", end_date="2020-01-02",
            flux_sources=list(fluxes), surface_transport=transport, split_by_sectors=True,
            obs_store="offline-obs", footprint_store="offline-fp", flux_store="offline-flux",
            bc_store="offline-bc", bc_input="offline-curtains", fp_species="co2", use_bc=True,
            boundary_reference=reference, boundary_reference_units="ppm", boundary_mass_tolerance=1e-6,
        )
        supplied = prepare_surface_site(
            obs, footprint, fluxes, species=species, transport=transport, boundary_data=boundary,
            boundary_reference=reference,
            boundary_reference_units="ppm", boundary_mass_tolerance=1e-6, output_units="ppm",
        )
    site = merged.site_data["TEST"]
    assert merged.acquisition.species == species
    assert merged.site_options.inlet == ("10m",)
    assert merged.acquisition.start_date == "2020-01-01"
    assert merged.acquisition.end_date == "2020-01-02"
    assert site.attrs["species"] == species
    assert site.time.identical(obs.time)
    assert site.sizes["time"] == 2
    assert site.bc_reference.dims == ()
    assert site.bc_reference.item() == pytest.approx(reference)
    for field in ("mf", "bc_mod", "fp_x_flux_sectoral", "mf_mod_sectoral"):
        xr.testing.assert_allclose(site[field], supplied[field])
        assert mole_fraction_unit_scale(site[field].attrs["units"], context=field) == pytest.approx(1e-6)
    assert set(site.source.values) == set(fluxes)
    assert isinstance(site.fp_x_flux_sectoral.data, da.Array)
    assert isinstance(site.bc_anomaly_n.data, da.Array)
    xr.testing.assert_identical(footprint, before)
    for source in fluxes:
        assert merged.flux_data[source].flux.data is fluxes[source].flux.data
        assert merged.provenance.flux[source].dataversion == "v2"
    assert merged.provenance.observations["TEST"].uuid == f"{species}-obs"
    assert merged.provenance.footprints["TEST"].dataversion == "v2"
    assert merged.provenance.boundary.dataversion == "v2"
    assert calls["obs"]["species"] == calls["flux"]["species"] == species
    assert calls["obs"]["site"] == calls["footprint"]["site"] == "TEST"
    assert calls["obs"]["inlet"] == calls["footprint"]["fp_height"] == "10m"
    assert calls["boundary"]["uuid"] == [f"{species}-boundary"]
    assert calls["flux"]["allow_older_flux"] is False
    np.testing.assert_allclose(site.mf_mod_sectoral.sel(source="biosphere"), [2., 3.])
    np.testing.assert_allclose(site.mf_mod_sectoral.sel(source="ocean"), [-4., -6.])
    np.testing.assert_allclose(site.bc_mod, reference + np.array([1e-4, 2e-4]), rtol=0, atol=6e-11)
    np.testing.assert_allclose(
        sum(site[f"bc_anomaly_{direction}"].sum(("height", "lon" if direction in "ns" else "lat"))
            for direction in "nesw"), [1e-4, 2e-4], rtol=0, atol=6e-11,
    )


def test_o2_observations_convert_once_and_reference_is_independent(native_channel):
    obs, footprint, fluxes, boundary, reference = native_channel("o2")
    obs["o2"] = (obs.o2 * 1e-6).assign_attrs(units="mol/mol")
    result = prepare_surface_site(
        obs, footprint, fluxes, species="o2", transport=dict.fromkeys(fluxes, "integrated"),
        boundary_data=boundary, boundary_reference=209000.,
        boundary_reference_units="ppm", boundary_mass_tolerance=1e-6, output_units="ppm",
    )
    np.testing.assert_allclose(result.mf, reference + np.array([1e-4, 2e-4]), rtol=0, atol=6e-11)
    np.testing.assert_allclose(result.bc_mod, result.mf, rtol=0, atol=6e-11)
    assert result.bc_reference.item() == 209000.
    assert result.mf.dtype == result.bc_mod.dtype == np.dtype("float64")


@pytest.mark.parametrize("invalid", ["missing_endpoint", "flux_units"])
def test_native_channel_rejects_invalid_scientific_inputs(native_channel, invalid):
    obs, footprint, fluxes, boundary, reference = native_channel("o2")
    if invalid == "missing_endpoint":
        footprint = footprint.drop_vars("particle_locations_w")
    else:
        fluxes["ocean"].flux.attrs["units"] = "kg"
    message = "particle_locations_w" if invalid == "missing_endpoint" else "incompatible units"
    with pytest.raises(ValueError, match=message):
        prepare_surface_site(
            obs, footprint, fluxes, species="o2", transport=dict.fromkeys(fluxes, "integrated"),
            boundary_data=boundary, boundary_reference=reference,
            boundary_reference_units="ppm", boundary_mass_tolerance=1e-6,
        )
