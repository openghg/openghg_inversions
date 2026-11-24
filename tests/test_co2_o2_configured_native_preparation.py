"""Configured linked CO2/O2 preparation from native responses (#1051, B2).

Offline synthetic surface data for both tracers on a 3 x 4 grid: CO2 and O2
each have GPP, TER, FF and ocean flux sources (O2 land fluxes are signed
O2 fluxes), observed at one site on different times. Configuration maps the
shared land sources and the two independent ocean sources onto five native
sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import basis_functions_from_flat_basis
from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.inversion_data.surface import prepare_surface_site
from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs
from openghg_inversions.rhime.co2.co2_native_preparation import NATIVE_SOURCE_DIM
from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2PreparationConfig, load_co2_o2_params
from openghg_inversions.rhime.co2.co2_o2_native_preparation import (
    LinkedNativeChannel,
    prepare_multisource_native_co2_o2_inputs,
)
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2

WINDOW = {"start_date": "2020-01-16", "end_date": "2020-01-17", "domain": "TOY"}
LAT = [49.0, 50.0, 51.0]
LON = [-3.0, -2.0, -1.0, 0.0]
NATIVE = ("gpp", "ter", "ff", "co2_ocean", "o2_ocean")
# Native source -> (sector, scope, CO2 flux source, O2 flux source).
STATES = {
    "gpp": ("gpp", "shared", "co2_gpp", "o2_gpp"),
    "ter": ("ter", "shared", "co2_ter", "o2_ter"),
    "ff": ("ff", "shared", "co2_ff", "o2_ff"),
    "co2_ocean": ("ocean", "co2", "co2_ocean", None),
    "o2_ocean": ("ocean", "o2", None, "o2_ocean"),
}
RATIO = {"gpp": -1.1, "ter": -1.1, "ff": -1.4}
TIMES = {"co2": pd.date_range("2020-01-16", periods=8, freq="3h"),
         "o2": pd.date_range("2020-01-16T01:30", periods=6, freq="3h")}


def _channel_data(channel: str) -> MergedData:
    rng = np.random.default_rng({"co2": 1, "o2": 2}[channel])
    times = TIMES[channel]
    grid = {"lat": LAT, "lon": LON}
    shape = (len(LAT), len(LON))
    obs = xr.Dataset(
        {
            channel: ("time", (402.0 if channel == "co2" else 5.0) + rng.normal(0.0, 1.0, len(times)), {"units": "ppm"}),
            f"{channel}_repeatability": ("time", np.full(len(times), 0.3), {"units": "ppm"}),
            f"{channel}_variability": ("time", np.full(len(times), 0.2), {"units": "ppm"}),
        },
        coords={"time": times},
        attrs={"platform": "surface", "sampling_period": 10800, "scale": "WMO"},
    )
    fp = xr.Dataset(
        {"fp": (("lat", "lon", "time"), rng.uniform(0.2, 3.0, (*shape, len(times))), {"units": "s m2 / mol"})},
        coords={**grid, "time": times, "height": [10.0]},
        attrs={"model": "NAME", "met_model": "toy"},
    )
    fp["release_lon"] = xr.DataArray(np.zeros(len(times)), dims="time")
    fp["release_lat"] = xr.DataArray(np.full(len(times), 50.0), dims="time")
    base = np.random.default_rng(7)
    patterns = {"gpp": -base.uniform(0.5, 1.5, shape), "ter": base.uniform(0.3, 1.2, shape),
                "ff": base.uniform(0.5, 2.0, shape)}
    ocean = {"co2": -base.uniform(0.2, 0.8, shape), "o2": base.uniform(0.5, 1.5, shape)}
    # The O2:CO2 ratio varies in space, as in a native paired O2 flux.
    ratio_field = np.random.default_rng(11).uniform(0.6, 1.4, shape)
    values = {f"{channel}_{sector}": pattern * (1.0 if channel == "co2" else RATIO[sector] * ratio_field)
              for sector, pattern in patterns.items()}
    values[f"{channel}_ocean"] = ocean[channel]
    fluxes = {
        source: xr.Dataset(
            {"flux": (("lat", "lon", "time"), np.repeat(value[..., None], len(times), axis=2) * 1e-6,
                      {"units": "mol / m2 / s"})},
            coords={**grid, "time": times},
        )
        for source, value in values.items()
    }
    bc = xr.Dataset(coords={**grid, "height": [10.0], "time": [pd.Timestamp("2020-01-01")]},
                    attrs={"time_period": "1 month"})
    reference = 400.0 if channel == "co2" else 5.0
    for direction, offset in zip("nesw", [0.1, -0.2, 0.3, 0.0], strict=True):
        side = "lon" if direction in "ns" else "lat"
        fp[f"particle_locations_{direction}"] = xr.DataArray(
            np.full((1, fp.sizes[side], len(times)), 0.25 / fp.sizes[side]), dims=("height", side, "time"),
            attrs={"units": "1"},
        )
        bc[f"vmr_{direction}"] = xr.DataArray(
            np.full((1, fp.sizes[side], 1), reference + offset), dims=("height", side, "time"), attrs={"units": "ppm"},
        )
    site = prepare_surface_site(
        obs, fp, fluxes, species=channel, transport=dict.fromkeys(fluxes, "integrated"), boundary_data=bc,
        boundary_reference=reference, boundary_reference_units="ppm", boundary_mass_tolerance=1e-6,
    )
    return MergedData(
        site_data={"TOY": site}, flux_data=fluxes, boundary_data=bc,
        site_options=SiteOptions.from_inputs(sites=["TOY"], averaging_period="3h"),
    )


@pytest.fixture(scope="module")
def merged() -> dict[str, MergedData]:
    return {channel: _channel_data(channel) for channel in ("co2", "o2")}


def _partition() -> xr.DataArray:
    land = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 2]])
    ocean = np.array([[1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2]])
    return xr.DataArray(
        np.stack([land, land, land, ocean, ocean]), dims=("source", "lat", "lon"),
        coords={"source": list(NATIVE), "lat": LAT, "lon": LON}, name="basis",
    )


@pytest.fixture(scope="module")
def partition_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("linked") / "partition.nc"
    _partition().to_dataset().to_netcdf(path)
    return path


def _acquisition(channel: str) -> dict:
    sources = [STATES[native][2 if channel == "co2" else 3] for native in NATIVE]
    sources = [source for source in sources if source is not None]
    return {
        "sites": ["TOY"], "averaging_period": "3h", "flux_sources": sources,
        "surface_transport": dict.fromkeys(sources, "integrated"),
        "boundary_reference": 400.0 if channel == "co2" else 5.0, "boundary_reference_units": "ppm",
        "boundary_mass_tolerance": 1e-6,
    }


def _source_states(flux_tracer: str | None = None) -> dict:
    states = {}
    for native, (sector, scope, co2, o2) in STATES.items():
        table = {"source": sector, "tracer_scope": scope}
        if co2 is not None:
            table["co2_flux_source"] = co2
        if o2 is not None:
            table["o2_flux_source"] = o2
        if flux_tracer is not None and scope == "shared":
            table["flux_tracer"] = flux_tracer
        states[native] = table
    return states


def _native_prior(restriction: str = "bucket", **marginals: str) -> dict:
    sigma = {"gpp": 0.5, "ter": 0.4, "ff": 0.3, "co2_ocean": 0.6, "o2_ocean": 0.8}
    lengths = {"latitude_correlation_length": 1.5, "longitude_correlation_length": 2.0}
    return {
        "restriction": restriction, "gpp_ter_correlation": 0.6, "gpp_source": "gpp", "ter_source": "ter",
        "sources": {source: {"sigma": value, "mean": 1.0, **lengths,
                             **({"marginal": marginals[source]} if source in marginals else {})}
                    for source, value in sigma.items()},
    }


def _request(partition_path: Path, *, restriction: str = "bucket", flux_tracer: str | None = None, **extra) -> dict:
    return {
        "format_version": 1, "recipe": "co2_o2",
        "prepared_inputs": {
            **WINDOW,
            "acquisition": {channel: _acquisition(channel) for channel in ("co2", "o2")},
            "basis": {"partition_path": str(partition_path)},
            "native_prior": extra.pop("native_prior", _native_prior(restriction)),
            "source_states": _source_states(flux_tracer),
            **extra,
        },
        "channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}},
    }


def _prepare(tmp_path: Path, request: dict, merged: dict[str, MergedData], name: str = "prepare") -> handoffs.Co2O2Preparation:
    arguments = Co2O2PreparationConfig.from_params(load_co2_o2_params(params=request), require_source=True)
    manifest = handoffs.prepare(**arguments.preparation_arguments(), merged_data=merged, output_dir=tmp_path / name)
    return handoffs.load_preparation(manifest)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_configuration_parses_each_channel_and_source_state(partition_path: Path) -> None:
    config = Co2O2PreparationConfig.from_params(load_co2_o2_params(params=_request(partition_path)), require_source=True)
    native = config.native
    assert native is not None and config.prepared_path is None
    assert native.flux_sources("co2") == {"co2_gpp": "gpp", "co2_ter": "ter", "co2_ff": "ff", "co2_ocean": "co2_ocean"}
    assert native.flux_sources("o2") == {"o2_gpp": "gpp", "o2_ter": "ter", "o2_ff": "ff", "o2_ocean": "o2_ocean"}
    assert native.source_states["o2_ocean"].tracer_scope == "o2"
    assert native.native_prior.sources["o2_ocean"].sigma == 0.8
    assert native.acquisition["o2"]["boundary_reference"] == 5.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["prepared_inputs"]["native_prior"].update(restriction="absolute_flux_weighted_mean"),
         r"source_states.gpp.flux_tracer is required"),
        (lambda r: r["prepared_inputs"]["source_states"]["co2_ocean"].update(o2_flux_source="o2_ocean"),
         r"o2_flux_source is not allowed"),
        (lambda r: r["prepared_inputs"]["source_states"]["ff"].pop("o2_flux_source"),
         r"o2_flux_source is required"),
        (lambda r: r["prepared_inputs"]["source_states"]["co2_ocean"].update(flux_tracer="co2"),
         r"flux_tracer applies only to shared"),
        (lambda r: r["prepared_inputs"]["source_states"]["ff"].update(co2_flux_source="co2_gpp"),
         r"map each co2 flux source .* exactly once"),
        (lambda r: r["prepared_inputs"]["source_states"].pop("o2_ocean"),
         r"source_states must name exactly the native sources"),
        (lambda r: r["prepared_inputs"].update(aggregation_error_rank=256), r"aggregation_error_rank is not available"),
        (lambda r: r["prepared_inputs"]["basis"].pop("partition_path"),
         r"requires prepared_inputs.basis.partition_path"),
        (lambda r: r["prepared_inputs"].update(path="saved.nc"), r"remove the build options"),
    ],
)
def test_configuration_rejects_inconsistent_choices(partition_path: Path, change, message) -> None:
    request = _request(partition_path)
    change(request)
    with pytest.raises(ValueError, match=message):
        Co2O2PreparationConfig.from_params(load_co2_o2_params(params=request), require_source=True)


def test_flux_weighting_tracer_is_accepted_when_set(partition_path: Path) -> None:
    request = _request(partition_path, restriction="absolute_flux_weighted_mean", flux_tracer="o2")
    config = Co2O2PreparationConfig.from_params(load_co2_o2_params(params=request), require_source=True)
    assert config.native.source_states["gpp"].flux_tracer == "o2"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("representation", ["ordinary", "centered"])
def test_configured_preparation_saves_both_channels(
    tmp_path: Path, partition_path: Path, merged: dict[str, MergedData], representation: str,
) -> None:
    request = _request(partition_path, boundary_representation=representation)
    preparation = _prepare(tmp_path, request, merged)
    prepared = preparation.prepared
    species = prepared.observations["species"].values
    assert (species == "co2").sum() == len(TIMES["co2"]) and (species == "o2").sum() == len(TIMES["o2"])
    assert set(prepared.observations["observation_units"].values) == {"ppm"}
    assert prepared.independent_error_sd is not None
    index = prepared.retained_prior.mean.indexes["region"]
    assert set(zip(index.get_level_values("source"), index.get_level_values("tracer_scope"), strict=True)) == {
        (sector, scope) for sector, scope, _, _ in STATES.values()
    }
    if representation == "centered":
        assert set(prepared.boundary_operands) == {"co2", "o2"} and not prepared.boundary_sensitivity
    else:
        assert set(prepared.boundary_sensitivity) == {"co2", "o2"} and not prepared.boundary_operands
    record = prepared.provenance["native_preparation"]
    assert record["flux_sources"]["o2"]["o2_ocean"] == "o2_ocean"
    assert record["boundary_representation"] == {"co2": representation, "o2": representation}
    assert (preparation.start_date, preparation.end_date, preparation.domain) == (
        WINDOW["start_date"], WINDOW["end_date"], WINDOW["domain"],
    )


def test_configured_preparation_matches_the_direct_linked_reduction(
    tmp_path: Path, partition_path: Path, merged: dict[str, MergedData],
) -> None:
    """The configured route gathers the same responses the Python core reduces."""
    prepared = _prepare(tmp_path, _request(partition_path), merged).prepared
    channels = {}
    for channel, data in merged.items():
        site = data.site_data["TOY"]
        dim = f"{channel}_measure"
        index = pd.MultiIndex.from_arrays([["TOY"] * len(TIMES[channel]), TIMES[channel]], names=("site", "time"))
        coords = xr.Coordinates.from_pandas_multiindex(index, dim)
        renamed = {flux: native for native, (_, _, co2, o2) in STATES.items()
                   for flux in ((co2 if channel == "co2" else o2),) if flux is not None}
        response = (site.fp_x_flux_sectoral.sel(source=list(renamed)).transpose("time", "source", "lat", "lon")
                    .rename(time=dim, source=NATIVE_SOURCE_DIM).drop_vars(dim)
                    .assign_coords({**coords, NATIVE_SOURCE_DIM: list(renamed.values())}))
        observations = xr.DataArray(site.mf.values, dims=(dim,), coords=coords)
        channels[channel] = LinkedNativeChannel(observations, response, "ppm")
    flux = xr.concat(
        [merged["co2" if STATES[n][1] != "o2" else "o2"].flux_data[STATES[n][2] or STATES[n][3]].flux
         .mean("time").expand_dims(source=[n]) for n in NATIVE], dim="source",
    )
    direct = prepare_multisource_native_co2_o2_inputs(
        co2=channels["co2"], o2=channels["o2"],
        basis=basis_functions_from_flat_basis(flux=flux, basis_flat={n: _partition().sel(source=n, drop=True)
                                                                      for n in NATIVE}),
        prior=Co2O2PreparationConfig.from_params(load_co2_o2_params(params=_request(partition_path)),
                                                 require_source=True).native.native_prior,
        source_states={n: STATES[n][:2] for n in NATIVE},
    )
    np.testing.assert_allclose(prepared.retained_prior.arithmetic_covariance.values,
                               direct.retained_prior.arithmetic_covariance.values, rtol=1e-10)
    np.testing.assert_allclose(prepared.co2_sensitivity.values, direct.co2_sensitivity.values, rtol=1e-10)
    np.testing.assert_allclose(prepared.o2_sensitivity.values, direct.o2_sensitivity.values, rtol=1e-10)
    np.testing.assert_allclose(prepared.aggregation_error.covariance.values,
                               direct.aggregation_error.covariance.values, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(prepared.observations.values, direct.observations.values)


def test_flux_tracer_selects_the_weighting_flux(tmp_path: Path, partition_path: Path, merged) -> None:
    results = {
        tracer: _prepare(tmp_path, _request(partition_path, restriction="absolute_flux_weighted_mean",
                                            flux_tracer=tracer), merged, name=tracer).prepared
        for tracer in ("co2", "o2")
    }
    scope = results["co2"].retained_prior.mean["tracer_scope"].values
    shared = scope == "shared"
    covariance = {tracer: result.retained_prior.arithmetic_covariance.values for tracer, result in results.items()}
    # The ocean states are weighted by their own tracer's flux either way; shared states follow flux_tracer.
    np.testing.assert_allclose(covariance["co2"][np.ix_(~shared, ~shared)], covariance["o2"][np.ix_(~shared, ~shared)])
    assert not np.allclose(covariance["co2"][np.ix_(shared, shared)], covariance["o2"][np.ix_(shared, shared)])
    assert results["o2"].provenance["native_preparation"]["source_state_options"]["gpp"]["flux_tracer"] == "o2"


def test_configured_marginals_build_the_linked_model(tmp_path: Path, partition_path: Path, merged) -> None:
    request = _request(partition_path, native_prior=_native_prior(gpp="weibull", ter="weibull"))
    prepared = _prepare(tmp_path, request, merged).prepared
    sector = prepared.retained_prior.mean["source"].values
    np.testing.assert_array_equal(prepared.retained_prior_marginal.values,
                                  np.where(np.isin(sector, ["gpp", "ter"]), "weibull", "lognormal"))
    with pytensor.config.change_flags(floatX="float64"):
        built = build_rhime_co2_o2(prepared_inputs=prepared)
        with built.model:
            draws = pm.sample_prior_predictive(draws=20, random_seed=1051, var_names=["flux_scaling"])
    assert np.isfinite(draws.prior["flux_scaling"].values).all()
    assert built.output_contract.metadata["prior_marginals"]["weibull"] == 6  # three regions each
