"""Linked CO2/O2 preparation from per-source native priors (#1051).

Five native sources on a 3 x 4 grid: GPP, TER and FF shared by both tracers
(GPP and TER correlated under one kernel), and independent CO2 and O2 ocean
sources. CO2 and O2 have unequal observation axes. Every prepared product is
checked against a dense reference assembled from the defining formulas.
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
from openghg_inversions.rhime.co2 import Co2O2PreparedInputs
from openghg_inversions.rhime.co2.co2_native_preparation import NATIVE_SOURCE_DIM
from openghg_inversions.rhime.co2.co2_o2_native_preparation import (
    LinkedNativeChannel,
    prepare_multisource_native_co2_o2_inputs,
)
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2
from openghg_inversions.rhime.co2.configuration import Co2MultiSourceNativePrior
from openghg_inversions.utils import areagrid

LAT = [49.0, 50.0, 51.0]
LON = [-3.0, -2.0, -1.0, 0.0]
SOURCES = ("gpp", "ter", "ff", "co2_ocean", "o2_ocean")
STATES = {
    "gpp": ("gpp", "shared"),
    "ter": ("ter", "shared"),
    "ff": ("ff", "shared"),
    "co2_ocean": ("ocean", "co2"),
    "o2_ocean": ("ocean", "o2"),
}
SIGMA = {"gpp": 0.5, "ter": 0.4, "ff": 0.3, "co2_ocean": 0.6, "o2_ocean": 0.8}
MEAN = {"gpp": 1.2, "ter": 0.9, "ff": 1.0, "co2_ocean": 1.1, "o2_ocean": 0.7}
LENGTHS = {"gpp": (1.5, 2.5), "ter": (1.5, 2.5), "ff": (1.0, 1.0), "co2_ocean": (2.0, 2.0), "o2_ocean": (1.0, 3.0)}
RHO = 0.6
NCO2, NO2 = 4, 6


def _prior(restriction: str = "bucket", **marginals: str) -> Co2MultiSourceNativePrior:
    sources = {}
    for source in SOURCES:
        table: dict[str, object] = {
            "sigma": SIGMA[source], "mean": MEAN[source],
            "latitude_correlation_length": LENGTHS[source][0], "longitude_correlation_length": LENGTHS[source][1],
        }
        if source in marginals:
            table["marginal"] = marginals[source]
        sources[source] = table
    params: dict[str, object] = {
        "sources": sources, "gpp_ter_correlation": RHO, "gpp_source": "gpp", "ter_source": "ter",
        "restriction": restriction,
    }
    return Co2MultiSourceNativePrior.from_params(params)


def _partition() -> dict[str, xr.DataArray]:
    land = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 2]])
    ocean = np.array([[1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2]])
    layers = {"gpp": land, "ter": land, "ff": land, "co2_ocean": ocean, "o2_ocean": ocean}
    return {source: xr.DataArray(values, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
            for source, values in layers.items()}


def _flux() -> xr.DataArray:
    rng = np.random.default_rng(1051)
    signs = {"gpp": -1.0, "ter": 1.0, "ff": 1.0, "co2_ocean": -1.0, "o2_ocean": 1.0}
    values = np.stack([signs[source] * rng.uniform(0.5, 2.0, (len(LAT), len(LON))) for source in SOURCES])
    return xr.DataArray(
        values * 1e-6, dims=("source", "lat", "lon"), coords={"source": list(SOURCES), "lat": LAT, "lon": LON},
        attrs={"units": "mol / m2 / s"},
    )


def _channels(errors: bool = True) -> tuple[LinkedNativeChannel, LinkedNativeChannel]:
    rng = np.random.default_rng(20261124)
    co2_index = pd.MultiIndex.from_arrays(
        [["MHD"] * 2 + ["TAC"] * 2, pd.date_range("2021-07-01", periods=NCO2, freq="6h")], names=("site", "time"),
    )
    o2_index = pd.MultiIndex.from_arrays(
        [["MHD"] * 3 + ["WAO"] * 3, pd.date_range("2021-07-01T03", periods=NO2, freq="4h")], names=("site", "time"),
    )
    channels = []
    for name, index, own_ocean, sign in (("co2", co2_index, "co2_ocean", 1.0), ("o2", o2_index, "o2_ocean", -1.4)):
        dim = f"{name}_measure"
        coords = xr.Coordinates.from_pandas_multiindex(index, dim)
        sources = ["gpp", "ter", "ff", own_ocean]
        response = rng.uniform(0.1, 1.0, (len(index), len(sources), len(LAT), len(LON)))
        # O2 responses are signed: the land sources' O2 fluxes oppose CO2.
        response[:, :3] *= sign
        sensitivity = xr.DataArray(
            response, dims=(dim, NATIVE_SOURCE_DIM, "lat", "lon"),
            coords={**coords, NATIVE_SOURCE_DIM: sources, "lat": LAT, "lon": LON},
        )
        observations = xr.DataArray(
            (400.0 if name == "co2" else 0.0) + rng.normal(0.0, 1.0, len(index)), dims=(dim,), coords=coords,
        )
        error = xr.full_like(observations, 0.5) if errors else None
        channels.append(LinkedNativeChannel(observations, sensitivity, units="ppm", independent_error_sd=error))
    return channels[0], channels[1]


def _prepare(restriction: str = "bucket", channels=None, **kwargs) -> Co2O2PreparedInputs:
    co2, o2 = channels or _channels()
    basis = basis_functions_from_flat_basis(flux=_flux(), basis_flat=_partition())
    prior = kwargs.pop("prior", None) or _prior(restriction)
    return prepare_multisource_native_co2_o2_inputs(
        co2=co2, o2=o2, basis=basis, prior=prior, source_states=kwargs.pop("source_states", STATES), **kwargs,
    )


def _dense_b() -> np.ndarray:
    lat, lon = np.meshgrid(LAT, LON, indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()

    def kernel(lengths):
        return np.exp(-np.abs(lat[:, None] - lat[None]) / lengths[0] - np.abs(lon[:, None] - lon[None]) / lengths[1])

    n = lat.size
    b = np.zeros((len(SOURCES) * n, len(SOURCES) * n))
    for i, source in enumerate(SOURCES):
        b[i * n:(i + 1) * n, i * n:(i + 1) * n] = SIGMA[source] ** 2 * kernel(LENGTHS[source])
    cross = RHO * SIGMA["gpp"] * SIGMA["ter"] * kernel(LENGTHS["gpp"])
    b[:n, n:2 * n] = cross
    b[n:2 * n, :n] = cross.T
    return b


def _dense_reference(prepared: Co2O2PreparedInputs, restriction: str) -> dict[str, np.ndarray]:
    """C_alpha, H_alpha, A and the intercept from the defining dense formulas."""
    co2, o2 = _channels()
    n = len(LAT) * len(LON)
    h = np.zeros((NCO2 + NO2, len(SOURCES) * n))
    for rows, channel in ((slice(None, NCO2), co2), (slice(NCO2, None), o2)):
        for source in channel.native_sensitivity[NATIVE_SOURCE_DIM].values:
            i = SOURCES.index(str(source))
            h[rows, i * n:(i + 1) * n] = channel.native_sensitivity.sel({NATIVE_SOURCE_DIM: source}).values.reshape(
                channel.observations.size, n
            )
    native_of = {state: source for source, state in STATES.items()}
    partition = _partition()
    flux = _flux()
    area = areagrid(np.asarray(LAT), np.asarray(LON)).ravel()
    states = prepared.retained_prior.mean.indexes["region"]
    pi = np.zeros((len(states), len(SOURCES) * n))
    for row, (sector, scope, region) in enumerate(states):
        source = native_of[(sector, scope)]
        i = SOURCES.index(source)
        layer = partition[source].values.ravel()
        cells = layer == np.unique(layer)[region]
        weights = np.abs(flux.sel(source=source).values.ravel()) * area if restriction != "bucket" else 1.0
        pi[row, i * n:(i + 1) * n] = cells * weights
    b = _dense_b()
    if restriction == "bucket":
        u = pi.T
        binv_u = np.linalg.solve(b, u)
        pi = np.linalg.solve(u.T @ binv_u, binv_u.T)
    else:
        pi /= pi.sum(axis=1, keepdims=True)
    m = np.concatenate([np.full(n, MEAN[source]) for source in SOURCES])
    c = pi @ b @ pi.T
    h_alpha = h @ b @ pi.T @ np.linalg.inv(c)
    return {
        "C": c, "H": h_alpha, "A": h @ b @ h.T - h_alpha @ c @ h_alpha.T,
        "intercept": h @ m - h_alpha @ pi @ m, "mean": pi @ m,
    }


@pytest.mark.parametrize("restriction", ["bucket", "absolute_flux_weighted_mean"])
def test_linked_native_preparation_matches_the_dense_joint_reduction(restriction: str) -> None:
    prepared = _prepare(restriction)
    expected = _dense_reference(prepared, restriction)
    prior = prepared.retained_prior
    np.testing.assert_allclose(prior.mean.values, expected["mean"], rtol=1e-10)
    np.testing.assert_allclose(prior.arithmetic_covariance.values, expected["C"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(prepared.co2_sensitivity.values, expected["H"][:NCO2], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(prepared.o2_sensitivity.values, expected["H"][NCO2:], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(prepared.aggregation_error.covariance.values, expected["A"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(prepared.fixed_prior_contribution.values, expected["intercept"], rtol=1e-8, atol=1e-10)


def test_retained_states_are_labelled_by_source_and_tracer_scope() -> None:
    prepared = _prepare()
    index = prepared.retained_prior.mean.indexes["region"]
    assert index.names == ["source", "tracer_scope", "region_in_source"]
    assert sorted(set(zip(index.get_level_values(0), index.get_level_values(1), strict=True))) == sorted(STATES.values())
    assert prepared.co2_sensitivity.dims == ("co2_measure", "region")
    assert prepared.o2_sensitivity.dims == ("o2_measure", "region")
    assert prepared.observations.sizes["observation"] == NCO2 + NO2
    assert prepared.o2_co2_flux_ratio is None and "embed" in prepared.o2_co2_flux_ratio_unavailable_reason
    record = prepared.provenance["native_preparation"]
    assert record["source_states"]["o2_ocean"] == ["ocean", "o2"]
    assert record["native_source_priors"]["sources"]["o2_ocean"]["sigma"] == SIGMA["o2_ocean"]


def test_ocean_states_are_independent_and_each_tracer_sees_only_its_own() -> None:
    prepared = _prepare()
    scope = prepared.retained_prior.mean["tracer_scope"].values
    sector = prepared.retained_prior.mean["source"].values
    co2_ocean = (sector == "ocean") & (scope == "co2")
    o2_ocean = (sector == "ocean") & (scope == "o2")
    covariance = prepared.retained_prior.arithmetic_covariance.values
    np.testing.assert_array_equal(covariance[np.ix_(co2_ocean, o2_ocean)], 0.0)
    np.testing.assert_array_equal(covariance[np.ix_(co2_ocean, ~co2_ocean)], 0.0)
    np.testing.assert_array_equal(prepared.co2_sensitivity.values[:, o2_ocean], 0.0)
    np.testing.assert_array_equal(prepared.o2_sensitivity.values[:, co2_ocean], 0.0)
    assert np.abs(prepared.o2_sensitivity.values[:, o2_ocean]).sum() > 0


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_marginals_and_floor_survive_save_and_load(tmp_path: Path, suffix: str) -> None:
    prior = Co2MultiSourceNativePrior.from_params({
        **_prior().to_params(),
        "sources": {
            source: {**table, **({"marginal": "weibull"} if source in ("gpp", "ter") else {})}
            for source, table in _prior().to_params()["sources"].items()
        },
        "latent_correlation_floor": 1e-6,
    })
    prepared = _prepare(prior=prior)
    sector = prepared.retained_prior.mean["source"].values
    expected = np.where(np.isin(sector, ["gpp", "ter"]), "weibull", "lognormal")
    np.testing.assert_array_equal(prepared.retained_prior_marginal.values, expected)
    assert prepared.latent_correlation_floor == 1e-6
    path = tmp_path / f"linked{suffix}"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    xr.testing.assert_identical(restored.retained_prior_marginal, prepared.retained_prior_marginal)
    assert restored.latent_correlation_floor == 1e-6
    xr.testing.assert_allclose(restored.aggregation_error.covariance, prepared.aggregation_error.covariance)


def test_without_marginals_none_are_stored() -> None:
    prepared = _prepare()
    assert prepared.retained_prior_marginal is None and prepared.latent_correlation_floor is None


def test_prepared_inputs_build_the_linked_model() -> None:
    prepared = _prepare(prior=Co2MultiSourceNativePrior.from_params({
        **_prior().to_params(),
        "sources": {
            source: {**table, **({"marginal": "gamma"} if source == "ff" else {})}
            for source, table in _prior().to_params()["sources"].items()
        },
    }))
    with pytensor.config.change_flags(floatX="float64"):
        built = build_rhime_co2_o2(prepared_inputs=prepared)
        with built.model:
            draws = pm.sample_prior_predictive(draws=50, random_seed=1051, var_names=["flux_scaling"])
    assert np.isfinite(draws.prior["flux_scaling"].values).all()
    assert built.output_contract.metadata["prior_marginals"]["gamma"] == 3


def test_rejects_inconsistent_sources_and_scopes() -> None:
    co2, o2 = _channels()
    with pytest.raises(ValueError, match="tracer_scope must be one of"):
        _prepare(source_states={**STATES, "o2_ocean": ("ocean", "ocean")})
    with pytest.raises(ValueError, match="distinct \\(source, tracer_scope\\)"):
        _prepare(source_states={**STATES, "o2_ocean": ("ocean", "co2")}, channels=(
            LinkedNativeChannel(co2.observations, xr.concat(
                [co2.native_sensitivity, o2.native_sensitivity.sel({NATIVE_SOURCE_DIM: ["o2_ocean"]})
                 .isel(o2_measure=slice(NCO2)).rename(o2_measure="co2_measure")
                 .assign_coords(co2.observations.coords)], dim=NATIVE_SOURCE_DIM,
            ), "ppm", co2.independent_error_sd),
            LinkedNativeChannel(o2.observations, o2.native_sensitivity.sel({NATIVE_SOURCE_DIM: ["gpp", "ter", "ff"]}),
                                "ppm", o2.independent_error_sd),
        ))
    leaky = o2.native_sensitivity.reindex({NATIVE_SOURCE_DIM: ["gpp", "ter", "ff", "co2_ocean", "o2_ocean"]},
                                          fill_value=0.0)
    with pytest.raises(ValueError, match="O2 native_sensitivity must hold exactly"):
        _prepare(channels=(co2, LinkedNativeChannel(o2.observations, leaky, "ppm", o2.independent_error_sd)))
    with pytest.raises(ValueError, match="source_states .* differ from basis sources"):
        _prepare(source_states={key: value for key, value in STATES.items() if key != "ff"})
    with pytest.raises(ValueError, match="both channels or neither"):
        _prepare(channels=(co2, LinkedNativeChannel(o2.observations, o2.native_sensitivity, "ppm")))
