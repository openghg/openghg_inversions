"""Configured per-source native priors: configuration, joint reduction and staged handoffs.

The synthetic case has three signed sources (FF, GPP, TER) on a 3 x 4 grid
with two native classes. GPP and TER are correlated under one shared
class-blocked separable kernel; FF uses a geodesic kernel in one class and
the separable kernel in the other. Every prepared product is checked against
a dense reference assembled here from the defining formulas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from examples.surface_centered_pilot import acquire_supplied
from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.native_covariance import DEFAULT_EARTH_RADIUS_KM
from openghg_inversions.rhime.co2 import handoffs
from openghg_inversions.rhime.co2.co2_affine_output import load_and_bind_affine_flux_map
from openghg_inversions.rhime.co2.configuration import (
    Co2MultiSourceNativePrior,
    Co2NativePrior,
    Co2PreparationConfig,
)
from openghg_inversions.rhime.co2.execution import load_co2_params, run_co2
from openghg_inversions.utils import areagrid

WINDOW = {"start_date": "2020-01-16", "end_date": "2020-01-17", "domain": "TOY"}
SOURCES = ("ff", "gpp", "ter")  # surface acquisition orders sources by label
LAT = [49.0, 50.0, 51.0]
LON = [-3.0, -2.0, -1.0, 0.0]
RHO = 0.7
SIGMA = {"ff": 0.3, "gpp": 0.5, "ter": 0.4}
MEAN = {"ff": 1.0, "gpp": 1.2, "ter": 0.9}
BIO_LENGTHS = (1.5, 2.5)  # latitude, longitude in degrees
FF_LENGTHS = (1.0, 1.0)
FF_GEODESIC_KM = 150.0


def _surface_inputs() -> tuple[xr.Dataset, xr.Dataset, dict[str, xr.Dataset], xr.Dataset]:
    """Eight releases on a 3 x 4 grid with three signed flux sources."""
    rng = np.random.default_rng(20261113)
    times = pd.date_range("2020-01-16", periods=8, freq="3h")
    grid = {"lat": LAT, "lon": LON}
    shape = (len(LAT), len(LON))
    obs = xr.Dataset(
        {
            "co2": ("time", 402.0 + rng.normal(0.0, 1.0, len(times)), {"units": "ppm"}),
            "co2_repeatability": ("time", np.full(len(times), 0.3), {"units": "ppm"}),
            "co2_variability": ("time", np.full(len(times), 0.2), {"units": "ppm"}),
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
    patterns = {
        "ff": rng.uniform(0.5, 2.0, shape),
        "gpp": -rng.uniform(0.5, 1.5, shape),
        "ter": rng.uniform(0.3, 1.2, shape),
    }
    fluxes = {
        source: xr.Dataset(
            {"flux": (("lat", "lon", "time"), np.repeat(values[..., None], len(times), axis=2) * 1e-6,
                      {"units": "mol / m2 / s"})},
            coords={**grid, "time": times},
        )
        for source, values in patterns.items()
    }
    bc = xr.Dataset(coords={**grid, "height": [10.0], "time": [pd.Timestamp("2020-01-01")]},
                    attrs={"time_period": "1 month"})
    for direction, concentration in zip("nesw", [400.1, 399.8, 400.3, 400.0], strict=True):
        side = "lon" if direction in "ns" else "lat"
        fp[f"particle_locations_{direction}"] = xr.DataArray(
            np.full((1, fp.sizes[side], len(times)), 0.25 / fp.sizes[side]), dims=("height", side, "time"),
            attrs={"units": "1"},
        )
        bc[f"vmr_{direction}"] = xr.DataArray(
            np.full((1, fp.sizes[side], 1), concentration), dims=("height", side, "time"), attrs={"units": "ppm"},
        )
    return obs, fp, fluxes, bc


@pytest.fixture(scope="module")
def merged() -> MergedData:
    """Offline three-source surface acquisition with centered boundary operands."""
    obs, fp, fluxes, bc = _surface_inputs()
    return MergedData(
        site_data={"TOY": acquire_supplied(obs, fp, fluxes, bc)}, flux_data=fluxes, boundary_data=bc,
        site_options=SiteOptions.from_inputs(sites=["TOY"], averaging_period="3h"),
    )


def _partition() -> xr.DataArray:
    """Source-specific regions: FF has three, GPP and TER two each."""
    ff = np.array([[1, 1, 2, 2], [1, 3, 3, 2], [3, 3, 3, 2]])
    bio = np.array([[1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2]])
    return xr.DataArray(
        np.stack([ff, bio, bio]), dims=("source", "lat", "lon"),
        coords={"source": list(SOURCES), "lat": LAT, "lon": LON}, name="basis",
    )


def _classes() -> xr.DataArray:
    """Two native classes shared by all sources (west 'land' 1, east 'sea' 2)."""
    return xr.DataArray(
        np.array([[1, 1, 1, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int64), dims=("lat", "lon"),
        coords={"lat": LAT, "lon": LON}, name="class_labels",
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Partition, class labels and retained-region files."""
    root = tmp_path_factory.mktemp("multisource")
    paths = {"partition": root / "partition.nc", "classes": root / "classes.nc", "retained": root / "retained.json"}
    _partition().to_dataset().to_netcdf(paths["partition"])
    _classes().to_dataset().to_netcdf(paths["classes"])
    paths["retained"].write_text(json.dumps({"ff": [1, 3], "gpp": [1, 2], "ter": [2]}))
    return paths


def _native_prior(files: dict[str, Path], restriction: str = "bucket", **extra: object) -> dict:
    """The per-source native prior table of the synthetic case."""
    bio = {"latitude_correlation_length": BIO_LENGTHS[0], "longitude_correlation_length": BIO_LENGTHS[1]}
    return {
        "restriction": restriction,
        "class_labels_path": str(files["classes"]),
        "gpp_ter_correlation": RHO, "gpp_source": "gpp", "ter_source": "ter",
        "sources": {
            "gpp": {"sigma": SIGMA["gpp"], "mean": MEAN["gpp"], **bio},
            "ter": {"sigma": SIGMA["ter"], "mean": MEAN["ter"], **bio},
            "ff": {
                "sigma": SIGMA["ff"], "mean": MEAN["ff"],
                "latitude_correlation_length": FF_LENGTHS[0], "longitude_correlation_length": FF_LENGTHS[1],
                "geodesic_correlation_length_km": {"1": FF_GEODESIC_KM},
            },
        },
        **extra,
    }


def _request(files: dict[str, Path], native_prior: dict) -> dict:
    """A complete configured CO2 request building a per-source preparation."""
    return {
        "format_version": 1, "recipe": "co2",
        "prepared_inputs": {
            **WINDOW,
            "acquisition": {
                "sites": ["TOY"], "averaging_period": "3h", "flux_sources": list(SOURCES),
                "surface_transport": dict.fromkeys(SOURCES, "integrated"),
                "boundary_reference": 400.0, "boundary_reference_units": "ppm", "boundary_mass_tolerance": 1e-6,
            },
            "basis": {"partition_path": str(files["partition"])},
            "native_prior": native_prior,
            "aggregation_error_rank": "exact",
        },
        "likelihood": {"kind": "additive_sigma", "no_model_error": True, "fixed_model_mismatch": 0.0},
    }


def _prepare(tmp_path: Path, config: dict, merged: MergedData, name: str) -> Path:
    """Run the configured staged prepare and return its manifest."""
    arguments = Co2PreparationConfig.from_params(load_co2_params(params=config), require_source=True)
    return handoffs.prepare(**arguments.preparation_arguments(), merged_data=merged, output_dir=tmp_path / name)


# ---------------------------------------------------------------------------
# Dense reference
# ---------------------------------------------------------------------------


def _separable(lengths: tuple[float, float], classes: np.ndarray) -> np.ndarray:
    lat, lon = np.meshgrid(LAT, LON, indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    kernel = np.exp(-np.abs(lat[:, None] - lat[None]) / lengths[0] - np.abs(lon[:, None] - lon[None]) / lengths[1])
    return kernel * (classes.ravel()[:, None] == classes.ravel()[None])


def _geodesic(length_km: float) -> np.ndarray:
    lat, lon = np.meshgrid(np.deg2rad(LAT), np.deg2rad(LON), indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    haversine = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
                 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2)
    return np.exp(-2 * DEFAULT_EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))) / length_km)


def _dense_b() -> np.ndarray:
    """Dense native covariance over (source, cell) in acquisition source order."""
    classes = _classes().values
    land = classes.ravel() == 1
    bio = _separable(BIO_LENGTHS, classes)
    ff = _separable(FF_LENGTHS, classes)
    ff[np.ix_(land, land)] = _geodesic(FF_GEODESIC_KM)[np.ix_(land, land)]
    kernels = {"ff": ff, "gpp": bio, "ter": bio}
    n = classes.size
    b = np.zeros((3 * n, 3 * n))
    for i, s in enumerate(SOURCES):
        b[i * n:(i + 1) * n, i * n:(i + 1) * n] = SIGMA[s] ** 2 * kernels[s]
    cross = RHO * SIGMA["gpp"] * SIGMA["ter"] * bio
    b[n:2 * n, 2 * n:] = cross
    b[2 * n:, n:2 * n] = cross.T
    return b


def _dense_reference(merged: MergedData, prepared, restriction: str) -> dict[str, np.ndarray]:
    """C_alpha, H_alpha, A and the intercept from the defining dense formulas."""
    inputs = prepared.inv_inputs
    states = inputs.indexes["region"]
    times = inputs.indexes["nmeasure"].get_level_values("time")
    site = merged.site_data["TOY"]
    h = (site.fp_x_flux_sectoral.sel(time=times, source=list(SOURCES))
         .transpose("time", "source", "lat", "lon").values.reshape(len(times), -1))
    b = _dense_b()
    partition = _partition()
    n = len(LAT) * len(LON)
    pi = np.zeros((len(states), 3 * n))
    flux = {s: merged.flux_data[s].flux.isel(time=0).values.ravel() for s in SOURCES}
    area = areagrid(np.asarray(LAT), np.asarray(LON)).ravel()
    for row, (source, region) in enumerate(states):
        i = SOURCES.index(source)
        layer = partition.sel(source=source).values.ravel()
        cells = layer == np.unique(layer)[region]
        weights = cells * (np.abs(flux[source]) * area if restriction != "bucket" else 1.0)
        pi[row, i * n:(i + 1) * n] = weights
    if restriction == "bucket":
        u = pi.T
        binv_u = np.linalg.solve(b, u)
        pi = np.linalg.solve(u.T @ binv_u, binv_u.T)
    else:
        pi /= pi.sum(axis=1, keepdims=True)
    m = np.concatenate([np.full(n, MEAN[s]) for s in SOURCES])
    c = pi @ b @ pi.T
    lift = b @ pi.T @ np.linalg.inv(c)
    h_alpha = h @ lift
    return {
        "C": c, "H": h_alpha, "A": h @ b @ h.T - h_alpha @ c @ h_alpha.T,
        "intercept": h @ m - h_alpha @ pi @ m, "mean": pi @ m, "lift": lift,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_multisource_configuration_round_trips(files: dict[str, Path]) -> None:
    calibrated = _native_prior(files, "absolute_flux_weighted_mean",
                               retained_regions_path=str(files["retained"]),
                               calibration={"targets_path": str(files["partition"]), "statistic": "median"})
    del calibrated["sources"]["gpp"]["sigma"], calibrated["sources"]["ter"]["sigma"]
    for table in (_native_prior(files), calibrated):
        config = Co2PreparationConfig.from_params(_request(files, table), require_source=True)
        prior = config.native.native_prior
        assert isinstance(prior, Co2MultiSourceNativePrior)
        assert Co2MultiSourceNativePrior.from_params(prior.to_params()) == prior
        again = Co2PreparationConfig.from_params(_request(files, prior.to_params()), require_source=True)
        assert again.native.native_prior == prior
    assert prior.sources["gpp"].sigma is None and prior.calibration.statistic == "median"
    assert prior.sources["ff"].geodesic_correlation_length_km == {"1": FF_GEODESIC_KM}
    single = Co2PreparationConfig.from_params(_request(files, {"sigma": 0.2}), require_source=True)
    assert single.native.native_prior == Co2NativePrior(sigma=0.2)


@pytest.mark.parametrize("change,message", [
    ({"sigma": 0.5}, "belongs to the single-field prior"),
    ({"gpp_source": None}, "requires prepared_inputs.native_prior.gpp_source"),
    ({"gpp_ter_correlation": 1.0}, r"\|rho\| < 1"),
    ({"restriction": "nearest"}, "restriction must be one of"),
    ({"retained_regions_path": "r.json"}, "requires restriction = 'absolute_flux_weighted_mean'"),
    ({"ff": None}, "exactly the acquisition flux_sources"),
    ({"ter": {"sigma": 0.4, "latitude_correlation_length": 9.0}}, "one shared kernel"),
    ({"ter": {"mean": 1.0, "latitude_correlation_length": BIO_LENGTHS[0],
              "longitude_correlation_length": BIO_LENGTHS[1]}}, "sources.ter.sigma is required"),
    ({"ff": {"sigma": 0.3, "colour": "red"}}, "Unknown configuration option"),
    ({"calibration": {"targets_path": "t.nc"}}, "no source to calibrate"),
    ({"class_labels_path": None}, "requires prepared_inputs.native_prior.class_labels_path"),
])
def test_multisource_configuration_rejects_inconsistent_options(
    files: dict[str, Path], change: dict, message: str,
) -> None:
    table = _native_prior(files)
    for key, value in change.items():
        if key in SOURCES and value is None:
            del table["sources"][key]
        elif key in SOURCES:
            table["sources"][key] = value
        elif value is None:
            del table[key]
        else:
            table[key] = value
    with pytest.raises((TypeError, ValueError), match=message):
        Co2PreparationConfig.from_params(_request(files, table), require_source=True)


def test_single_field_options_reject_multisource_keys(files: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="requires per-source priors"):
        Co2PreparationConfig.from_params(_request(files, {"sigma": 0.2, "gpp_ter_correlation": 0.5}))


@pytest.mark.parametrize("template", ["co2.toml", "co2_fixed_ou.toml"])
@pytest.mark.parametrize("example", ["E-r07", "geodesic"])
def test_template_multisource_examples_parse(template: str, example: str) -> None:
    import tomllib

    from openghg_inversions.rhime.co2 import co2_config_templates

    lines = co2_config_templates()[template].read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"# Per-source native priors ({example}"))
    block = []
    for line in lines[start + 1:]:
        if not line.startswith("#"):
            break
        block.append(line.removeprefix("#").removeprefix(" "))
    prior = Co2MultiSourceNativePrior.from_params(tomllib.loads("\n".join(block))["prepared_inputs"]["native_prior"])
    gpp, ter = prior.sources["co2_BASE_GPP"], prior.sources["co2_BASE_TER"]
    assert prior.gpp_ter_correlation == pytest.approx(0.7) and gpp.kernel_key == ter.kernel_key
    if example == "E-r07":
        assert prior.calibration is not None and gpp.sigma is None
        assert gpp.kernel_key == (3.0, 6.0, None) and prior.sources["co2_BASE_FF"].kernel_key == (1.0, 2.5, None)
    else:
        assert gpp.geodesic_correlation_length_km == {"1": 500.0}
        assert prior.sources["co2_BASE_FF"].geodesic_correlation_length_km == {"1": 200.0}


# ---------------------------------------------------------------------------
# Joint reduction against the dense reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("restriction", ["bucket", "absolute_flux_weighted_mean"])
def test_multisource_prepare_matches_dense_reference(
    tmp_path: Path, merged: MergedData, files: dict[str, Path], restriction: str,
) -> None:
    manifest = _prepare(tmp_path, _request(files, _native_prior(files, restriction)), merged, restriction)
    prepared = handoffs.load_preparation(manifest).prepared
    inputs = prepared.inv_inputs
    assert list(inputs.indexes["region"].names) == ["source", "region_in_source"]
    # Bucket states number each source's partition labels from zero.
    assert list(inputs.indexes["region"]) == [("ff", 0), ("ff", 1), ("ff", 2), ("gpp", 0), ("gpp", 1),
                                              ("ter", 0), ("ter", 1)]
    expected = _dense_reference(merged, prepared, restriction)
    covariance = inputs.alpha_prior_covariance.values
    np.testing.assert_allclose(covariance, expected["C"], rtol=1e-9, atol=1e-12)
    # The GPP/TER cross block is present and the FF blocks are independent.
    source = inputs.source.values
    assert np.abs(covariance[np.ix_(source == "gpp", source == "ter")]).max() > 1e-3
    assert np.abs(covariance[np.ix_(source == "ff", source != "ff")]).max() < 1e-14
    np.testing.assert_allclose(inputs.H.values, expected["H"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(inputs.aggregation_error_covariance.values, expected["A"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(inputs.fixed_prior_contribution.values, expected["intercept"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(inputs.alpha_prior_mean.values, expected["mean"], rtol=1e-12)

    record = prepared.provenance["native_preparation"]
    assert record["native_source_priors"]["gpp_ter_correlation"] == RHO
    assert record["native_source_priors"]["sources"]["ff"]["kernel"] == "class_mixed_geodesic_exponential"
    assert record["native_source_priors"]["sources"]["gpp"]["sigma"] == SIGMA["gpp"]
    assert record["restriction"] == restriction
    assert Co2MultiSourceNativePrior.from_params(record["native_prior"]).gpp_ter_correlation == RHO

    folder = manifest.parent
    bound = load_and_bind_affine_flux_map(folder / "affine-flux-map.nc", folder / "prepared-inputs.nc")
    assert bound.artifact.source_provenance["native_prior"]["restriction"] == restriction
    # The affine map evaluates E[x | alpha] = m + U_* (alpha - Pi m) for the retained state.
    alpha = (xr.ones_like(inputs.alpha_prior_mean) * 1.3).assign_attrs(units="1")
    native = bound.state_to_native(alpha).transpose("native_source", "lat", "lon").values.ravel()
    m = np.concatenate([np.full(len(LAT) * len(LON), MEAN[s]) for s in SOURCES])
    np.testing.assert_allclose(native, m + expected["lift"] @ (1.3 - expected["mean"]), rtol=1e-8, atol=1e-10)


def test_retained_regions_marginalize_the_others(
    tmp_path: Path, merged: MergedData, files: dict[str, Path],
) -> None:
    table = _native_prior(files, "absolute_flux_weighted_mean", retained_regions_path=str(files["retained"]))
    prepared = handoffs.load_preparation(_prepare(tmp_path, _request(files, table), merged, "retained")).prepared
    inputs = prepared.inv_inputs
    assert list(inputs.indexes["region"]) == [("ff", 0), ("ff", 2), ("gpp", 0), ("gpp", 1), ("ter", 1)]
    expected = _dense_reference(merged, prepared, "absolute_flux_weighted_mean")
    np.testing.assert_allclose(inputs.alpha_prior_covariance.values, expected["C"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(inputs.H.values, expected["H"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(inputs.aggregation_error_covariance.values, expected["A"], rtol=1e-7, atol=1e-10)
    assert prepared.provenance["native_preparation"]["native_source_priors"]["sources"]["ter"]["retained_states"] == 1


def test_calibrated_amplitudes_meet_the_geometric_mean_anchor(
    tmp_path: Path, merged: MergedData, files: dict[str, Path],
) -> None:
    classes = _classes()
    weights = xr.DataArray(
        np.stack([(classes.values == 1) * 1.0, (classes.values == 2) * 1.0]) * areagrid(
            np.asarray(LAT), np.asarray(LON))[None],
        dims=("functional", "lat", "lon"), coords={"functional": ["west", "east"], "lat": LAT, "lon": LON},
    )
    # Relative targets; GPP and TER fluxes have one sign, so the scale is |total|.
    target = xr.DataArray([[0.5, 0.3], [0.4, 0.6]], dims=("source", "functional"),
                          coords={"source": ["gpp", "ter"], "functional": ["west", "east"]})
    path = tmp_path / "targets.nc"
    xr.Dataset({"functional_weights": weights, "target_relative_sd": target}).to_netcdf(path)
    table = _native_prior(files, calibration={"targets_path": str(path)})
    del table["sources"]["gpp"]["sigma"], table["sources"]["ter"]["sigma"]
    prepared = handoffs.load_preparation(_prepare(tmp_path, _request(files, table), merged, "calibrated")).prepared
    record = prepared.provenance["native_preparation"]
    kernel = _separable(BIO_LENGTHS, classes.values)
    for source in ("gpp", "ter"):
        q = weights.values.reshape(2, -1) * merged.flux_data[source].flux.isel(time=0).values.ravel()
        unit_sd = np.sqrt(np.einsum("fi,ij,fj->f", q, kernel, q))
        target_sd = target.sel(source=source).values * np.abs(q.sum(axis=1))
        sigma = float(np.exp(np.mean(np.log(target_sd / unit_sd))))
        assert record["native_source_priors"]["sources"][source]["sigma"] == pytest.approx(sigma, rel=1e-10)
        assert record["native_source_priors"]["sources"][source]["sigma_origin"] == "calibrated"
        assert record["calibration"]["sources"][source]["sigma"] == pytest.approx(sigma, rel=1e-10)
    assert record["native_source_priors"]["sources"]["ff"]["sigma_origin"] == "configured"


@pytest.mark.usefixtures("double_precision")
def test_multisource_preparation_samples_through_the_runner(
    merged: MergedData, files: dict[str, Path],
) -> None:
    config = _request(files, _native_prior(files))
    config["sampling"] = {"draws": 20, "tune": 20, "chains": 1, "progressbar": False, "random_seed": 3}
    config["outputs"] = {"output_format": "none"}
    result = run_co2(params=config, merged_data=merged)
    posterior = result.idata["posterior"]
    assert posterior.sizes["draw"] == 20
    state = next(name for name in posterior.data_vars if "region" in posterior[name].dims)
    assert posterior[state].sizes["region"] == 7


def test_single_field_route_is_bit_identical_to_the_direct_reduction(tmp_path: Path) -> None:
    """The default single shared field still reduces exactly as the public recipe does."""
    from examples.surface_centered_pilot import surface_inputs
    from openghg_inversions.basis import PreserveBucketProlongation, prepare_basis
    from openghg_inversions.coherent_reduction import reduce_native_gaussian
    from openghg_inversions.inversion_data import select_boundary_representation
    from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
    from openghg_inversions.native_covariance import SeparableExponentialCovariance
    from openghg_inversions.native_flux import resolve_runtime_flux
    from openghg_inversions.rhime.co2 import prepare_co2_inputs
    from openghg_inversions.rhime.co2.co2_native_preparation import _native_sensitivity
    from openghg_inversions.rhime.preparation import (
        assemble_inputs, build_sensitivities, filter_observations, resolve_basis_inputs,
    )

    obs, fp, fluxes, bc = surface_inputs()
    pilot = MergedData(
        site_data={"TOY": acquire_supplied(obs, fp, fluxes, bc)}, flux_data=fluxes, boundary_data=bc,
        site_options=SiteOptions.from_inputs(sites=["TOY"], averaging_period="3h"),
    )
    partition = tmp_path / "partition.nc"
    xr.ones_like(fluxes["fossil"].flux.isel(time=0, drop=True), dtype=int).rename("basis").to_dataset().to_netcdf(
        partition)
    config = _request({"partition": partition}, {"sigma": 0.2, "mean": 1.1, "correlation_length": 1.0})
    config["prepared_inputs"]["acquisition"].update(
        flux_sources=["biosphere", "fossil"], surface_transport={"biosphere": "integrated", "fossil": "integrated"})
    resolved = Co2PreparationConfig.from_params(load_co2_params(params=config), require_source=True)
    prepared = handoffs.load_preparation(
        handoffs.prepare(**resolved.preparation_arguments(), merged_data=pilot, output_dir=tmp_path / "p")
    ).prepared

    native = resolved.native
    site_data = select_boundary_representation(pilot.site_data, "ordinary")
    filtered, site_options = filter_observations(site_data, pilot.site_options, filters=None)
    filtered = prepare_observation_errors(filtered, averaging_error=True)
    basis = prepare_basis(
        site_data=filtered, runtime_flux=resolve_runtime_flux(pilot.flux_data, explicit_surface=True),
        weighting_flux=pilot.flux_data["biosphere"]["flux"], domain="TOY", start_date=WINDOW["start_date"],
        basis_algorithm=native.basis.basis_algorithm, nbasis=native.basis.nbasis,
        fix_outer_regions=False, **resolve_basis_inputs(native.basis),
    )
    canonical = assemble_inputs(
        build_sensitivities(filtered, basis, flux_sources=("biosphere", "fossil"), multisector=False),
        basis, site_options=site_options, domain="TOY", start_date=WINDOW["start_date"], use_bc=True,
    )
    sensitivity = _native_sensitivity(filtered, canonical)
    mean = xr.full_like(sensitivity.isel(observation=0, drop=True), 1.1).assign_attrs(units="1")
    reduction = reduce_native_gaussian(
        covariance=SeparableExponentialCovariance(sensitivity.lat, sensitivity.lon, sigma=0.2, correlation_length=1.0),
        strategy=PreserveBucketProlongation(basis.operator.basis_matrix), state_dim=basis.operator.meta.state_dim,
        native_mean=mean, native_sensitivity=sensitivity, observation_dim="observation",
    )
    direct = prepare_co2_inputs(canonical, reduction, aggregation_error_rank=None)
    for name in ("H", "alpha_prior_mean", "alpha_prior_covariance", "fixed_prior_contribution",
                 "aggregation_error_covariance", "mf", "mf_error", "H_bc"):
        np.testing.assert_array_equal(prepared.inv_inputs[name].values, direct.inv_inputs[name].values)
    assert prepared.provenance["native_preparation"]["native_prior"] == {
        "sigma": 0.2, "mean": 1.1, "correlation_length": 1.0,
        "latitude_correlation_length": None, "longitude_correlation_length": None,
    }
