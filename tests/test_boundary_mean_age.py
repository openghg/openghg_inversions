"""Per-cell mean-age boundary timing (``bc_time_method="mean_age"``).

The oracle below is independent of the vendored helpers: it finds each
lagged time's interval with pandas label lookup on one release/cell at a time.
"""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import MergedData, SiteOptions, check_surface_request, prepare_surface_site
from openghg_inversions.inversion_data._boundary import (
    baseline_reference_sensitivities,
    mean_age_history_start,
)
from openghg_inversions.rhime.co2 import configuration
from tests.test_surface_acquisition import boundary_inputs, inputs as inputs


def lagged_inputs(release_start="2021-07-01 00:00", hours=6, *, max_age=60.0, seed=1, constant_in_time=False):
    """Hourly releases with random weights/fractional ages and 3-hourly mid-labelled BCs."""
    rng = np.random.default_rng(seed)
    release = pd.date_range(release_start, periods=hours, freq="h")
    labels = pd.date_range(release[0] - pd.Timedelta(hours=max_age + 6), release[-1], freq="3h") + pd.Timedelta(
        minutes=90
    )
    labels = labels[labels <= release[-1]]
    space = {"height": [500.0, 1500.0], "lon": [1.0, 2.0, 3.0], "lat": [4.0, 5.0]}
    bc = xr.Dataset(coords={"time": labels, **space}, attrs={"time_period": "3 hours"})
    fp = xr.Dataset(coords={"time": release, **space})
    weights = {}
    for direction in "nesw":
        axis = "lon" if direction in "ns" else "lat"
        n = len(space[axis])
        values = 410 + rng.normal(0, 3, (len(labels), 2, n))
        if constant_in_time:
            values = np.broadcast_to(values[:1], values.shape).copy()
        bc[f"vmr_{direction}"] = (("time", "height", axis), values, {"units": "ppm"})
        weights[direction] = rng.uniform(0.1, 1, (hours, 2, n))
        weights[direction][:, 0, 0] = 0.0  # inactive cells
        age = rng.uniform(0, max_age, (hours, 2, n))
        age[:, 0, 0] = np.nan  # no usable age where there is no weight
        # Dimension order differs from the weights on purpose.
        fp[f"mean_age_particles_{direction}"] = (
            ("time", axis, "height"), np.moveaxis(age, 1, 2).astype("float32"), {"units": "h"},
        )
    total = sum(w.reshape(hours, -1).sum(1) for w in weights.values())
    for direction, w in weights.items():
        axis = "lon" if direction in "ns" else "lat"
        fp[f"particle_locations_{direction}"] = (
            ("time", "height", axis), (w / total[:, None, None]).astype("float32"), {"units": "1"},
        )
    for ds in (bc, fp):
        ds.height.attrs["units"] = "m"
        ds.lon.attrs["units"] = "degrees_east"
        ds.lat.attrs["units"] = "degrees_north"
    return bc, fp


def calculate(bc, fp, method, **kwargs):
    return baseline_reference_sensitivities(
        bc, fp, species="co2", reference=400.0, reference_units="ppm", mass_tolerance=1e-5,
        bc_time_method=method, **kwargs,
    )


def total(result, prefix="bc_anomaly"):
    return sum(
        result[f"{prefix}_{d}"].sum([dim for dim in result[f"{prefix}_{d}"].dims if dim != "time"])
        for d in "nesw"
    )


def absolute(result):
    return total(result) + result.bc_reference * total(result, "bc_mass")


def oracle(bc, fp, lagged):
    """Direct ``sum_q w_iq C_q(t_i - a_iq) / sum_q w_iq`` with pandas label lookup."""
    labels = bc.indexes["time"]
    out = np.zeros(fp.sizes["time"])
    mass = np.zeros(fp.sizes["time"])
    for direction in "nesw":
        w = fp[f"particle_locations_{direction}"]
        age = fp[f"mean_age_particles_{direction}"].transpose(*w.dims)
        c = bc[f"vmr_{direction}"].transpose(*w.dims)
        for i, t in enumerate(fp.indexes["time"]):
            for h in range(w.sizes["height"]):
                for q in range(w.shape[2]):
                    weight = float(w.values[i, h, q])
                    if weight == 0:
                        continue
                    hours = float(age.values[i, h, q]) if lagged else 0.0
                    target = t - pd.Timedelta(hours * 3600e9, unit="ns").round("ns")
                    row = labels.get_indexer([target], method="ffill")[0]
                    assert row >= 0 and target < labels[row] + pd.Timedelta(hours=3)
                    out[i] += weight * float(c.values[row, h, q])
                    mass[i] += weight
    return out / mass


@pytest.mark.parametrize("release_start", [
    "2021-07-01 00:00",  # month seam: lags reach back into June
    "2021-01-01 00:00",  # year seam: lags reach back into December 2020
    "2021-07-15 10:00",
])
def test_lagged_baseline_matches_independent_oracle_across_seams(release_start):
    bc, fp = lagged_inputs(release_start)
    result = calculate(bc, fp, "mean_age")
    np.testing.assert_allclose(absolute(result), oracle(bc, fp, lagged=True), rtol=0, atol=1e-10)
    np.testing.assert_allclose(absolute(calculate(bc, fp, "release_time")), oracle(bc, fp, lagged=False),
                               rtol=0, atol=1e-10)
    assert result.attrs["bc_time_method"] == "mean_age"
    if release_start.endswith("01 00:00"):
        # Some lagged times really are in the previous month/year.
        ages = max(float(fp[f"mean_age_particles_{d}"].max()) for d in "nesw")
        assert (fp.indexes["time"][0] - pd.Timedelta(hours=ages)).month != fp.indexes["time"][0].month


def test_weighted_lagged_minus_release_difference():
    bc, fp = lagged_inputs(seed=7)
    lagged, release = calculate(bc, fp, "mean_age"), calculate(bc, fp, "release_time")
    expected = oracle(bc, fp, lagged=True) - oracle(bc, fp, lagged=False)
    np.testing.assert_allclose(absolute(lagged) - absolute(release), expected, rtol=0, atol=1e-10)
    assert np.abs(expected).max() > 0.1  # the lag matters on this fixture


def test_mass_is_unchanged_by_the_lag_and_sums_to_one():
    bc, fp = lagged_inputs()
    lagged, release = calculate(bc, fp, "mean_age"), calculate(bc, fp, "release_time")
    np.testing.assert_allclose(total(lagged, "bc_mass"), 1.0, rtol=0, atol=1e-12)
    for d in "nesw":
        xr.testing.assert_identical(lagged[f"bc_mass_{d}"], release[f"bc_mass_{d}"])


def test_constant_field_identity():
    bc, fp = lagged_inputs(constant_in_time=True)
    lagged, release = calculate(bc, fp, "mean_age"), calculate(bc, fp, "release_time")
    for d in "nesw":
        xr.testing.assert_allclose(lagged[f"bc_anomaly_{d}"], release[f"bc_anomaly_{d}"], rtol=0, atol=1e-13)


def test_zero_age_identity():
    bc, fp = lagged_inputs()
    for d in "nesw":
        fp[f"mean_age_particles_{d}"] = xr.zeros_like(fp[f"mean_age_particles_{d}"]).assign_attrs(units="h")
    lagged, release = calculate(bc, fp, "mean_age"), calculate(bc, fp, "release_time")
    for d in "nesw":
        xr.testing.assert_identical(lagged[f"bc_anomaly_{d}"], release[f"bc_anomaly_{d}"])


def test_fractional_ages_are_not_rounded_and_differ_per_cell():
    bc, fp = lagged_inputs(hours=1, max_age=10)
    release = fp.indexes["time"][0]
    label = bc.indexes["time"][bc.indexes["time"] <= release][-1]
    just_inside = (release - label).total_seconds() / 3600 - 1e-4
    just_outside = just_inside + 2e-4
    ages = fp.mean_age_particles_n.copy()
    ages.loc[{"height": 1500.0, "lon": 2.0}] = just_inside
    ages.loc[{"height": 1500.0, "lon": 3.0}] = just_outside
    fp["mean_age_particles_n"] = ages.astype("float64")
    selected = calculate(bc, fp, "mean_age").bc_anomaly_n / fp.particle_locations_n.astype("float64") + 400
    current, previous = bc.vmr_n.sel(time=label), bc.vmr_n.sel(time=label - pd.Timedelta(hours=3))
    np.testing.assert_allclose(selected.sel(height=1500.0, lon=2.0), current.sel(height=1500.0, lon=2.0), atol=1e-9)
    np.testing.assert_allclose(selected.sel(height=1500.0, lon=3.0), previous.sel(height=1500.0, lon=3.0), atol=1e-9)


def test_missing_history_fails_but_inactive_cells_need_none():
    bc, fp = lagged_inputs()
    with pytest.raises(ValueError, match="do not cover positive-weight lagged time"):
        calculate(bc.isel(time=slice(6, None)), fp, "mean_age")
    # A huge age on a zero-weight cell needs no BC support or valid value.
    ages = fp.mean_age_particles_n.copy()
    ages[:, 0, 0] = 1e6
    fp["mean_age_particles_n"] = ages
    calculate(bc, fp, "mean_age")


@pytest.mark.parametrize("change,match", [
    (lambda fp: fp.drop_vars("mean_age_particles_e"), "mean_age_particles_e"),
    (lambda fp: fp.assign(mean_age_particles_s=fp.mean_age_particles_s.assign_attrs(units="")),
     "duration units"),
    (lambda fp: fp.assign(mean_age_particles_w=-fp.mean_age_particles_w.fillna(1).assign_attrs(units="h")),
     "nonnegative"),
])
def test_invalid_ages_are_rejected(change, match):
    bc, fp = lagged_inputs()
    with pytest.raises(ValueError, match=match):
        calculate(bc, change(fp), "mean_age").compute()


def test_unknown_method_is_rejected():
    bc, fp = lagged_inputs()
    with pytest.raises(ValueError, match="bc_time_method"):
        calculate(bc, fp, "exit_time")


def test_lazy_inputs_match_eager_inputs():
    bc, fp = lagged_inputs()
    lazy = calculate(bc.chunk({"time": 5}), fp.chunk({"time": 2}), "mean_age")
    assert isinstance(lazy.bc_anomaly_n.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), calculate(bc, fp, "mean_age"), rtol=0, atol=1e-12)


def test_history_start_is_earliest_active_lagged_time():
    _, fp = lagged_inputs()
    release_ns = fp.indexes["time"].asi8.astype("float64")
    earliest = np.inf
    for d in "nesw":
        weight = fp[f"particle_locations_{d}"]
        age = fp[f"mean_age_particles_{d}"].transpose(*weight.dims).values.astype("float64")
        lagged = release_ns[:, None, None] - np.where(weight.values > 0, age, np.nan) * 3.6e12
        earliest = min(earliest, np.nanmin(lagged))
    assert abs(mean_age_history_start(fp).value - earliest) < 1e3  # within a microsecond


def test_surface_preparation_records_timing_and_requires_ages_by_schema(inputs):
    obs, fp, flux = inputs
    fp, bc = boundary_inputs(fp)
    options = dict(species="co2", transport={"source": "integrated"}, boundary_data=bc,
                   boundary_reference=400.0, boundary_reference_units="ppm", boundary_mass_tolerance=1e-6)
    with pytest.raises(ValueError, match="mean_age_particles"):
        prepare_surface_site(obs, fp, {"source": flux}, bc_time_method="mean_age", **options)
    for d in "nesw":
        weight = fp[f"particle_locations_{d}"]
        fp[f"mean_age_particles_{d}"] = xr.zeros_like(weight).assign_attrs(units="h")
    lagged = prepare_surface_site(obs, fp, {"source": flux}, bc_time_method="mean_age", **options)
    release = prepare_surface_site(obs, fp, {"source": flux}, **options)
    assert lagged.attrs["bc_time_method"] == "mean_age"
    assert lagged.attrs["bc_time_policy"] == "interval_start"
    assert release.attrs["bc_time_method"] == "release_time"
    xr.testing.assert_allclose(lagged.bc_mod, release.bc_mod)


def test_request_and_configuration_checks():
    sites = SiteOptions.from_inputs(sites=["MHD"], inlet=["10m"], averaging_period=["1h"])
    common = dict(species="co2", site_options=sites, flux_sources=["a"], transport={"a": "integrated"},
                  boundary_reference=400.0, boundary_reference_units="ppm", boundary_mass_tolerance=1e-4)
    check_surface_request(**common, bc_time_method="mean_age")
    with pytest.raises(ValueError, match="bc_time_method"):
        check_surface_request(**common, bc_time_method="exit")
    with pytest.raises(ValueError, match="use_bc"):
        check_surface_request(**{**common, "boundary_reference": None, "boundary_mass_tolerance": None},
                              use_bc=False, bc_time_method="mean_age")
    parsed = configuration._acquisition({
        "sites": ["MHD"], "inlet": ["10m"], "averaging_period": ["1h"], "flux_sources": ["a"],
        "surface_transport": {"a": "integrated"}, "boundary_reference": 400.0,
        "boundary_reference_units": "ppm", "boundary_mass_tolerance": 1e-4, "bc_time_method": "mean_age",
    })
    assert parsed["bc_time_method"] == "mean_age"
    with pytest.raises(ValueError, match="bc_time_method"):
        configuration._acquisition({
            "sites": ["MHD"], "inlet": ["10m"], "averaging_period": ["1h"], "flux_sources": ["a"],
            "surface_transport": {"a": "integrated"}, "boundary_reference": 400.0,
            "boundary_reference_units": "ppm", "boundary_mass_tolerance": 1e-4, "bc_time_method": "exit",
        })


def test_model_scenario_route_rejects_mean_age():
    sites = SiteOptions.from_inputs(sites=["MHD"], inlet=["10m"], averaging_period=["1h"])
    with pytest.raises(ValueError, match="requires surface_transport"):
        MergedData.from_options(
            species="co2", site_options=sites, domain="EUROPE", start_date="2021-07-01",
            end_date="2021-07-02", flux_sources=["a"], bc_time_method="mean_age",
        )
