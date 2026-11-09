"""Generated-basis weights from retained observation responses (#1045).

Shared-state weights are ``mean_o(abs(sum_s R_s))``; separate-state
(multisector) weights are ``sum_s mean_o(abs(R_s))``. Both pool retained
observations across sites by count and never read raw footprints.
"""

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr
from dask import delayed

import openghg_inversions.basis._functions as basis_module
from openghg_inversions.basis import basis_weights_from_data, fixed_outer_regions_basis_from_data, prepare_basis
from openghg_inversions.filters import local_influence
from openghg_inversions.inversion_data import MergedData, SiteOptions
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.multisector import prepare_multisector
from openghg_inversions.rhime.standard import prepare_standard

GRID = {"lat": [50.0, 51.0], "lon": [-2.0, -1.0]}
SOURCES = ["gpp", "ter"]


def _site(sectoral: np.ndarray, *, start: str = "2020-01-01") -> xr.Dataset:
    """Return a response-only site from ``(lat, lon, time, source)`` sector responses."""
    time = pd.date_range(start, periods=sectoral.shape[2], freq="h")
    coords = {**GRID, "time": time, "source": SOURCES}
    response = xr.DataArray(sectoral, dims=("lat", "lon", "time", "source"), coords=coords)
    return xr.Dataset({"fp_x_flux": response.sum("source"), "fp_x_flux_sectoral": response})


def _cancelling_site() -> xr.Dataset:
    """GPP and TER cancel exactly at cell (0, 0) and agree in sign at cell (1, 1)."""
    sectoral = np.zeros((2, 2, 2, 2))
    sectoral[0, 0, :, 0] = 3.0
    sectoral[0, 0, :, 1] = -3.0
    sectoral[1, 1, :, :] = 1.0
    return _site(sectoral)


def test_shared_state_sums_sectors_before_absolute_value():
    """Signed sector cancellation removes weight when sectors share one state."""
    weights = basis_weights_from_data({"SITE": _cancelling_site()})

    assert weights.dims == ("lat", "lon")
    np.testing.assert_array_equal(weights, [[0.0, 0.0], [0.0, 2.0]])


def test_separate_sector_states_sum_sector_magnitudes():
    """Cancellation between separately scaled sectors does not remove resolution."""
    weights = basis_weights_from_data({"SITE": _cancelling_site()}, multisector=True)

    np.testing.assert_array_equal(weights, [[6.0, 0.0], [0.0, 2.0]])


def test_sector_order_cannot_select_the_weighting_policy():
    """Each sector contributes its own response regardless of its position."""
    site = _cancelling_site()
    reordered = site.assign(fp_x_flux_sectoral=site.fp_x_flux_sectoral.isel(source=[1, 0]))

    for multisector in (False, True):
        xr.testing.assert_equal(
            basis_weights_from_data({"SITE": reordered}, multisector=multisector),
            basis_weights_from_data({"SITE": site}, multisector=multisector),
        )


@pytest.mark.parametrize("multisector", [False, True])
def test_temporal_sign_changes_cannot_cancel(multisector):
    """The absolute value is taken per observation, before averaging."""
    sectoral = np.zeros((2, 2, 2, 2))
    sectoral[0, 1, :, 0] = [2.0, -2.0]
    weights = basis_weights_from_data({"SITE": _site(sectoral)}, multisector=multisector)

    np.testing.assert_array_equal(weights, [[0.0, 2.0], [0.0, 0.0]])


def test_time_varying_response_deliberately_differs_from_mean_footprint_times_mean_flux():
    """Paired provider responses replace the product of independently averaged fields.

    The footprint sees cell (0, 0) only at the first observation, while the flux
    is non-zero there only at the second. The provider response is zero at both
    observations, but the retired rule averaged each field separately.
    """
    time = pd.date_range("2020-01-01", periods=2, freq="h")
    coords = {**GRID, "time": time}
    footprint = xr.DataArray(np.zeros((2, 2, 2)), dims=("lat", "lon", "time"), coords=coords)
    flux = xr.zeros_like(footprint)
    footprint[0, 0, 0] = 1.0
    flux[0, 0, 1] = 1.0
    footprint[1, 1, :] = 1.0
    flux[1, 1, :] = [1.0, 3.0]
    site = xr.Dataset({"fp_x_flux": footprint * flux})

    weights = basis_weights_from_data({"SITE": site})
    retired = footprint.mean("time") * flux.mean("time")

    np.testing.assert_array_equal(weights, [[0.0, 0.0], [0.0, 2.0]])
    np.testing.assert_array_equal(retired, [[0.25, 0.0], [0.0, 2.0]])


def test_sites_are_pooled_by_retained_observation_count():
    """Every retained observation contributes once; sites do not get equal weight."""
    one = np.zeros((2, 2, 1, 2))
    one[0, 0, 0, 0] = 3.0
    two = np.zeros((2, 2, 2, 2))
    sites = {"A": _site(one), "B": _site(two, start="2020-02-01")}

    weights = basis_weights_from_data(sites)

    assert float(weights.sel(lat=50.0, lon=-2.0)) == pytest.approx(1.0)  # site means would give 1.5


def test_only_retained_observations_contribute():
    """Callers pass filtered sites; removed observations cannot affect weights."""
    sectoral = np.ones((2, 2, 3, 2))
    sectoral[:, :, 2, :] = 1.0e6
    site = _site(sectoral)

    weights = basis_weights_from_data({"SITE": site.isel(time=[0, 1])})

    np.testing.assert_array_equal(weights, np.full((2, 2), 2.0))


def test_permuted_dimensions_and_mask_preserve_labels():
    """Labelled spatial coordinates survive permuted responses and masks."""
    site = _cancelling_site()
    permuted = site.transpose("source", "time", "lon", "lat")
    mask = xr.DataArray([[False, True], [False, True]], dims=("lon", "lat"),
                        coords={"lon": GRID["lon"], "lat": GRID["lat"]})

    weights = basis_weights_from_data({"A": site, "B": permuted}, multisector=True, mask=mask)

    assert weights.dims == ("lat", "lon")
    expected = basis_weights_from_data({"SITE": site}, multisector=True).where(mask.T, drop=True)
    xr.testing.assert_equal(weights, expected)
    np.testing.assert_array_equal(weights.lat, [51.0])


def test_sites_with_different_grids_are_rejected():
    """Positional mixing of site grids is never silently accepted."""
    site = _cancelling_site()
    shifted = site.assign_coords(lon=[-3.0, -2.0])

    with pytest.raises(ValueError, match="exact spatial coordinates"):
        basis_weights_from_data({"A": site, "B": shifted})


def test_missing_response_and_empty_selection_are_reported():
    """Generated weights need responses and at least one retained observation."""
    site = _cancelling_site()
    with pytest.raises(ValueError, match="fp_x_flux_sectoral"):
        basis_weights_from_data({"SITE": site.drop_vars("fp_x_flux_sectoral")}, multisector=True)
    with pytest.raises(ValueError, match="at least one retained observation"):
        basis_weights_from_data({"SITE": site.isel(time=[])})
    with pytest.raises(ValueError, match="at least one site"):
        basis_weights_from_data({})


def test_sparse_responses_are_densified_at_the_fitting_boundary():
    """Sparse payloads (eager and Dask-chunked) yield dense eager weights."""
    site = _cancelling_site()
    eager = site.copy(data={name: sparse.COO.from_numpy(site[name].to_numpy()) for name in site.data_vars})
    lazy = site.copy(data={name: da.from_array(sparse.COO.from_numpy(site[name].to_numpy()), chunks=1)
                           for name in site.data_vars})
    expected = basis_weights_from_data({"SITE": site}, multisector=True)

    for candidate in (eager, lazy):
        weights = basis_weights_from_data({"SITE": candidate}, multisector=True)
        assert isinstance(weights.data, np.ndarray)
        xr.testing.assert_equal(weights, expected)


def _time_resolved_site(executions: list[str]) -> tuple[xr.Dataset, xr.DataArray]:
    """Return a site whose response contracts a forbidden ``H_back`` footprint."""
    n_lat, n_lon, n_time, n_back = 4, 4, 6, 24
    time = pd.date_range("2020-01-01", periods=n_time, freq="h")
    coords = {"lat": np.arange(n_lat) + 50.0, "lon": np.arange(n_lon) - 2.0, "time": time}

    @delayed
    def forbidden_history():
        executions.append("fp_time_resolved")
        raise AssertionError("Basis weights executed the history-resolved footprint.")

    @delayed
    def response_payload():
        executions.append("fp_x_flux")
        rng = np.random.default_rng(1045)
        history = rng.random((n_lat, n_lon, n_time, n_back))
        flux = rng.normal(size=(n_lat, n_lon, n_time, n_back))  # signed, time-varying flux
        return (history * flux).sum(axis=-1)  # contract H_back as the transport provider does

    fp_time_resolved = xr.DataArray(
        da.from_delayed(forbidden_history(), shape=(n_lat, n_lon, n_time, n_back), dtype=float),
        dims=("lat", "lon", "time", "H_back"), coords=coords,
    )
    response = xr.DataArray(
        da.from_delayed(response_payload(), shape=(n_lat, n_lon, n_time), dtype=float),
        dims=("lat", "lon", "time"), coords=coords,
    )
    site = xr.Dataset({"fp_time_resolved": fp_time_resolved, "fp_x_flux": response})
    flux = xr.DataArray(np.ones((n_lat, n_lon)), dims=("lat", "lon"),
                        coords={"lat": coords["lat"], "lon": coords["lon"]}, name="flux")
    return site, flux


def test_time_resolved_response_needs_no_history_payload_and_inputs_are_borrowed():
    """Weights read the contracted response once, never ``H_back``, and leave inputs unchanged.

    Representative sizes here: ``fp_time_resolved`` (4, 4, 6, 24) is 18,432 bytes
    of float64, while the contracted ``fp_x_flux`` (4, 4, 6) is 768 bytes.
    """
    executions: list[str] = []
    site, _ = _time_resolved_site(executions)
    assert "H_back" not in site.fp_x_flux.dims
    assert (site.fp_time_resolved.nbytes, site.fp_x_flux.nbytes) == (18_432, 768)
    payloads = {name: site[name].data for name in site.data_vars}
    graph_keys = {name: payload.name for name, payload in payloads.items()}

    weights = basis_weights_from_data({"SITE": site})

    assert executions == ["fp_x_flux"]
    assert isinstance(weights.data, np.ndarray)
    assert weights.dims == ("lat", "lon")
    assert all(site[name].data is payload for name, payload in payloads.items())
    assert all(isinstance(payload, da.Array) for payload in payloads.values())
    assert {name: site[name].data.name for name in site.data_vars} == graph_keys
    with dask.config.set(scheduler="sync"):
        expected = abs(site.fp_x_flux).mean("time").compute()
    xr.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize("multisector", [False, True])
def test_fixed_outer_inner_weights_follow_the_state_layout(monkeypatch, multisector):
    """Fixed-outer generation fits its inner region to the state-layout weights."""
    lat, lon = np.arange(4.0), np.arange(4.0)
    sectoral = np.zeros((4, 4, 2, 2))
    sectoral[1:3, 1:3, :, 0] = 5.0
    sectoral[1:3, 1:3, :, 1] = -5.0
    sectoral[1, 1, :, :] = 1.0
    time = pd.date_range("2020-01-01", periods=2, freq="h")
    response = xr.DataArray(sectoral, dims=("lat", "lon", "time", "source"),
                            coords={"lat": lat, "lon": lon, "time": time, "source": SOURCES})
    sites = {"SITE": xr.Dataset({"fp_x_flux": response.sum("source"), "fp_x_flux_sectoral": response})}
    flux = xr.DataArray(np.ones((2, 4, 4)), dims=("source", "lat", "lon"),
                        coords={"source": SOURCES, "lat": lat, "lon": lon})
    outer = xr.DataArray([[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 2, 1], [0, 0, 1, 1]],
                         dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="region")
    seen = {}

    def fake_quadtree(weights, start_date, domain, nbasis=100):
        seen["weights"] = weights
        inner = xr.ones_like(weights, dtype=int)
        return inner.expand_dims(time=[pd.Timestamp(start_date)], axis=-1)

    monkeypatch.setattr(basis_module, "quadtree_basis_from_weights", fake_quadtree)
    fixed_outer_regions_basis_from_data(sites, flux, "2020-01-01", "quadtree", "TEST", nbasis=1,
                                        multisector=multisector, outer_regions=outer)

    expected = [[2.0, 10.0], [10.0, 10.0]] if multisector else [[2.0, 0.0], [0.0, 0.0]]
    np.testing.assert_array_equal(seen["weights"], expected)
    np.testing.assert_array_equal(seen["weights"].lat, [1.0, 2.0])
    np.testing.assert_array_equal(seen["weights"].lon, [1.0, 2.0])


def _response_only_merged(sectoral: bool) -> MergedData:
    """Build response-only merged inputs with no ``fp`` or ``fp_time_resolved``."""
    time = pd.date_range("2020-01-01", periods=2, freq="h")
    grid = {"lat": np.arange(4.0) + 50.0, "lon": np.arange(4.0) - 2.0}
    rng = np.random.default_rng(1045)
    sector_response = xr.DataArray(rng.normal(size=(4, 4, 2, 2)), dims=("lat", "lon", "time", "source"),
                                   coords={**grid, "time": time, "source": SOURCES})
    site = xr.Dataset({
        "mf": ("time", [1.0, 2.0], {"units": "ppm"}),
        "mf_error": ("time", [0.1, 0.2]),
        "fp_x_flux": sector_response.sum("source"),
    }, coords={**grid, "time": time}).chunk()
    if sectoral:
        site["fp_x_flux_sectoral"] = sector_response.chunk()
    flux = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"), coords=grid, attrs={"units": "mol m-2 s-1"})
    sources = {name: flux.to_dataset(name="flux").chunk() for name in (SOURCES if sectoral else SOURCES[:1])}
    return MergedData(
        site_data={"TAC": site}, flux_data=sources,
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"), split_by_sectors=sectoral,
    )


@pytest.mark.parametrize("sectoral", [False, True])
def test_generated_workflow_accepts_response_only_merged_inputs(monkeypatch, sectoral):
    """Standard and multisector preparation generate a basis without raw footprints."""
    merged = _response_only_merged(sectoral)
    site = merged.site_data["TAC"]
    borrowed = site.copy(deep=True)
    assert not {"fp", "fp_time_resolved"} & set(site.data_vars)
    seen = []
    original = basis_module.basis_weights_from_data

    def record(site_data, **kwargs):
        seen.append(kwargs["multisector"])
        return original(site_data, **kwargs)

    monkeypatch.setattr(basis_module, "basis_weights_from_data", record)
    import openghg_inversions.basis._wrapper as wrapper

    monkeypatch.setattr(wrapper, "basis_weights_from_data", record)
    prepare = prepare_multisector if sectoral else prepare_standard
    prepared = prepare(
        merged, domain="TEST", start_date="2020-01-01", flux_sources=tuple(merged.flux_data),
        averaging_error=True, filters=None, basis_algorithm="quadtree", nbasis=2,
        fix_basis_outer_regions=False, use_bc=False, basis_inputs={},
    )

    assert seen == [sectoral]
    assert prepared.basis_artifact_source == "generated"
    xr.testing.assert_identical(site, borrowed)


def test_response_only_prepare_basis_matches_direct_weights():
    """prepare_basis forwards the state layout to the public weight builder."""
    merged = _response_only_merged(True)
    runtime_flux = resolve_runtime_flux(merged.flux_data, split_by_sectors=True)
    shared = prepare_basis(site_data=merged.site_data, runtime_flux=runtime_flux, domain="TEST",
                           start_date="2020-01-01", nbasis=2, basis_algorithm="quadtree")
    separate = prepare_basis(site_data=merged.site_data, runtime_flux=runtime_flux, domain="TEST",
                             start_date="2020-01-01", nbasis=2, basis_algorithm="quadtree", multisector=True)
    expected_shared = basis_module.quadtree_basis_from_weights(
        basis_weights_from_data(merged.site_data), "2020-01-01", "TEST", nbasis=2)
    expected_separate = basis_module.quadtree_basis_from_weights(
        basis_weights_from_data(merged.site_data, multisector=True), "2020-01-01", "TEST", nbasis=2)
    xr.testing.assert_equal(shared.flat_basis(), expected_shared.squeeze("time", drop=True).rename("basis"))
    xr.testing.assert_equal(separate.flat_basis(), expected_separate.squeeze("time", drop=True).rename("basis"))


def test_local_influence_reports_missing_raw_footprints():
    """Response-only sites cannot use local_influence without ``fp`` or a cached local ratio."""
    site = _response_only_merged(False).site_data["TAC"]
    with pytest.raises(ValueError, match="local_influence filter requires the raw footprint 'fp'"):
        local_influence(site)
