"""Data-backed checkerboard benchmark inspired by Lunt et al. (2016).

This is a local implementation benchmark, not a reproduction of the paper.
It uses the repository's 2019 one-week NAME footprints for TAC and MHD rather
than two months of 2014 footprints for four sites, and its hybrid UKGHG/EDGAR7
inventory differs from the EDGAR inventory used in the paper. The available
boundary-condition test data are known to be corrupt and are deliberately
never opened. Instead, emissions outside the inversion window are treated as
a known fixed contribution and subtracted, following the inner/outer split
used by the InTEM fixed-outer-regions option without fitting outer scalings.
The packaged InTEM map resolves that contribution into labels 0 through 5 and
the part of inner label 6 outside the rectangular checkerboard crop.

The 48 by 56 inner window and sixteen-block 0.5/1.5 checkerboard preserve the
paper's principal spatial structure. Only 56 six-hour observations constrain
2,688 cells, however, so observation-space prediction is the primary recovery
diagnostic. Native-grid summaries remain useful illustrations but should not
be interpreted as a unique spatial reconstruction or a production comparison.
The slow comparison includes both a truth-informed oracle basis and a
conventional non-oracle quadtree basis derived only from mean sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
import xarray as xr

from openghg_inversions.basis import quadtree_basis_from_weights
from openghg_inversions.tdmcmc.core import (
    TransDimensionalProblem,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.tdmcmc.postprocessing import summarize_fine_grid_posterior
from openghg_inversions.tdmcmc.rhime_adapter import problem_from_rhime_inputs
from openghg_inversions.tdmcmc.sampling import SamplerConfig, sample

FloatArray = NDArray[np.float64]

_DATA_DIRECTORY = Path(__file__).parents[1] / "data"
_FLUX_PATH = _DATA_DIRECTORY / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
_INTEM_REGIONS_PATH = (
    Path(__file__).parents[2] / "openghg_inversions" / "basis" / "outer_region_definition_EUROPE.nc"
)
_FOOTPRINT_SPECS = (
    (
        _DATA_DIRECTORY / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc",
        "fp",
        "lat",
        "lon",
    ),
    (
        _DATA_DIRECTORY / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc",
        "srr",
        "latitude",
        "longitude",
    ),
)
_LATITUDE_SLICE = slice(157, 205)
_LONGITUDE_SLICE = slice(244, 300)
_GRID_SHAPE = (48, 56)
_BLOCK_SHAPE = (12, 14)
_OBSERVATION_SD = 5.0
_COORDINATE_ATOL = 2.0e-5
_BENCHMARK_ITERATIONS = 20_000
_BENCHMARK_START = 8_000
_BENCHMARK_THIN = 10


@dataclass(frozen=True, slots=True)
class _NameEdgarCheckerboardCase:
    """Complete two-site NAME/EDGAR checkerboard inversion case."""

    problem: TransDimensionalProblem
    truth: FloatArray
    noise: FloatArray
    inner_noiseless: FloatArray
    fixed_outer: FloatArray
    fixed_outer_regions: FloatArray
    full_noiseless: FloatArray
    full_observations: FloatArray
    site_observation_counts: tuple[int, ...]
    latitudes: FloatArray
    longitudes: FloatArray
    crop_intem_labels: NDArray[np.int64]


def _validate_positional_grid(
    footprint_latitudes: FloatArray,
    footprint_longitudes: FloatArray,
    flux_latitudes: FloatArray,
    flux_longitudes: FloatArray,
) -> None:
    """Require coordinate-compatible arrays before positional multiplication."""
    if footprint_latitudes.shape != flux_latitudes.shape:
        raise ValueError("Footprint and flux latitude shapes differ.")
    if footprint_longitudes.shape != flux_longitudes.shape:
        raise ValueError("Footprint and flux longitude shapes differ.")
    if not np.allclose(
        footprint_latitudes,
        flux_latitudes,
        rtol=0.0,
        atol=_COORDINATE_ATOL,
    ):
        raise ValueError("Footprint and flux latitudes are not positionally compatible.")
    if not np.allclose(
        footprint_longitudes,
        flux_longitudes,
        rtol=0.0,
        atol=_COORDINATE_ATOL,
    ):
        raise ValueError("Footprint and flux longitudes are not positionally compatible.")


def _six_hour_means(values: FloatArray) -> FloatArray:
    """Average consecutive hourly rows into non-overlapping six-hour blocks."""
    if values.shape[0] % 6:
        raise ValueError("Hourly inputs must contain complete six-hour blocks.")
    return values.reshape(values.shape[0] // 6, 6, *values.shape[1:]).mean(axis=1)


def _site_sensitivities_and_outer(
    path: Path,
    variable_name: str,
    latitude_name: str,
    longitude_name: str,
    flux: FloatArray,
    flux_latitudes: FloatArray,
    flux_longitudes: FloatArray,
    intem_labels: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build inner sensitivity, scalar outer signal, and seven InTEM columns."""
    with xr.open_dataset(path) as dataset:
        footprint_array = dataset[variable_name].transpose(
            "time",
            latitude_name,
            longitude_name,
        )
        footprint = np.asarray(footprint_array.values)
        footprint_latitudes = np.asarray(dataset[latitude_name].values, dtype=np.float64)
        footprint_longitudes = np.asarray(dataset[longitude_name].values, dtype=np.float64)

    _validate_positional_grid(
        footprint_latitudes,
        footprint_longitudes,
        flux_latitudes,
        flux_longitudes,
    )
    if footprint.shape[1:] != flux.shape:
        raise ValueError("Footprint and flux spatial shapes differ.")

    # Coordinates differ at the few-micrometre-in-degree level, so xarray's
    # exact label alignment would silently discard most cells. Validate above,
    # then deliberately multiply the arrays by position in NAME grid order.
    full_hourly = (
        np.einsum(
            "tij,ij->t",
            footprint,
            flux,
            dtype=np.float64,
            optimize=True,
        )
        * 1.0e9
    )
    inner_footprint = np.asarray(
        footprint[:, _LATITUDE_SLICE, _LONGITUDE_SLICE],
        dtype=np.float64,
    )
    inner_flux = flux[_LATITUDE_SLICE, _LONGITUDE_SLICE]
    inner_hourly = (inner_footprint * inner_flux[np.newaxis, :, :] * 1.0e9).reshape(
        footprint.shape[0],
        -1,
    )
    outer_hourly = full_hourly - inner_hourly.sum(axis=1)

    flattened_footprint = footprint.reshape(footprint.shape[0], -1)
    flattened_flux = flux.reshape(-1)
    flattened_labels = intem_labels.reshape(-1)
    crop = np.zeros(flux.shape, dtype=np.bool_)
    crop[_LATITUDE_SLICE, _LONGITUDE_SLICE] = True
    flattened_crop = crop.reshape(-1)
    outer_regions_hourly = np.empty((footprint.shape[0], 7), dtype=np.float64)
    for label in range(7):
        region_mask = flattened_labels == label
        if label == 6:
            region_mask = region_mask & ~flattened_crop
        outer_regions_hourly[:, label] = (
            np.einsum(
                "tc,c->t",
                flattened_footprint[:, region_mask],
                flattened_flux[region_mask],
                dtype=np.float64,
                optimize=True,
            )
            * 1.0e9
        )
    return (
        _six_hour_means(inner_hourly),
        _six_hour_means(outer_hourly),
        _six_hour_means(outer_regions_hourly),
    )


def _checkerboard_truth() -> FloatArray:
    """Return a 0.5/1.5 checkerboard on sixteen 12 by 14 blocks."""
    rows, columns = np.indices(_GRID_SHAPE)
    parity = (rows // _BLOCK_SHAPE[0] + columns // _BLOCK_SHAPE[1]) % 2
    return np.where(parity == 0, 0.5, 1.5).reshape(-1).astype(np.float64)


def _build_name_edgar_checkerboard_case() -> _NameEdgarCheckerboardCase:
    """Read raw inputs and adapt a canonical RHIME dataset into the problem."""
    with xr.open_dataset(_FLUX_PATH) as dataset:
        flux = np.asarray(
            dataset["flux"].transpose("time", "lat", "lon").isel(time=0).values,
            dtype=np.float64,
        )
        flux_latitudes = np.asarray(dataset["lat"].values, dtype=np.float64)
        flux_longitudes = np.asarray(dataset["lon"].values, dtype=np.float64)

    with xr.open_dataset(_INTEM_REGIONS_PATH) as dataset:
        intem_labels = np.asarray(dataset["region"].transpose("lat", "lon").values, dtype=np.int64)
        intem_latitudes = np.asarray(dataset["lat"].values, dtype=np.float64)
        intem_longitudes = np.asarray(dataset["lon"].values, dtype=np.float64)
    _validate_positional_grid(
        intem_latitudes,
        intem_longitudes,
        flux_latitudes,
        flux_longitudes,
    )
    if intem_labels.shape != flux.shape:
        raise ValueError("InTEM region and flux spatial shapes differ.")
    if not np.array_equal(np.unique(intem_labels), np.arange(7)):
        raise ValueError("Expected InTEM outer-region labels 0 through 6.")
    crop_intem_labels = intem_labels[_LATITUDE_SLICE, _LONGITUDE_SLICE]
    if not np.all(crop_intem_labels == 6):
        raise ValueError("The checkerboard crop must lie wholly inside InTEM region 6.")

    site_sensitivities: list[FloatArray] = []
    site_outer: list[FloatArray] = []
    site_outer_regions: list[FloatArray] = []
    for path, variable_name, latitude_name, longitude_name in _FOOTPRINT_SPECS:
        sensitivities, fixed_outer, fixed_outer_regions = _site_sensitivities_and_outer(
            path,
            variable_name,
            latitude_name,
            longitude_name,
            flux,
            flux_latitudes,
            flux_longitudes,
            intem_labels,
        )
        site_sensitivities.append(sensitivities)
        site_outer.append(fixed_outer)
        site_outer_regions.append(fixed_outer_regions)

    sensitivities = np.concatenate(site_sensitivities, axis=0)
    fixed_outer = np.concatenate(site_outer)
    fixed_outer_regions = np.concatenate(site_outer_regions, axis=0)
    truth = _checkerboard_truth()
    inner_noiseless = sensitivities @ truth
    full_noiseless = fixed_outer + inner_noiseless
    noise = np.random.default_rng(2017).normal(
        scale=_OBSERVATION_SD,
        size=inner_noiseless.size,
    )
    full_observations = full_noiseless + noise
    inner_observations = full_observations - fixed_outer

    inv_inputs = xr.Dataset(
        {
            "fp_x_flux": (
                ("nmeasure", "lat", "lon"),
                sensitivities.reshape(inner_observations.size, *_GRID_SHAPE),
            ),
            "mf": (("nmeasure",), inner_observations),
            "mf_error": (
                ("nmeasure",),
                np.full(inner_observations.size, _OBSERVATION_SD),
            ),
        },
        coords={
            "nmeasure": np.arange(inner_observations.size),
            "lat": np.arange(_GRID_SHAPE[0]),
            "lon": np.arange(_GRID_SHAPE[1]),
        },
    )
    problem = problem_from_rhime_inputs(
        inv_inputs,
        k_min=5,
        k_max=100,
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=1.0,
    )
    return _NameEdgarCheckerboardCase(
        problem=problem,
        truth=truth,
        noise=noise,
        inner_noiseless=inner_noiseless,
        fixed_outer=fixed_outer,
        fixed_outer_regions=fixed_outer_regions,
        full_noiseless=full_noiseless,
        full_observations=full_observations,
        site_observation_counts=tuple(values.shape[0] for values in site_sensitivities),
        latitudes=flux_latitudes[_LATITUDE_SLICE],
        longitudes=flux_longitudes[_LONGITUDE_SLICE],
        crop_intem_labels=crop_intem_labels,
    )


def _block_labels() -> NDArray[np.int64]:
    """Return the fixed 4 by 4 block label assigned to every inner cell."""
    rows, columns = np.indices(_GRID_SHAPE)
    return ((rows // _BLOCK_SHAPE[0]) * 4 + columns // _BLOCK_SHAPE[1]).reshape(-1).astype(np.int64)


def _fixed_basis_problem(
    case: _NameEdgarCheckerboardCase,
    labels: NDArray[np.int64],
) -> TransDimensionalProblem:
    """Aggregate the fine design over one complete fixed-basis label field."""
    if labels.shape != (case.problem.n_grid_cells,):
        raise ValueError("Fixed-basis labels must contain one value per grid cell.")
    unique_labels = np.unique(labels)
    design = np.column_stack(
        [case.problem.sensitivities[:, labels == label].sum(axis=1) for label in unique_labels]
    )
    region_count = unique_labels.size
    return TransDimensionalProblem(
        observations=case.problem.observations,
        observation_sd=case.problem.observation_sd,
        sensitivities=design,
        grid_coordinates=np.arange(region_count, dtype=np.float64)[:, np.newaxis],
        k_min=region_count,
        k_max=region_count,
        log_k_prior=uniform_log_k_prior(region_count, region_count),
        coefficient_prior_mean=case.problem.coefficient_prior_mean,
        coefficient_prior_sd=case.problem.coefficient_prior_sd,
    )


def _oracle_fixed_problem(case: _NameEdgarCheckerboardCase) -> TransDimensionalProblem:
    """Aggregate the fine design into the sixteen known checkerboard blocks."""
    return _fixed_basis_problem(case, _block_labels())


def _quadtree_fixed_problem(
    case: _NameEdgarCheckerboardCase,
) -> tuple[TransDimensionalProblem, NDArray[np.int64]]:
    """Build a conventional sixteen-region basis from mean sensitivity only."""
    weights = xr.DataArray(
        case.problem.sensitivities.mean(axis=0).reshape(_GRID_SHAPE),
        dims=("lat", "lon"),
        coords={"lat": case.latitudes, "lon": case.longitudes},
    )
    basis = quadtree_basis_from_weights(
        weights,
        "2019-01-01",
        "EUROPE",
        nbasis=16,
        seed=2016,
    )
    labels = np.asarray(basis.isel(time=0).values, dtype=np.int64).reshape(-1)
    return _fixed_basis_problem(case, labels), labels


def _sample_problem(
    problem: TransDimensionalProblem,
    initial_nuclei: NDArray[np.int64],
    comparison: FloatArray,
    *,
    seed: int,
) -> tuple[FloatArray, float, tuple[int, ...]]:
    """Run one seeded chain and return mean field, prediction RMSE, and k values."""
    initial_state = build_state(
        problem,
        initial_nuclei,
        np.ones(initial_nuclei.size),
        backend="numba",
    )
    trace = sample(
        problem,
        initial_state,
        SamplerConfig(
            iterations=_BENCHMARK_ITERATIONS,
            coefficient_proposal_sd=0.1,
            birth_proposal_sd=0.35,
            seed=seed,
            backend="numba",
            nucleus_move="local",
            local_move_scale=2.5,
        ),
    ).trace
    summary = summarize_fine_grid_posterior(
        problem,
        trace,
        comparison=comparison,
        start=_BENCHMARK_START,
        thin=_BENCHMARK_THIN,
        backend="numba",
    )
    return summary.mean, summary.rmse, tuple(int(value) for value in np.unique(trace.k))


def test_name_edgar_checkerboard_closes_the_raw_data_contract() -> None:
    """Raw NAME/EDGAR data should close the inner/outer forward calculation."""
    case = _build_name_edgar_checkerboard_case()

    assert case.site_observation_counts == (28, 28)
    assert case.problem.sensitivities.shape == (56, 48 * 56)
    assert case.problem.grid_coordinates.shape == (48 * 56, 2)
    assert (case.problem.k_min, case.problem.k_max) == (5, 100)
    rows, columns = np.indices(_GRID_SHAPE)
    np.testing.assert_array_equal(
        case.problem.grid_coordinates,
        np.column_stack((rows.reshape(-1), columns.reshape(-1))),
    )
    assert case.latitudes.shape == (48,)
    assert case.longitudes.shape == (56,)
    assert case.fixed_outer_regions.shape == (56, 7)
    np.testing.assert_array_equal(np.unique(case.crop_intem_labels), [6])
    np.testing.assert_array_equal(np.unique(case.truth), [0.5, 1.5])
    np.testing.assert_array_equal(
        case.truth.reshape(_GRID_SHAPE)[:12, :28],
        np.repeat([[0.5] * 14 + [1.5] * 14], 12, axis=0),
    )
    np.testing.assert_allclose(
        case.full_noiseless - case.fixed_outer,
        case.inner_noiseless,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        case.problem.sensitivities @ case.truth,
        case.inner_noiseless,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        case.fixed_outer_regions.sum(axis=1),
        case.fixed_outer,
        rtol=0.0,
        atol=3.0e-13,
    )
    assert np.all(np.any(case.fixed_outer_regions > 0.0, axis=0))
    np.testing.assert_allclose(
        case.full_observations - case.fixed_outer,
        case.problem.observations,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        case.problem.observations,
        case.inner_noiseless + case.noise,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.all(case.fixed_outer >= 0.0)
    assert np.median(case.inner_noiseless) > _OBSERVATION_SD


@pytest.mark.slow
def test_name_edgar_checkerboard_prediction_recovery() -> None:
    """Seeded adaptive and two fixed-basis fits should improve predictions.

    The oracle fixed comparator is intentionally advantaged by receiving the
    true rectangular blocks. The conventional fixed comparator instead uses a
    sixteen-region sensitivity-weighted quadtree without checkerboard truth or
    observations. Following the paper more closely, the adaptive chain starts
    with forty random nuclei under a uniform prior from 5 through 100 and can
    change both positions and count. All three receive 5,000 coefficient slots.
    This short, single-seed check is about predictive recovery under real
    transport, not convergence, model selection, runtime, or proof of
    native-grid recovery.
    """
    case = _build_name_edgar_checkerboard_case()
    fixed_problem = _oracle_fixed_problem(case)
    fixed_mean, fixed_rmse, fixed_visited_k = _sample_problem(
        fixed_problem,
        np.arange(16, dtype=np.int64),
        case.inner_noiseless,
        seed=1901,
    )
    quadtree_problem, quadtree_labels = _quadtree_fixed_problem(case)
    quadtree_mean, quadtree_rmse, quadtree_visited_k = _sample_problem(
        quadtree_problem,
        np.arange(16, dtype=np.int64),
        case.inner_noiseless,
        seed=1901,
    )
    initial_nuclei = np.sort(
        np.random.default_rng(1902).choice(
            case.problem.n_grid_cells,
            size=40,
            replace=False,
        )
    ).astype(np.int64)
    adaptive_mean, adaptive_rmse, adaptive_visited_k = _sample_problem(
        case.problem,
        initial_nuclei,
        case.inner_noiseless,
        seed=1901,
    )

    prior_prediction = case.problem.sensitivities @ np.ones(case.problem.n_grid_cells)
    prior_rmse = float(np.sqrt(np.mean(np.square(prior_prediction - case.inner_noiseless))))

    assert fixed_visited_k == (16,)
    assert np.unique(quadtree_labels).size == 16
    assert quadtree_visited_k == (16,)
    assert len(adaptive_visited_k) > 1
    assert fixed_rmse < 0.5 * prior_rmse
    assert quadtree_rmse < 0.35 * prior_rmse
    assert adaptive_rmse < 0.5 * prior_rmse
    assert adaptive_rmse < fixed_rmse + 1.0
    assert np.all(np.isfinite(fixed_mean))
    assert np.all(np.isfinite(quadtree_mean))
    assert np.all(np.isfinite(adaptive_mean))
