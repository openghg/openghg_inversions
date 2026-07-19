"""Run a paper-shaped RJMCMC checkerboard with local NAME/EDGAR test data.

This example is an implementation benchmark inspired by Lunt et al. (2016),
not a reproduction of the paper. It uses the repository's 2019 one-week NAME
footprints for TAC and MHD rather than two months of 2014 footprints for four
sites, and its hybrid UKGHG/EDGAR7 inventory differs from the paper inventory.

The available boundary-condition test data are known to be corrupt and are
deliberately never opened. Emissions outside the inversion window are instead
treated as a known fixed contribution and subtracted. The packaged InTEM map
resolves that contribution into labels 0 through 5 and the part of inner label
6 outside the rectangular checkerboard crop.

The 48 by 56 inner window and sixteen-block 0.5/1.5 checkerboard preserve the
paper's principal spatial structure. Only 56 six-hour observations constrain
2,688 cells, so observation-space prediction is the primary diagnostic. The
command compares RJMCMC with a truth-informed oracle basis and a conventional
non-oracle quadtree basis derived only from mean sensitivity. It prints a JSON
summary, including input hashes and run settings, and writes no result files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from openghg_inversions.basis import quadtree_basis_from_weights
from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.postprocessing import summarize_fine_grid_posterior
from openghg_inversions.experimental.rjmcmc.rhime_adapter import problem_from_rhime_inputs
from openghg_inversions.experimental.rjmcmc.sampling import SamplerConfig, sample

FloatArray = NDArray[np.float64]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "tests" / "data"
DEFAULT_INTEM_REGIONS_PATH = (
    REPOSITORY_ROOT / "openghg_inversions" / "basis" / "outer_region_definition_EUROPE.nc"
)

LATITUDE_SLICE = slice(157, 205)
LONGITUDE_SLICE = slice(244, 300)
GRID_SHAPE = (48, 56)
BLOCK_SHAPE = (12, 14)
OBSERVATION_SD = 5.0
COORDINATE_ATOL = 2.0e-5
DEFAULT_ITERATIONS = 20_000
DEFAULT_START = 8_000
DEFAULT_THIN = 10


@dataclass(frozen=True, slots=True)
class FootprintInput:
    """One NAME footprint input and its coordinate conventions."""

    path: Path
    variable_name: str
    latitude_name: str
    longitude_name: str


@dataclass(frozen=True, slots=True)
class CheckerboardInputPaths:
    """The complete, deliberately boundary-condition-free input set."""

    flux_path: Path
    intem_regions_path: Path
    footprints: tuple[FootprintInput, ...]

    @classmethod
    def repository_defaults(
        cls,
        *,
        data_directory: Path = DEFAULT_DATA_DIRECTORY,
        intem_regions_path: Path = DEFAULT_INTEM_REGIONS_PATH,
    ) -> CheckerboardInputPaths:
        """Return paths to the repository test fixtures used by the example."""
        return cls(
            flux_path=(data_directory / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"),
            intem_regions_path=intem_regions_path,
            footprints=(
                FootprintInput(
                    data_directory / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc",
                    "fp",
                    "lat",
                    "lon",
                ),
                FootprintInput(
                    data_directory / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc",
                    "srr",
                    "latitude",
                    "longitude",
                ),
            ),
        )

    @property
    def all_paths(self) -> tuple[Path, ...]:
        """Return every file that this example is allowed to open."""
        return (
            self.flux_path,
            self.intem_regions_path,
            *(footprint.path for footprint in self.footprints),
        )


@dataclass(frozen=True, slots=True, eq=False)
class NameEdgarCheckerboardCase:
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
    input_paths: CheckerboardInputPaths


@dataclass(frozen=True, slots=True)
class BenchmarkFit:
    """Compact result from one fixed- or variable-dimension chain."""

    name: str
    prediction_rmse: float
    visited_k: tuple[int, ...]
    runtime_seconds: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable fit summary."""
        return {
            "name": self.name,
            "prediction_rmse": self.prediction_rmse,
            "visited_k": list(self.visited_k),
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True, slots=True)
class CheckerboardBenchmarkSummary:
    """Provenance, configuration, and results printed by the example."""

    observation_count: int
    grid_shape: tuple[int, int]
    site_observation_counts: tuple[int, ...]
    observation_sd: float
    iterations: int
    start: int
    thin: int
    sampling_seed: int
    initial_seed: int
    prior_prediction_rmse: float
    oracle: BenchmarkFit
    quadtree: BenchmarkFit
    rjmcmc: BenchmarkFit
    input_provenance: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        """Return the complete machine-readable result contract."""
        return {
            "benchmark": "lunt_name_edgar_checkerboard",
            "scope": "implementation benchmark, not a paper reproduction",
            "boundary_conditions": "excluded: repository fixtures are known to be corrupt",
            "observations": {
                "count": self.observation_count,
                "site_counts": list(self.site_observation_counts),
                "standard_deviation": self.observation_sd,
            },
            "grid_shape": list(self.grid_shape),
            "sampler": {
                "iterations": self.iterations,
                "posterior_start": self.start,
                "thin": self.thin,
                "sampling_seed": self.sampling_seed,
                "initial_nucleus_seed": self.initial_seed,
                "backend": "numba",
            },
            "prior_prediction_rmse": self.prior_prediction_rmse,
            "fits": {
                "oracle_fixed_16": self.oracle.as_dict(),
                "sensitivity_quadtree_fixed_16": self.quadtree.as_dict(),
                "rjmcmc": self.rjmcmc.as_dict(),
            },
            "inputs": list(self.input_provenance),
        }


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
        atol=COORDINATE_ATOL,
    ):
        raise ValueError("Footprint and flux latitudes are not positionally compatible.")
    if not np.allclose(
        footprint_longitudes,
        flux_longitudes,
        rtol=0.0,
        atol=COORDINATE_ATOL,
    ):
        raise ValueError("Footprint and flux longitudes are not positionally compatible.")


def _six_hour_means(values: FloatArray) -> FloatArray:
    """Average consecutive hourly rows into non-overlapping six-hour blocks."""
    if values.shape[0] % 6:
        raise ValueError("Hourly inputs must contain complete six-hour blocks.")
    return values.reshape(values.shape[0] // 6, 6, *values.shape[1:]).mean(axis=1)


def _site_sensitivities_and_outer(
    footprint_input: FootprintInput,
    flux: FloatArray,
    flux_latitudes: FloatArray,
    flux_longitudes: FloatArray,
    intem_labels: NDArray[np.int64],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build inner sensitivity, scalar outer signal, and seven InTEM columns."""
    with xr.open_dataset(footprint_input.path) as dataset:
        footprint_array = dataset[footprint_input.variable_name].transpose(
            "time",
            footprint_input.latitude_name,
            footprint_input.longitude_name,
        )
        footprint = np.asarray(footprint_array.values)
        footprint_latitudes = np.asarray(
            dataset[footprint_input.latitude_name].values,
            dtype=np.float64,
        )
        footprint_longitudes = np.asarray(
            dataset[footprint_input.longitude_name].values,
            dtype=np.float64,
        )

    _validate_positional_grid(
        footprint_latitudes,
        footprint_longitudes,
        flux_latitudes,
        flux_longitudes,
    )
    if footprint.shape[1:] != flux.shape:
        raise ValueError("Footprint and flux spatial shapes differ.")

    # Coordinates differ at the few-micrometre-in-degree level. Validate them
    # above, then deliberately multiply by position in NAME grid order.
    full_hourly = np.einsum("tij,ij->t", footprint, flux, dtype=np.float64, optimize=True) * 1.0e9
    inner_footprint = np.asarray(
        footprint[:, LATITUDE_SLICE, LONGITUDE_SLICE],
        dtype=np.float64,
    )
    inner_flux = flux[LATITUDE_SLICE, LONGITUDE_SLICE]
    inner_hourly = (inner_footprint * inner_flux[np.newaxis, :, :] * 1.0e9).reshape(
        footprint.shape[0],
        -1,
    )
    outer_hourly = full_hourly - inner_hourly.sum(axis=1)

    flattened_footprint = footprint.reshape(footprint.shape[0], -1)
    flattened_flux = flux.reshape(-1)
    flattened_labels = intem_labels.reshape(-1)
    crop = np.zeros(flux.shape, dtype=np.bool_)
    crop[LATITUDE_SLICE, LONGITUDE_SLICE] = True
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


def checkerboard_truth() -> FloatArray:
    """Return a 0.5/1.5 checkerboard on sixteen 12 by 14 blocks."""
    rows, columns = np.indices(GRID_SHAPE)
    parity = (rows // BLOCK_SHAPE[0] + columns // BLOCK_SHAPE[1]) % 2
    return np.where(parity == 0, 0.5, 1.5).reshape(-1).astype(np.float64)


def build_name_edgar_checkerboard_case(
    input_paths: CheckerboardInputPaths | None = None,
) -> NameEdgarCheckerboardCase:
    """Read the declared raw inputs and construct the canonical RHIME problem."""
    if input_paths is None:
        input_paths = CheckerboardInputPaths.repository_defaults()

    with xr.open_dataset(input_paths.flux_path) as dataset:
        flux = np.asarray(
            dataset["flux"].transpose("time", "lat", "lon").isel(time=0).values,
            dtype=np.float64,
        )
        flux_latitudes = np.asarray(dataset["lat"].values, dtype=np.float64)
        flux_longitudes = np.asarray(dataset["lon"].values, dtype=np.float64)

    with xr.open_dataset(input_paths.intem_regions_path) as dataset:
        intem_labels = np.asarray(
            dataset["region"].transpose("lat", "lon").values,
            dtype=np.int64,
        )
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
    crop_intem_labels = intem_labels[LATITUDE_SLICE, LONGITUDE_SLICE]
    if not np.all(crop_intem_labels == 6):
        raise ValueError("The checkerboard crop must lie wholly inside InTEM region 6.")

    site_sensitivities: list[FloatArray] = []
    site_outer: list[FloatArray] = []
    site_outer_regions: list[FloatArray] = []
    for footprint_input in input_paths.footprints:
        sensitivities, fixed_outer, fixed_outer_regions = _site_sensitivities_and_outer(
            footprint_input,
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
    truth = checkerboard_truth()
    inner_noiseless = sensitivities @ truth
    full_noiseless = fixed_outer + inner_noiseless
    noise = np.random.default_rng(2017).normal(
        scale=OBSERVATION_SD,
        size=inner_noiseless.size,
    )
    full_observations = full_noiseless + noise
    inner_observations = full_observations - fixed_outer

    inv_inputs = xr.Dataset(
        {
            "fp_x_flux": (
                ("nmeasure", "lat", "lon"),
                sensitivities.reshape(inner_observations.size, *GRID_SHAPE),
            ),
            "mf": (("nmeasure",), inner_observations),
            "mf_error": (
                ("nmeasure",),
                np.full(inner_observations.size, OBSERVATION_SD),
            ),
        },
        coords={
            "nmeasure": np.arange(inner_observations.size),
            "lat": np.arange(GRID_SHAPE[0]),
            "lon": np.arange(GRID_SHAPE[1]),
        },
    )
    problem = problem_from_rhime_inputs(
        inv_inputs,
        k_min=5,
        k_max=100,
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=1.0,
    )
    return NameEdgarCheckerboardCase(
        problem=problem,
        truth=truth,
        noise=noise,
        inner_noiseless=inner_noiseless,
        fixed_outer=fixed_outer,
        fixed_outer_regions=fixed_outer_regions,
        full_noiseless=full_noiseless,
        full_observations=full_observations,
        site_observation_counts=tuple(values.shape[0] for values in site_sensitivities),
        latitudes=flux_latitudes[LATITUDE_SLICE],
        longitudes=flux_longitudes[LONGITUDE_SLICE],
        crop_intem_labels=crop_intem_labels,
        input_paths=input_paths,
    )


def block_labels() -> NDArray[np.int64]:
    """Return the fixed 4 by 4 block label assigned to every inner cell."""
    rows, columns = np.indices(GRID_SHAPE)
    return ((rows // BLOCK_SHAPE[0]) * 4 + columns // BLOCK_SHAPE[1]).reshape(-1).astype(np.int64)


def fixed_basis_problem(
    case: NameEdgarCheckerboardCase,
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


def oracle_fixed_problem(case: NameEdgarCheckerboardCase) -> TransDimensionalProblem:
    """Aggregate the fine design into the sixteen known checkerboard blocks."""
    return fixed_basis_problem(case, block_labels())


def quadtree_fixed_problem(
    case: NameEdgarCheckerboardCase,
) -> tuple[TransDimensionalProblem, NDArray[np.int64]]:
    """Build a conventional sixteen-region basis from mean sensitivity only."""
    weights = xr.DataArray(
        case.problem.sensitivities.mean(axis=0).reshape(GRID_SHAPE),
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
    return fixed_basis_problem(case, labels), labels


def _sample_problem(
    problem: TransDimensionalProblem,
    initial_nuclei: NDArray[np.int64],
    comparison: FloatArray,
    *,
    iterations: int,
    start: int,
    thin: int,
    seed: int,
    name: str,
) -> tuple[FloatArray, BenchmarkFit]:
    """Run one seeded chain and return its mean field and compact diagnostics."""
    initial_state = build_state(
        problem,
        initial_nuclei,
        np.ones(initial_nuclei.size),
        backend="numba",
    )
    started = perf_counter()
    trace = sample(
        problem,
        initial_state,
        SamplerConfig(
            iterations=iterations,
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
        start=start,
        thin=thin,
        backend="numba",
    )
    fit = BenchmarkFit(
        name=name,
        prediction_rmse=summary.rmse,
        visited_k=tuple(int(value) for value in np.unique(trace.k)),
        runtime_seconds=perf_counter() - started,
    )
    return summary.mean, fit


def _file_provenance(path: Path) -> dict[str, object]:
    """Return size and SHA-256 provenance for one input file."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def run_benchmark(
    case: NameEdgarCheckerboardCase,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    start: int = DEFAULT_START,
    thin: int = DEFAULT_THIN,
    sampling_seed: int = 1901,
    initial_seed: int = 1902,
) -> CheckerboardBenchmarkSummary:
    """Run matched oracle, quadtree, and RJMCMC prediction comparisons."""
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if start < 0 or start > iterations:
        raise ValueError("start must lie between zero and iterations.")
    if thin < 1:
        raise ValueError("thin must be positive.")

    oracle_problem = oracle_fixed_problem(case)
    _, oracle_fit = _sample_problem(
        oracle_problem,
        np.arange(16, dtype=np.int64),
        case.inner_noiseless,
        iterations=iterations,
        start=start,
        thin=thin,
        seed=sampling_seed,
        name="truth-informed fixed sixteen-block basis",
    )
    quadtree_problem, _ = quadtree_fixed_problem(case)
    _, quadtree_fit = _sample_problem(
        quadtree_problem,
        np.arange(16, dtype=np.int64),
        case.inner_noiseless,
        iterations=iterations,
        start=start,
        thin=thin,
        seed=sampling_seed,
        name="mean-sensitivity quadtree fixed sixteen-region basis",
    )
    initial_nuclei = np.sort(
        np.random.default_rng(initial_seed).choice(
            case.problem.n_grid_cells,
            size=40,
            replace=False,
        )
    ).astype(np.int64)
    _, rjmcmc_fit = _sample_problem(
        case.problem,
        initial_nuclei,
        case.inner_noiseless,
        iterations=iterations,
        start=start,
        thin=thin,
        seed=sampling_seed,
        name="variable-dimension Voronoi RJMCMC",
    )

    prior_prediction = case.problem.sensitivities @ np.ones(case.problem.n_grid_cells)
    prior_rmse = float(np.sqrt(np.mean(np.square(prior_prediction - case.inner_noiseless))))
    return CheckerboardBenchmarkSummary(
        observation_count=case.problem.n_observations,
        grid_shape=GRID_SHAPE,
        site_observation_counts=case.site_observation_counts,
        observation_sd=OBSERVATION_SD,
        iterations=iterations,
        start=start,
        thin=thin,
        sampling_seed=sampling_seed,
        initial_seed=initial_seed,
        prior_prediction_rmse=prior_rmse,
        oracle=oracle_fit,
        quadtree=quadtree_fit,
        rjmcmc=rjmcmc_fit,
        input_provenance=tuple(_file_provenance(path) for path in case.input_paths.all_paths),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the replayable benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--thin", type=int, default=DEFAULT_THIN)
    parser.add_argument("--sampling-seed", type=int, default=1901)
    parser.add_argument("--initial-seed", type=int, default=1902)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print its JSON summary without result-file output."""
    arguments = build_parser().parse_args(argv)
    case = build_name_edgar_checkerboard_case()
    summary = run_benchmark(
        case,
        iterations=arguments.iterations,
        start=arguments.start,
        thin=arguments.thin,
        sampling_seed=arguments.sampling_seed,
        initial_seed=arguments.initial_seed,
    )
    print(json.dumps(summary.as_dict(), indent=arguments.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
