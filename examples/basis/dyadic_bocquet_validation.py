"""Run a semi-synthetic TAC/MHD validation of dyadic Gaussian partitions.

This experimental command builds the independent-relative-error Gaussian
model on the repository's real TAC/MHD sensitivity matrix, but it never uses
the stored mole-fraction values or a stored boundary contribution.  Instead it
draws reproducible observation noise and combines it with a declared synthetic
relative-scaling truth.  A final blocked interval is held out, and all dyadic
partitions are selected using training rows only.

The command compares root, exact-coordinate land/ocean, rectangular
inner/outer, and exact fixed-count dyadic partitions selected separately by
DFS, base-error Fisher information, and the data-dependent Equation 45 score.
It writes a metrics CSV, provenance manifest, Markdown report, and summary PNG.
The default search resolution is the native grid (``coarsen_factor=1``);
coarsening is available only through the explicit benchmark option.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
import xarray as xr

from openghg_inversions.basis.experimental.dyadic.demo_data import (
    DemoDesignData,
    load_tac_mhd_demo_data,
    load_tac_mhd_week_demo_data,
)
from openghg_inversions.basis.experimental.dyadic.dynamic_programming import (
    optimal_additive_partition,
)
from openghg_inversions.basis.experimental.dyadic.partition_diagnostics import (
    build_partition_diagnostics,
    gaussian_partition_objectives,
)
from openghg_inversions.basis.experimental.dyadic.rhime_gaussian import RHIMEGaussianMultiscale
from openghg_inversions.basis.experimental.dyadic.state import PartitionState


_DAY_FIXTURES = (
    "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz",
    "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc",
    "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc",
    "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc",
)
_WEEK_FIXTURES = (
    "obs_mhd_ch4_10m_2019-01-01_2019-01-07_data.nc",
    "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc",
    "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc",
    "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc",
    "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc",
)
_LAND_OCEAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "openghg_inversions/basis/algorithms/country-EUROPE-UKMO-landsea-2023.nc"
)
_METRIC_NAMES = (
    "dfs",
    "fisher",
    "aggregation_aware_fisher",
    "equation45",
    "bayesian_information_gain",
)
_METRIC_LABELS = ("DFS", "Fisher R", "Fisher agg.", "Eq. 45", "Bayes info")


@dataclass(frozen=True, slots=True)
class SyntheticExperiment:
    """Deterministic truth, covariance, and centered synthetic innovation.

    Attributes:
        truth: Native relative-scaling anomaly, with zero at unsupported grid
            locations.
        noiseless_innovation: Real-design response ``G @ truth`` in ppb.
        innovation: Noiseless response plus a seeded draw from ``N(0, R)``.
        r_diag: Diagonal of ``R`` in ppb squared.
    """

    truth: np.ndarray
    noiseless_innovation: np.ndarray
    innovation: np.ndarray
    r_diag: np.ndarray


@dataclass(frozen=True, slots=True)
class PartitionCandidate:
    """One labelled comparison partition and its construction metadata.

    Attributes:
        name: Stable machine-readable partition name.
        title: Human-readable figure and report title.
        labels: Positive labels on the model search grid.
        selection_objective: Training-only construction criterion.
        selection_score: Additive training score for DP-selected partitions.
        construction_seconds: Wall-clock partition construction duration.
    """

    name: str
    title: str
    labels: np.ndarray
    selection_objective: str
    selection_score: float | None
    construction_seconds: float


def build_parser() -> argparse.ArgumentParser:
    """Build the semi-synthetic validation command-line parser.

    Returns:
        Parser containing all data, model, holdout, partition, and output
        options for the experimental command.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/dyadic_bocquet_validation"),
    )
    parser.add_argument(
        "--period",
        choices=("day", "week"),
        default="day",
        help="Use the quick frozen day by default or the aligned full week.",
    )
    parser.add_argument(
        "--coarsen-factor",
        type=int,
        default=1,
        help=(
            "Explicit square-block search benchmark. The default 1 searches at native "
            "resolution; no coarsening is applied silently."
        ),
    )
    parser.add_argument("--target-regions", type=int, default=32)
    parser.add_argument("--relative-prior-sd", type=float, default=0.5)
    parser.add_argument("--model-error-ppb", type=float, default=5.0)
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=None,
        help="Width of the final blocked holdout; defaults to 6 hours for day and 24 for week.",
    )
    parser.add_argument("--truth-mode", choices=("smooth", "prior-draw"), default="smooth")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--inner-lat-min", type=float, default=45.0)
    parser.add_argument("--inner-lat-max", type=float, default=60.0)
    parser.add_argument("--inner-lon-min", type=float, default=-15.0)
    parser.add_argument("--inner-lon-max", type=float, default=15.0)
    parser.add_argument("--posterior-chunk-size", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the validation and write all requested artifacts.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process status code zero after successful artifact generation.

    Raises:
        ValueError: If a numeric option, holdout, rectangle, or requested
            region count is invalid for the selected data.
        FileNotFoundError: If a required fixture or land/ocean mask is absent.
    """
    started = perf_counter()
    args = build_parser().parse_args(argv)
    _validate_arguments(args)
    holdout_hours = args.holdout_hours or (6 if args.period == "day" else 24)

    load_started = perf_counter()
    data = _load_data(args.data_directory, args.period)
    load_seconds = perf_counter() - load_started
    training, holdout, holdout_metadata = _blocked_time_masks(data.times, holdout_hours)
    experiment = _generate_experiment(
        data,
        relative_prior_sd=args.relative_prior_sd,
        model_error_ppb=args.model_error_ppb,
        truth_mode=args.truth_mode,
        seed=args.seed,
    )
    land = _load_exact_land_mask(_LAND_OCEAN_PATH, data.lat, data.lon)

    timings: dict[str, float] = {"load_data": load_seconds}
    training_started = perf_counter()
    training_model = _model_for_rows(
        data,
        experiment.r_diag,
        training,
        coarsen_factor=args.coarsen_factor,
        relative_prior_sd=args.relative_prior_sd,
    )
    timings["build_training_model"] = perf_counter() - training_started
    candidates = _build_candidates(
        training_model,
        experiment.innovation[training],
        land,
        data.lat,
        data.lon,
        target_regions=args.target_regions,
        rectangle=(
            args.inner_lat_min,
            args.inner_lat_max,
            args.inner_lon_min,
            args.inner_lon_max,
        ),
    )
    training_rows, training_bounds, evaluation_seconds = _evaluate_candidates(
        training_model,
        experiment.innovation[training],
        candidates,
        subset="training",
    )
    timings["evaluate_training"] = evaluation_seconds
    del training_model

    holdout_started = perf_counter()
    holdout_model = _model_for_rows(
        data,
        experiment.r_diag,
        holdout,
        coarsen_factor=args.coarsen_factor,
        relative_prior_sd=args.relative_prior_sd,
    )
    timings["build_holdout_model"] = perf_counter() - holdout_started
    holdout_rows, holdout_bounds, evaluation_seconds = _evaluate_candidates(
        holdout_model,
        experiment.innovation[holdout],
        candidates,
        subset="holdout",
    )
    timings["evaluate_holdout"] = evaluation_seconds
    del holdout_model

    all_started = perf_counter()
    all_rows = np.ones(data.G.shape[0], dtype=bool)
    all_model = _model_for_rows(
        data,
        experiment.r_diag,
        all_rows,
        coarsen_factor=args.coarsen_factor,
        relative_prior_sd=args.relative_prior_sd,
    )
    timings["build_all_row_model"] = perf_counter() - all_started
    posterior_started = perf_counter()
    posterior = all_model.native_posterior_marginals(
        experiment.innovation,
        chunk_size=args.posterior_chunk_size,
    )
    all_bounds = _native_bounds(all_model, experiment.innovation)
    timings["native_posterior"] = perf_counter() - posterior_started
    del all_model

    metrics = training_rows + holdout_rows
    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "dyadic_bocquet_metrics.csv"
    manifest_path = output_directory / "dyadic_bocquet_manifest.json"
    report_path = output_directory / "dyadic_bocquet_report.md"
    figure_path = output_directory / "dyadic_bocquet_summary.png"
    _write_metrics(csv_path, metrics)
    figure_started = perf_counter()
    sensitivity_weight = _native_sensitivity_weight(
        data.G,
        experiment.r_diag,
        relative_prior_sd=args.relative_prior_sd,
    )
    _write_figure(
        figure_path,
        data,
        experiment,
        posterior.mean_increment,
        np.sqrt(posterior.marginal_variance),
        sensitivity_weight,
        candidates,
        metrics,
        period=args.period,
        coarsen_factor=args.coarsen_factor,
        target_regions=args.target_regions,
        seed=args.seed,
    )
    timings["write_figure"] = perf_counter() - figure_started
    timings["total_before_report"] = perf_counter() - started

    manifest = _build_manifest(
        args,
        data,
        experiment,
        candidates,
        training,
        holdout,
        holdout_metadata,
        training_bounds,
        holdout_bounds,
        all_bounds,
        timings,
        csv_path,
        report_path,
        figure_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, manifest, metrics, figure_path, csv_path)
    print(f"Wrote {len(metrics)} metric rows and four artifacts to {output_directory}")
    return 0


def _validate_arguments(args: argparse.Namespace) -> None:
    """Validate scalar CLI arguments before loading potentially large data."""
    if args.coarsen_factor < 1:
        raise ValueError("coarsen_factor must be positive.")
    if args.target_regions < 1:
        raise ValueError("target_regions must be positive.")
    if args.relative_prior_sd <= 0.0 or not np.isfinite(args.relative_prior_sd):
        raise ValueError("relative_prior_sd must be finite and positive.")
    if args.model_error_ppb < 0.0 or not np.isfinite(args.model_error_ppb):
        raise ValueError("model_error_ppb must be finite and non-negative.")
    if args.holdout_hours is not None and args.holdout_hours < 1:
        raise ValueError("holdout_hours must be positive when supplied.")
    if args.posterior_chunk_size < 1:
        raise ValueError("posterior_chunk_size must be positive.")
    if not args.inner_lat_min < args.inner_lat_max:
        raise ValueError("inner latitude bounds must be strictly increasing.")
    if not args.inner_lon_min < args.inner_lon_max:
        raise ValueError("inner longitude bounds must be strictly increasing.")


def _load_data(data_directory: Path, period: str) -> DemoDesignData:
    """Load the requested fixture-backed design while leaving ``data.y`` unused."""
    if period == "day":
        return load_tac_mhd_demo_data(data_directory)
    return load_tac_mhd_week_demo_data(data_directory)


def _blocked_time_masks(
    times: npt.NDArray[np.datetime64],
    holdout_hours: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select one final contiguous wall-clock block as the holdout.

    Args:
        times: Observation timestamps, potentially in site-major order.
        holdout_hours: Positive duration of the final blocked interval.

    Returns:
        Training mask, holdout mask, and serializable interval metadata.

    Raises:
        ValueError: If timestamps are invalid or either split would be empty.
    """
    timestamps = np.asarray(times, dtype="datetime64[ns]")
    if timestamps.ndim != 1 or timestamps.size == 0 or np.any(np.isnat(timestamps)):
        raise ValueError("times must be a non-empty one-dimensional timestamp vector.")
    stop = timestamps.max().astype("datetime64[h]") + np.timedelta64(1, "h")
    start = stop - np.timedelta64(holdout_hours, "h")
    holdout = (timestamps >= start) & (timestamps < stop)
    training = ~holdout
    if not np.any(training) or not np.any(holdout):
        raise ValueError("blocked holdout must leave at least one training and one holdout row.")
    return (
        training,
        holdout,
        {
            "kind": "final_contiguous_wall_clock_block",
            "start_inclusive": str(start),
            "stop_exclusive": str(stop),
            "duration_hours": holdout_hours,
        },
    )


def _generate_experiment(
    data: DemoDesignData,
    *,
    relative_prior_sd: float,
    model_error_ppb: float,
    truth_mode: str,
    seed: int,
) -> SyntheticExperiment:
    """Generate a reproducible relative truth and centered synthetic innovation.

    Args:
        data: Real TAC/MHD sensitivity design and observation-error metadata.
        relative_prior_sd: Declared independent native prior standard deviation.
        model_error_ppb: Explicit additional observation error in ppb.
        truth_mode: Either ``smooth`` or ``prior-draw``.
        seed: Root seed split into independent truth and observation-noise streams.

    Returns:
        Synthetic truth, noiseless response, noisy innovation, and diagonal R.

    Raises:
        ValueError: If the flux has no support or the generated truth degenerates.
    """
    support = np.abs(data.prior_flux) > 0.0
    if not np.any(support):
        raise ValueError("prior_flux must contain at least one supported grid location.")
    truth_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    truth_rng = np.random.default_rng(truth_seed)
    raw = truth_rng.normal(size=support.shape)
    if truth_mode == "smooth":
        sigma = (max(1.0, support.shape[0] / 24.0), max(1.0, support.shape[1] / 24.0))
        raw = gaussian_filter(raw, sigma=sigma, mode="reflect")
    supported_values = raw[support]
    supported_values = supported_values - np.mean(supported_values)
    rms = float(np.sqrt(np.mean(np.square(supported_values))))
    if not np.isfinite(rms) or rms == 0.0:
        raise ValueError("synthetic truth must have positive finite supported RMS.")
    truth = np.zeros(support.shape, dtype=float)
    truth[support] = relative_prior_sd * supported_values / rms

    r_diag = np.square(np.asarray(data.error, dtype=float)) + model_error_ppb**2
    if np.any(r_diag <= 0.0) or not np.all(np.isfinite(r_diag)):
        raise ValueError("error**2 + model_error_ppb**2 must be finite and positive.")
    noiseless = np.einsum("ijk,jk->i", data.G, truth, optimize=True)
    noise_rng = np.random.default_rng(noise_seed)
    innovation = noiseless + noise_rng.normal(scale=np.sqrt(r_diag), size=r_diag.size)
    return SyntheticExperiment(
        truth=truth,
        noiseless_innovation=noiseless,
        innovation=innovation,
        r_diag=r_diag,
    )


def _load_exact_land_mask(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Load and validate the vendored exact-coordinate binary land mask.

    Args:
        path: Vendored NetCDF path containing variable ``country``.
        lat: Expected native latitude coordinates.
        lon: Expected native longitude coordinates.

    Returns:
        Boolean native grid, where true denotes land.

    Raises:
        FileNotFoundError: If the vendored mask does not exist.
        ValueError: If variable shape, coordinates, or binary values differ.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required land/ocean mask not found: {path}")
    with xr.open_dataset(path) as dataset:
        if "country" not in dataset:
            raise ValueError("land/ocean mask must contain variable 'country'.")
        country = dataset["country"]
        if country.dims != ("lat", "lon"):
            raise ValueError("land/ocean country variable must have dimensions ('lat', 'lon').")
        if not np.array_equal(dataset["lat"].values, lat) or not np.array_equal(
            dataset["lon"].values,
            lon,
        ):
            raise ValueError("land/ocean coordinates must exactly match the TAC/MHD design grid.")
        values = np.asarray(country.values)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("land/ocean country values must be exactly 0 or 1.")
    return values.astype(bool)


def _model_for_rows(
    data: DemoDesignData,
    r_diag: np.ndarray,
    rows: np.ndarray,
    *,
    coarsen_factor: int,
    relative_prior_sd: float,
) -> RHIMEGaussianMultiscale:
    """Build one row-subset independent-relative-error Gaussian model."""
    return RHIMEGaussianMultiscale.from_native_grid(
        data.G[rows],
        data.prior_flux,
        r_diag[rows],
        coarsen_factor=coarsen_factor,
        relative_prior_sd=relative_prior_sd,
    )


def _build_candidates(
    model: RHIMEGaussianMultiscale,
    training_innovation: np.ndarray,
    land_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    target_regions: int,
    rectangle: tuple[float, float, float, float],
) -> list[PartitionCandidate]:
    """Construct baseline and three exact training-only DP partitions.

    Args:
        model: Training-row Gaussian model defining scores and search tree.
        training_innovation: Centered synthetic innovation on training rows.
        land_mask: Exact-coordinate native binary land mask.
        lat: Native latitude coordinates.
        lon: Native longitude coordinates.
        target_regions: Exact region count for each dynamic program.
        rectangle: Inner latitude minimum/maximum and longitude minimum/maximum.

    Returns:
        Six comparison candidates in stable display order.

    Raises:
        ValueError: If the target count exceeds search leaves or the configured
            rectangle contains no native grid location.
    """
    tree = model.design.tree
    if target_regions > tree.shape[0] * tree.shape[1]:
        raise ValueError("target_regions exceeds the number of search-grid leaves.")
    root_started = perf_counter()
    root_labels = PartitionState.root(tree).to_labels(tree)
    root_seconds = perf_counter() - root_started

    land_started = perf_counter()
    land_labels = _binary_search_labels(land_mask, model.coarsen_factor)
    land_seconds = perf_counter() - land_started

    lat_min, lat_max, lon_min, lon_max = rectangle
    inner = (
        (lat[:, np.newaxis] >= lat_min)
        & (lat[:, np.newaxis] <= lat_max)
        & (lon[np.newaxis, :] >= lon_min)
        & (lon[np.newaxis, :] <= lon_max)
    )
    if not np.any(inner):
        raise ValueError("configured inner rectangle contains no native grid location.")
    rectangle_started = perf_counter()
    rectangle_labels = _binary_search_labels(inner, model.coarsen_factor)
    rectangle_seconds = perf_counter() - rectangle_started

    candidates = [
        PartitionCandidate("root", "Root", root_labels, "none", None, root_seconds),
        PartitionCandidate(
            "land_ocean",
            "Land / ocean",
            land_labels,
            "fixed_exact_coordinate_mask",
            None,
            land_seconds,
        ),
        PartitionCandidate(
            "rectangular_inner_outer",
            "Rectangular inner / outer",
            rectangle_labels,
            "fixed_user_rectangle",
            None,
            rectangle_seconds,
        ),
    ]
    eq45_scores = model.data_dependent_tile_scores(training_innovation)
    score_specs = (
        ("dyadic_dfs", "Dyadic DP: DFS", "dfs", model.tile_scores),
        ("dyadic_fisher", "Dyadic DP: Fisher R", "base_error_fisher", model.fisher_tile_scores),
        ("dyadic_equation45", "Dyadic DP: Equation 45", "equation45", eq45_scores),
    )
    for name, title, objective, scores in score_specs:
        started = perf_counter()
        solution = optimal_additive_partition(tree, scores, target_regions)
        elapsed = perf_counter() - started
        candidates.append(
            PartitionCandidate(
                name=name,
                title=title,
                labels=solution.state.to_labels(tree),
                selection_objective=objective,
                selection_score=solution.score,
                construction_seconds=elapsed,
            )
        )
    return candidates


def _binary_search_labels(mask: np.ndarray, factor: int) -> np.ndarray:
    """Convert a native binary mask to positive search-grid labels.

    At native resolution this is an exact ``1 + mask`` conversion.  For an
    explicitly requested coarsening benchmark, each partial or complete block
    is classified by majority area, with ties assigned to true.

    Args:
        mask: Two-dimensional native Boolean mask.
        factor: Positive width of each square search block.

    Returns:
        Search-grid labels one and two.
    """
    values = np.asarray(mask, dtype=bool)
    rows = (values.shape[0] + factor - 1) // factor
    columns = (values.shape[1] + factor - 1) // factor
    reduced = np.empty((rows, columns), dtype=bool)
    for row in range(rows):
        for column in range(columns):
            block = values[
                row * factor : min((row + 1) * factor, values.shape[0]),
                column * factor : min((column + 1) * factor, values.shape[1]),
            ]
            reduced[row, column] = 2 * np.count_nonzero(block) >= block.size
    return reduced.astype(np.int64) + 1


def _evaluate_candidates(
    model: RHIMEGaussianMultiscale,
    innovations: np.ndarray,
    candidates: list[PartitionCandidate],
    *,
    subset: str,
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """Evaluate all named Gaussian objectives for one row subset.

    Args:
        model: Gaussian model built only from the evaluated rows.
        innovations: Centered innovation vector matching those rows.
        candidates: Partitions selected without access to holdout rows.
        subset: Stable ``training`` or ``holdout`` label.

    Returns:
        Metric rows, native bounds, and total evaluation wall time.
    """
    started = perf_counter()
    bounds = _native_bounds(model, innovations)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        diagnostics = build_partition_diagnostics(model, candidate.labels)
        objectives = gaussian_partition_objectives(model, diagnostics, innovations)
        bounded_values = {
            "dfs": objectives.dfs,
            "fisher": objectives.fisher,
            "equation45": objectives.equation45,
        }
        for metric_name, value in bounded_values.items():
            tolerance = 1e-10 * max(1.0, abs(bounds[metric_name]))
            if value > bounds[metric_name] + tolerance:
                raise ArithmeticError(
                    f"{candidate.name} {metric_name} exceeds its native-grid bound."
                )
        rows.append(
            {
                "partition": candidate.name,
                "title": candidate.title,
                "subset": subset,
                "selection_objective": candidate.selection_objective,
                "selection_score_training": candidate.selection_score,
                "label_regions": int(np.unique(candidate.labels).size),
                "supported_regions": int(diagnostics.supported_region_ids.size),
                "observations": int(innovations.size),
                "dfs": objectives.dfs,
                "fisher": objectives.fisher,
                "aggregation_aware_fisher": objectives.aggregation_aware_fisher,
                "equation45": objectives.equation45,
                "bayesian_information_gain": objectives.bayesian_information_gain,
                "native_dfs_bound": bounds["dfs"],
                "native_fisher_bound": bounds["fisher"],
                "native_equation45_bound": bounds["equation45"],
                "construction_seconds": candidate.construction_seconds,
            }
        )
    return rows, bounds, perf_counter() - started


def _native_bounds(model: RHIMEGaussianMultiscale, innovations: np.ndarray) -> dict[str, float]:
    """Return native upper bounds for the three additive selection objectives."""
    return {
        "dfs": model.full_grid_dfs,
        "fisher": model.full_grid_fisher,
        "equation45": model.full_grid_data_dependent_score(innovations),
    }


def _native_sensitivity_weight(
    design: np.ndarray,
    r_diag: np.ndarray,
    *,
    relative_prior_sd: float,
) -> np.ndarray:
    """Return native base-error Fisher contribution by grid location."""
    return relative_prior_sd**2 * np.sum(
        np.square(design) / r_diag[:, np.newaxis, np.newaxis],
        axis=0,
    )


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write stable-schema objective rows to CSV."""
    if not rows:
        raise ValueError("at least one metrics row is required.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_manifest(
    args: argparse.Namespace,
    data: DemoDesignData,
    experiment: SyntheticExperiment,
    candidates: list[PartitionCandidate],
    training: np.ndarray,
    holdout: np.ndarray,
    holdout_metadata: dict[str, object],
    training_bounds: dict[str, float],
    holdout_bounds: dict[str, float],
    all_bounds: dict[str, float],
    timings: dict[str, float],
    csv_path: Path,
    report_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    """Assemble the machine-readable experiment and provenance manifest."""
    fixtures = _DAY_FIXTURES if args.period == "day" else _WEEK_FIXTURES
    fixture_paths = [args.data_directory / filename for filename in fixtures]
    source_path = Path(__file__).resolve()
    resolution_mode = "native_grid" if args.coarsen_factor == 1 else "explicit_coarsening_benchmark"
    return {
        "experimental_only": True,
        "period": args.period,
        "seed": args.seed,
        "truth": {
            "mode": args.truth_mode,
            "relative_prior_sd": args.relative_prior_sd,
            "supported_rms": float(np.sqrt(np.mean(np.square(experiment.truth[data.prior_flux != 0])))),
            "prior_mean": 0.0,
        },
        "centered_innovation": {
            "formula": "innovation = G @ relative_truth + epsilon",
            "noise": "epsilon ~ N(0, R)",
            "r_diag_formula": "data.error**2 + explicit_model_error_ppb**2",
            "explicit_model_error_ppb": args.model_error_ppb,
            "stored_real_mole_fraction_values_used": False,
            "stored_boundary_contribution_used": False,
            "noiseless_rms_ppb": float(np.sqrt(np.mean(np.square(experiment.noiseless_innovation)))),
            "realized_innovation_rms_ppb": float(np.sqrt(np.mean(np.square(experiment.innovation)))),
        },
        "search_resolution": {
            "coarsen_factor": args.coarsen_factor,
            "mode": resolution_mode,
            "default_is_native": True,
            "implicit_coarsening": False,
            "search_grid_shape": list(candidates[0].labels.shape),
            "native_grid_shape": list(data.prior_flux.shape),
            "coarsened_fixed_mask_rule": (
                None if args.coarsen_factor == 1 else "block majority by area; ties assigned to inner/land"
            ),
        },
        "partition_selection": {
            "uses_training_rows_only": True,
            "target_regions_for_each_exact_dp": args.target_regions,
            "candidate_order": [candidate.name for candidate in candidates],
            "dynamic_program_objectives": ["dfs", "base_error_fisher", "equation45"],
        },
        "rectangle": {
            "lat_min": args.inner_lat_min,
            "lat_max": args.inner_lat_max,
            "lon_min": args.inner_lon_min,
            "lon_max": args.inner_lon_max,
        },
        "land_ocean_mask": {
            "path": str(_LAND_OCEAN_PATH.relative_to(source_path.parents[2])),
            "variable": "country",
            "meaning": {"0": "ocean", "1": "land"},
            "coordinates_required_to_match_exactly": True,
        },
        "holdout": {
            **holdout_metadata,
            "training_rows": int(np.count_nonzero(training)),
            "holdout_rows": int(np.count_nonzero(holdout)),
            "training_sites": _site_counts(data.sites[training]),
            "holdout_sites": _site_counts(data.sites[holdout]),
        },
        "objective_conventions": {
            "dfs": "Bocquet Equation 38 degrees of freedom for signal",
            "fisher": "base-error Fisher trace using diagonal R",
            "aggregation_aware_fisher": "Fisher trace using R plus aggregation covariance",
            "equation45": "squared prior-precision posterior-mean norm; no factor one half",
            "bayesian_information_gain": "projected-posterior KL to projected prior; includes factor one half",
        },
        "native_bounds_note": (
            "Bounds are reported for the three additive DP selection objectives only. "
            "Aggregation-aware Fisher and Bayesian information gain are evaluation metrics, "
            "not scalar-node DP objectives in this report."
        ),
        "native_bounds": {
            "training": training_bounds,
            "holdout": holdout_bounds,
            "all_rows": all_bounds,
        },
        "posterior_maps": {
            "conditioning_rows": "all training and holdout rows",
            "mean": "native relative-scaling posterior mean increment",
            "sd": "native relative-scaling posterior marginal standard deviation",
            "chunk_size": args.posterior_chunk_size,
        },
        "input_provenance": {
            "fixture_sha256": {path.name: _sha256(path) for path in fixture_paths},
            "land_ocean_sha256": _sha256(_LAND_OCEAN_PATH),
            "source_sha256": {str(source_path.relative_to(source_path.parents[2])): _sha256(source_path)},
            "fixture_error_description": data.benchmark_error_description,
        },
        "timings_seconds": timings,
        "artifacts": {
            "metrics_csv": csv_path.name,
            "manifest_json": "dyadic_bocquet_manifest.json",
            "report_markdown": report_path.name,
            "summary_png": figure_path.name,
        },
    }


def _site_counts(sites: np.ndarray) -> dict[str, int]:
    """Return stable site counts as plain JSON-compatible integers."""
    names, counts = np.unique(sites, return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts, strict=True)}


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    figure_path: Path,
    csv_path: Path,
) -> None:
    """Write a self-contained Markdown interpretation of the validation."""
    innovation = manifest["centered_innovation"]
    resolution = manifest["search_resolution"]
    holdout = manifest["holdout"]
    bounds = manifest["native_bounds"]
    timings = manifest["timings_seconds"]
    lines = [
        "# Semi-synthetic TAC/MHD dyadic Bocquet validation",
        "",
        "This is an experimental validation of the independent-relative-error Gaussian model. "
        "It does not alter or exercise a production inversion path.",
        "",
        "## Experiment contract",
        "",
        f"The centered innovation is `{innovation['formula']}` with `{innovation['noise']}` and "
        f"`R = {innovation['r_diag_formula']}`. The explicit model error is "
        f"{innovation['explicit_model_error_ppb']:.6g} ppb. Stored real mole-fraction values and "
        "stored boundary contributions are not used.",
        "",
        f"Search mode: **{resolution['mode']}**, `coarsen_factor={resolution['coarsen_factor']}`. "
        "The default is native resolution and no coarsening is applied silently.",
        "",
        f"The holdout is the blocked interval [{holdout['start_inclusive']}, "
        f"{holdout['stop_exclusive']}) with {holdout['training_rows']} training rows and "
        f"{holdout['holdout_rows']} holdout rows. Every selected dyadic partition uses training "
        "rows only.",
        "",
        f"![Validation summary]({figure_path.name})",
        "",
        "## Partition objectives",
        "",
        _metrics_markdown(rows),
        "",
        "DFS, base-error Fisher, aggregation-aware Fisher, Equation 45, and Bayesian information "
        "gain remain separate columns. Equation 45 omits a factor of one half; Bayesian information "
        "gain includes the conventional factor of one half.",
        "",
        "Dyadic partitions are selected from training rows only. Holdout rows then define a fresh "
        "Gaussian update used to score how much held-out DFS, Fisher information, posterior-mean "
        "update, and projected KL each fixed partition retains. Under the exact Bocquet reduction, "
        "held-out predictive density is partition-invariant because the unresolved covariance is "
        "retained; predictive density is therefore a closure check rather than a ranking metric.",
        "",
        "## Native-grid additive selection-objective bounds",
        "",
        _bounds_markdown(bounds),
        "",
        "These are bounds for the three additive dynamic-programming objectives. "
        "Aggregation-aware Fisher and Bayesian information gain are retained as separate evaluation "
        "metrics and are not assigned scalar-node bounds here. The PNG also shows the native "
        "base-error sensitivity weight, synthetic truth, and all-row native posterior mean increment "
        "and marginal SD maps.",
        "",
        "## Provenance and timings",
        "",
        f"Raw metrics: [{csv_path.name}]({csv_path.name}). The JSON manifest records the complete "
        "fixture/source hashes, rectangle, seed, covariance, objective conventions, and timings.",
        "",
        "| stage | seconds |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.3f} |" for name, value in timings.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_markdown(rows: list[dict[str, Any]]) -> str:
    """Render compact training and holdout objective rows as Markdown."""
    lines = [
        "| subset | partition | regions | DFS | Fisher | agg. Fisher | Eq. 45 | Bayes info |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset']} | {row['title']} | {row['supported_regions']} | "
            f"{float(row['dfs']):.5g} | {float(row['fisher']):.5g} | "
            f"{float(row['aggregation_aware_fisher']):.5g} | {float(row['equation45']):.5g} | "
            f"{float(row['bayesian_information_gain']):.5g} |"
        )
    return "\n".join(lines)


def _bounds_markdown(bounds: Any) -> str:
    """Render native DFS, Fisher, and Equation 45 bounds as Markdown."""
    bound_mapping = dict(bounds)
    lines = [
        "| rows | native DFS | native Fisher | native Eq. 45 |",
        "|---|---:|---:|---:|",
    ]
    for subset in ("training", "holdout", "all_rows"):
        values = bound_mapping[subset]
        lines.append(
            f"| {subset} | {values['dfs']:.6g} | {values['fisher']:.6g} | {values['equation45']:.6g} |"
        )
    return "\n".join(lines)


def _write_figure(
    path: Path,
    data: DemoDesignData,
    experiment: SyntheticExperiment,
    posterior_mean: np.ndarray,
    posterior_sd: np.ndarray,
    sensitivity_weight: np.ndarray,
    candidates: list[PartitionCandidate],
    metrics: list[dict[str, Any]],
    *,
    period: str,
    coarsen_factor: int,
    target_regions: int,
    seed: int,
) -> None:
    """Write the context, posterior, partition, and objective summary figure.

    Args:
        path: Destination PNG path.
        data: TAC/MHD grid coordinates and design metadata.
        experiment: Synthetic truth and innovation metadata.
        posterior_mean: Native posterior mean increment map.
        posterior_sd: Native posterior marginal SD map.
        sensitivity_weight: Native base-error Fisher weight map.
        candidates: Six labelled comparison partitions.
        metrics: Training and holdout objective rows.
        period: Data period label.
        coarsen_factor: Explicit search coarsening factor.
        target_regions: Exact K used for dynamic programming.
        seed: Reproducibility seed shown in the title.
    """
    figure, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    extent = (float(data.lon.min()), float(data.lon.max()), float(data.lat.min()), float(data.lat.max()))
    context = np.log10(np.maximum(sensitivity_weight, np.finfo(float).tiny))
    image = axes[0, 0].imshow(context, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[0, 0].set_title("log10 native sensitivity weight")
    figure.colorbar(image, ax=axes[0, 0], shrink=0.78)

    truth_limit = float(np.max(np.abs(experiment.truth)))
    image = axes[0, 1].imshow(
        experiment.truth,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-truth_limit,
        vmax=truth_limit,
    )
    axes[0, 1].set_title("Synthetic relative truth")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.78)

    mean_limit = max(float(np.max(np.abs(posterior_mean))), np.finfo(float).eps)
    image = axes[0, 2].imshow(
        posterior_mean,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-mean_limit,
        vmax=mean_limit,
    )
    axes[0, 2].set_title("Native posterior mean increment")
    figure.colorbar(image, ax=axes[0, 2], shrink=0.78)

    image = axes[0, 3].imshow(posterior_sd, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0, 3].set_title("Native posterior marginal SD")
    figure.colorbar(image, ax=axes[0, 3], shrink=0.78)

    candidate_axes = list(axes[1, :]) + list(axes[2, :2])
    for axis, candidate in zip(candidate_axes, candidates, strict=True):
        axis.imshow(candidate.labels, origin="lower", extent=extent, aspect="auto", cmap="turbo")
        axis.set_title(f"{candidate.title}\n{np.unique(candidate.labels).size} labels")

    _objective_heatmap(axes[2, 2], metrics, candidates, subset="training")
    _objective_heatmap(axes[2, 3], metrics, candidates, subset="holdout")
    for axis in axes.ravel():
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
    axes[2, 2].set_xlabel("objective (column-normalized)")
    axes[2, 2].set_ylabel("partition")
    axes[2, 3].set_xlabel("objective (column-normalized)")
    axes[2, 3].set_ylabel("partition")
    figure.suptitle(
        f"Semi-synthetic TAC/MHD Gaussian validation — {period}, factor={coarsen_factor}, "
        f"fixed K={target_regions}, seed={seed}",
        fontsize=15,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _objective_heatmap(
    axis: Any,
    metrics: list[dict[str, Any]],
    candidates: list[PartitionCandidate],
    *,
    subset: str,
) -> None:
    """Plot objective values normalized within each metric column."""
    by_partition = {str(row["partition"]): row for row in metrics if row["subset"] == subset}
    values = np.asarray(
        [
            [float(by_partition[candidate.name][metric]) for metric in _METRIC_NAMES]
            for candidate in candidates
        ]
    )
    maxima = np.max(values, axis=0)
    normalized = np.divide(values, maxima, out=np.zeros_like(values), where=maxima > 0.0)
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(_METRIC_LABELS)), _METRIC_LABELS, rotation=45, ha="right")
    axis.set_yticks(range(len(candidates)), [candidate.name for candidate in candidates])
    axis.set_title(f"{subset.title()} objectives\n(fraction of column maximum)")
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            color = "white" if normalized[row, column] > 0.55 else "black"
            axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center", color=color)
    axis.figure.colorbar(image, ax=axis, shrink=0.7)


if __name__ == "__main__":
    raise SystemExit(main())
