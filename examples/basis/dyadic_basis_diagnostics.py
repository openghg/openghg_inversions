"""Compare dyadic search and quadtree bases in a synthetic Gaussian inversion.

The experiment uses the repository's aligned one-day TAC/MHD emissions grid
and frozen boundary-condition sensitivity.  Basis construction sees training
rows only.  Synthetic observations contain known emissions, explicit boundary
coefficients, and seeded Gaussian noise, so the analytic posterior can test
the basis without asking emissions to explain a missing atmospheric baseline.

This remains an experimental diagnostic.  It does not run production RHIME,
modify ``fixedbasisMCMC``, or claim that the frozen real observations and
boundary contribution are mutually consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from openghg_inversions.basis.algorithms import quadtree_algorithm
from openghg_inversions.basis.experimental.dyadic.demo_data import (
    DemoDesignData,
    load_tac_mhd_demo_data,
)
from openghg_inversions.basis.experimental.dyadic.demo_runner import (
    VariableKSearchConfig,
    run_projected_variable_k_dfs_search,
)
from openghg_inversions.basis.experimental.dyadic.dynamic_programming import (
    optimal_additive_partition,
)
from openghg_inversions.basis.experimental.dyadic.initializers import greedy_partition
from openghg_inversions.basis.experimental.dyadic.multiscale import sum_coarsen_grid
from openghg_inversions.basis.experimental.dyadic.partition_diagnostics import (
    GaussianPartitionDiagnostics,
    build_partition_diagnostics,
    emissions_compression_quality,
    gaussian_posterior_mean,
)
from openghg_inversions.basis.experimental.dyadic.rhime_gaussian import RHIMEGaussianMultiscale
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree

_BASELINE_PRIOR_SD = 0.05


@dataclass(frozen=True, slots=True)
class CandidateBasis:
    """One labelled search-grid basis and its construction diagnostics."""

    name: str
    labels: np.ndarray
    wall_seconds: float
    timing_scope: str
    accepted_moves: int | None = None
    initial_temperature: float | None = None


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    """One deterministic synthetic emissions and boundary truth."""

    name: str
    scaling: np.ndarray
    boundary_coefficients: np.ndarray
    observations: np.ndarray
    noiseless_emissions: np.ndarray
    noiseless_boundary: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    """Build the diagnostic command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/plans/figures/dyadic_basis_diagnostics"),
    )
    parser.add_argument(
        "--search-block-width",
        type=int,
        default=8,
        help="Native-cell width of one square search-grid leaf; this is spatial coarsening.",
    )
    parser.add_argument("--target-regions", type=int, default=31)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--pilot-proposals", type=int, default=300)
    parser.add_argument("--relative-prior-sd", type=float, default=0.5)
    parser.add_argument("--synthetic-model-error", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the basis and synthetic-inversion comparisons."""
    args = build_parser().parse_args(argv)
    data = load_tac_mhd_demo_data(args.data_directory)
    boundary_design = _load_boundary_design(args.data_directory, observations=data.G.shape[0])
    holdout = _blocked_holdout(data.times)
    training = ~holdout
    covariance_variants = _covariance_variants(data, training)
    synthetic_r_diag = covariance_variants[f"observation_plus_{args.synthetic_model_error:g}ppb"]
    synthetic_cases = _synthetic_cases(
        data,
        boundary_design,
        synthetic_r_diag,
        relative_prior_sd=args.relative_prior_sd,
        seed=args.seed,
    )

    rows: list[dict[str, object]] = []
    manifests: dict[str, object] = {}
    for covariance_name, r_diag in covariance_variants.items():
        result = _run_covariance_comparison(
            data,
            boundary_design,
            training,
            holdout,
            synthetic_cases,
            r_diag,
            covariance_name=covariance_name,
            block_width=args.search_block_width,
            target_regions=args.target_regions,
            relative_prior_sd=args.relative_prior_sd,
            iterations=args.iterations,
            pilot_proposals=args.pilot_proposals,
            seed=args.seed,
        )
        rows.extend(result[0])
        manifests[covariance_name] = result[1]

    args.output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_directory / "synthetic_basis_comparison.csv"
    report_path = args.output_directory / "synthetic_basis_comparison.md"
    figure_path = args.output_directory / "synthetic_basis_comparison.png"
    manifest_path = args.output_directory / "synthetic_basis_comparison_manifest.json"
    _write_csv(csv_path, rows)
    _write_figure(figure_path, rows, primary_covariance=f"observation_plus_{args.synthetic_model_error:g}ppb")
    consistency = _real_data_consistency(data, boundary_design)
    manifest = {
        "method": "training-only basis construction plus analytic synthetic Gaussian inversion",
        "search_block_width_native_cells": args.search_block_width,
        "search_block_is_grid_coarsening": True,
        "target_regions": args.target_regions,
        "relative_prior_sd": args.relative_prior_sd,
        "seed": args.seed,
        "iterations": args.iterations,
        "pilot_proposals": args.pilot_proposals,
        "synthetic_noise_covariance": f"observation_plus_{args.synthetic_model_error:g}ppb",
        "baseline_prior": {
            "mean": "one for every frozen boundary coefficient",
            "standard_deviation": _BASELINE_PRIOR_SD,
        },
        "synthetic_truth_cases": [case.name for case in synthetic_cases],
        "fixtures": {
            "emissions_and_observations": "tests/data reconstructed by load_tac_mhd_demo_data",
            "boundary_design": "tests/data/frozen_mhd_tac_make_inv_inputs_hbmcmc.npz:mcmc__Hbc",
        },
        "covariance_variants": {
            "observation_only": "error**2",
            "observation_plus_Nppb": "error**2 + N**2 for N in {5, 10, 20}",
            "training_percentile_floor": (
                "max(error, training-site median minus training-site fifth percentile)**2"
            ),
        },
        "training_percentile_floors_ppb": _training_percentile_floors(data, training),
        "training_observations": int(np.count_nonzero(training)),
        "holdout_observations": int(np.count_nonzero(holdout)),
        "holdout_start": str(data.times[holdout].min()),
        "holdout_stop": str(data.times[holdout].max()),
        "real_data_consistency": consistency,
        "baseline_holdout": _baseline_holdout_diagnostics(boundary_design, training, holdout),
        "covariance_runs": manifests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, rows, manifest, csv_path, figure_path)
    print(f"Wrote {len(rows)} comparison rows to {args.output_directory}")
    return 0


def _run_covariance_comparison(
    data: DemoDesignData,
    boundary_design: np.ndarray,
    training: np.ndarray,
    holdout: np.ndarray,
    synthetic_cases: tuple[SyntheticCase, ...],
    r_diag: np.ndarray,
    *,
    covariance_name: str,
    block_width: int,
    target_regions: int,
    relative_prior_sd: float,
    iterations: int,
    pilot_proposals: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Construct and evaluate all bases under one diagonal covariance.

    Args:
        data: Aligned one-day TAC/MHD emissions design and observation metadata.
        boundary_design: Frozen observation-by-boundary sensitivity matrix.
        training: Boolean rows used to construct bases and condition posteriors.
        holdout: Complementary Boolean rows used only for evaluation.
        synthetic_cases: Synthetic truths evaluated with every basis.
        r_diag: Assumed diagonal base observation covariance.
        covariance_name: Stable label for the covariance specification.
        block_width: Native-cell width of each search-grid leaf.
        target_regions: Requested basis size.
        relative_prior_sd: Native relative-scaling prior standard deviation.
        iterations: Fixed-count SLS proposal count.
        pilot_proposals: Discarded temperature-pilot proposal count.
        seed: Reproducible quadtree and SLS seed.

    Returns:
        Comparison rows and machine-readable construction diagnostics.
    """
    training_model = RHIMEGaussianMultiscale.from_native_grid(
        data.G[training],
        data.prior_flux,
        r_diag[training],
        coarsen_factor=block_width,
        relative_prior_sd=relative_prior_sd,
    )
    evaluation_model = RHIMEGaussianMultiscale.from_native_grid(
        data.G,
        data.prior_flux,
        r_diag,
        coarsen_factor=block_width,
        relative_prior_sd=relative_prior_sd,
    )
    candidates = _candidate_bases(
        data,
        training,
        r_diag,
        training_model,
        block_width=block_width,
        target_regions=target_regions,
        relative_prior_sd=relative_prior_sd,
        iterations=iterations,
        pilot_proposals=pilot_proposals,
        seed=seed,
    )
    baseline_prior_mean = np.ones(boundary_design.shape[1])
    baseline_prior_variances = np.full(boundary_design.shape[1], _BASELINE_PRIOR_SD**2)
    rows: list[dict[str, object]] = []
    candidate_manifest: dict[str, object] = {}

    for candidate in candidates:
        training_diagnostics = build_partition_diagnostics(training_model, candidate.labels)
        diagnostics = build_partition_diagnostics(evaluation_model, candidate.labels)
        candidate_manifest[candidate.name] = {
            "requested_regions": target_regions,
            "actual_regions": int(np.unique(candidate.labels).size),
            "effective_regions": int(diagnostics.supported_region_ids.size),
            "wall_seconds": candidate.wall_seconds,
            "timing_scope": candidate.timing_scope,
            "accepted_moves": candidate.accepted_moves,
            "initial_temperature": candidate.initial_temperature,
            "training_dfs": training_diagnostics.dfs,
            "all_rows_dfs": diagnostics.dfs,
        }
        for synthetic_case in synthetic_cases:
            row = _evaluate_synthetic_case(
                synthetic_case,
                candidate,
                diagnostics,
                evaluation_model,
                data,
                boundary_design,
                baseline_prior_mean,
                baseline_prior_variances,
                training,
                holdout,
                covariance_name=covariance_name,
                training_dfs=training_diagnostics.dfs,
            )
            rows.append(row)

    for synthetic_case in synthetic_cases:
        rows.append(
            _native_reference_row(
                synthetic_case,
                evaluation_model,
                data,
                boundary_design,
                baseline_prior_mean,
                baseline_prior_variances,
                training,
                holdout,
                covariance_name=covariance_name,
            )
        )
    return rows, {
        "training_native_dfs": training_model.full_grid_dfs,
        "all_rows_native_dfs": evaluation_model.full_grid_dfs,
        "candidates": candidate_manifest,
    }


def _candidate_bases(
    data: DemoDesignData,
    training: np.ndarray,
    r_diag: np.ndarray,
    training_model: RHIMEGaussianMultiscale,
    *,
    block_width: int,
    target_regions: int,
    relative_prior_sd: float,
    iterations: int,
    pilot_proposals: int,
    seed: int,
) -> tuple[CandidateBasis, ...]:
    """Build deterministic and stochastic candidate bases from training rows."""
    tree = training_model.design.tree

    started = perf_counter()
    greedy = greedy_partition(tree, target_regions, training_model.split_gain).state
    greedy_candidate = CandidateBasis(
        "dyadic_greedy",
        _state_labels(tree, greedy),
        perf_counter() - started,
        "partition selection only; shared Gaussian model excluded",
    )

    started = perf_counter()
    exact = optimal_additive_partition(tree, training_model.tile_scores, target_regions)
    exact_candidate = CandidateBasis(
        "dyadic_exact_dp",
        _state_labels(tree, exact.state),
        perf_counter() - started,
        "partition selection only; shared Gaussian model excluded",
    )

    coarse_training = sum_coarsen_grid(data.G[training], block_width).values
    weights = np.sqrt(np.sum(np.square(coarse_training) / r_diag[training, np.newaxis, np.newaxis], axis=0))
    started = perf_counter()
    quadtree_labels = np.asarray(
        quadtree_algorithm(weights, nbasis=target_regions, seed=seed), dtype=np.int64
    )
    quadtree_candidate = CandidateBasis(
        "existing_quadtree",
        quadtree_labels,
        perf_counter() - started,
        "quadtree threshold optimization only; coarse proxy construction excluded",
    )

    stochastic_candidates: list[CandidateBasis] = []
    for acceptance in (0.5, 0.1):
        config = VariableKSearchConfig(
            initial_regions=target_regions,
            free_regions=target_regions,
            min_regions=target_regions,
            max_regions=target_regions,
            penalty_per_extra_region=0.0,
            paired_move_probability=1.0,
            iterations=iterations,
            pilot_proposals=pilot_proposals,
            tau=relative_prior_sd,
            seed=seed,
            record_every=max(1, iterations // 100),
            initial_loss_acceptance=acceptance,
            final_loss_acceptance=0.01,
            hold_fraction=0.05,
            polish_fraction=0.2,
        )
        started = perf_counter()
        run = run_projected_variable_k_dfs_search(
            data.G[training],
            data.prior_flux,
            r_diag[training],
            config,
            coarsen_factor=block_width,
        )
        stochastic_candidates.append(
            CandidateBasis(
                f"dyadic_sls_p{int(acceptance * 100):02d}",
                _state_labels(run.model.design.tree, run.result.best_state),
                perf_counter() - started,
                "Gaussian model rebuild, greedy initializer, pilot, and SLS",
                accepted_moves=run.result.accepted_moves,
                initial_temperature=run.schedule.initial_temperature,
            )
        )
    return tuple([greedy_candidate, exact_candidate, quadtree_candidate, *stochastic_candidates])


def _evaluate_synthetic_case(
    case: SyntheticCase,
    candidate: CandidateBasis,
    diagnostics: GaussianPartitionDiagnostics,
    model: RHIMEGaussianMultiscale,
    data: DemoDesignData,
    boundary_design: np.ndarray,
    baseline_prior_mean: np.ndarray,
    baseline_prior_variances: np.ndarray,
    training: np.ndarray,
    holdout: np.ndarray,
    *,
    covariance_name: str,
    training_dfs: float,
) -> dict[str, object]:
    """Evaluate one basis against known- and uncertain-baseline inversions."""
    coefficients, joint_prediction = gaussian_posterior_mean(
        diagnostics,
        case.observations,
        baseline_design=boundary_design,
        baseline_prior_mean=baseline_prior_mean,
        baseline_prior_variances=baseline_prior_variances,
        training_subset=training,
    )
    region_count = diagnostics.regional_design.shape[1]
    joint_emissions = diagnostics.regional_design @ coefficients[:region_count]
    joint_boundary = boundary_design @ coefficients[region_count:]

    known_coefficients, known_emissions = gaussian_posterior_mean(
        diagnostics,
        case.observations - case.noiseless_boundary,
        training_subset=training,
    )
    del known_coefficients
    known_total = known_emissions + case.noiseless_boundary
    return {
        "covariance": covariance_name,
        "truth_case": case.name,
        "basis": candidate.name,
        "actual_regions": int(np.unique(candidate.labels).size),
        "effective_regions": int(region_count),
        "wall_seconds": candidate.wall_seconds,
        "timing_scope": candidate.timing_scope,
        "accepted_moves": candidate.accepted_moves,
        "initial_temperature": candidate.initial_temperature,
        "training_dfs": training_dfs,
        "all_rows_dfs": diagnostics.dfs,
        "native_dfs": model.full_grid_dfs,
        "pooled_compression": emissions_compression_quality(model, diagnostics),
        "mhd_compression": emissions_compression_quality(
            model, diagnostics, observation_subset=data.sites == "MHD"
        ),
        "tac_compression": emissions_compression_quality(
            model, diagnostics, observation_subset=data.sites == "TAC"
        ),
        "holdout_compression": emissions_compression_quality(model, diagnostics, observation_subset=holdout),
        "known_baseline_holdout_emissions_rmse": _rmse(
            known_emissions[holdout], case.noiseless_emissions[holdout]
        ),
        "known_baseline_holdout_total_rmse": _rmse(
            known_total[holdout],
            (case.noiseless_emissions + case.noiseless_boundary)[holdout],
        ),
        "joint_holdout_emissions_rmse": _rmse(joint_emissions[holdout], case.noiseless_emissions[holdout]),
        "joint_holdout_boundary_rmse": _rmse(joint_boundary[holdout], case.noiseless_boundary[holdout]),
        "joint_holdout_total_rmse": _rmse(
            joint_prediction[holdout],
            (case.noiseless_emissions + case.noiseless_boundary)[holdout],
        ),
        "joint_training_total_rmse": _rmse(
            joint_prediction[training],
            (case.noiseless_emissions + case.noiseless_boundary)[training],
        ),
    }


def _native_reference_row(
    case: SyntheticCase,
    model: RHIMEGaussianMultiscale,
    data: DemoDesignData,
    boundary_design: np.ndarray,
    baseline_prior_mean: np.ndarray,
    baseline_prior_variances: np.ndarray,
    training: np.ndarray,
    holdout: np.ndarray,
    *,
    covariance_name: str,
) -> dict[str, object]:
    """Return the no-reduction observation-space Gaussian reference."""
    emissions_prior_mean = data.G.reshape(data.G.shape[0], -1).sum(axis=1)
    boundary_prior_mean = boundary_design @ baseline_prior_mean
    boundary_covariance = (boundary_design * baseline_prior_variances) @ boundary_design.T
    retained_covariance = model.full_signal_covariance + boundary_covariance
    training_covariance = retained_covariance[np.ix_(training, training)] + np.diag(model.r_diag[training])
    residual = case.observations[training] - emissions_prior_mean[training] - boundary_prior_mean[training]
    solved = np.linalg.solve(training_covariance, residual)
    emissions_prediction = emissions_prior_mean + model.full_signal_covariance[:, training] @ solved
    boundary_prediction = boundary_prior_mean + boundary_covariance[:, training] @ solved
    total_prediction = emissions_prediction + boundary_prediction
    known_training_covariance = model.full_signal_covariance[np.ix_(training, training)] + np.diag(
        model.r_diag[training]
    )
    known_residual = (
        case.observations[training] - case.noiseless_boundary[training] - emissions_prior_mean[training]
    )
    known_solved = np.linalg.solve(known_training_covariance, known_residual)
    known_emissions_prediction = (
        emissions_prior_mean + model.full_signal_covariance[:, training] @ known_solved
    )
    supported_cells = int(np.count_nonzero(model.native_support))
    return {
        "covariance": covariance_name,
        "truth_case": case.name,
        "basis": "native_no_reduction",
        "actual_regions": supported_cells,
        "effective_regions": supported_cells,
        "wall_seconds": 0.0,
        "timing_scope": "reference calculation included in shared Gaussian model",
        "accepted_moves": None,
        "initial_temperature": None,
        "training_dfs": None,
        "all_rows_dfs": model.full_grid_dfs,
        "native_dfs": model.full_grid_dfs,
        "pooled_compression": 1.0,
        "mhd_compression": 1.0,
        "tac_compression": 1.0,
        "holdout_compression": 1.0,
        "known_baseline_holdout_emissions_rmse": _rmse(
            known_emissions_prediction[holdout],
            case.noiseless_emissions[holdout],
        ),
        "known_baseline_holdout_total_rmse": _rmse(
            (known_emissions_prediction + case.noiseless_boundary)[holdout],
            (case.noiseless_emissions + case.noiseless_boundary)[holdout],
        ),
        "joint_holdout_emissions_rmse": _rmse(
            emissions_prediction[holdout],
            case.noiseless_emissions[holdout],
        ),
        "joint_holdout_boundary_rmse": _rmse(
            boundary_prediction[holdout],
            case.noiseless_boundary[holdout],
        ),
        "joint_holdout_total_rmse": _rmse(
            total_prediction[holdout],
            (case.noiseless_emissions + case.noiseless_boundary)[holdout],
        ),
        "joint_training_total_rmse": _rmse(
            total_prediction[training],
            (case.noiseless_emissions + case.noiseless_boundary)[training],
        ),
    }


def _synthetic_cases(
    data: DemoDesignData,
    boundary_design: np.ndarray,
    r_diag: np.ndarray,
    *,
    relative_prior_sd: float,
    seed: int,
) -> tuple[SyntheticCase, ...]:
    """Build null, model-draw, and smooth misspecification truth cases."""
    rng = np.random.default_rng(seed)
    support = np.abs(data.prior_flux) > 0.0
    emission_design = data.G.reshape(data.G.shape[0], -1)
    cases = []
    for name in ("null", "model_draw", "smooth"):
        scaling = np.ones(data.prior_flux.shape, dtype=float)
        boundary_coefficients = np.ones(boundary_design.shape[1], dtype=float)
        if name == "model_draw":
            scaling[support] += relative_prior_sd * rng.normal(size=int(np.count_nonzero(support)))
            boundary_coefficients += 0.05 * rng.normal(size=boundary_coefficients.size)
        elif name == "smooth":
            scaling = _smooth_scaling_truth(data.lat, data.lon)
            scaling[~support] = 1.0
            boundary_coefficients += 0.04 * np.sin(np.linspace(0.0, 3.0 * np.pi, boundary_coefficients.size))

        emissions = emission_design @ scaling.ravel()
        boundary = boundary_design @ boundary_coefficients
        noise = rng.normal(scale=np.sqrt(r_diag))
        cases.append(
            SyntheticCase(
                name=name,
                scaling=scaling,
                boundary_coefficients=boundary_coefficients,
                observations=emissions + boundary + noise,
                noiseless_emissions=emissions,
                noiseless_boundary=boundary,
            )
        )
    return tuple(cases)


def _smooth_scaling_truth(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return a bounded smooth scaling field not aligned to the basis grids."""
    y = (lat - lat.min()) / (lat.max() - lat.min())
    x = (lon - lon.min()) / (lon.max() - lon.min())
    yy, xx = np.meshgrid(y, x, indexing="ij")
    positive_blob = 0.45 * np.exp(-((xx - 0.68) ** 2 / 0.018 + (yy - 0.58) ** 2 / 0.030))
    negative_blob = 0.28 * np.exp(-((xx - 0.38) ** 2 / 0.028 + (yy - 0.36) ** 2 / 0.020))
    wave = 0.12 * np.sin(2.0 * np.pi * xx) * np.cos(np.pi * yy)
    return 1.0 + positive_blob - negative_blob + wave


def _covariance_variants(data: DemoDesignData, training: np.ndarray) -> dict[str, np.ndarray]:
    """Return diagonal-R variants without using held-out values for floors."""
    variants = {"observation_only": np.square(data.error)}
    for mismatch in (5.0, 10.0, 20.0):
        variants[f"observation_plus_{mismatch:g}ppb"] = np.square(data.error) + mismatch**2
    floors = _training_percentile_floors(data, training)
    training_floor = np.empty(data.error.shape, dtype=float)
    for site in np.unique(data.sites):
        training_floor[data.sites == site] = floors[str(site)]
    variants["training_percentile_floor"] = np.square(np.maximum(data.error, training_floor))
    return variants


def _training_percentile_floors(data: DemoDesignData, training: np.ndarray) -> dict[str, float]:
    """Compute the legacy per-site error floors from training observations."""
    floors: dict[str, float] = {}
    for site in np.unique(data.sites):
        site_values = data.y[training & (data.sites == site)]
        if site_values.size == 0:
            raise ValueError(f"Training rows contain no observations for site {site!r}.")
        floors[str(site)] = float(np.median(site_values) - np.percentile(site_values, 5.0))
    return floors


def _load_boundary_design(data_directory: Path, *, observations: int) -> np.ndarray:
    """Load the frozen boundary sensitivity in observation-by-state order."""
    path = data_directory / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"
    with np.load(path, allow_pickle=False) as frozen:
        design = np.asarray(frozen["mcmc__Hbc"], dtype=float).T
    if design.ndim != 2 or design.shape[0] != observations or not np.all(np.isfinite(design)):
        raise ValueError("Frozen Hbc must be finite with one row per demo observation.")
    return design


def _blocked_holdout(times: np.ndarray) -> np.ndarray:
    """Hold out the common 12:00 through 17:00 UTC block at both sites."""
    start = np.datetime64("2019-01-01T12:00:00")
    stop = np.datetime64("2019-01-01T18:00:00")
    mask = (times >= start) & (times < stop)
    if np.count_nonzero(mask) < 2 or np.count_nonzero(~mask) < 2:
        raise ValueError("Blocked holdout must leave at least two training and holdout rows.")
    return mask


def _state_labels(tree: DyadicTree, state: PartitionState) -> np.ndarray:
    """Convert one exact dyadic frontier to positive search-grid labels."""
    state.validate(tree)
    labels = np.zeros(tree.shape, dtype=np.int64)
    for label, node_id in enumerate(state.ordered_active(), start=1):
        tile = tree.tile(node_id)
        labels[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop] = label
    if np.any(labels == 0):
        raise RuntimeError("Valid dyadic state did not cover the search grid.")
    return labels


def _real_data_consistency(data: DemoDesignData, boundary_design: np.ndarray) -> dict[str, object]:
    """Summarize the frozen real-data mismatch without attempting inversion."""
    prior_emissions = data.G.reshape(data.G.shape[0], -1).sum(axis=1)
    prior_boundary = boundary_design.sum(axis=1)
    residual = data.y - prior_emissions - prior_boundary
    return {
        "prior_emissions_min_ppb": float(prior_emissions.min()),
        "prior_emissions_max_ppb": float(prior_emissions.max()),
        "prior_boundary_min_ppb": float(prior_boundary.min()),
        "prior_boundary_max_ppb": float(prior_boundary.max()),
        "residual_min_ppb": float(residual.min()),
        "residual_max_ppb": float(residual.max()),
        "residual_rmse_ppb": float(np.sqrt(np.mean(np.square(residual)))),
        "interpretation": "fails real-data inversion gate; synthetic data only",
    }


def _baseline_holdout_diagnostics(
    boundary_design: np.ndarray,
    training: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, int]:
    """Count boundary directions unavailable from the blocked training rows."""
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(boundary_design))))
    training_support = np.any(np.abs(boundary_design[training]) > tolerance, axis=0)
    holdout_support = np.any(np.abs(boundary_design[holdout]) > tolerance, axis=0)
    return {
        "boundary_coefficients": int(boundary_design.shape[1]),
        "used_by_holdout": int(np.count_nonzero(holdout_support)),
        "used_by_holdout_but_unseen_in_training": int(np.count_nonzero(holdout_support & ~training_support)),
    }


def _rmse(values: np.ndarray, truth: np.ndarray) -> float:
    """Return root mean squared error for equally shaped finite vectors."""
    return float(np.sqrt(np.mean(np.square(values - truth))))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write comparison rows with a stable union of fields."""
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, rows: list[dict[str, object]], *, primary_covariance: str) -> None:
    """Plot basis score/compression and smooth-truth synthetic performance."""
    selected = [
        row
        for row in rows
        if row["covariance"] == primary_covariance
        and row["truth_case"] == "smooth"
        and row["basis"] != "native_no_reduction"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    positions = np.arange(len(selected))
    display_labels = [_display_basis(str(row["basis"])) for row in selected]
    axes[0].bar(positions, [_as_float(row["training_dfs"]) for row in selected])
    axes[0].set_xticks(positions, display_labels, rotation=25, ha="right")
    for index, row in enumerate(selected):
        label = display_labels[index]
        axes[1].scatter(
            _as_float(row["holdout_compression"]),
            _as_float(row["known_baseline_holdout_emissions_rmse"]),
            label=label,
        )
    axes[0].set(ylabel="Training projected DFS", title="Emissions-only basis score at K=31")
    axes[1].set(
        xlabel="Held-out emissions compression quality",
        ylabel="Held-out emissions RMSE (ppb)",
        title="Synthetic inversion with known baseline",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    rows: list[dict[str, object]],
    manifest: dict[str, object],
    csv_path: Path,
    figure_path: Path,
) -> None:
    """Write a compact interpretation report for the generated diagnostics."""
    primary_covariance = str(manifest["synthetic_noise_covariance"])
    baseline_holdout = cast(dict[str, int], manifest["baseline_holdout"])
    percentile_floors = cast(dict[str, float], manifest["training_percentile_floors_ppb"])
    primary_rows = [
        row for row in rows if row["covariance"] == primary_covariance and row["truth_case"] == "smooth"
    ]
    ordered = sorted(primary_rows, key=lambda row: _as_float(row["joint_holdout_emissions_rmse"]))
    lines = [
        "# Synthetic TAC/MHD Basis Diagnostics",
        "",
        "## Scope",
        "",
        "This is a controlled analytic Gaussian inversion, not a fit to the stored real observations. "
        "The real-data consistency gate fails because the frozen prior emissions plus boundary contribution "
        "leave residuals far larger than the supplied errors. Synthetic observations use the same emissions "
        "and boundary sensitivity matrices, so the baseline is explicit and internally consistent.",
        "",
        f"The search block width is {manifest['search_block_width_native_cells']} native cells along each spatial axis. "
        "That is grid coarsening; it is unrelated to the up-to-eightfold storage bound for a fully precomputed "
        "space-time multiscale Jacobian.",
        "",
        "## Primary smooth-truth result",
        "",
        f"Assumed covariance: `{primary_covariance}`. Bases use training rows only.",
        "",
        "| Basis | K effective | Train DFS | Holdout compression | Known-base emissions RMSE | Joint emissions RMSE | Boundary RMSE | Total RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            f"| {row['basis']} | {_as_int(row['effective_regions'])} | "
            f"{_format_optional(row['training_dfs'])} | {_format_optional(row['holdout_compression'])} | "
            f"{_format_optional(row['known_baseline_holdout_emissions_rmse'])} | "
            f"{_format_optional(row['joint_holdout_emissions_rmse'])} | "
            f"{_format_optional(row['joint_holdout_boundary_rmse'])} | "
            f"{_format_optional(row['joint_holdout_total_rmse'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Exact DP is the emissions-only, known-baseline Gaussian oracle; it is not jointly optimal for uncertain boundary coefficients.",
            "- Quadtree uses its existing cellwise precision-weighted proxy. Post-construction DFS and RMSE are comparable, but construction objectives and partition dictionaries both differ.",
            "- Compression quality is emissions-only and does not use the synthetic baseline.",
            f"- The holdout uses {baseline_holdout['used_by_holdout']} boundary directions; "
            f"{baseline_holdout['used_by_holdout_but_unseen_in_training']} are absent from training. Those directions "
            "remain at their prior mean, so boundary and total RMSE diagnose baseline extrapolation rather than basis quality.",
            "- Held-out posterior means predict retained emissions and boundary components. They do not conditionally "
            "predict the unresolved fine-grid aggregation residual.",
            "- Covariance sensitivity rows are in the CSV; the percentile floor is recomputed from training rows only.",
            f"  It is {percentile_floors['MHD']:.2f} ppb for MHD and {percentile_floors['TAC']:.2f} ppb for TAC in this split, so it remains a diagnostic rather than a recommended error model.",
            "",
            f"Artifacts: `{csv_path.name}`, `{figure_path.name}`, and `synthetic_basis_comparison_manifest.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_optional(value: object) -> str:
    """Format an optional numeric table cell."""
    if value is None:
        return "-"
    return f"{_as_float(value):.5g}"


def _display_basis(name: str) -> str:
    """Return compact plot text for a stable basis identifier."""
    return {
        "dyadic_greedy": "Greedy",
        "dyadic_exact_dp": "Exact DP",
        "existing_quadtree": "Quadtree",
        "dyadic_sls_p50": "SLS p=.5",
        "dyadic_sls_p10": "SLS p=.1",
    }.get(name, name)


def _as_float(value: object) -> float:
    """Narrow a numeric result field for static type checking."""
    return float(cast(float | int, value))


def _as_int(value: object) -> int:
    """Narrow an integral result field for static type checking."""
    return int(cast(int, value))


if __name__ == "__main__":
    raise SystemExit(main())
