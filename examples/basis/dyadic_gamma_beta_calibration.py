"""Calibrate the Gamma--Beta prior against UK country-total uncertainty.

This executable experiment compares the original prior settings with a
moment-controlled depth policy. It solves the inner-land Gamma root variance
analytically for 20% and 50% prior relative standard deviation of the UK total,
then repeats the controlled calibration with sensitivity and flat topology
weights. The flat case also receives the exact projected exponential-distance
diagnostic used by ``dyadic_gamma_beta_intem_demo.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from dyadic_gamma_beta_intem_demo import (
    IntemDistanceCovarianceComparison,
    IntemGammaBetaCase,
    IntemGammaBetaSummary,
    build_case,
    build_distance_covariance_comparison,
    load_country_mask,
    summarize_case,
)
from openghg_inversions.basis.experimental.dyadic.calibration import (
    AggregatePriorMoments,
    RootVarianceCalibration,
    aggregate_prior_moments,
    calibrate_group_root_variance,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta import MomentSplitConstraint

_DEFAULT_OUTPUT = Path("docs/plans/figures/dyadic_gamma_beta_calibration")
_UK_NAME = "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND"
_CONTROLLED_BASE_KAPPA = 40.0
_CONTROLLED_DEPTH_MULTIPLIER = 1.5
_CONTROLLED_MAX_KAPPA = 96.0
_CONTROLLED_MIN_BETA_SHAPE = 1.0
_CONTROLLED_MAX_CHILD_VARIANCE = 9.0
_CONTROLLED_INNER_OCEAN_ROOT_VARIANCE = 0.25


@dataclass(frozen=True, slots=True, eq=False)
class CalibratedCase:
    """One converged aggregate calibration and its exact diagnostics."""

    topology_weight_mode: Literal["sensitivity", "flat"]
    target_relative_standard_deviation: float
    case: IntemGammaBetaCase
    summary: IntemGammaBetaSummary
    aggregate: AggregatePriorMoments
    calibration: RootVarianceCalibration
    topology_iterations: int
    distance_comparison: IntemDistanceCovarianceComparison | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible calibration diagnostics."""
        distance = self.distance_comparison
        return {
            "topology_weight_mode": self.topology_weight_mode,
            "target_relative_standard_deviation": self.target_relative_standard_deviation,
            "achieved_relative_standard_deviation": self.aggregate.relative_standard_deviation,
            "calibrated_inner_land_root_variance": self.calibration.calibrated_root_variance,
            "fixed_inner_ocean_root_variance": _CONTROLLED_INNER_OCEAN_ROOT_VARIANCE,
            "contrast_only_minimum_relative_standard_deviation": (
                self.calibration.minimum_relative_standard_deviation
            ),
            "requested_inner_regions": sum(self.case.inner_region_targets),
            "requested_inner_ocean_regions": self.case.inner_region_targets[0],
            "requested_inner_land_regions": self.case.inner_region_targets[1],
            "achieved_inner_regions": self.summary.inner_ocean_regions
            + self.summary.inner_land_regions,
            "achieved_inner_ocean_regions": self.summary.inner_ocean_regions,
            "achieved_inner_land_regions": self.summary.inner_land_regions,
            "minimum_beta_shape": self.summary.minimum_beta_shape,
            "maximum_leaf_variance": self.summary.maximum_leaf_variance,
            "median_inner_land_correlation": self.summary.median_inner_land_correlation,
            "topology_iterations": self.topology_iterations,
            "correlation_length_scale_degrees": (
                None if distance is None else distance.correlation_fit.length_scale
            ),
            "correlation_length_scale_rmse": (
                None if distance is None else distance.correlation_fit.rmse
            ),
            "correlation_length_scale_pair_correlation": (
                None if distance is None else distance.correlation_fit.target_model_correlation
            ),
        }


def build_calibrated_case(
    *,
    data_directory: Path,
    topology_weight_mode: Literal["sensitivity", "flat"],
    target_relative_standard_deviation: float,
    draws: int = 2,
    inner_regions: int = 250,
    max_depth: int = 8,
    seed: int = 20260718,
    include_distance_comparison: bool = False,
) -> CalibratedCase:
    """Build a moment-controlled topology and solve its UK root variance.

    Topology admissibility depends on the current root variance. The method
    therefore alternates deterministic topology construction with the exact
    affine root-variance solution until both are unchanged.

    Args:
        data_directory: Directory containing OGI and country fixtures.
        topology_weight_mode: Sensitivity or flat grid-cell topology weights.
        target_relative_standard_deviation: Requested UK prior relative SD.
        draws: Minimal prior draws retained for demonstration diagnostics.
        inner_regions: Upper terminal-region budget across inner land/ocean.
        max_depth: Maximum candidate effective split depth.
        seed: Reproducible NumPy seed.
        include_distance_comparison: Whether to fit the exact projected
            exponential-distance diagnostic after calibration.

    Returns:
        Converged case, exact aggregate moments, and optional distance fit.

    Raises:
        RuntimeError: If the target is below the split-contrast floor or the
            constrained topology does not stabilize.
    """
    split_constraint = MomentSplitConstraint(
        min_beta_shape=_CONTROLLED_MIN_BETA_SHAPE,
        max_child_variance=_CONTROLLED_MAX_CHILD_VARIANCE,
        allow_fewer_regions=True,
    )
    root_variance = 0.0
    previous_labels: np.ndarray | None = None

    for iteration in range(1, 9):
        case = build_case(
            data_directory=data_directory,
            draws=draws,
            inner_regions=inner_regions,
            max_depth=max_depth,
            base_kappa=_CONTROLLED_BASE_KAPPA,
            depth_multiplier=_CONTROLLED_DEPTH_MULTIPLIER,
            max_kappa=_CONTROLLED_MAX_KAPPA,
            inner_land_root_variance=root_variance,
            inner_ocean_root_variance=_CONTROLLED_INNER_OCEAN_ROOT_VARIANCE,
            topology_weight_mode=topology_weight_mode,
            split_constraint=split_constraint,
            seed=seed,
        )
        uk_mask = load_country_mask(case, data_directory, _UK_NAME)
        included_mass = case.expected_mass * uk_mask
        calibration = calibrate_group_root_variance(
            case.samples,
            included_mass,
            group_name="inner_land",
            target_relative_standard_deviation=target_relative_standard_deviation,
        )
        if not calibration.feasible or calibration.calibrated_root_variance is None:
            raise RuntimeError(
                f"UK target {target_relative_standard_deviation:.1%} is below the "
                f"contrast-only floor {calibration.minimum_relative_standard_deviation:.1%}."
            )
        labels = case.forest.leaf_labels()
        next_root_variance = calibration.calibrated_root_variance
        if previous_labels is not None:
            stable_topology = np.array_equal(labels, previous_labels)
            stable_variance = np.isclose(next_root_variance, root_variance, rtol=0.0, atol=1.0e-12)
            if stable_topology and stable_variance:
                aggregate = aggregate_prior_moments(case.samples, included_mass)
                if not np.isclose(
                    aggregate.relative_standard_deviation,
                    target_relative_standard_deviation,
                    rtol=0.0,
                    atol=1.0e-10,
                ):
                    raise RuntimeError("Converged topology did not reproduce the requested UK target.")
                summary = summarize_case(case)
                distance = (
                    build_distance_covariance_comparison(case)
                    if include_distance_comparison
                    else None
                )
                return CalibratedCase(
                    topology_weight_mode=topology_weight_mode,
                    target_relative_standard_deviation=target_relative_standard_deviation,
                    case=case,
                    summary=summary,
                    aggregate=aggregate,
                    calibration=calibration,
                    topology_iterations=iteration,
                    distance_comparison=distance,
                )
        previous_labels = labels
        root_variance = next_root_variance

    raise RuntimeError("Moment-constrained topology did not stabilize within eight iterations.")


def run_calibration(
    *,
    data_directory: Path,
    draws: int = 2,
    inner_regions: int = 250,
    max_depth: int = 8,
    seed: int = 20260718,
) -> tuple[dict[str, Any], tuple[CalibratedCase, ...]]:
    """Run baseline diagnostics and four controlled target calibrations.

    Args:
        data_directory: Directory containing OGI and country fixtures.
        draws: Minimal prior draws retained for demonstration diagnostics.
        inner_regions: Upper inner land/ocean terminal-region budget.
        max_depth: Maximum candidate effective split depth.
        seed: Reproducible NumPy seed.

    Returns:
        Baseline dictionary and controlled sensitivity/flat cases at 20% and
        50% UK prior relative SD.
    """
    baseline = build_case(
        data_directory=data_directory,
        draws=draws,
        inner_regions=inner_regions,
        max_depth=max_depth,
        seed=seed,
    )
    uk_mask = load_country_mask(baseline, data_directory, _UK_NAME)
    included_mass = baseline.expected_mass * uk_mask
    baseline_floor = aggregate_prior_moments(
        baseline.samples,
        included_mass,
        root_variances={"inner_land": 0.0},
    )
    baseline_current = aggregate_prior_moments(baseline.samples, included_mass)
    baseline_diagnostics = {
        "description": "original depth policy on the sensitivity-weighted topology",
        "base_kappa": baseline.strategy.base_kappa,
        "depth_multiplier": baseline.strategy.depth_multiplier,
        "max_kappa": baseline.strategy.max_kappa,
        "inner_land_root_variance_for_floor": 0.0,
        "contrast_only_minimum_relative_standard_deviation": (
            baseline_floor.relative_standard_deviation
        ),
        "original_inner_land_root_variance": 1.0,
        "original_relative_standard_deviation": baseline_current.relative_standard_deviation,
        "uk_terminal_region_count": int(np.count_nonzero(baseline_floor.terminal_weights)),
    }

    calibrated = tuple(
        build_calibrated_case(
            data_directory=data_directory,
            topology_weight_mode=mode,
            target_relative_standard_deviation=target,
            draws=draws,
            inner_regions=inner_regions,
            max_depth=max_depth,
            seed=seed,
            include_distance_comparison=mode == "flat",
        )
        for mode in ("sensitivity", "flat")
        for target in (0.2, 0.5)
    )
    return baseline_diagnostics, calibrated


def write_calibration_report(
    baseline: dict[str, Any],
    calibrated: tuple[CalibratedCase, ...],
    output_directory: Path,
) -> None:
    """Write numeric results, a summary plot, and a Markdown interpretation.

    Args:
        baseline: Original-prior diagnostics from :func:`run_calibration`.
        calibrated: Four controlled calibration results.
        output_directory: Destination directory for report artifacts.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline,
        "controlled_policy": {
            "base_kappa": _CONTROLLED_BASE_KAPPA,
            "depth_multiplier": _CONTROLLED_DEPTH_MULTIPLIER,
            "max_kappa": _CONTROLLED_MAX_KAPPA,
            "minimum_beta_shape": _CONTROLLED_MIN_BETA_SHAPE,
            "maximum_child_variance": _CONTROLLED_MAX_CHILD_VARIANCE,
            "fixed_inner_ocean_root_variance": _CONTROLLED_INNER_OCEAN_ROOT_VARIANCE,
        },
        "calibrations": [result.as_dict() for result in calibrated],
    }
    (output_directory / "gamma_beta_uk_calibration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _plot_calibration(baseline, calibrated, output_directory / "gamma_beta_uk_calibration.png")
    (output_directory / "gamma_beta_uk_calibration.md").write_text(
        _calibration_markdown(baseline, calibrated)
    )


def _plot_calibration(
    baseline: dict[str, Any],
    calibrated: tuple[CalibratedCase, ...],
    output_path: Path,
) -> None:
    """Plot aggregate uncertainty, achieved region counts, and fitted scales."""
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    labels = [
        "Original\nroot fixed",
        *[
            f"{result.topology_weight_mode.title()}\n{result.target_relative_standard_deviation:.0%}"
            for result in calibrated
        ],
    ]
    relative_sd = [
        100.0 * baseline["contrast_only_minimum_relative_standard_deviation"],
        *[100.0 * result.aggregate.relative_standard_deviation for result in calibrated],
    ]
    bar_colors = ["#555555", "#2878b5", "#2878b5", "#d95f02", "#d95f02"]
    axes[0].bar(np.arange(len(labels)), relative_sd, color=bar_colors)
    axes[0].axhspan(20.0, 50.0, color="#7fbf7b", alpha=0.18, label="20--50% target")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axes[0].set_ylabel("UK prior relative SD (%)")
    axes[0].set_title("Exact country-total calibration")
    axes[0].legend(frameon=False, fontsize=8)

    achieved_regions = [
        result.summary.inner_ocean_regions + result.summary.inner_land_regions
        for result in calibrated
    ]
    axes[1].bar(np.arange(len(calibrated)), achieved_regions, color=bar_colors[1:])
    axes[1].axhline(250, color="black", linestyle="--", linewidth=1, label="requested upper budget")
    axes[1].set_xticks(np.arange(len(calibrated)), labels[1:], rotation=20, ha="right")
    axes[1].set_ylabel("Inner terminal regions")
    axes[1].set_title("Moment-constrained achieved K")
    axes[1].legend(frameon=False, fontsize=8)

    flat_results = [result for result in calibrated if result.topology_weight_mode == "flat"]
    targets = [100.0 * result.target_relative_standard_deviation for result in flat_results]
    scales = [
        result.distance_comparison.correlation_fit.length_scale
        for result in flat_results
        if result.distance_comparison is not None
    ]
    axes[2].plot(targets, scales, marker="o", color="#8c510a", linewidth=1.5)
    for target, scale in zip(targets, scales):
        axes[2].annotate(f"{scale:.1f} deg", (target, scale), xytext=(4, 5), textcoords="offset points")
    axes[2].set_xlabel("Calibrated UK prior relative SD (%)")
    axes[2].set_ylabel("Fitted correlation length (degrees)")
    axes[2].set_title("Flat-topology distance diagnostic")
    axes[2].grid(alpha=0.25)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _calibration_markdown(
    baseline: dict[str, Any],
    calibrated: tuple[CalibratedCase, ...],
) -> str:
    """Return the complete calibration report as Markdown."""
    rows = []
    for result in calibrated:
        diagnostics = result.as_dict()
        length_scale = diagnostics["correlation_length_scale_degrees"]
        length_text = "--" if length_scale is None else f"{length_scale:.3g}"
        rows.append(
            "| {mode} | {target:.0%} | {root:.5g} | {achieved:.2%} | "
            "{ocean}/{land} | {actual} | {shape:.3g} | {variance:.3g} | {length} |".format(
                mode=result.topology_weight_mode,
                target=result.target_relative_standard_deviation,
                root=result.calibration.calibrated_root_variance,
                achieved=result.aggregate.relative_standard_deviation,
                ocean=result.case.inner_region_targets[0],
                land=result.case.inner_region_targets[1],
                actual=result.summary.inner_ocean_regions + result.summary.inner_land_regions,
                shape=result.summary.minimum_beta_shape,
                variance=result.summary.maximum_leaf_variance,
                length=length_text,
            )
        )
    table = "\n".join(rows)
    flat_results = [result for result in calibrated if result.topology_weight_mode == "flat"]
    flat_details = "\n".join(
        "- At {target:.0%} UK relative SD: correlation fit `ell={ell:.4g}` degrees, "
        "RMSE `{rmse:.4g}`, target/model pair correlation `{correlation:.3f}` over "
        "{pairs} inner-land pairs.".format(
            target=result.target_relative_standard_deviation,
            ell=result.distance_comparison.correlation_fit.length_scale,
            rmse=result.distance_comparison.correlation_fit.rmse,
            correlation=result.distance_comparison.correlation_fit.target_model_correlation,
            pairs=result.distance_comparison.correlation_fit.pair_count,
        )
        for result in flat_results
        if result.distance_comparison is not None
    )
    return f"""# Gamma--Beta UK prior calibration

This experiment calibrates the prior uncertainty of the UK country total on
the committed EUROPE grid. The aggregate is absolute prior flux times grid area
inside the country mask. The methane fixture is positive over the UK, so this
also equals the signed prior total for this demonstration.

![Calibration summary](gamma_beta_uk_calibration.png)

## Exact aggregate calculation

For terminal-region scaling vector `x`, analytic covariance `C`, and UK mass in
each terminal region `w`,

```text
E[T_UK] = sum(w)
Var(T_UK) = w.T @ C @ w
relative_SD(T_UK) = sqrt(Var(T_UK)) / E[T_UK].
```

No finite-draw covariance estimate enters this calibration. For fixed topology
and split concentrations, aggregate variance is affine in the inner-land root
variance. Evaluating exact covariance at root variance zero and one therefore
gives the root solution directly.

The original depth policy has a root-fixed UK relative SD of
{baseline['contrast_only_minimum_relative_standard_deviation']:.1%}; restoring its original root
variance of one raises this to {baseline['original_relative_standard_deviation']:.1%}. A non-negative root
variance can only add uncertainty, so neither the 20% nor 50% target is feasible
under those split contrasts.

## Controlled policy

The comparison policy is

```text
kappa(d) = min(96, 40 * 1.5**d)
minimum Beta shape = 1
maximum exact child scaling variance = 9
```

The existing priority queue skips a proposed split when either moment limit
fails, then continues with the next highest-weight admissible candidate. The
requested 250 inner regions are therefore an upper budget rather than a
promise to create unstable regions.

Only the inner-land root is calibrated. The independent inner-ocean root
variance is held fixed at {_CONTROLLED_INNER_OCEAN_ROOT_VARIANCE:g}; UK tuning
therefore cannot change ocean uncertainty or ocean split admissibility.

| Topology weight | UK target | Inner-land root variance | Achieved UK SD | Requested ocean/land | Achieved inner K | Min Beta shape | Max leaf variance | Fitted ell (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

The two targets do not identify a unique prior. Here the controlled split
policy fixes the allocation-contrast floor close to 20%; the shared inner-land
root then supplies the additional common-mode uncertainty needed to reach 50%.
The 20% calibration consequently has an almost fixed land root, whereas the
50% calibration has root variance about 0.20.

Sensitivity topology spends nearly all resolution over land because it uses
mean absolute footprint-times-flux sensitivity. Flat topology gives every
mapped grid cell equal selection weight and therefore allocates the 250-region
budget approximately evenly between inner ocean and land. These are different
topologies, not merely different covariance parameters.

## Effective distance scale

For the flat cases, a native-grid covariance

```text
exp(-abs(delta_lat) / ell) * exp(-abs(delta_lon) / ell)
```

is projected to the exact terminal regions with the expected-mass restriction,
then fitted to the Gamma--Beta inner-land correlation matrix.

{flat_details}

This `ell` is descriptive, not an identity implied by flat weights. A dyadic
Gamma--Beta prior is organized by common-ancestor depth. Nearby grid cells on
opposite sides of an early tree boundary can be less correlated than more
distant grid cells sharing a recent ancestor. Irregular land support and
expected-flux split fractions add further departures from a distance-only
model. The shared inner-land root also acts as a continent-wide common mode.
The much larger fitted scale at 50% is consistent with that added common mode,
although the exponential model remains only a moderate approximation.

Degrees are retained because the diagnostic kernel is separable in latitude
and longitude. One latitude degree is roughly 111 km, but longitude distance
depends on latitude, so these fitted values should not be converted to a single
physical range without replacing the kernel by a physical-distance model.

## Interpretation and next checks

This produces numerically controlled priors at both requested UK uncertainty
levels while keeping `kappa <= 96`. It does not establish that this depth
profile is scientifically unique. One country total cannot identify base
kappa, depth growth, cap, stopping thresholds, and root variance separately.
The same diagnostics should next be checked for other countries, land/ocean
totals, and sectors. If country totals are intended as hard prior contracts,
country groups would provide a cleaner parameterization: a UK root variance of
0.04 or 0.25 gives 20% or 50% relative SD directly, while within-country kappa
would control only spatial allocation.
"""


def build_parser() -> argparse.ArgumentParser:
    """Return command-line arguments for the calibration report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument("--output-directory", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--inner-regions", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run exact aggregate calibrations and write report artifacts."""
    args = build_parser().parse_args(argv)
    baseline, calibrated = run_calibration(
        data_directory=args.data_directory,
        draws=args.draws,
        inner_regions=args.inner_regions,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    write_calibration_report(baseline, calibrated, args.output_directory)
    print(
        json.dumps(
            {
                "baseline": baseline,
                "calibrations": [result.as_dict() for result in calibrated],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
