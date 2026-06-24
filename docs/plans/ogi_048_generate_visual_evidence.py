"""Generate visual evidence for OGI-048 lat/lon split geometry.

This script uses repository test data to compare count-based split steps with
and without ``LatLonGridGeometry``. It writes PNG figures plus a markdown report
under ``docs/plans`` so the output can be attached to the PR and later refined
into documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from openghg_inversions.basis.algorithms import (
    AxisParallelSplitStep,
    GreedyAxisParallelSplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    region_constrained_basis,
)
from openghg_inversions.basis.basis_functions import BasisFunctions


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "plans" / "figures" / "ogi_048_visual_evidence"
REPORT_PATH = ROOT / "docs" / "plans" / "ogi_048_visual_evidence.md"

FLUX_PATH = ROOT / "tests" / "data" / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
FOOTPRINT_PATH = ROOT / "tests" / "data" / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
COUNTRY_PATH = ROOT / "tests" / "data" / "country_EUROPE.nc"
BUCKET_BASIS_PATH = ROOT / "tests" / "data" / "basis" / "EUROPE" / "bucket_ch4-test_basis_EUROPE_2019.nc"
QUADTREE_BASIS_PATH = ROOT / "tests" / "data" / "basis" / "EUROPE" / "quadtree_ch4-test_basis_EUROPE_2019.nc"

BASIS_COUNTS = (50, 100, 250, 500)


@dataclass(frozen=True)
class Scenario:
    """One generated basis comparison scenario."""

    split_step: str
    geometry: str
    classes: str
    nbasis: int
    actual_regions: int
    basis_functions: BasisFunctions


def main() -> None:
    """Generate all figures and the markdown report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    weights, flux, country = load_inputs()
    landsea_classes = make_landsea_classes(country, weights)
    all_classes = xr.DataArray(
        np.full(weights.shape, "all", dtype=object),
        dims=weights.dims,
        coords=weights.coords,
        name="region_class",
    )
    geometry = LatLonGridGeometry.from_dataarray(weights)

    scenarios: list[Scenario] = []
    labels_by_key: dict[tuple[str, str, str, int], xr.DataArray] = {}

    for split_step_name in ("axis-parallel", "inertial"):
        for class_name, region_classes in (("all-cells", all_classes), ("ocean-land", landsea_classes)):
            for geometry_name, split_geometry in (("row-column", None), ("lat-lon-metres", geometry)):
                for nbasis in BASIS_COUNTS:
                    labels = build_basis_labels(
                        weights=weights,
                        region_classes=region_classes,
                        nbasis=nbasis,
                        split_step_name=split_step_name,
                        geometry=split_geometry,
                    )
                    if labels_cross_region_classes(labels, region_classes):
                        raise AssertionError(
                            f"{split_step_name}/{class_name}/{geometry_name}/{nbasis} crosses class boundaries."
                        )
                    basis_functions = BasisFunctions.from_flat_basis(
                        basis_flat=labels.rename("basis"),
                        flux=flux.rename("flux"),
                        region_labels="basis_values",
                        metadata={
                            "openghg_inversions:basis_artifact_source": "ogi-048-visual-evidence",
                            "openghg_inversions:split_step": split_step_name,
                            "openghg_inversions:geometry": geometry_name,
                            "openghg_inversions:region_classes": class_name,
                        },
                    )
                    labels_by_key[(split_step_name, class_name, geometry_name, nbasis)] = (
                        validated_flat_basis(basis_functions, labels)
                    )
                    scenarios.append(
                        Scenario(
                            split_step=split_step_name,
                            geometry=geometry_name,
                            classes=class_name,
                            nbasis=nbasis,
                            actual_regions=count_regions(labels),
                            basis_functions=basis_functions,
                        )
                    )

    input_figure = plot_inputs(weights, landsea_classes)
    matrix_figures = []
    for split_step_name in ("axis-parallel", "inertial"):
        for class_name in ("all-cells", "ocean-land"):
            matrix_figures.append(
                plot_matrix(
                    labels_by_key=labels_by_key,
                    split_step_name=split_step_name,
                    class_name=class_name,
                    landsea_classes=landsea_classes,
                )
            )

    write_report(
        scenarios=scenarios,
        input_figure=input_figure,
        matrix_figures=matrix_figures,
    )


def load_inputs() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load and align the test-data weight, flux, and country fields."""
    flux = (
        xr.open_dataset(FLUX_PATH)
        .flux.squeeze("time", drop=True)
        .transpose("lat", "lon")
        .astype(np.float64)
        .load()
    )
    footprint = (
        xr.open_dataset(FOOTPRINT_PATH).fp.mean("time").transpose("lat", "lon").astype(np.float64).load()
    )
    flux = align_to_test_grid(reference=footprint, target=flux, target_name="flux")
    weights = (footprint * flux).fillna(0.0)
    max_weight = float(weights.max())
    if max_weight > 0.0:
        weights = weights / max_weight
    weights = weights.rename("weight")

    country = xr.open_dataset(COUNTRY_PATH).country.transpose("lat", "lon").load()
    country = align_to_test_grid(reference=weights, target=country, target_name="country")
    flux = flux.assign_coords({dim: weights.coords[dim] for dim in weights.dims})
    return weights, flux, country


def align_to_test_grid(
    *,
    reference: xr.DataArray,
    target: xr.DataArray,
    target_name: str,
    atol: float = 1e-4,
) -> xr.DataArray:
    """Align a test-data field after checking it uses the same near-identical grid."""
    if target.dims != reference.dims:
        raise ValueError(
            f"{target_name} dims {target.dims!r} do not match reference dims {reference.dims!r}."
        )

    for dim in reference.dims:
        if target.sizes[dim] != reference.sizes[dim]:
            raise ValueError(
                f"{target_name} dimension {dim!r} has size {target.sizes[dim]}, "
                f"expected {reference.sizes[dim]}."
            )
        if dim in target.coords and dim in reference.coords:
            target_coord = np.asarray(target.coords[dim].values, dtype=float)
            reference_coord = np.asarray(reference.coords[dim].values, dtype=float)
            if not np.allclose(target_coord, reference_coord, rtol=0.0, atol=atol):
                max_delta = float(np.max(np.abs(target_coord - reference_coord)))
                raise ValueError(
                    f"{target_name} coordinate {dim!r} differs from the reference grid "
                    f"by up to {max_delta:g}, exceeding tolerance {atol:g}."
                )

    return target.assign_coords({dim: reference.coords[dim] for dim in reference.dims})


def make_landsea_classes(country: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Convert the test country grid into ocean/land region classes."""
    classes = xr.where(country > 0, "land", "ocean").astype(object)
    return classes.rename("region_class").assign_coords({dim: weights.coords[dim] for dim in weights.dims})


def validated_flat_basis(basis_functions: BasisFunctions, labels: xr.DataArray) -> xr.DataArray:
    """Return the flat basis after checking the BasisFunctions object is coherent."""
    flat_basis = basis_functions.flat_basis()
    if isinstance(flat_basis, dict):
        raise TypeError("Expected a single-source BasisFunctions object, got a multi-source flat basis.")

    state_dim = basis_functions.operator.meta.state_dim
    actual_regions = count_regions(labels)
    matrix_regions = basis_functions.operator.basis_matrix.sizes[state_dim]
    if matrix_regions != actual_regions:
        raise AssertionError(
            f"BasisFunctions state dimension has {matrix_regions} regions, expected {actual_regions}."
        )

    xr.testing.assert_equal(flat_basis.load(), labels)
    return cast(xr.DataArray, flat_basis)


def labels_cross_region_classes(labels: xr.DataArray, region_classes: xr.DataArray) -> bool:
    """Return true if any positive basis label spans multiple region classes."""
    label_values = np.asarray(labels.values).ravel()
    class_values = np.asarray(region_classes.values, dtype=object).ravel()
    _, class_codes = np.unique(class_values, return_inverse=True)
    order = np.lexsort((class_codes, label_values))
    sorted_labels = label_values[order]
    sorted_class_codes = class_codes[order]
    positive = sorted_labels > 0
    sorted_labels = sorted_labels[positive]
    sorted_class_codes = sorted_class_codes[positive]
    if sorted_labels.size == 0:
        return False

    label_change = np.r_[True, sorted_labels[1:] != sorted_labels[:-1]]
    class_change = np.r_[False, sorted_class_codes[1:] != sorted_class_codes[:-1]]
    return bool(np.any(class_change & ~label_change))


def build_basis_labels(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: int,
    split_step_name: str,
    geometry: LatLonGridGeometry | None,
) -> xr.DataArray:
    """Build one count-based region-constrained basis label map."""
    if split_step_name == "axis-parallel":
        split_step = AxisParallelSplitStep(balanced=False, clean_splits=True, geometry=geometry)
    elif split_step_name == "inertial":
        split_step = InertialSplitStep(balanced=False, geometry=geometry)
    else:
        raise ValueError(f"Unknown split step {split_step_name!r}.")

    labels = region_constrained_basis(
        weights,
        region_classes,
        nbasis,
        allocation="area",
        split_strategy=GreedyAxisParallelSplitStrategy(split_step=split_step),
    )
    return labels.astype(np.int32).rename("basis")


def count_regions(labels: xr.DataArray) -> int:
    """Return the number of positive basis regions."""
    values = np.asarray(labels.values)
    return int(np.count_nonzero(np.unique(values[values > 0])))


def plot_inputs(weights: xr.DataArray, landsea_classes: xr.DataArray) -> str:
    """Plot source weights, ocean/land classes, and existing test basis files."""
    bucket = (
        xr.open_dataset(BUCKET_BASIS_PATH).basis.squeeze("time", drop=True).transpose("lat", "lon").load()
    )
    quadtree = (
        xr.open_dataset(QUADTREE_BASIS_PATH).basis.squeeze("time", drop=True).transpose("lat", "lon").load()
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()

    log_weight = np.log10(weights.where(weights > 0.0))
    mesh = axes[0].pcolormesh(weights.lon, weights.lat, log_weight, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[0], shrink=0.8, label="log10 normalized footprint x flux")
    axes[0].set_title("TAC test weight field")

    land_values = xr.where(landsea_classes == "land", 1, 0)
    mesh = axes[1].pcolormesh(
        weights.lon, weights.lat, land_values, shading="auto", cmap="BrBG", vmin=0, vmax=1
    )
    fig.colorbar(mesh, ax=axes[1], shrink=0.8, ticks=[0, 1], label="0=ocean, 1=land")
    axes[1].set_title("Country-derived ocean/land classes")

    plot_basis_panel(axes[2], bucket, title=f"Existing bucket test basis ({count_regions(bucket)} regions)")
    plot_basis_panel(
        axes[3], quadtree, title=f"Existing quadtree test basis ({count_regions(quadtree)} regions)"
    )

    for ax in axes:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    filename = "inputs_and_existing_test_basis.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_visual_evidence/{filename}"


def plot_matrix(
    *,
    labels_by_key: dict[tuple[str, str, str, int], xr.DataArray],
    split_step_name: str,
    class_name: str,
    landsea_classes: xr.DataArray,
) -> str:
    """Plot one split-step/class matrix across region counts and geometry modes."""
    fig, axes = plt.subplots(
        nrows=len(BASIS_COUNTS),
        ncols=2,
        figsize=(11, 13),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    geometry_names = ("row-column", "lat-lon-metres")

    for row, nbasis in enumerate(BASIS_COUNTS):
        for col, geometry_name in enumerate(geometry_names):
            labels = labels_by_key[(split_step_name, class_name, geometry_name, nbasis)]
            ax = axes[row, col]
            title = f"{geometry_name}, target {nbasis}, actual {count_regions(labels)}"
            plot_basis_panel(ax, labels, title=title)
            overlay_land_outline(ax, landsea_classes)
            if row == len(BASIS_COUNTS) - 1:
                ax.set_xlabel("lon")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("lat")
            else:
                ax.set_ylabel("")

    fig.suptitle(f"{split_step_name}; classes={class_name}; count-based splits", fontsize=14)
    filename = f"{split_step_name.replace('-', '_')}_{class_name.replace('-', '_')}.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_visual_evidence/{filename}"


def plot_basis_panel(ax: Axes, labels: xr.DataArray, *, title: str) -> None:
    """Plot one basis label map without a colorbar."""
    values = labels.values.astype(float)
    ax.pcolormesh(labels.lon, labels.lat, values, shading="auto", cmap="nipy_spectral", rasterized=True)
    ax.set_title(title, fontsize=10)


def overlay_land_outline(ax: Axes, landsea_classes: xr.DataArray) -> None:
    """Draw a thin outline around land cells for spatial orientation."""
    land_values = xr.where(landsea_classes == "land", 1.0, 0.0)
    ax.contour(
        landsea_classes.lon,
        landsea_classes.lat,
        land_values,
        levels=[0.5],
        colors="black",
        linewidths=0.25,
    )


def write_report(
    *,
    scenarios: list[Scenario],
    input_figure: str,
    matrix_figures: list[str],
) -> None:
    """Write the markdown evidence report."""
    lines = [
        "# OGI-048 Visual Evidence: Lat/Lon Split Geometry",
        "",
        "Generated from repository test data on the EUROPE grid.",
        "",
        "Inputs:",
        "",
        "- Flux: `tests/data/flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc`.",
        "- Footprint: `tests/data/footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc`.",
        "- Region classes: `tests/data/country_EUROPE.nc`, using country code `0` as an `ocean`/non-country proxy and positive country codes as `land`.",
        "- Existing baseline plots: `bucket_ch4-test_basis_EUROPE_2019.nc` and `quadtree_ch4-test_basis_EUROPE_2019.nc`.",
        "",
        "Method:",
        "",
        "- Each generated map is wrapped in a `BasisFunctions.from_flat_basis(...)` object before plotting; the script checks the object state dimension and flat-basis round trip.",
        "- `row-column` uses the old count-based coordinate system with no `SplitGeometry`.",
        "- `lat-lon-metres` uses `LatLonGridGeometry.from_dataarray(weights)`.",
        "- `axis-parallel` uses `AxisParallelSplitStep(balanced=False, clean_splits=True, ...)`.",
        "- `inertial` uses `InertialSplitStep(balanced=False, ...)`.",
        '- Class allocation uses `allocation="area"` so the ocean/land split changes class boundaries but not contribution-weight allocation.',
        "- Flux, footprint, and country fixtures have near-identical grids; the script checks dimensions and coordinates before assigning a common test grid.",
        "- Generated labels are checked so no basis label crosses the configured `all`, `ocean`, or `land` class boundary.",
        "- These are algorithm-level visual checks only; PR #482 does not add config or wrapper routing for geometry.",
        "",
        f"![Inputs and existing test basis]({input_figure})",
        "",
        "## Generated Basis Matrices",
        "",
    ]

    for figure in matrix_figures:
        title = Path(figure).stem.replace("_", " ")
        lines.extend([f"### {title.title()}", "", f"![{title}]({figure})", ""])

    lines.extend(
        [
            "## Scenario Summary",
            "",
            "| split step | classes | geometry | target regions | actual regions |",
            "|---|---|---|---:|---:|",
        ]
    )
    for scenario in scenarios:
        lines.append(
            f"| {scenario.split_step} | {scenario.classes} | {scenario.geometry} | "
            f"{scenario.nbasis} | {scenario.actual_regions} |"
        )

    lines.extend(
        [
            "",
            "## Notes For PR Review",
            "",
            "- The clearest visual difference is expected at high latitudes, where one degree of longitude is physically shorter than one degree at lower latitudes.",
            "- The ocean/land split should prevent a generated label from crossing the ocean/land class boundary.",
            "- Inertial splits use physical coordinates for projection only when `LatLonGridGeometry` is supplied.",
            "- Color values are label IDs, so colors are useful for shape inspection but should not be interpreted as stable region identity across panels.",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
