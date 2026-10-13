"""Prepare retained basis operators from exact artifacts or resolved scientific inputs."""

from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Literal

import xarray as xr


from .basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
    basis_functions_from_flat_basis,
)
from ._functions import (
    basis_from_weights,
    _validate_basis_algorithm,
    basis_weights_from_data,
    fixed_outer_regions_basis_from_data,
)


def prepare_basis(
    *,
    site_data: Mapping[str, xr.Dataset],
    runtime_flux: xr.DataArray,
    weighting_flux: xr.DataArray,
    domain: str,
    start_date: str,
    nbasis: int,
    basis_algorithm: str | None = None,
    fix_outer_regions: bool = False,
    basis_path: str | Path | None = None,
    basis_partition: xr.DataArray | Mapping[str, xr.DataArray] | None = None,
    output_path: str | Path | None = None,
    landsea_classes: xr.DataArray | None = None,
    outer_regions: xr.DataArray | None = None,
    region_classes: xr.DataArray | None = None,
    region_allocation: Literal["weight", "area"] = "weight",
    min_regions_per_class: int = 1,
    split_acceptance: Literal["none", "contrast_score"] = "none",
    contrast_contribution: xr.DataArray | None = None,
    contrast_cell_weight: xr.DataArray | None = None,
    min_contrast_delta_eig: float | None = None,
    min_contrast_lambda: float | None = None,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | None = None,
    allow_empty_inner_region: bool = False,
) -> BasisFunctions:
    """Load or generate an operator, bind current flux, and optionally save it.

    ``basis_path`` and ``output_path`` are exact retained-artifact destinations
    (.nc or .zarr). Scientific support maps are borrowed labelled arrays;
    callers resolve paths and packaged resources before acquisition. Loading
    reuses the saved operator with current runtime flux. Reconstruction uses
    ``BasisFunctions.load`` to restore the complete saved object instead.
    Generation fits an eager numerical kernel; saving is an explicit
    serialization/execution boundary. Borrowed inputs are not modified.

    Args:
        site_data: Site datasets with footprints used to construct weights.
            Unused when an artifact or partition is selected.
        runtime_flux: Resolved native prior containing all retained sources on
            their inventory time axis, independent of observation-aligned
            responses. Replaces any saved prior when an operator is reused.
            Resolve source combination with
            :func:`~openghg_inversions.native_flux.resolve_runtime_flux`.
        weighting_flux: One source flux used for basis-design weights.
            Standard RHIME callers select the first requested source. Unused
            when an artifact or partition is selected.
        domain: Domain label for generated-basis metadata.
        start_date: Date attached to a generated partition's singleton time axis.
        nbasis: Requested generated-region count; with fixed outer regions,
            this counts inner regions only.
        basis_algorithm: Generation algorithm: ``"quadtree"``, ``"weighted"``
            or ``"region_constrained"``. Required only without a saved input.
        fix_outer_regions: Fit only the inner region of ``outer_regions``.
        basis_path: Exact retained .nc or .zarr artifact to load. Mutually
            exclusive with ``basis_partition``; either bypasses generation.
        basis_partition: Labelled partition, or source-keyed partitions, to
            bind to runtime flux. Spatial coordinates must match the flux grid.
        output_path: Exact retained-artifact destination. None disables saving;
            saving also applies to loaded operators and supplied partitions.
        landsea_classes: Resolved labelled map required for weighted fitting.
            Values 0 and 1 are split separately; other cells retain label 0.
        outer_regions: Resolved fixed-outer map on the flux grid, required when
            generating with ``fix_outer_regions=True``. The ``inner_region_label``
            attribute selects the inner label, falling back to the maximum label.
        region_classes: Resolved class map required for region-constrained fitting.
        region_allocation: Distribute the inner region budget by ``"weight"``
            (default) or ``"area"`` for region-constrained fitting.
        min_regions_per_class: Minimum automatic budget per non-empty class.
        split_acceptance: ``"none"`` accepts ordinary geometric splits;
            ``"contrast_score"`` applies the supplied contrast design criterion.
        contrast_contribution: Design contributions for contrast scoring.
        contrast_cell_weight: Optional prior-flux or split-mass field for scoring.
        min_contrast_delta_eig: Optional minimum contrast information gain.
        min_contrast_lambda: Optional minimum contrast ``lambda`` score.
        contrast_tau: Contrast prior standard deviation; None uses uncalibrated 1.
        contrast_sigma_design: Optional scalar design standard deviation.
        contrast_s_diag: Optional diagonal design covariance entries.
        allow_empty_inner_region: Keep a finite, all-zero inner region unsplit
            during fixed-outer fitting; intended for nested outer domains.

    Returns:
        Retained operator bound to runtime flux with artifact-source metadata
        and, when loaded or saved, the selected artifact path.

    Raises:
        ValueError: If saved input selections conflict, no input or algorithm is
            supplied, or required maps or scientific fitting inputs are invalid.
        OSError: If an artifact cannot be opened or written.
    """
    if basis_path is not None and basis_partition is not None:
        raise ValueError("Supply either basis_path or basis_partition.")
    if basis_path is None and basis_partition is None and basis_algorithm is not None:
        _validate_basis_algorithm(basis_algorithm)
    basis_data_array: xr.DataArray | None = None
    basis_start = time()

    if basis_path is not None:
        basis_path = Path(basis_path)
        basis_functions_object = BasisFunctions.reuse_operator(basis_path, flux=runtime_flux).with_metadata(
            {
                BASIS_ARTIFACT_SOURCE_ATTR: "datatree",
                BASIS_ARTIFACT_PATH_ATTR: str(basis_path),
            }
        )
    elif basis_partition is not None:
        basis_functions_object = basis_functions_from_flat_basis(
            flux=runtime_flux,
            basis_flat=basis_partition,
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: "partition"},
        )
    elif basis_algorithm is None:
        raise ValueError("One of basis_path or basis_algorithm must be specified.")
    elif fix_outer_regions is True:
        if outer_regions is None:
            raise ValueError("outer_regions must be resolved before fixed-outer basis construction.")
        print("Using fixed outer regions for basis functions.")
        basis_data_array = fixed_outer_regions_basis_from_data(
            site_data,
            weighting_flux,
            start_date,
            basis_algorithm,
            domain,
            nbasis,
            outer_regions=outer_regions,
            landsea_classes=landsea_classes,
            region_classes=region_classes,
            region_allocation=region_allocation,
            min_regions_per_class=min_regions_per_class,
            split_acceptance=split_acceptance,
            contrast_contribution=contrast_contribution,
            contrast_cell_weight=contrast_cell_weight,
            min_contrast_delta_eig=min_contrast_delta_eig,
            min_contrast_lambda=min_contrast_lambda,
            contrast_tau=contrast_tau,
            contrast_sigma_design=contrast_sigma_design,
            contrast_s_diag=contrast_s_diag,
            allow_empty_inner_region=allow_empty_inner_region,
        )
        print(f"Using InTEM regions with {basis_algorithm} to derive basis functions for inner region.")
        print("Using generated in-memory basis artifact.")
        basis_functions_object = basis_functions_from_flat_basis(
            flux=runtime_flux,
            basis_flat=basis_data_array,
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: "generated"},
        )

    else:
        algorithm_kwargs: dict[str, Any] = {"landsea_classes": landsea_classes}
        if basis_algorithm == "region_constrained":
            algorithm_kwargs.update(
                {
                    "region_classes": region_classes,
                    "allocation": region_allocation,
                    "min_regions_per_class": min_regions_per_class,
                    "split_acceptance": split_acceptance,
                    "contrast_contribution": contrast_contribution,
                    "contrast_cell_weight": contrast_cell_weight,
                    "min_contrast_delta_eig": min_contrast_delta_eig,
                    "min_contrast_lambda": min_contrast_lambda,
                    "contrast_tau": contrast_tau,
                    "contrast_sigma_design": contrast_sigma_design,
                    "contrast_s_diag": contrast_s_diag,
                }
            )
        weights = basis_weights_from_data(site_data, weighting_flux)
        basis_data_array = basis_from_weights(
            weights,
            start_date,
            domain,
            basis_algorithm,
            nbasis,
            **algorithm_kwargs,
        )
        print("Using generated in-memory basis artifact.")
        basis_functions_object = basis_functions_from_flat_basis(
            flux=runtime_flux,
            basis_flat=basis_data_array,
            metadata={BASIS_ARTIFACT_SOURCE_ATTR: "generated"},
        )

    print(f"Computing basis took {time() - basis_start}s.")

    if output_path is not None:
        basis_functions_object.save(output_path)
        basis_functions_object = basis_functions_object.with_metadata(
            {BASIS_ARTIFACT_PATH_ATTR: str(output_path)}
        )
    return basis_functions_object
