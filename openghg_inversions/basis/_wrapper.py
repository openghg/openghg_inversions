"""Wrappers for creating basis functions and applying them to sensitivities."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import time
from typing import Any, Literal

import xarray as xr


from .basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
    basis_functions_from_flat_basis,
    flux_from_data,
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
    flux_data: Mapping[str, xr.Dataset],
    split_by_sectors: bool = False,
    domain: str,
    start_date: str,
    flux_sources: Sequence[str] | None = None,
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
    Saving is an explicit serialization/execution boundary.
    """
    if basis_path is not None and basis_partition is not None:
        raise ValueError("Supply either basis_path or basis_partition.")
    if basis_path is None and basis_partition is None and basis_algorithm is not None:
        _validate_basis_algorithm(basis_algorithm)
    basis_data_array: xr.DataArray | None = None
    basis_start = time()

    if basis_path is not None:
        basis_path = Path(basis_path)
        basis_functions_object = BasisFunctions.load(basis_path)
        current_flux = flux_from_data(flux_data, split_by_sectors=split_by_sectors)
        if "source" in current_flux.dims:
            basis_functions_object = basis_functions_object.select_sources(
                [str(source) for source in current_flux.source.values]
            )
        basis_functions_object = basis_functions_object.with_flux(current_flux).with_metadata({
            BASIS_ARTIFACT_SOURCE_ATTR: "datatree", BASIS_ARTIFACT_PATH_ATTR: str(basis_path),
        })
    elif basis_partition is not None:
        basis_functions_object = basis_functions_from_flat_basis(
            flux_data=flux_data, split_by_sectors=split_by_sectors,
            basis_flat=basis_partition, metadata={BASIS_ARTIFACT_SOURCE_ATTR: "partition"},
        )
    elif basis_algorithm is None:
        raise ValueError("One of basis_path or basis_algorithm must be specified.")
    elif fix_outer_regions is True:
        print("Using fixed outer regions for basis functions.")
        basis_data_array = fixed_outer_regions_basis_from_data(
            site_data,
            flux_data,
            start_date,
            basis_algorithm,
            domain,
            flux_sources,
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
            flux_data=flux_data,
            split_by_sectors=split_by_sectors,
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
        weights = basis_weights_from_data(site_data, flux_data, flux_sources)
        basis_data_array = basis_from_weights(
            weights, start_date, domain, basis_algorithm, nbasis, **algorithm_kwargs,
        )
        print("Using generated in-memory basis artifact.")
        basis_functions_object = basis_functions_from_flat_basis(
            flux_data=flux_data,
            split_by_sectors=split_by_sectors,
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
