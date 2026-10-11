"""Compatibility adapters for explicitly legacy ``fp_all`` workflows until 0.9.

Numerical algorithms and retained basis artifact loaders live in ``basis``.
"""

import warnings
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from openghg_inversions.basis._functions import (
    _mean_fp_times_mean_flux, quadtree_basis_from_weights,
    bucket_basis_from_weights, region_constrained_basis_from_weights,
    fixed_outer_regions_basis_from_data,
)
from openghg_inversions.basis.algorithms import AllocationMode, NbasisAllocation
from openghg_inversions.basis.basis_functions import (
    BasisFunctions, FluxWeightedBasis, basis_functions_from_flat_basis, flux_from_data,
)
from openghg_inversions.basis._wrapper import make_basis_functions, load_basis_functions

def _flux_fp_from_fp_all(
    fp_all: dict, emissions_name: list[str] | None = None
) -> tuple[xr.DataArray, list[xr.DataArray]]:
    """Extract a flux field and site footprints from a legacy ``fp_all`` mapping.

    Args:
        fp_all: Dictionary returned by the merged-data preparation path. Flux
            data are expected under the ``".flux"`` key, while measurement-site
            footprint datasets are stored under the remaining site keys.
        emissions_name: Optional list of OpenGHG flux source names. When
            supplied, the first source is selected from ``fp_all[".flux"]``.
            When omitted, the first available flux entry is used.

    Returns:
        Tuple containing the selected flux ``DataArray`` and the list of
        footprint ``DataArray`` objects for all sites.
    """
    if emissions_name is not None:
        flux = fp_all[".flux"][emissions_name[0]].data.flux
    else:
        first_flux = next(iter(fp_all[".flux"].values()))
        flux = first_flux.data.flux

    flux = cast(xr.DataArray, flux)

    footprints: list[xr.DataArray] = [v.fp for k, v in fp_all.items() if not k.startswith(".")]

    return flux, footprints


def basis_weights_from_fp_all(
    fp_all: dict,
    emissions_name: list[str] | None = None,
    *,
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Build the standard 2D basis weight field from a legacy ``fp_all`` mapping.

    The generated basis algorithms historically computed weights internally
    from mean footprints multiplied by mean flux. This helper exposes that
    adapter step so algorithms and experiments can use already computed weight
    fields directly.

    Args:
        fp_all: Legacy merged-data dictionary produced by the data preparation
            path.
        emissions_name: Optional list of OpenGHG flux source names used to
            select emissions from ``fp_all``.
        abs_flux: If true, use absolute flux values before averaging.
        mask: Optional Boolean spatial mask. When supplied, weights outside the
            mask are dropped from the returned field.

    Returns:
        Two-dimensional weight field with spatial coordinates preserved.
    """
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    return _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()


def quadtree_basis_function(
    fp_all: dict,
    start_date: str,
    domain: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    country_directory: str | None = None,
    abs_flux: bool = False,
    seed: int | None = None,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Create a basis field with the quadtree algorithm.

    The domain is split with smaller grid cells for regions which contribute
    more to the a priori above-baseline mole fraction. This is based on the
    average footprint over the inversion period and the a priori emissions field.

    Dual annealing selects the split threshold to approach ``nbasis`` regions.

    Args:
        fp_all: Legacy merged-data dictionary produced by the data preparation
            path.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        emissions_name: Optional list of OpenGHG flux source names used to
            select emissions from ``fp_all``.
        nbasis: Desired number of basis regions.
        country_directory: Accepted for a consistent basis-algorithm interface;
            the quadtree algorithm does not use it.
        abs_flux: If true, use absolute flux values when constructing weights.
        seed: Optional seed passed to ``scipy.optimize.dual_annealing``.
        mask: Optional Boolean spatial mask for fitting basis functions over a
            sub-region.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    weights = basis_weights_from_fp_all(fp_all, emissions_name, abs_flux=abs_flux, mask=mask)
    return quadtree_basis_from_weights(weights, start_date, domain, nbasis=nbasis, seed=seed)


def bucket_basis_function(
    fp_all: dict,
    start_date: str,
    domain: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    country_directory: str | None = None,
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
    landsea_indices: np.ndarray | None = None,
) -> xr.DataArray:
    """Create a basis field with the legacy weighted bucket algorithm.

    This algorithm recursively splits weighted rectangles so each scaling region
    contains approximately the same total weight. The implementation also uses
    land/sea masks from ``country_directory`` through the lower-level weighted
    algorithm.

    Args:
        fp_all: Legacy merged-data dictionary produced by the data preparation
            path.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        emissions_name: Optional list of OpenGHG flux source names used to
            select emissions from ``fp_all``.
        nbasis: Desired number of basis regions.
        country_directory: Optional directory containing land/sea files. When
            omitted, default package files are used.
        abs_flux: If true, use absolute flux values when constructing weights.
        mask: Optional Boolean spatial mask for fitting basis functions over a
            sub-region.
        landsea_indices: Optional pre-aligned land/sea mask for the selected
            sub-region.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    weights = basis_weights_from_fp_all(fp_all, emissions_name, abs_flux=abs_flux, mask=mask)
    return bucket_basis_from_weights(
        weights,
        start_date,
        domain,
        nbasis=nbasis,
        country_directory=country_directory,
        landsea_indices=landsea_indices,
    )


def region_constrained_basis_function(
    fp_all: dict,
    start_date: str,
    domain: str,
    emissions_name: list[str] | None = None,
    nbasis: NbasisAllocation = 100,
    country_directory: str | None = None,
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
    region_classes: xr.DataArray | None = None,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_acceptance: Literal["none", "contrast_score"] = "none",
    contrast_contribution: xr.DataArray | None = None,
    contrast_cell_weight: xr.DataArray | None = None,
    min_contrast_delta_eig: float | None = None,
    min_contrast_lambda: float | None = None,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | None = None,
) -> xr.DataArray:
    """Create weighted basis regions constrained by caller-supplied classes.

    This adapter keeps file loading outside the constrained algorithm: callers
    provide ``region_classes`` directly, for example from a country file,
    land/sea file, or a user-defined region-class field. It links the pure
    ``region_constrained_basis`` helper to the current ``fp_all``-based wrapper
    interface by constructing the usual footprint-times-flux weight field first.

    Args:
        fp_all: Legacy merged-data dictionary produced by the data preparation
            path.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        emissions_name: Optional list of OpenGHG flux source names used to
            select emissions from ``fp_all``.
        nbasis: Total number of basis regions, or class-local allocation
            accepted by ``region_constrained_basis``.
        country_directory: Accepted for a consistent basis-algorithm interface;
            file loading for ``region_classes`` must happen before calling this
            adapter.
        abs_flux: If true, use absolute flux values when constructing weights.
        mask: Optional Boolean spatial mask for fitting basis functions over a
            sub-region.
        region_classes: Two-dimensional class field on the same spatial grid as
            the generated weights. Positive basis labels are generated
            independently within each non-null class value.
        allocation: Automatic allocation mode used when ``nbasis`` is an
            integer. ``"weight"`` allocates regions by total class weight;
            ``"area"`` allocates by mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.
        split_acceptance: Optional split-acceptance criterion. The default
            ``"none"`` preserves existing behavior. ``"contrast_score"`` uses
            a mass-preserving observation-space contrast gate.
        contrast_contribution: Design contribution array for contrast scoring,
            with at least one design-observation dimension plus the two spatial
            dimensions. Observed mole-fraction values must not be used here.
        contrast_cell_weight: Optional prior flux or split-mass field used as
            ``mu`` in contrast scoring. When omitted, the unnormalised generated
            basis weight field is used as a split-mass proxy.
        min_contrast_delta_eig: Optional minimum ``delta_eig`` threshold. If
            both contrast thresholds are omitted, contrast diagnostics are
            computed but proposed splits are not rejected.
        min_contrast_lambda: Optional minimum ``lambda`` threshold.
        contrast_tau: Prior standard deviation of the split contrast
            coefficient ``delta = alpha_A - alpha_B``. If omitted, ``tau=1`` is
            used and scores are uncalibrated.
        contrast_sigma_design: Optional scalar design standard deviation,
            equivalent to ``S = contrast_sigma_design**2 I``.
        contrast_s_diag: Optional diagonal design covariance entries in the
            same row space as ``contrast_contribution``.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and globally unique integer labels that do not cross
        ``region_classes`` values.

    Raises:
        ValueError: If ``region_classes`` is not supplied.
    """
    if region_classes is None:
        raise ValueError("region_classes must be supplied for the region_constrained basis algorithm.")

    weights = basis_weights_from_fp_all(fp_all, emissions_name, abs_flux=abs_flux, mask=mask)
    return region_constrained_basis_from_weights(
        weights,
        start_date,
        domain,
        region_classes=region_classes,
        nbasis=nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        split_acceptance=split_acceptance,
        contrast_contribution=contrast_contribution,
        contrast_cell_weight=contrast_cell_weight,
        min_contrast_delta_eig=min_contrast_delta_eig,
        min_contrast_lambda=min_contrast_lambda,
        contrast_tau=contrast_tau,
        contrast_sigma_design=contrast_sigma_design,
        contrast_s_diag=contrast_s_diag,
    )


def quadtreebasisfunction(*args, **kwargs) -> xr.DataArray:
    """Deprecated alias for :func:`quadtree_basis_function`."""
    warnings.warn(
        "`quadtreebasisfunction` is deprecated; use `quadtree_basis_function` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return quadtree_basis_function(*args, **kwargs)


def bucketbasisfunction(*args, **kwargs) -> xr.DataArray:
    """Deprecated alias for :func:`bucket_basis_function`."""
    warnings.warn(
        "`bucketbasisfunction` is deprecated; use `bucket_basis_function` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bucket_basis_function(*args, **kwargs)


def fixed_outer_regions_basis(
    fp_all: dict,
    start_date: str,
    basis_algorithm: str,
    domain: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    country_directory: str | None = None,
    abs_flux: bool = False,
    *,
    outer_regions_path: str | Path | None = None,
    region_classes: xr.DataArray | None = None,
    region_allocation: AllocationMode = "weight",
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
) -> xr.DataArray:
    """Adapt legacy merged dictionaries to dataset fixed-outer basis fitting.

    See :func:`fixed_outer_regions_basis_from_data` for numerical options.
    """
    return fixed_outer_regions_basis_from_data(
        {key: value for key, value in fp_all.items() if not key.startswith(".")},
        {key: _extract_flux_dataarray(value, flux_key=key).to_dataset(name="flux")
         for key, value in fp_all[".flux"].items()},
        start_date, basis_algorithm, domain, emissions_name, nbasis, country_directory, abs_flux,
        outer_regions_path=outer_regions_path, region_classes=region_classes,
        region_allocation=region_allocation, min_regions_per_class=min_regions_per_class,
        split_acceptance=split_acceptance, contrast_contribution=contrast_contribution,
        contrast_cell_weight=contrast_cell_weight, min_contrast_delta_eig=min_contrast_delta_eig,
        min_contrast_lambda=min_contrast_lambda, contrast_tau=contrast_tau,
        contrast_sigma_design=contrast_sigma_design, contrast_s_diag=contrast_s_diag,
        allow_empty_inner_region=allow_empty_inner_region,
    )


def basis_functions_from_fp_all_flat_basis(
    *,
    fp_all: dict,
    basis_flat: xr.DataArray | Mapping[str, xr.DataArray],
    metadata: Mapping[str, Any] | None = None,
) -> FluxWeightedBasis:
    """Adapt legacy merged dictionaries to retained flat-basis construction.

    New callers should pass runtime datasets to
    :func:`basis_functions_from_flat_basis` instead.
    """
    return basis_functions_from_flat_basis(
        flux_data={key: _extract_flux_dataarray(value, flux_key=key).to_dataset(name="flux")
                   for key, value in fp_all[".flux"].items()},
        split_by_sectors=_is_multi_source_workflow(fp_all),
        basis_flat=basis_flat,
        metadata=metadata,
    )


def flux_from_fp_all(fp_all: dict) -> xr.DataArray:
    """Legacy adapter that builds representative flux from ``fp_all``.

    This compatibility helper attaches current-run flux when loading a legacy
    flat basis artifact as retained ``BasisFunctions``.

    Args:
        fp_all: Legacy merged-data dictionary containing ``fp_all[".flux"]`` and
            optional ``fp_all[".split_by_sectors"]`` metadata.

    Returns:
        A combined flux array for non-sectoral workflows, or a source-stacked
        flux array for sectoral workflows. Source-stacked output follows the
        insertion order of ``fp_all[".flux"]``.

    Raises:
        ValueError: If ``fp_all[".flux"]`` is missing or empty, or a sectoral
            source mapping mixes timed and timeless arrays or contains unequal,
            missing, or duplicate native time coordinates.
        TypeError: If a flux entry cannot be converted to a ``DataArray``.
    """
    if ".flux" not in fp_all or not fp_all[".flux"]:
        raise ValueError("Cannot construct BasisFunctions object: fp_all['.flux'] is missing or empty.")

    return flux_from_data(
        {key: _extract_flux_dataarray(value, flux_key=key).to_dataset(name="flux")
         for key, value in fp_all[".flux"].items()},
        split_by_sectors=_is_multi_source_workflow(fp_all),
    )


def _is_multi_source_workflow(fp_all: dict) -> bool:
    """Interpret legacy ``fp_all`` metadata to determine multi-source mode."""
    split_by_sectors = fp_all.get(".split_by_sectors")
    if split_by_sectors is not None:
        return bool(split_by_sectors)

    flux_entries = fp_all.get(".flux")
    return isinstance(flux_entries, dict) and len(flux_entries) > 1


def _extract_flux_dataarray(flux_entry: object, flux_key: str) -> xr.DataArray:
    """Extract flux from legacy ``fp_all[".flux"]`` entry containers.

    Args:
        flux_entry: Legacy flux entry, xarray ``Dataset``, or ``DataArray``.
        flux_key: Source key used only for error reporting.

    Returns:
        The ``flux`` DataArray.

    Raises:
        TypeError: If the entry does not expose a flux DataArray.
    """
    flux_entry_data = getattr(flux_entry, "data", None)
    if isinstance(flux_entry_data, xr.Dataset) and "flux" in flux_entry_data:
        return flux_entry_data["flux"]
    if isinstance(flux_entry, xr.Dataset) and "flux" in flux_entry:
        return flux_entry["flux"]
    if isinstance(flux_entry, xr.DataArray):
        return flux_entry

    raise TypeError(
        "Could not extract a flux DataArray from fp_all['.flux']. "
        f"Got type {type(flux_entry)!r} for flux entry {flux_key!r}."
    )


def make_basis_functions_from_fp_all(*, fp_all: dict, **kwargs: Any) -> BasisFunctions:
    """Adapt legacy merged dictionaries to dataset basis construction."""
    return make_basis_functions(
        site_data={key: value for key, value in fp_all.items() if not key.startswith(".")},
        flux_data={key: _extract_flux_dataarray(value, flux_key=key).to_dataset(name="flux")
                   for key, value in fp_all[".flux"].items()},
        split_by_sectors=_is_multi_source_workflow(fp_all),
        **kwargs,
    )


def load_basis_functions_from_fp_all(*, fp_all: dict, **kwargs: Any) -> BasisFunctions:
    """Adapt legacy merged dictionaries to dataset basis artifact loading."""
    return load_basis_functions(
        flux_data={key: _extract_flux_dataarray(value, flux_key=key).to_dataset(name="flux")
                   for key, value in fp_all[".flux"].items()},
        split_by_sectors=_is_multi_source_workflow(fp_all),
        **kwargs,
    )


# dict to retrieve basis function and description by algorithm name
BasisFunction = namedtuple("BasisFunction", ["description", "algorithm"])
basis_functions = {
    "quadtree": BasisFunction("quadtree algorithm", quadtree_basis_function),
    "weighted": BasisFunction("weighted by data algorithm", bucket_basis_function),
    "region_constrained": BasisFunction(
        "region-constrained weighted by data algorithm",
        region_constrained_basis_function,
    ),
}
