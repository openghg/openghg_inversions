"""Functions to create basis datasets from fluxes and footprints."""

import os
import warnings

import getpass
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import numpy as np
import xarray as xr
import logging

from .algorithms import (
    AllocationMode,
    ContrastScoreSplitAcceptance,
    GreedyAxisParallelSplitStrategy,
    NbasisAllocation,
    SplitStrategy,
    region_constrained_basis,
)
from .algorithms import quadtree_algorithm, weighted_algorithm

from openghg_inversions.config.paths import Paths
from openghg_inversions.utils import read_netcdfs

logger = logging.getLogger(__name__)


openghginv_path = Paths.openghginv


def basis(domain: str, basis_case: str, basis_directory: str | None = None) -> xr.Dataset:
    """Read in basis function(s) from file given basis case and domain, and return as an
    xarray Dataset.

    The basis function files should be stored as on paths of the form:
        <basis_directory>/<domain>/<basis_case>_<domain>*.nc

    For instance: domain = EUROPE, basis_directory = /group/chem/acrg/LPDM/basis_functions,
    and basis_case = sub_transd would find files such as:

        /group/chem/acrg/LPDM/basis_functions/EUROPE/sub_transd_EUROPE_2014.nc

    Basis functions created by algorithms in OpenGHG inversions will be stored using
    this path format.

    Args:
        domain: domain name. The basis files should be sub-categorised by the domain.
        basis_case: basis case to read in. Examples of basis cases are "voronoi", "sub-transd",
            "sub-country_mask", "INTEM".
        basis_directory: basis_directory can be specified if files are not in the default
            directory (i.e. `openghg_inversions/basis_functions`). Must point to a directory that
            contains subfolders organized by domain.

    Returns:
        xarray.Dataset: combined dataset of matching basis functions
    """
    if basis_directory is None:
        basis_path = openghginv_path / "basis_functions"
        if not basis_path.exists():
            basis_path.mkdir()
            raise ValueError(
                f"Default basis directory {basis_path} was empty. Add basis files or specify `basis_path`."
            )
    else:
        basis_path = Path(basis_directory)

    file_path = (basis_path / domain).glob(f"{basis_case}_{domain}*.nc")
    files = sorted(list(file_path))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Can't find basis function files for domain '{domain}' and basis_case '{basis_case}' "
        )

    basis_ds = read_netcdfs(files)

    return basis_ds


def basis_boundary_conditions(domain: str, basis_case: str, bc_basis_directory: str | None = None):
    """Read in basis function(s) from file given basis case and domain, and return as an
    xarray Dataset.

    The basis function files should be stored as on paths of the form:
        <bc_basis_directory>/<domain>/<basis_case>_<domain>*.nc

    For instance: domain = "EUROPE", bc_basis_directory = /group/chem/acrg/LPDM/bc_basis_functions,
    and basis_case = "NESW" would find files such as:

        /group/chem/acrg/LPDM/bc_basis_functions/EUROPE/NESW_EUROPE_2014.nc

    Args:
        domain: domain name. The basis files should be sub-categorised by the domain.
        basis_case: basis case to read in. Examples of BC basis cases are "NESW", "stratgrad".
        bc_basis_directory: bc_basis_directory can be specified if files are not in the default
            directory (i.e. `openghg_inversions/bc_basis_functions`). Must point to a directory that
            contains subfolders organized by domain.

    Returns:
        xarray.Dataset: combined dataset of matching basis functions
    """
    if bc_basis_directory is None:
        bc_basis_path = openghginv_path / "bc_basis_functions"
        if not bc_basis_path.exists():
            bc_basis_path.mkdir()
            raise ValueError(
                f"Default BC basis directory {bc_basis_path} was empty. "
                "Add basis files or specify `bc_basis_path`."
            )
    else:
        bc_basis_path = Path(bc_basis_directory)

    file_path = (bc_basis_path / domain).glob(f"{basis_case}_{domain}*.nc")
    files = sorted(list(file_path))

    # check for files that we can't access
    # NOTE: Hannah added this in 2021 to the ACRG code.
    # I don't know why it is only for BC boundary conditions -- BM, 2024
    file_no_acc = [ff for ff in files if not os.access(ff, os.R_OK)]
    if len(file_no_acc) > 0:
        print(
            "Warning: unable to read all boundary conditions basis function files which match this criteria:"
        )
        print("\n".join(map(str, file_no_acc)))

    # only use files we can access
    files = [ff for ff in files if ff not in file_no_acc]

    if len(files) == 0:
        raise FileNotFoundError(
            f"Can't find BC basis function files for domain '{domain}' and bc_basis_case '{basis_case}' "
        )

    basis_ds = read_netcdfs(files)

    return basis_ds


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


def _mean_fp_times_mean_flux(
    flux: xr.DataArray,
    footprints: list[xr.DataArray],
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Multiply mean flux by mean of footprints, optionally restricted to a Boolean mask.

    Args:
        flux: Flux field with a ``time`` dimension and spatial grid dimensions.
        footprints: Footprint fields for each site. Their time coordinates are
            outer-aligned before summing so every measurement contributes once.
        abs_flux: If true, use the absolute value of ``flux`` before averaging.
        mask: Optional Boolean spatial mask. When supplied, weights outside the
            mask are dropped from the returned field.

    Returns:
        Spatial weight field equal to temporal mean flux multiplied by the
        measurement-weighted temporal mean footprint.
    """
    if abs_flux is True:
        print("Using absolute value of flux array.")
        flux = abs(flux)

    mean_flux = flux.mean("time")

    # get total times before aligning
    n_measure = sum(len(fp.time) for fp in footprints)

    # align so that all times are used
    footprints = xr.align(*footprints, join="outer", fill_value=0.0)  # type: ignore  the docs say scalars are accepted as fill values, but type hints don't
    fp_total = sum(footprints)  # this seems to be faster than concatentating and summing over new axis

    fp_total = cast(xr.DataArray, fp_total)  # otherwise mypy complains about the next line
    mean_fp = fp_total.sum("time") / n_measure

    if mask is not None:
        # align to footprint lat/lon
        mean_fp, mean_flux, mask = xr.align(mean_fp, mean_flux, mask, join="override")
        return (mean_fp * mean_flux).where(mask, drop=True)

    mean_fp, mean_flux = xr.align(mean_fp, mean_flux, join="override")
    return mean_fp * mean_flux


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


def _sanitize_generated_basis_weights(
    weights: xr.DataArray,
    *,
    algorithm: str,
    require_nonzero: bool = False,
) -> xr.DataArray:
    """Replace non-finite generated-basis weights and reject empty weight fields."""
    weights = weights.as_numpy()
    finite = xr.apply_ufunc(np.isfinite, weights)
    if not bool(finite.any().item()):
        raise ValueError(f"{algorithm} generated-basis weights contain no finite values.")

    sanitized = weights.where(finite, 0.0)

    if require_nonzero and not bool((sanitized != 0.0).any().item()):
        raise ValueError(
            f"{algorithm} generated-basis weights contain no non-zero finite values "
            "after replacing non-finite values with zero."
        )

    return sanitized


def _normalise_weights_by_max(weights: xr.DataArray) -> xr.DataArray:
    """Return weights scaled by their maximum when the maximum is positive."""
    max_weight = float(weights.max())
    if max_weight > 0:
        return weights / max_weight
    return weights


def _normalise_weights_by_nonzero_max(weights: xr.DataArray) -> xr.DataArray:
    """Return weights scaled by their finite non-zero maximum."""
    max_weight = float(weights.max())
    if not np.isfinite(max_weight) or max_weight == 0.0:
        raise ValueError("generated-basis weights have no finite non-zero maximum.")
    return weights / max_weight


def _finalise_generated_basis(
    basis_field: xr.DataArray,
    *,
    start_date: str,
    domain: str,
) -> xr.DataArray:
    """Attach the legacy generated-basis dimensions, name, and metadata."""
    basis_field = basis_field.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    basis_field = basis_field.rename("basis")
    basis_field.attrs["creator"] = getpass.getuser()
    basis_field.attrs["date created"] = str(pd.Timestamp.today())
    basis_field.attrs["domain"] = domain
    return basis_field


def quadtree_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    nbasis: int = 100,
    seed: int | None = None,
) -> xr.DataArray:
    """Create a quadtree basis field from precomputed 2D weights.

    Args:
        weights: Two-dimensional basis weight field.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        nbasis: Desired number of basis regions.
        seed: Optional seed passed to ``scipy.optimize.dual_annealing``.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    weights = _sanitize_generated_basis_weights(weights, algorithm="quadtree", require_nonzero=True)
    func = partial(quadtree_algorithm, nbasis=nbasis, seed=seed)
    quad_basis = xr.apply_ufunc(func, weights)
    return _finalise_generated_basis(quad_basis, start_date=start_date, domain=domain)


def bucket_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    nbasis: int = 100,
    country_directory: str | None = None,
) -> xr.DataArray:
    """Create a legacy weighted bucket basis field from precomputed 2D weights.

    This is a weight-first version of :func:`bucket_basis_function`. It still
    delegates to the existing land/sea-aware weighted algorithm for
    compatibility.

    Args:
        weights: Two-dimensional basis weight field.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        nbasis: Desired number of basis regions.
        country_directory: Optional directory containing land/sea files.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    weights = _sanitize_generated_basis_weights(weights, algorithm="weighted bucket", require_nonzero=True)
    weights = _normalise_weights_by_nonzero_max(weights)
    func = partial(
        weighted_algorithm,
        nregion=nbasis,
        bucket=1,
        domain=domain,
        country_directory=country_directory,
    )
    bucket_basis = xr.apply_ufunc(func, weights)
    return _finalise_generated_basis(bucket_basis, start_date=start_date, domain=domain)


def region_constrained_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation = 100,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
    split_acceptance: Literal["none", "contrast_score"] = "none",
    contrast_contribution: xr.DataArray | None = None,
    contrast_cell_weight: xr.DataArray | None = None,
    min_contrast_delta_eig: float | None = None,
    min_contrast_lambda: float | None = None,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | None = None,
) -> xr.DataArray:
    """Create constrained basis labels from weights and region classes.

    This is the weight-first adapter for the current ``region_constrained``
    basis algorithm. Labels are generated independently within each non-null
    region class.

    Args:
        weights: Two-dimensional basis weight field.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        region_classes: Two-dimensional class field on the same spatial grid as
            ``weights``.
        nbasis: Total number of basis regions, or class-local allocation
            accepted by ``region_constrained_basis``.
        allocation: Automatic allocation mode used when ``nbasis`` is an
            integer. ``"weight"`` allocates by class total weight; ``"area"``
            allocates by mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.
        split_strategy: Class-local partition strategy. When omitted,
            ``region_constrained_basis`` uses its greedy axis-parallel default.
            Supplying a strategy keeps class composition/allocation independent
            from the partition algorithm.
        split_acceptance: Optional split-acceptance criterion. The default
            ``"none"`` preserves existing behavior. ``"contrast_score"`` uses
            a mass-preserving observation-space contrast gate.
        contrast_contribution: Design contribution array for contrast scoring.
        contrast_cell_weight: Optional prior flux or split-mass field used as
            ``mu`` in contrast scoring. When omitted, the unnormalised input
            weights are used as a split-mass proxy.
        min_contrast_delta_eig: Optional minimum ``delta_eig`` threshold.
        min_contrast_lambda: Optional minimum ``lambda`` threshold.
        contrast_tau: Prior standard deviation of the split contrast
            coefficient ``delta = alpha_A - alpha_B``.
        contrast_sigma_design: Optional scalar design standard deviation,
            equivalent to ``S = contrast_sigma_design**2 I``.
        contrast_s_diag: Optional diagonal design covariance entries in the
            same row space as ``contrast_contribution``.

    Returns:
        Basis field with globally unique integer labels that do not cross
        ``region_classes`` values.

    Raises:
        ValueError: If an explicit ``split_strategy`` is combined with
            greedy-specific contrast configuration, or if the core basis
            inputs or allocation are invalid.
    """
    raw_weights = _sanitize_generated_basis_weights(weights, algorithm="region-constrained")
    weights = _normalise_weights_by_max(raw_weights)
    region_classes = region_classes.transpose(*weights.dims)
    region_classes = region_classes.sel({dim: weights.coords[dim] for dim in weights.dims})
    split_strategy = _region_constrained_split_strategy(
        split_strategy=split_strategy,
        split_acceptance=split_acceptance,
        contrast_contribution=contrast_contribution,
        contrast_cell_weight=contrast_cell_weight,
        default_contrast_cell_weight=raw_weights,
        min_contrast_delta_eig=min_contrast_delta_eig,
        min_contrast_lambda=min_contrast_lambda,
        contrast_tau=contrast_tau,
        contrast_sigma_design=contrast_sigma_design,
        contrast_s_diag=contrast_s_diag,
    )

    constrained_basis = region_constrained_basis(
        weights,
        region_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        split_strategy=split_strategy,
    )
    return _finalise_generated_basis(constrained_basis, start_date=start_date, domain=domain)


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

    The number of basis functions is optimised using dual annealing. Probably
    not the best or fastest method as there should only be one minimum, but it
    does not require the Jacobian or Hessian for optimisation.

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
    split_strategy: SplitStrategy | None = None,
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
        split_strategy: Class-local partition strategy. When omitted, the core
            constrained helper uses its greedy axis-parallel default. An
            explicit strategy cannot be combined with the greedy-specific
            contrast options above.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and globally unique integer labels that do not cross
        ``region_classes`` values.

    Raises:
        ValueError: If ``region_classes`` is not supplied or an explicit
            ``split_strategy`` is combined with greedy-specific contrast
            configuration.
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
        split_strategy=split_strategy,
        split_acceptance=split_acceptance,
        contrast_contribution=contrast_contribution,
        contrast_cell_weight=contrast_cell_weight,
        min_contrast_delta_eig=min_contrast_delta_eig,
        min_contrast_lambda=min_contrast_lambda,
        contrast_tau=contrast_tau,
        contrast_sigma_design=contrast_sigma_design,
        contrast_s_diag=contrast_s_diag,
    )


def _region_constrained_split_strategy(
    *,
    split_strategy: SplitStrategy | None,
    split_acceptance: Literal["none", "contrast_score"],
    contrast_contribution: xr.DataArray | None,
    contrast_cell_weight: xr.DataArray | None,
    default_contrast_cell_weight: xr.DataArray,
    min_contrast_delta_eig: float | None,
    min_contrast_lambda: float | None,
    contrast_tau: float | None,
    contrast_sigma_design: float | None,
    contrast_s_diag: xr.DataArray | None,
) -> SplitStrategy | None:
    """Return an optional region-constrained split strategy."""
    contrast_options = (
        contrast_contribution,
        contrast_cell_weight,
        min_contrast_delta_eig,
        min_contrast_lambda,
        contrast_tau,
        contrast_sigma_design,
        contrast_s_diag,
    )
    if split_strategy is not None:
        if split_acceptance != "none" or any(option is not None for option in contrast_options):
            raise ValueError(
                "split_strategy cannot be combined with split_acceptance or contrast-scoring options."
            )
        return split_strategy
    if split_acceptance == "none":
        return None
    if split_acceptance != "contrast_score":
        raise ValueError("split_acceptance must be 'none' or 'contrast_score'.")
    if contrast_contribution is None:
        raise ValueError("contrast_contribution is required when split_acceptance='contrast_score'.")
    return GreedyAxisParallelSplitStrategy(
        split_acceptance=ContrastScoreSplitAcceptance(
            contribution=contrast_contribution,
            cell_weight=(
                contrast_cell_weight if contrast_cell_weight is not None else default_contrast_cell_weight
            ),
            min_contrast_delta_eig=min_contrast_delta_eig,
            min_contrast_lambda=min_contrast_lambda,
            contrast_tau=contrast_tau,
            contrast_sigma_design=contrast_sigma_design,
            contrast_s_diag=contrast_s_diag,
        )
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


# dict to retrieve basis function and description by algorithm name
BasisFunction = namedtuple("BasisFunction", ["description", "algorithm"])
basis_functions = {
    "quadtree": BasisFunction("quadtree algorithm", quadtree_basis_function),
    "weighted": BasisFunction("weighted by data algorithm", bucket_basis_function),
    "region_constrained": BasisFunction(
        "class-constrained basis with a selectable split strategy",
        region_constrained_basis_function,
    ),
}


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
    region_classes: xr.DataArray | None = None,
    region_allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
    split_acceptance: Literal["none", "contrast_score"] = "none",
    contrast_contribution: xr.DataArray | None = None,
    contrast_cell_weight: xr.DataArray | None = None,
    min_contrast_delta_eig: float | None = None,
    min_contrast_lambda: float | None = None,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | None = None,
) -> xr.DataArray:
    """Use fixed InTEM outer regions and fit inner regions with an algorithm.

    The InTEM outer-region file defines known outer labels. The largest region
    value is treated as the inner inversion region; this inner mask is passed to
    ``basis_algorithm`` and then inserted back into the fixed outer map.

    Args:
        fp_all: Legacy merged-data dictionary produced by the data preparation
            path.
        start_date: Start date of the inversion period.
        basis_algorithm: Algorithm used to fit the inner region. Supported
            values are ``"quadtree"``, ``"weighted"``, and
            ``"region_constrained"``.
        domain: Domain across which to calculate basis functions.
        emissions_name: Optional list of OpenGHG flux source names used to
            select emissions from ``fp_all``.
        nbasis: Desired number of inner-region basis labels.
        country_directory: Optional directory containing land/sea files and the
            InTEM outer-region file. When omitted, default package files are
            used.
        abs_flux: If true, use absolute flux values when constructing weights.
        region_classes: Region or country class field used only with
            ``basis_algorithm="region_constrained"``. File loading should
            happen before calling this helper.
        region_allocation: Allocation mode for ``region_constrained``. One of
            ``"weight"`` or ``"area"``.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class when using ``region_constrained``.
        split_strategy: Class-local partition strategy used only with
            ``basis_algorithm="region_constrained"``. When omitted, the core
            constrained helper uses its greedy axis-parallel default.
        split_acceptance: Optional split-acceptance criterion for
            ``region_constrained`` inner-region splitting.
        contrast_contribution: Design contribution array used only when
            ``split_acceptance="contrast_score"``.
        contrast_cell_weight: Optional prior flux or split-mass field used for
            contrast scoring.
        min_contrast_delta_eig: Optional minimum contrast ``delta_eig``.
        min_contrast_lambda: Optional minimum contrast ``lambda``.
        contrast_tau: Prior standard deviation of the split contrast
            coefficient. If omitted, ``tau=1`` is uncalibrated.
        contrast_sigma_design: Optional scalar design standard deviation.
        contrast_s_diag: Optional diagonal design covariance entries.

    Returns:
        Basis field with fixed outer labels and generated inner labels.

    Raises:
        ValueError: If the selected basis algorithm rejects its inputs, or an
            explicit ``split_strategy`` is combined with greedy-specific
            contrast configuration.
    """
    if country_directory is None:
        logger.info(f"Loading default InTEM outer region file for domain {domain}.")
        intem_regions_path = Path(__file__).parent / f"outer_region_definition_{domain}.nc"
    else:
        logger.info(f"Loading InTEM outer region file for domain {domain} from {country_directory}.")
        intem_regions_path = Path(country_directory) / f"outer_region_definition_{domain}.nc"
    intem_regions = xr.open_dataset(intem_regions_path).region

    # force intem_regions to use flux coordinates
    flux, _ = _flux_fp_from_fp_all(fp_all, emissions_name)
    _, intem_regions = xr.align(flux, intem_regions, join="override")

    inner_index = intem_regions.values.max()

    mask = intem_regions == inner_index

    basis_function = basis_functions[basis_algorithm].algorithm
    algorithm_kwargs = {"country_directory": country_directory, "abs_flux": abs_flux, "mask": mask}
    if basis_algorithm == "region_constrained":
        algorithm_kwargs.update(
            {
                "region_classes": region_classes,
                "allocation": region_allocation,
                "min_regions_per_class": min_regions_per_class,
                "split_strategy": split_strategy,
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
    inner_region = basis_function(
        fp_all,
        start_date,
        domain,
        emissions_name,
        nbasis,
        **algorithm_kwargs,
    )

    basis = intem_regions.rename("basis")

    loc_dict = {
        "lat": slice(inner_region.lat.min(), inner_region.lat.max() + 0.1),
        "lon": slice(inner_region.lon.min(), inner_region.lon.max() + 0.1),
    }
    basis.loc[loc_dict] = (inner_region + inner_index - 1).squeeze().values

    basis += 1  # intem_region_definitions.nc regions start at 0, not 1

    basis = basis.expand_dims({"time": [pd.to_datetime(start_date)]})

    return basis
