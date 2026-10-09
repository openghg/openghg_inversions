"""Load and generate basis functions from files, fluxes, and footprints."""

import getpass
import logging
import os
from collections.abc import Hashable, Mapping, Sequence
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from .algorithms import (
    AllocationMode,
    AxisParallelSplitStep,
    ContrastScoreSplitAcceptance,
    GreedySplitStrategy,
    NbasisAllocation,
    SplitStrategy,
    allocate_nbasis_by_class,
    combine_inner_outer_region_classes,
    normalize_spatial_grid,
    region_constrained_basis,
)
from .algorithms import quadtree_algorithm, weighted_algorithm

from openghg_inversions.config.paths import Paths
from openghg_inversions.utils import read_netcdfs

logger = logging.getLogger(__name__)


openghginv_path = Paths.openghginv
_INNER_REGION_LABEL_ATTR = "inner_region_label"


def basis(domain: str, basis_case: str, basis_directory: str | None = None) -> xr.Dataset:
    """Load matching basis files from ``<directory>/<domain>/<basis_case>_<domain>*.nc``.

    Args:
        domain: Domain subdirectory and filename suffix.
        basis_case: Filename prefix used to select basis files.
        basis_directory: Optional root directory. Defaults to the repository's
            ``basis_functions`` directory.

    Returns:
        Combined dataset of matching basis files.

    Raises:
        ValueError: If the default directory is missing; the directory is
            created before this error is raised.
        FileNotFoundError: If no files match the domain and basis case.
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
    """Load matching boundary-condition basis files for a domain and case.

    Files match ``<directory>/<domain>/<basis_case>_<domain>*.nc``.
    Unreadable matches are reported and skipped.

    Args:
        domain: Domain subdirectory and filename suffix.
        basis_case: Filename prefix used to select boundary-condition basis files.
        bc_basis_directory: Optional root directory. Defaults to the
            repository's ``bc_basis_functions`` directory.

    Returns:
        Combined dataset of readable matching basis files.

    Raises:
        ValueError: If the default directory is missing; the directory is
            created before this error is raised.
        FileNotFoundError: If no readable files match the domain and basis case.
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


def paired_abs_response_weights(
    flux: xr.DataArray,
    footprints: list[xr.DataArray],
    *,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Build weights from retained footprint observations paired with prior flux.

    Each retained footprint time is multiplied by the prior flux at the same
    timestamp, and absolute responses are averaged over retained observations:
    ``mean_o(abs(fp_o * flux_at_o))``. This helper is intentionally pure and
    lower-level; callers are responsible for passing footprints after any
    observation filters have been applied.

    Args:
        flux: Prior flux field with ``time`` plus spatial dimensions.
        footprints: Retained footprint fields for each site.
        mask: Optional Boolean spatial mask. When supplied, weights outside the
            mask are dropped from the returned field.

    Returns:
        Two-dimensional response weight field with spatial coordinates
        preserved.

    Raises:
        ValueError: If no footprints are supplied, required ``time`` dimensions
            are absent, or a retained footprint time cannot be paired with a
            flux time. Also raised when paired arrays do not have exactly two
            matching spatial dimensions and coordinates, or when a mask is not
            Boolean and defined on those spatial dimensions.
    """
    if "time" not in flux.dims:
        raise ValueError("flux must have a time dimension for paired response weights.")
    spatial_dims = tuple(dim for dim in flux.dims if dim != "time")
    if len(spatial_dims) != 2:
        raise ValueError("flux must have exactly two spatial dimensions for paired response weights.")
    if any(dim not in flux.indexes for dim in spatial_dims):
        raise ValueError("flux spatial dimensions must have coordinates for paired response weights.")
    if not footprints:
        raise ValueError("at least one retained footprint is required for paired response weights.")

    response_sums: list[xr.DataArray] = []
    n_measure = 0
    flux_times = flux.get_index("time")

    for footprint in footprints:
        if "time" not in footprint.dims:
            raise ValueError("footprints must have a time dimension for paired response weights.")
        if set(footprint.dims) != set(flux.dims):
            raise ValueError(
                "footprints and flux must have the same time and spatial dimensions "
                "for paired response weights."
            )
        if any(dim not in footprint.indexes for dim in spatial_dims):
            raise ValueError(
                "footprint spatial dimensions must have coordinates for paired response weights."
            )
        footprint = footprint.transpose(*flux.dims)
        missing_times = footprint.get_index("time").difference(flux_times)
        if not missing_times.empty:
            raise ValueError(
                "footprint times must be present in flux time coordinates for paired response weights."
            )

        paired_flux = flux.sel(time=footprint.time)
        try:
            footprint, paired_flux = xr.align(footprint, paired_flux, join="exact")
        except xr.AlignmentError as exc:
            raise ValueError(
                "footprints and flux must share exact time and spatial coordinates for paired response weights."
            ) from exc
        response_sums.append(abs(footprint * paired_flux).sum("time"))
        n_measure += footprint.sizes["time"]

    if n_measure == 0:
        raise ValueError("at least one retained footprint time is required for paired response weights.")

    try:
        response_sums = list(xr.align(*response_sums, join="exact"))
    except xr.AlignmentError as exc:
        raise ValueError(
            "retained footprints must share exact spatial coordinates for paired response weights."
        ) from exc
    weights = cast(xr.DataArray, sum(response_sums)) / n_measure

    if mask is not None:
        if set(mask.dims) != set(spatial_dims):
            raise ValueError("mask must have the same spatial dimensions as paired response weights.")
        if mask.dtype.kind != "b":
            raise ValueError("mask must be Boolean for paired response weights.")
        mask = mask.transpose(*spatial_dims)
        try:
            weights, mask = xr.align(weights, mask, join="exact")
        except xr.AlignmentError as exc:
            raise ValueError(
                "mask must share exact spatial coordinates with paired response weights."
            ) from exc
        return weights.where(mask, drop=True)

    return weights


def basis_weights_from_data(
    site_data: Mapping[str, xr.Dataset],
    flux_data: Mapping[str, xr.Dataset],
    flux_sources: Sequence[str] | None = None,
    *,
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Materialize mean-footprint times mean-flux weights for basis fitting.

    Select only the first requested emissions source, or the first mapping
    entry when omitted. All site footprint times contribute to the mean. This
    named eager algorithm boundary leaves the borrowed datasets unchanged.
    ``abs_flux=True`` takes the absolute flux before temporal averaging.
    ``mask`` optionally drops cells outside the fitting region.
    """
    source = flux_sources[0] if flux_sources is not None else next(iter(flux_data))
    flux = flux_data[source]["flux"]
    footprints = [dataset["fp"] for dataset in site_data.values()]
    return _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()


def _validate_basis_algorithm(basis_algorithm: str) -> None:
    """Reject unsupported generated algorithms before fitting inputs are materialized."""
    if basis_algorithm not in ("quadtree", "weighted", "region_constrained"):
        raise ValueError(
            "Basis algorithm not recognised. Please use 'quadtree', 'weighted', "
            "'region_constrained', or input a basis function file"
        )


def basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    basis_algorithm: str,
    nbasis: int,
    *,
    country_directory: str | None = None,
    landsea_indices: np.ndarray | None = None,
    **region_kwargs: Any,
) -> xr.DataArray:
    """Fit a generated basis with the existing weight-array algorithms."""
    _validate_basis_algorithm(basis_algorithm)
    if basis_algorithm == "quadtree":
        return quadtree_basis_from_weights(weights, start_date, domain, nbasis=nbasis)
    if basis_algorithm == "weighted":
        return bucket_basis_from_weights(
            weights, start_date, domain, nbasis=nbasis,
            country_directory=country_directory, landsea_indices=landsea_indices,
        )
    if region_kwargs.get("region_classes") is None:
        raise ValueError("region_classes must be supplied for the region_constrained basis algorithm.")
    return region_constrained_basis_from_weights(
        weights, start_date, domain, nbasis=nbasis, **region_kwargs,
    )


def load_intem_outer_regions(
    domain: str,
    outer_regions_path: str | Path | None = None,
) -> xr.DataArray:
    """Load a coordinate-preserving InTEM fixed-outer region map.

    Args:
        domain: Domain suffix used to select
            ``outer_region_definition_{domain}.nc`` when
            ``outer_regions_path`` is omitted.
        outer_regions_path: Optional direct path to an outer-region NetCDF
            file. A bare relative filename may name a packaged file in
            :mod:`openghg_inversions.basis`. When omitted, the packaged file
            for ``domain`` is used.

    Returns:
        Loaded two-dimensional ``region`` field, including its spatial
        coordinates and metadata.

    Raises:
        FileNotFoundError: If the selected region file does not exist.
        KeyError: If the selected dataset does not contain ``region``.
    """
    if outer_regions_path is None:
        logger.info(f"Loading default InTEM outer region file for domain {domain}.")
        outer_regions_path = Path(__file__).parent / f"outer_region_definition_{domain}.nc"
    else:
        outer_regions_path = Path(outer_regions_path)
        if not outer_regions_path.is_absolute() and not outer_regions_path.exists():
            packaged_path = Path(__file__).parent / outer_regions_path
            if packaged_path.exists():
                outer_regions_path = packaged_path
        logger.info(f"Loading InTEM outer region file for domain {domain} from {outer_regions_path}.")

    with xr.open_dataset(outer_regions_path, decode_coords="all") as dataset:
        regions = dataset["region"].load()
        for name, value in dataset.attrs.items():
            regions.attrs.setdefault(name, value)
        return regions


def _fixed_outer_inner_region_label(regions: xr.DataArray) -> int:
    """Return the explicitly marked inner label, with legacy max-label fallback."""
    configured_label = regions.attrs.get(_INNER_REGION_LABEL_ATTR)
    if configured_label is None:
        return int(regions.max().item())
    if isinstance(configured_label, bool) or not isinstance(configured_label, Integral):
        raise ValueError(
            f"Fixed outer-region map attribute {_INNER_REGION_LABEL_ATTR!r} must be an integer."
        )
    inner_label = int(configured_label)
    if not bool((regions == inner_label).any().item()):
        raise ValueError(
            f"Fixed outer-region map marks inner label {inner_label}, but that label is absent."
        )
    return inner_label


def load_country_region_classes(
    domain: str,
    country_directory: str | Path | None = None,
) -> xr.DataArray:
    """Load a coordinate-preserving country or land/sea class map.

    The path-selection behavior matches the legacy weighted-basis loader. A
    caller-supplied directory uses ``country-land-sea_{domain}.nc``. Packaged
    EUROPE data use ``country-EUROPE-UKMO-landsea-2023.nc``; other packaged
    domains use ``country-land-sea_{domain}.nc`` and fall back to the EUROPE
    file when that domain file is unavailable. Values are returned unchanged:
    every distinct non-null value can be used as a separate region class, so a
    caller-supplied multi-country integer map is not coerced to binary.

    Args:
        domain: Domain used to select the country/land-sea file.
        country_directory: Optional directory containing the class-map file.

    Returns:
        Loaded two-dimensional ``country`` field with its original values,
        spatial coordinates, and metadata.

    Raises:
        FileNotFoundError: If the selected file does not exist.
        KeyError: If the selected dataset does not contain ``country``.
    """
    default_directory = Path(__file__).parent / "algorithms"
    if country_directory is not None:
        class_map_path = Path(country_directory) / f"country-land-sea_{domain}.nc"
    elif domain == "EUROPE":
        class_map_path = default_directory / "country-EUROPE-UKMO-landsea-2023.nc"
    else:
        class_map_path = default_directory / f"country-land-sea_{domain}.nc"
        if not class_map_path.exists():
            logger.warning(
                f"No land-sea file found for domain {domain}. Defaulting to EUROPE "
                "(country-EUROPE-UKMO-landsea-2023.nc)"
            )
            class_map_path = default_directory / "country-EUROPE-UKMO-landsea-2023.nc"

    with xr.open_dataset(class_map_path, decode_coords="all") as dataset:
        return dataset["country"].load()


def _sanitize_generated_basis_weights(
    weights: xr.DataArray,
    *,
    algorithm: str,
    require_nonzero: bool = False,
) -> xr.DataArray:
    """Materialize weights, replace non-finite cells with zero, and reject empty fields.

    Args:
        weights: Generated-basis spatial weight field.
        algorithm: Name included in validation errors.
        require_nonzero: Also reject a field with no non-zero finite weights.

    Returns:
        Eager weight field with non-finite values replaced by zero.

    Raises:
        ValueError: If no finite values remain, or ``require_nonzero`` is true
            and all finite values are zero.
    """
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
    landsea_indices: np.ndarray | None = None,
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
        landsea_indices: Optional pre-aligned land/sea mask for ``weights``.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    weights = _sanitize_generated_basis_weights(weights, algorithm="weighted bucket", require_nonzero=True)
    weights = _normalise_weights_by_nonzero_max(weights)
    algorithm_kwargs: dict[str, Any] = {
        "nregion": nbasis,
        "bucket": 1,
        "domain": domain,
        "country_directory": country_directory,
    }
    if landsea_indices is not None:
        algorithm_kwargs["landsea_indices"] = landsea_indices
    func = partial(weighted_algorithm, **algorithm_kwargs)
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
            ``weights``. A full-domain field may be supplied when ``weights``
            are a rectangular crop of that grid.
        nbasis: Total number of basis regions, or class-local allocation
            accepted by ``region_constrained_basis``.
        allocation: Automatic allocation mode used when ``nbasis`` is an
            integer. ``"weight"`` allocates by class total weight; ``"area"``
            allocates by mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.
        split_strategy: Optional class-local label generator. When omitted, the
            default greedy generator is used, optionally configured by
            ``split_acceptance``.
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
        ValueError: If both ``split_strategy`` and a non-default
            ``split_acceptance`` are supplied.
    """
    raw_weights = _sanitize_generated_basis_weights(weights, algorithm="region-constrained")
    weights = _normalise_weights_by_max(raw_weights)
    region_classes = _spatial_field_on_weights_grid(
        weights,
        region_classes,
        candidate_name="region_classes",
    )
    if split_strategy is not None and split_acceptance != "none":
        raise ValueError("split_strategy cannot be combined with a non-default split_acceptance.")
    if split_strategy is None:
        split_strategy = _region_constrained_split_strategy(
            split_acceptance=split_acceptance,
            contrast_contribution=contrast_contribution,
            contrast_cell_weight=contrast_cell_weight if contrast_cell_weight is not None else raw_weights,
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


def _spatial_field_on_weights_grid(
    weights: xr.DataArray,
    field: xr.DataArray,
    *,
    candidate_name: str,
) -> xr.DataArray:
    """Subset or reorder a spatial field before strict grid normalization.

    When both inputs have matching one-dimensional dimension coordinates and
    the candidate grid is at least as large, each weights coordinate selects its
    unique nearest numeric candidate coordinate or unique exact nonnumeric match.
    Physical-coordinate tolerances, metadata, and CRS compatibility are then
    validated by :func:`normalize_spatial_grid`.

    Args:
        weights: Two-dimensional weight field defining the target grid.
        field: Two-dimensional candidate field on the target grid or a
            larger grid containing it.
        candidate_name: Semantic field name used in alignment errors.

    Returns:
        Candidate field subset, transposed, and normalized to the weights grid.

    Raises:
        ValueError: If either input is not two-dimensional or their dimension
            names differ.
        xarray.AlignmentError: If the candidate field cannot be normalized to the
            physical weights grid.
    """
    if weights.ndim != 2 or field.ndim != 2 or set(weights.dims) != set(field.dims):
        return normalize_spatial_grid(
            weights,
            field,
            reference_name="weights",
            candidate_name=candidate_name,
        )

    field = field.transpose(*weights.dims)
    indexers: dict[Hashable, np.ndarray] = {}
    for dimension in weights.dims:
        if weights.sizes[dimension] > field.sizes[dimension]:
            break
        if dimension not in weights.coords or dimension not in field.coords:
            break

        weight_coordinate = weights.coords[dimension]
        field_coordinate = field.coords[dimension]
        if weight_coordinate.dims != (dimension,) or field_coordinate.dims != (dimension,):
            break

        weight_values = weight_coordinate.to_numpy()
        field_values = field_coordinate.to_numpy()
        if np.issubdtype(weight_values.dtype, np.number) and np.issubdtype(field_values.dtype, np.number):
            distances = np.abs(
                np.asarray(field_values, dtype=np.float64)[:, np.newaxis]
                - np.asarray(weight_values, dtype=np.float64)[np.newaxis, :]
            )
            nearest = np.argmin(distances, axis=0)
        else:
            nearest_matches: list[int] = []
            for value in weight_values:
                matches = np.flatnonzero(field_values == value)
                if matches.size != 1:
                    break
                nearest_matches.append(int(matches[0]))
            if len(nearest_matches) != weight_values.size:
                break
            nearest = np.asarray(nearest_matches, dtype=np.intp)

        if np.unique(nearest).size != nearest.size:
            break
        indexers[dimension] = nearest
    else:
        field = field.isel(indexers)

    return normalize_spatial_grid(
        weights,
        field,
        reference_name="weights",
        candidate_name=candidate_name,
    )


class _FixedOuterSplitStrategy:
    """Keep outer class maps fixed while dispatching an inner generator."""

    def __init__(
        self,
        inner_mask: np.ndarray,
        inner_strategy: SplitStrategy | None,
    ) -> None:
        """Initialize fixed-outer dispatch.

        Args:
            inner_mask: Boolean mask selecting cells handled by the inner
                splitting strategy.
            inner_strategy: Optional strategy for inner cells. When omitted,
                ``GreedySplitStrategy`` configured with
                ``AxisParallelSplitStep`` is used.
        """
        self.inner_mask = np.asarray(inner_mask, dtype=bool)
        self.inner_strategy = (
            inner_strategy
            if inner_strategy is not None
            else GreedySplitStrategy(split_step=AxisParallelSplitStep())
        )

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return one fixed label for outer IDs or dispatch an inner class.

        Args:
            weights: Non-negative spatial weights passed to the inner strategy.
            class_mask: Boolean mask selecting the current class.
            target_regions: Requested label count for the current class.

        Returns:
            Integer labels with the same shape as ``class_mask``, positive
            inside the selected class and zero outside. Outer classes receive
            one fixed label; inner classes use the configured strategy.

        Raises:
            ValueError: Propagated if the inner strategy rejects the request.
        """
        if np.any(class_mask & ~self.inner_mask):
            labels = np.zeros(class_mask.shape, dtype=np.int64)
            labels[class_mask] = 1
            return labels
        return self.inner_strategy(weights, class_mask, target_regions)


def region_constrained_fixed_outer_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    nbasis: int | np.integer = 100,
    outer_regions: xr.DataArray | None = None,
    region_classes: xr.DataArray | None = None,
    country_directory: str | Path | None = None,
    outer_regions_path: str | Path | None = None,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
) -> xr.DataArray:
    """Create a constrained fixed-outer basis from precomputed weights.

    The largest value in the InTEM outer-region map selects the bounded inner
    domain. ``nbasis`` is distributed only among the inner ``region_classes``;
    each distinct non-null outer-region value outside that maximum-valued inner
    domain receives one fixed basis label. Inner and outer class values are
    tagged before constrained splitting, so their final positive integer labels
    are globally disjoint.

    Args:
        weights: Non-negative, two-dimensional whole-domain basis weight field
            whose grid defines the output coordinates. Non-finite cells are
            replaced by zero, but at least one finite cell is required. Unlike
            :func:`region_constrained_basis_from_weights`, this fixed-outer
            adapter does not accept cropped weights because it emits the outer
            states as well as the bounded inner states.
        start_date: Start date of the inversion period.
        domain: Domain used for output metadata and default file loading.
        nbasis: Total number of requested inner-region basis labels.
        outer_regions: Optional already-loaded InTEM outer-region map. When
            omitted, :func:`load_intem_outer_regions` is used. Supplying this
            field takes precedence over ``outer_regions_path``.
        region_classes: Optional already-loaded inner classification, such as
            a binary land/sea or multi-country integer map. Each distinct
            non-null selected value is a class. When omitted,
            :func:`load_country_region_classes` is used.
        country_directory: Optional directory containing the country or
            land/sea class-map file. Used only when ``region_classes`` is
            omitted.
        outer_regions_path: Optional direct path to an outer-region NetCDF
            file. Used only when ``outer_regions`` is omitted. When both are
            omitted, the packaged file for ``domain`` is used.
        allocation: Mode used to distribute ``nbasis`` across selected inner
            classes.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            selected inner class.
        split_strategy: Optional class-local label generator applied only to
            bounded inner classes after layout composition. Outer IDs remain
            fixed maps even when they are disconnected. The inner default is
            ``GreedySplitStrategy`` configured with
            ``AxisParallelSplitStep``.

    Returns:
        Basis field on the weights grid with a singleton ``time`` dimension,
        standard generated-basis metadata, one target per non-null outer class,
        outside the maximum-valued inner domain, and ``nbasis`` targets
        allocated across the inner classes. Null outer cells remain label
        ``0``.

    Raises:
        TypeError: If ``nbasis`` is not a non-Boolean integral value.
        ValueError: If weights, dimensions, dimension names, outer-map values,
            inner classes, allocation, or split-strategy labels are invalid.
        FileNotFoundError: If a required default or caller-selected map file is
            missing.
        KeyError: If a selected outer map lacks ``region`` or a selected class
            map lacks ``country``.
        xarray.AlignmentError: If the supplied or loaded fields are not on
            physically compatible spatial grids.
    """
    if isinstance(nbasis, (bool, np.bool_)) or not isinstance(nbasis, (Integral, np.integer)):
        raise TypeError("nbasis must be an integer inner-region target.")
    nbasis = int(nbasis)

    loaded_outer_regions = (
        load_intem_outer_regions(domain, outer_regions_path) if outer_regions is None else outer_regions
    )
    loaded_region_classes = (
        load_country_region_classes(domain, country_directory) if region_classes is None else region_classes
    )

    loaded_outer_regions = _spatial_field_on_weights_grid(
        weights,
        loaded_outer_regions,
        candidate_name="outer_regions",
    )
    loaded_region_classes = _spatial_field_on_weights_grid(
        weights,
        loaded_region_classes,
        candidate_name="region_classes",
    )

    try:
        inner_region_value = loaded_outer_regions.max(skipna=True).compute().item()
    except (TypeError, ValueError) as exc:
        raise ValueError("outer_regions must contain a finite maximum region value.") from exc
    if not isinstance(inner_region_value, (int, float, np.integer, np.floating)) or not np.isfinite(
        inner_region_value
    ):
        raise ValueError("outer_regions must contain a finite maximum region value.")

    inner_mask = loaded_outer_regions == inner_region_value
    inner_classes = loaded_region_classes.where(inner_mask)
    raw_weights = _sanitize_generated_basis_weights(weights, algorithm="fixed-outer region-constrained")
    raw_inner_targets = allocate_nbasis_by_class(
        raw_weights,
        inner_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
    )
    if not raw_inner_targets:
        raise ValueError("The maximum-valued outer region contains no mapped inner classes.")

    targets: dict[Hashable, int] = {}
    outer_values = loaded_outer_regions.where(~inner_mask).to_numpy().ravel()
    for value in pd.unique(outer_values):
        if bool(pd.isna(value)):
            continue
        targets[("outer", cast(Hashable, value))] = 1
    targets.update({("inner", class_value): target for class_value, target in raw_inner_targets.items()})

    composed_classes = combine_inner_outer_region_classes(
        inner_mask,
        loaded_region_classes,
        loaded_outer_regions,
    )
    fixed_outer_strategy = _FixedOuterSplitStrategy(inner_mask.to_numpy(), split_strategy)
    return region_constrained_basis_from_weights(
        raw_weights,
        start_date,
        domain,
        region_classes=composed_classes,
        nbasis=targets,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        split_strategy=fixed_outer_strategy,
    )


def _region_constrained_split_strategy(
    *,
    split_acceptance: Literal["none", "contrast_score"],
    contrast_contribution: xr.DataArray | None,
    contrast_cell_weight: xr.DataArray,
    min_contrast_delta_eig: float | None,
    min_contrast_lambda: float | None,
    contrast_tau: float | None,
    contrast_sigma_design: float | None,
    contrast_s_diag: xr.DataArray | None,
):
    """Build the greedy split strategy with an optional contrast acceptance gate.

    Args:
        split_acceptance: ``"none"`` or ``"contrast_score"``.
        contrast_contribution: Design contribution array required for contrast
            scoring.
        contrast_cell_weight: Spatial weight field used as contrast split mass.
        min_contrast_delta_eig: Optional minimum ``delta_eig`` score.
        min_contrast_lambda: Optional minimum ``lambda`` score.
        contrast_tau: Optional prior standard deviation of the split contrast.
        contrast_sigma_design: Optional scalar design standard deviation.
        contrast_s_diag: Optional diagonal design covariance entries.

    Returns:
        Greedy split strategy configured for the requested acceptance mode.

    Raises:
        ValueError: If the acceptance mode is unknown or contrast scoring lacks
            a design contribution array.
    """
    if split_acceptance == "none":
        return GreedySplitStrategy(split_step=AxisParallelSplitStep())
    if split_acceptance != "contrast_score":
        raise ValueError("split_acceptance must be 'none' or 'contrast_score'.")
    if contrast_contribution is None:
        raise ValueError("contrast_contribution is required when split_acceptance='contrast_score'.")
    return GreedySplitStrategy(
        split_step=AxisParallelSplitStep(),
        split_acceptance=ContrastScoreSplitAcceptance(
            contribution=contrast_contribution,
            cell_weight=contrast_cell_weight,
            min_contrast_delta_eig=min_contrast_delta_eig,
            min_contrast_lambda=min_contrast_lambda,
            contrast_tau=contrast_tau,
            contrast_sigma_design=contrast_sigma_design,
            contrast_s_diag=contrast_s_diag,
        ),
    )


def fixed_outer_regions_basis_from_data(
    site_data: Mapping[str, xr.Dataset],
    flux_data: Mapping[str, xr.Dataset],
    start_date: str,
    basis_algorithm: str,
    domain: str,
    flux_sources: Sequence[str] | None = None,
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
    """Use fixed InTEM outer regions and fit inner regions with an algorithm.

    The InTEM outer-region file defines known outer labels. Its optional
    ``inner_region_label`` attribute identifies the inner inversion region;
    legacy files without that metadata continue to use their largest label.
    This inner mask is passed to ``basis_algorithm`` and then inserted back
    into the fixed outer map.

    For nested outer-domain preparation, ``allow_empty_inner_region=True``
    keeps the inner label unsplit when its masked weights contain finite values
    but all are zero. Otherwise the selected basis algorithm handles the inner
    weights normally.

    Args:
        site_data: Borrowed site datasets containing footprints.
        flux_data: Borrowed flux datasets keyed by runtime source label.
        start_date: Start date of the inversion period.
        basis_algorithm: Algorithm used to fit the inner region. Supported
            values are ``"quadtree"``, ``"weighted"``, and
            ``"region_constrained"``.
        domain: Domain across which to calculate basis functions.
        flux_sources: Optional list of OpenGHG flux source names used to
            select the first source for weighting.
        nbasis: Desired number of inner-region basis labels.
        country_directory: Optional directory containing land/sea files and the
            InTEM outer-region file. When omitted, default package files are
            used.
        abs_flux: If true, use absolute flux values when constructing weights.
        outer_regions_path: Optional direct path to the fixed outer-region
            NetCDF file. When omitted, the packaged
            ``outer_region_definition_<domain>.nc`` file is used.
        region_classes: Region or country class field used only with
            ``basis_algorithm="region_constrained"``. File loading should
            happen before calling this helper.
        region_allocation: Allocation mode for ``region_constrained``. One of
            ``"weight"`` or ``"area"``.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class when using ``region_constrained``.
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
        allow_empty_inner_region: If true, keep the inner label unsplit when
            its masked weights contain finite values but all are zero. Intended
            for nested outer-domain preparation.

    Returns:
        Basis field with fixed outer labels and generated inner labels. When
        ``allow_empty_inner_region`` is true and finite inner weights are all
        zero, its label is kept unsplit.
    """
    _validate_basis_algorithm(basis_algorithm)
    if outer_regions_path is not None:
        selected_outer_regions_path = Path(outer_regions_path)
        if not selected_outer_regions_path.is_absolute() and country_directory is not None:
            country_path = Path(country_directory) / selected_outer_regions_path
            if country_path.exists():
                selected_outer_regions_path = country_path
        intem_regions = load_intem_outer_regions(domain, selected_outer_regions_path)
    elif country_directory is None:
        intem_regions = load_intem_outer_regions(domain)
    else:
        intem_regions = load_intem_outer_regions(
            domain,
            Path(country_directory) / f"outer_region_definition_{domain}.nc",
        )

    # Validate the physical grid before adopting the authoritative flux coordinates.
    source = flux_sources[0] if flux_sources is not None else next(iter(flux_data))
    flux = flux_data[source]["flux"]
    flux_grid = flux.isel(
        {dimension: 0 for dimension in flux.dims if dimension not in intem_regions.dims},
        drop=True,
    )
    intem_regions = normalize_spatial_grid(
        flux_grid,
        intem_regions,
        reference_name="flux",
        candidate_name="fixed outer-region map",
    )

    inner_index = _fixed_outer_inner_region_label(intem_regions)

    mask = intem_regions == inner_index

    weights = basis_weights_from_data(
        site_data, flux_data, flux_sources, abs_flux=abs_flux, mask=mask,
    )
    if allow_empty_inner_region:
        finite_inner_weights = weights.to_numpy()
        finite_inner_weights = finite_inner_weights[np.isfinite(finite_inner_weights)]
        if finite_inner_weights.size and not bool((finite_inner_weights != 0.0).any()):
            logger.warning(
                f"Fixed outer-region map's inner label {inner_index} has no non-zero footprint*flux "
                "response; keeping it as a single fixed region instead of subdividing it further "
                "(allow_empty_inner_region=True)."
            )
            basis = intem_regions.copy().rename("basis")
            basis += 1  # intem_region_definitions.nc regions start at 0, not 1
            basis = basis.expand_dims({"time": [pd.to_datetime(start_date)]})
            return basis

    algorithm_kwargs: dict[str, Any] = {"country_directory": country_directory}
    if basis_algorithm == "weighted":
        landsea_classes = load_country_region_classes(domain, country_directory=country_directory)
        landsea_classes = normalize_spatial_grid(
            intem_regions,
            landsea_classes,
            reference_name="fixed outer-region map",
            candidate_name="land/sea class map",
        )
        inner_landsea = landsea_classes.where(mask, drop=True)
        algorithm_kwargs["landsea_indices"] = inner_landsea.to_numpy()
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
    inner_region = basis_from_weights(
        weights, start_date, domain, basis_algorithm, nbasis, **algorithm_kwargs,
    )

    basis = intem_regions.copy().rename("basis")

    fixed_outer_values = intem_regions.where(~mask).to_numpy()
    finite_outer_values = fixed_outer_values[np.isfinite(fixed_outer_values)]
    if finite_outer_values.size == 0:
        raise ValueError("Fixed outer-region map must retain at least one label outside the inner region.")
    max_fixed_outer_label = int(finite_outer_values.max())

    loc_dict = {
        "lat": slice(inner_region.lat.min(), inner_region.lat.max() + 0.1),
        "lon": slice(inner_region.lon.min(), inner_region.lon.max() + 0.1),
    }
    # Generated algorithms use positive labels starting at one. Place those
    # labels above every retained fixed-outer label before converting the
    # complete field from zero-based to one-based labels below. Offsetting
    # from ``inner_index`` is only safe for legacy maps whose inner label is
    # already the maximum; EUHROB deliberately marks a non-maximum label.
    basis.loc[loc_dict] = (inner_region + max_fixed_outer_label).squeeze().values

    basis += 1  # intem_region_definitions.nc regions start at 0, not 1

    basis = basis.expand_dims({"time": [pd.to_datetime(start_date)]})

    return basis
