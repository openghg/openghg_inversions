"""Functions to create basis datasets from fluxes and footprints."""

import os
import warnings

import getpass
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import xarray as xr
import logging

from .algorithms import AllocationMode, region_constrained_basis
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
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()

    # use xr.apply_ufunc to keep xarray coords
    func = partial(quadtree_algorithm, nbasis=nbasis, seed=seed)
    quad_basis = xr.apply_ufunc(func, fps)

    quad_basis = quad_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    quad_basis = quad_basis.rename("basis")  # this will be used in merges

    quad_basis.attrs["creator"] = getpass.getuser()
    quad_basis.attrs["date created"] = str(pd.Timestamp.today())
    quad_basis.attrs["domain"] = domain

    return quad_basis


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
    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    fps = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()
    fps = fps / fps.max()

    # use xr.apply_ufunc to keep xarray coords
    func = partial(
        weighted_algorithm, nregion=nbasis, bucket=1, domain=domain, country_directory=country_directory
    )
    bucket_basis = xr.apply_ufunc(func, fps)

    bucket_basis = bucket_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    bucket_basis = bucket_basis.rename("basis")  # this will be used in merges

    bucket_basis.attrs["creator"] = getpass.getuser()
    bucket_basis.attrs["date created"] = str(pd.Timestamp.today())
    bucket_basis.attrs["domain"] = domain

    return bucket_basis


def region_constrained_basis_function(
    fp_all: dict,
    start_date: str,
    domain: str,
    emissions_name: list[str] | None = None,
    nbasis: int = 100,
    country_directory: str | None = None,
    abs_flux: bool = False,
    mask: xr.DataArray | None = None,
    region_classes: xr.DataArray | None = None,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
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

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and globally unique integer labels that do not cross
        ``region_classes`` values.

    Raises:
        ValueError: If ``region_classes`` is not supplied.
    """
    if region_classes is None:
        raise ValueError("region_classes must be supplied for the region_constrained basis algorithm.")

    flux, footprints = _flux_fp_from_fp_all(fp_all, emissions_name)
    weights = _mean_fp_times_mean_flux(flux, footprints, abs_flux=abs_flux, mask=mask).as_numpy()
    max_weight = float(weights.max())
    if max_weight > 0:
        weights = weights / max_weight

    region_classes = region_classes.transpose(*weights.dims)
    region_classes = region_classes.sel({dim: weights.coords[dim] for dim in weights.dims})

    constrained_basis = region_constrained_basis(
        weights,
        region_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
    )

    constrained_basis = constrained_basis.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    constrained_basis = constrained_basis.rename("basis")

    constrained_basis.attrs["creator"] = getpass.getuser()
    constrained_basis.attrs["date created"] = str(pd.Timestamp.today())
    constrained_basis.attrs["domain"] = domain

    return constrained_basis


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
        "region-constrained weighted by data algorithm",
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

    Returns:
        Basis field with fixed outer labels and generated inner labels.
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
