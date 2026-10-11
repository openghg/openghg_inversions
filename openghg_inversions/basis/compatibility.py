"""Deprecated basis discovery, artifact adaptation, and filename conventions."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import xarray as xr

from openghg_inversions._flux_options import UNSET, _Unset, resolve_deprecated_keyword

from openghg_inversions.utils import read_netcdfs
from .support import packaged_basis_support_path, load_basis_support_map

from .basis_functions import (
    BASIS_ARTIFACT_PATH_ATTR,
    BASIS_ARTIFACT_SOURCE_ATTR,
    BasisFunctions,
)
from ._functions import (
    _validate_basis_algorithm,
    openghginv_path,
)

_VALID_BASIS_OUTPUT_FORMATS = ("legacy", "datatree")


def make_basis_functions(
    *,
    site_data: Mapping[str, xr.Dataset],
    flux_data: Mapping[str, xr.Dataset],
    split_by_sectors: bool = False,
    species: str,
    domain: str,
    start_date: str,
    flux_sources: Sequence[str] | None | _Unset = UNSET,
    emissions_name: Sequence[str] | None | _Unset = UNSET,
    nbasis: int,
    basis_algorithm: str | None = None,
    fix_outer_regions: bool = False,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    country_directory: str | None = None,
    outer_regions_path: str | Path | None = None,
    outputname: str | None = None,
    output_path: str | None = None,
    basis_output_format: Literal["legacy", "datatree"] = "datatree",
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
    """Translate historical basis keywords once, then call modern preparation.

    Calling this adapter emits DeprecationWarning. Directory/name output
    conventions are deprecated in 0.9; legacy flat output remains explicitly
    available, with no automatic removal date. Prefer
    :func:`~openghg_inversions.basis.prepare_basis` for the shared generation
    arguments and scientific-map contracts.

    ``fp_basis_case`` bypasses generation and discovers matching saved files
    below ``basis_directory/domain`` (default: repository ``basis_functions``).
    Otherwise ``basis_algorithm`` is required, and ``country_directory`` selects
    historical support filenames before numerical work. ``outer_regions_path``
    follows the historical relative-path convention. Inputs remain borrowed;
    selected support maps are loaded eagerly.

    ``emissions_name`` aliases ``flux_sources`` and cannot be supplied with it.
    ``output_path`` is a historical root directory, with a domain subdirectory
    and filename derived from ``species``, algorithm, month and ``outputname``.
    It is used only for generated bases. ``basis_output_format="datatree"``
    writes a retained operator/flux artifact; ``"legacy"`` writes its flat
    partition. ``output_path=None`` disables saving. Returns the operator bound to runtime flux
    with source and, where applicable, artifact-path metadata.
    """
    from ._wrapper import prepare_basis
    from .support import resolve_basis_support_maps
    import warnings

    warnings.warn(
        "make_basis_functions is deprecated; use prepare_basis with exact paths and resolved maps.",
        DeprecationWarning,
        stacklevel=2,
    )
    flux_sources = cast(
        Sequence[str] | None,
        resolve_deprecated_keyword("emissions_name", "flux_sources", emissions_name, flux_sources),
    )
    if basis_output_format not in _VALID_BASIS_OUTPUT_FORMATS:
        raise ValueError(f"Unknown basis_output_format {basis_output_format!r}.")
    if fp_basis_case is None and basis_algorithm is not None:
        _validate_basis_algorithm(basis_algorithm)
    if fp_basis_case is not None:
        result = load_basis_functions(
            flux_data=flux_data,
            split_by_sectors=split_by_sectors,
            domain=domain,
            basis_case=fp_basis_case,
            basis_directory=basis_directory,
        )
    else:
        maps = resolve_basis_support_maps(
            domain=domain,
            basis_algorithm=basis_algorithm,
            fix_outer_regions=fix_outer_regions,
            landsea_path=(Path(country_directory) / f"country-land-sea_{domain}.nc")
            if country_directory is not None and basis_algorithm == "weighted"
            else None,
            outer_regions_path=resolve_legacy_outer_path(
                domain, outer_regions_path, country_directory if fix_outer_regions else None
            ),
        )
        result = prepare_basis(
            site_data=site_data,
            flux_data=flux_data,
            split_by_sectors=split_by_sectors,
            domain=domain,
            start_date=start_date,
            flux_sources=flux_sources,
            nbasis=nbasis,
            basis_algorithm=basis_algorithm,
            fix_outer_regions=fix_outer_regions,
            landsea_classes=maps["landsea_classes"],
            outer_regions=maps["outer_regions"],
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
    if output_path is not None and fp_basis_case is None:
        flat = result.flat_basis()
        if not isinstance(flat, xr.DataArray):
            raise TypeError("Historical output requires one flat partition.")
        assert basis_algorithm is not None
        flat = flat.expand_dims(time=[start_date])
        if basis_output_format == "legacy":
            path = _save_basis(flat, basis_algorithm, output_path, domain, species, outputname)
        else:
            path = _save_basis_datatree(
                result, flat, basis_algorithm, output_path, domain, species, outputname
            )
        result = result.with_metadata({BASIS_ARTIFACT_PATH_ATTR: str(path)})
    return result


def load_basis_functions(
    *,
    flux_data: Mapping[str, xr.Dataset],
    split_by_sectors: bool = False,
    domain: str,
    basis_case: str,
    basis_directory: str | Path | None = None,
) -> BasisFunctions:
    """Load a saved basis artifact as retained ``BasisFunctions``.

    DataTree artifacts are preferred when the matching file carries the
    ``openghg_inversions.flux_weighted_basis`` schema and are loaded through
    ``BasisFunctions.load``. Otherwise the existing legacy flat artifact loader
    is used and a retained basis object is built from runtime flux in
    ``flux_data``.

    Args:
        flux_data: Borrowed runtime flux datasets used for flat artifacts and
            to replace serialized DataTree flux.
        split_by_sectors: Retain all runtime sources separately in mapping order.
        domain: Inversion domain used in the artifact path convention.
        basis_case: Basis case prefix used in the artifact path convention.
        basis_directory: Optional root directory containing per-domain basis
            artifact subdirectories.

    Returns:
        Loaded retained basis object with ``basis_artifact_source`` metadata.

    Raises:
        FileNotFoundError: If no matching artifact files are found.
        ValueError: If more than one matching DataTree artifact is found.
    """
    from ._wrapper import prepare_basis

    selected = resolve_legacy_basis_inputs(
        domain=domain,
        fp_basis_case=basis_case,
        basis_directory=basis_directory,
        country_directory=None,
        outer_regions_path=None,
    )
    result = prepare_basis(
        site_data={},
        flux_data=flux_data,
        split_by_sectors=split_by_sectors,
        domain=domain,
        start_date="",
        nbasis=0,
        **selected,
    )
    if "basis_partition" in selected:
        files = _basis_artifact_files(domain=domain, basis_case=basis_case, basis_directory=basis_directory)
        result = result.with_metadata(
            {
                BASIS_ARTIFACT_SOURCE_ATTR: "legacy_flat",
                BASIS_ARTIFACT_PATH_ATTR: _basis_artifact_path_metadata(files),
            }
        )
    return result


def _basis_artifact_files(
    *,
    domain: str,
    basis_case: str,
    basis_directory: str | Path | None = None,
) -> list[Path]:
    """Find basis artifact files using the legacy basis directory convention."""
    basis_path = Path(basis_directory) if basis_directory is not None else openghginv_path / "basis_functions"
    files = sorted((basis_path / domain).glob(f"{basis_case}_{domain}*.nc"))
    if not files:
        raise FileNotFoundError(
            f"Can't find basis function files for domain '{domain}' and basis_case '{basis_case}' "
        )
    return files


def _basis_artifact_path_metadata(files: list[Path]) -> str:
    """Return deterministic artifact path metadata for one or more matched files."""
    return ";".join(str(file) for file in files)


def _is_basis_datatree_artifact(path: Path) -> bool:
    """Return true when a file contains the BasisFunctions DataTree schema."""
    try:
        with xr.open_datatree(path) as dt:
            return dt.attrs.get("schema") == "openghg_inversions.flux_weighted_basis"
    except (OSError, ValueError, KeyError):
        return False


def _save_basis(
    basis: xr.DataArray,
    basis_algorithm: str,
    output_dir: str,
    domain: str,
    species: str,
    output_name: str | None = None,
) -> Path:
    """Save basis functions to netCDF.

    Args:
      basis (xarray.DataArray):
        basis dataset to save
      basis_algorithm (str):
        name of basis algorithm (e.g. "quadtree", "weighted", or "region_constrained")
      output_dir (str):
        root directory to save basis functions
      domain (str):
        domain of inversion; basis is saved in a "domain" directory inside `output_dir`
      species (str):
        species of inversion
      output_name (str,optional):
        File output name
        Default None

    Returns:
        Path to the written basis artifact.
    """
    basis_out_path = Path(output_dir, domain.upper())

    if not basis_out_path.exists():
        basis_out_path.mkdir(parents=True)

    start_date = str(basis.time.min().values)[:7]  # year and month

    if output_name is None:
        output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}.nc"
    else:
        output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}.nc"

    output_path = basis_out_path / output_name
    basis.to_netcdf(output_path, mode="w")
    return output_path


def _save_basis_datatree(
    basis_functions: BasisFunctions,
    basis: xr.DataArray,
    basis_algorithm: str,
    output_dir: str,
    domain: str,
    species: str,
    output_name: str | None = None,
) -> Path:
    """Save ``BasisFunctions`` using the wrapper's DataTree file convention.

    This historical filename convention wraps ``BasisFunctions.save``.
    Modern retained artifacts are the default; the flat writer is opt-in.
    """
    basis_out_path = Path(output_dir, domain.upper())

    if not basis_out_path.exists():
        basis_out_path.mkdir(parents=True)

    start_date = str(basis.time.min().values)[:7]  # year and month

    if output_name is None:
        output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}_basis_datatree.nc"
    else:
        output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}_basis_datatree.nc"

    output_path = basis_out_path / output_name
    basis_functions.save(output_path)
    return output_path


def resolve_legacy_basis_inputs(
    *,
    domain: str,
    fp_basis_case: str | None,
    basis_directory: str | Path | None,
    country_directory: str | Path | None,
    outer_regions_path: str | Path | None,
) -> dict[str, Any]:
    """Translate historical discovery choices into exact preparation inputs.

    ``fp_basis_case`` discovers files below ``basis_directory/domain``; None
    selects no saved basis. A single retained artifact is preferred over flat
    files; otherwise matching flat files are combined into a labelled partition.
    An omitted directory uses the repository's ``basis_functions`` root.
    ``country_directory`` supplies historical land/sea and outer-map filenames;
    ``outer_regions_path`` can override the outer-map selection.

    Returns selected ``basis_path`` or ``basis_partition`` and optional
    ``landsea_path``/``outer_regions_path`` entries. Flat partitions are read
    here; support maps and retained artifacts are read by the calling boundary.
    Emits DeprecationWarning on every call. Missing artifacts raise
    FileNotFoundError; multiple retained matches raise ValueError.
    """
    import warnings

    warnings.warn(
        "Basis case/directory options are deprecated; select exact artifact and support-map paths.",
        DeprecationWarning,
        stacklevel=2,
    )
    selected: dict[str, Any] = {}
    if fp_basis_case is not None:
        files = _basis_artifact_files(
            domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory
        )
        retained = [file for file in files if _is_basis_datatree_artifact(file)]
        if retained:
            if len(retained) != 1:
                raise ValueError(
                    "Historical basis case matched multiple retained artifacts: "
                    + ", ".join(str(path) for path in retained)
                    + ". Use a more specific basis_case or an exact basis_path."
                )
            selected["basis_path"] = retained[0]
        else:
            selected["basis_partition"] = basis(
                domain, fp_basis_case, str(basis_directory) if basis_directory else None
            ).basis
    resolved_outer_path = resolve_legacy_outer_path(domain, outer_regions_path, country_directory)
    if resolved_outer_path is not None:
        selected["outer_regions_path"] = resolved_outer_path
    if country_directory is not None:
        selected["landsea_path"] = Path(country_directory) / f"country-land-sea_{domain}.nc"
    return selected


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
        ValueError: If the historical default directory is missing.
        FileNotFoundError: If no files match the domain and basis case.
    """
    if basis_directory is None:
        basis_path = openghginv_path / "basis_functions"
        if not basis_path.exists():
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


def load_country_region_classes(domain: str, country_directory: str | Path | None = None) -> xr.DataArray:
    """Compatibility selector for historical country/land-sea directories.

    Every class value is preserved. A missing domain never selects another
    domain. Modern callers use an exact support path and resolved map arrays.
    ``country_directory`` selects ``country-land-sea_<domain>.nc``; None selects
    the exact domain's packaged map. Returns an eager ``country`` array with
    file/variable attributes, without grid alignment. File and variable errors
    propagate to the caller.
    """
    path = (
        Path(country_directory) / f"country-land-sea_{domain}.nc"
        if country_directory is not None
        else packaged_basis_support_path("landsea", domain)
    )
    return load_basis_support_map(path, "country")


def load_intem_outer_regions(domain: str, outer_regions_path: str | Path | None = None) -> xr.DataArray:
    """Eagerly read ``region`` from a historical fixed-outer map selection.

    None selects the exact domain's packaged map. Explicit paths are used as
    given; an absent relative path can resolve beside this module for historical
    packaged filenames. Returns labelled geometry without grid alignment, and
    preserves the ``inner_region_label`` attribute when present. File and
    variable errors propagate. Modern callers use an exact support-file path.
    """
    if outer_regions_path is None:
        path = packaged_basis_support_path("outer_regions", domain)
    else:
        path = Path(outer_regions_path)
        if not path.is_absolute() and not path.exists() and (Path(__file__).parent / path).exists():
            path = Path(__file__).parent / path
    return load_basis_support_map(path, "region")


def load_landsea_indices(domain: str, country_directory: str | None = None):
    """Return the historical country map as an eager unlabelled NumPy array.

    Selection follows :func:`load_country_region_classes`; every class value
    and the file's axis order are preserved. The caller must align the array to
    its weight grid before passing it to a numerical kernel.
    """
    return load_country_region_classes(domain, country_directory).to_numpy()


def resolve_legacy_outer_path(
    domain: str,
    outer_regions_path: str | Path | None,
    country_directory: str | Path | None,
) -> Path | None:
    """Resolve historical relative outer-map selections at the old boundary.

    Directory-relative selections take precedence when present. Bare packaged
    names remain supported here; canonical support loaders use exact paths.
    """
    if outer_regions_path is None:
        return (
            Path(country_directory) / f"outer_region_definition_{domain}.nc"
            if country_directory is not None
            else None
        )
    path = Path(outer_regions_path)
    if not path.is_absolute():
        if country_directory is not None and (Path(country_directory) / path).exists():
            return Path(country_directory) / path
        packaged = Path(__file__).parent / path
        if not path.exists() and packaged.exists():
            return packaged
    return path
