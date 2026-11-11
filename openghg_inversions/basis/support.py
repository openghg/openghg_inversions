"""Explicit selection and reading of basis-construction support resources.

These maps define basis geometry. Country reporting maps are independent,
caller-supplied inputs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import xarray as xr

from openghg_inversions.utils import read_netcdfs


def packaged_basis_support_path(kind: Literal["landsea", "outer_regions"], domain: str) -> Path:
    """Select a packaged support file for exactly the requested domain.

    Raises FileNotFoundError when the domain has no packaged resource. Selection
    never substitutes another domain or creates a directory. ``kind`` selects
    the land/sea class map or fixed-outer region map; ``domain`` is matched
    exactly, including case. Returns a path without reading the map.
    """
    directory = Path(__file__).parent
    if kind == "landsea":
        filename = (
            "country-EUROPE-UKMO-landsea-2023.nc" if domain == "EUROPE" else f"country-land-sea_{domain}.nc"
        )
        path = directory / "algorithms" / filename
    elif kind == "outer_regions":
        filename = (
            "intem_region_definition_EUHROB.nc"
            if domain == "EUHROB"
            else f"outer_region_definition_{domain}.nc"
        )
        path = directory / filename
    else:
        raise ValueError(f"Unknown basis support resource {kind!r}.")
    if not path.is_file():
        raise FileNotFoundError(f"No packaged {kind} resource for domain {domain!r}: {path}")
    return path


def load_basis_support_map(path: str | Path, variable: str) -> xr.DataArray:
    """Eagerly read one labelled map and close its exact selected file.

    Args:
        path: Exact NetCDF support-file path, without directory discovery.
        variable: Map variable to select, normally ``"country"`` or ``"region"``.

    Returns:
        In-memory array with coordinates and dataset attributes, overridden by
        variable attributes where both supply the same key. No grid alignment
        or class interpretation is performed here.

    Raises:
        KeyError: If the selected variable is absent.
    """
    with xr.open_dataset(path, decode_coords="all") as dataset:
        result = dataset[variable].load()
        result.attrs = {**dataset.attrs, **result.attrs}
        return result


def resolve_basis_support_maps(
    *,
    domain: str,
    basis_algorithm: str | None,
    fix_outer_regions: bool,
    landsea_path: str | Path | None = None,
    outer_regions_path: str | Path | None = None,
) -> dict[str, xr.DataArray | None]:
    """Resolve selected support maps before acquisition or numerical work.

    Explicit selections are checked even when the chosen algorithm does not
    consume that map. Omitted maps select packaged resources only when needed:
    ``basis_algorithm="weighted"`` needs land/sea classes and
    ``fix_outer_regions=True`` needs outer regions. ``domain`` selects the
    exact packaged domain; it never falls back to a different grid.

    Returns a mapping with ``landsea_classes`` and ``outer_regions`` entries,
    each either an eagerly loaded labelled array or None. Exact selected files
    must contain ``country`` and ``region`` respectively. File or variable
    errors propagate before acquisition; grid alignment belongs to fitting.
    """
    landsea = None
    outer = None
    if landsea_path is not None or basis_algorithm == "weighted":
        path = landsea_path if landsea_path is not None else packaged_basis_support_path("landsea", domain)
        landsea = load_basis_support_map(path, "country")
    if outer_regions_path is not None or fix_outer_regions:
        path = (
            outer_regions_path
            if outer_regions_path is not None
            else packaged_basis_support_path("outer_regions", domain)
        )
        outer = load_basis_support_map(path, "region")
    return {"landsea_classes": landsea, "outer_regions": outer}


def load_region_classes(
    path: str | Path,
    variable: str | None = None,
    *,
    unmapped_values: Sequence[float] = (),
) -> xr.DataArray:
    """Eagerly read a region-class map for constrained basis fitting.

    The map is read like a reporting country file, including its fallbacks for
    known HDF5 backend problems. The last-resort direct HDF5 reader requires
    ``lat``, ``lon`` and ``name`` and exposes the class variable as ``country``,
    so an explicit ``variable="region"`` or a map without ``name`` only reads
    through the xarray backends. Grid alignment and shape checks belong to
    fitting, which also accepts a map on a larger grid containing the flux grid.

    Args:
        path: Exact NetCDF file, without directory discovery.
        variable: Class variable to select. ``None`` selects ``country`` when
            present, otherwise ``region``.
        unmapped_values: Class values to replace with missing values, leaving
            those cells outside every generated region.

    Returns:
        In-memory class map with file and variable attributes; selected
        unmapped values are missing.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        OSError: If every backend fails to read the file.
        KeyError: If the selected variable is absent.
        ValueError: If no mapped cells remain after excluding unmapped values.
    """
    from openghg_inversions._country_file import load_country_dataset

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Region-class map not found: {path}")
    dataset = load_country_dataset(path)
    if variable is None:
        variable = "country" if "country" in dataset or "region" not in dataset else "region"
    if variable not in dataset.data_vars:
        raise KeyError(
            f"Region-class map {path} has no variable {variable!r}; "
            f"available variables: {sorted(map(str, dataset.data_vars))!r}."
        )
    classes = dataset[variable]
    classes.attrs = {**dataset.attrs, **classes.attrs}
    if unmapped_values:
        classes = classes.where(~classes.isin(list(unmapped_values)))
    if not bool(classes.notnull().any()):
        raise ValueError(
            f"Region-class variable {variable!r} in {path} has no mapped cells "
            f"after excluding unmapped values {tuple(unmapped_values)!r}."
        )
    return classes


def load_basis_partition(path: str | Path) -> xr.DataArray:
    """Read an exact selected labelled partition with variable ``basis``.

    This scientific representation is separate from retained operator/flux
    artifacts; historical multi-file discovery belongs to compatibility.
    ``path`` is an exact NetCDF file. The returned array and its coordinates
    are loaded into memory before the file closes, without grid alignment or
    label validation. A missing ``basis`` variable raises KeyError.
    """
    with xr.open_dataset(path, decode_coords="all") as dataset:
        return dataset["basis"].load()


def load_bc_basis(paths: Sequence[str | Path]) -> xr.Dataset:
    """Eagerly load every explicitly selected BC basis file before acquisition.

    Files are sorted by path and concatenated along time, with latitude and
    longitude reindexed to the first file within the existing 1e-5 tolerance.
    The returned dataset is independent of closed files. Missing or unreadable
    files propagate their read error; an empty selection raises ValueError.
    NESW and disabled BCs do not need this loader.
    """
    if not paths:
        raise ValueError("Select at least one boundary-condition basis file.")
    return read_netcdfs(sorted(Path(path) for path in paths), verbose=False)
