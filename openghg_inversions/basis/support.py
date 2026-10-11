"""Explicit selection and reading of basis-construction support resources.

These maps define basis geometry. Country reporting maps are independent,
caller-supplied inputs.
"""

from pathlib import Path
from typing import Literal

import xarray as xr


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
