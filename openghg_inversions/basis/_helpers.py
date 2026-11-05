"""Boundary-condition sensitivity helpers."""

import xarray as xr

from .boundary import BoundaryBasis, nesw_basis


def bc_sensitivity(fp_and_data: dict, bc_basis: BoundaryBasis = nesw_basis) -> dict:
    """Project absolute or centered directional boundary contributions.

    Absolute ``bc_n/e/s/w`` inputs produce ``H_bc``. Centered
    ``bc_anomaly_n/e/s/w`` and ``bc_mass_n/e/s/w`` inputs produce
    ``H_bc_anomaly`` and ``H_bc_mass`` through the same spatial basis; their
    scalar ``bc_reference`` is preserved unchanged. Centered inputs require
    ``boundary_representation="centered-v1"``. See
    :ref:`surface-centered-boundary-contract`.

    Args:
        fp_and_data: Site datasets with all four directional contributions
            for the selected representation. Centered anomaly units must
            match the reference concentration units; mass units must be
            ``"1"``. Dot-prefixed metadata entries are retained unchanged.
        bc_basis: Resolved labelled basis with ``bc_basis_<direction>`` variables
            and a ``region`` dimension. A singleton time applies to every
            observation; multiple times align by their labels. A callable
            builds a basis from each site's dataset. The default assigns each
            whole curtain to its own ``n/e/s/w`` region, without resource I/O.

    Returns:
        New mapping and site datasets, sharing input storage where possible.
        Projected sensitivities have ``bc_region`` and ``time`` dimensions;
        Each sensitivity keeps its directional operand attributes (including
        units) plus the basis ``bc_basis_*`` provenance attributes.
        Centered operands remain spatial, without fitted boundary-period
        expansion. Inputs remain unchanged and lazy arrays remain lazy.

    Raises:
        ValueError: If centered directional operands, representation, reference
            shape, units or basis alignment are invalid.
    """
    result = dict(fp_and_data)
    if not callable(bc_basis):
        bc_basis = _normalise_basis(bc_basis)

    for site, ds in fp_and_data.items():
        if site.startswith("."):
            continue
        site_basis = _normalise_basis(bc_basis(ds)) if callable(bc_basis) else bc_basis
        representation = ds.attrs.get("boundary_representation")
        if representation is not None and representation != "centered-v1":
            raise ValueError(f"Unsupported boundary_representation {representation!r}.")
        centered = any(str(name).startswith(("bc_anomaly_", "bc_mass_")) for name in ds) or (
            ds.attrs.get("boundary_representation") == "centered-v1"
        )
        if centered:
            projected = _project_centered_boundaries(ds, site_basis)
        else:
            sensitivity = _project_boundaries(ds[[f"bc_{d}" for d in "nesw"]], site_basis)
            sensitivity.attrs = dict(ds.bc_n.attrs)
            projected = {"H_bc": sensitivity}
        provenance = {key: value for key, value in site_basis.attrs.items() if key.startswith("bc_basis_")}
        for sensitivity in projected.values():
            sensitivity.attrs.update(provenance)
        result[site] = ds.assign(projected)
    return result


def _normalise_basis(basis: xr.Dataset) -> xr.Dataset:
    """Rename stored weights to ``bc_<d>``; squeeze one time or sort several.

    Applies equally to resolved datasets and to builder results.
    """
    basis = basis.rename({dv: str(dv).replace("basis_", "") for dv in basis.data_vars})
    if basis.sizes.get("time") == 1:
        return basis.squeeze("time")
    if "time" in basis.dims:
        return basis.sortby("time")
    return basis


def _project_boundaries(ds: xr.Dataset, basis: xr.Dataset, *, centered: bool = False) -> xr.DataArray:
    """Contract directional curtains with the same spatial boundary basis.

    A curtain cell contributes only where its weight is nonzero, so a
    non-finite cell never reaches a region it does not belong to. In-memory
    weights are also narrowed to the regions they touch; lazy weights are not
    computed.
    """
    skipna = False if centered else None
    if centered:
        if basis.sizes.get("time", 0) > 1:
            try:
                basis = basis.sel(time=ds.time)
            except KeyError as exc:
                raise ValueError("Centered boundary basis must cover every retained site timestamp.") from exc
        ds, basis = xr.align(ds, basis, join="exact", copy=False)
    else:
        # Absolute projection keeps the shared labels, e.g. retained observation times.
        ds, basis = xr.align(ds, basis, join="inner", copy=False)
    parts = []
    for direction in "nesw":
        weights = basis[f"bc_{direction}"]
        if weights.chunks is None:
            weights = weights.isel(region=(weights != 0).any([d for d in weights.dims if d != "region"]).values)
        curtain = (ds[f"bc_{direction}"] * weights).where(weights != 0, 0)
        spatial = [dim for dim in ("lat", "lon", "height") if dim in curtain.dims]
        parts.append(curtain.sum(spatial, skipna=skipna).reindex(region=basis.region, fill_value=0))
    total = sum(parts[1:], parts[0])
    return total.rename(region="bc_region").transpose("bc_region", ...)


def _project_centered_boundaries(ds: xr.Dataset, basis: xr.Dataset) -> dict[str, xr.DataArray]:
    """Validate the directional centered handoff and project anomaly and mass."""
    prefixes = ("bc_anomaly", "bc_mass")
    required = {f"{prefix}_{d}" for prefix in prefixes for d in "nesw"} | {"bc_reference"}
    if not required <= set(ds.variables) or ds.bc_reference.ndim:
        raise ValueError("Centered site boundaries require all directional A/M operands and scalar bc_reference.")
    if ds.attrs.get("boundary_representation") != "centered-v1":
        raise ValueError("Centered site boundaries require boundary_representation='centered-v1'.")
    projected = {}
    for prefix in prefixes:
        curtains = ds[[f"{prefix}_{d}" for d in "nesw"]].rename(
            {f"{prefix}_{d}": f"bc_{d}" for d in "nesw"}
        )
        expected = "1" if prefix == "bc_mass" else ds.bc_reference.attrs.get("units")
        if not expected or any(value.attrs.get("units") != expected for value in curtains.data_vars.values()):
            raise ValueError("Directional boundary operands have incompatible units.")
        sensitivity = _project_boundaries(curtains, basis, centered=True)
        sensitivity.attrs = dict(ds[f"{prefix}_n"].attrs)
        sensitivity.attrs["boundary_sensitivity_preparation"] = "unexpanded"
        projected[prefix.replace("bc", "H_bc", 1)] = sensitivity
    return projected
