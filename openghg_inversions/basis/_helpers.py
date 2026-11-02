"""Boundary-condition sensitivity helpers."""

import xarray as xr


def bc_sensitivity(fp_and_data: dict, bc_basis: xr.Dataset | None = None) -> dict:
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
            observation; multiple times align by their labels. None uses the
            acquired NESW contributions directly, without resource I/O.

    Returns:
        New mapping and site datasets, sharing input storage where possible.
        Projected sensitivities have ``bc_region`` and ``time`` dimensions;
        centered anomaly retains concentration units and mass retains ``"1"``.
        Centered operands remain spatial, without fitted boundary-period
        expansion. Inputs remain unchanged and lazy arrays remain lazy.

    Raises:
        ValueError: If centered directional operands, representation, reference
            shape, units or basis alignment are invalid.
    """
    result = dict(fp_and_data)
    if bc_basis is not None:
        if bc_basis.sizes.get("time") == 1:
            bc_basis = bc_basis.squeeze("time")
        elif "time" in bc_basis.dims:
            bc_basis = bc_basis.sortby("time")
        bc_basis = bc_basis.rename({dv: str(dv).replace("basis_", "") for dv in bc_basis.data_vars})

    for site, ds in fp_and_data.items():
        if site.startswith("."):
            continue
        representation = ds.attrs.get("boundary_representation")
        if representation is not None and representation != "centered-v1":
            raise ValueError(f"Unsupported boundary_representation {representation!r}.")
        centered = any(str(name).startswith(("bc_anomaly_", "bc_mass_")) for name in ds) or (
            ds.attrs.get("boundary_representation") == "centered-v1"
        )
        if centered:
            projected = _project_centered_boundaries(ds, bc_basis)
        else:
            projected = {"H_bc": _project_boundaries(ds[[f"bc_{d}" for d in "nesw"]], bc_basis)}
        result[site] = ds.assign(projected)
    return result


def _project_boundaries(
    ds: xr.Dataset, basis: xr.Dataset | None, *, centered: bool = False,
) -> xr.DataArray:
    """Contract directional curtains with the same spatial boundary basis."""
    if basis is None:
        return ds.rename({f"bc_{d}": d for d in "nesw"}).sum(
            ["lat", "lon", "height"], skipna=False if centered else None,
        ).to_dataarray(dim="bc_region")
    if centered:
        if basis.sizes.get("time", 0) > 1:
            try:
                basis = basis.sel(time=ds.time)
            except KeyError as exc:
                raise ValueError("Centered boundary basis must cover every retained site timestamp.") from exc
        ds, basis = xr.align(ds, basis, join="exact", copy=False)
    return (
        (ds * basis).sum(["lat", "lon", "height"], skipna=False if centered else None)
        .to_dataarray(dim="__newdim__").sum("__newdim__", skipna=False if centered else None)
        .rename(region="bc_region")
    )


def _project_centered_boundaries(ds: xr.Dataset, basis: xr.Dataset | None) -> dict[str, xr.DataArray]:
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
