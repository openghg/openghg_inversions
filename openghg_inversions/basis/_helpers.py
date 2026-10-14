"""Boundary-condition sensitivity helpers."""

import xarray as xr


def bc_sensitivity(fp_and_data: dict, bc_basis: xr.Dataset | None = None) -> dict:
    """Return site datasets with boundary-condition sensitivity ``H_bc`` added.

    Args:
        fp_and_data: Site datasets containing acquired directional ``bc_n``,
            ``bc_e``, ``bc_s`` and ``bc_w`` contributions. Dot-prefixed metadata
            entries are retained unchanged.
        bc_basis: Resolved labelled basis with ``bc_basis_<direction>`` variables
            and a ``region`` dimension. A singleton time applies to every
            observation; multiple times align by their labels. None uses the
            acquired NESW contributions directly, without resource I/O.

    Returns:
        New mapping and site datasets, sharing input storage where possible,
        with ``H_bc`` indexed by ``bc_region``. Inputs remain unchanged and
        lazy arrays remain lazy.
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
        bc_ds = ds[[f"bc_{d}" for d in "nesw"]]
        if bc_basis is None:
            bc_ds = bc_ds.rename({f"bc_{d}": d for d in "nesw"})
            sensitivity = bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region")
        else:
            sensitivity = (
                (bc_ds * bc_basis).sum(["lat", "lon", "height"])
                .to_dataarray(dim="__newdim__").sum("__newdim__")
                .rename(region="bc_region")
            )
        result[site] = ds.assign(H_bc=sensitivity)
    return result
