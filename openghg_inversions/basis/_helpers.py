"""Boundary-condition sensitivity helpers."""

from ._functions import basis_boundary_conditions


def bc_sensitivity(
    fp_and_data: dict, domain: str, basis_case: str, bc_basis_directory: str | None = None
) -> dict:
    """Add boundary conditions sensitivity matrix `H_bc` to each site xr.Dataframe in fp_and_data.

    Args:
        fp_and_data: dict containing xr.Datasets output by `ModelScenario.footprints_data_merge`
            keyed by site code.
        domain: inversion domain. For instance "EUROPE"
        basis_case: BC basis case to read in. Examples of basis cases are "NESW","stratgrad".
        bc_basis_directory: bc_basis_directory can be specified if files are not in the default
            directory. Must point to a directory which contains subfolders organized
            by domain. (optional)

    Returns:
        dict of xr.Datasets in same format as fp_and_data with `H_bc` sensitivity matrix added.

    """
    sites = [key for key in list(fp_and_data.keys()) if key[0] != "."]

    if basis_case.lower() == "nesw":
        for site in sites:
            ds = fp_and_data[site]
            bc_ds = ds[[f"bc_{d}" for d in "nesw"]].rename({f"bc_{d}": d for d in "nesw"})
            sensitivity = bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region")
            fp_and_data[site]["H_bc"] = sensitivity

        return fp_and_data

    basis_func = basis_boundary_conditions(
        domain=domain, basis_case=basis_case, bc_basis_directory=bc_basis_directory
    )

    # drop time if there is only one value
    if basis_func.sizes.get("time", -1) == 1:
        basis_func = basis_func.squeeze("time")
    else:
        basis_func = basis_func.sortby("time")

    # align basis data var names with baseline sensitivity data var names from ModelScenario
    bc_basis = basis_func.rename({dv: str(dv).replace("basis_", "") for dv in basis_func.data_vars})

    for site in sites:
        ds = fp_and_data[site]
        bc_ds = ds[[f"bc_{d}" for d in "nesw"]]
        sensitivity = (
            (bc_ds * bc_basis).sum(["lat", "lon", "height"]).to_dataarray(dim="__newdim__").sum("__newdim__")
        )
        sensitivity = sensitivity.rename(region="bc_region")
        fp_and_data[site]["H_bc"] = sensitivity

    return fp_and_data
