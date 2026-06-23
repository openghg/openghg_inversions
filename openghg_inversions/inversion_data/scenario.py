import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData

def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    inner_flux_dict: dict[str, FluxData] | None,
    bc_data: BoundaryConditionsData | None = None,
    inner_footprint_data: FootprintData | None = None,
    platform: str | None = None,
    max_level: int | None = None
) -> xr.DataTree:
    """Create ModelScenario and get result of `footprint_data_merge`."""
    # Create ModelScenario object for all emissions_sectors
    # and combine into one object
    if platform is not None and "satellite" in platform:
        model_scenario = ModelScenario(
            obs_column=obs_data,
            footprint=footprint_data,
            flux=flux_dict,
            bc=bc_data,
            platform=platform,
            max_level=max_level
        )
    else: 
        model_scenario = ModelScenario(
            obs=obs_data,
            footprint=footprint_data,
            flux=flux_dict,
            bc=bc_data,
        )

    # TODO: should we make this option explicit? Multiple fluxes can be stacked and used as a single flux
    split_by_sectors = len(flux_dict) > 1
    scenario_combined = model_scenario.footprints_data_merge(
        platform=platform,
        calc_fp_x_flux=True,
        split_by_sectors=split_by_sectors,
        calc_bc_sensitivity=True,
        cache=False,
    )

    dt_dict: dict[str, xr.Dataset] = {"/standard": scenario_combined}
    if inner_footprint_data is not None:
        if inner_flux_dict is None:
            raise ValueError(
                "Inner-domain footprints were loaded but no inner-domain flux was supplied. "
                "Set `inner_emissions_store` so inner fp_x_flux is computed from native inner flux."
            )

        inner_scenario = ModelScenario(obs=obs_data, footprint=inner_footprint_data, flux=inner_flux_dict, bc=None)
        inner_domain_merged = inner_scenario.footprints_data_merge(
            calc_fp_x_flux=True,
            calc_bc_sensitivity=False,
            cache=False,
        )

        # Align inner to outer time axis.
        # If inner footprint is missing any timestamps that exist in the outer
        # scenario (e.g. sparse inner store coverage), fill those with 0 so
        # both nodes share exactly the same time dimension in the DataTree.
        inner_domain_merged = inner_domain_merged.reindex(
            time=scenario_combined.time, fill_value=0.0
        )

        dt_dict["/inner"] = inner_domain_merged

    return xr.DataTree.from_dict(dt_dict)
