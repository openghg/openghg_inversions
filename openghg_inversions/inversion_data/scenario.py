import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData


def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    bc_data: BoundaryConditionsData | None = None,
    inner_footprint_data: FootprintData | None = None,
    platform: str | None = None,
    max_level: int | None = None
) -> xr.Dataset:
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

    if inner_footprint_data is not None:
        inner_scenario = ModelScenario(obs=obs_data, footprint=inner_footprint_data, flux=flux_dict, bc=None)
        inner_domain_merged = inner_scenario.footprints_data_merge(
            calc_fp_x_flux=True,
            calc_bc_sensitivity=False,
            cache=False,
        )
        scenario_combined = scenario_combined.copy()

        # 6km fp_x_flux is added as a separate variable to the combined dataset, and can be merged with the EUROPE fp_x_flux.
        scenario_combined["fp_x_flux_inner"] = inner_domain_merged["fp_x_flux"]

        return scenario_combined
    else:
        return scenario_combined
