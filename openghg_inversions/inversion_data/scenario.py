import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData

from .getters import convert_bc_units


def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    bc_data: BoundaryConditionsData | None = None,
) -> xr.Dataset:
    """Create ModelScenario and get result of `footprint_data_merge`."""
    # convert bc units, if using bc
    use_bc = bc_data is not None
    unit = float(obs_data.data.mf.units)
    bc_data = convert_bc_units(bc_data, unit) if use_bc else None

    # Create ModelScenario object for all emissions_sectors
    # and combine into one object
    model_scenario = ModelScenario(
        obs=obs_data,
        footprint=footprint_data,
        flux=flux_dict,
        bc=bc_data,
    )

    if len(flux_dict) == 1:
        scenario_combined = model_scenario.footprints_data_merge()
        if use_bc is True:
            scenario_combined.bc_mod.values *= unit

    else:
        # Create model scenario object for each flux sector
        model_scenario_dict = {}

        for source in flux_dict:
            scenario_sector = model_scenario.footprints_data_merge(sources=source, recalculate=True)

            if model_scenario.species.lower() == "co2":
                model_scenario_dict["mf_mod_high_res_" + source] = scenario_sector["mf_mod_high_res"]
            else:
                model_scenario_dict["mf_mod_" + source] = scenario_sector["mf_mod"]

        scenario_combined = model_scenario.footprints_data_merge(recalculate=True)

        for k, v in model_scenario_dict.items():
            scenario_combined[k] = v
            if use_bc is True:
                scenario_combined.bc_mod.values *= unit

    return scenario_combined
