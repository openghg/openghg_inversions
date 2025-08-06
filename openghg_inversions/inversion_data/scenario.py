import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData

from .getters import convert_bc_units


def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    bc_data: BoundaryConditionsData | None = None,
    platform: str | None = None,
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

    split_by_sectors = len(flux_dict) > 1
    scenario_combined = model_scenario.footprints_data_merge(platform=platform, calc_fp_x_flux=True, split_by_sectors=split_by_sectors)

    return scenario_combined
