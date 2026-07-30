"""Build one surface or column OpenGHG model scenario.

``merged_scenario_data`` forwards a shared ``output_units`` target so fresh
multi-site retrieval delegates unit conversion to OpenGHG itself.
"""

import xarray as xr
from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData, ObsData

from openghg_inversions.inversion_data._site_options import is_column_platform


def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    bc_data: BoundaryConditionsData | None = None,
    platform: str | None = None,
    max_level: int | None = None,
    split_by_sectors: bool = False,
    output_units: str | None = None,
) -> xr.Dataset:
    """Create a ``ModelScenario`` and return its merged, unit-aligned data.

    Args:
        obs_data: Surface or column observation data.
        footprint_data: Footprint data paired with the observations.
        flux_dict: Flux data keyed by source.
        bc_data: Optional boundary-condition data.
        platform: Observation platform used for column handling.
        max_level: Maximum vertical level for column observations.
        split_by_sectors: Preserve sector-resolved modelled concentrations.
        output_units: Shared unit target. ``None`` uses this scenario's
            observation units; later sites receive the first retained target.

    Returns:
        The merged OpenGHG scenario dataset.
    """
    # Create ModelScenario object for all emissions_sectors
    # and combine into one object
    if is_column_platform(platform):
        model_scenario = ModelScenario(
            obs_column=obs_data,
            footprint=footprint_data,
            flux=flux_dict,
            bc=bc_data,
            platform=platform,
            max_level=max_level,
        )
    else:
        model_scenario = ModelScenario(
            obs=obs_data,
            footprint=footprint_data,
            flux=flux_dict,
            bc=bc_data,
        )

    scenario_combined = model_scenario.footprints_data_merge(
        platform=platform,
        calc_fp_x_flux=True,
        split_by_sectors=split_by_sectors,
        calc_bc_sensitivity=True,
        cache=False,
        output_units=output_units,
    )

    return scenario_combined
