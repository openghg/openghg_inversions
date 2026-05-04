import xarray as xr
import numpy as np
import pandas as pd
from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData


def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    bc_data: BoundaryConditionsData | None = None,
    platform: str | None = None,
    max_level: int | None = None,
    split_by_sectors: bool = False,
) -> xr.Dataset:
    """Create ModelScenario and get result of `footprint_data_merge`."""
    # Create ModelScenario object for all emissions_sectors
    # and combine into one object
    if platform is not None and "satellite" in platform:
        time_resolved = footprint_data.metadata.get("time_resolved", False)
        if isinstance(time_resolved, str):
            time_resolved = time_resolved.lower() == "true"
        
        # Align obs timestamps to footprint timestamps for both integrated and
        # time-resolved (HR) footprints. Obs timestamps may carry sub-microsecond
        # nanosecond noise that prevents exact matching; floor to microseconds first,
        # then snap to the nearest footprint timestamp within a 1 us tolerance.
        obs_times = pd.to_datetime(obs_data.data.time.values)
        fp_times = pd.to_datetime(footprint_data.data.time.values)

        obs_times_floored = obs_times.floor("us")
        idx = fp_times.get_indexer(obs_times_floored, method="nearest", tolerance=pd.Timedelta("1us"))

        new_obs_times = obs_times.to_numpy().copy()
        matched = idx >= 0
        new_obs_times[matched] = fp_times[idx[matched]].to_numpy()

        obs_data.data = obs_data.data.assign_coords(time=("time", new_obs_times))
        print(f"Timestamp alignment: {matched.sum()}/{len(obs_times)} obs timestamps snapped to footprint.")
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

    scenario_combined = model_scenario.footprints_data_merge(
        platform=platform,
        calc_fp_x_flux=True,
        split_by_sectors=split_by_sectors,
        calc_bc_sensitivity=True,
        cache=False,
    )

    return scenario_combined
