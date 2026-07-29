import logging

import numpy as np
import pandas as pd
import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData

logger = logging.getLogger(__name__)


def _snap_footprint_times_to_obs(
    obs_data: ObsData,
    footprint_data: FootprintData,
    *,
    tolerance: str | pd.Timedelta = "1s",
) -> None:
    """Snap satellite footprint times to matching obs times within a small tolerance.

    OCO-2 column observations can retain nanosecond precision while matching
    footprint times may be stored at lower precision. Xarray's exact coordinate
    alignment then treats the same retrievals as different times, creating
    mostly-NaN merged datasets that are later dropped by inversion preparation.
    """
    obs_time = obs_data.data.indexes.get("time")
    fp_time = footprint_data.data.indexes.get("time")
    if obs_time is None or fp_time is None or len(obs_time) == 0 or len(fp_time) == 0:
        return

    tolerance = pd.Timedelta(tolerance)
    indexer = obs_time.get_indexer(fp_time, method="nearest", tolerance=tolerance)
    matched_positions = np.flatnonzero(indexer >= 0)
    if matched_positions.size == 0:
        return

    targets, counts = np.unique(indexer[matched_positions], return_counts=True)
    ambiguous_targets = set(targets[counts > 1])
    positions_to_snap = [
        position for position in matched_positions if indexer[position] not in ambiguous_targets
    ]
    if not positions_to_snap:
        return

    snapped_times = fp_time.values.copy()
    snapped_times[positions_to_snap] = obs_time.values[indexer[positions_to_snap]]
    if pd.Index(snapped_times).has_duplicates:
        logger.warning("Skipping satellite timestamp snapping because it would create duplicate footprint times.")
        return

    footprint_data.data = footprint_data.data.assign_coords(time=snapped_times)
    logger.info(
        "Snapped %s satellite footprint timestamp(s) to observation timestamps within %s.",
        len(positions_to_snap),
        tolerance,
    )


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
        _snap_footprint_times_to_obs(obs_data, footprint_data)
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
    )

    return scenario_combined
