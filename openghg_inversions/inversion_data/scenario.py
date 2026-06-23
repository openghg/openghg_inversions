import pandas as pd
import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData


def _time_alignment_tolerance(averaging_period: str | None) -> pd.Timedelta | None:
    if averaging_period is None:
        return None

    try:
        return pd.Timedelta(averaging_period) / 2
    except ValueError:
        return None


def align_inner_to_outer_time(
    inner_domain_merged: xr.Dataset,
    scenario_combined: xr.Dataset,
    averaging_period: str | None,
) -> xr.Dataset:
    """Align inner-domain scenario times to the outer/observation time axis."""
    tolerance = _time_alignment_tolerance(averaging_period)
    if tolerance is None:
        return inner_domain_merged.reindex(time=scenario_combined.time, fill_value=0.0)

    return inner_domain_merged.reindex(
        time=scenario_combined.time,
        method="nearest",
        tolerance=tolerance,
        fill_value=0.0,
    )


def merged_scenario_data(
    obs_data: ObsData,
    footprint_data: FootprintData,
    flux_dict: dict[str, FluxData],
    inner_flux_dict: dict[str, FluxData] | None,
    bc_data: BoundaryConditionsData | None = None,
    inner_footprint_data: FootprintData | None = None,
    platform: str | None = None,
    max_level: int | None = None,
    averaging_period: str | None = None,
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

        # Align inner to the outer/obs time axis. Use nearest matching within
        # half the averaging window so small timestamp offsets do not remove a
        # valid inner contribution. Genuinely missing inner coverage remains 0.
        inner_domain_merged = align_inner_to_outer_time(
            inner_domain_merged=inner_domain_merged,
            scenario_combined=scenario_combined,
            averaging_period=averaging_period,
        )

        dt_dict["/inner"] = inner_domain_merged

    return xr.DataTree.from_dict(dt_dict)
