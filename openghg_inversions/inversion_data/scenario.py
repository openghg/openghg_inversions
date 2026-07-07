import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData


def align_inner_to_outer_time(
    inner_domain_merged: xr.Dataset,
    scenario_combined: xr.Dataset,
    averaging_period: str | None,
) -> xr.Dataset:
    """Align inner-domain scenario times to the outer/observation time axis.

    Missing inner times are assigned the nearest available inner value rather
    than zero so the inner-domain contribution stays consistent with the
    nearest-time footprint/flux treatment.
    """
    outer_count = int(scenario_combined.sizes.get("time", 0))
    inner_count = int(inner_domain_merged.sizes.get("time", 0))
    exact_matches = 0
    if "time" in scenario_combined.coords and "time" in inner_domain_merged.coords:
        exact_matches = int(scenario_combined.time.isin(inner_domain_merged.time).sum().compute())

    try:
        aligned = inner_domain_merged.reindex(
            time=scenario_combined.time,
            method="nearest",
        )
        align_method = "nearest"
    except ValueError:
        # If there is no usable inner time coordinate to choose a nearest
        # value from, fall back to zero contribution rather than leaving NaNs.
        aligned = inner_domain_merged.reindex(time=scenario_combined.time, fill_value=0.0)
        align_method = "zero-fill"

    print(
        "DIAGNOSTIC inner_time_alignment | "
        f"outer_times={outer_count} inner_times={inner_count} exact_matches={exact_matches} "
        f"outer_without_exact_inner={outer_count - exact_matches} method={align_method}",
        flush=True,
    )

    return _zero_entirely_missing_time_vars(aligned)


def _zero_entirely_missing_time_vars(dataset: xr.Dataset) -> xr.Dataset:
    """Replace variables that are entirely missing on time with zero."""
    updated = dataset
    for name, data_var in dataset.data_vars.items():
        if "time" not in data_var.dims:
            continue
        if bool(data_var.isnull().all().compute()):
            print(
                "DIAGNOSTIC zero_fill | "
                f"variable={name} reason=entire_time_variable_missing size={data_var.size}",
                flush=True,
            )
            updated = updated.assign({name: data_var.fillna(0.0)})
    return updated


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
) -> xr.Dataset | xr.DataTree:
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

        # Align inner to the outer/obs time axis using nearest values. Do not
        # fill missing inner contribution with zeros.
        inner_domain_merged = align_inner_to_outer_time(
            inner_domain_merged=inner_domain_merged,
            scenario_combined=scenario_combined,
            averaging_period=averaging_period,
        )

        return xr.DataTree.from_dict({"/standard": scenario_combined, "/inner": inner_domain_merged})

    return scenario_combined
