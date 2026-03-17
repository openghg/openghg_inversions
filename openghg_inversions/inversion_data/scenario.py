import xarray as xr

from openghg.analyse import ModelScenario
from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData

def _mask_flux_to_inner_domain(
    flux_dict: dict[str, FluxData],
    inner_footprint_data: FootprintData,
) -> dict[str, FluxData]:
    """Mask EUROPE-domain flux values to zero outside the inner footprint extent.

    The inner footprint (``inner_fp``) is non-zero only within the inner
    domain (e.g. 6 km grid).  We use this spatial coverage to build a
    boolean mask, regrid the EUROPE flux to the inner footprint lat/lon
    coordinates, and then zero out any flux cells that fall outside the
    inner domain extent.

    The masked flux is returned as a new dict of ``FluxData`` objects so
    that ``ModelScenario`` can use it unmodified to compute ``fp_x_flux``
    on the correct inner grid.

    Args:
        flux_dict: EUROPE-domain flux, keyed by source name.
        inner_footprint_data: FootprintData for the inner domain whose
            raw ``fp`` defines the spatial extent and lat/lon grid.

    Returns:
        New dict of ``FluxData`` with flux regridded to the inner grid
        and zeroed outside the inner domain footprint coverage.
    """
    # Inner fp: dims (time, lat, lon) on the inner (e.g. 6 km) grid.
    inner_fp: xr.DataArray = inner_footprint_data.data.fp

    # Boolean mask: True where inner domain has any non-zero fp at any time.
    inner_domain_mask: xr.DataArray = (inner_fp != 0).any("time")  # (lat, lon)

    inner_lat = inner_fp.lat
    inner_lon = inner_fp.lon

    masked_flux_dict: dict[str, FluxData] = {}

    for source, flux_data in flux_dict.items():
        flux_da: xr.DataArray = flux_data.data.flux  # EUROPE grid (time, lat, lon)

        # 1. Regrid EUROPE flux to inner footprint lat/lon grid
        flux_on_inner = (
            flux_da
            .interp(lat=inner_lat, lon=inner_lon, method="nearest")
            .assign_coords(lat=inner_lat, lon=inner_lon)
            .fillna(0.0)
        )

        # 2. Mask: zero out flux cells outside the inner domain extent
        flux_masked = flux_on_inner.where(inner_domain_mask, other=0.0)

        # 3. Build a new FluxData with the masked flux dataset,
        #    preserving all original metadata and dataset attributes.
        masked_ds = flux_data.data.copy()
        masked_ds["flux"] = flux_masked

        masked_flux_dict[source] = FluxData(data=masked_ds, metadata=flux_data.metadata)

    return masked_flux_dict

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
        # Mask the EUROPE flux to the inner domain extent (zero outside),
        # regridded to the inner footprint lat/lon grid.
        # ModelScenario then computes fp_x_flux on the inner grid correctly.
        flux_dict_inner = _mask_flux_to_inner_domain(flux_dict, inner_footprint_data)

        inner_scenario = ModelScenario(obs=obs_data, footprint=inner_footprint_data, flux=flux_dict_inner, bc=None)
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
