"""Retrieve and merge observations, footprints, fluxes, and boundary data.

``data_processing_surface_notracer`` expands scalar site options, keeps their
positions aligned during retrieval failures, assembles per-site model
scenarios, and can save the resulting merged-data artifact. Retrieval and
optional saving have external object-store and filesystem side effects; loading
saved merged data is handled by :mod:`openghg_inversions.inversion_data.serialise`.
The first successful scenario supplies the unit target forwarded to later
OpenGHG ``ModelScenario`` merges.
"""

import logging
import warnings
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any, Literal

import numpy as np
import xarray as xr
from openghg.retrieve import get_bc
from openghg.types import SearchError

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data._site_options import (
    expand_site_boolean_option,
    expand_site_option,
    is_column_observation,
    is_column_platform,
    is_satellite_platform,
)
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.inversion_data.getters import (
    get_flux_data,
    get_footprint_data,
    get_obs_data,
)
from openghg_inversions.inversion_data.scenario import merged_scenario_data
from openghg_inversions.inversion_data.serialise import _save_merged_data

logger = logging.getLogger(__name__)


def add_obs_error(sites: list[str], fp_all: dict, add_averaging_error: bool = True) -> None:
    """Create `mf_error` variable.

    The `mf_error` variables contains either `mf_repeatablility`, `mf_variability`
    or the square root of the sum of the squares of both, if `add_averaging_error` is True.

    This function modifies `fp_all` in place, adding `mf_error` and making sure that both
    `mf_repeatability` and `mf_variability` are present.

    Note: if `averaging_period` is specified in `data_processing_surface_notracer`, then OpenGHG
    will add an `mf_variability` variable with the standard deviation of the obs over the specified
    period. If `mf_variability` is already present (for instance, for Picarro data), then the existing
    variable is over-written. If the `averaging_period` matches the frequency of the data, this will
    make `mf_variability` zero (since the stdev of one value is 0).

    Args:
        sites: list of site names to process
        fp_all: dictionary of `ModelScenario` objects, keyed by site names
        add_averaging_error: if True, combine repeatability and variability to make `mf_error`
            variable. Otherwise, `mf_error` will equal `mf_repeatability` if it is present, otherwise
            it will equal `mf_variability`.

    Returns:
        None, modifies `fp_all` in place.
    """
    # TODO: do we want to fill missing values in repeatability or variability?
    for site in sites:
        ds = fp_all[site]
        mf_long_name = ds.mf.attrs.get("long_name", "")
        mf_units = ds.mf.attrs.get("units", None)

        variability_missing = False
        if "mf_variability" not in ds:
            ds["mf_variability"] = xr.zeros_like(ds.mf)
            variability_missing = True
        ds["mf_variability"].attrs["long_name"] = mf_long_name + "_variability"
        ds["mf_variability"].attrs["units"] = mf_units

        if "mf_repeatability" not in ds:
            if variability_missing:
                raise ValueError(f"Obs data for site {site} is missing both repeatability and variability.")

            ds["mf_repeatability"] = xr.zeros_like(ds.mf_variability)

            ds["mf_error"] = ds["mf_variability"]

            if add_averaging_error:
                logger.info(
                    "`mf_repeatability` not present; using `mf_variability` for `mf_error` at site %s", site
                )

        elif add_averaging_error:
            # Fill with zeros so that if one of repeatability and variability is not NaN, then mf_error will not be NaN.
            ds["mf_error"] = np.sqrt(
                ds["mf_repeatability"].fillna(0) ** 2 + ds["mf_variability"].fillna(0) ** 2
            )
        else:
            ds["mf_error"] = ds["mf_repeatability"]

        ds["mf_repeatability"].attrs["long_name"] = mf_long_name + "_repeatability"
        ds["mf_repeatability"].attrs["units"] = mf_units
        ds["mf_error"].attrs["long_name"] = mf_long_name + "_error"
        ds["mf_error"].attrs["units"] = mf_units

        # warnings/info for debugging
        err0 = (ds["mf_error"] == 0) | (
            ds["mf_error"].isnull()
        )  # might have NaN if add_averaging_error is False

        if err0.any():
            percent0 = 100 * err0.mean()
            logger.warning(
                (
                    "`mf_error` is zero/nan for %.2f percent of times at site %s;"
                    "filling with max(median(mf_error), std(mf))."
                ),
                percent0,
                site,
            )

            mf_err_da = ds["mf_error"].as_numpy()  # load into memory to avoid Dask issues
            fill_value = np.nanmax(
                [
                    mf_err_da.where(mf_err_da != 0).dropna(dim="time").median(),
                    ds["mf"].std(dim="time"),
                ]
            )
            ds["mf_error"] = mf_err_da.where(mf_err_da != 0, fill_value)
            info_msg = (
                "If `averaging_period` matches the frequency of the obs data, then `mf_variability` "
                "will be zero. Try setting `averaging_period = None`."
            )
            logger.info(info_msg)


def convert_to_list(
    x: Iterable[Any] | str | slice | int | Integral | None,
    length: int,
    name: str | None = None,
) -> list[Any]:
    """Convert a scalar or sequence to a list of the expected size.

    Args:
        x: Scalar string/integer/slice/``None`` to broadcast, or an iterable
            to copy.
        length: Required output length.
        name: Optional argument name used in error messages.

    Returns:
        A new list of the requested length.

    Raises:
        ValueError: If an iterable has the wrong length, or if ``x`` is neither
            a supported scalar nor an iterable.
    """
    return list(expand_site_option(x, nsites=length, name=name or "value"))


def data_processing_surface_notracer(
    species: str,
    sites: Sequence[str] | str,
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    obs_data_level: list[str | None] | str | None = None,
    platform: list[str | None] | str | None = None,
    inlet: Sequence[str | slice | None] | str | None = None,
    instrument: list[str | None] | str | None = None,
    max_level: Sequence[int | None] | int | None = None,
    calibration_scale: str | None = None,
    met_model: list[str | None] | str | None = None,
    fp_model: str | None = None,
    fp_height: list[str | None | Literal["auto"]] | Literal["auto"] | str | None = None,
    fp_species: str | None = None,
    time_resolved: Sequence[bool | None] | bool | None = None,
    emissions_name: list | None = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    bc_store: str | None = None,
    obs_store: str | list[str] | None = None,
    footprint_store: str | list[str] | None = None,
    emissions_store: str | None = None,
    split_by_sectors: bool = False,
    averagingerror: bool = True,
    save_merged_data: bool = False,
    merged_data_name: str | None = None,
    merged_data_dir: str | None = None,
    output_name: str | None = None,
    time_resolved: Sequence[bool | None] | bool | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> tuple[dict, list, list, list, list, list]:
    """Retrieve and prepare surface or column datasets from OpenGHG stores.

    Use for forward simulations and model-data comparisons that do not
    use tracers.

    Args:
        species: Atmospheric trace gas species of interest
            e.g. "co2"
        sites: Measurement station/site abbreviation, or a sequence of them,
            e.g. ``"MHD"`` or ``["MHD", "TAC"]``.
            NOTE: for satellite, pass as "satellitename-obs_region" eg "GOSAT-BRAZIL" and pass corresponding platform as "satellite"
        domain: Model domain region of interest; e.g. "EUROPE"
        averaging_period: Averaging period to apply to mole fraction data,
            either scalar or aligned to ``sites``.
        start_date: Date from which to gather data; e.g. "2020-01-01"
        end_date: Date until which to gather data; e.g. "2020-02-01"
        obs_data_level: ICOS observation data level, either scalar or aligned
            to ``sites``. For non-ICOS sites use ``None``.
        platform: Observation platform, either scalar or aligned to ``sites``.
        inlet: Observation inlet selector, either scalar or aligned to
            ``sites``. Entries may be strings, legacy ``slice`` selectors, or
            ``None``.
        instrument: Observation instrument, either scalar or aligned to
            ``sites``.
        max_level: Maximum atmospheric level to extract, either scalar or
            aligned to ``sites``. This is required for satellite/site-column
            data.
        calibration_scale: Convert measurements to defined calibration scale
        met_model: Meteorological model used in the LPDM, either scalar or
            aligned to ``sites``.
        fp_model: LPDM used for generating footprints.
        fp_height: Inlet height used in footprints for corresponding sites.
        fp_species: Species name associated with footprints in the object store
        time_resolved: Select integrated (``False``) or time-resolved
            high-frequency (``True``) footprints, either as one value for all
            sites or aligned to ``sites``. ``None`` leaves selection to the
            OpenGHG search metadata.
        emissions_name: List of keywords args associated with emissions files in the object store.
            Corresponds to `source` in OpenGHG.
        use_bc: Option to include boundary conditions in model
        bc_input: Variable for calling BC data from 'bc_store' - equivalent of 'emissions_name' for fluxes.
        bc_store: Name of object store to retrieve boundary conditions data from.
        obs_store: Name of object store to retrieve observations data from.
        footprint_store: Name of object store to retrieve footprints data from.
        emissions_store: Name of object store to retrieve emissions data from.
        flux_non_finite_check: Non-finite flux handling mode. ``"lazy"``
            applies zero-fill lazily and records attrs; ``"count"`` computes
            count metadata once and warns if non-finite values are present.
        split_by_sectors: If True, calculate sector-resolved ``fp_x_flux_sectoral`` in ModelScenario.
            If False (default), combine all flux sources into a single ``fp_x_flux`` pathway.
        averagingerror: Adds the variability in the averaging period to the measurement
            error if set to True.
        save_merged_data: Save forward simulations data and observations.
        merged_data_name: Filename for saved forward simulations data and observations.
        merged_data_dir: Directory path for for saved forward simulations data and observations.
        output_name: Optional name used to create merged data name.

    Returns:
        tuple: containing

            - fp_all: dictionary containing flux data (key ".flux"), bc data (key ".bc"),
              and observations data (site short name as key)
            - sites: Updated list of sites. All put in upper case and if data was not extracted
              correctly for any sites, drop these from the rest of the inversion.
            - inlet: List of inlet height for the updated list of sites
            - fp_height: List of footprint height for the updated list of sites
            - instrument: List of instrument for the updated list of sites
            - averaging_period: List of averaging_period for the updated list of sites

    Raises:
        SearchError: If no requested site has both observations and footprints.
        ValueError: If aligned options have invalid lengths, emissions are not
            specified, observation units are unavailable or incompatible, or
            required error inputs are absent.

    Notes:
        This function reads OpenGHG stores, emits progress messages and
        warnings, and may save a merged-data artifact. The first retained
        scenario defines the unit target requested for later sites;
        ``fp_all[".units"]`` stores that unit's scale against ``mol/mol``.
    """
    site_values = [sites] if isinstance(sites, str) else sites
    sites = [site.upper() for site in site_values]

    # Convert 'None' args to list
    nsites = len(sites)
    inlet = convert_to_list(inlet, nsites, "inlet")
    instrument = convert_to_list(instrument, nsites, "instrument")
    fp_height = convert_to_list(fp_height, nsites, "fp_height")
    obs_data_level = convert_to_list(obs_data_level, nsites, "obs_data_level")
    met_model = convert_to_list(met_model, nsites, "met_model")
    averaging_period = convert_to_list(averaging_period, nsites, "averaging_period")
    platform = convert_to_list(platform, nsites, "platform")
    max_level = convert_to_list(max_level, nsites, "max_level")
    time_resolved = list(
        expand_site_boolean_option(time_resolved, nsites=nsites, name="time_resolved")
    )
    invalid_max_levels = [
        value
        for value in max_level
        if value is not None and (not isinstance(value, Integral) or isinstance(value, bool))
    ]
    if invalid_max_levels:
        raise ValueError(
            f"`max_level` entries must be integers or None. Invalid value(s): {invalid_max_levels!r}."
        )
    max_level = [None if value is None else int(value) for value in max_level]

    fp_all = {}
    fp_all[".species"] = species.upper()

    # Get flux data
    if emissions_name is None:
        raise ValueError("`emissions_name` must be specified")

    flux_dict = get_flux_data(
        sources=emissions_name,
        species=species,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        store=emissions_store,
        flux_non_finite_check=flux_non_finite_check,
    )
    fp_all[".flux"] = flux_dict
    fp_all[".split_by_sectors"] = split_by_sectors

    # Get BC data
    if use_bc is True:
        try:
            bc_data = get_bc(
                species=species,
                domain=domain,
                bc_input=bc_input,
                start_date=start_date,
                end_date=end_date,
                store=bc_store,
            )
        except SearchError as e:
            raise SearchError("Could not find matching boundary conditions.") from e
        else:
            fp_all[".bc"] = bc_data
    else:
        bc_data = None

    # get obs and footprints, and make scenarios for each site
    scales = {}
    check_scales = set()
    site_indices_to_keep = []
    output_units: str | None = None

    keep_variables = [
        f"{species}",
        f"{species}_variability",
        f"{species}_repeatability",
        f"{species}_number_of_observations",
        "inlet",  # needed if multiple inlets combined
        "inlet_height",  # sometimes needed if inlet='multiple' (may be outdated soon)
    ]
    warnings.warn(f"Dropping all variables besides {keep_variables}")
    for i, site in enumerate(sites):
        # Get observations data
        site_platform = platform[i]
        if isinstance(site_platform, str) and site_platform.lower() == "flask":
            avg_period = None
        else:
            avg_period = averaging_period[i]

        site_data = get_obs_data(
            site=site,
            species=species,
            inlet=inlet[i],
            start_date=start_date,
            domain=domain,
            platform=site_platform,
            end_date=end_date,
            data_level=obs_data_level[i],
            average=avg_period,
            instrument=instrument[i],
            calibration_scale=calibration_scale,
            max_level=max_level[i],
            stores=obs_store,
            keep_variables=keep_variables,
        )

        if site_data is None:
            print(f"No obs. found, continuing model run without {site}.\n")
            continue

        # Get footprints data
        footprint_data = get_footprint_data(
            site=site,
            domain=domain,
            platform=site_platform,
            fp_height=fp_height[i],
            start_date=start_date,
            end_date=end_date,
            model=fp_model,
            met_model=met_model[i],
            fp_species=fp_species,
            averaging_period=averaging_period[i],
            time_resolved=time_resolved[i],
            obs_data=site_data,
            stores=footprint_store,
        )
        if footprint_data is None:
            print(
                f"\nNo footprint data found for {site} with inlet/height {fp_height[i]}, model {fp_model}, and domain {domain}.",
                f"Check these values.\nContinuing model run without {site}.\n",
            )
            continue  # skip this site

        scenario_platform = (
            "site-column"
            if is_column_observation(inlet[i], site_platform) and not is_column_platform(site_platform)
            else site_platform
        )
        try:
            scenario_combined = merged_scenario_data(
                site_data,
                footprint_data,
                flux_dict,
                bc_data,
                platform=scenario_platform,
                max_level=max_level[i],
                split_by_sectors=split_by_sectors,
                output_units=output_units,
            )
        except (TypeError, ValueError) as exc:
            if output_units is None:
                raise
            raise ValueError(
                f"Could not merge site {site!r} using target observation units {output_units!r}."
            ) from exc
        if output_units is None:
            scenario_units = scenario_combined["mf"].attrs.get("units")
            if not isinstance(scenario_units, str) or not scenario_units:
                raise ValueError(f"No observation units detected for the first retained site {site!r}.")
            output_units = scenario_units
        fp_all[site] = scenario_combined

        if not is_satellite_platform(site_platform):
            scales[site] = scenario_combined.scale
            check_scales.add(scenario_combined.scale)

        site_indices_to_keep.append(i)
    if len(site_indices_to_keep) == 0:
        raise SearchError("No site data found. Exiting process.")

    # If data was not extracted correctly for any sites, drop these from the rest of the inversion
    if len(site_indices_to_keep) < len(sites):
        sites = [sites[s] for s in site_indices_to_keep]
        inlet = [inlet[s] for s in site_indices_to_keep]
        fp_height = [fp_height[s] for s in site_indices_to_keep]
        instrument = [instrument[s] for s in site_indices_to_keep]
        averaging_period = [averaging_period[s] for s in site_indices_to_keep]

    # if "satellite" not in footprint_data.metadata:
    # check for consistency of calibration scales
    if len(check_scales) > 1:
        msg = f"Not all sites using the same calibration scale: {len(check_scales)} scales found."
        logger.warning(msg)

    fp_all[".scales"] = scales

    # create `mf_error`
    add_obs_error(sites, fp_all, add_averaging_error=averagingerror)
    if output_units is None:
        raise ValueError("No observation units detected.")
    fp_all[".units"] = mole_fraction_unit_scale(
        output_units,
        context=f"site {sites[0]!r} variable 'mf'",
    )

    if save_merged_data:
        if merged_data_dir is None:
            print("`merged_data_dir` not specified; could not save merged data")
        else:
            _save_merged_data(
                fp_all,
                merged_data_dir,
                merged_data_name=merged_data_name,
                species=species,
                start_date=start_date,
                output_name=output_name,
            )
            print(f"\nfp_all saved in {merged_data_dir}\n")

    return fp_all, sites, inlet, fp_height, instrument, averaging_period
