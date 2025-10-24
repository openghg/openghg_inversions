"""Functions for retrieving data from an OpenGHG store.

These functions customise the behavior of `get_flux`, `get_footprint`, etc.

- `get_flux_data` calls `get_flux` after adjusting the start date to begin at the
  start of a year or month; if nothing is found, then the start date restriction
  is dropped and the most recent result found is used.

TODO: add more docs (and add more detailed docstrings)
"""

import logging
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from openghg.dataobjects import ObsData, FluxData, FootprintData
from openghg.retrieve import get_flux, get_footprint, get_obs_column, get_obs_surface, search_footprints, search_flux
from openghg.types import SearchError


logger = logging.getLogger(__name__)


# Flux
def adjust_flux_start_date(
    start_date: str, species: str, source: str, domain: str, store: str | None = None
) -> pd.Timestamp:
    """Adjusts the flux start_date to align with the flux data's temporal resolution."""
    flux_search = search_flux(species=species, source=source, domain=domain, store=store)
    if flux_search.results.empty:
        raise SearchError(
            f"No flux found with species={species}, source={source}, domain={domain}, store={store}."
        )
    flux_period = flux_search.results["time_period"][0]

    start_date_flux = pd.to_datetime(start_date)

    if flux_period == "1 year" and not start_date_flux.is_year_start:
        start_date_flux = start_date_flux - pd.offsets.YearBegin()
    elif flux_period == "1 month" and not start_date_flux.is_month_start:
        start_date_flux = start_date_flux - pd.offsets.MonthBegin()

    return start_date_flux


def get_flux_data(
    sources: list[str],
    species: str,
    domain: str,
    start_date: str,
    end_date: str,
    store: str | None = None,
) -> dict[str, FluxData]:
    """Get flux data and add to dict."""
    flux_dict = {}

    for source in sources:
        logging.Logger.disabled = True  # suppress confusing OpenGHG warnings

        try:
            start_date_flux = adjust_flux_start_date(start_date, species, source, domain, store)

            flux_data = get_flux(
                species=species,
                domain=domain,
                source=source,
                start_date=start_date_flux,
                end_date=end_date,
                store=store,
            )

            # fix to prevent empty time coordinate:
            if len(flux_data.data.time) == 0:
                raise SearchError

        except SearchError:
            print(f"No flux data found between {start_date} and {end_date}.")
            print(f"Searching for flux data from before {start_date}.")

            # re-try without start date
            try:
                flux_data = get_flux(
                    species=species,
                    domain=domain,
                    source=source,
                    start_date=None,
                    end_date=end_date,
                    store=store,
                )
            except SearchError as e:
                raise SearchError(f"No flux data found before {start_date}") from e
            else:
                flux_data.data = flux_data.data.isel(time=[-1])  # select the last time step
                print(f"Using flux data from {str(flux_data.data.time.values[0]).split(':')[0]}.")
                flux_data.data = flux_data.data.assign_coords(
                    time=[pd.to_datetime(start_date)]
                )  # set time to start_date

        logging.Logger.disabled = False  # resume confusing OpenGHG warnings

        # try to guess flux time period
        # PARIS post-processing uses the time period of the flux
        time_period = pd.to_datetime(end_date) - pd.to_datetime(start_date)

        # check number of days, with extra day at start and end for buffer
        if time_period.days in (27, 28, 29, 30, 31, 32):
            inferred_time_period_str = "monthly"
        elif time_period.days in (364, 365, 366, 367):
            inferred_time_period_str = "yearly"
        else:
            inferred_time_period_str = "other"

        existing_time_period_str = flux_data.data.attrs.get("time_period", "")

        if (
            ("year" in inferred_time_period_str and "year" in existing_time_period_str.lower())
            or ("month" in inferred_time_period_str and "month" in existing_time_period_str.lower())
            or (inferred_time_period_str == "other")
        ):
            flux_data.data.flux.attrs["time_period"] = existing_time_period_str
        elif "month" in existing_time_period_str.lower():
            logger.warning("Monthly flux detected, but inversion period is {time_period.days} days. Setting flux time_period to 'monthly'.")
            flux_data.data.flux.attrs["time_period"] = existing_time_period_str
        else:
            flux_data.data.flux.attrs["time_period"] = inferred_time_period_str

        # add flux data to result dict
        flux_dict[source] = flux_data

    # cast to float32 to avoid up-casting H matrix
    for v in flux_dict.values():
        if v.data.flux.dtype != "float32":
            v.data["flux"] = v.data.flux.astype("float32")

    return flux_dict


# Obs data
def get_obs_data(
    site: str,
    species: str,
    inlet: str | None,
    start_date: str,
    end_date: str,
    domain: str | None = None,
    platform : str | None = None,
    satellite : str | None = None,
    max_level : int | None = None,
    data_level: str | None = None,
    average: str | None = None,
    instrument: str | None = None,
    calibration_scale: str | None = None,
    stores: str | None | Iterable[str | None] = None,
    keep_variables: list | None = None,
) -> ObsData | None:
    """Try to retrieve obs. data from listed stores."""

    if platform == "satellite":
        if max_level is None:
            raise AttributeError(
                "If you are using column-based data (i.e. platform is 'satellite' or 'site-column'), you need to pass max_level"
            )
            
    if stores is None or isinstance(stores, str):
        stores = [stores]

    for store in stores:
        try:
            if platform == "satellite":
                # current convention: for satellite data, the site name
                # has format satellitename-obs_region
                # or format satellitename-obs_region-selection
                split_site_name = site.split("-")
                satellite = split_site_name[0]
                obs_region = split_site_name[1]
                if len(split_site_name) == 3:
                    selection = split_site_name[2]
                else:
                    selection = None

                obs_data = get_obs_column(
                    species=species,
                    max_level=max_level,
                    satellite=satellite,
                    platform = "satellite", 
                    domain=domain,
                    selection=selection,
                    start_date=start_date,
                    end_date=end_date,
                    store=store,
                )
            else:
                obs_data = get_obs_surface(
                    site=site,
                    species=species.lower(),
                    inlet=inlet,
                    start_date=start_date,
                    end_date=end_date,
                    icos_data_level=data_level,
                    average=average,
                    instrument=instrument,
                    calibration_scale=calibration_scale,
                    store=store,
                    keep_variables=keep_variables,
                )
        except SearchError:
            print(
                f"\nNo obs data found for {site} with inlet {inlet} and instrument {instrument} in store {store}."
            )
            continue  # skip this site
        except AttributeError:
            print(f"\nNo data found for {site} between {start_date} and {end_date} in store {store}.")
            continue  # skip this site
        else:
            if obs_data is None or obs_data.data.sizes["time"] == 0:
                print(f"\nNo data found for {site} between {start_date} and {end_date} in store {store}.")
                continue  # skip this site
            else:
                return obs_data
    return None


# Footprints
def _convert_inlets_to_float(inlets):
    return np.array(list(map(lambda x: float(x[:-1]), inlets)))


def get_fp_indexer(obs: xr.Dataset, fp: xr.Dataset, averaging_period: str):
    obs_idx = obs.indexes["time"]
    fp_idx = fp.indexes["time"]

    period = pd.Timedelta(averaging_period)
    tol = period / 2  # type: ignore

    fp_reidx = fp_idx.get_indexer(obs_idx, method="nearest", tolerance=tol)
    int_idx = pd.IntervalIndex.from_arrays(fp_idx[fp_reidx], fp_idx[fp_reidx] + period, closed="left")
    return pd.DatetimeIndex([t for t in fp_idx if int_idx.contains(t).any()])


def get_footprint_to_match(
    obs: ObsData,
    domain: str,
    model: str | None = None,
    platform : str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    met_model: str | None = None,
    fp_species: str | None = None,
    fp_height: str | None = None,
    store: str | None = None,
    averaging_period: str | None = None,
    tolerance: float = 10.0,
) -> FootprintData:
    site = obs.metadata["site"]
    species = fp_species or obs.metadata.get("species", "inert")
    if store is None:
        store = Path(obs.metadata.get("object_store", "")).name or "user"
    start_date = start_date or obs._start_date
    end_date = end_date or obs._end_date

    # get available footprint heights
    met_model = met_model or "not_set"  # replace None with 'not_set'
    fp_kwargs = {
        "site": site,
        "species": species,
        "domain": domain,
        "model": model,
        "met_model": met_model,
        "store": store,
        "start_date": start_date,
        "end_date": end_date,
    }

    if platform == "satellite":
        # current convention: for satellite data, the site name
        # has format satellitename-obs_region
        # or format satellitename-obs_region-selection
        split_site_name = site.split("-")
        satellite = split_site_name[0]
        obs_region = split_site_name[1]
        if len(split_site_name) == 3:
            selection = split_site_name[2]
        else:
            selection = None

        # get available footprint heights
        fp_kwargs = {
            "domain": domain,
            "satellite": satellite,
            "obs_region": obs_region,
            "inlet": fp_height,
            "model": model,
            "met_model": met_model,
            "store": store,
            "start_date": start_date,
            "end_date": end_date,
        }

    results = search_footprints(**fp_kwargs)

    # check that we got results with inlet values
    # Note: results `search_footprints` should always have inlet values
    # so the second check shouldn't be necessary...
    if results.results.empty or "inlet" not in results.results:
        raise SearchError(f"No footprints found for search terms {fp_kwargs}")

    fp_heights_strs = list(results.results.inlet.unique())
    fp_heights = _convert_inlets_to_float(fp_heights_strs)

    # special case: only a single inlet height
    if "inlet" not in obs.data.data_vars:
        inlet = float(obs.metadata["inlet"][:-1])

        if np.min(np.abs(fp_heights - inlet)) > tolerance:
            raise SearchError("No footprints found with inlet heights matching given obs.")

        fp_height_idx = np.argmin(np.abs(fp_heights - inlet))
        fp_height = fp_heights_strs[fp_height_idx]
        return get_footprint(**fp_kwargs, inlet=fp_height)

    # get inlet values
    inlets = obs.data.inlet.values

    # match inlets to fp heights
    distances = np.abs(inlets.reshape((-1, 1)) - fp_heights.reshape((1, -1)))
    inlets_to_heights = np.argmin(distances, axis=1)

    # check tolerance
    inlet_tolerance_passed = np.min(distances, axis=1) <= tolerance
    inlet_tolerance_failed = ~inlet_tolerance_passed
    if (s := np.sum(inlet_tolerance_failed)) > 0:
        logger.warning(
            f"For site {site}: {s} times where obs. inlet height was not within {tolerance}m of a footprint height."
        )
    inlets_to_heights = inlets_to_heights[inlet_tolerance_passed]

    # footprint heights to load
    matched_fp_heights = [fp_heights_strs[i] for i in np.unique(inlets_to_heights)]
    footprints = []

    for fp_height in matched_fp_heights:
        fp_data = get_footprint(**fp_kwargs, inlet=fp_height)
        if fp_data.data.time.size > 0:
            footprints.append(fp_data)

    if not footprints:
        raise SearchError("No footprints found with inlet heights matching given obs.")

    # select footprints to match inlets
    # note: we need to take into account the difference between the obs frequency
    # and the footprint frequency
    try:
        averaging_period = averaging_period or obs.metadata["averaged_period_str"]
    except KeyError:
        if "sampling_period" in obs.metadata and "sampling_period_unit" in obs.metadata:
            averaging_period = f"{obs.metadata['sampling_period']}{obs.metadata['sampling_period_unit']}"
        else:
            raise ValueError("`averaging_period` could not be inferred from ObsData; please provide a value.")

    # select times from footprints to match with obs
    for i, fp in zip(np.unique(inlets_to_heights), footprints):
        i_idx = np.where(inlets_to_heights == i)[0]
        fp_idx = get_fp_indexer(obs.data.isel(time=i_idx), fp.data, averaging_period=averaging_period)
        fp.data = fp.data.sel(time=fp_idx)

    # make FootprintData to return
    metadata = footprints[0].metadata

    if len(footprints) > 1:
        metadata["inlet"] = "varies"
        metadata["height"] = "varies"

    data = xr.concat([fp.data for fp in footprints], dim="time").sortby("time")

    return FootprintData(data=data, metadata=metadata)


def get_footprint_data(
    domain: str,
    start_date: str,
    end_date: str,
    model: str | None,
    met_model: str | None,
    fp_species: str | None,
    fp_height: str | None,
    site: str | None = None,
    platform : str | None = None,
    averaging_period: str | None = None,
    obs_data: ObsData | None = None,
    stores: str | None | Iterable[str | None] = None,
) -> FootprintData | None:
    """Try to retrieve Footprint data from given stores.

    If `fp_height` is 'auto', then `get_footprint_to_match`
    is used to search for a footprint matching the given ObsData.
    Otherwise, `get_footprint` is used.
    """
    # if fp_height is 'auto', use `get_footprint_to_match`
    # otherwise, use `get_footprint`
    satellite = None
    obs_region = None
    try:
        if fp_height == "auto":
            if obs_data is None:
                raise ValueError("If `fp_height` is 'auto', you must provide `obs_data`.")

            def get_func(store):
                return get_footprint_to_match(
                    obs_data,
                    domain=domain,
                    start_date=start_date,
                    end_date=end_date,
                    model=model,
                    met_model=met_model,
                    store=store,
                    fp_species=fp_species,
                    averaging_period=averaging_period,
                )
        elif platform=="satellite":
            # current convention: for satellite data, the site name
            # has format satellitename-obs_region
            # or format satellitename-obs_region-selection
            split_site_name = site.split("-")
            satellite = split_site_name[0]
            obs_region = split_site_name[1]
            if len(split_site_name) == 3:
                selection = split_site_name[2]
            else:
                selection = None
            def get_func(store):
                return get_footprint(domain=domain,
                    satellite=satellite,
                    obs_region=obs_region,
                    inlet=fp_height,
                    model=model,
                    start_date=start_date,
                    end_date=end_date,
                    species=fp_species,
                    store=store)
        else:

            def get_func(store):
                return get_footprint(
                    site=site,
                    height=fp_height,
                    domain=domain,
                    model=model,
                    met_model=met_model,
                    start_date=start_date,
                    end_date=end_date,
                    store=store,
                    species=fp_species,
                )
    except SearchError:
        print(
            f"\nNo obs data found for {site} with inlet {fp_height} and in store {store}."
            )

    if stores is None or isinstance(stores, str):
        stores = [stores]

    for store in stores:
        try:
            footprint_data = get_func(store)
            if footprint_data.data.time.size == 0:
                raise SearchError
        except SearchError:
            continue  # try next store
        else:
            return footprint_data
    return None
