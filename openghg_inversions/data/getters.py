import logging
from pathlib import Path
from typing import Any, Iterable, cast, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg.dataobjects import ObsData, BoundaryConditionsData, FluxData, FootprintData
from openghg.retrieve import get_bc, get_flux, get_footprint, get_obs_surface, search_footprints, search_flux
from openghg.types import SearchError
from openghg.util import timestamp_now


logger = logging.getLogger(__name__)


# Flux
def adjust_flux_start_date(
    start_date: str, species: str, source: str, domain: str, store: str | None = None
) -> pd.Timestamp:
    """Adjusts the flux start_date to align with the flux data's temporal resolution."""
    flux_search = search_flux(species=species, source=source, domain=domain, store=store)
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
                flux_data.data = flux_data.data.isel(time=[-1])
                print(f"Using flux data from {str(flux_data.data.time.values[0]).split(':')[0]}.")

        logging.Logger.disabled = False  # resume confusing OpenGHG warnings

        flux_dict[source] = flux_data

    return flux_dict


# Boundary conditions data
def convert_bc_units(bc_data: BoundaryConditionsData, unit: float) -> BoundaryConditionsData:
    # Divide by trace gas species units
    # See if R+G can include this 'behind the scenes'
    bc_data.data.vmr_n.values /= unit
    bc_data.data.vmr_e.values /= unit
    bc_data.data.vmr_s.values /= unit
    bc_data.data.vmr_w.values /= unit

    return BoundaryConditionsData(
        data=bc_data.data.transpose("height", "lat", "lon", "time"),
        metadata=bc_data.metadata,
    )


# Obs data
def get_obs_data(
    site: str,
    species: str,
    inlet: str | None,
    start_date: str,
    end_date: str,
    data_level: str | None = None,
    average: str | None = None,
    instrument: str | None = None,
    calibration_scale: str | None = None,
    stores: str | None | Iterable[str | None] = None,
) -> ObsData | None:
    """Try to retrieve obs. data from listed stores."""
    if stores is None or isinstance(stores, str):
        stores = [stores]

    for store in stores:
        try:
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
            if obs_data is None:
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
    start_date: str | None = None,
    end_date: str | None = None,
    met_model: str | None = None,
    fp_species: str | None = None,
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

    met_model = met_model or "not_set"  # replace None with 'not_set'

    # get available footprint heights
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
    if (s := np.sum(inlet_tolerance_passed)) > 0:
        logger.warning(
            f"For site {site}: {s} times where obs. inlet height was not within {tolerance}m of a footprint height."
        )
        inlets_to_heights = inlets_to_heights[inlet_tolerance_passed]

    # footprint heights to load
    matched_fp_heights = [fp_heights_strs[i] for i in np.unique(inlets_to_heights)]
    footprints = []

    for fp_height in matched_fp_heights:
        fp_data = get_footprint(**fp_kwargs, inlet=fp_height)
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
    site: str,
    domain: str,
    fp_height: str | None,
    start_date: str,
    end_date: str,
    model: str | None,
    met_model: str | None,
    fp_species: str | None,
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
