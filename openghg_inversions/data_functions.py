
import logging
from typing import cast, Literal, overload

import numpy as np
import pandas as pd
import xarray as xr

from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData
from openghg.analyse import ModelScenario


logger = logging.getLogger("openghg_inversions.data_functions")
logger.setLevel(logging.INFO)  # Have to set level for logger as well as handler


def fp_x_flux(fp: FootprintData, flux: FluxData | dict[str, FluxData], units: float | None = None) -> xr.DataArray:
    """Calculate array with footprints * flux.

    TODO: make this a function that accepts a model scenario and source?
    """
    ms = ModelScenario(footprint=fp, flux=flux)
    ms.scenario = fp.data  # HACK

    if ms.species.lower() == "co2":
        result = ms._calc_modelled_obs_HiTRes(output_TS=False, output_fpXflux=True)
    else:
        result = ms._calc_modelled_obs_integrated(output_TS=False, output_fpXflux=True)

    if units is not None:
        result /= units

    return result


def fp_x_bc(fp: FootprintData, bc: BoundaryConditionsData, units: float | None = None) -> xr.Dataset:
    pass


def nesw_bc(fp: FootprintData, bc: BoundaryConditionsData, units: float | None = None) -> xr.DataArray:
    """This only works for inert footprints, and only applies the NESW boundary conditions."""
    directions = ["n", "e", "s", "w"]

    fp_bc = fp.data[[f"particle_locations_{x}" for x in directions]].rename({f"particle_locations_{x}": x for x in directions})
    bc_aligned = bc.data.rename({f"vmr_{x}": x for x in directions}).reindex_like(fp_bc, method="ffill")

    result = (fp_bc * bc_aligned).sum(["lat", "lon", "height"]).to_dataarray(dim="region", name="h_bc")

    if units is not None:
        result /= units

    return result


@overload
def align_obs_and_other(
    obs: xr.DataArray,
    other: xr.DataArray,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray]: ...


@overload
def align_obs_and_other(
    obs: xr.Dataset,
    other: xr.DataArray,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.Dataset, xr.DataArray]: ...


@overload
def align_obs_and_other(
    obs: xr.DataArray,
    other: xr.Dataset,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.DataArray, xr.Dataset]: ...


@overload
def align_obs_and_other(
    obs: xr.Dataset,
    other: xr.Dataset,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset]: ...


def align_obs_and_other(
    obs: xr.DataArray | xr.Dataset,
    other: xr.DataArray | xr.Dataset,
    resample_to: Literal["obs", "other", "coarsest"] | str = "coarsest",
    platform: str | None = None,
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]:
    """
    Slice and resample obs and footprint data to align along time

    This slices the date to the smallest time frame
    spanned by both the footprint and obs, using the sliced start date
    The time dimension is resampled based on the resample_to input using the mean.
    The resample_to options are:
     - "coarsest" - resample to the coarsest resolution between obs and footprints
     - "obs" - resample to observation data frequency
     - "footprint" - resample to footprint data frequency
     - a valid resample period e.g. "2H"

    Args:
        resample_to: Resample option to use: either data based or using a valid pandas resample period.
        platform: Observation platform used to decide whether to resample

    Returns:
        tuple: Two xarray DataArrays with aligned time dimensions
    """
    resample_keyword_choices = ("obs", "other", "coarsest")

    # Check whether resample has been requested by specifying a specific period rather than a keyword
    if resample_to in resample_keyword_choices:
        force_resample = False
    else:
        force_resample = True

    if platform is not None:
        platform = platform.lower()
        # Do not apply resampling for "satellite" (but have re-included "flask" for now)
        if platform == "satellite":
            return obs, other

    # Whether sampling period is present or we need to try to infer this
    infer_sampling_period = False
    # Get the period of measurements in time
    obs_attributes = obs.attrs
    if "averaged_period" in obs_attributes:
        obs_data_period_s = float(obs_attributes["averaged_period"])
    elif "sampling_period" in obs_attributes:
        sampling_period = obs_attributes["sampling_period"]
        if sampling_period == "NOT_SET":
            infer_sampling_period = True
        elif sampling_period == "multiple":
            # If we have a varying sampling_period, make sure we always resample to footprint
            obs_data_period_s = 1.0
        else:
            obs_data_period_s = float(sampling_period)
    elif "sampling_period_estimate" in obs_attributes:
        estimate = obs_attributes["sampling_period_estimate"]
        logger.warning(f"Using estimated sampling period of {estimate}s for observational data")
        obs_data_period_s = float(estimate)
    else:
        infer_sampling_period = True

    if infer_sampling_period:
        # Attempt to derive sampling period from frequency of data
        obs_data_period_s = np.nanmedian((obs.time.data[1:] - obs.time.data[0:-1]) / 1e9).astype("float32")

        obs_data_period_s_min = np.diff(obs.time.data).min() / 1e9
        obs_data_period_s_max = np.diff(obs.time.data).max() / 1e9

        max_diff = (obs_data_period_s_max - obs_data_period_s_min).astype(float)

        # Check if the periods differ by more than 1 second
        if max_diff > 1.0:
            raise ValueError("Sample period can be not be derived from observations")

        estimate = f"{obs_data_period_s:.1f}"
        logger.warning(f"Sampling period was estimated (inferred) from data frequency: {estimate}s")
        obs.attrs["sampling_period_estimate"] = estimate

    # TODO: Check regularity of the data - will need this to decide is resampling
    # is appropriate or need to do checks on a per time point basis

    obs_data_period_ns = obs_data_period_s * 1e9
    obs_data_timeperiod = pd.Timedelta(obs_data_period_ns, unit="ns")

    # Derive the footprints period from the frequency of the data
    other_data_period_ns = np.nanmedian((other.time.data[1:] - other.time.data[0:-1]).astype("int64"))
    other_data_timeperiod = pd.Timedelta(other_data_period_ns, unit="ns")

    # If resample_to is set to "coarsest", check whether "obs" or "footprint" have lower resolution
    if resample_to == "coarsest":
        if obs_data_timeperiod >= other_data_timeperiod:
            resample_to = "obs"
        elif obs_data_timeperiod < other_data_timeperiod:
            resample_to = "footprint"

    # Here we want timezone naive pd.Timestamps
    # Add sampling period to end date to make sure resample includes these values when matching
    obs_startdate = pd.Timestamp(obs.time[0].values)
    obs_enddate = pd.Timestamp(obs.time[-1].values) + obs_data_timeperiod
    footprint_startdate = pd.Timestamp(other.time[0].values)
    footprint_enddate = pd.Timestamp(other.time[-1].values) + other_data_timeperiod

    start_date = max(obs_startdate, footprint_startdate)
    end_date = min(obs_enddate, footprint_enddate)

    # Ensure lower range is covered for obs
    start_obs_slice = start_date - pd.Timedelta("1ns")
    # Ensure extra buffer is added for footprint based on fp timeperiod.
    # This is to ensure footprint can be forward-filled to obs (in later steps)
    start_other_slice = start_date - (other_data_timeperiod - pd.Timedelta("1ns"))
    # Subtract very small time increment (1 nanosecond) to make this an exclusive selection
    end_slice = end_date - pd.Timedelta("1ns")

    obs = obs.sel(time=slice(start_obs_slice, end_slice))
    other = other.sel(time=slice(start_other_slice, end_slice))

    if obs.time.size == 0 or other.time.size == 0:
        raise ValueError("Obs data and Footprint data don't overlap")
    # Only non satellite datasets with different periods need to be resampled
    timeperiod_diff_s = np.abs(obs_data_timeperiod - other_data_timeperiod).total_seconds()
    tolerance = 1e-9  # seconds

    if timeperiod_diff_s >= tolerance or force_resample:
        offset = pd.Timedelta(hours=start_date.hour + start_date.minute / 60.0 + start_date.second / 3600.0)
        offset = cast(pd.Timedelta, offset)

        if resample_to == "obs":
            resample_period = str(round(obs_data_timeperiod / np.timedelta64(1, "h"), 5)) + "H"
            other = other.resample(indexer={"time": resample_period}, offset=offset).mean()

        elif resample_to == "footprint":
            resample_period = str(round(other_data_timeperiod / np.timedelta64(1, "h"), 5)) + "H"
            obs = obs.resample(indexer={"time": resample_period}, offset=offset).mean()

        else:
            resample_period = resample_to
            other = other.resample(indexer={"time": resample_period}, offset=offset).mean()
            obs = obs.resample(indexer={"time": resample_period}, offset=offset).mean()

    return obs, other
