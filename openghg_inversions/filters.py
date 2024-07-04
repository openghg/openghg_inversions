import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.utils import combine_datasets

def filtering(datasets_in, filters, keep_missing=False):
    """
    Applies time filtering to entire dataset.
    Filters supplied in a list and then applied in order.
    For example if you wanted a daily, daytime average, you could do this:

        datasets_dictionary = filtering(datasets_dictionary,
                                    ["daytime", "daily_median"])

    The order of the filters reflects the order they are applied, so for
    instance when applying the "daily_median" filter if you only wanted
    to look at daytime values the filters list should be
    ["daytime","daily_median"]
    -----------------------------------
    Args:
      datasets_in (dict):
        Output from ModelScenario.footprints_merge(). Dictionary of datasets.
      filters (list):
        Filters to apply to the datasets.
          All options are:
            "daytime"           : selects data between 1100 and 1500 local solar time
            "daytime9to5"       : selects data between 0900 and 1700 local solar time
            "nighttime"         : Only b/w 23:00 - 03:00 inclusive
            "noon"              : Only 12:00 fp and obs used
            "daily_median"      : calculates the daily median
            "pblh_min"          : Only keeps times when pblh is > threshold (default 200m)
            "pblh_inlet_diff"   : Only keeps times when inlet is at least a threshold (default 50m) below the pblh
            "local_influence"   : Only keep times when localness is low
            "six_hr_mean"       :
            "local_lapse"       :
      keep_missing (bool) : Whether to reindex to retain missing data.

    Returns:
       Same format as datasets_in : Datasets with filters applied.
    -----------------------------------
    """
    # Get list of sites
    sites = [key for key in list(datasets_in.keys()) if key[0] != "."]

    # Put the filters in a dict of list
    if not isinstance(filters, dict):
        if not isinstance(filters, list):
            filters = [filters]
        filters = {site: filters for site in sites}
    else:
        for site, filt in filters.items():
            if filt is not None and not isinstance(filt, list):
                filters[site] = [filt]

    # Check that filters are defined for all sites
    tmp = [(site in filters) for site in sites]
    if not all(tmp):
        raise ValueError(f"Missing entry for sites {np.array(sites)[~np.array(tmp)]} in filters.")


    datasets = datasets_in.copy()

    def local_solar_time(dataset):
        """
        Returns hour of day as a function of local solar time
        relative to the Greenwich Meridian.
        """
        sitelon = dataset.release_lon.values[0]
        # convert lon to [-180,180], so time offset is negative west of 0 degrees
        if sitelon > 180:
            sitelon = sitelon - 360.0
        dataset["time"] = dataset.time + pd.Timedelta(minutes=float(24 * 60 * sitelon / 360.0))
        hours = dataset.time.to_pandas().index.hour
        return hours

    def local_ratio(dataset):
        """
        Calculates the local ratio in the surrounding grid cells
        """
        dlon = dataset.lon[1].values - dataset.lon[0].values
        dlat = dataset.lat[1].values - dataset.lat[0].values
        local_sum = np.zeros((len(dataset.mf)))

        for ti in range(len(dataset.mf)):
            release_lon = dataset.release_lon[ti].values
            release_lat = dataset.release_lat[ti].values
            wh_rlon = np.where(abs(dataset.lon.values - release_lon) < dlon / 2.0)
            wh_rlat = np.where(abs(dataset.lat.values - release_lat) < dlat / 2.0)
            if np.any(wh_rlon[0]) and np.any(wh_rlat[0]):
                local_sum[ti] = np.sum(
                    dataset.fp[
                        wh_rlat[0][0] - 2 : wh_rlat[0][0] + 3, wh_rlon[0][0] - 2 : wh_rlon[0][0] + 3, ti
                    ].values
                ) / np.sum(dataset.fp[:, :, ti].values)
            else:
                local_sum[ti] = 0.0

        return local_sum

    # Filter functions
    def daily_median(dataset, keep_missing=False):
        """Calculate daily median"""
        if keep_missing:
            return dataset.resample(indexer={"time": "1D"}).median()
        else:
            return dataset.resample(indexer={"time": "1D"}).median().dropna(dim="time")

    def six_hr_mean(dataset, keep_missing=False):
        """Calculate six-hour median"""
        if keep_missing:
            return dataset.resample(indexer={"time": "6H"}).mean()
        else:
            return dataset.resample(indexer={"time": "6H"}).mean().dropna(dim="time")

    def daytime(dataset, site, keep_missing=False):
        """Subset during daytime hours (11:00-15:00)"""
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h >= 11 and h <= 15]

        if keep_missing:
            dataset_temp = dataset[dict(time=ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def daytime9to5(dataset, site, keep_missing=False):
        """Subset during daytime hours (9:00-17:00)"""
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h >= 9 and h <= 17]

        if keep_missing:
            dataset_temp = dataset[dict(time=ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def nighttime(dataset, site, keep_missing=False):
        """Subset during nighttime hours (23:00 - 03:00)"""
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h >= 23 or h <= 3]

        if keep_missing:
            dataset_temp = dataset[dict(time=ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def noon(dataset, site, keep_missing=False):
        """Select only 12pm data"""
        hours = local_solar_time(dataset)
        ti = [i for i, h in enumerate(hours) if h == 12]

        if keep_missing:
            dataset_temp = dataset[dict(time=ti)]
            dataset_out = dataset_temp.reindex_like(dataset)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def local_influence(dataset, site, keep_missing=False):
        """
        Subset for times when local influence is below threshold.
        Local influence expressed as a fraction of the sum of entire footprint domain.
        """
        if not dataset.filter_by_attrs(standard_name="local_ratio"):
            lr = local_ratio(dataset)
        else:
            lr = dataset.local_ratio

        pc = 0.1
        ti = [i for i, local_ratio in enumerate(lr) if local_ratio <= pc]
        if keep_missing is True:
            mf_data_array = dataset.mf
            dataset_temp = dataset.drop("mf")

            dataarray_temp = mf_data_array[dict(time=ti)]

            mf_ds = xr.Dataset(
                {"mf": (["time"], dataarray_temp)}, coords={"time": (dataarray_temp.coords["time"])}
            )

            dataset_out = combine_datasets(dataset_temp, mf_ds, method=None)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def pblh_min(dataset, pblh_threshold=200.0, keep_missing=False):
        """
        Subset for times when the PBLH is greater than 200m.
        """
        pblh_da = dataset.PBLH if "PBLH" in dataset.data_vars else dataset.atmosphere_boundary_layer_thickness

        ti = [
            i for i, pblh in enumerate(pblh_da) if pblh > pblh_threshold
        ]

        if keep_missing is True:
            mf_data_array = dataset.mf
            dataset_temp = dataset.drop("mf")

            dataarray_temp = mf_data_array[dict(time=ti)]

            mf_ds = xr.Dataset(
                {"mf": (["time"], dataarray_temp)}, coords={"time": (dataarray_temp.coords["time"])}
            )

            dataset_out = combine_datasets(dataset_temp, mf_ds, method=None)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def pblh_inlet_diff(dataset, diff_threshold=50.0, keep_missing=False):
        """
        Subset for times when observations are taken at a height of less than 50 m below the PBLH.
        """
        if "inlet_height_magl" in dataset.attrs:
            inlet_height = float(dataset.inlet_height_magl)
        elif "inlet" in dataset.attrs:
            m = re.search(r"\d+", dataset.attrs["inlet"])
            if m is not None:
                inlet_height = float(m.group(0))
        else:
            raise ValueError("Could not find inlet height from `inlet_height_magl` or `inlet` dataset attributes.")

        pblh_da = dataset.PBLH if "PBLH" in dataset.data_vars else dataset.atmosphere_boundary_layer_thickness

        ti = [
            i for i, pblh in enumerate(pblh_da) if inlet_height < pblh - diff_threshold
        ]

        if keep_missing is True:
            mf_data_array = dataset.mf
            dataset_temp = dataset.drop("mf")

            dataarray_temp = mf_data_array[dict(time=ti)]

            mf_ds = xr.Dataset(
                {"mf": (["time"], dataarray_temp)}, coords={"time": (dataarray_temp.coords["time"])}
            )

            dataset_out = combine_datasets(dataset_temp, mf_ds, method=None)
            return dataset_out
        else:
            return dataset[dict(time=ti)]

    def pblh(dataset, keep_missing=False):
        raise NotImplementedError("pblh is now called pblh_inlet_diff")

    filtering_functions = {
        "daily_median": daily_median,
        "daytime": daytime,
        "daytime9to5": daytime9to5,
        "nighttime": nighttime,
        "noon": noon,
        "local_influence": local_influence,
        "six_hr_mean": six_hr_mean,
        "pblh_inlet_diff": pblh_inlet_diff,
        "pblh_min": pblh_min,
        "pblh": pblh,
    }

    # Apply filtering
    for site in sites:
        if filters[site] is not None:
            for filt in filters[site]:
                n_nofilter = datasets[site].time.values.shape[0]
                if filt in ["daily_median", "six_hr_mean", "pblh_inlet_diff", "pblh_min", "pblh"]:
                    datasets[site] = filtering_functions[filt](datasets[site], keep_missing=keep_missing)
                else:
                    datasets[site] = filtering_functions[filt](datasets[site], site, keep_missing=keep_missing)
                n_filter = datasets[site].time.values.shape[0]
                n_dropped = n_nofilter - n_filter
                perc_dropped = np.round(n_dropped / n_nofilter * 100, 2)
                print(f"{filt} filter removed {n_dropped} ({perc_dropped} %) obs at site {site}")

    return datasets
