import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Literal

import numpy as np
import xarray as xr

from openghg.dataobjects import BoundaryConditionsData, FluxData
from openghg.util import timestamp_now


def _make_merged_data_name(species: str, start_date: str, output_name: str) -> str:
    return f"{species}_{start_date}_{output_name}_merged-data"


def _save_merged_data(
    fp_all: dict,
    merged_data_dir: str | Path,
    species: str | None = None,
    start_date: str | None = None,
    output_name: str | None = None,
    merged_data_name: str | None = None,
    output_format: Literal["pickle", "netcdf", "zarr", "zarr.zip"] = "zarr.zip",
) -> None:
    """Save `fp_all` dictionary to `merged_data_dir`.

    The name of the pickle file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    The default output format is a zarr store. If zarr is not installed, then netCDF is used.
    Alternatively, "pickle" can be specified.

    Args:
        fp_all: dictionary of merged data to save
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.
        output_format: format to save merged data to (default: "zarr").

    Returns:
        None
    """
    if merged_data_name is None:
        if any(arg is None for arg in [species, start_date, output_name]):
            raise ValueError(
                "If `merged_date_name` isn't given, then "
                "`species`, `start_date`, and `output_name` must be provided."
            )
        merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore

    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    # write to specified output
    if output_format == "pickle":
        with open(merged_data_dir / (merged_data_name + ".pickle"), "wb") as f:
            pickle.dump(fp_all, f)
    elif output_format in {"netcdf", "zarr", "zarr.zip"}:
        ds = make_combined_scenario(fp_all)

        if "zarr" in output_format:
            try:
                import zarr
            except ModuleNotFoundError:
                # zarr not found
                ds.to_netcdf(merged_data_dir / (merged_data_name + ".nc"))
            else:
                if output_format == "zarr":
                    ds.to_zarr(merged_data_dir / (merged_data_name + ".zarr"), mode="w")
                else:
                    with zarr.ZipStore(merged_data_dir / (merged_data_name + ".zarr.zip"), mode="w") as store:
                        ds.to_zarr(store, mode="w")
        else:
            ds.to_netcdf(merged_data_dir / (merged_data_name + ".nc"))
    else:
        raise ValueError(
            f"Output format should be 'pickle', 'netcdf', 'zarr', or 'zarr.zip'. Given '{output_format}'."
        )


def load_merged_data(
    merged_data_dir: str | Path,
    species: str | None = None,
    start_date: str | None = None,
    output_name: str | None = None,
    merged_data_name: str | None = None,
    output_format: Literal["pickle", "netcdf", "zarr", "zarr.zip"] | None = None,
) -> dict:
    """Load `fp_all` dictionary from a file in `merged_data_dir`.

    The name of the pickle file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    This function tries to automatically find a compatible format of merged data, if a format is not specified.
    First, it checks for data in "zarr" format, then in netCDF, and finally in pickle.

    Args:
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.
        output_format: format of data to load (if not specified, this will be inferred).

    Returns:
        `fp_all` dictionary
    """
    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    if merged_data_name is not None:
        err_msg = (
            f"No merged data with file name {merged_data_name} in merged data directory {merged_data_dir}"
        )
    elif any(arg is None for arg in [species, start_date, output_name]):
        raise ValueError(
            "If `merged_date_name` isn't given, then "
            "`species`, `start_date`, and `output_name` must be provided."
        )
    else:
        merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore
        err_msg = (
            f"No merged data for species {species}, start date {start_date}, and "
            f"output name {output_name} found in merged data directory {merged_data_dir}"
        )

    if output_format is not None:
        ext = output_format
        merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
        if not merged_data_file.exists():
            raise ValueError(f"No merged data found at {merged_data_file}.")
    else:
        for ext in ["zarr.zip", "zarr", "nc", "pickle"]:
            # skip "zarr" if zarr not installed...
            if "zarr" in ext:
                try:
                    import zarr
                except ModuleNotFoundError:
                    continue

            merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
            if merged_data_file.exists():
                break
        else:
            # no `break` occurred, so no file found
            raise ValueError(err_msg)

    # load merged data
    if merged_data_file.suffix == "pickle":
        with open(merged_data_file, "rb") as f:
            fp_all = pickle.load(f)
    else:
        if merged_data_file.suffixes == [".zarr", ".zip"]:
            import zarr

            with zarr.ZipStore(merged_data_file, mode="r") as store:
                ds = xr.open_zarr(store).load()
        elif merged_data_file.suffix == ".zarr":
            ds = xr.open_zarr(merged_data_file)
        else:
            # suffix is probably ".nc", but could be something else if name passed directly
            # try `open_dataset`
            ds = xr.open_dataset(merged_data_file)

        fp_all = fp_all_from_dataset(ds)

    return fp_all


list_keys = [
    "site",
    "inlet",
    "instrument",
    "sampling_period",
    "sampling_period_unit",
    "averaged_period_str",
    "scale",
    "network",
    "data_owner",
    "data_owner_email",
]


def combine_scenario_attrs(attrs_list: list[dict[str, Any]], context) -> dict[str, Any]:
    """Combine attributes when concatenating scenarios from different sites.

    The `ModelScenario.scenario`s in `get_combined_scenario` have the key "scenario" added
    to their attributes as a flag so this function can process the dataset attributes and
    the data variable attributes differently.

    TODO: add 'time_period', 'high_time/spatial_resolution', 'short_lifetime', 'heights'?
        Is 'time_period' from the footprint? Need to check model scenario...

    Args:
        attrs_list: list of attributes from datasets being concatenated
        context: additional parameter supplied by concatenate (this is required/supplied by xarray)

    Returns:
        dict that will be used as attributes for concatenated dataset
    """
    single_keys = [
        "species",
        "start_date",
        "end_date",
        "model",
        "metmodel",
        "domain",
        "max_longitude",
        "min_longitude",
        "max_latitude",
        "min_latitude",
    ]

    # take attributes from first element of attrs_list if key "scenario" is not in attributes
    # this is a flag set in `get_combined_scenarios` to facilitate combining attributes
    if "scenario" not in attrs_list[0]:
        return attrs_list[0]

    # processing for scenarios
    single_attrs = {
        k: attrs_list[0].get(k, "None") for k in single_keys
    }  # NoneType can't be saved to netCDF, use string instead
    list_attrs = defaultdict(list)
    for attrs in attrs_list:
        for key in list_keys:
            list_attrs[key].append(attrs.get(key, "None"))

    list_attrs = cast(dict, list_attrs)
    list_attrs.update(single_attrs)
    list_attrs["file_created"] = str(timestamp_now())
    return list_attrs


def make_combined_scenario(fp_all: dict) -> xr.Dataset:
    """Combine scenarios and merge in fluxes and boundary conditions.

    If fluxes and boundary conditions only have one coordinate for their
    "time" dimension, then "time" will be dropped.

    Otherwise, it is assumed that the time axis for fluxes and boundary conditions
    have the same length as the time axis for the model scenarios.

    """
    # combine scenarios by site
    scenarios = [v.expand_dims({"site": [k]}) for k, v in fp_all.items() if not k.startswith(".")]

    # add flag to top level attributes to help combine scenario attributes, without combining the
    # attributes of every data variable
    for scenario in scenarios:
        scenario.attrs["scenario"] = True

    combined_scenario = xr.concat(scenarios, dim="site", combine_attrs=combine_scenario_attrs)

    # make dtype of 'site' coordinate "<U3" (little-endian Unicode string of length 3)
    combined_scenario = combined_scenario.assign_coords(site=combined_scenario.site.astype(np.dtype("<U3")))

    # concat fluxes over source before merging into combined scenario
    fluxes = [v.data.expand_dims({"source": [k]}) for k, v in fp_all[".flux"].items()]
    combined_fluxes = xr.concat(fluxes, dim="source")

    if "time" in combined_fluxes.dims and combined_fluxes.sizes["time"] == 1:
        combined_fluxes = combined_fluxes.squeeze("time")

    # merge with override in case coordinates slightly off
    # (data should already be aligned by `ModelScenario`)
    combined_scenario = combined_scenario.merge(combined_fluxes, join="override")

    # merge in boundary conditions
    if ".bc" in fp_all:
        bc = fp_all[".bc"].data
        if "time" in bc.dims and bc.sizes["time"] == 1:
            bc = bc.squeeze("time")
        bc = bc.reindex_like(combined_scenario, method="nearest")
        combined_scenario = combined_scenario.merge(bc)

    return combined_scenario


def fp_all_from_dataset(ds: xr.Dataset) -> dict:
    """Recover "fp_all" dictionary from "combined scenario" dataset.

    This is the inverse of `make_combined_scenario`, except that the attributes of the
    scenarios, fluxes, and boundary conditions may be different.

    Args:
        ds: dataset created by `make_combined_scenario`

    Returns:
        dictionary containing model scenarios keyed by site, as well as flux and boundary conditions.
    """
    fp_all = {}

    # we'll get scales as we get scenarios
    fp_all[".scales"] = {}

    # get scenarios
    bc_vars = ["vmr_n", "vmr_e", "vmr_s", "vmr_w"]

    for i, site in enumerate(ds.site.values):
        scenario = (
            ds.sel(site=site, drop=True).drop_vars(["flux", *bc_vars], errors="ignore").drop_dims("source")
        )

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = scenario.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"

            if k == "scale":
                fp_all[".scales"][site] = val
            else:
                scenario.attrs[k] = val

        fp_all[site] = scenario.dropna("time")

    # get fluxes
    fp_all[".flux"] = {}

    for i, source in enumerate(ds.source.values):
        flux_ds = (
            ds[["flux"]]  # double brackets to get dataset
            .sel(source=source, drop=True)
            .expand_dims({"time": [ds.time.min().values]})
            .transpose(..., "time")
        )

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = flux_ds.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"
            flux_ds.attrs[k] = val

        fp_all[".flux"][source] = FluxData(data=flux_ds, metadata={})

    try:
        bc_ds = ds[bc_vars]
    except KeyError:
        pass
    else:
        bc_ds = bc_ds.expand_dims({"time": [ds.time.min().values]}).transpose(..., "time")
        fp_all[".bc"] = BoundaryConditionsData(data=bc_ds, metadata={})

    species = ds.attrs.get("species", None)
    if species is not None:
        species = species.upper()
    fp_all[".species"] = species

    try:
        fp_all[".units"] = float(ds.mf.attrs.get("units", 1.0))
    except ValueError:
        # conversion to float failed
        fp_all[".units"] = 1.0

    return fp_all
