"""Functions for saving and loading data used for inversions.

- `_save_merged_data` saves the `fp_all` dict created by `get_data.data_processing_surface_notracer`
  to disk (either as a pickle file, netCDF, or zarr)
- `load_merged_data` restores the `fp_all` dict from these saved formats
- `make_combined_scenario` converts the `fp_all` dict into a xr.Dataset
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, cast, Literal

from numcodecs import Blosc
import numpy as np
import xarray as xr
import zarr

from openghg.dataobjects import BoundaryConditionsData, FluxData
from openghg.dataobjects._basedata import _BaseData
from openghg.util import timestamp_now
from openghg_inversions.utils import datatree_ncdf_encoding


OutputFormat = Literal["pickle", "netcdf", "zarr", "zarr.zip"]  # for internal type hints


def _make_merged_data_name(species: str, start_date: str, output_name: str) -> str:
    return f"{species}_{start_date}_{output_name}_merged-data"


def _split_suffix(merged_data_name: str) -> tuple[str, OutputFormat | None]:
    for suffix in ("pickle", "nc", "zarr", "zarr.zip"):
        if merged_data_name.endswith("." + suffix):
            if suffix == "nc":
                return merged_data_name.removesuffix("." + suffix), "netcdf"
            return merged_data_name.removesuffix("." + suffix), suffix
    return merged_data_name, None


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

    If `merged_data_name` ends with one of "pickle", "nc", "zarr", or "zarr.zip", the output format
    will be set accordingly. Otherwise, the output format defaults to zipped zarr store. If zarr is
    not installed, then netCDF is used.

    The output can be saved to a pickle file, but this isn't
    recommended because data can only be unpickled reliably with the exact same environment that created
    the pickle.

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

    # if suffix corresponds to an output format, strip the suffix and set the output
    # format accordingly
    merged_data_name, suffix = _split_suffix(merged_data_name)
    output_format = suffix or output_format

    merged_data_dir = Path(merged_data_dir)

    if not merged_data_dir.exists():
        merged_data_dir.mkdir(parents=True)

    # write to specified output
    if output_format == "pickle":
        with open(merged_data_dir / (merged_data_name + ".pickle"), "wb") as f:
            pickle.dump(fp_all, f)
    elif output_format in {"netcdf", "zarr", "zarr.zip"}:
        dt = fp_all_to_datatree(fp_all, netcdf_safe_attrs=(output_format == "netcdf"))
        dt = clear_datatree_encoding(dt)
        print(dt)  # TODO: remove, for debugging
        if "zarr" in output_format:
            # make sure chunks are reasonable and uniform
            dt = dt.chunk({"time": 600})
            dt = dt.map_over_datasets(
                lambda x: xr.unify_chunks(x)[0]
            )  # unify_chunks returns a tuple, select first item

            assert isinstance(dt, xr.DataTree)  # narrow type since the previous operation could return tuple

            # update encoding
            comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
            encoding = datatree_compression_encoding(dt, comp)

            if output_format == "zarr":
                dt.to_zarr(merged_data_dir / (merged_data_name + ".zarr"), mode="w", encoding=encoding)
            else:
                with zarr.ZipStore(merged_data_dir / (merged_data_name + ".zarr.zip"), mode="w") as store:
                    dt.to_zarr(store, mode="w", encoding=encoding)
        else:
            dt.to_netcdf(merged_data_dir / (merged_data_name + ".nc"), encoding=datatree_ncdf_encoding(dt))
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

    # if suffix corresponds to an output format, strip the suffix and set the output
    # format accordingly
    merged_data_name, suffix = _split_suffix(merged_data_name)
    output_format = suffix or output_format

    if output_format is not None:
        ext = "nc" if output_format == "netcdf" else output_format
        merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
        if not merged_data_file.exists():
            raise ValueError(f"No merged data found at {merged_data_file}.")
    else:
        for ext in ["zarr.zip", "zarr", "nc", "pickle"]:
            merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
            if merged_data_file.exists():
                break
        else:
            # no `break` occurred, so no file found
            raise ValueError(err_msg)

    # load merged data
    if merged_data_file.suffix == ".pickle":
        with open(merged_data_file, "rb") as f:
            return pickle.load(f)
    elif merged_data_file.suffixes == [".zarr", ".zip"]:
        with zarr.ZipStore(merged_data_file, mode="r") as store:
            with xr.open_datatree(store, engine="zarr") as dt:  # type: ignore[arg-type, unused-ignore]
                if dt.is_leaf:
                    return fp_all_from_dataset(dt.to_dataset().load())
                return datatree_to_fp_all(dt.load())
    elif merged_data_file.suffix == ".zarr":
        with xr.open_datatree(merged_data_file, engine="zarr") as dt:
            if dt.is_leaf:
                return fp_all_from_dataset(dt.to_dataset())
            return datatree_to_fp_all(dt)
    else:
        # suffix is probably ".nc", but could be something else if name passed directly
        # try `open_dataset`
        with xr.open_datatree(merged_data_file) as dt:
            if dt.is_leaf:
                return fp_all_from_dataset(dt.to_dataset())
            return datatree_to_fp_all(dt)


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

        fp_all[site] = scenario.dropna("time", subset=["mf"])

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

        fp_all[".flux"][source] = FluxData(data=flux_ds, metadata={"data_type": "flux"})

    try:
        bc_ds = ds[bc_vars]
    except KeyError:
        pass
    else:
        if "time" not in bc_ds.dims:
            bc_ds = bc_ds.expand_dims({"time": [ds.time.min().values]})

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


# ----------------------------------------
# DataTree conversions
# ----------------------------------------


def openghg_data_to_dataset(openghg_data: _BaseData, netcdf_safe_attrs: bool = False) -> xr.Dataset:
    ds = openghg_data.data

    if netcdf_safe_attrs:
        ds.attrs["_openghg_metadata"] = json.dumps(openghg_data.metadata)
    else:
        ds.attrs["_openghg_metadata"] = openghg_data.metadata
    return ds


def dataset_to_flux_data(ds: xr.Dataset) -> FluxData:
    if "flux" not in ds.data_vars:
        raise ValueError("Dataset must have `flux` data variable to convert to FluxData.")
    ds = ds.copy()
    metadata = ds.attrs.pop("_openghg_metadata")

    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return FluxData(metadata=metadata, data=ds)


def dataset_to_bc_data(ds: xr.Dataset) -> BoundaryConditionsData:
    if any(f"vmr_{d}" not in ds.data_vars for d in "nesw"):
        raise ValueError(
            "Dataset must have `vmr_n`, `vmr_e`, `vmr_s`, `vmr_w` data "
            "variables to convert to BoundaryConditionsData."
        )
    ds = ds.copy()
    metadata = ds.attrs.pop("_openghg_metadata")

    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return BoundaryConditionsData(metadata=metadata, data=ds)


def flux_dict_to_datatree(flux_dict: dict[str, FluxData], netcdf_safe_attrs: bool = False) -> xr.DataTree:
    dt_dict = {k: openghg_data_to_dataset(v, netcdf_safe_attrs) for k, v in flux_dict.items()}
    return xr.DataTree.from_dict(dt_dict)


def datatree_to_flux_dict(dt: xr.DataTree) -> dict[str, FluxData]:
    return {str(k): dataset_to_flux_data(v.to_dataset()) for k, v in dt.items()}


def fp_all_to_datatree(fp_all: dict, netcdf_safe_attrs: bool = False) -> xr.DataTree:
    dt_dict: dict[str, xr.Dataset | xr.DataTree] = {}
    scenario_dict = {}
    dt_attrs = {}

    if ".flux" in fp_all:
        dt_dict["fluxes"] = flux_dict_to_datatree(fp_all[".flux"], netcdf_safe_attrs)

    for k, v in fp_all.items():
        if k == ".flux":
            continue
        if isinstance(v, BoundaryConditionsData):
            dt_dict[k.removeprefix(".")] = openghg_data_to_dataset(v, netcdf_safe_attrs)
        elif not k.startswith(".") and isinstance(v, xr.Dataset):
            scenario_dict[k] = v
        else:
            dt_attrs[k] = v

    dt_dict["scenarios"] = xr.DataTree.from_dict(scenario_dict)

    dt = xr.DataTree.from_dict(dt_dict)
    dt.attrs = dt_attrs

    return dt


def datatree_to_fp_all(dt: xr.DataTree) -> dict:
    fp_all = {}

    if "fluxes" in dt:
        fp_all[".flux"] = datatree_to_flux_dict(dt.fluxes)

    if "bc" in dt:
        fp_all[".bc"] = dataset_to_bc_data(dt.bc.to_dataset())

    for k, v in dt.scenarios.items():
        fp_all[str(k)] = v.to_dataset()

    fp_all.update({str(k): v for k, v in dt.attrs.items()})

    return fp_all


def datatree_compression_encoding(dt: xr.DataTree, compressor: Blosc) -> dict:
    """Creating encoding dictionary for saving DataTree to zarr."""
    encoding = defaultdict(dict)

    for g in dt.groups:
        if not dt[g].data_vars:
            continue
        for dv in dt[g].data_vars:
            encoding[g][dv] = {"compressor": compressor, "compressors": (compressor,)}

    return encoding


def clear_datatree_encoding(dt: xr.DataTree) -> xr.DataTree:
    """Clean encoding attribute of variables to avoid issues when writing."""
    result = dt.copy()

    for g in result.groups:
        for v in result[g].data_vars.values():
            v.encoding = {}

        for c in result[g].coords.values():
            c.encoding = {}

    return result
