"""Explicit legacy acquisition and ``fp_all`` file utilities, supported until 0.9.

``retrieve_inversion_data`` returns the historical six-tuple. Call
``_save_merged_data`` and ``load_merged_data`` explicitly to persist or reload
its mapping; modern RHIME acquisition and preparation do not use this codec.
"""

import json
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc
from openghg.dataobjects import BoundaryConditionsData, FluxData
from openghg.dataobjects._basedata import _BaseData
from openghg.util import timestamp_now

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data._provenance import (
    InputProvenance, MergedDataProvenance, selected_provenance,
)
from openghg_inversions.inversion_data._site_options import SiteOptions
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.inversion_data.acquisition import AcquisitionFacts, RhimeMergedData
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.utils import _flux_period_is_missing, datatree_ncdf_encoding

OutputFormat = Literal["netcdf", "zarr", "zarr.zip"]  # for internal type hints
_OBSOLETE_FP_ALL_METADATA = frozenset({".scales", ".species", ".units"})


def _make_merged_data_name(species: str, start_date: str, output_name: str) -> str:
    return f"{species}_{start_date}_{output_name}_merged-data"


def _split_suffix(merged_data_name: str) -> tuple[str, OutputFormat | None]:
    if merged_data_name.endswith(".pickle"):
        raise ValueError("Pickle merged-data files are no longer supported; use netCDF or Zarr.")
    for suffix in ("nc", "zarr", "zarr.zip"):
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
    output_format: OutputFormat = "zarr.zip",
) -> None:
    """Save `fp_all` dictionary to `merged_data_dir`.

    The file name can be specified using `merged_data_name`, or a standard name
    will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    If `merged_data_name` ends with "nc", "zarr", or "zarr.zip", the output
    format is inferred from the suffix. Otherwise, it defaults to zipped zarr.

    Args:
        fp_all: dictionary of merged data to save
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.
        output_format: format to save merged data to (default: "zarr.zip").

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
    if output_format not in {"netcdf", "zarr", "zarr.zip"}:
        raise ValueError(f"Unsupported merged-data format {output_format!r}; use netCDF or Zarr.")

    merged_data_dir = Path(merged_data_dir)

    if not merged_data_dir.exists():
        merged_data_dir.mkdir(parents=True)

    # write to specified output
    dt = fp_all_to_datatree(fp_all, netcdf_safe_attrs=(output_format == "netcdf"))
    dt = clear_datatree_encoding(dt)
    dt = clear_datatree_time_attrs(dt)

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
            dt.to_zarr(merged_data_dir / (merged_data_name + ".zarr"), mode="w-", encoding=encoding)
        else:
            with zarr.ZipStore(merged_data_dir / (merged_data_name + ".zarr.zip"), mode="w") as store:
                dt.to_zarr(store, mode="w-", encoding=encoding)
    else:
        # OpenGHG can attach None when a coordinate's units are unknown.
        # Omit that absent metadata, without inventing a physical unit.
        for node in dt.subtree:
            for coordinate in node.coords.values():
                if coordinate.attrs.get("units") is None:
                    coordinate.attrs.pop("units", None)
        dt.to_netcdf(merged_data_dir / (merged_data_name + ".nc"), encoding=datatree_ncdf_encoding(dt))


def load_merged_data(
    merged_data_dir: str | Path,
    species: str | None = None,
    start_date: str | None = None,
    output_name: str | None = None,
    merged_data_name: str | None = None,
    output_format: OutputFormat | None = None,
) -> dict:
    """Load `fp_all` dictionary from a file in `merged_data_dir`.

    The file name can be specified using `merged_data_name`, or a standard name
    will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    This function tries to automatically find a compatible format of merged data, if a format is not specified.
    It checks for zipped zarr, zarr, then netCDF data. Pickle files are not supported.

    Obsolete ``.species``, ``.units``, and ``.scales`` entries stored in older
    netCDF or Zarr artifacts are omitted from the returned mapping.

    Note: if data is stored in a zarr ZipStore, then the data is eagerly loaded, since the data needs to
    loaded before the zip file is closed.

    Args:
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.
        output_format: format of data to load (if not specified, this will be inferred).

    Returns:
        `fp_all` dictionary.
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
    if output_format is not None and output_format not in {"netcdf", "zarr", "zarr.zip"}:
        raise ValueError(f"Unsupported merged-data format {output_format!r}; use netCDF or Zarr.")

    if output_format is not None:
        ext = "nc" if output_format == "netcdf" else output_format
        merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
        if not merged_data_file.exists():
            raise ValueError(f"No merged data found at {merged_data_file}.")
    else:
        for ext in ["zarr.zip", "zarr", "nc"]:
            merged_data_file = merged_data_dir / (merged_data_name + "." + ext)
            if merged_data_file.exists():
                break
        else:
            # no `break` occurred, so no file found
            if (merged_data_dir / (merged_data_name + ".pickle")).exists():
                raise ValueError("Pickle merged-data files are no longer supported; use netCDF or Zarr.")
            raise ValueError(err_msg)

    # load merged data
    if merged_data_file.suffixes == [".zarr", ".zip"]:
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

    Flux time coordinates are stored on a separate ``flux_time`` dimension,
    with explicit per-source timestamp presence, so that source periods
    beginning before the observations and all-NaN slices are preserved.
    Singleton boundary-condition time dimensions are still dropped.

    Args:
        fp_all: Inversion data keyed by site, with flux sources under
            ``".flux"`` and optional boundary conditions under ``".bc"``.

    Returns:
        A combined dataset containing site scenarios, source-specific flux
        periods, and optional boundary conditions.

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

    # Record which timestamps belong to each source before concat introduces
    # outer-join padding. Flux values cannot serve as this mask because a
    # legitimate flux slice may itself contain only NaNs.
    fluxes = []
    for source, flux_data in fp_all[".flux"].items():
        source_flux = flux_data.data
        if "time" in source_flux.dims:
            source_flux = source_flux.assign(
                flux_time_present=("time", np.ones(source_flux.sizes["time"], dtype=np.int8))
            )
        fluxes.append(source_flux.expand_dims({"source": [source]}))

    # concat fluxes over source before merging into combined scenario
    combined_fluxes = xr.concat(fluxes, dim="source")
    if "flux_time_present" in combined_fluxes:
        combined_fluxes["flux_time_present"] = combined_fluxes["flux_time_present"].fillna(0).astype(np.int8)
        combined_fluxes["flux_time_present"].attrs["long_name"] = "flux source includes this timestamp"
    flux_time_periods = []
    for flux_data in fp_all[".flux"].values():
        variable_period = flux_data.data["flux"].attrs.get("time_period")
        dataset_period = flux_data.data.attrs.get("time_period")
        source_period = dataset_period if _flux_period_is_missing(variable_period) else variable_period
        flux_time_periods.append("" if _flux_period_is_missing(source_period) else str(source_period))
    combined_fluxes["flux_time_period"] = ("source", np.asarray(flux_time_periods, dtype=str))
    combined_fluxes["flux"].attrs.pop("time_period", None)

    if "time" in combined_fluxes.dims:
        combined_fluxes = combined_fluxes.rename(time="flux_time")

    # Merge with override in case coordinates are slightly off. Fresh data are
    # already unit-aligned by ModelScenario.
    combined_scenario = combined_scenario.merge(combined_fluxes, join="override")

    # merge in boundary conditions
    if ".bc" in fp_all:
        bc = fp_all[".bc"].data
        if "time" in bc.dims and bc.sizes["time"] == 1:
            bc = bc.squeeze("time")
        bc = bc.reindex_like(combined_scenario, method="nearest")
        combined_scenario = combined_scenario.merge(bc)

    combined_scenario.attrs["split_by_sectors"] = bool(fp_all.get(".split_by_sectors", False))

    return combined_scenario


def fp_all_from_dataset(ds: xr.Dataset) -> dict:
    """Recover "fp_all" dictionary from "combined scenario" dataset.

    This is the inverse of `make_combined_scenario`, except that the attributes of the
    scenarios, fluxes, and boundary conditions may be different. New datasets
    retain source timestamps on ``flux_time`` with explicit source presence;
    older datasets without that metadata use value-based padding removal or
    fall back to the first observation time for compatibility.

    Args:
        ds: dataset created by `make_combined_scenario`

    Returns:
        dictionary containing model scenarios keyed by site, as well as flux and boundary conditions.

    Raises:
        ValueError: If serialized ``mf`` units are invalid or are not a molar
            mixing ratio. Missing units default to ``mol/mol``.
    """
    mole_fraction_unit_scale(
        ds.mf.attrs.get("units", "mol/mol"),
        context="serialized merged observations",
    )
    fp_all: dict[str, Any] = {}

    # get scenarios
    bc_vars = ["vmr_n", "vmr_e", "vmr_s", "vmr_w"]

    for i, site in enumerate(ds.site.values):
        scenario = (
            ds.sel(site=site, drop=True)
            .drop_vars(["flux", "flux_time_period", "flux_time_present", *bc_vars], errors="ignore")
            .drop_dims(["source", "flux_time"], errors="ignore")
        )

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = scenario.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"

            scenario.attrs[k] = val

        fp_all[site] = scenario.dropna("time", subset=["mf"])

    # get fluxes
    fp_all[".flux"] = {}

    for i, source in enumerate(ds.source.values):
        flux_time_present = None
        if "flux_time_present" in ds:
            flux_time_present = ds["flux_time_present"].sel(source=source, drop=True)

        flux_ds = ds[["flux"]].sel(source=source, drop=True)
        if "flux_time" in flux_ds.dims:
            flux_ds = flux_ds.rename(flux_time="time")
            if flux_time_present is not None:
                flux_time_present = flux_time_present.rename(flux_time="time")
        elif "time" not in flux_ds.dims:
            # Backward compatibility for old combined datasets that squeezed a
            # singleton flux time coordinate during serialization.
            flux_ds = flux_ds.expand_dims({"time": [ds.time.min().values]})

        if flux_time_present is not None:
            flux_ds = flux_ds.isel(time=flux_time_present.load().values.astype(bool))
        else:
            # Older stores did not record timestamp presence, so retain their
            # best-effort value-based padding removal.
            flux_ds = flux_ds.dropna("time", how="all", subset=["flux"])
        flux_ds = flux_ds.transpose(..., "time")
        if "flux_time_period" in ds:
            time_period = str(ds["flux_time_period"].sel(source=source).load().item())
            if time_period:
                flux_ds["flux"].attrs["time_period"] = time_period

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

    if bool(ds.attrs.get("split_by_sectors", False)):
        warnings.warn(
            "Legacy `fp_all_from_dataset` drops scenario `source` dimensions, so sector-resolved "
            "state cannot be reconstructed. Setting `fp_all['.split_by_sectors'] = False` on load.",
            UserWarning,
            stacklevel=2,
        )
    fp_all[".split_by_sectors"] = False

    return fp_all


# ----------------------------------------
# DataTree conversions
# ----------------------------------------


def openghg_data_to_dataset(openghg_data: _BaseData, netcdf_safe_attrs: bool = False) -> xr.Dataset:
    """Attach serialization metadata without modifying borrowed dataset attributes.

    The shallow copy preserves the underlying numerical arrays and Dask graphs.
    """
    ds = openghg_data.data.copy(deep=False)

    if netcdf_safe_attrs:
        ds.attrs["openghg_metadata"] = json.dumps(openghg_data.metadata)
    else:
        ds.attrs["openghg_metadata"] = openghg_data.metadata
    return ds


def dataset_to_flux_data(ds: xr.Dataset) -> FluxData:
    if "flux" not in ds.data_vars:
        raise ValueError("Dataset must have `flux` data variable to convert to FluxData.")
    ds = ds.copy()
    metadata = ds.attrs.pop("openghg_metadata")

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
    metadata = ds.attrs.pop("openghg_metadata")

    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return BoundaryConditionsData(metadata=metadata, data=ds)


def flux_dict_to_datatree(flux_dict: dict[str, FluxData], netcdf_safe_attrs: bool = False) -> xr.DataTree:
    dt_dict = {k: openghg_data_to_dataset(v, netcdf_safe_attrs) for k, v in flux_dict.items()}
    return xr.DataTree.from_dict(dt_dict)


def datatree_to_flux_dict(dt: xr.DataTree) -> dict[str, FluxData]:
    """Convert an xarray DataTree to a dict of FluxData objects.

    Args:
        dt: DataTree whose child nodes are converted to datasets and then to FluxData.

    Returns:
        Mapping from node keys (as strings) to FluxData instances.
    """
    return {str(k): dataset_to_flux_data(v.to_dataset()) for k, v in dt.items()}


def fp_all_to_datatree(fp_all: dict, netcdf_safe_attrs: bool = False) -> xr.DataTree:
    dt_dict: dict[str, xr.Dataset | xr.DataTree] = {}
    scenario_dict = {}
    dt_attrs = {}

    if ".flux" in fp_all:
        dt_dict["fluxes"] = flux_dict_to_datatree(fp_all[".flux"], netcdf_safe_attrs)

    for k, v in fp_all.items():
        if k in {".flux", ".provenance"} or k in _OBSOLETE_FP_ALL_METADATA:
            continue
        if isinstance(v, BoundaryConditionsData):
            dt_dict[k.removeprefix(".")] = openghg_data_to_dataset(v, netcdf_safe_attrs)
        elif not k.startswith(".") and isinstance(v, xr.Dataset):
            scenario_dict[k] = v
        else:
            if k == ".artifact_stage":
                dt_attrs["artifact_stage"] = v
            elif netcdf_safe_attrs and k == ".split_by_sectors":
                # NetCDF forbids leading dots in names and Boolean attributes.
                dt_attrs["split_by_sectors"] = int(v)
            else:
                dt_attrs[k] = v

    dt_dict["scenarios"] = xr.DataTree.from_dict(scenario_dict)

    dt = xr.DataTree.from_dict(dt_dict)
    dt.attrs = dt_attrs

    return dt


def datatree_to_fp_all(dt: xr.DataTree) -> dict:
    if "scenarios" not in dt:
        raise ValueError("Can only convert DataTree to fp_all if 'scenarios' group is present.")

    fp_all: dict[str, Any] = {}
    if "artifact_stage" in dt.attrs:
        fp_all[".artifact_stage"] = dt.attrs["artifact_stage"]

    if "fluxes" in dt:
        fp_all[".flux"] = datatree_to_flux_dict(dt.fluxes)

    if "bc" in dt:
        fp_all[".bc"] = dataset_to_bc_data(dt.bc.to_dataset())

    for k, v in dt.scenarios.items():
        fp_all[str(k)] = v.to_dataset()

    fp_all.update(
        {
            str(k): v
            for k, v in dt.attrs.items()
            if str(k) not in _OBSOLETE_FP_ALL_METADATA and str(k) != "artifact_stage"
        }
    )
    if "split_by_sectors" in fp_all:
        fp_all[".split_by_sectors"] = bool(fp_all.pop("split_by_sectors"))

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


def clear_datatree_time_attrs(dt: xr.DataTree) -> xr.DataTree:
    result = dt.copy()

    for g in result.groups:
        if "time" in result[g].coords:
            result[g].coords["time"].attrs.pop("units", None)

    return result


def to_legacy_fp_all(merged: RhimeMergedData) -> dict:
    """Adapt to remaining wrapper-based scientific consumers (#821; remove in 0.9).

    Only this explicit adapter reconstructs OpenGHG wrappers. Flux and boundary
    dataset containers are shallow copies; site datasets remain borrowed.
    Consumers must copy site containers before assigning variables or attrs.
    Numerical arrays remain shared. Wrapper metadata carries selected source
    identities, not arbitrary OpenGHG catalogue fields.
    """
    result = dict(merged.site_data)
    result[".flux"] = {
        source: FluxData(
            data=data.copy(deep=False),
            metadata={"data_type": "flux", **asdict(merged.provenance.flux[source])},
        )
        for source, data in merged.flux_data.items()
    }
    if merged.boundary_data is not None:
        result[".bc"] = BoundaryConditionsData(
            data=merged.boundary_data.copy(deep=False),
            metadata=asdict(merged.provenance.boundary),
        )
    result[".split_by_sectors"] = merged.split_by_sectors
    return result


def retrieve_inversion_data(
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
    emissions_domain: str | None = None,
    split_by_sectors: bool = False,
    averagingerror: bool = True,
    save_merged_data: bool = False,
    merged_data_name: str | None = None,
    merged_data_dir: str | None = None,
    output_name: str | None = None,
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
        emissions_domain: Optional flux-domain metadata selector. When it is
            different from ``domain``, flux density is interpolated with
            nearest neighbours onto each footprint grid before merging.
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
        scenario defines the unit target requested for later sites. Acquisition
        constructs ``RhimeMergedData`` first; this compatibility API projects
        its datasets into historical wrappers rather than retaining arbitrary
        OpenGHG wrapper metadata. Selected input identities belong to the
        modern record's provenance.
    """
    site_values = [sites] if isinstance(sites, str) else sites
    site_options = SiteOptions.from_inputs(
        sites=site_values,
        averaging_period=averaging_period,
        inlet=inlet,
        fp_height=fp_height,
        instrument=instrument,
        platform=platform,
        obs_data_level=obs_data_level,
        met_model=met_model,
        max_level=max_level,
        time_resolved=time_resolved,
    )
    merged = RhimeMergedData.from_options(
        site_options=site_options,
        species=species,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        calibration_scale=calibration_scale,
        fp_model=fp_model,
        fp_species=fp_species,
        flux_sources=emissions_name,
        use_bc=use_bc,
        bc_input=bc_input,
        bc_store=bc_store,
        obs_store=obs_store,
        footprint_store=footprint_store,
        emissions_store=emissions_store,
        emissions_domain=emissions_domain,
        split_by_sectors=split_by_sectors,
        flux_non_finite_check=flux_non_finite_check,
    )

    merged = prepare_observation_errors(merged, averaging_error=averagingerror)

    # Keep the historical public tuple and its mapping layout at this adapter.
    legacy = to_legacy_fp_all(merged)
    fp_all = {".flux": legacy[".flux"], ".split_by_sectors": merged.split_by_sectors}
    if ".bc" in legacy:
        fp_all[".bc"] = legacy[".bc"]
    fp_all.update(merged.site_data)
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

    retained = merged.site_options
    return (
        fp_all, list(retained.sites), list(retained.inlet), list(retained.fp_height),
        list(retained.instrument), list(retained.averaging_period),
    )


@wraps(retrieve_inversion_data)
def data_processing_surface_notracer(*args, **kwargs):
    warnings.warn(
        "data_processing_surface_notracer is deprecated; use retrieve_inversion_data instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return retrieve_inversion_data(*args, **kwargs)


def from_legacy_fp_all(
    fp_all: dict, site_options: SiteOptions, *,
    acquisition: AcquisitionFacts | None = None,
) -> RhimeMergedData:
    """Borrow legacy datasets with resolved selectors; unknown identities remain unknown."""
    fluxes = fp_all.get(".flux", {})
    boundary = fp_all.get(".bc")
    provenance = MergedDataProvenance(
        observations={site: InputProvenance() for site in site_options.sites},
        footprints={site: InputProvenance() for site in site_options.sites},
        flux={source: selected_provenance(value) for source, value in fluxes.items()},
        boundary=selected_provenance(boundary) if boundary is not None else None,
    )
    return RhimeMergedData(
        site_data={site: fp_all[site] for site in site_options.sites},
        flux_data={
            source: value if isinstance(value, xr.Dataset) else value.data
            for source, value in fluxes.items()
        },
        boundary_data=boundary if isinstance(boundary, xr.Dataset) else getattr(boundary, "data", None),
        site_options=site_options,
        split_by_sectors=bool(fp_all.get(".split_by_sectors", False)),
        provenance=provenance,
        acquisition=acquisition or AcquisitionFacts(),
    )
