from pathlib import Path
from typing_extensions import Self
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, TypeVar, cast
import json

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import align_sparse_lat_lon
from openghg_inversions.basis.basis_functions import BasisFunctions


MODERN_INVERSION_OUTPUT_SCHEMA = "openghg_inversions.inversion_output"
MULTIINDEX_DIMS_ATTR = "openghg_inversions:multiindex_dims"


def _json_default(value: object) -> str:
    """JSON fallback for metadata values stored on output artifacts."""
    return str(value)


def _json_attr(value: dict[str, Any]) -> str:
    """Encode output metadata for xarray attrs."""
    return json.dumps(value, default=_json_default)


def _load_json_attr(attrs: dict[Any, Any], key: str) -> dict[str, Any]:
    """Decode optional JSON metadata from xarray attrs."""
    raw = attrs.get(key)
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        try:
            raw = raw.decode()
        except UnicodeDecodeError:
            return {}
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_datatree(
    dt: xr.DataTree, output_file: str | Path, output_format: Literal["netcdf", "zarr"] | None
) -> None:
    """Save a DataTree to NetCDF or Zarr, inferring format from the path when needed."""
    output_file = Path(output_file)

    if output_format is None:
        try:
            output_format = {".nc": "netcdf", ".zarr": "zarr"}[output_file.suffix]  # type: ignore
        except KeyError:
            raise ValueError(
                f"Output file {output_file} does not end in '.nc' or '.zarr'; please specify `output_format`."
            )

    if output_format == "netcdf":
        if output_file.suffix != ".nc":
            output_file = output_file.with_suffix(".nc")
        dt.to_netcdf(output_file)
    elif output_format == "zarr":
        if output_file.suffix != ".zarr":
            output_file = output_file.with_suffix(".zarr")
        dt.to_zarr(output_file)
    else:
        raise ValueError(f"Unsupported output_format: {output_format!r}")


def _open_datatree_loaded(file_path: str | Path) -> xr.DataTree:
    """Open and load a DataTree artifact with the same engine fallback as legacy outputs."""
    open_errors: list[Exception] = []
    for engine in ("h5netcdf", None):
        try:
            dt = (
                xr.open_datatree(file_path, engine=engine)
                if engine is not None
                else xr.open_datatree(file_path)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            open_errors.append(exc)
        else:
            with dt:
                return dt.load()

    raise open_errors[-1]


def _inferencedata_to_datatree(idata: az.InferenceData) -> xr.DataTree:
    """Convert an ArviZ InferenceData object to a DataTree."""
    return xr.DataTree.from_dict(
        {group: _reset_serialisation_multiindexes(idata[group]) for group in idata.groups()}
    )


def _inferencedata_from_datatree(dt: xr.DataTree) -> az.InferenceData:
    """Convert a DataTree of InferenceData groups back to ArviZ InferenceData."""
    return cast(Any, az.InferenceData)(
        **{group: _restore_serialisation_multiindexes(child.to_dataset()) for group, child in dt.items()}
    )


def _reset_serialisation_multiindexes(ds: xr.Dataset) -> xr.Dataset:
    """Expand xarray MultiIndexes before DataTree serialisation."""
    result = ds
    multiindex_dims: list[dict[str, object]] = []
    for dim, index in ds.indexes.items():
        if dim in result.dims and isinstance(index, pd.MultiIndex):
            level_names = list(index.names)
            if any(name is None for name in level_names):
                raise ValueError(f"Cannot serialise unnamed MultiIndex levels for dimension {dim!r}.")
            result = result.reset_index(dim)
            multiindex_dims.append({"dim": str(dim), "levels": [str(name) for name in level_names]})
    if multiindex_dims:
        result = result.copy()
        result.attrs = dict(result.attrs)
        result.attrs[MULTIINDEX_DIMS_ATTR] = json.dumps({"dims": multiindex_dims})
    return result


def _restore_serialisation_multiindexes(ds: xr.Dataset) -> xr.Dataset:
    """Restore MultiIndexes expanded for DataTree serialisation."""
    raw_multiindex_dims = ds.attrs.get(MULTIINDEX_DIMS_ATTR)
    if raw_multiindex_dims is None:
        return ds

    result = ds.copy()
    result.attrs = dict(result.attrs)
    del result.attrs[MULTIINDEX_DIMS_ATTR]

    if isinstance(raw_multiindex_dims, bytes):
        try:
            raw_multiindex_dims = raw_multiindex_dims.decode()
        except UnicodeDecodeError:
            return result
    if not isinstance(raw_multiindex_dims, str):
        return result

    try:
        payload = json.loads(raw_multiindex_dims)
    except json.JSONDecodeError:
        return result

    records = payload.get("dims") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return result

    for record in records:
        if not isinstance(record, dict):
            continue
        dim = record.get("dim")
        levels = record.get("levels")
        if not isinstance(dim, str) or not isinstance(levels, list):
            continue
        if not all(isinstance(level, str) for level in levels):
            continue
        if dim in result.dims and all(level in result and result[level].dims == (dim,) for level in levels):
            result = result.set_index({dim: levels})
    return result


def filter_data_vars_by_prefix(
    ds: xr.Dataset, var_name_prefixes: str | list[str], sep: str = "_"
) -> xr.Dataset:
    """Select data variables that match the specified filters.

    For instance, if var_name_prefixes = 'prior', then any data variable
    whose name begins with 'prior_' will be selected. The underscore '_' is
    added by default, but can be changed by specifying sep.

    Args:
        ds: Dataset to filter.
        var_name_prefixes: (List of) prefix(s) to filter data variables by.
        sep: Separator for prefix; default is "_".

    Returns:
        xr.Dataset: Dataset restricted to data variables whose names match the filter.
    """
    if isinstance(var_name_prefixes, str):
        var_name_prefixes = [var_name_prefixes]

    var_name_prefixes = [f"{name}{sep}" for name in var_name_prefixes]

    data_vars = []
    for dv in ds.data_vars:
        for name in var_name_prefixes:
            if str(dv).startswith(name):
                data_vars.append(dv)

    return ds[data_vars]


def _filter_trace_data_vars_by_name(ds: xr.Dataset, var_names: str | list[str]) -> xr.Dataset:
    """Select trace variables by exact base variable name.

    Trace datasets use names like ``x_prior`` or ``mu_bc_posterior``. This
    helper matches the full base variable name before the trace-group suffix so
    ``var_names="x"`` does not also select ``x_latent_prior``.
    """
    if isinstance(var_names, str):
        var_names = [var_names]

    group_suffixes = (
        "_prior_predictive",
        "_posterior_predictive",
        "_prior",
        "_posterior",
    )
    selected: list[Hashable] = []
    wanted = set(var_names)

    for dv in ds.data_vars:
        name = str(dv)
        for suffix in group_suffixes:
            if name.endswith(suffix):
                base_name = name.removesuffix(suffix)
                if base_name in wanted:
                    selected.append(dv)
                break

    return ds[selected]


def convert_idata_to_dataset(
    idata: az.InferenceData, group_filters=["prior", "posterior"], add_suffix=True
) -> xr.Dataset:
    """Merge all groups in an arviz InferenceData object into a single xr.Dataset.

    Args:
        idata: arviz InferenceData containing traces (and other data)
        group_filters: Filters for the groups of the InferenceData. A group will
          be selected if a filter is a substring of the group name. So the groups
          "prior" and "prior_predictive" will both match the filter "prior". The
          default filters select the "prior", "prior_predictive", "posterior", and
          "posterior_predictive" groups.
        add_suffix: if True, rename the data variables so that they end in the
          name of the group they came from.

    Returns:
        xr.Dataset containing all data variables in the selected groups of the
        InferenceData

    """
    traces = []
    for group in idata.groups():
        if any(filt in group for filt in group_filters):
            trace = idata[group]
            if add_suffix:
                rename_dict = {dv: f"{dv}_{group}" for dv in trace.data_vars}
                trace = trace.rename_vars(rename_dict)
            if "chain" in trace.dims:
                trace = trace.isel(chain=0, drop=True)
            traces.append(trace)
    return xr.merge(traces)


def _add_attributes_to_trace_dataset(trace_ds: xr.Dataset, obs_units: str, obs_longname: str) -> None:
    """Add attributes to trace dataset.

    Args:
        trace_ds: trace dataset (probably created by `convert_idata_to_dataset`)
        obs_units: units for observation data used in inversion
        obs_longname: long name for observation data used in inversion

    Returns:
        None: updates Dataset in-place

    """
    for dv in trace_ds.data_vars:
        if str(dv).endswith("prior_predictive"):
            trace_ds[dv].attrs["units"] = obs_units
            trace_ds[dv].attrs["long_name"] = "prior_predictive_" + obs_longname
        elif str(dv).endswith("posterior_predictive"):
            trace_ds[dv].attrs["units"] = obs_units
            trace_ds[dv].attrs["long_name"] = "posterior_predictive_" + obs_longname
        elif str(dv).startswith("mu_bc"):
            suffix = str(dv).removeprefix("mu_bc_")
            trace_ds[dv].attrs["units"] = obs_units
            trace_ds[dv].attrs["long_name"] = suffix + "_modelled_baseline"
        elif str(dv).endswith("prior"):
            prefix = str(dv).removesuffix("_prior")
            if prefix == "x":
                name = "flux_scaling_factor"
            elif "sig" in prefix:
                name = "pollution_event_scaling_factor"
            elif prefix == "bc":
                name = "boundary_conditions_scaling_factor"
            else:
                name = str(dv)
            trace_ds[dv].attrs["long_name"] = f"prior_trace_of_{name}"
        elif str(dv).endswith("posterior"):
            prefix = str(dv).removesuffix("_posterior")
            if prefix == "x":
                name = "flux_scaling_factor"
            elif "sig" in prefix:
                name = "pollution_event_scaling_factor"
            elif prefix == "bc":
                name = "boundary_conditions_scaling_factor"
            else:
                name = str(dv)
            trace_ds[dv].attrs["long_name"] = f"posterior_trace_of_{name}"


XrDataArrayOrSet = TypeVar("XrDataArrayOrSet", xr.DataArray, xr.Dataset)


def _nmeasure_to_site_time(
    data: XrDataArrayOrSet,
    site_indicators: xr.DataArray | np.ndarray,
    times: xr.DataArray | np.ndarray,
    site_names: xr.DataArray | dict | None = None,
) -> XrDataArrayOrSet:
    """Convert `nmeasure` dimension to multi-index over `site` and `time`.

    This uses an array of `site_indicators` and an array of times to construct
    coordinates for the dimension `nmeasure`. If the `site_indicators` are
    numbers, `site_names` can be provided to convert these numbers into site
    names.

    Args:
        data: xr.DataArray or xr.Dataset. Typically, this has a `nmeasure`
          coordinate, but this isn't a strict requirement.
        site_indicators: array specifying the site where a measurement was taken
        times: array specifying the time a measurement was taken
        site_names: optional DataArray or dict mapping the values of
          `site_indicator` to strings. If `None`, the values of `site_indicator`
          will be used unchanged.

    Returns:
        xr.DataArray or xr.Dataset (same type as input) with `nmeasure`
          coordinate consisting of stacked `site` and `time` coordinates.

    Raises:
        ValueError: if `site_indicators` and `times` have different lengths.

    """
    if len(site_indicators) != len(times):
        raise ValueError(
            "Site indicators and times must be same length, got:"
            f"\nsite indicators:\n{site_indicators}\ntimes:\n{times}"
        )

    time_vals = times.values if isinstance(times, xr.DataArray) else times
    site_codes = site_indicators.values if isinstance(site_indicators, xr.DataArray) else site_indicators

    if site_names is not None:
        if isinstance(site_names, xr.DataArray):
            site_names = dict(site_names.to_series())

        site_codes = [site_names.get(x) for x in site_codes]

    nmeasure_multiindex = pd.MultiIndex.from_arrays([site_codes, time_vals], names=["site", "time"])
    xr_nmeasure_multiindex = xr.Coordinates.from_pandas_multiindex(nmeasure_multiindex, "nmeasure")

    result = data.assign_coords(xr_nmeasure_multiindex)
    result.time.attrs = times.attrs if isinstance(times, xr.DataArray) else {}

    return result


@dataclass
class InversionOutput:
    """Modern RHIME inversion output contract.

    This object carries the runtime artifacts needed to reproduce and extend
    RHIME outputs without exposing fixedbasis ``fp_data`` or legacy
    ``inferpymc_postprocessouts`` dictionaries.
    """

    trace: az.InferenceData
    inv_inputs: xr.Dataset
    basis_functions: BasisFunctions
    run_metadata: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    output_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def start_date(self) -> str | None:
        """Inversion start date from run metadata."""
        value = self.run_metadata.get("start_date")
        return None if value is None else str(value)

    @property
    def end_date(self) -> str | None:
        """Inversion end date from run metadata."""
        value = self.run_metadata.get("end_date")
        return None if value is None else str(value)

    @property
    def species(self) -> str | None:
        """Species name from model metadata."""
        value = self.model_metadata.get("species")
        return None if value is None else str(value)

    @property
    def domain(self) -> str | None:
        """Domain name from model metadata."""
        value = self.model_metadata.get("domain")
        return None if value is None else str(value)

    def to_datatree(self) -> xr.DataTree:
        """Convert the modern output to a serialisable DataTree."""
        dt = xr.DataTree.from_dict(
            {
                "trace": _inferencedata_to_datatree(self.trace),
                "inv_inputs": xr.DataTree(_reset_serialisation_multiindexes(self.inv_inputs)),
                "basis_functions": self.basis_functions.to_datatree(),
            }
        )
        dt.attrs = {
            "schema": MODERN_INVERSION_OUTPUT_SCHEMA,
            "schema_version": 1,
            "run_metadata": _json_attr(self.run_metadata),
            "model_metadata": _json_attr(self.model_metadata),
            "output_metadata": _json_attr(self.output_metadata),
            "provenance": _json_attr(self.provenance),
        }
        return dt

    def save(self, output_file: str | Path, output_format: Literal["netcdf", "zarr"] | None = None) -> None:
        """Save modern InversionOutput to NetCDF or Zarr."""
        _save_datatree(self.to_datatree(), output_file, output_format)

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> Self:
        """Construct a modern InversionOutput from a serialised DataTree."""
        schema = dt.attrs.get("schema")
        if schema is not None and schema != MODERN_INVERSION_OUTPUT_SCHEMA:
            raise ValueError(f"Unexpected InversionOutput schema: {schema!r}")

        trace = _inferencedata_from_datatree(cast(xr.DataTree, dt["trace"]))
        inv_inputs = _restore_serialisation_multiindexes(cast(xr.DataTree, dt["inv_inputs"]).to_dataset())
        basis_functions = BasisFunctions.from_datatree(cast(xr.DataTree, dt["basis_functions"]))
        return cls(
            trace=trace,
            inv_inputs=inv_inputs,
            basis_functions=basis_functions,
            run_metadata=_load_json_attr(dt.attrs, "run_metadata"),
            model_metadata=_load_json_attr(dt.attrs, "model_metadata"),
            output_metadata=_load_json_attr(dt.attrs, "output_metadata"),
            provenance=_load_json_attr(dt.attrs, "provenance"),
        )

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        """Load a modern InversionOutput artifact."""
        return cls.from_datatree(_open_datatree_loaded(file_path))


_OBS_INPUT_RENAMES = {
    "mf": "y_obs",
    "mf_error": "y_obs_error",
    "mf_prior_factor": "y_obs_prior_factor",
    "mf_prior_upper_level_factor": "y_obs_prior_upper_level_factor",
    "mf_repeatability": "y_obs_repeatability",
    "mf_variability": "y_obs_variability",
}
_REQUIRED_OBS_INPUTS = ("mf", "mf_error", "mf_repeatability", "mf_variability")


def require_standard_postprocessing_metadata(inv_out: InversionOutput) -> tuple[str, str, str, str]:
    """Return metadata required by current single-sector postprocessing helpers."""
    if inv_out.run_metadata.get("split_by_sectors"):
        raise ValueError("Standard postprocessing supports only single-sector RHIME outputs.")

    species = inv_out.species
    domain = inv_out.domain
    start_date = inv_out.start_date
    end_date = inv_out.end_date
    if species is None or domain is None or start_date is None or end_date is None:
        raise ValueError(
            "Modern InversionOutput metadata must include species, domain, start_date, and end_date."
        )
    return species, domain, start_date, end_date


def standard_species(inv_out: InversionOutput) -> str:
    """Return the species name for standard postprocessing."""
    species, _, _, _ = require_standard_postprocessing_metadata(inv_out)
    return species


def standard_domain(inv_out: InversionOutput) -> str:
    """Return the domain name for standard postprocessing."""
    _, domain, _, _ = require_standard_postprocessing_metadata(inv_out)
    return domain


def standard_start_time(inv_out: InversionOutput) -> pd.Timestamp:
    """Return the inversion start time for standard postprocessing."""
    _, _, start_date, _ = require_standard_postprocessing_metadata(inv_out)
    return pd.to_datetime(start_date)


def standard_end_time(inv_out: InversionOutput) -> pd.Timestamp:
    """Return the inversion end time for standard postprocessing."""
    _, _, _, end_date = require_standard_postprocessing_metadata(inv_out)
    return pd.to_datetime(end_date)


def standard_period_midpoint(inv_out: InversionOutput) -> pd.Timestamp:
    """Return the midpoint of the inversion period."""
    start_time = standard_start_time(inv_out)
    return start_time + (standard_end_time(inv_out) - start_time) / 2


def standard_site_names(inv_out: InversionOutput) -> xr.DataArray:
    """Return site names from inversion inputs or run metadata."""
    if "site_names" in inv_out.inv_inputs:
        return inv_out.inv_inputs["site_names"]

    sites = inv_out.run_metadata.get("sites", [])
    return xr.DataArray(list(sites), dims="nsite", coords={"nsite": np.arange(len(sites))})


def _standard_basis_from_basis_functions(
    basis_functions: BasisFunctions, inv_inputs: xr.Dataset
) -> xr.DataArray:
    """Return the flat standard-postprocessing basis from modern basis functions."""
    basis = basis_functions.operator.basis_matrix
    current_state_dim = basis_functions.operator.meta.state_dim
    if current_state_dim != "region":
        basis = basis.rename({current_state_dim: "region"})
    if "region" in inv_inputs.coords:
        basis = basis.reindex(region=inv_inputs.region)
    return basis


def _modern_observation_inputs(inv_inputs: xr.Dataset) -> xr.Dataset:
    """Extract the modern observation-input dataset used by postprocessing."""
    missing = [name for name in _REQUIRED_OBS_INPUTS if name not in inv_inputs]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"Modern InversionOutput.inv_inputs is missing required variables: {missing_names}.")

    available = [name for name in _OBS_INPUT_RENAMES if name in inv_inputs]
    rename = {name: _OBS_INPUT_RENAMES[name] for name in available}
    return inv_inputs[available].rename(rename)


def _has_site_time_nmeasure_index(data: xr.DataArray | xr.Dataset) -> bool:
    """Return True when ``data`` already has a site/time ``nmeasure`` index."""
    nmeasure_index = data.indexes.get("nmeasure")
    return isinstance(nmeasure_index, pd.MultiIndex) and list(nmeasure_index.names) == ["site", "time"]


def standard_nmeasure_to_site_time(inv_out: InversionOutput, data: XrDataArrayOrSet) -> XrDataArrayOrSet:
    """Convert an ``nmeasure`` dimension to a site/time MultiIndex."""
    if "nmeasure" not in data.dims or _has_site_time_nmeasure_index(data):
        return data

    if {"site", "time"}.issubset(data.coords) and all(
        data[coord].dims == ("nmeasure",) for coord in ("site", "time")
    ):
        return data.set_index(nmeasure=["site", "time"])

    return _nmeasure_to_site_time(
        data,
        inv_out.inv_inputs["site_indicator"],
        inv_out.inv_inputs["time"],
        standard_site_names(inv_out),
    )


def standard_obs_inputs(inv_out: InversionOutput) -> xr.Dataset:
    """Return observation and observation-error inputs for standard products."""
    require_standard_postprocessing_metadata(inv_out)
    return standard_nmeasure_to_site_time(inv_out, _modern_observation_inputs(inv_out.inv_inputs))


def standard_trace_dataset(inv_out: InversionOutput, var_names: str | list[str] | None = None) -> xr.Dataset:
    """Return prior and posterior trace samples for standard products."""
    obs = standard_obs_inputs(inv_out)["y_obs"]
    trace_ds = convert_idata_to_dataset(inv_out.trace)

    if "longname" in obs.attrs:
        obs_long_name = obs.attrs["longname"]
    else:
        obs_long_name = obs.attrs.get("long_name", "observed_mole_fraction")

    _add_attributes_to_trace_dataset(trace_ds, obs.attrs.get("units", ""), obs_long_name)
    result = standard_nmeasure_to_site_time(inv_out, trace_ds)

    if var_names is not None:
        result = _filter_trace_data_vars_by_name(result, var_names)

    return result


def standard_model_data(inv_out: InversionOutput, var_names: str | list[str] | None = None) -> xr.Dataset:
    """Return model input data from the ``InferenceData`` constant groups."""
    result = convert_idata_to_dataset(inv_out.trace, group_filters=["data"], add_suffix=False)
    result = standard_nmeasure_to_site_time(inv_out, result)

    if var_names is not None:
        result = filter_data_vars_by_prefix(result, var_names, sep="")

    return result


def standard_total_err(inv_out: InversionOutput, take_mean: bool = True) -> xr.DataArray:
    """Return the posterior model-data mismatch error."""
    result = standard_trace_dataset(inv_out, var_names="epsilon").epsilon_posterior

    if take_mean:
        result = result.mean("draw")

    result.attrs["units"] = standard_obs_inputs(inv_out)["y_obs"].attrs.get("units", "")
    result.attrs["long_name"] = "total model-data mismatch error"

    return result.rename("total_error")


def standard_model_err(inv_out: InversionOutput) -> xr.DataArray:
    """Return the inferred model-error component."""
    total_err = standard_total_err(inv_out, take_mean=False)
    obs = standard_obs_inputs(inv_out)
    total_obs_err = obs["y_obs_error"]

    result = np.sqrt(np.maximum(total_err**2 - total_obs_err**2, 0)).mean("draw")  # type: ignore
    result.attrs["units"] = obs["y_obs"].attrs.get("units", "")
    result.attrs["long_name"] = "inferred model error"
    return result.rename("model_error")


def standard_obs_and_errors(inv_out: InversionOutput) -> xr.Dataset:
    """Return observations and derived uncertainty terms."""
    result = xr.merge(
        [standard_obs_inputs(inv_out), standard_model_err(inv_out), standard_total_err(inv_out)]
    )
    result.attrs = {}
    return result


def standard_flux(inv_out: InversionOutput) -> xr.DataArray:
    """Return prior flux normalised for current standard products."""
    flux = inv_out.basis_functions.flux
    if "flux_time" in flux.dims:
        return flux
    if "time" in flux.dims:
        return flux.rename(time="flux_time")
    return flux.expand_dims(flux_time=[standard_start_time(inv_out)])


def standard_basis_matrix(inv_out: InversionOutput) -> xr.DataArray:
    """Return the current flat basis matrix for products that still report it."""
    basis = _standard_basis_from_basis_functions(inv_out.basis_functions, inv_out.inv_inputs)
    basis = align_sparse_lat_lon(basis, standard_flux(inv_out))
    if "time" in basis.dims:
        basis = basis.rename(time="flux_time")
    elif "time" in basis.coords:
        basis = basis.drop_vars("time")
    return basis


def standard_flat_basis(inv_out: InversionOutput) -> xr.DataArray:
    """Return a two-dimensional basis-region map for standard output files."""
    basis = standard_basis_matrix(inv_out)
    if len(basis.dims) == 2:
        return basis

    region_dim = next(str(dim) for dim in basis.dims if dim not in ["lat", "lon", "latitude", "longitude"])
    return (basis * basis[region_dim]).sum(region_dim).as_numpy().rename("basis")
