from pathlib import Path
from typing_extensions import Self
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, TypeVar, cast
import json

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies, align_sparse_lat_lon
from openghg_inversions.basis.basis_functions import BasisFunctions


MODERN_INVERSION_OUTPUT_SCHEMA = "openghg_inversions.inversion_output"
LEGACY_INVERSION_OUTPUT_SCHEMA = "openghg_inversions.legacy_inversion_output"
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
        raw = raw.decode()
    if not isinstance(raw, str):
        return {}
    return json.loads(raw)


def _save_datatree(dt: xr.DataTree, output_file: str | Path, output_format: Literal["netcdf", "zarr"] | None) -> None:
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
            dt = xr.open_datatree(file_path, engine=engine) if engine is not None else xr.open_datatree(file_path)
        except (OSError, RuntimeError, ValueError) as exc:
            open_errors.append(exc)
        else:
            with dt:
                return dt.load()

    raise open_errors[-1]


def _inferencedata_to_datatree(idata: az.InferenceData) -> xr.DataTree:
    """Convert an ArviZ InferenceData object to a DataTree."""
    return xr.DataTree.from_dict({group: idata[group] for group in idata.groups()})


def _inferencedata_from_datatree(dt: xr.DataTree) -> az.InferenceData:
    """Convert a DataTree of InferenceData groups back to ArviZ InferenceData."""
    return cast(Any, az.InferenceData)(
        **{group: child.to_dataset() for group, child in dt.items()}
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


def _restore_inv_inputs_indexes(ds: xr.Dataset) -> xr.Dataset:
    """Restore RHIME inv_inputs MultiIndexes expanded for serialisation."""
    raw_multiindex_dims = ds.attrs.get(MULTIINDEX_DIMS_ATTR)
    if raw_multiindex_dims is None:
        return ds

    result = ds.copy()
    result.attrs = dict(result.attrs)
    del result.attrs[MULTIINDEX_DIMS_ATTR]
    records = json.loads(raw_multiindex_dims)["dims"]
    for record in records:
        dim = record["dim"]
        levels = record["levels"]
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
        inv_inputs = _restore_inv_inputs_indexes(cast(xr.DataTree, dt["inv_inputs"]).to_dataset())
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


@dataclass
class LegacyInversionOutput:
    """Legacy-shaped postprocessing carrier.

    This is the object consumed by the current ``postprocessing`` submodule. It
    is distinct from the true legacy ``inferpymc_postprocessouts`` dataset.
    """

    obs: xr.DataArray
    obs_err: xr.DataArray
    obs_repeatability: xr.DataArray
    obs_variability: xr.DataArray
    flux: xr.DataArray
    basis: xr.DataArray
    trace: az.InferenceData
    site_indicators: xr.DataArray
    times: xr.DataArray
    start_date: str
    end_date: str
    species: str
    domain: str
    site_names: xr.DataArray | None = None
    obs_prior_factor: xr.DataArray | None = None
    obs_prior_upper_level_factor: xr.DataArray | None = None

    def __post_init__(self) -> None:
        """Check that trace has posterior traces, and fix flux time values."""
        if not hasattr(self.trace, "posterior"):
            raise ValueError("`trace` InferenceData must have `posterior` traces.")

        # check if flux has time coordinate, and add one if necessary
        if "time" not in self.flux.dims:
            self.flux = self.flux.expand_dims(time=[self.start_time])

        # change "time" dim of flux to "flux_time" to avoid NaNs when merging with obs times
        self.flux = self.flux.rename(time="flux_time")

        # align basis with flux; this is necessary due to an issue with sparse matrices
        self.basis = align_sparse_lat_lon(self.basis, self.flux)

        # if basis has time, make sure it is aligned to flux
        if "time" in self.basis.dims:
            self.basis = self.basis.rename(time="flux_time")
        elif "time" in self.basis.coords:
            # time not in dims, so just delete the coord
            self.basis = self.basis.drop_vars("time")

        self._refresh_derived_datasets()

    def _refresh_derived_datasets(self) -> None:
        """Refresh derived datasets cached from trace and observation inputs."""
        trace_ds = convert_idata_to_dataset(self.trace)

        if "longname" in self.obs.attrs:
            obs_long_name = self.obs.attrs["longname"]
        else:
            obs_long_name = self.obs.attrs.get("long_name", "observed_mole_fraction")

        _add_attributes_to_trace_dataset(trace_ds, self.obs.attrs["units"], obs_long_name)
        self.trace_ds = self.nmeasure_to_site_time(trace_ds)

        # format obs data and errors
        self.obs = self.nmeasure_to_site_time(self.obs.rename("y_obs"))
        self.obs_err = self.nmeasure_to_site_time(self.obs_err.rename("y_obs_error"))

        if self.obs_prior_factor is not None:
            self.obs_prior_factor = self.nmeasure_to_site_time(
                self.obs_prior_factor.rename("y_obs_prior_factor")
            )
        if self.obs_prior_upper_level_factor is not None:
            self.obs_prior_upper_level_factor = self.nmeasure_to_site_time(
                self.obs_prior_upper_level_factor.rename("y_obs_prior_upper_level_factor")
            )
        self.obs_repeatability = self.nmeasure_to_site_time(
            self.obs_repeatability.rename("y_obs_repeatability")
        )
        self.obs_variability = self.nmeasure_to_site_time(self.obs_variability.rename("y_obs_variability"))
        obs_inputs = [
            self.obs,
            self.obs_err,
            self.obs_prior_factor,
            self.obs_prior_upper_level_factor,
            self.obs_repeatability,
            self.obs_variability,
        ]
        self.obs_inputs = xr.merge([x for x in obs_inputs if x is not None])

    @classmethod
    def from_modern_output(cls, inv_out: InversionOutput) -> Self:
        """Build the current postprocessing carrier from a modern RHIME output.

        This temporary adapter keeps existing ``basic`` and ``paris``
        postprocessing available while issue #383/#429 move those consumers onto
        retained ``BasisFunctions`` directly.
        """
        inv_inputs = inv_out.inv_inputs
        basis = inv_out.basis_functions.operator.basis_matrix
        current_state_dim = inv_out.basis_functions.operator.meta.state_dim
        if current_state_dim != "region":
            basis = basis.rename({current_state_dim: "region"})
        if "region" in inv_inputs.coords:
            basis = basis.reindex(region=inv_inputs.region)

        nmeasure = np.arange(inv_inputs.sizes["nmeasure"])
        sites = inv_out.run_metadata.get("sites", [])
        site_names = (
            inv_inputs["site_names"]
            if "site_names" in inv_inputs
            else xr.DataArray(list(sites), dims="nsite", coords={"nsite": np.arange(len(sites))})
        )
        obs_prior_factor = inv_inputs["mf_prior_factor"] if "mf_prior_factor" in inv_inputs else None
        obs_prior_upper_level_factor = (
            inv_inputs["mf_prior_upper_level_factor"] if "mf_prior_upper_level_factor" in inv_inputs else None
        )

        def nmeasure_array(name: str, source: xr.DataArray) -> xr.DataArray:
            """Create a clean nmeasure DataArray without inherited indexes."""
            result = xr.DataArray(
                source.values,
                dims=["nmeasure"],
                coords={"nmeasure": nmeasure},
                name=name,
            )
            result.attrs = source.attrs
            return result

        species = inv_out.species
        domain = inv_out.domain
        start_date = inv_out.start_date
        end_date = inv_out.end_date
        if species is None or domain is None or start_date is None or end_date is None:
            raise ValueError("Modern InversionOutput metadata must include species, domain, start_date, and end_date.")

        return cls(
            obs=nmeasure_array("Yobs", inv_inputs["mf"]),
            obs_err=nmeasure_array("Yerror", inv_inputs["mf_error"]),
            obs_repeatability=nmeasure_array("Yerror_repeatability", inv_inputs["mf_repeatability"]),
            obs_variability=nmeasure_array("Yerror_variability", inv_inputs["mf_variability"]),
            obs_prior_factor=(
                nmeasure_array("Yobs_prior_factor", obs_prior_factor) if obs_prior_factor is not None else None
            ),
            obs_prior_upper_level_factor=(
                nmeasure_array("Yobs_prior_upper_level_factor", obs_prior_upper_level_factor)
                if obs_prior_upper_level_factor is not None
                else None
            ),
            site_indicators=nmeasure_array("site_indicator", inv_inputs["site_indicator"]),
            flux=inv_out.basis_functions.flux,
            basis=basis,
            trace=inv_out.trace,
            site_names=site_names,
            times=nmeasure_array("times", inv_inputs["time"]),
            start_date=start_date,
            end_date=end_date,
            species=species,
            domain=domain,
        )

    def __eq__(self, other: Any) -> bool:
        """Check equality between LegacyInversionOutput objects.

        The `dataclass` default `__eq__` method doesn't work because the
        `.basis` attribute is a sparse matrix, which causes problems when
        testing equality.

        Args:
            other: object to compare with

        Returns:
            True if obs and errors, flux, flat basis, trace, start/end dates,
              species, and domain are equal.

        Raises:
            NotImplementedError: if equality is tested with an object that is
              not LegacyInversionOutput.

        """
        if not isinstance(other, self.__class__):
            raise NotImplementedError

        checks = [
            (self.obs == other.obs).all(),
            (self.obs_err == other.obs_err).all(),
            (self.obs_repeatability == other.obs_repeatability).all(),
            (self.obs_variability == other.obs_variability).all(),
            (self.flux == other.flux).all(),
            (self.get_flat_basis() == other.get_flat_basis()).all(),
            (self.get_trace_dataset() == other.get_trace_dataset()).all(),
            str(self.start_date) == str(other.start_date),
            str(self.end_date) == str(other.end_date),
            self.species == other.species,
            self.domain == other.domain,
        ]
        return all(checks)

    def nmeasure_to_site_time(self, data: XrDataArrayOrSet) -> XrDataArrayOrSet:
        """Convert `nmeasure` coordinate of dataset to stacked (site, time) coordinate.

        Args:
            data: xr.DataArray or xr.Dataset

        Returns:
            data with `nmeasure` converted to a stacked (site, time) coordinate.

        """
        return _nmeasure_to_site_time(data, self.site_indicators, self.times, self.site_names)

    def get_trace_dataset(self, var_names: str | list[str] | None = None) -> xr.Dataset:
        """Return an xarray Dataset containing a prior/posterior parameter/predictive samples.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "x" will return "x_prior" and "x_posterior".

        Returns:
            xarray Dataset containing a prior/posterior parameter/predictive samples.
        """
        result = self.trace_ds

        if var_names is not None:
            result = _filter_trace_data_vars_by_name(result, var_names)

        return result

    def get_model_data(self, var_names: str | list[str] | None = None) -> xr.Dataset:
        """Return an xarray Dataset containing the data input to the model.

        This data is captured using `pm.Data`, or when data is observed.

        Args:
            convert_nmeasure: if True, convert `nmeasure` coordinate to multi-index comprising `time` and `site`.
            var_names: (list of) variables to select. For instance, "hx" or "min_error"

        Returns:
            xarray Dataset containing model data
        """
        result = convert_idata_to_dataset(self.trace, group_filters=["data"], add_suffix=False)
        result = self.nmeasure_to_site_time(result)

        if var_names is not None:
            result = filter_data_vars_by_prefix(result, var_names, sep="")

        return result

    @property
    def start_time(self) -> pd.Timestamp:
        """Start date of inversion."""
        if self.start_date is not None:
            return pd.to_datetime(self.start_date)
        return pd.to_datetime(self.times.min().values[0])

    @property
    def end_time(self) -> pd.Timestamp:
        """End date of inversion."""
        if self.end_date is not None:
            return pd.to_datetime(self.end_date)
        return pd.to_datetime(self.times.max().values[0])

    @property
    def period_midpoint(self) -> pd.Timestamp:
        """Midpoint of inversion period."""
        return self.start_time + (self.end_time - self.start_time) / 2

    def get_total_err(self, take_mean: bool = True) -> xr.DataArray:
        """Return the posterior model-data mismatch error.

        This is the variable `epsilon` in the RHIME model. It can be thought of
        as sqrt(repeatability**2 + variability**2 + model_error**2), although the
        actual definition is more complicated.

        Args:
            take_mean: if True, take mean over trace of error term, otherwise
              return the full trace.

        Returns:
            xr.DataArray containing total error

        """
        result = self.get_trace_dataset(var_names="epsilon").epsilon_posterior

        if take_mean:
            result = result.mean("draw")

        result.attrs["units"] = self.obs.attrs["units"]
        result.attrs["long_name"] = "total model-data mismatch error"

        return result.rename("total_error")

    def get_model_err(self) -> xr.DataArray:
        """Return model_error.

        The model error is calculated by subtracting the square of the obs error
        from the square of the total error, and then taking a square root.

        Returns:
            xr.DataArray containing model error

        """
        total_err = self.get_total_err(take_mean=False)
        total_obs_err = self.obs_err

        result = np.sqrt(np.maximum(total_err**2 - total_obs_err**2, 0)).mean("draw")  # type: ignore
        result.attrs["units"] = self.obs.attrs["units"]
        result.attrs["long_name"] = "inferred model error"
        return result.rename("model_error")

    def get_obs_and_errors(self) -> xr.Dataset:
        """Return dataset containing observations and related error terms.

        The dataset return contains: obs, obs error (as used by the inversion),
        obs repeatability and variability, model error, and total error (i.e.
        the model-data mismatch error).

        Returns:
            xr.Dataset containing obs and error data

        """
        result = xr.merge([self.obs_inputs, self.get_model_err(), self.get_total_err()])
        result.attrs = {}

        return result

    def get_flat_basis(self) -> xr.DataArray:
        """Return 2D DataArray encoding basis regions.

        The `LegacyInversionOutput.basis` matrix is sparse, with three dimensions: latitude, longitude, and region
        (which corresponds to a basis function). A sparse matrix cannot be saved directly to disk, or compared
        with other matrices of this type, and converting directly to a dense matrix could use a very large
        amount of memory.

        This function converts the basis matrix to a 2D array with latitude and longitude coordinates, and basis
        regions encoded by numbers in this 2D array.

        Returns:
            xr.DataArray encoding basis functions, with latitude and longitude coordinates

        """
        if len(self.basis.dims) == 2:
            return self.basis

        region_dim = next(
            str(dim) for dim in self.basis.dims if dim not in ["lat", "lon", "latitude", "longitude"]
        )

        return (self.basis * self.basis[region_dim]).sum(region_dim).as_numpy().rename("basis")

    def to_datatree(self) -> xr.DataTree:
        """Convert LegacyInversionOutput to xarray DataTree.

        The output of this method can be saved to netCDF or zarr.

        To make it possible to save the data, the `nmeasure` multi-index needs to be removed.
        The multi-index is restored by the `from_datatree` method.

        Returns:
            xr.DataTree containing the trace (as a sub-DataTree), obs and errors, the flat basis
              functions, and the flux, as well as the start/end dates, species, and domain in its
              attributes.

        """
        to_merge = [
            self.obs,
            self.obs_err,
            self.obs_prior_factor,
            self.obs_prior_upper_level_factor,
            self.obs_repeatability,
            self.obs_variability,
        ]
        to_merge = [x for x in to_merge if x is not None]
        dt_dict = {
            "trace": _inferencedata_to_datatree(self.trace),
            "obs_and_errors": xr.merge(to_merge).reset_index("nmeasure"),
            "basis": self.get_flat_basis().to_dataset(),
            "flux": self.flux.rename(flux_time="time").rename("flux").to_dataset(),
        }
        dt = xr.DataTree.from_dict(dt_dict)
        dt.attrs = {
            "schema": LEGACY_INVERSION_OUTPUT_SCHEMA,
            "schema_version": 1,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "species": self.species,
            "domain": self.domain,
        }
        return dt

    def save(self, output_file: str | Path, output_format: Literal["netcdf", "zarr"] | None = None) -> None:
        """Save LegacyInversionOutput to netCDF or Zarr.

        There is a corresponding `load` method to recover the LegacyInversionOutput
        from a saved version.

        Args:
            output_file: path to file where the LegacyInversionOutput should be saved
            output_format: format to save to; if `None`, this will be inferred by the
              extension of `output_file`

        Raises:
            ValueError: If `output_format` is not specified and cannot be inferred
              from the output file extension.

        """
        _save_datatree(self.to_datatree(), output_file, output_format)

    @classmethod
    def from_datatree(cls: type[Self], dt: xr.DataTree) -> Self:
        """Construct LegacyInversionOutput from serialised DataTree.

        This method is the inverse of `to_datatree`.

        Args:
            dt: xr.DataTree constructed using `LegacyInversionOutput.to_datatree`

        Returns:
            LegacyInversionOutput: reconstructed from datatree

        """
        schema = dt.attrs.get("schema")
        if schema is not None and schema != LEGACY_INVERSION_OUTPUT_SCHEMA:
            raise ValueError(f"Unexpected LegacyInversionOutput schema: {schema!r}")

        obs_and_errs_ds = dt.obs_and_errors.to_dataset().drop_vars(["site", "time"])
        obs_and_errs = dict(obs_and_errs_ds)
        obs_and_errs = {
            k.replace("y_", "").replace("obs_error", "obs_err"): v for k, v in obs_and_errs.items()
        }
        inv_info = {
            "start_date": dt.attrs.get("start_date"),
            "end_date": dt.attrs.get("end_date"),
            "species": dt.attrs.get("species"),
            "domain": dt.attrs.get("domain"),
        }
        if any(value is None for value in inv_info.values()):
            raise ValueError("LegacyInversionOutput DataTree is missing required inversion metadata.")
        basis_dim = "nx" if "nx" in dt.trace.posterior.coords else "region"
        basis = get_xr_dummies(
            dt.basis.basis,
            cat_dim=basis_dim,
            categories=dt.trace.posterior[basis_dim],
        )
        trace = _inferencedata_from_datatree(dt.trace)
        return cls(
            obs=obs_and_errs["obs"],
            obs_err=obs_and_errs["obs_err"],
            obs_repeatability=obs_and_errs["obs_repeatability"],
            obs_variability=obs_and_errs["obs_variability"],
            obs_prior_factor=obs_and_errs.get("obs_prior_factor"),
            obs_prior_upper_level_factor=obs_and_errs.get("obs_prior_upper_level_factor"),
            flux=dt.flux.flux,
            basis=basis,
            trace=trace,
            site_indicators=dt.obs_and_errors.site,
            times=dt.obs_and_errors.time,
            start_date=str(inv_info["start_date"]),
            end_date=str(inv_info["end_date"]),
            species=str(inv_info["species"]),
            domain=str(inv_info["domain"]),
        )

    @classmethod
    def load(cls: type[Self], file_path: str | Path) -> Self:
        """Load LegacyInversionOutput from file.

        Use this to load `LegacyInversionOutput` that was previously saved using
        `LegacyInversionOutput.save`.

        Args:
            file_path: path to saved LegacyInversionOutput

        Returns:
            LegacyInversionOutput loaded from saved file

        """
        return cls.from_datatree(_open_datatree_loaded(file_path))


def make_inv_out_for_fixed_basis_mcmc(
    fp_data: dict,
    Y: np.ndarray,
    Ytime: np.ndarray,
    error: np.ndarray,
    obs_repeatability: np.ndarray,
    obs_variability: np.ndarray,
    site_indicator: np.ndarray,
    site_names: np.ndarray | list[str],  # could be a list?
    mcmc_results: dict,
    start_date: str,
    end_date: str,
    species: str,
    domain: str,
    obs_prior_factor: np.ndarray | None = None,
    obs_prior_upper_level_factor: np.ndarray | None = None,
) -> LegacyInversionOutput:
    """Create LegacyInversionOutput in `fixedbasisMCMC`."""
    nmeasure = np.arange(len(Y))
    y_obs = xr.DataArray(Y, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yobs")
    times = xr.DataArray(Ytime, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="times")
    y_error = xr.DataArray(error, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror")
    y_obs_prior_factor = (
        xr.DataArray(
            obs_prior_factor, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yobs_prior_factor"
        )
        if obs_prior_factor is not None
        else None
    )
    y_obs_prior_upper_level_factor = (
        xr.DataArray(
            obs_prior_upper_level_factor,
            dims=["nmeasure"],
            coords={"nmeasure": nmeasure},
            name="Yobs_prior_upper_level_factor",
        )
        if obs_prior_upper_level_factor is not None
        else None
    )
    y_error_repeatability = xr.DataArray(
        obs_repeatability, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror_repeatability"
    )
    y_error_variability = xr.DataArray(
        obs_variability, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="Yerror_variability"
    )
    site_indicator_da = xr.DataArray(
        site_indicator, dims=["nmeasure"], coords={"nmeasure": nmeasure}, name="site_indicator"
    )
    site_names_da = xr.DataArray(
        site_names, dims=["nsite"], coords={"nsite": np.arange(len(site_names))}, name="site_names"
    )

    _, nx = mcmc_results["xouts"].shape
    nx = np.arange(nx)

    basis = get_xr_dummies(fp_data[".basis"], cat_dim="nx", categories=nx)

    scenarios = [v for k, v in fp_data.items() if not k.startswith(".")]

    try:
        flux = scenarios[0].flux_stacked
    except AttributeError:
        flux_source = next(iter(fp_data[".flux"].values()))
        flux = getattr(flux_source, "data", flux_source)

    # TODO: this only works if there is one flux used (or if multiple, but ModelScenario stacks them)
    if isinstance(flux, xr.Dataset):
        flux = flux.flux if "flux" in flux else flux[flux.data_vars[0]]

    if not isinstance(flux, xr.DataArray):
        raise ValueError("Flux from `fp_data` could not be converted to a xr.DataArray.")

    # add attributes
    scenario = scenarios[0]
    y_obs.attrs = scenario.mf.attrs
    times.attrs = scenario.time.attrs
    y_error.attrs = scenario.mf_error.attrs
    y_error_variability.attrs = scenario.mf_variability.attrs
    y_error_repeatability.attrs = scenario.mf_repeatability.attrs
    for scenario in scenarios:
        if not ("mf_prior_factor" in scenario and "mf_prior_upper_level_factor" in scenario):
            continue
        if y_obs_prior_factor is not None:
            y_obs_prior_factor.attrs = scenario.mf_prior_factor.attrs
        if y_obs_prior_upper_level_factor is not None:
            y_obs_prior_upper_level_factor.attrs = scenario.mf_prior_upper_level_factor.attrs

    return LegacyInversionOutput(
        obs=y_obs,
        obs_err=y_error,
        obs_prior_factor=y_obs_prior_factor,
        obs_prior_upper_level_factor=y_obs_prior_upper_level_factor,
        obs_repeatability=y_error_repeatability,
        obs_variability=y_error_variability,
        site_indicators=site_indicator_da,
        flux=flux,
        basis=basis,
        trace=mcmc_results["trace"],
        site_names=site_names_da,
        times=times,
        start_date=start_date,
        end_date=end_date,
        species=species,
        domain=domain,
    )


# Functions to re-run post-processing on standard RHIME outputs
#
# This is a temporary fix. A more general fix will be possible when we can recreate the
# RHIME model using the modular model set-up.
#
# In particular, there are not prior or predictive traces, so only flux and country totals
# can be recomputed.
def _clean_rhime_output(ds: xr.Dataset) -> xr.Dataset:
    """Take raw RHIME output and rename/drop/create variables to get dataset ready for further processing."""
    use_bc = "bctrace" in ds.data_vars

    rename_vars_dict = dict(stepnum="draw", paramnum="nlatent", measurenum="nmeasure")

    rename_dict = {
        "nsites": "nsite",
        "nparam": "nx",
        "xtrace": "x",
        "sigtrace": "sigma",
    }

    if use_bc:
        rename_vars_dict["numBC"] = "nBC"
        rename_dict.update({"bctrace": "bc", "nBC": "nbc"})

    ds = (
        ds.rename_vars(rename_vars_dict)
        .drop_dims(["nUI", "nlatent"])
        .swap_dims(nsite="nsites", steps="draw")
        .rename(rename_dict)
    )
    ds["x"] = ds.x.assign_coords(nx=("nx", ds.basisfunctions.to_series().sort_values().unique()))

    data_vars = [
        "Yobs",
        "Yerror",
        "Yerror_repeatability",
        "Yerror_variability",
        "Ytime",
        "x",
        "sigma",
        "siteindicator",
        "sigmafreqindex",
        "sitenames",
        "fluxapriori",
        "basisfunctions",
        "xsensitivity",
    ]

    if "Yobs_prior_factor" in ds.data_vars or "Yobs_prior_upper_level_factor" in ds.data_vars:
        data_vars.extend(["Yobs_prior_factor", "Yobs_prior_upper_level_factor"])
    if use_bc:
        data_vars.extend(["bc", "bcsensitivity"])

    ds = ds[data_vars]

    return ds


def _make_idata_from_rhime_outs(rhime_out_ds: xr.Dataset) -> az.InferenceData:
    """Create arviz InferenceData with posterior group created from RHIME output."""
    trace_dvs = [dv for dv in rhime_out_ds.data_vars if "draw" in list(rhime_out_ds[dv].coords)]
    traces = rhime_out_ds[trace_dvs].expand_dims({"chain": [0]})
    prior = xr.ones_like(traces[["x"]])
    constant_data = rhime_out_ds[["xsensitivity"]]
    constant_data["min_model_error"] = xr.DataArray(
        rhime_out_ds.attrs["min_model_error"], coords={"nmeasure": rhime_out_ds.nmeasure}, dims="nmeasure"
    )
    return az.InferenceData(posterior=traces, prior=prior, constant_data=constant_data)


def make_inv_out_from_rhime_outputs(
    ds: xr.Dataset, species: str, domain: str, start_date: str | None = None, end_date: str | None = None
) -> LegacyInversionOutput:
    """Create inversion output from RHIME outputs.

    This can be used to re-run flux and country total outputs using the PARIS postprocessing.
    However, this doesn't recover enough information to re-compute concentration outputs.
    """
    flux = ds.fluxapriori

    ds_clean = _clean_rhime_output(ds)
    site_indicators = ds_clean.siteindicator
    basis = get_xr_dummies(ds_clean.basisfunctions, cat_dim="nx", categories=ds_clean.nx.values)

    start_date = str(start_date or ds_clean.Ytime.min().values)
    end_date = str(end_date or ds_clean.Ytime.max().values)

    trace = _make_idata_from_rhime_outs(ds_clean)

    return LegacyInversionOutput(
        obs=ds_clean.Yobs,
        obs_err=ds_clean.Yerror,
        obs_prior_factor=getattr(ds_clean, "Yobs_prior_factor", None),
        obs_prior_upper_level_factor=getattr(ds_clean, "Yobs_prior_upper_level_factor", None),
        obs_repeatability=ds_clean.Yerror_repeatability,
        obs_variability=ds_clean.Yerror_variability,
        flux=flux,
        basis=basis,
        trace=trace,
        site_indicators=site_indicators,
        site_names=ds_clean.sitenames,
        times=ds_clean.Ytime,
        species=species,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
    )
