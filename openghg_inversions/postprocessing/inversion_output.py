"""Modern inversion output container and serialization helpers.

``InversionOutput`` is the durable artifact produced by modern RHIME and by
fixedbasis compatibility paths that have been routed through the modern
postprocessing flow. It stores the sampled trace, canonical inversion inputs,
retained ``BasisFunctions``, and run/model/output metadata needed to reproduce
postprocessing products.

The class deliberately stays product-neutral. Product modules such as
``make_outputs``, ``make_paris_outputs``, and ``legacy_outputs`` decide which
output formats they support, how variables are named in those formats, and
whether a product can handle multisector data. ``InversionOutput`` only exposes
modern semantic access to the underlying inputs and trace, including variable
role lookup for model-specific variable names. The current role mapping is a
small bridge until the project decides whether to use CF metadata via
``cf_xarray`` or a custom accessor.

Serialization is DataTree-based: object-specific ``to_datatree`` methods own
their durable representation, while the local helper functions expand xarray
MultiIndexes around NetCDF/Zarr limitations. Those generic helpers should move
to shared utilities once the serialization surface settles.
"""

from pathlib import Path
from collections.abc import Iterable, Mapping
from typing_extensions import Self
from dataclasses import dataclass, field
from typing import  Union, Any, Optional, TypeVar, Hashable, Literal, cast
import json
import logging
import warnings

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

XrDataArrayOrSet = Union[xr.DataArray, xr.Dataset]
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


DEFAULT_VARIABLE_ROLES: dict[str, str] = {
    "observation": "mf",
    "observation_error": "mf_error",
    "observation_prior_factor": "mf_prior_factor",
    "observation_prior_upper_level_factor": "mf_prior_upper_level_factor",
    "observation_repeatability": "mf_repeatability",
    "observation_variability": "mf_variability",
    "flux_scale": "x",
    "model_error": "epsilon",
    "concentration": "y",
    "baseline": "mu_bc",
    "offset": "offset",
    "emissions_sensitivity": "hx",
    "baseline_sensitivity": "hbc",
    "minimum_error": "min_error",
}


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
    def __post_init__(self) -> None:
        """Perform OCO2 specific alignment and coordinate transformation."""
        try:
            # 1. Access DataArrays using 0.19.0 paths
            # basis is in operator, flux is in inv_inputs
            basis_da = self.basis_functions.operator.basis
            flux_da = self.inv_inputs.flux

            # 2. Apply your custom OCO2 alignment fix
            basis_da = align_sparse_lat_lon(basis_da, flux_da)
            
            if hasattr(basis_da.data, 'todense'):
                basis_da = basis_da.copy(data=basis_da.data.todense())
                
            if "time" in basis_da.dims:
                basis_da = basis_da.rename(time="flux_time")
            elif "time" in basis_da.coords:
                basis_da = basis_da.drop_vars("time")

            # 3. Write back to frozen attributes using object.__setattr__
            object.__setattr__(self.basis_functions.operator, 'basis', basis_da)

            # 4. Process Trace Data
            trace_ds = convert_idata_to_dataset(self.trace)
            
            obs_data = self.inv_inputs.obs
            obs_units = obs_data.attrs.get("units", "unknown")
            obs_long_name = obs_data.attrs.get("longname", obs_data.attrs.get("long_name", "observed_mole_fraction"))

            _add_attributes_to_trace_dataset(trace_ds, obs_units, obs_long_name)
            
            # Map trace_ds back to self
            object.__setattr__(self, 'trace_ds', self.nmeasure_to_site_time(trace_ds))

            # 5. Format inv_inputs variables
            self.inv_inputs["obs"] = self.nmeasure_to_site_time(self.inv_inputs.obs.rename("y_obs"))
            
            err_name = "obs_error" if "obs_error" in self.inv_inputs else "obs_err"
            if err_name in self.inv_inputs:
                self.inv_inputs[err_name] = self.nmeasure_to_site_time(self.inv_inputs[err_name].rename("y_obs_error"))

            # 6. Create shortcuts for backward compatibility
            object.__setattr__(self, 'obs', self.inv_inputs.obs)
            object.__setattr__(self, 'flux', self.inv_inputs.flux)

            logging.info("InversionOutput: Custom alignment successful.")

        except Exception as e:
            logging.warning(f"InversionOutput: Post-init alignment encountered an issue: {e}")
    
    def __eq__(self, other: Any) -> bool:
        """Check equality between InversionOutput objects.

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
              not InversionOutput.

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

    def sample_predictive_distributions(self, ndraw: int | None = None) -> None:
        """Sample prior and posterior predictive distributions.

        This creates prior samples as a side-effect.

        Args:
            ndraw: optional number of prior samples to draw; defaults to the number of
              posterior samples.

        """
        if self.model is None:
            warnings.warn("Cannot sample predictive distributions without PyMC model.")
            return None

        # don't recompute if prior and predictive samples already present
        if all(group in self.trace for group in ("posterior_predictive", "prior", "prior_predictive")):
            return None

        if ndraw is None:
            ndraw = self.trace.posterior.sizes["draw"]

        self.trace.extend(pm.sample_prior_predictive(ndraw, self.model))
        self.trace.extend(pm.sample_posterior_predictive(self.trace, model=self.model, var_names=["y"]))

    def nmeasure_to_site_time(self, data: XrDataArrayOrSet) -> XrDataArrayOrSet:
        """Convert `nmeasure` coordinate of dataset to stacked (site, time) coordinate.

        Args:
            data: xr.DataArray or xr.Dataset

        Returns:
            data with `nmeasure` converted to a stacked (site, time) coordinate.

        """
        #return _nmeasure_to_site_time(data, self.site_indicators, self.times, self.site_names)
        return _nmeasure_to_site_time(
            data, 
            self.run_metadata.get("site_indicators"), 
            self.run_metadata.get("times"), 
            self.run_metadata.get("site_names")
        )
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
            result = filter_data_vars_by_prefix(result, var_names)

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
    def is_multisector(self) -> bool:
        """Whether this output represents a multisector RHIME run."""
        return bool(self.run_metadata.get("split_by_sectors"))

    @property
    def start_time(self) -> pd.Timestamp:
        """Start time for the inversion period."""
        if self.start_date is None:
            raise ValueError("InversionOutput run metadata is missing `start_date`.")
        return pd.to_datetime(self.start_date)

    @property
    def end_time(self) -> pd.Timestamp:
        """End time for the inversion period."""
        if self.end_date is None:
            raise ValueError("InversionOutput run metadata is missing `end_date`.")
        return pd.to_datetime(self.end_date)

    @property
    def period_midpoint(self) -> pd.Timestamp:
        """Midpoint of the inversion period."""
        return self.start_time + (self.end_time - self.start_time) / 2

    @property
    def flux(self) -> xr.DataArray:
        """Prior flux carried by the retained basis functions."""
        flux = self.basis_functions.flux
        if "flux_time" in flux.dims:
            return flux
        if "time" in flux.dims:
            return flux.rename(time="flux_time")
        return flux.expand_dims(flux_time=[self.start_time])

    @property
    def site_names(self) -> xr.DataArray:
        """Site names from inversion inputs or run metadata."""
        if "site_names" in self.inv_inputs:
            return self.inv_inputs["site_names"]

        sites = self.run_metadata.get("sites", [])
        return xr.DataArray(list(sites), dims="nsite", coords={"nsite": np.arange(len(sites))})

    @property
    def variable_roles(self) -> dict[str, str]:
        """Mapping from modern semantic roles to concrete model variable names."""
        roles = dict(DEFAULT_VARIABLE_ROLES)
        overrides = self.model_metadata.get("variable_roles", {})
        if overrides:
            if not isinstance(overrides, Mapping):
                raise ValueError("InversionOutput model metadata `variable_roles` must be a mapping.")
            roles.update({str(role): str(name) for role, name in overrides.items()})
        return roles

    def variable_name(self, role: str) -> str:
        """Return the concrete variable name for a semantic role."""
        try:
            return self.variable_roles[role]
        except KeyError as exc:
            raise KeyError(f"Unknown InversionOutput variable role: {role!r}") from exc

    def input_dataset(
        self,
        required_roles: Iterable[str] | str | None = None,
        *,
        optional_roles: Iterable[str] | str = (),
    ) -> xr.Dataset:
        """Return canonical inversion-input variables selected by semantic role."""
        required = self._normalise_roles(required_roles)
        optional = self._normalise_roles(optional_roles)
        selected: list[str] = []
        missing: list[str] = []

        for role in required:
            name = self.variable_name(role)
            if name in self.inv_inputs:
                selected.append(name)
            else:
                missing.append(f"{role} ({name})")

        for role in optional:
            name = self.variable_name(role)
            if name in self.inv_inputs:
                selected.append(name)

        if missing:
            raise ValueError(
                "InversionOutput.inv_inputs is missing required variable role(s): " + ", ".join(missing) + "."
            )

        return self.inv_inputs[list(dict.fromkeys(selected))]

    def trace_dataset(self, var_roles: Iterable[str] | str | None = None) -> xr.Dataset:
        """Return prior and posterior trace samples selected by semantic role."""
        result = convert_idata_to_dataset(self.trace)
        obs_name = self.variable_name("observation")
        if obs_name in self.inv_inputs:
            obs = self.inv_inputs[obs_name]
            obs_long_name = (
                obs.attrs["longname"]
                if "longname" in obs.attrs
                else obs.attrs.get("long_name", "observed_mole_fraction")
            )
            _add_attributes_to_trace_dataset(result, obs.attrs.get("units", ""), obs_long_name)

        if var_roles is not None:
            result = _filter_trace_data_vars_by_name(result, self._variable_names_for_roles(var_roles))

        return result

    def model_data(self, var_roles: Iterable[str] | str | None = None) -> xr.Dataset:
        """Return model input data from the ``InferenceData`` constant groups."""
        result = convert_idata_to_dataset(self.trace, group_filters=["data"], add_suffix=False)
        if var_roles is not None:
            result = filter_data_vars_by_prefix(result, self._variable_names_for_roles(var_roles), sep="")
        return result

    @staticmethod
    def _normalise_roles(roles: Iterable[str] | str | None) -> list[str]:
        """Return role input as a list of strings."""
        if roles is None:
            return []
        if isinstance(roles, str):
            return [roles]
        return [str(role) for role in roles]

    def _variable_names_for_roles(self, roles: Iterable[str] | str) -> list[str]:
        """Return concrete variable names for role input."""
        return [self.variable_name(role) for role in self._normalise_roles(roles)]

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
