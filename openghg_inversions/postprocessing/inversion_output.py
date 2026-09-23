"""Modern inversion output container and serialization helpers.

``InversionOutput`` is the durable artifact produced by modern RHIME. It stores
the sampled trace, canonical inversion inputs, retained ``BasisFunctions``, and
run/model/output metadata needed to reproduce postprocessing products.

The class deliberately stays product-neutral. Product modules such as
``make_outputs``, ``make_paris_outputs``, and ``legacy_outputs`` decide which
output formats they support, how variables are named in those formats, and
whether a product can handle multisector data. ``InversionOutput`` only exposes
modern semantic access to the underlying inputs and trace, including variable
role lookup for model-specific variable names. The current role mapping is a
small bridge until the project decides whether to use CF metadata via
``cf_xarray`` or a custom accessor.

Serialization is DataTree-based: object-specific ``to_datatree`` methods own
their durable representation, while shared helpers expand xarray MultiIndexes
around NetCDF/Zarr limitations.
"""

from pathlib import Path
from collections.abc import Iterable, Mapping
from typing_extensions import Self
from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Literal, cast
import json

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.serialization import (
    MULTIINDEX_DIMS_ATTR as _MULTIINDEX_DIMS_ATTR,
    open_datatree_loaded as _open_datatree_loaded,
    reset_serialisation_multiindexes as _reset_serialisation_multiindexes,
    restore_serialisation_multiindexes as _restore_serialisation_multiindexes,
    save_datatree as _save_datatree,
    trace_from_datatree as _trace_from_datatree,
    trace_to_datatree as _trace_to_datatree,
)


MODERN_INVERSION_OUTPUT_SCHEMA = "openghg_inversions.inversion_output"
MULTIINDEX_DIMS_ATTR = _MULTIINDEX_DIMS_ATTR
TRACE_SAMPLE_GROUPS = ("prior", "prior_predictive", "posterior", "posterior_predictive")
TRACE_MODEL_DATA_GROUPS = ("constant_data",)


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


def trace_group(trace: xr.DataTree, name: str) -> xr.Dataset:
    """Return one exact, direct trace group as a Dataset.

    Args:
        trace: Native xarray trace tree.
        name: Exact direct-child group name.

    Returns:
        The group's Dataset.

    Raises:
        KeyError: If ``name`` is not a direct child of ``trace``.
    """
    try:
        return trace.children[name].to_dataset()
    except KeyError as exc:
        raise KeyError(f"Trace has no direct group {name!r}.") from exc


def merge_trace_groups(
    trace: xr.DataTree,
    groups: Iterable[str] = TRACE_SAMPLE_GROUPS,
    *,
    add_suffix: bool = True,
) -> xr.Dataset:
    """Merge selected direct trace groups into a Dataset.

    Args:
        trace: Native xarray trace tree.
        groups: Exact direct-child group names to merge. Missing groups are
            ignored so posterior-only and prior-only traces remain valid.
        add_suffix: If true, suffix each data variable with its group name.

    Returns:
        Dataset containing all variables in the selected groups. Native chain
        and draw dimensions are retained.
    """
    datasets = []
    for group in groups:
        if group not in trace.children:
            continue
        dataset = trace_group(trace, group)
        if add_suffix:
            dataset = dataset.rename_vars({name: f"{name}_{group}" for name in dataset.data_vars})
        datasets.append(dataset)
    return xr.merge(datasets, join="outer")


def convert_idata_to_dataset(
    trace: xr.DataTree,
    group_filters: Iterable[str] = TRACE_SAMPLE_GROUPS,
    add_suffix: bool = True,
) -> xr.Dataset:
    """Forward the former conversion helper to exact DataTree group merging.

    Args:
        trace: Native xarray trace tree.
        group_filters: Exact direct-child group names to merge.
        add_suffix: If true, suffix each data variable with its group name.

    Returns:
        Dataset containing variables from the selected groups.
    """
    return merge_trace_groups(trace, group_filters, add_suffix=add_suffix)


def _add_attributes_to_trace_dataset(trace_ds: xr.Dataset, obs_units: str, obs_longname: str) -> None:
    """Add attributes to trace dataset.

    Args:
        trace_ds: Trace dataset, probably created by ``merge_trace_groups``.
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
    "boundary": "mu_bc",
    "offset": "offset",
    "emissions_sensitivity": "hx",
    "baseline_sensitivity": "hbc",
    "minimum_error": "min_error",
}


@dataclass
class InversionOutput:
    """Modern RHIME inversion output contract.

    This object carries the runtime artifacts needed to reproduce and extend
    RHIME outputs without exposing model-runner implementation details.

    Args:
        trace: Native xarray trace tree with ArviZ-compatible direct groups.
        inv_inputs: Canonical labelled inversion inputs.
        basis_functions: Retained basis functions used by the inversion.
        run_metadata: Run configuration and temporal metadata.
        model_metadata: Model identity, variable roles, and scientific metadata.
        output_metadata: Product and persistence metadata.
        provenance: Source and processing provenance.
    """

    trace: xr.DataTree
    inv_inputs: xr.Dataset
    basis_functions: BasisFunctions
    run_metadata: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    output_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def select_chain(self, index: int) -> Self:
        """Return a derived-output view containing one chain.

        Args:
            index: Zero-based chain position to retain.

        Returns:
            A shallowly replaced output whose trace retains only that chain.
        """
        return replace(self, trace=self.trace.isel(chain=[index]))

    def trace_group(self, name: str) -> xr.Dataset:
        """Return one exact, direct trace group as a Dataset.

        Args:
            name: Exact direct-child group name.

        Returns:
            The group's Dataset.

        Raises:
            KeyError: If ``name`` is not a direct trace group.
        """
        return trace_group(self.trace, name)

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

    @property
    def state_dimension_mapping(self) -> dict[str, str] | None:
        """Explicit selected-trace to retained-basis state-dimension mapping."""
        raw_mapping = self.model_metadata.get("state_dimension_mapping")
        if raw_mapping is None:
            return None
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("InversionOutput model metadata `state_dimension_mapping` must be a mapping.")
        mapping = {str(key): str(value) for key, value in raw_mapping.items()}
        if set(mapping) != {"trace", "basis"}:
            raise ValueError(
                "InversionOutput state-dimension mapping requires exactly the keys 'trace' and 'basis'."
            )
        operator_state_dim = self.basis_functions.operator.meta.state_dim
        if mapping["basis"] != operator_state_dim:
            raise ValueError(
                "InversionOutput state-dimension mapping does not match the retained basis operator: "
                f"metadata={mapping['basis']!r}, operator={operator_state_dim!r}."
            )
        return mapping

    def _normalise_selected_trace_state_dimension(self, trace: xr.Dataset) -> xr.Dataset:
        """Rename one selected recipe state dimension to its basis dimension."""
        mapping = self.state_dimension_mapping
        if mapping is None or mapping["trace"] == mapping["basis"]:
            return trace
        trace_state_dim = mapping["trace"]
        basis_state_dim = mapping["basis"]
        if trace_state_dim not in trace.dims:
            return trace
        if basis_state_dim in trace.dims:
            raise ValueError(
                "Selected trace contains both mapped state dimensions "
                f"{trace_state_dim!r} and {basis_state_dim!r}; select one domain before reconstruction."
            )
        return trace.rename({trace_state_dim: basis_state_dim})

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
        result = merge_trace_groups(self.trace)
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
            result = self._normalise_selected_trace_state_dimension(result)

        return result

    def model_data(self, var_roles: Iterable[str] | str | None = None) -> xr.Dataset:
        """Return model input data from exact constant-data trace groups."""
        result = merge_trace_groups(self.trace, TRACE_MODEL_DATA_GROUPS, add_suffix=False)
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
                "trace": _trace_to_datatree(self.trace),
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

        trace = _trace_from_datatree(cast(xr.DataTree, dt["trace"]))
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
