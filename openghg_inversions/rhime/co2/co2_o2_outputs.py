"""Scientific products from one saved linked CO2/O2 posterior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path

import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.operators import BucketBasisOperator
from openghg_inversions.postprocessing.stats import calculate_stats
from openghg_inversions.rhime._stage_artifacts import _validate_filename_component, check_checkpoint_targets
from openghg_inversions.rhime.outputs import _define_output_filename, _resolve_output_path
from openghg_inversions.serialization import reset_serialisation_multiindexes, save_datatree, save_trace
from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs

from .co2_o2_preparation import Co2O2PreparedInputs


@dataclass
class Co2O2Output:
    """Borrowed linked posterior, preparation, reporting window and products.

    ``idata`` retains one joint posterior and its covariance; channel products
    are marginals with shared draw labels. Replay has no live inference object.
    """

    prepared: Co2O2PreparedInputs
    idata: xr.DataTree
    start_date: str
    end_date: str
    domain: str
    output_metadata: dict = field(default_factory=dict)
    outputs: dict[str, xr.Dataset] = field(default_factory=dict)
    sampler: None = field(default=None, init=False)


def make_co2_o2_output(
    *, prepared: Co2O2PreparedInputs, trace: xr.DataTree,
    start_date: str, end_date: str, domain: str,
    output_metadata: Mapping | None = None,
) -> Co2O2Output:
    """Join borrowed scientific inputs and draws without graph execution."""
    validate_co2_o2_trace(trace, prepared)
    return Co2O2Output(prepared, trace, start_date, end_date, domain, dict(output_metadata or {}))


@dataclass(frozen=True)
class Co2O2OutputOptions:
    """Linked concentration, state and supported bucket-native reporting choices.

    ``basic`` accepts posterior-only draws; ``paris`` requires saved prior draws.
    ``native_flux_bases`` maps co2/o2 to explicit ``BasisFunctions`` or saved basis
    paths, for the existing conditional single-source bucket PARIS products.
    ``save_concentrations`` also exports full component draws. Paths resolve from
    the invocation directory; validation and calculation precede writing.
    ``none`` suppresses derived products while honoring an explicit ``save_trace``.
    """

    output_format: str = "basic"
    output_path: str | Path | None = None
    output_name: str = "rhime"
    save_trace: str | Path | bool = False
    save_concentrations: bool = False
    country_file: str | Path | None = None
    paris_postprocessing_kwargs: Mapping | None = None
    native_flux_bases: Mapping[str, BasisFunctions | str | Path] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_format, str) or self.output_format not in {"none", "basic", "paris"}:
            raise ValueError("Linked outputs support only 'none', 'basic', or 'paris'.")
        _validate_filename_component("output_name", self.output_name)
        for name in ("output_path", "country_file", "save_trace"):
            value = getattr(self, name)
            if value is None or (name == "save_trace" and isinstance(value, bool)):
                continue
            if not isinstance(value, str | Path):
                raise TypeError(f"Linked {name} must be a filesystem path.")
            object.__setattr__(self, name, Path(value).expanduser().resolve())
        if not isinstance(self.save_concentrations, bool):
            raise TypeError("Linked save_concentrations must be boolean.")
        if self.save_trace is True and self.output_path is None:
            raise ValueError("Linked output_path is required when save_trace=True.")
        kwargs = self.paris_postprocessing_kwargs
        if kwargs is not None:
            if not isinstance(kwargs, Mapping):
                raise TypeError("Linked paris_postprocessing_kwargs must be a mapping.")
            supported = {"template_version", "obs_avg_period", "flux_frequency", "country_selections"}
            if unknown := kwargs.keys() - supported:
                raise ValueError(f"Unsupported linked PARIS options: {sorted(unknown)!r}.")
            if kwargs.get("template_version", "latest") != "latest":
                raise ValueError("Linked PARIS products support only template_version='latest'.")
            object.__setattr__(self, "paris_postprocessing_kwargs", dict(kwargs))
        bases = self.native_flux_bases
        if bases is not None:
            if not isinstance(bases, Mapping) or bases.keys() - {"co2", "o2"}:
                raise ValueError("Linked native_flux_bases must map co2/o2 to explicit bases or paths.")
            resolved = {}
            for species, basis in bases.items():
                if not isinstance(basis, BasisFunctions | str | Path):
                    raise TypeError("Linked native_flux_bases values must be BasisFunctions or paths.")
                resolved[species] = Path(basis).expanduser().resolve() if isinstance(basis, str | Path) else basis
            object.__setattr__(self, "native_flux_bases", resolved)
        if self.output_format == "basic" and (self.native_flux_bases or self.country_file):
            raise ValueError("Linked native flux and country reporting require output_format='paris'.")
        if self.country_file and not self.native_flux_bases:
            raise ValueError("Linked country reporting requires explicit native_flux_bases.")

    @classmethod
    def from_params(cls, params: Mapping) -> Co2O2OutputOptions:
        """Resolve only the linked recipe's supported output settings."""
        if unknown := params.keys() - cls.__dataclass_fields__.keys():
            raise ValueError(f"Unsupported linked output option(s): {sorted(unknown)!r}.")
        return cls(**params)


def resolve_co2_o2_output(params: Mapping) -> dict[str, Co2O2OutputOptions]:
    """Resolve the outputs section independently of model or inference options."""
    from .co2_o2_configuration import _validate_co2_o2_option_names

    _validate_co2_o2_option_names(params)
    return {"output": Co2O2OutputOptions.from_params(params.get("outputs", {}))}


def validate_co2_o2_trace(trace: xr.DataTree, prepared: Co2O2PreparedInputs | None = None) -> None:
    """Check linked roles and exact saved labels without constructing a model.

    A posterior requires full state scaling, modelled concentrations and total
    marginal errors. Present prior groups require full states and any declared
    baselines. Optional predictive draws are checked against observation labels.
    """
    from openghg_inversions.postprocessing.linked_paris_outputs import _role

    if trace.attrs.get("rhime_recipe") not in {"co2_o2", "co2_o2_cached_sigma_fixed_ou"}:
        raise ValueError("Linked outputs require one joint co2_o2 posterior.")
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    if not isinstance(roles, dict) or any(not isinstance(value, str) or not value for value in roles.values()):
        raise ValueError("Linked scientific roles must map to nonempty variable names.")
    required = {"flux_scale", "modelled_concentration", "total_marginal_error"}
    if missing := required - roles.keys():
        raise ValueError(f"Linked trace lacks scientific role(s): {sorted(missing)!r}.")
    if "posterior" not in trace.children:
        raise ValueError("Linked outputs require posterior draws.")
    for group in ("prior", "posterior"):
        if group not in trace.children:
            continue
        needed = {"flux_scale"} if group == "prior" else required
        needed = needed | ({"boundary_concentration", "offset_concentration"} & roles.keys())
        for role in needed:
            value = _role(trace, role, group)
            sample_dims = {f"prior_{dim}" if group == "prior" else dim for dim in ("chain", "draw")}
            if role in {"flux_scale", "modelled_concentration"} and not sample_dims.issubset(value.dims):
                raise ValueError(f"Linked role {role!r} in {group} requires chain/draw dimensions.")
            if prepared is not None:
                expected = prepared.retained_prior.mean if role == "flux_scale" else prepared.observations
                if set(value.dims) - sample_dims != set(expected.dims):
                    raise ValueError(f"Linked role {role!r} in {group} has incompatible dimensions.")
                xr.align(value, expected, join="exact", copy=False)
                coordinates = ("source", "tracer_scope") if role == "flux_scale" else ("observation_units",)
                for name in coordinates:
                    if name not in value.coords or not value[name].variable.equals(expected[name].variable):
                        raise ValueError(f"Linked role {role!r} in {group} has incompatible {name} labels.")
    if prepared is not None:
        for group in ("prior_predictive", "posterior_predictive"):
            if group in trace.children and "concentration" in roles:
                value = _role(trace, "concentration", group)
                if set(value.dims) != {"chain", "draw", "observation"}:
                    raise ValueError(f"Linked concentration in {group} requires chain/draw/observation dimensions.")
                xr.align(value, prepared.observations, join="exact", copy=False)
                if "observation_units" not in value.coords or not value.observation_units.variable.equals(
                    prepared.observations.observation_units.variable
                ):
                    raise ValueError(f"Linked concentration in {group} has incompatible observation_units labels.")


def validate_co2_o2_product_request(
    output: Co2O2OutputOptions, prepared: Co2O2PreparedInputs,
) -> dict[str, BasisFunctions]:
    """Validate common-unit reporting and load requested native reporting bases.

    This boundary creates no model and opens no output destination. Native
    products retain the existing single-source bucket representation; the PARIS
    owner checks signed private-tracer exclusions when constructing products.
    """
    from .co2_o2_configuration import validate_co2_o2_units

    if output.output_format == "none":
        return {}
    validate_co2_o2_units(prepared)
    bases = {}
    for species, value in (output.native_flux_bases or {}).items():
        basis = BasisFunctions.load(value) if isinstance(value, str | Path) else value
        if basis.source_labels is not None or not isinstance(basis.operator, BucketBasisOperator):
            raise ValueError("Linked native PARIS flux requires a single-source bucket total-flux basis.")
        matrix = basis.operator.basis_matrix.rename({basis.operator.meta.state_dim: prepared.retained_prior.state_dim})
        xr.align(matrix, prepared.retained_prior.mean, join="exact", copy=False)
        bases[species] = basis
    return bases


def _summaries(components: xr.Dataset) -> xr.Dataset:
    """Compute related summaries together within each independent sample group."""
    groups: dict[tuple, list] = {}
    for name, value in components.data_vars.items():
        axes = tuple(sorted(str(dim) for dim in value.dims if str(dim).endswith(("chain", "draw"))))
        groups.setdefault(axes, []).append(name)
    summaries = []
    for axes, names in groups.items():
        selected = components[names]
        sample_dims = {str(dim): "chain" if str(dim).endswith("chain") else "draw"
                       for dim in axes}
        if sample_dims:
            summaries.append(calculate_stats(selected.rename(sample_dims), stats=["mean", "stdev", "quantiles"]))
        else:
            summaries.append(selected)
    result = xr.merge(summaries, compat="no_conflicts")
    result.attrs = dict(components.attrs)
    return result


def calculate_co2_o2_products(*, result: Co2O2Output, output: Co2O2OutputOptions) -> None:
    """Attach labelled channel and state products without model construction or inference.

    Basic products support posterior-only traces. PARIS preserves its saved-prior
    requirement and conditional bucket-native limitations. Predictions come only
    from saved predictive groups. Product validation finishes before writing.
    """
    from openghg_inversions.postprocessing.linked_paris_outputs import (
        _role, make_co2_o2_paris_outputs, reconstruct_co2_o2_concentrations,
    )

    if output.output_format == "none":
        result.outputs.clear()
        return
    bases = validate_co2_o2_product_request(output, result.prepared)
    components = reconstruct_co2_o2_concentrations(result.idata, result.prepared, require_prior=False)
    metadata = {
        "rhime_variable_roles": result.idata.attrs["rhime_variable_roles"],
        "rhime_model_metadata": result.idata.attrs.get("rhime_model_metadata", "{}"),
        "linked_preparation_provenance": json.dumps(dict(result.prepared.provenance), sort_keys=True),
        "linked_ratio_provenance": result.prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"],
    }
    products = {}
    for species, values in components.items():
        values.attrs.update(metadata)
        products[f"{species}_concentration_components"] = values
        products[f"{species}_basic"] = _summaries(values)
    states = xr.Dataset({f"flux_scale_{group}": _role(result.idata, "flux_scale", group)
                         for group in ("prior", "posterior") if group in result.idata.children})
    states.attrs.update(metadata, units="dimensionless flux scale")
    products["state_scaling"] = states
    if output.output_format == "paris":
        paris = make_co2_o2_paris_outputs(
            result.idata, result.prepared, domain=result.domain,
            start_date=result.start_date, end_date=result.end_date,
            native_flux_bases=bases, country_file=output.country_file,
            **dict(output.paris_postprocessing_kwargs or {}),
        )
        for species, channel in paris.items():
            for product, value in channel.items():
                products[f"{species}_paris_{product}"] = value
    result.outputs.clear()
    result.outputs.update(products)


def save_co2_o2_products(
    result: Co2O2Output, *, output: Co2O2OutputOptions, referenced: tuple[Path, ...] = (),
) -> None:
    """Export requested products and traces after rejecting aliases and containment overlaps."""
    if output.output_format == "none" and not output.save_trace:
        return
    _validate_filename_component("domain", result.domain)
    _validate_filename_component("start_date", result.start_date)
    paths = {}
    if output.output_format != "none" and output.output_path is not None:
        for name, product in result.outputs.items():
            if name.endswith("_concentration_components") and not output.save_concentrations:
                continue
            species = name.split("_", 1)[0] if name.startswith(("co2_", "o2_")) else "co2_o2"
            path = _define_output_filename(output.output_path, species, result.domain,
                                           f"{output.output_name}_{name}", result.start_date,
                                           filename_convention="rhime")
            paths[name] = path
    trace_path = _resolve_output_path(output.save_trace, output.output_path,
                                      f"{output.output_name}{result.start_date}_trace.nc")
    reporting_inputs = [value for value in (output.native_flux_bases or {}).values() if isinstance(value, Path)]
    if output.country_file is not None:
        reporting_inputs.append(Path(output.country_file))
    protected = tuple(path.resolve() for path in (*referenced, *reporting_inputs))
    targets = [*paths.values(), *([trace_path] if trace_path is not None else [])]
    for path in targets:
        check_checkpoint_targets(path.parent, stage=None, payloads=(path.name,), referenced=protected)
        target = path.resolve()
        if any(target.is_relative_to(artifact) or artifact.is_relative_to(target) for artifact in protected):
            raise ValueError(f"Linked output target {target} would overlap another input or output artifact.")
        protected += (target,)
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        product = result.outputs[name]
        if "_paris_" in name:
            write_netcdf_preserving_bounds_attrs(product, path)
        else:
            save_datatree(xr.DataTree(reset_serialisation_multiindexes(product)), path)
        result.output_metadata[f"{name}_path"] = str(path)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        save_trace(result.idata, trace_path)
        result.output_metadata["trace_path"] = str(trace_path)


__all__ = [
    "Co2O2Output", "Co2O2OutputOptions", "make_co2_o2_output",
    "resolve_co2_o2_output", "validate_co2_o2_trace", "validate_co2_o2_product_request",
    "calculate_co2_o2_products", "save_co2_o2_products",
]
