"""Scientific output adapters for the ordinary and cached CO2 recipes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.postprocessing.co2_flux_outputs import (
    co2_country_flux_outputs,
    co2_native_flux_outputs,
)
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import trace_group
from openghg_inversions.postprocessing.stats import calculate_stats
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.outputs import (
    RhimeResult,
    _define_output_filename,
    _save_requested_trace,
)
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeOutputSpec, RhimeRunSpec
from openghg_inversions.serialization import reset_serialisation_multiindexes, save_datatree
from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs

from .co2_affine_output import BoundCo2AffineFluxMap
from .co2_preparation import Co2PreparedInputs, _require_equivalent_units


@dataclass
class Co2Output:
    """Borrowed CO2 scientific output and its requested reporting window.

    ``bound`` preserves conditional affine reconstruction rather than standard
    multiplicative flux scaling. ``idata`` remains the borrowed full DataTree,
    including distinct sample groups and their coordinates. Inference provenance
    remains descriptive in ``output_metadata``; no live sampler or writing policy
    belongs to this scientific view. ``sampler=None`` explicitly represents the
    absent inference object required by the replay contract. Products remain accessible in
    ``outputs``. Reporting choices are supplied to calculation/export operations.
    """

    prepared: Co2PreparedInputs
    idata: xr.DataTree
    start_date: str
    end_date: str
    domain: str
    bound: BoundCo2AffineFluxMap | None = None
    sampler: None = field(default=None, init=False)
    outputs: dict = field(default_factory=dict)
    output_metadata: dict = field(default_factory=dict)


def validate_co2_trace(trace: xr.DataTree, prepared: Co2PreparedInputs | None = None) -> None:
    """Validate saved output roles and labelled reconstruction without computing draws."""
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    if not isinstance(roles, dict) or any(not isinstance(value, str) or not value for value in roles.values()):
        raise ValueError("CO2 trace requires scientific role names mapped to variable names.")
    required = {"model_mean", "pollution_concentration", "flux_scale"}
    missing = required - roles.keys()
    if missing:
        raise ValueError(f"CO2 trace lacks scientific role(s): {sorted(missing)!r}.")
    if "posterior" not in trace.children:
        raise ValueError("CO2 samples require posterior draws.")
    for group in ("prior", "posterior"):
        if group not in trace.children:
            continue
        dataset = trace[group].to_dataset()
        for role in required | ({"boundary_concentration", "offset_concentration", "model_error"} & roles.keys()):
            if roles[role] not in dataset:
                raise ValueError(f"CO2 scientific outputs require role {role!r} in {group}.")
            variable = dataset[roles[role]]
            if not {"chain", "draw"}.issubset(variable.dims):
                raise ValueError(f"CO2 role {role!r} in {group} requires chain/draw sample dimensions.")
            if prepared is not None:
                state_dim = prepared.inv_inputs["alpha_prior_mean"].dims[0]
                expected_dim = state_dim if role == "flux_scale" else "nmeasure"
                if set(variable.dims) != {"chain", "draw", expected_dim}:
                    raise ValueError(f"CO2 role {role!r} in {group} requires only chain/draw and dimension {expected_dim!r}.")
                expected_units = "1" if role == "flux_scale" else prepared.inv_inputs["mf"].attrs["units"]
                _require_equivalent_units(
                    variable.attrs.get("units"), expected_units, name=f"CO2 role {role!r} in {group}",
                )
                xr.align(variable, prepared.inv_inputs["mf"], join="exact", copy=False)
        if prepared is not None:
            xr.align(dataset[roles["flux_scale"]], prepared.inv_inputs["alpha_prior_mean"], join="exact", copy=False)


def make_co2_output(
    *, prepared: Co2PreparedInputs, trace: xr.DataTree, start_date: str,
    end_date: str, domain: str,
    bound: BoundCo2AffineFluxMap | None = None,
    output_metadata: Mapping | None = None,
) -> Co2Output:
    """Construct a scientific view without graph execution or array copying."""
    return Co2Output(prepared, trace, start_date, end_date, domain,
                     bound=bound,
                     output_metadata=dict(output_metadata or {}))


def _validate_co2_paris_options(kwargs: Mapping | None) -> None:
    """Validate supported reporting choices without reading scientific artifacts."""
    kwargs = kwargs or {}
    supported = {
        "template_version",
        "flux_frequency",
        "country_selections",
        "time_point",
        "report_mode",
        "inversion_grid",
    }
    if unknown := kwargs.keys() - supported:
        raise ValueError(f"Unsupported CO2 PARIS options: {sorted(unknown)}.")
    for name, expected in (
        ("template_version", "latest"),
        ("time_point", "midpoint"),
        ("report_mode", False),
        ("inversion_grid", False),
    ):
        if kwargs.get(name, expected) != expected:
            raise ValueError(f"CO2 PARIS outputs require {name}={expected!r}.")


@dataclass(frozen=True)
class Co2OutputOptions:
    """Ordinary CO2 concentration and conditional-flux reporting choices.

    ``none`` suppresses products and exports; ``basic`` retains scientific
    summaries, and ``paris`` adds supported latest-template products. With no
    output directory products remain in memory. ``save_trace`` exports the full
    annotated DataTree to an explicit path or the default trace filename.
    Paths resolve from the invocation directory, without loading resources.
    """

    output_format: str = "basic"
    output_path: str | Path | None = None
    output_name: str = "rhime"
    save_trace: str | Path | bool = False
    country_file: str | Path | None = None
    paris_postprocessing_kwargs: Mapping | None = None

    def __post_init__(self) -> None:
        from openghg_inversions.rhime._stage_artifacts import _validate_filename_component

        if not isinstance(self.output_format, str) or self.output_format not in {"none", "basic", "paris"}:
            raise ValueError("CO2 outputs support only 'none', 'basic', or 'paris'.")
        _validate_filename_component("output_name", self.output_name)
        for name in ("output_path", "country_file", "save_trace"):
            value = getattr(self, name)
            if value is None or (name == "save_trace" and isinstance(value, bool)):
                continue
            if not isinstance(value, str | Path):
                raise TypeError(f"CO2 {name} must be a filesystem path.")
            object.__setattr__(self, name, Path(value).expanduser().resolve())
        if self.output_format != "none" and self.save_trace is True and self.output_path is None:
            raise ValueError("CO2 output_path is required when save_trace=True.")
        if self.paris_postprocessing_kwargs is not None:
            if not isinstance(self.paris_postprocessing_kwargs, Mapping):
                raise TypeError("CO2 paris_postprocessing_kwargs must be a mapping.")
            object.__setattr__(self, "paris_postprocessing_kwargs", dict(self.paris_postprocessing_kwargs))
        if self.output_format == "paris":
            _validate_co2_paris_options(self.paris_postprocessing_kwargs)

    @classmethod
    def from_params(cls, params: Mapping) -> Co2OutputOptions:
        """Resolve only supported ordinary CO2 product and export options."""
        if unknown := params.keys() - cls.__dataclass_fields__.keys():
            raise ValueError(f"Unsupported CO2 output option(s): {sorted(unknown)!r}.")
        return cls(**params)


def validate_co2_output_spec(
    output_spec: Co2OutputOptions | RhimeOutputSpec,
    *,
    has_reconstruction: bool,
) -> None:
    """Reject unsupported CO2 requests before constructing or writing products.

    Basic products may report concentrations and states without native
    reconstruction. PARIS requires a bound affine reconstruction and supports
    latest templates, native grids, means, and midpoint flux timestamps only.
    The standard multiplicative inversion-output and legacy formats cannot
    represent coherent affine CO2 reconstruction.
    """
    if isinstance(output_spec, RhimeOutputSpec):
        if output_spec.output_format not in {"none", "basic", "paris"}:
            raise ValueError(
                "CO2 outputs support output_format='none', 'basic', or 'paris'; inv_out and legacy are unsupported."
            )
        if output_spec.save_inversion_output:
            raise ValueError(
                "CO2 save_inversion_output is unsupported; retain the prepared, posterior, and reconstruction stage artifacts."
            )
    if output_spec.country_file and not has_reconstruction:
        raise ValueError("CO2 country flux outputs require a bound affine reconstruction.")
    if output_spec.output_format == "paris":
        if not has_reconstruction:
            raise ValueError("CO2 PARIS flux outputs require a bound affine reconstruction.")
        if isinstance(output_spec, RhimeOutputSpec):
            _validate_co2_paris_options(output_spec.paris_postprocessing_kwargs)


def validate_co2_outputs(
    output_spec: Co2OutputOptions | RhimeOutputSpec,
    *,
    bound: BoundCo2AffineFluxMap | None = None,
    source_to_sector: Mapping[str, str] | None = None,
) -> None:
    """Validate output capabilities and exact native source reporting labels."""
    validate_co2_output_spec(output_spec, has_reconstruction=bound is not None)
    validate_co2_source_mapping(bound, source_to_sector)


def validate_co2_source_mapping(
    bound: BoundCo2AffineFluxMap | None, mapping: Mapping[str, str] | None,
) -> None:
    """Require a complete native-source mapping before reporting sector sums."""
    if mapping is not None:
        if not isinstance(mapping, Mapping) or not mapping or any(
            not isinstance(key, str) or not isinstance(value, str) or not value.isidentifier()
            for key, value in mapping.items()
        ):
            raise ValueError("CO2 source_to_sector must map native sources to identifier sector names.")
        if bound is None or len(bound.affine_map.native_dims) != 3:
            raise ValueError("CO2 source_to_sector requires a source-resolved affine reconstruction.")
        labels = set(map(str, bound.affine_map.native_mean[bound.affine_map.native_dims[0]].values))
        if set(mapping) != labels:
            raise ValueError(
                "CO2 source_to_sector must map every native source exactly once to an identifier sector name."
            )


def validate_co2_reporting_inputs(
    output: Co2OutputOptions | RhimeOutputSpec, prepared: Co2PreparedInputs, bound: BoundCo2AffineFluxMap | None,
) -> Countries | None:
    """Load requested country masks and align them before inference or writing.

    The returned reporting input is reused by product calculation. No scientific
    arrays are materialized here; indexed native grid labels own alignment.
    """
    if output.output_format == "none":
        return None
    if output.output_format == "paris" and len(set(prepared.averaging_period)) != 1:
        raise ValueError("CO2 PARIS concentration requires one common observation averaging period.")
    if output.country_file is None and output.output_format != "paris":
        return None
    if output.output_format == "paris" and output.country_file is None:
        raise ValueError("CO2 PARIS outputs require an explicit outputs.country_file.")
    kwargs = output.paris_postprocessing_kwargs or {}
    countries = Countries.from_file(country_file=output.country_file,
                                   country_selections=kwargs.get("country_selections"))
    if bound is None:
        raise ValueError("CO2 country flux outputs require a bound affine reconstruction.")
    xr.align(bound.affine_map.flux, countries.matrix, join="exact", copy=False)
    return countries


def validate_co2_product_request(
    output: Co2OutputOptions | RhimeOutputSpec, *, prepared: Co2PreparedInputs,
    bound: BoundCo2AffineFluxMap | None = None,
    source_to_sector: Mapping[str, str] | None = None,
) -> Countries | None:
    """Validate one product request and load its reporting inputs before execution.

    Full execution calls this before model construction; replay calls it with
    loaded scientific context. Calculation trusts these choices and reuses the
    returned country matrix. Saved prior-draw availability is checked separately
    when products consume the completed trace.
    """
    validate_co2_outputs(output, bound=bound, source_to_sector=source_to_sector)
    return validate_co2_reporting_inputs(output, prepared, bound)


def make_co2_rhime_result(
    *,
    prepared: Co2PreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler | None,
    model_build_result: RhimeModelBuildResult | None = None,
    idata: xr.DataTree,
    build_and_sample_seconds: float = 0.0,
) -> RhimeResult:
    """Join a retained cached-stage posterior with its validated preparation.

    The cached caller supplies the matched trace and preparation. The adapter borrows
    those scientific inputs without rebuilding a graph or serializing.
    An in-memory runner may additionally retain its existing model build.
    Passing ``sampler=None`` retains no live sampler; saved-output orchestration
    supplies recorded descriptive values in ``result.output_metadata["sampler"]``.
    """
    return RhimeResult(
        run_spec=run_spec,
        model_spec=run_spec.model,
        output_spec=run_spec.output,
        inv_inputs=prepared.inv_inputs,
        idata=idata,
        sampler=sampler,
        model=None if model_build_result is None else model_build_result.model,
        basis_functions=prepared.basis_functions,
        model_build_result=model_build_result,
        output_metadata={"build_and_sample_seconds": build_and_sample_seconds},
    )


def reconstruct_co2_concentrations(trace: xr.DataTree, prepared: Co2PreparedInputs) -> xr.Dataset:
    """Select full scientific components and predictions through explicit roles.

    Returns observations, affine intercept, full/active state draws, complete
    model mean, pollution, boundary, offset, residual, and represented marginal
    error. Predictive draws are included only when generated by the model's
    likelihood; this adapter never invents independent observation noise.
    Prior axes are renamed to preserve unequal prior/posterior sample counts.
    Absent optional baseline components are zero. Inputs remain borrowed.
    """
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    observations = prepared.inv_inputs["mf"]
    fixed = prepared.inv_inputs["fixed_prior_contribution"]
    result = {"observed": observations, "affine": fixed}
    component_roles = {
        "modelled": "model_mean",
        "pollution": "pollution_concentration",
        "boundary": "boundary_concentration",
        "offset": "offset_concentration",
        "flux_scale": "flux_scale",
        "active_flux_scale": "active_flux_scale",
        "boundary_scale": "boundary_scale",
        "offset_coefficient": "offset_coefficient",
        "total_error": "model_error",
    }
    for group in ("prior", "posterior"):
        if group not in trace.children:
            if group == "prior":
                continue
            raise ValueError(f"CO2 scientific outputs require {group} draws.")
        dataset = trace_group(trace, group)
        for component_name, role in component_roles.items():
            name = roles.get(role)
            if name in dataset:
                value = dataset[name]
            elif component_name in {"boundary", "offset"} and role not in roles:
                value = xr.zeros_like(dataset[roles["model_mean"]])
            elif component_name in {"modelled", "pollution", "flux_scale"} or role in roles:
                raise ValueError(f"CO2 scientific outputs require role {role!r} in {group}.")
            else:
                continue
            value, _ = xr.align(value, observations, join="exact", copy=False)
            if group == "prior":
                value = value.rename({dim: f"prior_{dim}" for dim in ("chain", "draw") if dim in value.dims})
            result[f"{component_name}_{group}"] = value
        result[f"residual_{group}"] = observations - result[f"modelled_{group}"]
        predictive_group = f"{group}_predictive"
        if predictive_group in trace.children:
            predictive = trace_group(trace, predictive_group)
            name = roles.get("concentration")
            if name in predictive:
                value, _ = xr.align(predictive[name], observations, join="exact", copy=False)
                if group == "prior":
                    value = value.rename(
                        {dim: f"prior_{dim}" for dim in ("chain", "draw") if dim in value.dims}
                    )
                result[f"predictive_{group}"] = value
    output = xr.Dataset(result)
    output.attrs = {
        "species": "co2",
        "units": str(observations.attrs["units"]),
        "rhime_variable_roles": json.dumps(roles, sort_keys=True),
        "rhime_model_metadata": trace.attrs.get("rhime_model_metadata", "{}"),
        "co2_preparation_provenance": json.dumps(dict(prepared.provenance), sort_keys=True),
        "has_offset": int("offset_concentration" in roles),
    }
    for name, value in output.data_vars.items():
        value.attrs = dict(value.attrs)
        value.attrs["units"] = (
            "1" if "flux_scale" in name or "boundary_scale" in name else observations.attrs["units"]
        )
    return output


def _basic_product(components: xr.Dataset) -> xr.Dataset:
    """Reduce each sample group independently, preserving observation labels."""
    summaries = [components[["observed", "affine"]]]
    for group in ("prior", "posterior"):
        names = [name for name in components.data_vars if str(name).endswith(f"_{group}")]
        if not names:
            continue
        selected = components[names]
        if group == "prior":
            selected = selected.rename(
                {
                    name: name.removeprefix("prior_")
                    for name in ("prior_chain", "prior_draw")
                    if name in selected.dims
                }
            )
        summaries.append(calculate_stats(selected, stats=["mean", "stdev", "quantiles"]))
    result = xr.merge(summaries)
    result.attrs = dict(components.attrs)
    return result


def _affine_products(trace, bound, countries, source_to_sector):
    """Apply reporting source combinations before statistics, using signed draws."""
    native = co2_native_flux_outputs(trace, bound)
    country = co2_country_flux_outputs(trace, bound, countries) if countries is not None else None
    if source_to_sector:
        source_dim = bound.affine_map.native_dims[0]
        for sector in sorted(set(source_to_sector.values())):
            sources = [source for source, mapped in source_to_sector.items() if mapped == sector]
            flux = bound.affine_map.flux.where(bound.affine_map.flux[source_dim].isin(sources), 0)
            flux.attrs = dict(bound.affine_map.flux.attrs)
            mapped = replace(
                bound, artifact=replace(bound.artifact, affine_map=replace(bound.affine_map, flux=flux))
            )
            sector_native = co2_native_flux_outputs(trace, mapped)
            native = native.merge(
                sector_native[[name for name in sector_native if str(name).startswith("flux_total_")]].rename(
                    {
                        name: str(name).replace("flux_total_", f"flux_{sector}_")
                        for name in sector_native
                        if str(name).startswith("flux_total_")
                    }
                )
            )
            if countries is not None:
                sector_country = co2_country_flux_outputs(trace, mapped, countries)
                country = country.merge(
                    sector_country[
                        [name for name in sector_country if str(name).startswith("country_total_")]
                    ].rename(
                        {
                            name: str(name).replace("country_total_", f"country_{sector}_")
                            for name in sector_country
                            if str(name).startswith("country_total_")
                        }
                    )
                )
        native.attrs["source_to_sector"] = json.dumps(dict(source_to_sector), sort_keys=True)
        if country is not None:
            country.attrs["source_to_sector"] = native.attrs["source_to_sector"]
    return native, country


def _paris_products(components, native, country, countries, result, source_to_sector, *, output):
    """Adapt conditional signed summaries directly to latest PARIS templates."""
    from openghg_inversions.postprocessing.linked_paris_outputs import _concentration_product
    from openghg_inversions.postprocessing.make_paris_outputs import (
        _assign_flux_time_bounds,
        _cast_data_vars_to_template_dtypes,
        _convert_flux_time_and_bounds_to_epoch_days,
        _expand_sector_template_mapping,
        _prepare_latest_paris_netcdf_encoding,
        add_variable_attrs,
        get_data_var_attrs,
        infer_flux_frequency,
        make_global_attrs,
        paris_template_files,
    )

    # The complete pollution role includes the affine intercept already. The
    # template helper expects separate affine and sampled pollution components.
    concentration = components.rename(nmeasure="observation")
    for group in ("prior", "posterior"):
        concentration[f"emissions_{group}"] = concentration[f"pollution_{group}"] - concentration.affine
    concentration = concentration.drop_vars(
        [
            name
            for name in concentration
            if "flux_scale" in str(name)
            or "boundary_scale" in str(name)
            or "offset_coefficient" in str(name)
            or "predictive" in str(name)
            or "pollution" in str(name)
        ]
    )
    concentration["total_error"] = concentration.total_error_posterior
    concentration = concentration.drop_vars(["total_error_prior", "total_error_posterior", "residual_prior"])
    domain = result.domain if isinstance(result, Co2Output) else result.model_spec.domain
    start_date = result.start_date if isinstance(result, Co2Output) else result.run_spec.start_date
    end_date = result.end_date if isinstance(result, Co2Output) else result.run_spec.end_date
    conc = _concentration_product(concentration.compute(), domain=domain, obs_avg_period=prepared_period(result))
    kwargs = output.paris_postprocessing_kwargs or {}
    sectors = sorted(set(source_to_sector.values())) if source_to_sector else []
    # Affine flux artifacts may retain the producer's ordinary time axis.
    # Normalize it before adding a static interval or inferring its frequency.
    if "time" in native.dims:
        native = native.rename(time="flux_time")
    if "time" in country.dims:
        country = country.rename(time="flux_time")
    arrays = {}
    for prefix, dataset in (("flux", native), ("country", country)):
        for name, value in dataset.data_vars.items():
            text = str(name)
            if prefix == "flux" and not (
                text.startswith("flux_total_")
                or any(text.startswith(f"flux_{sector}_") for sector in sectors)
            ):
                continue
            if prefix == "country" and not (
                text.startswith("country_total_")
                or any(text.startswith(f"country_{sector}_") for sector in sectors)
            ):
                continue
            label = text.replace("country_", "flux_", 1) if prefix == "country" else text
            if label.endswith("_mean"):
                label = label.removesuffix("_mean")
            elif label.endswith("_stdev"):
                label = "stdev_" + label.removesuffix("_stdev")
            else:
                label = "percentile_" + label.removesuffix("_quantile")
            if prefix == "country":
                label += "_country"
                value = value / 1000  # Existing conditional helper reports grams; PARIS uses kg.
            arrays[label] = value
    arrays["country_fraction"] = countries.matrix
    arrays["cell_area"] = countries.area_grid
    flux = xr.Dataset(arrays)
    if "flux_time" not in flux.dims:
        flux = flux.expand_dims(flux_time=[np.datetime64(start_date)])
    flux = flux.rename(
        {
            name: replacement
            for name, replacement in {
                "lat": "latitude",
                "lon": "longitude",
                "flux_time": "time",
                "quantile": "percentile",
            }.items()
            if name in flux.dims
        }
    )
    frequency = kwargs.get("flux_frequency")
    if frequency is None:
        frequency = (
            infer_flux_frequency(native.flux_total_prior_mean) if "flux_time" in native.dims else "yearly"
        )
    flux = _assign_flux_time_bounds(flux, frequency, pd.Timestamp(start_date), pd.Timestamp(end_date))
    flux = _convert_flux_time_and_bounds_to_epoch_days(flux)
    template = paris_template_files("latest")
    attrs = _expand_sector_template_mapping(get_data_var_attrs(template.flux, "co2"), sectors)
    flux = add_variable_attrs(flux, attrs).compute()
    flux.attrs = make_global_attrs("flux", species="co2", domain=domain)
    flux.attrs.update(native.attrs, paris_flux_template_version=template.flux_version)
    conc.attrs.update({key: value for key, value in components.attrs.items() if key != "has_offset"})
    flux = _prepare_latest_paris_netcdf_encoding(
        _cast_data_vars_to_template_dtypes(flux, template.flux, sector_names=sectors)
    )
    return conc, flux


def prepared_period(result: Co2Output | RhimeResult) -> str:
    """Return the observation interval used by the PARIS timestamp contract."""
    averaging_period = result.prepared.averaging_period if isinstance(result, Co2Output) else result.run_spec.averaging_period
    periods = set(averaging_period)
    if len(periods) != 1:
        raise ValueError("CO2 PARIS concentration requires one common observation averaging period.")
    return averaging_period[0] or "0h"


def calculate_co2_products(
    *,
    result: Co2Output | RhimeResult,
    prepared: Co2PreparedInputs,
    output: Co2OutputOptions | RhimeOutputSpec,
    bound: BoundCo2AffineFluxMap | None = None,
    source_to_sector: Mapping[str, str] | None = None,
    countries: Countries | None = None,
) -> None:
    """Calculate and attach CO2 components and requested conditional products.

    Call :func:`validate_co2_product_request` before this operation and pass its
    reporting inputs. Calculation trusts that preflight; it checks actual saved
    prior draws when a product requires them. All products are constructed before
    any destination is opened. Basic
    summaries preserve independent prior and posterior counts. Native/country
    products use the supplied exact affine map; PARIS reports their conditional
    scope. Files use the ordinary RHIME naming convention. Full component
    draws remain available in ``result.outputs['concentration_components']``.
    """
    if output.output_format == "none":
        if isinstance(result, Co2Output):
            result.outputs.clear()
        return
    if output.output_format == "paris" and "prior" not in result.idata.children:
        raise ValueError("CO2 PARIS outputs require prior draws; enable sample_prior_predictive.")
    components = reconstruct_co2_concentrations(result.idata, prepared)
    products = {"basic": _basic_product(components)}
    if bound is not None:
        native, country = _affine_products(result.idata, bound, countries, source_to_sector)
        products["native_flux"] = native
        if country is not None:
            products["country_flux"] = country
        result.output_metadata["uncertainty_scope"] = bound.affine_map.uncertainty_scope
        if isinstance(result, RhimeResult):
            result.outputs["affine_reconstruction"] = bound
        if output.output_format == "paris":
            conc, flux = _paris_products(components, native, country, countries, result, source_to_sector, output=output)
            products.update(paris_concentration=conc, paris_flux=flux)
    if isinstance(result, Co2Output):
        result.outputs.clear()
    result.outputs.update(products, concentration_components=components)


def save_co2_products(result: Co2Output | RhimeResult, *, output: Co2OutputOptions | RhimeOutputSpec) -> None:
    """Export calculated products and the optional trace through existing naming policy."""
    if output.output_format == "none":
        return
    domain = result.domain if isinstance(result, Co2Output) else result.model_spec.domain
    start_date = result.start_date if isinstance(result, Co2Output) else result.run_spec.start_date
    if output.output_path is not None:
        Path(output.output_path).mkdir(parents=True, exist_ok=True)
        for name, product in result.outputs.items():
            if name in {"affine_reconstruction", "concentration_components"}:
                continue
            path = _define_output_filename(
                output.output_path, "co2", domain, f"{output.output_name}_{name}", start_date,
                filename_convention=output.output_filename_convention if isinstance(output, RhimeOutputSpec) else "rhime",
            )
            if name == "basic":
                save_datatree(xr.DataTree(reset_serialisation_multiindexes(product)), path)
            else:
                write_netcdf_preserving_bounds_attrs(product, path)
            result.output_metadata[f"{name}_path"] = str(path)
    if isinstance(result, Co2Output):
        from openghg_inversions.rhime.outputs import _resolve_output_path
        from openghg_inversions.serialization import save_trace

        trace_path = _resolve_output_path(output.save_trace, output.output_path,
                                          f"{output.output_name}{start_date}_trace.nc")
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            save_trace(result.idata, trace_path)
            result.output_metadata["trace_path"] = str(trace_path)
    else:
        result.output_metadata.update(_save_requested_trace(result))


def make_co2_rhime_outputs(
    *, result: RhimeResult, prepared: Co2PreparedInputs,
    bound: BoundCo2AffineFluxMap | None = None, source_to_sector: Mapping[str, str] | None = None,
) -> None:
    """Calculate and export CO2 products for the retained cached-stage adapter."""
    countries = validate_co2_product_request(
        result.output_spec, prepared=prepared, bound=bound, source_to_sector=source_to_sector,
    )
    calculate_co2_products(result=result, prepared=prepared, output=result.output_spec,
                           bound=bound, source_to_sector=source_to_sector, countries=countries)
    save_co2_products(result, output=result.output_spec)


__all__ = [
    "Co2Output",
    "Co2OutputOptions",
    "make_co2_output",
    "calculate_co2_products",
    "save_co2_products",
    "validate_co2_trace",
    "validate_co2_product_request",
    "validate_co2_output_spec",
    "validate_co2_outputs",
    "validate_co2_source_mapping",
    "make_co2_rhime_result",
    "make_co2_rhime_outputs",
    "reconstruct_co2_concentrations",
]
