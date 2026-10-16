"""Configured ordinary CO2 execution and independently resolved phase choices."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime._stage_artifacts import _validate_filename_component
from openghg_inversions.rhime.builders import callable_metadata
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.sampling import sample_rhime_model

from .co2_affine_output import _bind_affine_flux_map, load_and_bind_affine_flux_map
from .co2_outputs import (
    Co2Output, Co2OutputOptions, calculate_co2_products, save_co2_products, make_co2_output,
    validate_co2_output_spec, validate_co2_reporting_inputs,
)
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .co2_runner import (
    annotate_co2_trace, materialize_co2_inputs, build_co2_from_materialized, describe_co2_likelihood_kwargs,
)
from .configuration import load_co2_family_config, resolve_co2_model, resolve_co2_sampler


def load_co2_params(
    *, config_file: str | Path | None = None, params_file: str | Path | None = None,
    params: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate one TOML/JSON/Python request without touching arrays.

    Overrides replace top-level sections before normalization. Each concrete
    owner resolves its own paths from the invocation working directory. Saved
    references are resolved separately beside their declaring manifest.
    """
    if sum(value is not None for value in (config_file, params_file, params)) != 1:
        raise ValueError("Pass exactly one of config_file, params_file or params.")
    if config_file is not None:
        options = dict(load_co2_family_config(config_file))
    elif params_file is not None:
        options = json.loads(Path(params_file).read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            raise ValueError("CO2 params file must contain one JSON object.")
    else:
        options = dict(params or {})
    options.update(overrides or {})
    if type(options.get("format_version", 1)) is not int or options.get("format_version", 1) != 1:
        raise ValueError("config.format_version must be the integer 1.")
    if options.get("recipe", "co2") != "co2":
        raise ValueError("CO2 execution requires recipe='co2'.")
    if options.get("variant", "ordinary") not in {"ordinary", "cached_fixed_ou"}:
        raise ValueError("CO2 variant must be 'ordinary' or 'cached_fixed_ou'.")
    allowed = {"format_version", "recipe", "variant", "prepared_inputs", "model", "likelihood", "sampling", "outputs"}
    if unknown := options.keys() - allowed:
        raise ValueError(f"Unknown CO2 configuration option(s): {sorted(unknown)!r}.")
    return options


def resolve_co2_preparation(params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve explicit prepared/affine locations and original requested facts.

    This owner reads no inference or reporting choices. Imported artifact paths
    resolve from the invocation working directory; numerical loaders own facts
    that can only be checked against their arrays.
    """
    prepared = dict(params.get("prepared_inputs", {}))
    if unknown := prepared.keys() - {"path", "start_date", "end_date", "domain"}:
        raise ValueError(f"Unknown prepared_inputs option(s): {sorted(unknown)!r}.")
    source = prepared.get("path")
    companion = params.get("outputs", {}).get("reconstruction_path")
    return {
        "prepared_path": Path(source).expanduser().resolve() if source is not None else None,
        "affine_path": Path(companion).expanduser().resolve() if companion is not None else None,
        "start_date": prepared.get("start_date"), "end_date": prepared.get("end_date"),
        "domain": prepared.get("domain"),
    }


def resolve_co2_output(params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve supported CO2 products and their current reporting choices only."""
    choices = dict(params.get("outputs", {}))
    choices.pop("reconstruction_path", None)
    source_to_sector = choices.pop("source_to_sector", None)
    return {"output": Co2OutputOptions.from_params(choices), "source_to_sector": source_to_sector}


def run_co2_from_config(
    *, config_file: str | Path | None = None, params_file: str | Path | None = None,
    params: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None, start_date: str | None = None, end_date: str | None = None,
    prepared: Co2PreparedInputs | None = None, canonical_inputs: RhimePreparedInputs | None = None,
    reduction: CoherentGaussianReduction | None = None, affine_artifact: AffineFluxMapArtifact | None = None,
    aggregation_error_rank: int | None = 512,
) -> Co2Output:
    """Prepare or load ordinary CO2 inputs, sample and create requested products.

    This configured product entry point consumes ``model``, ``likelihood``,
    ``sampling`` and ``outputs`` without intermediate manifests. The direct
    :func:`run_rhime_co2` retains its numerical trace return. Supplied canonical
    inputs and their coherent reduction are prepared together; an existing
    preparation is reused unchanged. Imported runs must supply requested dates
    and domain explicitly or retain them in preparation provenance.

    Returns:
        Scientific result containing the trace, concentration components and
        requested conditional affine products, with configured files written.

    Raises:
        ValueError: If inputs conflict, required run facts are unavailable, or
            a requested scientific/output combination is unsupported.
    """
    from .co2_preparation import requested_co2_facts

    options = load_co2_params(config_file=config_file, params_file=params_file, params=params, overrides=overrides)
    if output_path is not None:
        options["outputs"] = {**options.get("outputs", {}), "output_path": Path(output_path).expanduser().resolve()}
    if options.get("variant", "ordinary") != "ordinary":
        raise ValueError("run_co2_from_config requires variant='ordinary'; cached execution has its own runner.")
    preparation = resolve_co2_preparation(options)
    model = resolve_co2_model(options)
    sampler = resolve_co2_sampler(options)
    output_choices = resolve_co2_output(options)
    output = output_choices["output"]
    affine_path = preparation["affine_path"]
    validate_co2_output_spec(output, has_reconstruction=affine_path is not None or affine_artifact is not None)
    if output.output_format == "paris" and not output.country_file:
        raise ValueError("CO2 PARIS outputs require an explicit outputs.country_file.")
    if (canonical_inputs is None) != (reduction is None):
        raise ValueError("Pass canonical_inputs and reduction together.")
    if prepared is not None and canonical_inputs is not None:
        raise ValueError("Pass prepared or canonical_inputs/reduction, not both.")
    source = preparation["prepared_path"]
    supplied_prepared = prepared is not None
    if canonical_inputs is not None and reduction is not None:
        prepared = prepare_co2_inputs(canonical_inputs, reduction, aggregation_error_rank=aggregation_error_rank)
    elif prepared is None:
        if source is None:
            raise ValueError("Pass prepared inputs or configure prepared_inputs.path.")
        prepared = Co2PreparedInputs.load(source)
    facts = requested_co2_facts(
        prepared, start_date=start_date or preparation["start_date"],
        end_date=end_date or preparation["end_date"], domain=preparation["domain"],
    )
    for name in ("domain", "start_date"):
        _validate_filename_component(name, facts[name])
    bound = None
    if affine_artifact is not None:
        if affine_path is not None:
            raise ValueError("Pass affine_artifact or outputs.reconstruction_path, not both.")
        bound = _bind_affine_flux_map(affine_artifact, prepared, prepared_inputs_id=affine_artifact.prepared_inputs_id)
    elif affine_path is not None:
        if canonical_inputs is not None or supplied_prepared or source is None:
            raise ValueError("Supplied or new coherent preparation requires its matching affine_artifact.")
        bound = load_and_bind_affine_flux_map(affine_path, source)
    from .co2_outputs import validate_co2_outputs

    validate_co2_outputs(output, bound=bound, source_to_sector=output_choices["source_to_sector"])
    countries = validate_co2_reporting_inputs(output, prepared, bound)
    if bound is not None and output.output_format != "none" and not sampler.sample_prior_predictive:
        raise ValueError("CO2 affine products require prior draws; enable sample_prior_predictive.")
    active_sampler = resolve_sampler(sampler)
    model_inputs, materialized_options = materialize_co2_inputs(prepared_inputs=prepared, **model)
    built = build_co2_from_materialized(model_inputs, prepared_inputs=prepared,
                                       **{**model, **materialized_options})
    trace = sample_rhime_model(built, active_sampler)
    annotate_co2_trace(trace, built, concentration_units=prepared.inv_inputs["mf"].attrs.get("units"))
    if "likelihood_builder" in model:
        annotate_likelihood_trace(
            trace, builder_identity=callable_metadata(model["likelihood_builder"]),
            likelihood_kwargs=describe_co2_likelihood_kwargs(model.get("likelihood_kwargs")),
        )
    result = make_co2_output(prepared=prepared, trace=trace, bound=bound,
                            output_metadata={"sampler": active_sampler.describe(),
                            "provenance": installed_ogi_provenance()}, **facts)
    calculate_co2_products(result=result, prepared=prepared, bound=bound,
                           source_to_sector=output_choices["source_to_sector"], countries=countries, output=output)
    save_co2_products(result, output=output)
    return result


__all__ = ["load_co2_params", "resolve_co2_preparation", "resolve_co2_output", "run_co2_from_config"]
