"""Configured ordinary CO2 execution and independently resolved phase choices."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Literal

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime._stage_artifacts import _validate_filename_component
from openghg_inversions.rhime.builders import callable_metadata
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.sampling import sample_rhime_model
from openghg_inversions.rhime.specs import RhimeOutputSpec

from .co2_affine_output import _bind_affine_flux_map, load_and_bind_affine_flux_map
from .co2_outputs import (
    Co2Output, calculate_co2_products, save_co2_products, make_co2_output,
    validate_co2_output_spec, validate_co2_reporting_inputs,
)
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .co2_runner import (
    annotate_co2_trace, materialize_co2_inputs, build_co2_from_materialized, describe_co2_likelihood_kwargs,
)
from .configuration import load_co2_family_config, resolve_co2_inference


def load_co2_params(
    *, config_file: str | Path | None = None, params_file: str | Path | None = None,
    params: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
    phase: Literal["prepare", "sample", "prior", "postprocess"] | None = None,
) -> dict[str, Any]:
    """Load one TOML/JSON/Python request, resolving paths from the invocation.

    Overrides replace top-level sections before normalization. Configuration
    resources use the same working-directory base in full and saved execution;
    checkpoint references are resolved separately beside their manifest.
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
    for section, keys in (
        ("prepared_inputs", ("path",)), ("likelihood", ("eigenbasis_path",)),
        ("outputs", ("reconstruction_path", "country_file", "output_path", "save_trace")),
    ):
        if section not in options:
            continue
        if phase in {"sample", "prior"} and section != "likelihood":
            continue
        if phase == "postprocess" and section != "outputs":
            continue
        if phase == "prepare" and section == "likelihood":
            continue
        values = dict(options[section])
        for key in keys:
            if phase == "postprocess" and key == "reconstruction_path":
                continue
            if phase == "prepare" and section == "outputs" and key != "reconstruction_path":
                continue
            if values.get(key) is not None and not isinstance(values[key], bool):
                values[key] = Path(values[key]).expanduser().resolve()
        options[section] = values
    return options


def resolve_co2_phase_options(
    params: Mapping[str, Any], *, phase: Literal["prepare", "sample", "prior", "postprocess"],
) -> dict[str, Any]:
    """Resolve only the preparation, inference or output choices a phase uses.

    Phase-only requests need no unrelated sections. Defining coherent native
    priors and operators remain numerical preparation inputs, rather than
    inference overrides. Linked and cached recipes retain their own routes.
    """
    if type(params.get("format_version", 1)) is not int or params.get("format_version", 1) != 1:
        raise ValueError("config.format_version must be the integer 1.")
    if params.get("recipe", "co2") != "co2" or params.get("variant", "ordinary") != "ordinary":
        raise ValueError("Ordinary CO2 phases require recipe='co2', variant='ordinary'.")
    allowed = {"format_version", "recipe", "variant", "prepared_inputs", "model", "likelihood", "sampling", "outputs"}
    if unknown := params.keys() - allowed:
        raise ValueError(f"Unknown CO2 configuration option(s): {sorted(unknown)!r}.")
    if phase == "prepare":
        prepared = dict(params.get("prepared_inputs", {}))
        if unknown := prepared.keys() - {"path", "start_date", "end_date", "domain"}:
            raise ValueError(f"Unknown prepared_inputs option(s): {sorted(unknown)!r}.")
        return {
            "prepared_path": prepared.get("path"), "start_date": prepared.get("start_date"),
            "end_date": prepared.get("end_date"), "domain": prepared.get("domain"),
            "affine_path": params.get("outputs", {}).get("reconstruction_path"),
        }
    if phase in {"sample", "prior"}:
        science = {"format_version": 1, "recipe": "co2", "variant": "ordinary", **params}
        return {"model": resolve_co2_inference(science, sampling=phase == "sample")}
    choices = dict(params.get("outputs", {}))
    # A saved sample owns its affine association; a new reporting request cannot replace it.
    choices.pop("reconstruction_path", None)
    source_to_sector = choices.pop("source_to_sector", None)
    if source_to_sector is not None and (
        not isinstance(source_to_sector, Mapping) or not source_to_sector
        or any(not isinstance(key, str) or not isinstance(value, str) or not value.isidentifier()
               for key, value in source_to_sector.items())
    ):
        raise ValueError("outputs.source_to_sector must map sources to identifier sector names.")
    choices.setdefault("output_format", "basic")
    choices.setdefault("save_inversion_output", False)
    output = RhimeOutputSpec.from_params(choices, multisector=False)
    _validate_filename_component("output_name", output.output_name)
    return {"output": output, "source_to_sector": source_to_sector}


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
    preparation = resolve_co2_phase_options(options, phase="prepare")
    model = resolve_co2_phase_options(options, phase="sample")["model"]
    output_choices = resolve_co2_phase_options(options, phase="postprocess")
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
    if canonical_inputs is not None:
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
    if output.output_format == "paris":
        if not model.sampler.sample_prior_predictive:
            raise ValueError("CO2 PARIS outputs require prior draws; enable sample_prior_predictive.")
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
    active_sampler = resolve_sampler(model.sampler)
    model_inputs, materialized_options = materialize_co2_inputs(prepared_inputs=prepared, **model.runner_kwargs)
    built = build_co2_from_materialized(model_inputs, prepared_inputs=prepared,
                                       **{**model.runner_kwargs, **materialized_options})
    trace = sample_rhime_model(built, active_sampler)
    annotate_co2_trace(trace, built, concentration_units=prepared.inv_inputs["mf"].attrs.get("units"))
    if "likelihood_builder" in model.runner_kwargs:
        annotate_likelihood_trace(
            trace, builder_identity=callable_metadata(model.runner_kwargs["likelihood_builder"]),
            likelihood_kwargs=describe_co2_likelihood_kwargs(model.runner_kwargs.get("likelihood_kwargs")),
        )
    result = make_co2_output(prepared=prepared, output=output, trace=trace, bound=bound,
                            source_to_sector=output_choices["source_to_sector"],
                            sampler=active_sampler, output_metadata={"sampler": active_sampler.describe(),
                            "provenance": installed_ogi_provenance()}, **facts)
    calculate_co2_products(result=result, prepared=prepared, bound=bound,
                           source_to_sector=output_choices["source_to_sector"], countries=countries)
    save_co2_products(result)
    return result


__all__ = ["load_co2_params", "resolve_co2_phase_options", "run_co2_from_config"]
