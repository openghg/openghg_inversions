"""Configured ordinary CO2 execution and independently resolved phase choices."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact, load as load_affine
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime._stage_artifacts import _validate_filename_component

from .co2_affine_output import _bind_affine_flux_map, prepared_inputs_content_id
from .co2_outputs import (
    Co2Output, Co2OutputOptions, calculate_co2_products, save_co2_products, make_co2_output,
    validate_co2_product_request,
)
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs, requested_co2_facts
from .inference import sample_co2_inference, validate_co2_inference
from .configuration import (
    load_co2_family_config, validate_co2_config, resolve_co2_preparation,
    resolve_co2_model, resolve_co2_sampler,
)


def load_co2_params(
    *, config_file: str | Path | None = None, params_file: str | Path | None = None,
    params: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate one TOML/JSON/Python request without touching arrays.

    Overrides update keys within their owned sections before normalization. Each concrete
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
    for name, value in (overrides or {}).items():
        if name in {"prepared_inputs", "model", "likelihood", "sampling", "outputs"} and isinstance(value, Mapping):
            previous = options.get(name, {})
            if not isinstance(previous, Mapping):
                raise ValueError(f"CO2 {name} must be a mapping.")
            options[name] = {**previous, **value}
        else:
            options[name] = value
    return validate_co2_config(options)


def resolve_co2_output(params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve supported CO2 products and their current reporting choices only."""
    choices = dict(params.get("outputs", {}))
    source_to_sector = choices.pop("source_to_sector", None)
    return {"output": Co2OutputOptions.from_params(choices), "source_to_sector": source_to_sector}


def run_co2(
    *, config_file: str | Path | None = None, params_file: str | Path | None = None,
    params: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None, start_date: str | None = None, end_date: str | None = None,
    prepared: Co2PreparedInputs | None = None, canonical_inputs: RhimePreparedInputs | None = None,
    reduction: CoherentGaussianReduction | None = None, affine_artifact: AffineFluxMapArtifact | None = None,
    aggregation_error_rank: int | None = 512,
) -> Co2Output:
    """Prepare or load ordinary CO2 inputs, sample and create requested products.

    This configured product entry point consumes ``model``, ``likelihood``,
    ``sampling`` and ``outputs`` without intermediate manifests. Trace-only
    callers use the model-selected inference steps directly. Supplied canonical
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
    options = load_co2_params(config_file=config_file, params_file=params_file, params=params, overrides=overrides)
    if output_path is not None:
        options["outputs"] = {**options.get("outputs", {}), "output_path": Path(output_path).expanduser().resolve()}
    if options.get("variant", "ordinary") != "ordinary":
        raise ValueError("run_co2 requires variant='ordinary'; cached execution has its own runner.")
    preparation = resolve_co2_preparation(options)
    model = resolve_co2_model(options)
    sampler = resolve_co2_sampler(options)
    validate_co2_inference(model, sampler)
    output_choices = resolve_co2_output(options)
    output = output_choices["output"]
    affine_path = preparation["affine_path"]
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
        prepared, start_date=start_date, end_date=end_date,
        recorded_facts={name: preparation[name] for name in ("start_date", "end_date", "domain")},
    )
    for name in ("domain", "start_date"):
        _validate_filename_component(name, facts[name])
    bound = None
    if affine_artifact is not None:
        if affine_path is not None:
            raise ValueError("Pass affine_artifact or prepared_inputs.reconstruction_path, not both.")
        bound = _bind_affine_flux_map(affine_artifact, prepared, prepared_inputs_id=affine_artifact.prepared_inputs_id)
    elif affine_path is not None:
        if canonical_inputs is not None or supplied_prepared or source is None:
            raise ValueError("Supplied or new coherent preparation requires its matching affine_artifact.")
        bound = _bind_affine_flux_map(
            load_affine(affine_path), prepared, prepared_inputs_id=prepared_inputs_content_id(source),
        )
    countries = validate_co2_product_request(
        output, prepared=prepared, bound=bound, source_to_sector=output_choices["source_to_sector"],
    )
    if bound is not None and output.output_format != "none" and not sampler.sample_prior_predictive:
        raise ValueError("CO2 affine products require prior draws; enable sample_prior_predictive.")
    active_sampler = resolve_sampler(sampler)
    trace = sample_co2_inference(prepared, model, active_sampler)
    result = make_co2_output(prepared=prepared, trace=trace, bound=bound,
                            output_metadata={"sampler": active_sampler.describe(),
                            "provenance": installed_ogi_provenance()}, **facts)
    calculate_co2_products(result=result, prepared=prepared, bound=bound,
                           source_to_sector=output_choices["source_to_sector"], countries=countries, output=output)
    save_co2_products(result, output=output)
    return result


__all__ = ["load_co2_params", "resolve_co2_output", "run_co2"]
