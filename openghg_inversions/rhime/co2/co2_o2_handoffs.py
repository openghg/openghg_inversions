"""Linked CO2/O2 preparation, sampling and scientific-output checkpoints.

Numerical codecs own the joint labelled arrays. These records retain original
requested facts and references to the matched preparation and posterior.
Full runners call the same scientific operations without creating checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pymc as pm
import xarray as xr
from dask.array import Array as DaskArray

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.inference.sampling import RhimeSampler, describe_recorded_sampler, resolve_sampler
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime._stage_artifacts import (
    _output_path, check_checkpoint_targets, finish_checkpoint, read_manifest,
    reference, start_checkpoint, write_json,
)
from openghg_inversions.rhime._stage_checks import (
    PREPARATION_CHECK_NAME, _check_draws, _check_result, _check_stage,
)
from openghg_inversions.rhime.sampling import sample_rhime_model
from openghg_inversions.serialization import load_trace, save_trace

from .co2_o2_configuration import (
    Co2O2ModelConfig, load_co2_o2_params, resolve_co2_o2_sampler, validate_co2_o2_units,
)
from .co2_o2_outputs import (
    Co2O2Output, Co2O2OutputOptions, calculate_co2_o2_products, make_co2_o2_output,
    save_co2_o2_products, validate_co2_o2_trace,
)
from .co2_o2_preparation import Co2O2PreparedInputs
from .co2_affine_output import prepared_inputs_content_id
from .co2_o2_runner import annotate_co2_o2_trace, build_rhime_co2_o2
from .co2_preparation import requested_co2_facts


@dataclass(frozen=True)
class Co2O2Preparation:
    """Joint prepared inputs and their original requested interval/domain.

    ``payload`` locates the existing numerical artifact. Pass ``generation`` and
    ``inputs_id`` back to :func:`save_samples` so replacing preparation invalidates
    samples from its previous generation or numerical content.
    """

    prepared: Co2O2PreparedInputs
    generation: str
    start_date: str
    end_date: str
    domain: str
    payload: Path
    inputs_id: str


def _preparation_record(manifest: str | Path) -> tuple[Path, dict[str, Any]]:
    path, record = read_manifest(manifest, "openghg_inversions.co2_o2.prepare", (
        "prepared_inputs", "generation", "start_date", "end_date", "domain", "recipe", "prepared_inputs_id",
    ), schema_version=2)
    if not isinstance(record["generation"], str) or not record["generation"]:
        raise ValueError(f"Linked preparation {path} requires a nonempty generation.")
    return path, record


def _finish_preparation(destination: Path, *, payload: Path, facts: Mapping[str, str], recipe: str) -> Path:
    return finish_checkpoint(destination, "prepare", {
        "schema": "openghg_inversions.co2_o2.prepare", "schema_version": 2,
        "producer": "openghg_inversions", "stage": "prepare", "generation": uuid4().hex,
        "prepared_inputs": os.path.relpath(payload, destination), "recipe": recipe, **facts,
        "prepared_inputs_id": prepared_inputs_content_id(payload),
        "provenance": installed_ogi_provenance(),
    })


def _check_saved_payload(path: Path, expected: str, *, name: str) -> str:
    """Bind an explicit file-load boundary to the bytes recorded at serialization."""
    actual = prepared_inputs_content_id(path)
    if not isinstance(expected, str) or not expected or expected != actual:
        raise ValueError(f"{name} content does not match its saved artifact identity: {path}")
    return actual


def save_preparation(
    prepared: Co2O2PreparedInputs, *, output_dir: str | Path,
    start_date: str | None = None, end_date: str | None = None, domain: str | None = None,
) -> Path:
    """Save joint inputs using their numerical codec, then publish a manifest.

    Requested dates/domain must be explicit or present in preparation provenance.
    Saving materializes the borrowed arrays at the existing serialization boundary.
    A successful replacement creates a new preparation generation.
    """
    facts = requested_co2_facts(prepared, start_date=start_date, end_date=end_date, domain=domain)
    lazy_units = isinstance(prepared.observations.observation_units.data, DaskArray)
    if not lazy_units:
        validate_co2_o2_units(prepared)
    destination = start_checkpoint(output_dir, "prepare", payloads=("prepared-inputs.nc",))
    payload = destination / "prepared-inputs.nc"
    prepared.save(payload)
    if lazy_units:
        # Validate written units without executing their shared input graph again.
        with xr.open_datatree(payload) as stored:
            units = stored["joint"]["observation_units"].values
        observations = prepared.observations.assign_coords(observation_units=("observation", units))
        validate_co2_o2_units(replace(prepared, observations=observations))
    return _finish_preparation(destination, payload=payload, facts=facts,
                               recipe="co2_o2_shared_baseline" if prepared.shared_baseline else "co2_o2")


def prepare(
    *, prepared_path: str | Path, output_dir: str | Path,
    start_date: str | None = None, end_date: str | None = None, domain: str | None = None,
) -> Path:
    """Reference an existing NetCDF/Zarr linked preparation without copying it.

    The caller keeps the referenced numerical artifact for later phases.
    Missing original request facts require explicit values; retained observation
    times cannot recover an interval after filtering. New in-memory preparation
    uses :func:`openghg_inversions.rhime.co2.prepare_co2_o2_inputs` followed by
    :func:`save_preparation`.
    """
    payload = Path(prepared_path).resolve()
    prepared = Co2O2PreparedInputs.load(payload)
    validate_co2_o2_units(prepared)
    facts = requested_co2_facts(prepared, start_date=start_date, end_date=end_date, domain=domain)
    destination = start_checkpoint(output_dir, "prepare", payloads=(), referenced=(payload,))
    return _finish_preparation(destination, payload=payload, facts=facts,
                               recipe="co2_o2_shared_baseline" if prepared.shared_baseline else "co2_o2")


def load_preparation(
    manifest: str | Path, *, expected_generation: str | None = None, expected_inputs_id: str | None = None,
) -> Co2O2Preparation:
    """Load a completed joint preparation and validate its original request facts.

    References resolve beside the manifest. Numerical loading validates labels,
    covariance, errors and ratio provenance. ``expected_generation`` rejects
    a preparation replaced after sampling. Current records pin exact payload
    bytes; ``expected_inputs_id`` binds a sample to those bytes even if someone
    replaces its preparation record. Only current records with required digests are accepted.
    """
    path, record = _preparation_record(manifest)
    if expected_generation is not None and record["generation"] != expected_generation:
        raise ValueError(f"Linked preparation {path} was replaced; samples require generation {expected_generation!r}.")
    payload = reference(path, record["prepared_inputs"])
    actual_id = _check_saved_payload(payload, record["prepared_inputs_id"], name="Linked preparation")
    if expected_inputs_id is not None and expected_inputs_id != actual_id:
        raise ValueError(f"Sampled linked preparation content does not match its saved artifact identity: {payload}")
    prepared = Co2O2PreparedInputs.load(payload)
    validate_co2_o2_units(prepared)
    recipe = "co2_o2_shared_baseline" if prepared.shared_baseline else "co2_o2"
    if record["recipe"] != recipe:
        raise ValueError("Linked preparation recipe does not match its numerical payload.")
    facts = requested_co2_facts(prepared, **{name: record[name] for name in ("start_date", "end_date", "domain")})
    return Co2O2Preparation(prepared=prepared, generation=record["generation"], payload=payload,
                             inputs_id=actual_id, **facts)


def save_samples(
    preparation_manifest: str | Path, trace: xr.DataTree, *, actual_inference: Mapping[str, Any],
    preparation_generation: str, output_dir: str | Path,
    preparation_inputs_id: str,
) -> Path:
    """Write the annotated joint trace and record its preparation generation.

    The trace owns the sampled scientific roles and prior descriptions.
    Inference provenance is descriptive; current configuration is not a replay
    gate. Pass ``preparation_inputs_id`` from the loaded preparation to reject
    file replacement during inference, using the current preparation identity. Completion
    is published only after posterior serialization succeeds.
    """
    preparation_path, preparation = _preparation_record(preparation_manifest)
    if preparation["generation"] != preparation_generation:
        raise ValueError(f"Linked preparation {preparation_path} was replaced; samples require generation {preparation_generation!r}.")
    validate_co2_o2_trace(trace)
    shared_trace = trace.attrs.get("rhime_recipe") == "co2_o2_shared_baseline"
    if shared_trace != (preparation["recipe"] == "co2_o2_shared_baseline"):
        raise ValueError("Linked sample recipe does not match its preparation recipe.")
    payload = reference(preparation_path, preparation["prepared_inputs"])
    prepared_id = _check_saved_payload(payload, preparation["prepared_inputs_id"], name="Linked preparation")
    if preparation_inputs_id != prepared_id:
        raise ValueError(f"Linked preparation content changed after it was loaded for inference: {payload}")
    destination = start_checkpoint(
        output_dir, "sample", payloads=("posterior.nc",), referenced=(preparation_path, payload),
    )
    save_trace(trace, destination / "posterior.nc")
    return finish_checkpoint(destination, "sample", {
        "schema": "openghg_inversions.co2_o2.sample", "schema_version": 2,
        "producer": "openghg_inversions", "stage": "sample", "inference_kind": "mcmc",
        "recipe": trace.attrs["rhime_recipe"],
        "preparation": os.path.relpath(preparation_path, destination),
        "preparation_generation": preparation_generation, "posterior": "posterior.nc",
        "prepared_inputs_id": prepared_id,
        "posterior_id": prepared_inputs_content_id(destination / "posterior.nc"),
        "actual_inference": dict(actual_inference), "provenance": installed_ogi_provenance(),
    })


def sample(
    preparation_manifest: str | Path, *, model: Co2O2ModelConfig | Mapping[str, Any], output_dir: str | Path,
    sampler: RhimeSampler | Literal["default"] = "default", referenced: tuple[str | Path, ...] = (),
) -> Path:
    """Sample one joint model using current scientific choices and saved inputs.

    Compatible baseline and fixed mismatch changes reuse preparation; defining
    retained priors, operators and cross-channel covariance remain prepared facts.
    The full prepared-input runner and this route use the same construction,
    sampling and scientific annotation operations.
    ``referenced`` names consumed request files, such as the CLI configuration,
    that this stage's outputs must not overwrite or enclose.
    """
    options = load_co2_o2_params(params=model) if isinstance(model, Mapping) else {}
    model_config = model if isinstance(model, Co2O2ModelConfig) else Co2O2ModelConfig.from_params(options)
    if sampler == "default":
        active_sampler = resolve_co2_o2_sampler(options, model=model_config)
    else:
        active_sampler = resolve_sampler(sampler)
    if model_config.tau_hours is not None and active_sampler.nuts_sampler != "pymc":
        raise ValueError("Linked fixed-OU requires nuts_sampler='pymc'.")
    preparation = load_preparation(preparation_manifest)
    check_checkpoint_targets(
        Path(output_dir).resolve(), stage="sample", payloads=("posterior.nc",),
        referenced=(Path(preparation_manifest).resolve(), preparation.payload) + model_config.artifact_paths()
        + tuple(Path(path).resolve() for path in referenced),
    )
    model_options = model_config.bind(preparation.prepared)
    if model_config.recipe == "co2_o2_shared_baseline":
        from .shared_workflow import sample_linked_shared_workflow
        trace = sample_linked_shared_workflow(preparation.prepared, sampler=active_sampler, **model_options)
    else:
        built = build_rhime_co2_o2(prepared_inputs=preparation.prepared, **model_options)
        trace = sample_rhime_model(built, active_sampler, recompute_deterministics=True)
        trace = annotate_co2_o2_trace(trace, built=built, biosphere_prior=model_options.get("biosphere_prior"))
    return save_samples(
        preparation_manifest, trace, actual_inference=active_sampler.describe(),
        preparation_generation=preparation.generation, output_dir=output_dir,
        preparation_inputs_id=preparation.inputs_id,
    )


def load_samples(sample_manifest: str | Path) -> Co2O2Output:
    """Reopen the joint posterior and scientific context without constructing a graph.

    Saved inference provenance remains descriptive. Original requested dates and
    domain come from preparation; replay does not resolve current inference
    choices or generate missing predictions.
    """
    path, record = read_manifest(sample_manifest, "openghg_inversions.co2_o2.sample", (
        "preparation", "preparation_generation", "posterior", "inference_kind", "recipe", "prepared_inputs_id", "posterior_id",
    ), schema_version=2)
    if record["inference_kind"] != "mcmc":
        raise ValueError(f"Linked samples {path} require inference_kind='mcmc'.")
    generation = record["preparation_generation"]
    if not isinstance(generation, str) or not generation:
        raise ValueError(f"Linked samples {path} require a nonempty preparation generation.")
    preparation = load_preparation(reference(path, record["preparation"]), expected_generation=generation,
                                   expected_inputs_id=record["prepared_inputs_id"])
    posterior = reference(path, record["posterior"])
    _check_saved_payload(posterior, record["posterior_id"], name="Linked posterior")
    trace = load_trace(posterior)
    if record["recipe"] != trace.attrs.get("rhime_recipe"):
        raise ValueError("Linked sample recipe does not match its posterior.")
    return make_co2_o2_output(
        prepared=preparation.prepared, trace=trace, start_date=preparation.start_date,
        end_date=preparation.end_date, domain=preparation.domain,
        output_metadata={
            "sampler": describe_recorded_sampler(record.get("actual_inference")),
            "provenance": record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {},
        },
    )


def postprocess(
    sample_manifest: str | Path, *, output: Co2O2OutputOptions, referenced: tuple[str | Path, ...] = (),
) -> Co2O2Output:
    """Calculate and export requested products from saved linked scientific meaning.

    Products are validated and calculated before any output destination opens.
    The posterior and preparation stay borrowed. Prior/predictive outputs are
    available only when the corresponding draws were saved.
    ``referenced`` names consumed request files, such as the CLI configuration,
    that this stage's outputs must not overwrite or enclose.
    """
    result = load_samples(sample_manifest)
    calculate_co2_o2_products(result=result, output=output)
    path, record = read_manifest(sample_manifest, "openghg_inversions.co2_o2.sample", schema_version=2)
    preparation_path = reference(path, record["preparation"])
    _, preparation = _preparation_record(preparation_path)
    save_co2_o2_products(result, output=output, referenced=(
        path, preparation_path, reference(preparation_path, preparation["prepared_inputs"]),
        reference(path, record["posterior"]),
    ) + tuple(Path(path).resolve() for path in referenced))
    return result


def prior_predictive(
    preparation_manifest: str | Path, *, model: Co2O2ModelConfig | Mapping[str, Any], output_dir: str | Path,
    draws: int = 100, check_output: str | Path | None = None, stage: str = "prior-predictive",
    random_seed: int | None = None, referenced: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Generate native joint prior predictions and report finite-draw readiness.

    This builds the configured joint model without creating an MCMC step.
    Construction errors propagate; evaluated missing, empty or nonfinite
    predictive evidence produces a failed readiness report.
    ``referenced`` names consumed request files, such as the CLI configuration,
    that this stage's outputs must not overwrite or enclose.
    """
    _check_draws(draws)
    stage = _check_stage(stage)
    options = load_co2_o2_params(params=model) if isinstance(model, Mapping) else {}
    model_config = model if isinstance(model, Co2O2ModelConfig) else Co2O2ModelConfig.from_params(options)
    preparation = load_preparation(preparation_manifest)
    destination = Path(output_dir).resolve()
    report = _output_path(destination, check_output, "prior-predictive-readiness.json")
    check_checkpoint_targets(
        destination, stage=None, payloads=("prior-predictive.nc", str(report.relative_to(destination))),
        referenced=(Path(preparation_manifest).resolve(), preparation.payload) + model_config.artifact_paths()
        + tuple(Path(path).resolve() for path in referenced),
    )
    report.unlink(missing_ok=True)
    model_options = model_config.bind(preparation.prepared)
    if model_config.recipe == "co2_o2_shared_baseline":
        from .shared_workflow import build_linked_shared_workflow
        built, _, materialized_prepared = build_linked_shared_workflow(preparation.prepared, **model_options)
    else:
        built = build_rhime_co2_o2(prepared_inputs=preparation.prepared, **model_options)
    with built.model:
        prior = pm.sample_prior_predictive(draws, random_seed=random_seed)
    registry = get_coord_registry(built.model)
    if registry is not None:
        prior = restore_inferencedata_coords(prior, registry)
    if model_config.recipe == "co2_o2_shared_baseline":
        from .shared_workflow import annotate_linked_shared_workflow
        prior = annotate_linked_shared_workflow(prior, built=built, prepared=materialized_prepared)
    else:
        prior = annotate_co2_o2_trace(prior, built=built, biosphere_prior=model_options.get("biosphere_prior"))
    concentration = built.output_contract.variable_roles.get("concentration")
    predictive = prior.children.get("prior_predictive")
    values = [np.asarray(group[name].values) for group in prior.children.values() for name in group.data_vars]
    non_finite = sum(int(value.size - np.isfinite(value).sum()) for value in values)
    status = "fail"
    if predictive is None or concentration not in predictive:
        message = "Prior prediction is missing the declared concentration draws."
    elif predictive[concentration].size == 0:
        message = "Prior prediction contains no concentration draws."
    elif non_finite:
        message = f"Prior predictive contains {non_finite} non-finite values."
    else:
        status = "pass"
        message = f"Prior predictive produced {draws} finite joint draws."
    destination.mkdir(parents=True, exist_ok=True)
    prior_path = destination / "prior-predictive.nc"
    save_trace(prior, prior_path)
    result = _check_result(
        name=PREPARATION_CHECK_NAME, status=status,
        measured_values={"draws": draws, "non_finite_values": non_finite},
        thresholds={"max_non_finite_values": 0}, message=message,
        artifact_paths=[str(prior_path)], stage=stage,
    )
    write_json(report, result)
    return result


__all__ = [
    "Co2O2Preparation", "prepare", "save_preparation", "load_preparation",
    "sample", "save_samples", "load_samples", "postprocess", "prior_predictive",
]
