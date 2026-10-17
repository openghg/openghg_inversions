"""Save and reopen the ordinary CO2 recipe's scientific handoffs.

Manifests retain original requested dates and point to existing numerical
codecs. They do not authenticate model configuration or execute recorded code.
Callers allocate checkpoint directories; writers publish completion last.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact, save as save_affine
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime._stage_artifacts import (
    _output_path, check_checkpoint_targets, finish_checkpoint, read_manifest,
    reference, start_checkpoint, write_json,
)
from openghg_inversions.rhime._stage_checks import (
    PREPARATION_CHECK_NAME, _check_draws, _check_stage, _check_result,
)
from openghg_inversions.rhime.sampling import sample_rhime_model
from openghg_inversions.serialization import load_trace, save_trace

from .co2_affine_output import (
    load_and_bind_affine_flux_map, prepared_inputs_content_id,
)
from .co2_outputs import (
    Co2Output, Co2OutputOptions, make_co2_output, calculate_co2_products, save_co2_products,
    validate_co2_trace, validate_co2_product_request,
)
from .co2_preparation import (
    Co2PreparedInputs, prepare_co2_inputs, requested_co2_facts,
)
from .co2_runner import (
    annotate_co2_trace, materialize_co2_inputs, build_co2_from_materialized,
)
from .configuration import resolve_co2_model


@dataclass(frozen=True)
class Co2Preparation:
    """Loaded coherent inputs, reconstruction reference and original request.

    ``generation`` must be returned to :func:`save_samples` after inference.
    ``payload`` identifies the numerical prepared artifact, independently of its
    recorded original inventory paths. Missing request facts are never inferred
    from retained observations.
    """

    prepared: Co2PreparedInputs
    generation: str
    start_date: str
    end_date: str
    domain: str
    payload: Path
    affine_payload: Path | None = None


def _preparation_record(manifest: str | Path) -> tuple[Path, dict[str, Any]]:
    path, record = read_manifest(manifest, "openghg_inversions.co2.prepare", (
        "prepared_inputs", "generation", "start_date", "end_date", "domain",
    ))
    if not isinstance(record["generation"], str) or not record["generation"]:
        raise ValueError(f"CO2 preparation {path} requires a nonempty generation.")
    return path, record


def _finish_preparation(
    destination: Path, *, prepared_path: Path, affine_path: Path | None,
    facts: Mapping[str, str],
) -> Path:
    """Publish one preparation record after saving or validating its numerical pair."""
    record = {
        "schema": "openghg_inversions.co2.prepare", "schema_version": 1,
        "producer": "openghg_inversions", "stage": "prepare", "generation": uuid4().hex,
        "prepared_inputs": os.path.relpath(prepared_path, destination), **facts,
        "provenance": installed_ogi_provenance(),
    }
    if affine_path is not None:
        record["affine_reconstruction"] = os.path.relpath(affine_path, destination)
    return finish_checkpoint(destination, "prepare", record)


def save_preparation(
    prepared: Co2PreparedInputs, *, output_dir: str | Path,
    start_date: str | None = None, end_date: str | None = None, domain: str | None = None,
    affine_artifact: AffineFluxMapArtifact | None = None,
) -> Path:
    """Serialize coherent inputs and an optional new affine companion, manifest last.

    Requested dates/domain must be explicit or present in preparation provenance;
    observation extrema cannot recover a filtered request. The affine companion
    is rebound to the saved numerical bytes and checked by the existing loader,
    including prepared projection, source/grid labels, signed flux and operator.
    A successful replacement assigns a new generation and invalidates old samples.
    """
    facts = requested_co2_facts(prepared, start_date=start_date, end_date=end_date, domain=domain)
    payloads = ("prepared-inputs.nc",) + (("affine-flux-map.nc",) if affine_artifact is not None else ())
    destination = start_checkpoint(output_dir, "prepare", payloads=payloads)
    prepared_path = destination / "prepared-inputs.nc"
    prepared.save(prepared_path)
    companion = None
    if affine_artifact is not None:
        companion = destination / "affine-flux-map.nc"
        save_affine(replace(affine_artifact, prepared_inputs_id=prepared_inputs_content_id(prepared_path)), companion)
        load_and_bind_affine_flux_map(companion, prepared_path)
    return _finish_preparation(
        destination, prepared_path=prepared_path, affine_path=companion, facts=facts,
    )


def prepare(
    *, output_dir: str | Path, prepared_path: str | Path | None = None,
    start_date: str | None = None, end_date: str | None = None, domain: str | None = None,
    canonical_inputs: RhimePreparedInputs | None = None,
    reduction: CoherentGaussianReduction | None = None,
    aggregation_error_rank: int | None = 512,
    affine_artifact: AffineFluxMapArtifact | None = None, affine_path: str | Path | None = None,
) -> Path:
    """Prepare a coherent reduction or import a saved prepared/affine pair.

    Pass canonical inputs and their defining coherent reduction together, or one
    saved prepared path. Imported pairs are validated and referenced directly;
    callers retain those files for later sampling/replay. New pairs use
    :func:`save_preparation` and the existing numerical codecs. Defining priors, covariance, operators and
    projections belong to the coherent preparation, never an inference override.
    """
    if (canonical_inputs is None) != (reduction is None):
        raise ValueError("Pass canonical_inputs and reduction together.")
    if (prepared_path is None) == (canonical_inputs is None):
        raise ValueError("Pass either prepared_path or canonical_inputs with reduction.")
    if affine_artifact is not None and affine_path is not None:
        raise ValueError("Pass either affine_artifact or affine_path.")
    if canonical_inputs is not None:
        if affine_path is not None:
            raise ValueError("A new preparation requires affine_artifact, not an imported companion path.")
        prepared = prepare_co2_inputs(
            canonical_inputs, cast(CoherentGaussianReduction, reduction),
            aggregation_error_rank=aggregation_error_rank,
        )
        return save_preparation(
            prepared, output_dir=output_dir, start_date=start_date, end_date=end_date, domain=domain,
            affine_artifact=affine_artifact,
        )
    source = Path(cast(str | Path, prepared_path)).resolve()
    prepared = Co2PreparedInputs.load(source)
    if affine_artifact is not None:
        raise ValueError("Imported preparation requires affine_path; use save_preparation for a new pair.")
    facts = requested_co2_facts(prepared, start_date=start_date, end_date=end_date, domain=domain)
    companion = Path(affine_path).resolve() if affine_path is not None else None
    if companion is not None:
        load_and_bind_affine_flux_map(companion, source)
    destination = start_checkpoint(
        output_dir, "prepare", payloads=(), referenced=(source,) + ((companion,) if companion else ()),
    )
    return _finish_preparation(
        destination, prepared_path=source, affine_path=companion, facts=facts,
    )


def load_preparation(manifest: str | Path, *, expected_generation: str | None = None) -> Co2Preparation:
    """Load one completed numerical preparation and original requested facts.

    Relative references resolve beside the manifest. Missing artifacts and
    invalid scientific bindings propagate from their owning numerical loaders.
    An expected generation rejects replacement of a preparation after sampling.
    Affine references stay unloaded until scientific output replay consumes them.
    """
    path, record = _preparation_record(manifest)
    if expected_generation is not None and record["generation"] != expected_generation:
        raise ValueError(f"CO2 preparation {path} was replaced; samples require generation {expected_generation!r}.")
    payload = reference(path, record["prepared_inputs"])
    prepared = Co2PreparedInputs.load(payload)
    facts = requested_co2_facts(prepared, **{name: record[name] for name in ("start_date", "end_date", "domain")})
    affine_payload = reference(path, record["affine_reconstruction"]) if "affine_reconstruction" in record else None
    return Co2Preparation(
        prepared=prepared, generation=record["generation"], payload=payload, affine_payload=affine_payload, **facts,
    )


def save_samples(
    preparation_manifest: str | Path, trace: xr.DataTree, *, actual_inference: Mapping[str, Any],
    preparation_generation: str, output_dir: str | Path,
) -> Path:
    """Save annotated samples and inference provenance bound to a preparation generation.

    The scientific trace owns model roles and used prior descriptions. Current
    configuration identity and callable representations do not gate reuse. Model
    choices are live inference inputs; replay uses only saved scientific meaning.
    """
    preparation_path, preparation = _preparation_record(preparation_manifest)
    if preparation["generation"] != preparation_generation:
        raise ValueError(f"CO2 preparation {preparation_path} was replaced; samples require generation {preparation_generation!r}.")
    validate_co2_trace(trace)
    prepared_path = reference(preparation_path, preparation["prepared_inputs"])
    affine_path = (
        reference(preparation_path, preparation["affine_reconstruction"])
        if "affine_reconstruction" in preparation else None
    )
    destination = start_checkpoint(
        output_dir, "sample", payloads=("posterior.nc",),
        referenced=(preparation_path, prepared_path) + ((affine_path,) if affine_path is not None else ()),
    )
    save_trace(trace, destination / "posterior.nc")
    return finish_checkpoint(destination, "sample", {
        "schema": "openghg_inversions.co2.sample", "schema_version": 1,
        "producer": "openghg_inversions", "stage": "sample", "inference_kind": "mcmc",
        "preparation": os.path.relpath(preparation_path, destination),
        "preparation_generation": preparation_generation, "posterior": "posterior.nc",
        "actual_inference": dict(actual_inference), "provenance": installed_ogi_provenance(),
    })


def sample(
    preparation_manifest: str | Path, *, model: Mapping[str, Any], output_dir: str | Path,
    sampler: RhimeSampler | Literal["default"] = "default",
) -> Path:
    """Build and sample ordinary CO2 using current science and its resolved sampler.

    Compatible independent BC/offset priors and supported likelihood choices can
    reuse preparation; the defining coherent native reduction stays unchanged.
    Full execution and saved sampling share the explicit build and sample owners.
    ``sampler="default"`` selects fresh defaults; an explicit sampler supplies
    configured sampling choices. Direct runner trace returns remain unchanged.
    """
    active_sampler = resolve_sampler(sampler)
    preparation = load_preparation(preparation_manifest)
    model_options = resolve_co2_model(model)
    check_checkpoint_targets(
        Path(output_dir).resolve(), stage="sample", payloads=("posterior.nc",),
        referenced=(Path(preparation_manifest).resolve(), preparation.payload)
        + ((preparation.affine_payload,) if preparation.affine_payload else ()),
    )
    model_inputs, priors = materialize_co2_inputs(
        prepared_inputs=preparation.prepared, **model_options,
    )
    built = build_co2_from_materialized(
        model_inputs, prepared_inputs=preparation.prepared,
        priors=priors, **model_options,
    )
    trace = sample_rhime_model(built, active_sampler)
    annotate_co2_trace(
        trace, built, concentration_units=preparation.prepared.inv_inputs["mf"].attrs.get("units"),
        likelihood_builder=model_options.get("likelihood_builder"),
        likelihood_kwargs=model_options.get("likelihood_kwargs"),
    )
    return save_samples(
        preparation_manifest, trace, actual_inference=active_sampler.describe(), preparation_generation=preparation.generation,
        output_dir=output_dir,
    )


def load_samples(sample_manifest: str | Path) -> Co2Output:
    """Load saved scientific meaning without acquisition, construction or inference.

    Replayed results retain ``sampler=None`` and plain recorded inference and
    installed-version provenance. Original requested dates are taken from the
    preparation, independently of observation extrema or current model settings.
    """
    path, record = read_manifest(sample_manifest, "openghg_inversions.co2.sample", (
        "preparation", "preparation_generation", "posterior", "inference_kind",
    ))
    if record["inference_kind"] != "mcmc":
        raise ValueError(f"CO2 samples {path} require inference_kind='mcmc'.")
    generation = record["preparation_generation"]
    if not isinstance(generation, str) or not generation:
        raise ValueError(f"CO2 samples {path} require a nonempty preparation generation.")
    preparation = load_preparation(reference(path, record["preparation"]), expected_generation=generation)
    trace = load_trace(reference(path, record["posterior"]))
    validate_co2_trace(trace, preparation.prepared)
    bound = (
        load_and_bind_affine_flux_map(preparation.affine_payload, preparation.payload)
        if preparation.affine_payload is not None else None
    )
    result = make_co2_output(
        prepared=preparation.prepared, trace=trace, bound=bound,
        start_date=preparation.start_date, end_date=preparation.end_date, domain=preparation.domain,
        output_metadata={
            "sampler": record.get("actual_inference") if isinstance(record.get("actual_inference"), Mapping) else {},
            "provenance": record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {},
        },
    )
    return result


def postprocess(
    sample_manifest: str | Path, *, output: Co2OutputOptions,
    source_to_sector: Mapping[str, str] | None = None,
) -> Co2Output:
    """Reopen one sample manifest, calculate requested products and export them.

    Products use saved coherent/affine scientific meaning and current reporting
    choices. All product validation/calculation precedes opening output paths.
    """
    loaded = load_samples(sample_manifest)
    countries = validate_co2_product_request(
        output, prepared=loaded.prepared, bound=loaded.bound, source_to_sector=source_to_sector,
    )
    calculate_co2_products(
        result=loaded, prepared=loaded.prepared, bound=loaded.bound,
        output=output, source_to_sector=source_to_sector, countries=countries,
    )
    save_co2_products(loaded, output=output)
    return loaded


def prior_predictive(
    preparation_manifest: str | Path, *, model: Mapping[str, Any], output_dir: str | Path,
    draws: int = 100, check_output: str | Path | None = None, stage: str = "prior-predictive",
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Check exact graph prior predictions without posterior inference.

    Construction/inference errors propagate before creating a destination. Only
    evaluated missing, empty or non-finite predictive evidence becomes a failed
    report. The check never repairs absent posterior/replay evidence.
    """
    _check_draws(draws)
    stage = _check_stage(stage)
    preparation = load_preparation(preparation_manifest)
    model_options = resolve_co2_model(model)
    destination = Path(output_dir).resolve()
    report = _output_path(destination, check_output, "prior-predictive-readiness.json")
    check_checkpoint_targets(
        destination, stage=None, payloads=("prior-predictive.nc", str(report.relative_to(destination))),
        referenced=(Path(preparation_manifest).resolve(), preparation.payload)
        + ((preparation.affine_payload,) if preparation.affine_payload else ()),
    )
    report.unlink(missing_ok=True)
    model_inputs, priors = materialize_co2_inputs(
        prepared_inputs=preparation.prepared, **model_options,
    )
    built = build_co2_from_materialized(
        model_inputs, prepared_inputs=preparation.prepared,
        priors=priors, **model_options,
    )
    with built.model:
        prior = pm.sample_prior_predictive(draws, random_seed=random_seed)
    registry = get_coord_registry(built.model)
    if registry is not None:
        prior = restore_inferencedata_coords(prior, registry)
    annotate_co2_trace(
        prior, built, concentration_units=preparation.prepared.inv_inputs["mf"].attrs.get("units"),
        likelihood_builder=model_options.get("likelihood_builder"),
        likelihood_kwargs=model_options.get("likelihood_kwargs"),
    )
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
        message = f"Prior predictive produced {draws} finite draws."
    destination.mkdir(parents=True, exist_ok=True)
    prior_path = destination / "prior-predictive.nc"
    save_trace(prior, prior_path)
    artifacts = [str(prior_path)]
    result = _check_result(
        name=PREPARATION_CHECK_NAME, status=status,
        measured_values={"draws": draws, "non_finite_values": non_finite},
        thresholds={"max_non_finite_values": 0}, message=message, artifact_paths=artifacts, stage=stage,
    )
    write_json(report, result)
    return result


__all__ = [
    "Co2Preparation", "prepare", "save_preparation", "load_preparation",
    "sample", "save_samples", "load_samples", "postprocess", "prior_predictive",
]
