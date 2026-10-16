"""Save and reopen the ordinary CO2 recipe's scientific handoffs.

Manifests retain original requested dates and point to existing numerical
codecs. They do not authenticate model configuration or execute recorded code.
Callers allocate checkpoint directories; writers publish completion last.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal
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
from openghg_inversions.rhime.specs import RhimeOutputSpec
from openghg_inversions.rhime.builders import callable_metadata
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.sampling import sample_rhime_model
from openghg_inversions.serialization import load_trace, save_trace

from .co2_affine_output import (
    BoundCo2AffineFluxMap, load_and_bind_affine_flux_map, prepared_inputs_content_id,
)
from .co2_outputs import (
    Co2Output, make_co2_output, calculate_co2_products, save_co2_products, validate_co2_outputs,
)
from .co2_preparation import (
    Co2PreparedInputs, prepare_co2_inputs, requested_co2_facts, _require_equivalent_units,
)
from .co2_runner import (
    annotate_co2_trace, materialize_co2_inputs, build_co2_from_materialized, describe_co2_likelihood_kwargs,
)
from .configuration import Co2RunSetup


@dataclass(frozen=True)
class Co2Preparation:
    """Loaded coherent inputs, optional bound reconstruction and original request.

    ``generation`` must be returned to :func:`save_samples` after inference.
    ``payload`` identifies the numerical prepared artifact, independently of its
    recorded original inventory paths. Missing request facts are never inferred
    from retained observations.
    """

    prepared: Co2PreparedInputs
    bound: BoundCo2AffineFluxMap | None
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
    record = {
        "schema": "openghg_inversions.co2.prepare", "schema_version": 1,
        "producer": "openghg_inversions", "stage": "prepare", "generation": uuid4().hex,
        "prepared_inputs": "prepared-inputs.nc", **facts,
        "provenance": installed_ogi_provenance(),
    }
    if affine_artifact is not None:
        companion = destination / "affine-flux-map.nc"
        save_affine(replace(affine_artifact, prepared_inputs_id=prepared_inputs_content_id(prepared_path)), companion)
        load_and_bind_affine_flux_map(companion, prepared_path)
        record["affine_reconstruction"] = "affine-flux-map.nc"
    return finish_checkpoint(destination, "prepare", record)


def _copy_artifact(source: Path, destination: Path) -> None:
    """Copy an explicitly imported artifact in its existing numerical format."""
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)


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
    saved prepared path. Imported affine pairs are copied byte-for-byte to retain
    their numerical binding. New pairs use :func:`save_preparation` and the
    existing numerical codecs. Defining priors, covariance, operators and
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
        prepared = prepare_co2_inputs(canonical_inputs, reduction, aggregation_error_rank=aggregation_error_rank)
        return save_preparation(
            prepared, output_dir=output_dir, start_date=start_date, end_date=end_date, domain=domain,
            affine_artifact=affine_artifact,
        )
    source = Path(prepared_path).resolve()
    prepared = Co2PreparedInputs.load(source)
    if affine_artifact is not None:
        return save_preparation(
            prepared, output_dir=output_dir, start_date=start_date, end_date=end_date, domain=domain,
            affine_artifact=affine_artifact,
        )
    facts = requested_co2_facts(prepared, start_date=start_date, end_date=end_date, domain=domain)
    companion = Path(affine_path).resolve() if affine_path is not None else None
    if companion is not None:
        load_and_bind_affine_flux_map(companion, source)
    prepared_name = "prepared-inputs.zarr" if source.is_dir() else "prepared-inputs.nc"
    affine_name = "affine-flux-map.zarr" if companion is not None and companion.is_dir() else "affine-flux-map.nc"
    payloads = (prepared_name,) + ((affine_name,) if companion is not None else ())
    destination = start_checkpoint(
        output_dir, "prepare", payloads=payloads, referenced=(source,) + ((companion,) if companion else ()),
    )
    _copy_artifact(source, destination / prepared_name)
    record = {
        "schema": "openghg_inversions.co2.prepare", "schema_version": 1,
        "producer": "openghg_inversions", "stage": "prepare", "generation": uuid4().hex,
        "prepared_inputs": prepared_name, **facts,
        "provenance": installed_ogi_provenance(),
    }
    if companion is not None:
        _copy_artifact(companion, destination / affine_name)
        record["affine_reconstruction"] = affine_name
    return finish_checkpoint(destination, "prepare", record)


def load_preparation(manifest: str | Path, *, expected_generation: str | None = None) -> Co2Preparation:
    """Load one completed numerical preparation and original requested facts.

    Relative references resolve beside the manifest. Missing artifacts and
    invalid scientific bindings propagate from their owning numerical loaders.
    An expected generation rejects replacement of a preparation after sampling.
    """
    path, record = _preparation_record(manifest)
    if expected_generation is not None and record["generation"] != expected_generation:
        raise ValueError(f"CO2 preparation {path} was replaced; samples require generation {expected_generation!r}.")
    payload = reference(path, record["prepared_inputs"])
    prepared = Co2PreparedInputs.load(payload)
    facts = requested_co2_facts(prepared, **{name: record[name] for name in ("start_date", "end_date", "domain")})
    affine_payload = reference(path, record["affine_reconstruction"]) if "affine_reconstruction" in record else None
    bound = load_and_bind_affine_flux_map(affine_payload, payload) if affine_payload is not None else None
    return Co2Preparation(
        prepared, bound, record["generation"], payload=payload, affine_payload=affine_payload, **facts,
    )


def _validate_trace(trace: xr.DataTree, prepared: Co2PreparedInputs | None = None) -> None:
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


def _require_ordinary(model: Co2RunSetup) -> None:
    if model.variant != "ordinary":
        raise ValueError("Ordinary CO2 handoffs require variant='ordinary'; cached adoption remains separate.")


def save_samples(
    preparation_manifest: str | Path, trace: xr.DataTree, *, actual_inference: Mapping[str, Any],
    preparation_generation: str, output_dir: str | Path,
    source_to_sector: Mapping[str, str] | None = None,
) -> Path:
    """Save annotated samples and inference provenance bound to a preparation generation.

    The scientific trace owns model roles and used prior descriptions. Current
    configuration identity and callable representations do not gate reuse. Model
    choices are live inference inputs; replay uses only saved scientific meaning.
    """
    preparation_path, preparation = _preparation_record(preparation_manifest)
    if preparation["generation"] != preparation_generation:
        raise ValueError(f"CO2 preparation {preparation_path} was replaced; samples require generation {preparation_generation!r}.")
    _validate_trace(trace)
    prepared_path = reference(preparation_path, preparation["prepared_inputs"])
    affine_path = (
        reference(preparation_path, preparation["affine_reconstruction"])
        if "affine_reconstruction" in preparation else None
    )
    if source_to_sector is not None and (
        not isinstance(source_to_sector, Mapping) or not source_to_sector
        or any(not isinstance(key, str) or not isinstance(value, str) or not value.isidentifier()
               for key, value in source_to_sector.items())
    ):
        raise ValueError("CO2 source_to_sector must map native sources to identifier sector names.")
    if source_to_sector is not None and affine_path is None:
        raise ValueError("CO2 source_to_sector requires a bound affine reconstruction.")
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
        "source_to_sector": dict(source_to_sector) if source_to_sector is not None else None,
    })


def sample(
    preparation_manifest: str | Path, *, model: Co2RunSetup, output_dir: str | Path,
    source_to_sector: Mapping[str, str] | None = None,
    sampler: RhimeSampler | Literal["default"] = "default",
) -> Path:
    """Build and sample ordinary CO2 using current science and its resolved sampler.

    Compatible independent BC/offset priors and supported likelihood choices can
    reuse preparation; the defining coherent native reduction stays unchanged.
    Full execution and saved sampling share the explicit build and sample owners.
    ``sampler="default"`` selects fresh defaults; pass ``model.sampler`` to use
    configured sampler choices. Direct runner trace returns remain unchanged.
    """
    _require_ordinary(model)
    active_sampler = resolve_sampler(sampler)
    preparation = load_preparation(preparation_manifest)
    check_checkpoint_targets(
        Path(output_dir).resolve(), stage="sample", payloads=("posterior.nc",),
        referenced=(Path(preparation_manifest).resolve(), preparation.payload)
        + ((preparation.affine_payload,) if preparation.affine_payload else ()),
    )
    validate_co2_outputs(
        RhimeOutputSpec(output_format="none", save_inversion_output=False),
        bound=preparation.bound, source_to_sector=source_to_sector,
    )
    model_inputs, materialized_options = materialize_co2_inputs(
        prepared_inputs=preparation.prepared, **model.runner_kwargs,
    )
    built = build_co2_from_materialized(
        model_inputs, prepared_inputs=preparation.prepared,
        **{**model.runner_kwargs, **materialized_options},
    )
    trace = sample_rhime_model(built, active_sampler)
    annotate_co2_trace(trace, built, concentration_units=preparation.prepared.inv_inputs["mf"].attrs.get("units"))
    likelihood_builder = model.runner_kwargs.get("likelihood_builder")
    if likelihood_builder is not None:
        annotate_likelihood_trace(
            trace, builder_identity=callable_metadata(likelihood_builder),
            likelihood_kwargs=describe_co2_likelihood_kwargs(model.runner_kwargs.get("likelihood_kwargs")),
        )
    return save_samples(
        preparation_manifest, trace, actual_inference=active_sampler.describe(), preparation_generation=preparation.generation,
        output_dir=output_dir, source_to_sector=source_to_sector,
    )


def load_samples(
    sample_manifest: str | Path, *, output: RhimeOutputSpec,
    source_to_sector: Mapping[str, str] | None = None,
) -> Co2Output:
    """Load saved scientific meaning without acquisition, construction or inference.

    Replayed results retain ``sampler=None`` and plain recorded inference and
    installed-version provenance. Original requested dates are taken from the
    preparation, independently of observation extrema or current model settings.
    """
    path, record = read_manifest(sample_manifest, "openghg_inversions.co2.sample", (
        "preparation", "preparation_generation", "posterior", "source_to_sector", "inference_kind",
    ))
    if record["inference_kind"] != "mcmc":
        raise ValueError(f"CO2 samples {path} require inference_kind='mcmc'.")
    generation = record["preparation_generation"]
    if not isinstance(generation, str) or not generation:
        raise ValueError(f"CO2 samples {path} require a nonempty preparation generation.")
    preparation = load_preparation(reference(path, record["preparation"]), expected_generation=generation)
    trace = load_trace(reference(path, record["posterior"]))
    _validate_trace(trace, preparation.prepared)
    result = make_co2_output(
        prepared=preparation.prepared, trace=trace, sampler=None, bound=preparation.bound,
        source_to_sector=record["source_to_sector"] if source_to_sector is None else source_to_sector,
        start_date=preparation.start_date, end_date=preparation.end_date, domain=preparation.domain,
        output=output, output_metadata={
            "sampler": record.get("actual_inference") if isinstance(record.get("actual_inference"), Mapping) else {},
            "provenance": record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {},
        },
    )
    return result


def postprocess(
    sample_manifest: str | Path, *, output: RhimeOutputSpec,
    source_to_sector: Mapping[str, str] | None = None,
) -> Co2Output:
    """Reopen one sample manifest, calculate requested products and export them.

    Products use saved coherent/affine scientific meaning and current reporting
    choices. All product validation/calculation precedes opening output paths.
    """
    loaded = load_samples(sample_manifest, output=output, source_to_sector=source_to_sector)
    calculate_co2_products(
        result=loaded, prepared=loaded.prepared, bound=loaded.bound,
        source_to_sector=loaded.source_to_sector,
    )
    save_co2_products(loaded)
    return loaded


def prior_predictive(
    preparation_manifest: str | Path, *, model: Co2RunSetup, output_dir: str | Path,
    draws: int = 100, check_output: str | Path | None = None, stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Check exact graph prior predictions without posterior inference.

    Construction/inference errors propagate before creating a destination. Only
    evaluated missing, empty or non-finite predictive evidence becomes a failed
    report. The check never repairs absent posterior/replay evidence.
    """
    _require_ordinary(model)
    _check_draws(draws)
    stage = _check_stage(stage)
    preparation = load_preparation(preparation_manifest)
    destination = Path(output_dir).resolve()
    report = _output_path(destination, check_output, "prior-predictive-readiness.json")
    check_checkpoint_targets(
        destination, stage=None, payloads=("prior-predictive.nc", str(report.relative_to(destination))),
        referenced=(Path(preparation_manifest).resolve(), preparation.payload)
        + ((preparation.affine_payload,) if preparation.affine_payload else ()),
    )
    report.unlink(missing_ok=True)
    model_inputs, materialized_options = materialize_co2_inputs(
        prepared_inputs=preparation.prepared, **model.runner_kwargs,
    )
    built = build_co2_from_materialized(
        model_inputs, prepared_inputs=preparation.prepared,
        **{**model.runner_kwargs, **materialized_options},
    )
    seed = dict(model.sampler.sample_kwargs or {}).get("random_seed")
    with built.model:
        prior = pm.sample_prior_predictive(draws, random_seed=seed)
    registry = get_coord_registry(built.model)
    if registry is not None:
        prior = restore_inferencedata_coords(prior, registry)
    annotate_co2_trace(
        prior, built, concentration_units=preparation.prepared.inv_inputs["mf"].attrs.get("units"),
    )
    likelihood_builder = model.runner_kwargs.get("likelihood_builder")
    if likelihood_builder is not None:
        annotate_likelihood_trace(
            prior, builder_identity=callable_metadata(likelihood_builder),
            likelihood_kwargs=describe_co2_likelihood_kwargs(model.runner_kwargs.get("likelihood_kwargs")),
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
