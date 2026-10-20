"""Save and reload the scientific handoffs used by standard execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
from typing import Any, Literal
from uuid import uuid4

import xarray as xr

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions._record_schema import validate_json_keys
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.serialization import load_trace, save_trace

from ._stage_artifacts import (
    _output_path, check_checkpoint_targets, finish_checkpoint, json_value,
    reference, start_checkpoint,
)
from ._manifest_records import (
    StandardPreparationRecord, StandardSampleRecord, StandardSampleReadRecord, checkpoint_dict, read_checkpoint,
)
from ._stage_checks import _check_draws, _check_stage, prior_predictive_check
from .builders import RhimeLikelihoodBuilder
from .outputs import (
    annotate_sample_trace, make_model_output, postprocess_standard_outputs,
    save_inversion_output, validate_standard_output_resources,
)
from .sampling import sample_rhime_model
from .params import DataConfig, resolve_model_options
from .specs import RhimeModelSpec, RhimeOutputSpec
from .standard import build_standard_model, materialize_standard_inputs, prepare_standard
from .preparation import acquire_merged_data, resolve_basis_inputs


def _require_standard(data_config: DataConfig) -> None:
    """Reject directly constructed sector-split requests at the public boundary."""
    if data_config.split_by_sectors:
        raise ValueError("Standard handoffs require `split_by_sectors=False`; use the multisector stages.")


def _check_generation(manifest: Path, record: StandardPreparationRecord, expected: str) -> None:
    """Reject a preparation replaced since its inputs were loaded for sampling."""
    if record.generation != expected:
        raise ValueError(f"Preparation {manifest} was replaced; samples require generation {expected!r}.")


def save_preparation(
    prepared: RhimePreparedInputs, *, data_config: DataConfig, output_dir: str | Path,
) -> Path:
    """Save prepared inputs and their configuration, publishing the manifest last.

    Args:
        prepared: Borrowed scientific preparation to serialize.
        data_config: Acquisition and preparation choices that produced it.
        output_dir: Caller-selected checkpoint directory; existing files are replaced.

    Returns:
        Absolute path of the completed preparation manifest. Each successful
        save receives a new generation, invalidating samples of an earlier save.

    Raises:
        ValueError: If data selects sectors or concrete write targets redirect
            outside the directory or alias another checkpoint artifact.
    """
    _require_standard(data_config)
    choices = data_config.to_params()
    validate_json_keys(choices)
    record = StandardPreparationRecord(
        schema="openghg_inversions.standard.prepare", schema_version=1,
        producer="openghg_inversions", stage="prepare", generation=uuid4().hex,
        prepared_inputs="prepared-inputs.nc", data=json_value(choices),
    )
    destination = start_checkpoint(output_dir, "prepare", payloads=("prepared-inputs.nc",))
    prepared.save(destination / "prepared-inputs.nc")
    return finish_checkpoint(destination, "prepare", checkpoint_dict(record))


def _load_preparation(
    manifest: str | Path, *, expected_generation: str | None = None,
) -> tuple[RhimePreparedInputs, DataConfig, str, Path]:
    """Load numerical inputs and capture the generation from the same manifest."""
    path, record = read_checkpoint(manifest, StandardPreparationRecord)
    if expected_generation is not None:
        _check_generation(path, record, expected_generation)
    data_config = DataConfig.from_params(record.data)
    payload = reference(path, record.prepared_inputs)
    prepared = RhimePreparedInputs.load(payload)
    return prepared, data_config, record.generation, payload


def load_preparation(manifest: str | Path) -> tuple[RhimePreparedInputs, DataConfig, str]:
    """Load prepared inputs and their recorded acquisition/preparation choices.

    Args:
        manifest: Completed preparation manifest. Relative artifact references
            resolve beside it; absolute shared-cache references are accepted.

    Returns:
        Prepared inputs, their data configuration and the generation to pass
        to :func:`save_samples` after inference.

    Raises:
        ValueError: If the manifest, generation or recorded choices are invalid.
        FileNotFoundError: If the manifest or numerical payload is missing.
    """
    prepared, data_config, generation, _ = _load_preparation(manifest)
    return prepared, data_config, generation


def prepare(
    data_config: DataConfig, *, output_dir: str | Path, merged_data: RhimeMergedData | None = None,
) -> Path:
    """Run shared standard preparation and save its reusable checkpoint.

    Args:
        data_config: Resolved acquisition and preparation choices.
        output_dir: Caller-selected checkpoint directory.
        merged_data: Optional acquired inputs, skipping acquisition.

    Returns:
        Absolute path of the completed preparation manifest.

    Raises:
        ValueError: If the data configuration or supplied inputs conflict.
    """
    basis_inputs = resolve_basis_inputs(data_config)
    bc_basis = (
        load_bc_basis(data_config.bc_basis_paths)
        if data_config.use_bc and data_config.bc_basis_paths is not None else None
    )

    if merged_data is None:
        merged_data = acquire_merged_data(data_config)
    else:
        merged_data.validate_for_preparation(
            **data_config.select("species", "domain", "start_date", "end_date", "split_by_sectors"),
        )

    prepared = prepare_standard(
        merged_data, basis_inputs=basis_inputs, bc_basis=bc_basis,
        **data_config.select(
            "domain", "start_date", "flux_sources", "averaging_error", "filters",
            "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
            "min_error", "min_error_options",
        ),
    )

    if data_config.basis_output_path is not None:
        data_config.basis_output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.basis_functions.save(data_config.basis_output_path)
    return save_preparation(prepared, data_config=data_config, output_dir=output_dir)


def save_samples(
    preparation_manifest: str | Path, posterior: xr.DataTree, *,
    output_contract: OutputContract, model: RhimeModelSpec,
    actual_inference: Mapping[str, Any], output_dir: str | Path, preparation_generation: str,
) -> Path:
    """Save samples bound to one completed preparation generation.

    Args:
        preparation_manifest: Preparation supplying the scientific inputs.
        posterior: Trace with construction-owned ``rhime_priors`` metadata and
            every state required by the output contract. Empty prior records
            are valid when construction omitted unused priors.
        output_contract: Scientific roles of the sampled graph.
        model: Model choices used for sampling.
        actual_inference: Recorded sampler invocation.
        output_dir: Caller-selected sample checkpoint directory.
        preparation_generation: Generation returned by :func:`load_preparation`
            before inference; it must still match at this saving boundary.

    Returns:
        Absolute path of the completed sample manifest.

    Raises:
        ValueError: If the scientific record cannot be persisted, the preparation
            was replaced, or concrete targets redirect or alias dependencies.
    """
    output_contract.validate_trace(posterior)
    preparation_path, preparation = read_checkpoint(preparation_manifest, StandardPreparationRecord)
    _check_generation(preparation_path, preparation, preparation_generation)
    data_config = DataConfig.from_params(preparation.data)
    model_record = model.to_record(trace=posterior, data_config=data_config)
    record = StandardSampleRecord(
        schema="openghg_inversions.standard.sample", schema_version=1,
        producer="openghg_inversions", stage="sample",
        preparation=os.path.relpath(preparation_path, Path(output_dir).resolve()),
        preparation_generation=preparation.generation, posterior="posterior.nc",
        output_contract=output_contract.to_dict(), model=model_record,
        actual_inference=dict(actual_inference), provenance=installed_ogi_provenance(),
    )
    destination = start_checkpoint(
        output_dir, "sample", payloads=("posterior.nc",),
        referenced=(preparation_path, reference(preparation_path, preparation.prepared_inputs)),
    )
    save_trace(posterior, destination / "posterior.nc")
    return finish_checkpoint(destination, "sample", checkpoint_dict(record))


def sample(
    preparation_manifest: str | Path, *, model: Mapping[str, Any],
    sampler: RhimeSampler | Literal["default"] = "default", output_dir: str | Path,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Load preparation, build and sample the shared eager graph, then save samples.

    Args:
        preparation_manifest: Completed scientific preparation.
        model: Current model choices, bound to recorded data facts.
        sampler: Sampler configuration or ``"default"`` for fresh defaults.
        output_dir: Caller-selected sample checkpoint directory.
        likelihood_builder: Optional custom likelihood callable.
        likelihood_kwargs: Options belonging to that callable.

    Returns:
        Absolute path of the completed sample manifest.

    Raises:
        ValueError: If choices conflict, cannot be persisted, or the preparation
            is replaced during inference.
    """
    sampler = resolve_sampler(sampler)
    prepared, data_config, generation, payload = _load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, data_config=data_config)
    model_spec.validate_recordable()
    check_checkpoint_targets(
        Path(output_dir).resolve(), stage="sample", payloads=("posterior.nc",),
        referenced=(Path(preparation_manifest).resolve(), payload),
    )

    model_inputs, priors = materialize_standard_inputs(prepared, model_spec)
    built = build_standard_model(
        model_inputs, model_spec, priors=priors,
        likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
    )

    posterior = sample_rhime_model(built, sampler)
    annotate_sample_trace(built, posterior, likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs)

    return save_samples(
        preparation_manifest, posterior, output_contract=built.output_contract,
        model=model_spec, actual_inference=sampler.describe(),
        output_dir=output_dir,
        preparation_generation=generation,
    )


def load_samples(sample_manifest: str | Path) -> InversionOutput:
    """Decode a completed sample checkpoint into its existing scientific output view.

    Args:
        sample_manifest: Sample record and manifest-relative artifact references.

    Returns:
        Scientific inputs, posterior, model roles and recorded inference metadata,
        without constructing a model or applying output policy.

    Raises:
        ValueError: If the record or scientific trace is invalid, or its
            preparation has been replaced since sampling.
        FileNotFoundError: If a referenced checkpoint is missing.
    """
    path, record = read_checkpoint(sample_manifest, StandardSampleReadRecord)
    preparation_path = reference(path, record.preparation)
    prepared, data_config, _, _ = _load_preparation(
        preparation_path, expected_generation=record.preparation_generation,
    )
    posterior = load_trace(reference(path, record.posterior))
    output_contract = record.output_contract
    output_contract.validate_trace(posterior)
    model_metadata = record.model.to_output_metadata(
        data_config=data_config, trace=posterior, output_contract=output_contract,
    )
    return make_model_output(
        prepared=prepared,
        model_metadata=model_metadata,
        posterior=posterior,
        output_contract=output_contract,
        start_date=data_config.start_date,
        end_date=data_config.end_date,
        actual_inference=record.actual_inference,
        provenance=record.provenance,
        split_by_sectors=data_config.split_by_sectors,
    )


def postprocess(sample_manifest: str | Path, *, output: RhimeOutputSpec) -> InversionOutput:
    """Load the scientific output, write requested products and exports, and return it.

    Args:
        sample_manifest: Completed samples and their preparation reference.
        output: Current output choices, following full-run overwrite behavior.

    Returns:
        The same scientific output view used by full execution.

    Raises:
        ValueError: If checkpoint generation, scientific records or requested
            products are incompatible.
    """
    validate_standard_output_resources(output)
    inv_out = load_samples(sample_manifest)
    postprocess_standard_outputs(inv_out, output)
    save_inversion_output(inv_out, output)
    return inv_out


def prior_predictive(
    preparation_manifest: str | Path, *, model: Mapping[str, Any], output_dir: str | Path,
    draws: int = 100, check_output: str | Path | None = None, stage: str = "prior-predictive",
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared graph and record a finite prior-prediction readiness check.

    Args:
        preparation_manifest: Completed preparation to check.
        model: Current model choices, bound to recorded data facts.
        output_dir: Caller-selected check directory.
        draws: Positive number of prior-predictive draws.
        check_output: Optional report path beneath the check directory.
        stage: Label recorded in the report.
        likelihood_builder: Optional custom likelihood callable.
        likelihood_kwargs: Options belonging to that callable.

    Returns:
        Written readiness report. Model readiness errors become failed checks;
        loading and serialization failures propagate.
    """
    stage = _check_stage(stage)
    _check_draws(draws)
    prepared, data_config, _, payload = _load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, data_config=data_config)
    destination = Path(output_dir).resolve()
    report = _output_path(destination, check_output, "prior-predictive-readiness.json")
    check_checkpoint_targets(
        destination, stage=None,
        payloads=("prior-predictive.nc", str(report.relative_to(destination))),
        referenced=(Path(preparation_manifest).resolve(), payload),
    )
    destination.mkdir(parents=True, exist_ok=True)

    def build() -> Any:
        model_inputs, priors = materialize_standard_inputs(prepared, model_spec)
        return build_standard_model(
            model_inputs, model_spec, priors=priors,
            likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
        )

    return prior_predictive_check(
        build, draws=draws, destination=destination, report=report, stage=stage,
    )
