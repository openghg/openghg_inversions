"""Standard RHIME handoffs: save and reload what the full runner passes between steps.

``prepare``, ``sample``, ``postprocess`` and ``prior_predictive`` call the same
step functions as :func:`~openghg_inversions.rhime.run_standard`; they only add
loading a checkpoint before and saving one after.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import xarray as xr

from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions._timing import timer_seconds, timer_start
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.serialization import load_trace, save_trace
from openghg_inversions.models.priors import trace_prior_records

from ._stage_artifacts import _output_path, write_json as _write_json
from ._stage_checks import _check_draws, _check_stage, prior_predictive_check
from .builders import RhimeLikelihoodBuilder, callable_metadata
from .outputs import (
    RhimeResult,
    annotate_likelihood_trace,
    make_standard_rhime_outputs,
)
from .params import DataConfig, resolve_model_options, retained_run_spec
from .preparation import acquire_merged_data, export_basis, resolve_basis_inputs
from .sampling import sample_rhime_model
from .specs import RhimeModelSpec, RhimeOutputSpec


def _read_record(path: str | Path, *, stage: str) -> tuple[Path, dict[str, Any]]:
    """Read the current standard format once, without ambient path discovery."""
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = f"openghg_inversions.standard.{stage}"
    if not isinstance(record, dict) or (
        record.get("schema") != expected
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("family") != "standard"
    ):
        found = {name: record.get(name) for name in ("schema", "schema_version", "family")} if isinstance(record, dict) else type(record).__name__
        raise ValueError(f"{path} is not a {expected} version-1 manifest for the standard family; found {found!r}.")
    required = (
        ("prepared_inputs", "data") if stage == "prepare"
        else ("preparation", "posterior", "output_contract", "model", "build_and_sample_seconds")
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"Standard {stage} manifest {path} is missing {missing!r}.")
    if stage == "sample" and record.get("inference_kind") != "mcmc":
        raise ValueError(
            f"Standard sample manifest requires inference_kind='mcmc'; found {record.get('inference_kind')!r}."
        )
    return path, record


def _reference(path: Path, name: str, value: Any) -> Path:
    """Validate one manifest artifact reference and resolve it beside the manifest."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest {path} requires a nonempty `{name}` reference; found {value!r}.")
    return (path.parent / value).resolve()


def _save_record(destination: Path, *, stage: str, **contents: Any) -> Path:
    """Write a completed manifest last, so its presence marks a finished stage."""
    temporary = _write_json(destination / f".{stage}-manifest.tmp", {
        "schema": f"openghg_inversions.standard.{stage}", "schema_version": 1,
        "producer": "openghg_inversions", "family": "standard", "stage": stage,
        **contents,
    })
    return temporary.replace(destination / f"{stage}-manifest.json")


def _require_standard(data: DataConfig) -> None:
    """Reject sector-split data choices at the standard handoff boundary."""
    if data.split_by_sectors:
        raise ValueError("Standard handoffs require `split_by_sectors=False`; use the multisector stages.")


def _check_checkpoint_targets(
    destination: Path, *, stage: str, payload: str, referenced: tuple[Path, ...] = (),
) -> None:
    """Check only concrete writes for containment and aliases to dependencies."""
    protected = list(referenced)
    for name in (payload, f".{stage}-manifest.tmp", f"{stage}-manifest.json"):
        target = _output_path(destination, None, name)
        if any(target == path or (target.exists() and path.exists() and target.samefile(path))
               for path in protected):
            raise ValueError(f"Checkpoint target {target} would overwrite another checkpoint artifact.")
        protected.append(target)


def save_preparation(prepared: RhimePreparedInputs, *, data: DataConfig, output_dir: str | Path) -> Path:
    """Save prepared inputs and the data choices that produced them.

    Several samples may reuse one saved preparation. Saving materializes numerical
    payloads through their codec without mutating the borrowed inputs; the
    manifest is written last and marks completion.

    Args:
        prepared: Prepared inputs, e.g. from :func:`prepare_standard`.
        data: Data choices that produced ``prepared``. They are recorded in their
            external option form and reloaded by :func:`load_preparation`.
        output_dir: Directory for the preparation.

    Returns:
        Absolute path of the completed preparation manifest.

    Raises:
        ValueError: If ``data`` selects sector-split preparation or a checkpoint
            write target escapes its directory or aliases another checkpoint file.
    """
    _require_standard(data)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return _write_preparation(prepared, data=data, destination=destination)


def _write_preparation(prepared: RhimePreparedInputs, *, data: DataConfig, destination: Path) -> Path:
    """Write prepared inputs, then the manifest that marks the preparation complete."""
    _check_checkpoint_targets(destination, stage="prepare", payload="prepared-inputs.nc")
    (destination / "prepare-manifest.json").unlink(missing_ok=True)
    prepared.save(destination / "prepared-inputs.nc")
    return _save_record(
        destination, stage="prepare", prepared_inputs="prepared-inputs.nc",
        data=data.to_params(), basis_artifact_source=prepared.basis_artifact_source,
    )


def load_preparation(manifest: str | Path) -> tuple[RhimePreparedInputs, DataConfig]:
    """Load saved prepared inputs and the data choices that produced them.

    Args:
        manifest: Preparation manifest. Relative references resolve beside it;
            absolute shared-cache references are used directly.

    Returns:
        The prepared inputs and the recorded :class:`DataConfig`.

    Raises:
        ValueError: If the manifest is not a standard preparation manifest or its
            recorded data choices are invalid.
        FileNotFoundError: If the referenced numerical file is missing.
    """
    path, record = _read_record(manifest, stage="prepare")
    data = DataConfig.from_params(record["data"])
    _require_standard(data)
    return RhimePreparedInputs.load(_reference(path, "prepared_inputs", record["prepared_inputs"])), data


def prepare(data: DataConfig, *, output_dir: str | Path, merged_data: RhimeMergedData | None = None) -> Path:
    """Acquire and prepare standard inputs once, then save a reusable preparation.

    This runs the same steps as :func:`~openghg_inversions.rhime.run_standard` up
    to the basis export. Model, sampler and output choices are not needed.

    Args:
        data: Resolved data choices.
        output_dir: Directory for the preparation.
        merged_data: Optional supplied acquisition handoff; skips acquisition.

    Returns:
        Absolute path of the completed preparation manifest.

    Raises:
        ValueError: If ``data`` selects sector-split preparation or supplied data
            conflicts with it.
    """
    from .standard import prepare_standard

    _require_standard(data)
    basis_inputs = resolve_basis_inputs(data)
    bc_basis = load_bc_basis(data.bc_basis_paths) if data.use_bc and data.bc_basis_paths is not None else None
    merged = acquire_merged_data(data, merged_data)
    prepared = prepare_standard(
        merged, basis_inputs=basis_inputs, bc_basis=bc_basis,
        **data.select(
            "domain", "start_date", "flux_sources", "averaging_error", "filters",
            "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
            "min_error", "min_error_options",
        ),
    )
    # Create the destination only now, so a failed acquisition or preparation can be retried.
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if data.basis_output_path is not None:
        prepared = replace(prepared, basis_functions=export_basis(prepared.basis_functions, data.basis_output_path))
    return _write_preparation(prepared, data=data, destination=destination)


def save_samples(
    preparation_manifest: str | Path, posterior: xr.DataTree, *,
    output_contract: OutputContract, model: RhimeModelSpec,
    actual_inference: Mapping[str, Any], build_and_sample_seconds: float,
    output_dir: str | Path,
) -> Path:
    """Save samples and their scientific meaning beside a saved preparation.

    Several sample records may reference the same preparation. After preflight
    checks succeed, the old completion marker is removed before payload writes.
    The new manifest is written last, so a payload-write failure leaves no
    completed sample manifest; preflight failures preserve an existing record.

    Args:
        preparation_manifest: Preparation the samples were drawn from.
        posterior: Sampled trace. It must hold every state ``output_contract``
            requires.
        output_contract: Output roles of the sampled model.
        model: Resolved model choices; see :meth:`RhimeModelSpec.to_record`.
        actual_inference: Description of the sampler invocation that ran.
        build_and_sample_seconds: Duration of model construction and sampling.
        output_dir: Directory for the samples.

    Returns:
        Absolute path of the completed sample manifest.

    Raises:
        ValueError: If the posterior lacks a required state or the model choices
            cannot be recorded, or a checkpoint write target escapes its directory
            or aliases another checkpoint file or dependency. These checks precede writes.
    """
    output_contract.validate_trace(posterior)
    if (model.state_activity is not None or model.bc_state_activity is not None
            or any(sector.state_activity is not None for sector in model.sectors)):
        raise ValueError(
            "Saved sample records cannot represent `state_activity`, `bc_state_activity` "
            "or `SectorSpec.state_activity`; run these models in memory."
        )
    model_record = model.to_record(priors=trace_prior_records(posterior))
    preparation_path, preparation = _read_record(preparation_manifest, stage="prepare")
    destination = Path(output_dir).resolve()
    _check_checkpoint_targets(
        destination, stage="sample", payload="posterior.nc",
        referenced=(preparation_path, _reference(preparation_path, "prepared_inputs", preparation["prepared_inputs"])),
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sample-manifest.json").unlink(missing_ok=True)
    save_trace(posterior, destination / "posterior.nc")
    return _save_record(
        destination, stage="sample", preparation=str(preparation_path),
        posterior="posterior.nc", output_contract=output_contract.to_dict(),
        model=model_record, inference_kind="mcmc", actual_inference=dict(actual_inference),
        build_and_sample_seconds=build_and_sample_seconds, provenance=installed_ogi_provenance(),
    )


def _saved_model(
    preparation_manifest: str | Path, model: Mapping[str, Any],
) -> tuple[RhimePreparedInputs, RhimeModelSpec]:
    """Load a preparation and bind current model choices to its data choices."""
    prepared, data = load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, data=data)
    return prepared, model_spec


def sample(
    preparation_manifest: str | Path, *, model: Mapping[str, Any],
    sampler: RhimeSampler | Literal["default"] = "default", output_dir: str | Path,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Sample a saved preparation with current model and sampler choices.

    Builds through the same :func:`~openghg_inversions.rhime.standard.build_standard`
    as the full runner.

    Args:
        preparation_manifest: Saved preparation to sample.
        model: Canonical model choices, bound to the saved data choices.
        sampler: Sampler to copy for this invocation; ``"default"`` selects a
            fresh default sampler and ``None`` is rejected.
        output_dir: Directory for the samples.
        likelihood_builder: Optional Python custom likelihood.
        likelihood_kwargs: Options for ``likelihood_builder``.

    Returns:
        Absolute path of the completed sample manifest.

    Raises:
        ValueError: If model choices conflict with the preparation, no likelihood
            is selected, or the posterior lacks a required state.
    """
    from .standard import build_standard

    sampler = resolve_sampler(sampler)
    prepared, model_spec = _saved_model(preparation_manifest, model)
    started = timer_start()
    built = build_standard(
        prepared, model_spec, likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
    )
    posterior = sample_rhime_model(built, sampler)
    if likelihood_builder is not None:
        annotate_likelihood_trace(
            posterior, builder_identity=callable_metadata(likelihood_builder), likelihood_kwargs=likelihood_kwargs,
        )
    return save_samples(
        preparation_manifest, posterior, output_contract=built.output_contract,
        model=model_spec, actual_inference=sampler.describe(),
        build_and_sample_seconds=timer_seconds(started), output_dir=output_dir,
    )


def postprocess(sample_manifest: str | Path, *, output: RhimeOutputSpec) -> RhimeResult:
    """Make products from saved samples with current output choices, without a graph.

    The recorded model and inference describe the samples; ``output`` alone
    controls products and destinations. Keep the referenced preparation and
    posterior files available for each replay.

    Args:
        sample_manifest: Saved samples.
        output: Output choices, with the full runner's overwrite behavior.

    Returns:
        The replayed result with ``sampler=None`` and ``model=None``.

    Raises:
        ValueError: If the manifest is invalid, its model conflicts with its
            preparation, or the output is unsupported by the saved roles.
    """
    from .standard import make_standard_rhime_result

    path, record = _read_record(sample_manifest, stage="sample")
    output_contract = OutputContract.from_dict(record["output_contract"])
    output_contract.validate_requested_output(output.output_format)
    prepared, data = load_preparation(_reference(path, "preparation", record["preparation"]))
    posterior = load_trace(_reference(path, "posterior", record["posterior"]))
    output_contract.validate_trace(posterior)
    model = RhimeModelSpec.from_record(record["model"], trace=posterior)
    # The manifests are separate files; keep the sample bound to its preparation.
    conflicts = {name: (getattr(model, name), getattr(data, name))
                 for name in ("species", "domain") if getattr(model, name) != getattr(data, name)}
    if model.use_bc and not data.use_bc:
        conflicts["use_bc"] = (True, False)
    sources = tuple(sector.flux_source for sector in model.sectors)
    if sources != data.flux_sources:
        conflicts["flux_sources"] = (sources, data.flux_sources)
    if conflicts:
        raise ValueError(f"Sample model conflicts with its preparation as (sample, preparation): {conflicts!r}.")
    result = make_standard_rhime_result(
        prepared=prepared, run_spec=retained_run_spec(data, prepared, model=model, output=output), sampler=None,
        output_contract=output_contract, idata=posterior,
        build_and_sample_seconds=record["build_and_sample_seconds"],
        sampler_metadata=record.get("actual_inference"), provenance=record.get("provenance", {}),
    )
    return make_standard_rhime_outputs(result=result, prepared=prepared)


def prior_predictive(
    preparation_manifest: str | Path, *, model: Mapping[str, Any], output_dir: str | Path,
    draws: int = 100, check_output: str | Path | None = None, stage: str = "prior-predictive",
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check finite prior prediction from a saved preparation and current model choices.

    Never runs posterior inference. Model readiness failures produce a failed
    check; input loading and serialization errors propagate.

    Args:
        preparation_manifest: Saved preparation to check.
        model: Canonical model choices, bound to the saved data choices.
        output_dir: Directory for the evidence and report.
        draws: Positive number of prior-predictive draws.
        check_output: Optional report path beneath ``output_dir``.
        stage: Stage label recorded in the report.
        likelihood_builder: Optional Python custom likelihood.
        likelihood_kwargs: Options for ``likelihood_builder``.

    Returns:
        The readiness check written to the report.

    Raises:
        ValueError: If ``draws`` or ``stage`` is invalid or model choices conflict
            with the preparation.
    """
    from .standard import build_standard

    stage = _check_stage(stage)
    _check_draws(draws)
    prepared, model_spec = _saved_model(preparation_manifest, model)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return prior_predictive_check(
        lambda: build_standard(
            prepared, model_spec, likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
        ),
        draws=draws, destination=destination, check_output=check_output, stage=stage,
    )
