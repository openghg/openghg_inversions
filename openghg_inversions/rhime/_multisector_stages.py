"""Concrete multisector stages around the canonical scientific recipe.

Read ``prepare`` -> optional ``prior_predictive`` -> ``sample`` -> ``postprocess``.
Prepared inputs are the inference checkpoint; saved output bindings permit
strict graph-free replay under the family schema-3/identity-1 contract.
See :ref:`staged-rhime-lifecycle` for checkpoint ownership and
:ref:`staged-rhime-identity-lifecycle` for requested-to-retained identities."""
from __future__ import annotations
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any
from collections.abc import Mapping
import numpy as np
import pymc as pm
import xarray as xr
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.serialization import load_trace, reset_serialisation_multiindexes, save_trace
from .outputs import RhimeResult, make_multisector_rhime_outputs
from .params import StandardRecipeConfig, resolve_rhime_options
from .preparation import retrieve_or_reload_rhime_data, with_prepared_rhime_sites
from .sampling import sample_rhime_model
from .multisector import construct_multisector_rhime_model, make_multisector_rhime_result, prepare_multisector_rhime_inputs
from . import _stage_configuration as configuration
from ._stage_artifacts import (
    artifact_path as _artifact_path, file_identity as _file_identity,
    write_json as _write_json, _filename_component, _output_path, _stage_output_directory,
)
from ._stage_authentication import (
    _load_output_binding, _load_stage_manifest, _sampler_from_sample_manifest,
    _verify_manifest_artifact, _verify_sample_manifest,
)
from ._stage_checks import PREPARATION_CHECK_NAME, _check_result, _check_stage

RECIPE = "multisector"
SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (3,)
IDENTITY_VERSION = 1
load_params = configuration.load_stage_params


def resolve_config(params: Mapping[str, Any]) -> StandardRecipeConfig:
    """Resolve the multisector recipe's authoritative configuration."""
    return resolve_rhime_options(params=params, multisector=True)


def effective_configuration(setup: StandardRecipeConfig) -> dict[str, Any]:
    """Describe requested or retained choices for this concrete recipe."""
    return configuration.effective_configuration(setup, model=RECIPE)


def configuration_identity(setup: StandardRecipeConfig) -> str:
    """Project this recipe's explicit version-1 scientific identity."""
    return configuration.configuration_identity(setup, model=RECIPE)


def prepare(
    *,
    setup: StandardRecipeConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Acquire and prepare observations, then persist the retained handoff.

    Filtering, site alignment, basis and sensitivities use the canonical recipe.
    Optional merged caching saves acquisition output before filtering at its
    configured cache location. This stage writes prepared inputs and a manifest;
    it does not construct a model. Checkpoint serialization materializes arrays.

    Args:
        setup: Resolved requested science, sampler choices and output policy.
        output_dir: Stage checkpoint directory; requested basis saves go below it.

    Returns:
        Preparation envelope with requested configuration identity, requested
        and retained effective choices, prepared path and content digest.
        ``manifest_path`` is added to the returned mapping after persistence
        and is absent from the saved JSON.

    Raises:
        ValueError: If inputs, scientific options or output paths are invalid.
        OSError: If acquisition, cache or checkpoint I/O fails. Failures propagate;
            the destination and earlier optional artifacts may already exist.
    """
    destination = _stage_output_directory(output_dir)
    data_args = dict(setup.data_args)
    if data_args["basis_output_path"] is not None:
        data_args["basis_output_path"] = str(destination / "basis")
    merged = retrieve_or_reload_rhime_data(data_args, multisector=True)
    prepared = prepare_multisector_rhime_inputs(merged, data_args)
    executed_setup = StandardRecipeConfig(
        run_spec=with_prepared_rhime_sites(setup.run_spec, prepared),
        sampler_options=setup.sampler_options, data_args=data_args,
    )
    prepared_path = _output_path(destination, None, "prepared-inputs.nc")
    prepared.save(prepared_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity_version": IDENTITY_VERSION,
        "recipe": RECIPE,
        "producer": "openghg_inversions",
        "stage": "prepare",
        "configuration_identity": configuration_identity(setup),
        "requested_configuration": effective_configuration(setup),
        "effective_configuration": effective_configuration(executed_setup),
        "artifacts": {
            "prepared_inputs": _artifact_path(prepared_path),
        },
        "artifact_identities": {
            "prepared_inputs": _file_identity(prepared_path),
        },
    }
    manifest_path = _write_json(_output_path(destination, None, "prepare-manifest.json"), manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_prepared(
    path: str | Path,
    *,
    setup: StandardRecipeConfig,
    preparation_manifest: str | Path,
) -> tuple[RhimePreparedInputs, StandardRecipeConfig]:
    """Authenticate requested preparation, then align the run to retained sites.

    The family schema and identity versions, requested configuration identity
    and prepared content digest are checked before loading the handoff. The
    returned configuration keeps requested preparation choices while its
    run sites and averaging periods follow the loaded observations. Sampling
    and replay use this retained-run identity. Loading/authentication errors
    propagate rather than becoming prior-readiness results.
    """
    prepared_path = Path(path).resolve()
    manifest_path, manifest = _load_stage_manifest(
        preparation_manifest, stage="prepare", supported_versions=SUPPORTED_SCHEMA_VERSIONS,
        recipe=RECIPE, identity_version=IDENTITY_VERSION,
    )
    expected = configuration_identity(setup)
    if manifest.get("configuration_identity") != expected:
        raise ValueError(
            "Prepared inputs do not match the effective RHIME configuration: "
            f"manifest has {manifest.get('configuration_identity')!r}, current configuration has {expected!r}."
        )
    _verify_manifest_artifact(
        manifest,
        manifest_path=manifest_path,
        artifact_name="prepared_inputs",
        artifact_path=prepared_path,
    )
    prepared = RhimePreparedInputs.load(prepared_path)
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)
    return prepared, StandardRecipeConfig(run_spec=run_spec, sampler_options=setup.sampler_options, data_args=setup.data_args)


def _build_prepared_model(
    prepared: RhimePreparedInputs,
    setup: StandardRecipeConfig,
):
    return construct_multisector_rhime_model(prepared=prepared, run_spec=setup.run_spec)


def prior_predictive(
    *,
    setup: StandardRecipeConfig,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Assess prior readiness from authenticated fully prepared inputs.

    Construction materializes the model-selected inputs. Loading/authentication
    happens outside the readiness catch. Construction/prediction ``KeyError``
    or ``ValueError`` inside that catch yields a fail check without predictive
    artifacts. Returned empty/non-finite evidence yields fail while preserving
    predictive/report writes. Serialization errors propagate.

    Args:
        setup: Requested configuration matched to the preparation manifest.
        prepared_inputs: Saved fully prepared handoff; preparation is not repeated.
        output_dir: Predictive and readiness-report destination.
        check_output: Optional report path beneath the stage destination.
        preparation_manifest: Envelope authenticating requested science and inputs.
        draws: Positive integer prior-predictive draw count.
        stage: Non-empty stage label included in the check.

    Returns:
        Schema-version-1 preparation-readiness CheckResult with status, measured
        evidence, thresholds and any predictive artifact paths.

    Raises:
        ValueError: If invocation arguments, paths or authentication are invalid.
        OSError: If input loading or artifact persistence fails. External loading
            and uncaught execution errors also propagate; a destination may exist.
    """
    stage = _check_stage(stage)
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("Prior-predictive draws must be a positive integer.")
    destination = _stage_output_directory(output_dir)
    prior_path = _output_path(destination, None, "prior-predictive.nc")
    prepared, resolved = _load_prepared(
        prepared_inputs,
        setup=setup,
        preparation_manifest=preparation_manifest,
    )
    try:
        built = _build_prepared_model(prepared, resolved)
        with built.model:
            prior = pm.sample_prior_predictive(draws, built.model)
        values = [
            np.asarray(group[name].values) for group in prior.children.values() for name in group.data_vars
        ]
        non_finite = sum(int(np.size(value) - np.isfinite(value).sum()) for value in values)
        status = "pass" if any(value.size for value in values) and non_finite == 0 else "fail"
        message = (
            f"Prior predictive produced {draws} finite draws."
            if status == "pass"
            else f"Prior predictive contains {non_finite} non-finite values."
        )
    except (KeyError, ValueError) as exc:
        non_finite = None
        status = "fail"
        message = f"Prior-predictive readiness failed: {type(exc).__name__}: {exc}"
        artifacts = []
    else:
        save_trace(prior, prior_path)
        artifacts = [_artifact_path(prior_path)]
    result = _check_result(
        name=PREPARATION_CHECK_NAME,
        status=status,
        measured_values={"draws": draws, "non_finite_values": non_finite},
        thresholds={"max_non_finite_values": 0},
        message=message,
        artifact_paths=artifacts,
        stage=stage,
    )
    _write_json(_output_path(destination, check_output, "prior-predictive-readiness.json"), result)
    return result


def sample(
    *,
    setup: StandardRecipeConfig,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Construct and sample an authenticated prepared handoff, then bind outputs.

    No acquisition or preparation is repeated. Construction materializes selected
    inputs; sampling uses invocation-local state. The posterior, output binding
    and sample envelope authenticate both numerical artifacts and output meaning.

    Args:
        setup: Requested configuration and sampling choices for this invocation.
        prepared_inputs: Saved fully prepared numerical handoff.
        output_dir: Posterior, binding and sample-manifest destination.
        preparation_manifest: Requested-science and prepared-content authentication.

    Returns:
        Sample envelope with the retained-run configuration identity, recorded
        sampling choices, artifact paths and digests. Its ``manifest_path`` is
        added only to the returned mapping after the JSON has been written.

    Raises:
        ValueError: If authentication, construction or sampling options are invalid.
        OSError: If input or artifact I/O fails. Execution failures propagate;
            the stage destination or earlier artifacts may already exist.
    """
    destination = _stage_output_directory(output_dir)
    prepared, resolved = _load_prepared(
        prepared_inputs,
        setup=setup,
        preparation_manifest=preparation_manifest,
    )
    built = _build_prepared_model(prepared, resolved)
    idata = sample_rhime_model(built, resolved.sampler)
    trace_path = _output_path(destination, None, "posterior.nc")
    save_trace(idata, trace_path)
    identities = {
        "posterior": _file_identity(trace_path),
        "prepared_inputs": _file_identity(Path(prepared_inputs).resolve()),
    }
    binding_path = _write_json(
        _output_path(destination, None, "output-binding.json"),
        {
            "schema": "openghg_inversions.output_binding",
            "schema_version": 1,
            "artifact_identities": identities,
            "output_contract": built.output_contract.to_dict(),
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity_version": IDENTITY_VERSION,
        "recipe": RECIPE,
        "producer": "openghg_inversions",
        "stage": "sample",
        "configuration_identity": configuration_identity(resolved),
        "effective_configuration": effective_configuration(resolved),
        "artifacts": {
            "posterior": _artifact_path(trace_path),
            "prepared_inputs": _artifact_path(Path(prepared_inputs)),
            "output_binding": binding_path.name,
        },
        "artifact_identities": {
            **identities,
            "output_binding": _file_identity(binding_path),
        },
    }
    manifest_path = _write_json(_output_path(destination, None, "sample-manifest.json"), manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def postprocess(
    *,
    setup: StandardRecipeConfig,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
    sample_manifest: str | Path,
) -> RhimeResult:
    """Authenticate saved output meaning and construct products without a graph.

    Requested preparation identity, retained-run sample identity, numerical
    digests and output binding are verified before posterior loading or output
    destination creation. Replay does not materialize model inputs, construct
    a model or resample. It uses recorded sampling provenance and the caller's
    current output policy, with writes contained beneath ``output_dir``.

    Args:
        setup: Requested science and current product policy; sampler changes do
            not replace the provenance recorded by sampling.
        prepared_inputs: Saved fully prepared handoff.
        posterior: Posterior whose content digest matches the sample envelope.
        output_dir: Requested product and postprocess-manifest destination.
        preparation_manifest: Requested preparation and input authentication.
        sample_manifest: Retained-run, posterior and bound-output authentication.

    Returns:
        RhimeResult with requested products and saved output contract, with
        ``model`` and ``model_build_result`` set to None. ``output_metadata``
        includes ``postprocess_manifest_path`` for the written product envelope.

    Raises:
        ValueError: If schemas, identities, bindings, formats or paths are invalid.
        OSError: If input or product I/O fails. Product-construction and persistence
            errors propagate; earlier product writes are not rolled back.
    """
    configured_output = setup.run_spec.output
    _filename_component("output_name", configured_output.output_name)
    _filename_component("species", setup.run_spec.model.species)
    _filename_component("domain", setup.run_spec.model.domain)
    _filename_component("start_date", setup.run_spec.start_date)
    prepared, resolved = _load_prepared(
        prepared_inputs,
        setup=setup,
        preparation_manifest=preparation_manifest,
    )
    posterior_path = Path(posterior).resolve()
    sample_contract = _verify_sample_manifest(
        sample_manifest,
        posterior=posterior_path,
        configuration_identity=configuration_identity(resolved),
        prepared_inputs=prepared_inputs,
        supported_versions=SUPPORTED_SCHEMA_VERSIONS, recipe=RECIPE, identity_version=IDENTITY_VERSION,
    )
    sampled_sampler = _sampler_from_sample_manifest(sample_contract, path=sample_manifest)
    output_contract = _load_output_binding(sample_contract, manifest_path=Path(sample_manifest).resolve())
    output_contract.validate_requested_output(configured_output.output_format)
    destination = _stage_output_directory(output_dir)
    output_spec = replace(
        configured_output,
        output_path=str(destination),
        save_trace=False,
        save_inversion_output=bool(configured_output.save_inversion_output),
    )
    run_spec = replace(resolved.run_spec, output=output_spec)
    resolved = StandardRecipeConfig(run_spec=run_spec, sampler=sampled_sampler, data_args=resolved.data_args)
    idata = load_trace(posterior_path)
    result = make_multisector_rhime_result(
        prepared=prepared, run_spec=run_spec, sampler=resolved.sampler,
        model_build_result=None, output_contract=output_contract, idata=idata,
        build_and_sample_seconds=0.0,
    )
    make_multisector_rhime_outputs(result=result, prepared=prepared)
    artifacts = {
        name: path
        for name, path in result.output_metadata.items()
        if name.endswith("_path") and isinstance(path, str)
    }
    basic = result.outputs.get("basic")
    if isinstance(basic, xr.Dataset):
        basic_path = _output_path(destination, None, "basic.nc")
        reset_serialisation_multiindexes(basic).to_netcdf(basic_path)
        artifacts["basic_path"] = str(basic_path)
    for name, path in artifacts.items():
        try:
            Path(path).resolve().relative_to(destination)
        except ValueError:
            raise ValueError(
                f"Postprocessing artifact {name!r} at {path!r} escaped stage output directory {destination}."
            ) from None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "identity_version": IDENTITY_VERSION,
        "recipe": RECIPE,
        "producer": "openghg_inversions",
        "stage": "postprocess",
        "configuration_identity": configuration_identity(resolved),
        "effective_configuration": effective_configuration(resolved),
        "input_identities": dict(sample_contract["artifact_identities"]),
        "artifacts": {name: _artifact_path(Path(path)) for name, path in artifacts.items()},
    }
    manifest_path = _write_json(_output_path(destination, None, "postprocess-manifest.json"), manifest)
    result.output_metadata["postprocess_manifest_path"] = str(manifest_path)
    return result
