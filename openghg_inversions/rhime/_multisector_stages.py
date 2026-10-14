"""Deferred multisector file-backed RHIME stages (pre-handoff envelope).

Standard execution uses the handoffs in :mod:`._standard_stages`. These
functions remain until multisector adopts the same handoffs.
"""

from __future__ import annotations

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.basis import prepare_basis
from openghg_inversions.rhime.preparation import acquire_merged_data, export_basis, resolve_basis_inputs

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping


from openghg_inversions._timing import timed
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.inference.sampling import resolve_sampler
from .specs import RhimeRunSpec
from openghg_inversions.serialization import (
    load_trace,
    save_trace,
)

from .builders import RhimeModelBuildResult
from .multisector import (
    build_multisector_rhime_model_result,
    make_multisector_rhime_result,
    multisector_model_input_names,
)
from .outputs import (
    RhimeResult,
    make_multisector_rhime_outputs,
    validate_output_resources,
)
from .params import RhimeConfig
from .preparation import (
    assemble_rhime_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
)
from .sampling import sample_rhime_model
from .materialization import materialize_pymc_inputs

from ._stage_artifacts import (
    artifact_path as _artifact_path,
    json_value as _json_value,
    write_json as _write_json,
    _output_path,
    _stage_output_directory,
)
from ._stage_authentication import (
    _load_output_binding,
    _load_stage_manifest,
    _sampler_metadata_from_sample_manifest,
    _load_sample_manifest,
)
from ._stage_checks import _check_draws, _check_stage, prior_predictive_check


ModelKind = Literal["multisector"]

_PREPARATION_IDENTITY_EXCLUDED_OPTIONS = frozenset(
    {
        "basis_output_path",
        "output_name",
    }
)


def effective_configuration(setup: RhimeConfig, *, model: ModelKind) -> dict[str, Any]:
    """Project resolved choices into this workflow's manifest vocabulary.

    This is a staged artifact representation, not a configuration export API.
    No pre-preparation run description or raw shorthand is reconstructed.
    """
    return {
        "model": model,
        "model_spec": asdict(setup.model),
        "output": asdict(setup.output),
        "sampler": {name: getattr(setup.sampler, name) for name in setup.sampler.__slots__},
        "preparation": setup.data.to_params(),
    }


def configuration_identity(setup: RhimeConfig, *, model: ModelKind) -> str:
    """Hash resolved preparation and scientific model choices.

    The resolved encoding supersedes historical raw/default projections;
    existing artifacts with a different configuration identity require preparation.
    """
    identity_configuration = {
        "model": model,
        "preparation": {
            name: value for name, value in setup.data.to_params().items()
            if name not in _PREPARATION_IDENTITY_EXCLUDED_OPTIONS
        },
        "model_spec": asdict(setup.model),
    }
    encoded = json.dumps(
        _json_value(identity_configuration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def prepare_rhime_stage(
    *,
    setup: RhimeConfig,
    model: ModelKind,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare and persist independently inspectable RHIME inputs.

    Selected basis and support files are resolved before acquisition. Reporting
    resources are checked only when outputs are requested at postprocessing.
    ``setup.data.basis_output_path`` remains an exact caller-selected destination;
    it is not relocated below ``output_dir``. The stage writes prepared inputs
    and their manifest below ``output_dir`` and returns the manifest mapping.
    """
    basis_inputs = resolve_basis_inputs(setup.data)
    bc_basis = load_bc_basis(setup.data.bc_basis_paths) if setup.data.use_bc and setup.data.bc_basis_paths is not None else None
    destination = _stage_output_directory(output_dir)

    merged = acquire_merged_data(setup.data)
    with_errors = prepare_observation_errors(merged.site_data, averaging_error=setup.data.averaging_error)
    filtered, site_options = filter_observations(with_errors, merged.site_options, filters=setup.data.filters)
    runtime_flux = resolve_runtime_flux(merged.flux_data, split_by_sectors=merged.split_by_sectors)
    with timed("rhime.prepare_inputs.basis_build", basis_algorithm=setup.data.basis_algorithm):
        basis = prepare_basis(
            site_data=filtered,
            runtime_flux=runtime_flux,
            weighting_flux=merged.flux_data[setup.data.flux_sources[0]]["flux"],
            **setup.data.select(
                "domain", "start_date", "basis_algorithm", "nbasis",
            ),
            fix_outer_regions=setup.data.fix_basis_outer_regions,
            **basis_inputs,
        )
    basis = export_basis(basis, setup.data.basis_output_path)
    sensitivities = build_sensitivities(
        filtered,
        basis,
        **setup.data.select(
            "flux_sources", "use_bc",
        ),
        bc_basis=bc_basis,
        multisector=True,
    )
    prepared = assemble_rhime_inputs(
        sensitivities,
        basis,
        site_options=site_options,
        **setup.data.select(
            "domain", "start_date", "bc_freq", "min_error",
            "min_error_options", "use_bc",
        ),
    )
    # The deferred multisector stages require every requested site to survive preparation.
    prepared_sites = {str(site).upper() for site in prepared.sites}
    missing_sites = [site for site in setup.data.site_options.sites if str(site).upper() not in prepared_sites]
    if missing_sites:
        raise ValueError(
            "RHIME preparation could not produce required site input(s) "
            f"{missing_sites!r} for species {setup.data.species!r} and period "
            f"{setup.data.start_date} to {setup.data.end_date}."
        )
    prepared_path = _output_path(destination, None, "prepared-inputs.nc")
    prepared.save(prepared_path)
    manifest = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "prepare",
        "configuration_identity": configuration_identity(setup, model=model),
        "requested_configuration": effective_configuration(setup, model=model),
        "effective_configuration": effective_configuration(setup, model=model),
        "artifacts": {
            "prepared_inputs": _artifact_path(prepared_path),
        },
    }
    manifest_path = _write_json(_output_path(destination, None, "prepare-manifest.json"), manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_prepared(
    path: str | Path,
    *,
    setup: RhimeConfig,
    model: ModelKind,
    preparation_manifest: str | Path,
) -> tuple[RhimePreparedInputs, RhimeRunSpec]:
    prepared_path = Path(path).resolve()
    _, manifest = _load_stage_manifest(preparation_manifest, stage="prepare")
    expected = configuration_identity(setup, model=model)
    if manifest.get("configuration_identity") != expected:
        raise ValueError(
            "Prepared inputs do not match the effective RHIME configuration: "
            f"manifest has {manifest.get('configuration_identity')!r}, current configuration has {expected!r}."
        )
    prepared = RhimePreparedInputs.load(prepared_path)
    return prepared, setup.retained_run_spec(prepared)


def _build_prepared_model(prepared: RhimePreparedInputs, run_spec: RhimeRunSpec) -> RhimeModelBuildResult:
    """Materialize the multisector inputs and build its graph."""
    inputs = materialize_pymc_inputs(prepared, variable_names=multisector_model_input_names(prepared, run_spec.model))
    return build_multisector_rhime_model_result(prepared=prepared, model_inputs=inputs, run_spec=run_spec)


def prior_predictive_stage(
    *,
    setup: RhimeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Build the configured graph and check finite prior-predictive draws."""
    stage = _check_stage(stage)
    _check_draws(draws)
    destination = _stage_output_directory(output_dir)
    prepared, run_spec = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    return prior_predictive_check(
        lambda: _build_prepared_model(prepared, run_spec),
        draws=draws, destination=destination, check_output=check_output, stage=stage,
    )


def sample_rhime_stage(
    *,
    setup: RhimeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Sample prepared inputs and persist posterior and output-role records.

    Writes the trace, a versioned output contract and a schema-version-2
    sample manifest. Preparation is never invoked implicitly.
    """
    setup = replace(setup, sampler=resolve_sampler(setup.sampler))
    destination = _stage_output_directory(output_dir)
    prepared, run_spec = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    built = _build_prepared_model(prepared, run_spec)
    idata = sample_rhime_model(built, setup.sampler)
    trace_path = _output_path(destination, None, "posterior.nc")
    save_trace(idata, trace_path)
    binding_path = _write_json(
        _output_path(destination, None, "output-binding.json"),
        {
            "schema": "openghg_inversions.output_binding",
            "schema_version": 1,
            "output_contract": built.output_contract.to_dict(),
        },
    )
    manifest = {
        "schema_version": 2,
        "producer": "openghg_inversions",
        "stage": "sample",
        "configuration_identity": configuration_identity(setup, model=model),
        "effective_configuration": effective_configuration(setup, model=model),
        "artifacts": {
            "posterior": _artifact_path(trace_path),
            "prepared_inputs": _artifact_path(Path(prepared_inputs)),
            "output_binding": binding_path.name,
        },
    }
    manifest_path = _write_json(_output_path(destination, None, "sample-manifest.json"), manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def postprocess_rhime_stage(
    *,
    setup: RhimeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
    sample_manifest: str | Path,
) -> RhimeResult:
    """Build requested products from matched prepared, posterior and role artifacts.

    Version-2 sample manifests use their saved output bindings, without
    materializing model inputs or constructing a graph. Genuine version-1
    manifests retain graph reconstruction to recover their missing roles.
    Output settings may change; the sampled scientific configuration may not.
    """
    validate_output_resources(setup.output)
    destination = _stage_output_directory(output_dir)
    configured_output = setup.output
    prepared, run_spec = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    posterior_path = Path(posterior).resolve()
    sample_contract = _load_sample_manifest(
        sample_manifest,
        configuration_identity=configuration_identity(setup, model=model),
    )
    sampler_metadata = _sampler_metadata_from_sample_manifest(sample_contract)
    output_spec = replace(
        configured_output,
        output_path=str(destination),
        save_trace=False,
        save_inversion_output=bool(configured_output.save_inversion_output),
    )
    run_spec = replace(run_spec, output=output_spec)
    resolved = replace(setup, output=output_spec)
    if sample_contract["schema_version"] == 1:
        # Older manifests did not persist roles; preserve their explicit graph replay.
        built = _build_prepared_model(prepared, run_spec)
        output_contract = built.output_contract
    else:
        built = None
        output_contract = _load_output_binding(sample_contract, manifest_path=Path(sample_manifest).resolve())
    output_contract.validate_requested_output(output_spec.output_format)
    idata = load_trace(posterior_path)
    result = make_multisector_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=None,
        model_build_result=built,
        output_contract=output_contract,
        idata=idata,
        build_and_sample_seconds=0.0,
    )
    result.output_metadata["sampler"] = sampler_metadata
    make_multisector_rhime_outputs(result=result, prepared=prepared)
    artifacts = {
        name: path
        for name, path in result.output_metadata.items()
        if name.endswith("_path") and isinstance(path, str)
    }
    for name, path in artifacts.items():
        try:
            Path(path).resolve().relative_to(destination)
        except ValueError:
            raise ValueError(
                f"Postprocessing artifact {name!r} at {path!r} escaped stage output directory {destination}."
            ) from None
    effective = effective_configuration(resolved, model=model)
    recorded = sample_contract.get("effective_configuration")
    effective["sampler"] = recorded.get("sampler") if isinstance(recorded, Mapping) else None
    manifest = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "postprocess",
        "configuration_identity": configuration_identity(resolved, model=model),
        "effective_configuration": effective,
        "artifacts": {name: _artifact_path(Path(path)) for name, path in artifacts.items()},
    }
    manifest_path = _write_json(_output_path(destination, None, "postprocess-manifest.json"), manifest)
    result.output_metadata["postprocess_manifest_path"] = str(manifest_path)
    return result
