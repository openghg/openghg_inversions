"""Concrete standard and multisector file-backed RHIME workflows.

These functions compose the existing RHIME scientific stages.  They do not
define a second configuration schema and have no dependency on an
orchestrator or scheduler.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast
from collections.abc import Mapping

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions._timing import timed
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from .specs import RhimeRunSpec
from openghg_inversions.serialization import (
    load_trace,
    reset_serialisation_multiindexes,
    save_trace,
)

from .multisector import (
    build_multisector_rhime_model_result,
    make_multisector_rhime_result,
    multisector_model_input_names,
)
from .outputs import RhimeResult, make_multisector_rhime_outputs, make_standard_rhime_outputs
from .ini import read_rhime_ini
from .params import RHIME_PREPARATION_OPTION_NAMES, RhimeConfig
from .preparation import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
)
from .sampling import sample_rhime_model
from .standard import (
    build_standard_rhime_model_result,
    make_standard_rhime_result,
    standard_model_input_names,
)
from .materialization import materialize_pymc_inputs

from ._stage_artifacts import (
    artifact_path as _artifact_path,
    file_identity as _file_identity,
    json_value as _json_value,
    write_json as _write_json,
    _filename_component,
    _output_path,
    _stage_output_directory,
)
from ._stage_authentication import (
    _load_output_binding,
    _load_stage_manifest,
    _sampler_from_sample_manifest,
    _verify_manifest_artifact,
    _verify_sample_manifest,
)
from ._stage_checks import PREPARATION_CHECK_NAME, _check_result, _check_stage

ModelKind = Literal["standard", "multisector"]

_STAGE_PATH_OPTIONS = frozenset(
    {
        "basis_directory",
        "bc_basis_directory",
        "country_directory",
        "country_file",
    }
)
_PREPARATION_IDENTITY_EXCLUDED_OPTIONS = frozenset(
    {
        "basis_output_path",
        "output_name",
    }
)


def _resolve_stage_paths(params: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Resolve filesystem-valued staged options relative to their config file."""
    resolved = dict(params)
    for name in _STAGE_PATH_OPTIONS:
        value = resolved.get(name)
        if not isinstance(value, str | Path) or not value:
            continue
        path = Path(value).expanduser()
        resolved[name] = str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())
    return resolved


def load_stage_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load existing RHIME parameters from one explicit source.

    ``CONFIG_FILE`` is intentionally not an environment default: an unrelated
    ambient variable must never select the scientific configuration.
    """
    if (config_file is None) == (params_file is None):
        raise ValueError("Pass exactly one of `config_file` or `params_file`.")
    if config_file is not None:
        source_path = Path(config_file).resolve()
        params = read_rhime_ini(source_path)
    else:
        source_path = Path(cast(str | Path, params_file)).resolve()
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"RHIME params file {source_path} must contain one JSON object.")
        params = loaded
    if overrides:
        params.update(overrides)
    return _resolve_stage_paths(params, base_dir=source_path.parent)


def effective_configuration(setup: RhimeConfig, *, model: ModelKind) -> dict[str, Any]:
    """Project resolved choices into this workflow's manifest vocabulary.

    This is a staged artifact representation, not a configuration export API.
    No pre-preparation run description or raw shorthand is reconstructed.
    """
    site_names = {field.name for field in fields(setup.site_options)}
    preparation = {
        name: getattr(setup, name)
        for name in RHIME_PREPARATION_OPTION_NAMES if name not in site_names
    }
    preparation.update({name: getattr(setup.site_options, name) for name in site_names})
    return {
        "model": model,
        "model_spec": asdict(setup.model),
        "output": asdict(setup.output),
        "sampler": {name: getattr(setup.sampler, name) for name in setup.sampler.__slots__},
        "preparation": preparation,
    }


def configuration_identity(setup: RhimeConfig, *, model: ModelKind) -> str:
    """Hash resolved preparation and scientific model choices.

    The resolved encoding supersedes historical raw/default projections;
    existing artifacts with a different configuration identity require preparation.
    """
    site_names = {field.name for field in fields(setup.site_options)}
    preparation = setup.select(*(RHIME_PREPARATION_OPTION_NAMES - site_names))
    preparation.update({name: getattr(setup.site_options, name) for name in site_names})
    identity_configuration = {
        "model": model,
        "preparation": {
            name: value for name, value in preparation.items()
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
    """Prepare and persist independently inspectable RHIME inputs."""
    destination = _stage_output_directory(output_dir)
    multisector = model == "multisector"
    executed_setup = replace(
        setup,
        basis_output_path=str(destination / "basis") if setup.basis_output_path is not None else None,
    )
    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(executed_setup.site_options.sites),
        split_by_sectors=executed_setup.split_by_sectors,
    ):
        merged = RhimeMergedData.from_options(
            **executed_setup.select(
                "site_options", "species", "domain", "start_date",
                "end_date",  "flux_sources", "split_by_sectors",
                "bc_store", "obs_store", "footprint_store", "emissions_store",
                "emissions_domain", "fp_model", "fp_species", "calibration_scale",
                "use_bc", "bc_input", "averaging_error",
                "flux_non_finite_check",
            ),
        )
    filtered = filter_rhime_observations(merged, filters=executed_setup.filters)
    retained_sites = {str(site).upper() for site in filtered.sites}
    missing_sites = [site for site in executed_setup.site_options.sites if str(site).upper() not in retained_sites]
    if missing_sites:
        raise ValueError(
            "RHIME preparation could not produce required site input(s) "
            f"{missing_sites!r} for species {executed_setup.species!r} and period "
            f"{executed_setup.start_date} to {executed_setup.end_date}."
        )
    basis = build_rhime_basis(
        filtered,
        **executed_setup.select(
            "species", "domain", "start_date", "flux_sources",
            "output_name", "basis_algorithm", "nbasis", "fp_basis_case",
            "basis_directory", "country_directory", "outer_regions_path",
            "fix_basis_outer_regions", "basis_output_path",
        ),
    )
    site_data = build_rhime_sensitivities(
        filtered,
        basis,
        **executed_setup.select(
            "domain", "flux_sources", "use_bc", "bc_basis_case",
            "bc_basis_directory",
        ),
        multisector=multisector,
    )
    prepared = assemble_rhime_inputs(
        filtered,
        basis,
        site_data,
        **executed_setup.select(
            "domain", "start_date", "bc_freq", "min_error",
            "min_error_options", "use_bc",
        ),
    )
    prepared_sites = {str(site).upper() for site in prepared.sites}
    missing_sites = [site for site in setup.site_options.sites if str(site).upper() not in prepared_sites]
    if missing_sites:
        raise ValueError(
            "RHIME preparation could not produce required site input(s) "
            f"{missing_sites!r} for species {setup.species!r} and period "
            f"{setup.start_date} to {setup.end_date}."
        )
    prepared_path = _output_path(destination, None, "prepared-inputs.nc")
    prepared.save(prepared_path)
    manifest = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "prepare",
        "configuration_identity": configuration_identity(executed_setup, model=model),
        "requested_configuration": effective_configuration(setup, model=model),
        "effective_configuration": effective_configuration(executed_setup, model=model),
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
    setup: RhimeConfig,
    model: ModelKind,
    preparation_manifest: str | Path,
) -> tuple[RhimePreparedInputs, RhimeRunSpec]:
    prepared_path = Path(path).resolve()
    manifest_path, manifest = _load_stage_manifest(preparation_manifest, stage="prepare")
    expected = configuration_identity(setup, model=model)
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
    return prepared, setup.retained_run_spec(prepared)


def _build_prepared_model(
    prepared: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    *,
    model: ModelKind,
):
    if model == "multisector":
        names = multisector_model_input_names(prepared, run_spec.model)
        build = build_multisector_rhime_model_result
    else:
        names = standard_model_input_names(prepared, run_spec.model)
        build = build_standard_rhime_model_result
    inputs = materialize_pymc_inputs(prepared, variable_names=names)
    return build(prepared=prepared, model_inputs=inputs, run_spec=run_spec)


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
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("Prior-predictive draws must be a positive integer.")
    destination = _stage_output_directory(output_dir)
    prior_path = _output_path(destination, None, "prior-predictive.nc")
    prepared, run_spec = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    try:
        built = _build_prepared_model(prepared, run_spec, model=model)
        with built.model:
            prior = pm.sample_prior_predictive(draws, built.model)
        values = [
            np.asarray(group[name].values) for group in prior.children.values() for name in group.data_vars
        ]
        non_finite = sum(int(np.size(value) - np.isfinite(value).sum()) for value in values)
        status = "pass" if values and non_finite == 0 else "fail"
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


def sample_rhime_stage(
    *,
    setup: RhimeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Sample prepared inputs and persist matched posterior/output bindings.

    Writes the trace, a versioned output contract bound to both numerical
    artifacts, and a schema-version-2 sample manifest. Preparation is never
    invoked implicitly.
    """
    destination = _stage_output_directory(output_dir)
    prepared, run_spec = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    built = _build_prepared_model(prepared, run_spec, model=model)
    idata = sample_rhime_model(built, setup.sampler)
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
        "artifact_identities": {
            **identities,
            "output_binding": _file_identity(binding_path),
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

    Version-2 sample manifests use their verified saved output bindings, without
    materializing model inputs or constructing a graph. Genuine version-1
    manifests retain graph reconstruction to recover their missing roles.
    Output settings may change; the sampled scientific configuration may not.
    """
    destination = _stage_output_directory(output_dir)
    configured_output = setup.output
    _filename_component("output_name", configured_output.output_name)
    _filename_component("species", setup.model.species)
    _filename_component("domain", setup.model.domain)
    _filename_component("start_date", setup.start_date)
    prepared, run_spec = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    posterior_path = Path(posterior).resolve()
    sample_contract = _verify_sample_manifest(
        sample_manifest,
        posterior=posterior_path,
        configuration_identity=configuration_identity(setup, model=model),
        prepared_inputs=prepared_inputs,
    )
    sampled_sampler = _sampler_from_sample_manifest(sample_contract, path=sample_manifest)
    output_spec = replace(
        configured_output,
        output_path=str(destination),
        save_trace=False,
        save_inversion_output=bool(configured_output.save_inversion_output),
    )
    run_spec = replace(run_spec, output=output_spec)
    resolved = replace(setup, output=output_spec, sampler=sampled_sampler)
    if sample_contract["schema_version"] == 1:
        # Older manifests did not persist roles; preserve their explicit graph replay.
        built = _build_prepared_model(prepared, run_spec, model=model)
        output_contract = built.output_contract
    else:
        built = None
        output_contract = _load_output_binding(sample_contract, manifest_path=Path(sample_manifest).resolve())
    output_contract.validate_requested_output(output_spec.output_format)
    idata = load_trace(posterior_path)
    if model == "multisector":
        result = make_multisector_rhime_result(
            prepared=prepared,
            run_spec=run_spec,
            sampler=resolved.sampler,
            model_build_result=built,
            output_contract=output_contract,
            idata=idata,
            build_and_sample_seconds=0.0,
        )
        make_multisector_rhime_outputs(result=result, prepared=prepared)
    else:
        result = make_standard_rhime_result(
            prepared=prepared,
            run_spec=run_spec,
            sampler=resolved.sampler,
            model_build_result=built,
            output_contract=output_contract,
            idata=idata,
            build_and_sample_seconds=0.0,
        )
        make_standard_rhime_outputs(result=result, prepared=prepared)
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
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "postprocess",
        "configuration_identity": configuration_identity(resolved, model=model),
        "effective_configuration": effective_configuration(resolved, model=model),
        "input_identities": dict(sample_contract["artifact_identities"]),
        "artifacts": {name: _artifact_path(Path(path)) for name, path in artifacts.items()},
    }
    manifest_path = _write_json(_output_path(destination, None, "postprocess-manifest.json"), manifest)
    result.output_metadata["postprocess_manifest_path"] = str(manifest_path)
    return result
