"""Concrete standard and multisector file-backed RHIME workflows.

These functions compose the existing RHIME scientific stages.  They do not
define a second configuration schema and have no dependency on an
orchestrator or scheduler.
"""

from __future__ import annotations

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.basis_functions import BASIS_ARTIFACT_PATH_ATTR
from openghg_inversions.rhime.preparation import resolve_basis_inputs

from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast, get_type_hints
from collections.abc import Mapping

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions._timing import timed, timer_start, timer_seconds
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler
from .specs import RhimeRunSpec
from ._resources import resolve_resource_paths
from ._runner_compat import _ALIASES, translate_rhime_aliases
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
from .outputs import (
    RhimeResult,
    make_multisector_rhime_outputs,
    make_standard_rhime_outputs,
    validate_output_resources,
)
from .ini import read_rhime_ini
from .params import RHIME_PREPARATION_OPTION_NAMES, RhimeConfig, DataConfig, resolve_model_options, resolve_flux_sources
from .specs import RhimeModelSpec, RhimeOutputSpec, SectorSpec, PollutionEventSettings, AdditiveSigmaSettings, FixedErrorSettings
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.inversion_data import SiteOptions
from .preparation import (
    assemble_rhime_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
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
    json_value as _json_value,
    write_json as _write_json,
    _filename_component,
    _output_path,
    _stage_output_directory,
)
from ._stage_authentication import (
    _load_output_binding,
    _load_stage_manifest,
    _sampler_metadata_from_sample_manifest,
    _load_sample_manifest,
)
from ._stage_checks import PREPARATION_CHECK_NAME, _check_result, _check_stage

ModelKind = Literal["standard", "multisector"]

_PREPARATION_IDENTITY_EXCLUDED_OPTIONS = frozenset(
    {
        "basis_output_path",
        "output_name",
    }
)


def load_stage_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load existing RHIME parameters from one explicit source.

    ``CONFIG_FILE`` is intentionally not an environment default: an unrelated
    ambient variable must never select the scientific configuration. File
    resources resolve beside the source; explicit overrides resolve from the
    invocation working directory.
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
    annotations = get_type_hints(DataConfig) | get_type_hints(RhimeOutputSpec)

    def resolve(values: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
        winning = {**values, **updates}
        translated = translate_rhime_aliases(winning)
        invocation_names = {
            _ALIASES.get(name, name) for name in updates
            if name not in _ALIASES or _ALIASES[name] not in winning
        }
        return resolve_resource_paths(
            {name: value for name, value in translated.items() if name not in invocation_names},
            annotations=annotations, base_dir=source_path.parent,
        ) | resolve_resource_paths(
            {name: value for name, value in translated.items() if name in invocation_names},
            annotations=annotations, base_dir=Path.cwd(),
        )

    from .params import rhime_option_groups

    updates = dict(overrides or {})
    owners = rhime_option_groups()
    if (params.keys() | updates.keys()) & owners.keys():
        aliases = _ALIASES | {"emissions_store": "flux_store", "emissions_domain": "flux_domain"}

        def group_raw(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
            grouped: dict[str, dict[str, Any]] = {}
            unsupported = []
            for name, value in values.items():
                if name in owners:
                    if not isinstance(value, Mapping):
                        raise ValueError(f"`{name}` must be a mapping.")
                    grouped.setdefault(name, {}).update(value)
                    continue
                canonical = aliases.get(name, name)
                owner = next((group for group, names in owners.items() if canonical in names), None)
                if name == "use_tracer":
                    owner = "data"
                if owner is None:
                    unsupported.append(name)
                else:
                    grouped.setdefault(owner, {})[name] = value
            if unsupported:
                raise ValueError(f"Unsupported RHIME parameter(s): {sorted(unsupported)!r}")
            return grouped

        grouped_params, grouped_updates = group_raw(params), group_raw(updates)
        return {group: resolve(grouped_params.get(group, {}), grouped_updates.get(group, {}))
                for group in grouped_params.keys() | grouped_updates.keys()}
    return resolve(params, updates)


def effective_configuration(setup: RhimeConfig, *, model: ModelKind) -> dict[str, Any]:
    """Project resolved choices into this workflow's manifest vocabulary.

    This is a staged artifact representation, not a configuration export API.
    No pre-preparation run description or raw shorthand is reconstructed.
    """
    site_names = {field.name for field in fields(setup.data.site_options)}
    preparation = {
        name: getattr(setup.data, name)
        for name in RHIME_PREPARATION_OPTION_NAMES if name not in site_names
    }
    preparation.update({name: getattr(setup.data.site_options, name) for name in site_names})
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
    site_names = {field.name for field in fields(setup.data.site_options)}
    preparation = setup.data.select(*(RHIME_PREPARATION_OPTION_NAMES - site_names))
    preparation.update({name: getattr(setup.data.site_options, name) for name in site_names})
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
    """Prepare and persist independently inspectable RHIME inputs.

    Selected basis and support files are resolved before acquisition. Reporting
    resources are checked only when outputs are requested at postprocessing.
    ``setup.data.basis_output_path`` remains an exact caller-selected destination;
    it is not relocated below ``output_dir``. The stage writes prepared inputs
    and their manifest below ``output_dir`` and returns the manifest mapping.
    """
    multisector = model == "multisector"
    basis_inputs = resolve_basis_inputs(setup.data)
    bc_basis = load_bc_basis(setup.data.bc_basis_paths) if setup.data.use_bc and setup.data.bc_basis_paths is not None else None
    destination = _stage_output_directory(output_dir)

    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(setup.data.site_options.sites),
        split_by_sectors=setup.data.split_by_sectors,
    ):
        merged = RhimeMergedData.from_options(
            **setup.data.select(
                "site_options", "species", "domain", "start_date",
                "end_date",  "flux_sources", "split_by_sectors",
                "bc_store", "obs_store", "footprint_store", "flux_store",
                "flux_domain", "fp_model", "fp_species", "calibration_scale",
                "use_bc", "bc_input",
                "flux_non_finite_check",
            ),
        )
    merged = prepare_observation_errors(merged, averaging_error=setup.data.averaging_error)
    filtered = filter_observations(merged, filters=setup.data.filters)
    retained_sites = {str(site).upper() for site in filtered.sites}
    missing_sites = [site for site in setup.data.site_options.sites if str(site).upper() not in retained_sites]
    if missing_sites:
        raise ValueError(
            "RHIME preparation could not produce required site input(s) "
            f"{missing_sites!r} for species {setup.data.species!r} and period "
            f"{setup.data.start_date} to {setup.data.end_date}."
        )
    runtime_flux = resolve_runtime_flux(filtered.flux_data, split_by_sectors=filtered.split_by_sectors)
    with timed("rhime.prepare_inputs.basis_build", basis_algorithm=setup.data.basis_algorithm):
        basis = prepare_basis(
            site_data=filtered.site_data,
            runtime_flux=runtime_flux,
            weighting_flux=filtered.flux_data[setup.data.flux_sources[0]]["flux"],
            **setup.data.select(
                "domain", "start_date", "basis_algorithm", "nbasis",
            ),
            fix_outer_regions=setup.data.fix_basis_outer_regions,
            **basis_inputs,
        )
    if setup.data.basis_output_path is not None:
        basis.save(setup.data.basis_output_path)
        basis = basis.with_metadata(
            {BASIS_ARTIFACT_PATH_ATTR: str(setup.data.basis_output_path)}
        )
    site_data = build_sensitivities(
        filtered,
        basis,
        **setup.data.select(
            "flux_sources", "use_bc",
        ),
        bc_basis=bc_basis,
        multisector=multisector,
    )
    prepared = assemble_rhime_inputs(
        filtered,
        basis,
        site_data,
        **setup.data.select(
            "domain", "start_date", "bc_freq", "min_error",
            "min_error_options", "use_bc",
        ),
    )
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
    built = _build_prepared_model(prepared, run_spec, model=model)
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
    _filename_component("output_name", configured_output.output_name)
    _filename_component("species", setup.model.species)
    _filename_component("domain", setup.model.domain)
    _filename_component("start_date", setup.data.start_date)
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
        built = _build_prepared_model(prepared, run_spec, model=model)
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

# Standard handoffs are intentionally independent of the retired staged envelope
# still used by the deferred multisector route below.
_CONTEXT_FIELDS = (
    "species", "domain", "start_date", "end_date", "flux_sources",
    "use_bc", "bc_freq", "split_by_sectors",
)


def _read_record(path: str | Path, *, stage: str) -> tuple[Path, dict[str, Any]]:
    """Read the current standard format once, without ambient path discovery."""
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or (
        record.get("schema") != f"openghg_inversions.standard.{stage}"
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("family") != "standard"
    ):
        raise ValueError(f"Unsupported standard {stage} manifest: {path}.")
    required = (
        ("prepared_inputs", "context") if stage == "prepare"
        else ("preparation", "posterior", "output_contract", "model", "build_and_sample_seconds")
    )
    if any(name not in record for name in required):
        raise ValueError(f"Standard {stage} manifest is missing required contents: {path}.")
    if stage == "sample" and record.get("inference_kind") != "mcmc":
        raise ValueError("Standard sample manifest requires supported inference_kind='mcmc'.")
    return path, record


def _reference(path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest {path} requires a nonempty artifact reference.")
    return (path.parent / value).resolve()


def _save_record(destination: Path, *, stage: str, **contents: Any) -> Path:
    temporary = _write_json(destination / f".{stage}-manifest.tmp", {
        "schema": f"openghg_inversions.standard.{stage}", "schema_version": 1,
        "producer": "openghg_inversions", "family": "standard", "stage": stage,
        **contents,
    })
    return temporary.replace(destination / f"{stage}-manifest.json")


def save_preparation(
    prepared: RhimePreparedInputs, *, data: DataConfig, output_dir: str | Path,
) -> Path:
    """Save numerical preparation, then return its completed absolute manifest.

    The destination must be new. Keep both the manifest and its referenced
    numerical file available for later sampling and output replay.
    ``data`` describes the request that produced ``prepared``; retained sites
    and arrays come from ``prepared``. Saving materializes numerical payloads
    through their existing codec without mutating the borrowed inputs.
    """
    if data.split_by_sectors:
        raise ValueError("Standard handoffs require standard preparation.")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    return _write_preparation(prepared, data=data, destination=destination)


def _write_preparation(
    prepared: RhimePreparedInputs, *, data: DataConfig, destination: Path,
) -> Path:
    prepared.save(destination / "prepared-inputs.nc")
    return _save_record(
        destination, stage="prepare", prepared_inputs="prepared-inputs.nc",
        context=data.select(*_CONTEXT_FIELDS),
        preparation_provenance={
            **data.select(
                "averaging_error", "min_error", "min_error_options", "filters",
                "basis_path", "basis_partition_path", "basis_algorithm", "nbasis",
                "fix_basis_outer_regions",
            ),
            "requested_site_options": asdict(data.site_options),
            "basis_artifact_source": prepared.basis_artifact_source,
        },
    )


def load_preparation(manifest: str | Path) -> tuple[RhimePreparedInputs, dict[str, Any]]:
    """Load saved numerical inputs and original request facts from one manifest.

    Relative references resolve beside the manifest; absolute shared-cache
    references are used directly. Missing numerical files retain loader errors.
    """
    path, record = _read_record(manifest, stage="prepare")
    context = record["context"]
    if not isinstance(context, dict) or set(context) != set(_CONTEXT_FIELDS):
        raise ValueError("Standard preparation manifest has incomplete request facts.")
    if context["split_by_sectors"] is not False:
        raise ValueError("Standard preparation manifest declares a different family layout.")
    for name in ("species", "domain", "start_date", "end_date"):
        if not isinstance(context[name], str) or not context[name].strip():
            raise ValueError(f"Preparation manifest requires a nonempty `{name}`.")
    if not isinstance(context["use_bc"], bool):
        raise ValueError("Preparation manifest requires boolean `use_bc`.")
    if not isinstance(context["flux_sources"], list):
        raise ValueError("Preparation manifest requires a list of `flux_sources`.")
    context["flux_sources"] = tuple(resolve_flux_sources(flux_sources=context["flux_sources"]))
    return RhimePreparedInputs.load(_reference(path, record["prepared_inputs"])), context


def prepare(
    data: DataConfig, *, output_dir: str | Path, merged_data: RhimeMergedData | None = None,
) -> Path:
    """Prepare standard scientific inputs once and save a reusable manifest.

    Model, sampler and output choices are not needed. Observation, inventory,
    basis and baked error or BC-period changes require a new preparation.
    ``merged_data`` bypasses acquisition when supplied. Resource loading,
    acquisition and any exact ``data.basis_output_path`` export belong to this
    runner. ``output_dir`` must be new; the returned absolute manifest is written
    after the numerical inputs and is the completion marker.
    """
    from .standard import prepare_standard

    if data.split_by_sectors:
        raise ValueError("Standard handoffs require standard preparation.")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Preparation destination already exists: {destination}")
    basis_inputs = resolve_basis_inputs(data)
    bc_basis = load_bc_basis(data.bc_basis_paths) if data.use_bc and data.bc_basis_paths is not None else None
    destination.mkdir(parents=True, exist_ok=False)
    if merged_data is not None:
        merged_data.validate_for_preparation(
            **data.select("species", "domain", "start_date", "end_date", "split_by_sectors"),
        )
        merged = merged_data
    else:
        merged = RhimeMergedData.from_options(
            **data.select(
                "site_options", "species", "domain", "start_date", "end_date", "flux_sources",
                "split_by_sectors", "bc_store", "obs_store", "footprint_store", "flux_store",
                "flux_domain", "fp_model", "fp_species", "calibration_scale", "use_bc", "bc_input",
                "flux_non_finite_check",
            ),
        )
    prepared = prepare_standard(
        merged, basis_inputs=basis_inputs, bc_basis=bc_basis,
        **data.select(
            "domain", "start_date", "flux_sources", "averaging_error", "filters",
            "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
            "min_error", "min_error_options",
        ),
    )
    if data.basis_output_path is not None:
        prepared.basis_functions.save(data.basis_output_path)
        prepared = replace(prepared, basis_functions=prepared.basis_functions.with_metadata(
            {BASIS_ARTIFACT_PATH_ATTR: str(data.basis_output_path)}
        ))
    return _write_preparation(prepared, data=data, destination=destination)


def _run_spec(prepared: RhimePreparedInputs, context: Mapping[str, Any], model: RhimeModelSpec,
              output: RhimeOutputSpec) -> RhimeRunSpec:
    return RhimeRunSpec(
        start_date=context["start_date"], end_date=context["end_date"],
        sites=tuple(prepared.sites), averaging_period=tuple(prepared.averaging_period),
        model=model, output=output, split_by_sectors=context["split_by_sectors"],
    )


_MODEL_METADATA_FIELDS = (
    "species", "domain", "use_bc", "add_offset", "aggregation_error_mode",
    "bc_prior", "offset_prior", "offset_args",
)
_LIKELIHOOD_RECORDS = {
    "fixed_error": (FixedErrorSettings, ()),
    "additive_sigma": (AdditiveSigmaSettings, (
        "sigma_prior", "sigma_freq", "sigma_per_site", "sigma_freq_anchor", "use_minimum_error_floor",
    )),
    "pollution_event": (PollutionEventSettings, (
        "sigma_prior", "sigma_freq", "sigma_per_site", "sigma_freq_anchor", "pollution_events_from_obs", "power",
    )),
}


def _model_record(model: RhimeModelSpec) -> dict[str, Any]:
    """Save bounded scientific descriptions, independent of runtime record layout."""
    likelihood = None
    if model.likelihood is not None:
        kind = next(name for name, (cls, _) in _LIKELIHOOD_RECORDS.items()
                    if isinstance(model.likelihood, cls))
        likelihood = {"kind": kind, "options": {
            name: getattr(model.likelihood, name) for name in _LIKELIHOOD_RECORDS[kind][1]
        }}
    return {
        **{name: getattr(model, name) for name in _MODEL_METADATA_FIELDS},
        "sectors": [{"name": sector.name, "flux_source": sector.flux_source,
                     "x_prior": sector.x_prior, "variable_suffix": sector.variable_suffix}
                    for sector in model.sectors],
        "likelihood": likelihood,
    }


def _saved_model(value: Any) -> RhimeModelSpec:
    """Decode explicit scientific descriptions without executing saved callables."""
    if not isinstance(value, dict) or set(value) != set(_MODEL_METADATA_FIELDS) | {"sectors", "likelihood"}:
        raise ValueError("Sample manifest has incomplete resolved model metadata.")
    values = dict(value)
    likelihood = values["likelihood"]
    try:
        if likelihood is not None:
            if set(likelihood) != {"kind", "options"}:
                raise ValueError("Sample manifest has invalid likelihood metadata.")
            cls, option_names = _LIKELIHOOD_RECORDS[likelihood["kind"]]
            if set(likelihood["options"]) != set(option_names):
                raise ValueError("Sample manifest has incomplete likelihood metadata.")
            values["likelihood"] = cls(**likelihood["options"])
        values["sectors"] = tuple(SectorSpec(**sector) for sector in values["sectors"])
        return RhimeModelSpec(**values)
    except (KeyError, TypeError) as exc:
        raise ValueError("Sample manifest has invalid resolved model metadata.") from exc

def _provenance() -> dict[str, str | None]:
    from importlib.metadata import version, PackageNotFoundError
    import subprocess

    try:
        package_version = version("openghg_inversions")
    except PackageNotFoundError:
        package_version = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"package_version": package_version, "git_revision": revision}


def save_samples(
    preparation_manifest: str | Path, posterior: xr.DataTree, *,
    output_contract: OutputContract, model: RhimeModelSpec,
    actual_inference: Mapping[str, Any], build_and_sample_seconds: float,
    output_dir: str | Path,
) -> Path:
    """Save samples and their scientific meaning in a new operation directory.

    Several sample records may reference the same preparation. The completed
    manifest is written last; failed numerical saves leave no completed record.
    The output contract must be satisfied by the posterior, including retained
    scientific scales. Missing required states raise before creating the
    destination. ``actual_inference`` describes the completed invocation;
    it does not select future sampling defaults.
    """
    output_contract.validate_trace(posterior)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    save_trace(posterior, destination / "posterior.nc")
    return _save_record(
        destination, stage="sample", preparation=str(Path(preparation_manifest).resolve()),
        posterior="posterior.nc", output_contract=output_contract.to_dict(),
        model=_model_record(model), inference_kind="mcmc", actual_inference=dict(actual_inference),
        build_and_sample_seconds=build_and_sample_seconds, provenance=_provenance(),
    )


def sample(
    preparation_manifest: str | Path, *, model: Mapping[str, Any],
    sampler: RhimeSampler | Literal["default"] = "default", output_dir: str | Path,
) -> Path:
    """Sample a saved preparation using current decoded model and sampler choices.

    Scientific components validate required inputs before inference. Prepared
    BC policy and acquisition facts cannot be replaced by model choices.
    ``model`` contains canonical decoded choices, bound to the saved preparation.
    Omitted or ``"default"`` sampling selects a fresh sampler; explicit ``None``
    is rejected. A supplied sampler is copied for this invocation. The output
    directory must be new; return its absolute completed sample-manifest path.
    """
    sampler = resolve_sampler(sampler)
    if Path(output_dir).exists():
        raise FileExistsError(f"Sample destination already exists: {output_dir}")
    prepared, context = load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, preparation=DataConfig(
        site_options=SiteOptions.from_inputs(sites=tuple(prepared.sites), averaging_period=tuple(prepared.averaging_period)),
        **context,
    ))
    run_spec = _run_spec(prepared, context, model_spec, RhimeOutputSpec(output_format="none"))
    started = timer_start()
    built = _build_prepared_model(prepared, run_spec, model="standard")
    posterior = sample_rhime_model(built, sampler)
    return save_samples(
        preparation_manifest, posterior, output_contract=built.output_contract,
        model=model_spec, actual_inference=sampler.describe(),
        build_and_sample_seconds=timer_seconds(started), output_dir=output_dir,
    )


def postprocess(sample_manifest: str | Path, *, output: RhimeOutputSpec) -> RhimeResult:
    """Replay saved standard samples with current output choices, without a graph.

    The recorded model and inference describe the samples. Current output
    choices alone control products and destinations, including in-memory output.
    Keep referenced preparation and posterior files available for each replay.
    """
    path, record = _read_record(sample_manifest, stage="sample")
    output_contract = OutputContract.from_dict(record["output_contract"])
    output_contract.validate_requested_output(output.output_format)
    validate_output_resources(output)
    prepared, context = load_preparation(_reference(path, record["preparation"]))
    model = _saved_model(record["model"])
    for name in ("species", "domain"):
        if getattr(model, name) != context[name]:
            raise ValueError(f"Sample model `{name}` conflicts with its preparation.")
    if tuple(sector.flux_source for sector in model.sectors) != context["flux_sources"]:
        raise ValueError("Sample model flux sources conflict with its preparation.")
    if model.use_bc and not context["use_bc"]:
        raise ValueError("Sample model requires BC inputs absent from its preparation.")
    posterior = load_trace(_reference(path, record["posterior"]))
    result = make_standard_rhime_result(
        prepared=prepared, run_spec=_run_spec(prepared, context, model, output), sampler=None,
        output_contract=output_contract, idata=posterior,
        build_and_sample_seconds=record["build_and_sample_seconds"],
    )
    result.output_metadata["sampler"] = record.get("actual_inference")
    result.output_metadata["provenance"] = record.get("provenance", {})
    if output.output_format != "none":
        writes_directory = (
            output.save_trace is True or output.save_inversion_output is True
            or output.output_format in {"paris", "legacy"}
        )
        if writes_directory and output.output_path is not None and Path(output.output_path).exists():
            raise FileExistsError(f"Output destination already exists: {output.output_path}")
        for setting in (output.save_trace, output.save_inversion_output):
            if isinstance(setting, str | Path) and Path(setting).exists():
                raise FileExistsError(f"Output file already exists: {setting}")
    make_standard_rhime_outputs(result=result, prepared=prepared)
    return result


def prior_predictive(
    preparation_manifest: str | Path, *, model: Mapping[str, Any], output_dir: str | Path,
    draws: int = 100, check_output: str | Path | None = None, stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Check finite prior prediction from saved preparation and current model choices.

    Input loading and serialization errors propagate. Model readiness failures
    produce a failed check; this operation never runs posterior inference.
    """
    stage = _check_stage(stage)
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("Prior-predictive draws must be a positive integer.")
    prepared, context = load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, preparation=DataConfig(
        site_options=SiteOptions.from_inputs(sites=tuple(prepared.sites), averaging_period=tuple(prepared.averaging_period)),
        **context,
    ))
    run_spec = _run_spec(prepared, context, model_spec, RhimeOutputSpec(output_format="none"))
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        built = _build_prepared_model(prepared, run_spec, model="standard")
        with built.model:
            prior = pm.sample_prior_predictive(draws, built.model)
    except (KeyError, ValueError) as exc:
        non_finite, status = None, "fail"
        message = f"Prior-predictive readiness failed: {type(exc).__name__}: {exc}"
        artifacts = []
    else:
        concentration = built.output_contract.variable_roles.get("concentration")
        predictive = prior.children.get("prior_predictive")
        values = [np.asarray(group[name].values) for group in prior.children.values() for name in group.data_vars]
        non_finite = sum(int(np.size(value) - np.isfinite(value).sum()) for value in values)
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
        prior_path = destination / "prior-predictive.nc"
        save_trace(prior, prior_path)
        artifacts = [str(prior_path)]
    result = _check_result(
        name=PREPARATION_CHECK_NAME, status=status,
        measured_values={"draws": draws, "non_finite_values": non_finite},
        thresholds={"max_non_finite_values": 0}, message=message, artifact_paths=artifacts, stage=stage,
    )
    _write_json(_output_path(destination, check_output, "prior-predictive-readiness.json"), result)
    return result
