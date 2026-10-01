"""File-backed stages for the concrete ordinary and cached CO2 recipes.

Preparation accepts the public coherent handoff, or canonical observations
and a coherent reduction supplied directly through the Python API. Campaign
discovery stays with the caller. Saved handoffs are copied byte-for-byte so
their affine reconstruction binding remains valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
import shutil
from typing import Any, Mapping, cast

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.rhime.builders import callable_metadata
from openghg_inversions.rhime.outputs import RhimeResult
from openghg_inversions.rhime.specs import RhimeModelSpec, RhimeOutputSpec, RhimeRunSpec
from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.rhime._stage_artifacts import (
    artifact_path as _artifact_path,
    file_identity as _file_identity,
    json_value as _json_value,
    write_json as _write_json,
    _filename_component,
    _output_path,
    _stage_output_directory,
)
from openghg_inversions.rhime._stage_authentication import (
    _load_stage_manifest,
    _sampler_from_sample_manifest,
    _verify_manifest_artifact,
)
from openghg_inversions.rhime._stage_checks import PREPARATION_CHECK_NAME, _check_result, _check_stage
from openghg_inversions.serialization import load_trace, reset_serialisation_multiindexes, save_trace
from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs

from .co2_affine_output import BoundCo2AffineFluxMap, load_and_bind_affine_flux_map
from .co2_cached_sigma_runner import (
    build_rhime_co2_cached_sigma,
    co2_cached_sigma_input_names,
    run_rhime_co2_cached_sigma,
    sample_co2_cached_prior_predictive,
)
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .co2_runner import annotate_co2_trace, build_rhime_co2, co2_model_input_names
from .configuration import Co2RunSetup, resolve_co2_family_config


@dataclass(frozen=True)
class Co2StageSetup:
    """Resolved scientific replay and output settings for the CO2 stages.

    Attributes:
        recipe: Existing CO2 configuration, including its matched sampler.
        output: Requested common product and naming settings.
        reconstruction_path: Optional affine artifact bound to prepared inputs.
        source_to_sector: Explicit source-to-sector reporting labels.
    """

    recipe: Co2RunSetup
    output: RhimeOutputSpec
    reconstruction_path: Path | None
    source_to_sector: Mapping[str, str] | None = None


def load_co2_stage_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an explicit CO2 configuration and resolve paths beside its source."""
    if (config_file is None) == (params_file is None):
        raise ValueError("Pass exactly one of `config_file` or `params_file`.")
    if config_file is not None:
        from . import load_co2_family_config

        source_path = Path(config_file).resolve()
        params = dict(load_co2_family_config(source_path))
    else:
        source_path = Path(cast(str | Path, params_file)).resolve()
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"RHIME params file {source_path} must contain one JSON object.")
        params = loaded
    if overrides:
        params.update(overrides)
    return resolve_co2_stage_paths(params, base_dir=source_path.parent)


def resolve_co2_stage_paths(params: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Resolve CO2 artifact paths relative to the configuration file."""
    resolved = dict(params)
    for section, keys in (
        ("prepared_inputs", ("path",)),
        ("likelihood", ("eigenbasis_path",)),
        ("outputs", ("reconstruction_path", "country_file")),
    ):
        if section not in resolved:
            continue
        options = dict(resolved[section])
        for key in keys:
            if key in options:
                path = Path(options[key]).expanduser()
                options[key] = str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())
        resolved[section] = options
    return resolved


def resolve_co2_stage_setup(params: Mapping[str, Any]) -> Co2StageSetup:
    """Reuse the CO2 resolver and reject unsupported stage outputs early."""
    from .co2_outputs import validate_co2_output_spec

    options = dict(params)
    outputs = dict(options.pop("outputs", {}))
    reconstruction = outputs.pop("reconstruction_path", None)
    sectors = outputs.pop("source_to_sector", None)
    if sectors is not None and (
        not isinstance(sectors, Mapping)
        or not sectors
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in sectors.items()
        )
    ):
        raise ValueError("outputs.source_to_sector must map source names to non-empty sector names.")
    allowed = {"output_format", "output_name", "country_file", "save_inversion_output"}
    if unknown := outputs.keys() - allowed:
        raise ValueError(f"Unknown CO2 output option(s): {sorted(unknown)!r}.")
    output = RhimeOutputSpec(output_format="basic", save_inversion_output=False)
    output = RhimeOutputSpec(**{**asdict(output), **outputs})
    _filename_component("output_name", output.output_name)
    validate_co2_output_spec(output, has_reconstruction=reconstruction is not None)
    if output.output_format == "paris" and not output.country_file:
        raise ValueError("Staged CO2 PARIS outputs require an explicit outputs.country_file.")
    if sectors and reconstruction is None:
        raise ValueError("CO2 source_to_sector requires a bound affine reconstruction.")
    recipe = resolve_co2_family_config(options)
    if not isinstance(recipe, Co2RunSetup):
        raise ValueError("The staged co2 model requires recipe='co2'; linked CO2/O2 is not supported.")
    if (
        output.output_format != "none"
        and reconstruction is not None
        and not recipe.sampler.sample_prior_predictive
    ):
        raise ValueError("Conditional CO2 flux outputs require sampling.sample_prior_predictive draws.")
    return Co2StageSetup(
        recipe, output, Path(reconstruction) if reconstruction is not None else None, sectors
    )


def effective_co2_configuration(setup: Co2StageSetup) -> dict[str, Any]:
    """Return JSON-safe resolved science, sampler, and output provenance."""
    kwargs = dict(cast(Mapping[str, Any], setup.recipe.runner_kwargs))
    if "likelihood_builder" in kwargs:
        kwargs["likelihood_builder"] = callable_metadata(kwargs["likelihood_builder"])
    return _json_value(
        {
            "model": "co2",
            "runner": callable_metadata(setup.recipe.runner),
            "runner_kwargs": kwargs,
            "preparation": setup.recipe.preparation_kwargs,
            "sampler": {name: getattr(setup.recipe.sampler, name) for name in setup.recipe.sampler.__slots__},
            "output": asdict(setup.output),
            "reconstruction_path": setup.reconstruction_path,
            "source_to_sector": setup.source_to_sector,
        }
    )


def co2_configuration_identity(setup: Co2StageSetup) -> str:
    """Hash resolved recipe settings needed for scientific replay.

    Prepared data are authenticated separately by artifact content hashes.
    Transport paths, output settings, and sampler tuning may change without
    changing this configuration identity.
    """
    science = effective_co2_configuration(setup)
    science.pop("preparation")
    for key in ("sigma_target_accept", "state_target_accept"):
        science["runner_kwargs"].pop(key, None)
    for key in ("sampler", "output", "reconstruction_path", "source_to_sector"):
        science.pop(key)
    encoded = json.dumps(science, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def _manifest(setup: Co2StageSetup, stage: str) -> dict[str, Any]:
    """Build a CO2 manifest requiring an identifiable installed revision."""
    ogi = installed_ogi_provenance()
    if ogi["revision"] is None:
        raise ValueError(
            "Staged CO2 execution requires an identifiable installed Git revision (VCS install or checkout)."
        )
    return {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": stage,
        "configuration_identity": co2_configuration_identity(setup),
        "effective_configuration": effective_co2_configuration(setup),
        "ogi": ogi,
    }


def _validate_inputs(setup: Co2StageSetup, prepared: Co2PreparedInputs) -> None:
    options = cast(Mapping[str, Any], setup.recipe.runner_kwargs)
    sites = set(prepared.sites)
    likelihood = options.get("likelihood_kwargs", options)
    for key in ("tau_hours", "initial_site_amplitudes", "fixed_site_amplitudes"):
        value = likelihood.get(key)
        if isinstance(value, Mapping) and set(value) != sites:
            raise ValueError(f"CO2 {key} must cover the prepared site labels exactly: {sorted(sites)!r}.")
    if setup.output.output_format == "paris" and len(set(prepared.averaging_period)) != 1:
        raise ValueError("CO2 PARIS concentration requires one common observation averaging period.")
    if setup.output.country_file:
        countries = Countries.from_file(country_file=setup.output.country_file)
        xr.align(prepared.basis_functions.flux, countries.matrix, join="exact", copy=False)
    if setup.recipe.runner is run_rhime_co2_cached_sigma:
        co2_cached_sigma_input_names(prepared, use_bc=bool(options.get("use_bc", False)))
    else:
        co2_model_input_names(
            prepared,
            use_bc=bool(options.get("use_bc", False)),
            preserve_prepared_fixed_mismatch="likelihood_builder" not in options
            and "fixed_model_mismatch" not in options,
        )


def prepare_co2_stage(
    *,
    setup: Co2StageSetup,
    output_dir: str | Path,
    canonical_inputs: RhimePreparedInputs | None = None,
    reduction: CoherentGaussianReduction | None = None,
    aggregation_error_rank: int | None = 512,
) -> dict[str, Any]:
    """Persist a source-neutral coherent CO2 handoff and preparation manifest.

    Pass both ``canonical_inputs`` and ``reduction`` to call the public
    preparation boundary directly. Otherwise copy ``prepared_inputs.path``
    from the configuration after validation. Existing handoffs and their
    optional bound affine artifacts retain exact byte identities. Invalid
    science or output requests fail before creating the destination.
    """
    if (canonical_inputs is None) != (reduction is None):
        raise ValueError("Pass canonical_inputs and reduction together.")
    source = Path(cast(Path, setup.recipe.preparation_kwargs["path"]))
    if canonical_inputs is None and source.is_dir():
        raise ValueError("Installed staging requires a NetCDF handoff; save Co2PreparedInputs to a .nc file.")
    prepared = (
        prepare_co2_inputs(
            canonical_inputs,
            cast(CoherentGaussianReduction, reduction),
            aggregation_error_rank=aggregation_error_rank,
        )
        if canonical_inputs is not None
        else Co2PreparedInputs.load(source)
    )
    _validate_inputs(setup, prepared)
    if canonical_inputs is not None and setup.reconstruction_path is not None:
        raise ValueError(
            "Save the new prepared handoff and bind its reconstruction before staging affine outputs."
        )
    if setup.reconstruction_path is not None:
        from .co2_outputs import validate_co2_outputs

        bound = load_and_bind_affine_flux_map(setup.reconstruction_path, source)
        validate_co2_outputs(setup.output, bound=bound, source_to_sector=setup.source_to_sector)
    manifest = _manifest(setup, "prepare")
    destination = _stage_output_directory(output_dir)
    prepared_path = _output_path(destination, None, "prepared-inputs.nc")
    if canonical_inputs is not None:
        prepared.save(prepared_path)
    elif source.resolve() != prepared_path:
        shutil.copyfile(source, prepared_path)
    artifacts = {"prepared_inputs": prepared_path}
    if setup.reconstruction_path is not None:
        companion = _output_path(destination, None, "affine-flux-map.nc")
        if setup.reconstruction_path.resolve() != companion:
            shutil.copyfile(setup.reconstruction_path, companion)
        artifacts["affine_reconstruction"] = companion
    manifest["requested_configuration"] = effective_co2_configuration(setup)
    manifest["artifacts"] = {key: _artifact_path(path) for key, path in artifacts.items()}
    manifest["artifact_identities"] = {key: _file_identity(path) for key, path in artifacts.items()}
    manifest["manifest_path"] = str(_write_json(destination / "prepare-manifest.json", manifest))
    return manifest


def _load_prepared(
    setup: Co2StageSetup,
    prepared_inputs: str | Path,
    preparation_manifest: str | Path,
) -> tuple[Co2PreparedInputs, BoundCo2AffineFluxMap | None, dict[str, Any]]:
    manifest_path, manifest = _load_stage_manifest(preparation_manifest, stage="prepare")
    if manifest.get("configuration_identity") != co2_configuration_identity(setup):
        raise ValueError("Prepared inputs do not match the effective CO2 configuration.")
    _verify_manifest_artifact(
        manifest, manifest_path=manifest_path, artifact_name="prepared_inputs", artifact_path=prepared_inputs
    )
    prepared = Co2PreparedInputs.load(prepared_inputs)
    _validate_inputs(setup, prepared)
    bound = None
    if setup.reconstruction_path is not None:
        # The configured artifact is authenticated by content, allowing the
        # caller to relocate it without changing scientific identity.
        _verify_manifest_artifact(
            manifest,
            manifest_path=manifest_path,
            artifact_name="affine_reconstruction",
            artifact_path=setup.reconstruction_path,
        )
        bound = load_and_bind_affine_flux_map(setup.reconstruction_path, prepared_inputs)
    from .co2_outputs import validate_co2_outputs

    validate_co2_outputs(setup.output, bound=bound, source_to_sector=setup.source_to_sector)
    return prepared, bound, manifest


def prior_predictive_co2_stage(
    *,
    setup: Co2StageSetup,
    prepared_inputs: str | Path,
    preparation_manifest: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Check exact prior replicates from the selected CO2 graph without sampling a posterior."""
    stage = _check_stage(stage)
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws <= 0:
        raise ValueError("Prior-predictive draws must be a positive integer.")
    prepared, _, preparation = _load_prepared(setup, prepared_inputs, preparation_manifest)
    seed = dict(setup.recipe.sampler.sample_kwargs or {}).get("random_seed")
    kwargs = dict(cast(Mapping[str, Any], setup.recipe.runner_kwargs))
    if setup.recipe.runner is run_rhime_co2_cached_sigma:
        kwargs.pop("sigma_target_accept", None)
        kwargs.pop("state_target_accept", None)
        cached = build_rhime_co2_cached_sigma(prepared_inputs=prepared, **kwargs)
        prior = sample_co2_cached_prior_predictive(cached, prepared, draws=draws, random_seed=seed)
    else:
        built = build_rhime_co2(prepared_inputs=prepared, **kwargs)
        with built.model:
            prior = pm.sample_prior_predictive(draws, random_seed=seed)
        registry = get_coord_registry(built.model)
        if registry is not None:
            prior = restore_inferencedata_coords(prior, registry)
        annotate_co2_trace(prior, built, concentration_units=prepared.inv_inputs["mf"].attrs.get("units"))
    values = [np.asarray(group[name].values) for group in prior.children.values() for name in group.data_vars]
    non_finite = sum(int(value.size - np.isfinite(value).sum()) for value in values)
    manifest = _manifest(setup, "prior-predictive")
    destination = _stage_output_directory(output_dir)
    prior_path = _output_path(destination, None, "prior-predictive.nc")
    save_trace(prior, prior_path)
    status = "pass" if values and non_finite == 0 else "fail"
    result = _check_result(
        name=PREPARATION_CHECK_NAME,
        status=status,
        measured_values={"draws": draws, "non_finite_values": non_finite},
        thresholds={"max_non_finite_values": 0},
        message=f"CO2 prior predictive produced {draws} draws with {non_finite} non-finite values.",
        artifact_paths=[_artifact_path(prior_path)],
        stage=stage,
    )
    _write_json(_output_path(destination, check_output, "prior-predictive-readiness.json"), result)
    manifest["input_identities"] = preparation["artifact_identities"]
    manifest["artifacts"] = {"prior_predictive": _artifact_path(prior_path)}
    manifest["artifact_identities"] = {"prior_predictive": _file_identity(prior_path)}
    _write_json(destination / "prior-predictive-manifest.json", manifest)
    return result


def sample_co2_stage(
    *,
    setup: Co2StageSetup,
    prepared_inputs: str | Path,
    preparation_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Sample with the public ordinary or matched cached runner, then save canonically."""
    prepared, _, preparation = _load_prepared(setup, prepared_inputs, preparation_manifest)
    manifest = _manifest(setup, "sample")
    trace = setup.recipe.runner(**setup.recipe.runner_arguments(prepared))
    destination = _stage_output_directory(output_dir)
    posterior_path = _output_path(destination, None, "posterior.nc")
    save_trace(trace, posterior_path)
    manifest["input_identities"] = preparation["artifact_identities"]
    manifest["artifacts"] = {
        "posterior": _artifact_path(posterior_path),
        "prepared_inputs": _artifact_path(Path(prepared_inputs)),
    }
    manifest["artifact_identities"] = {
        "posterior": _file_identity(posterior_path),
        **preparation["artifact_identities"],
    }
    manifest["manifest_path"] = str(_write_json(destination / "sample-manifest.json", manifest))
    return manifest


def _run_spec(setup: Co2StageSetup, prepared: Co2PreparedInputs, output_dir: Path) -> RhimeRunSpec:
    options = cast(Mapping[str, Any], setup.recipe.runner_kwargs)
    time = prepared.inv_inputs["time"]
    start = str(np.asarray(time.min().values).astype("datetime64[D]"))
    end = str(np.asarray(time.max().values).astype("datetime64[D]") + np.timedelta64(1, "D"))
    model = RhimeModelSpec(
        species="co2",
        domain=prepared.inv_inputs.attrs.get("domain", "unspecified"),
        sectors=(),
        use_bc=bool(options.get("use_bc", False)),
        add_offset="offset_prior" in options,
        aggregation_error_mode=prepared.aggregation_error_mode,
        bc_prior=options.get("bc_prior"),
        offset_prior=options.get("offset_prior"),
        offset_args=dict(options.get("offset_args", {})),
    )
    output = RhimeOutputSpec(**{**asdict(setup.output), "output_path": None})
    return RhimeRunSpec(start, end, prepared.sites, prepared.averaging_period, model, output)


def postprocess_co2_stage(
    *,
    setup: Co2StageSetup,
    prepared_inputs: str | Path,
    preparation_manifest: str | Path,
    posterior: str | Path,
    sample_manifest: str | Path,
    output_dir: str | Path,
) -> RhimeResult:
    """Authenticate saved CO2 artifacts and create role-selected common products."""
    from .co2_outputs import make_co2_rhime_outputs, make_co2_rhime_result

    prepared, bound, preparation = _load_prepared(setup, prepared_inputs, preparation_manifest)
    path, sample = _load_stage_manifest(sample_manifest, stage="sample")
    if sample.get("configuration_identity") != co2_configuration_identity(setup):
        raise ValueError("Posterior does not match the effective CO2 configuration.")
    for name, artifact in (("posterior", posterior), ("prepared_inputs", prepared_inputs)):
        _verify_manifest_artifact(sample, manifest_path=path, artifact_name=name, artifact_path=artifact)
    if bound is not None and sample["artifact_identities"].get("affine_reconstruction") != preparation[
        "artifact_identities"
    ].get("affine_reconstruction"):
        raise ValueError("Posterior and preparation disagree on the affine reconstruction identity.")
    sampler = _sampler_from_sample_manifest(sample, path=sample_manifest)
    trace = load_trace(posterior)
    destination = Path(output_dir).resolve()
    result = make_co2_rhime_result(
        prepared=prepared,
        run_spec=_run_spec(setup, prepared, destination),
        sampler=sampler,
        idata=trace,
    )
    # Build all requested products before opening any output destination.
    make_co2_rhime_outputs(
        result=result, prepared=prepared, bound=bound, source_to_sector=setup.source_to_sector
    )
    manifest = _manifest(setup, "postprocess")
    destination = _stage_output_directory(destination)
    artifacts = {}
    for name, product in result.outputs.items():
        if isinstance(product, xr.Dataset):
            path = _output_path(destination, None, f"{name}.nc")
            if name.startswith("paris_"):
                write_netcdf_preserving_bounds_attrs(product, path)
            else:
                reset_serialisation_multiindexes(product).to_netcdf(path)
            artifacts[name] = _artifact_path(path)
            result.output_metadata[f"{name}_path"] = str(path)
    manifest["input_identities"] = {
        **preparation["artifact_identities"],
        "posterior": _file_identity(Path(posterior)),
    }
    manifest["sampling_configuration"] = sample["effective_configuration"]["sampler"]
    manifest["artifacts"] = artifacts
    manifest["artifact_identities"] = {
        name: _file_identity(Path(result.output_metadata[f"{name}_path"])) for name in artifacts
    }
    result.output_metadata["postprocess_manifest_path"] = str(
        _write_json(destination / "postprocess-manifest.json", manifest)
    )
    return result


__all__ = [
    "Co2StageSetup",
    "resolve_co2_stage_setup",
    "prepare_co2_stage",
    "prior_predictive_co2_stage",
    "sample_co2_stage",
    "postprocess_co2_stage",
]
