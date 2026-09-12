"""File-backed RHIME stages for workflow orchestrators.

These functions compose the existing RHIME scientific stages.  They do not
define a second configuration schema and have no dependency on an
orchestrator or scheduler.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.inversion_data import RhimePreparedInputs, _save_merged_data
from openghg_inversions.serialization import (
    load_inferencedata,
    reset_serialisation_multiindexes,
    save_inferencedata,
)

from .multisector import (
    build_multisector_rhime_model_result,
    make_multisector_rhime_result,
    multisector_model_input_names,
)
from .outputs import RhimeResult, make_multisector_rhime_outputs, make_standard_rhime_outputs
from .params import RhimeRunnerSetup, params_from_config, resolve_rhime_options
from .preparation import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
    retrieve_or_reload_rhime_data,
    with_prepared_rhime_sites,
)
from .sampling import sample_rhime_model
from .standard import (
    build_standard_rhime_model_result,
    make_standard_rhime_result,
    standard_model_input_names,
)
from .materialization import materialize_pymc_inputs

ModelKind = Literal["standard", "multisector"]

PREPARATION_CHECK_NAME = "prior-predictive-readiness"
CONVERGENCE_CHECK_NAME = "sampler-convergence"
CHECK_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Return a stable JSON-compatible value for configuration provenance."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _artifact_path(path: Path) -> str:
    """Prefer an OGR run-relative path while remaining scheduler-independent."""
    run_root = os.environ.get("RUN_ROOT")
    if run_root is not None:
        try:
            return str(path.resolve().relative_to(Path(run_root).resolve()))
        except ValueError:
            pass
    return str(path.resolve())


def _file_identity(path: Path) -> str:
    """Return the content identity used in compact stage manifests."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _output_path(output_dir: Path, requested: str | Path | None, default_name: str) -> Path:
    """Resolve an output path and keep it within the declared stage directory."""
    if requested is None:
        path = output_dir / default_name
    else:
        requested_path = Path(requested)
        path = requested_path.resolve() if requested_path.is_absolute() else (output_dir / requested_path).resolve()
    try:
        path.relative_to(output_dir)
    except ValueError:
        raise ValueError(f"Output path {path} must be beneath stage output directory {output_dir}.") from None
    return path


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
        params = params_from_config(Path(config_file).resolve(), normalise=False)
    else:
        params_path = Path(cast(str | Path, params_file)).resolve()
        loaded = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"RHIME params file {params_path} must contain one JSON object.")
        params = loaded
    if overrides:
        params.update(overrides)
    return params


def resolve_stage_setup(params: Mapping[str, Any], *, model: ModelKind) -> RhimeRunnerSetup:
    """Resolve stage parameters through the canonical RHIME boundary."""
    return resolve_rhime_options(params=params, multisector=model == "multisector")


def effective_configuration(setup: RhimeRunnerSetup, *, model: ModelKind) -> dict[str, Any]:
    """Return the resolved scientific configuration used by every stage."""
    return {
        "model": model,
        "run_spec": asdict(setup.run_spec),
        "sampler": {name: getattr(setup.sampler, name) for name in setup.sampler.__slots__},
        "preparation": setup.data_args,
    }


def configuration_identity(setup: RhimeRunnerSetup, *, model: ModelKind) -> str:
    """Hash resolved data, period, model, and prior choices."""
    run_spec = setup.run_spec
    identity_configuration = {
        "model": model,
        "preparation": setup.data_args,
        "run": {
            "start_date": run_spec.start_date,
            "end_date": run_spec.end_date,
            "sites": run_spec.sites,
            "averaging_period": run_spec.averaging_period,
            "model": asdict(run_spec.model),
            "split_by_sectors": run_spec.split_by_sectors,
        },
    }
    encoded = json.dumps(
        _json_value(identity_configuration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def prepare_rhime_stage(
    *,
    setup: RhimeRunnerSetup,
    model: ModelKind,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare and persist independently inspectable RHIME inputs."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    multisector = model == "multisector"
    data_args = dict(setup.data_args)
    data_args["save_merged_data"] = False
    if data_args["basis_output_path"] is not None:
        data_args["basis_output_path"] = str(destination / "basis")
    merged = retrieve_or_reload_rhime_data(data_args, multisector=multisector)
    filtered = filter_rhime_observations(merged, data_args)
    missing_sites = [site for site in data_args["sites"] if site not in filtered.sites]
    if missing_sites:
        raise ValueError(
            "RHIME preparation could not produce required site input(s) "
            f"{missing_sites!r} for species {data_args['species']!r} and period "
            f"{data_args['start_date']} to {data_args['end_date']}."
        )
    merged_dir = destination / "merged-data"
    _save_merged_data(filtered.fp_all, merged_dir, merged_data_name="merged-data.nc")
    merged_path = merged_dir / "merged-data.nc"
    basis = build_rhime_basis(filtered, data_args)
    site_data = build_rhime_sensitivities(filtered, basis, data_args, multisector=multisector)
    prepared = assemble_rhime_inputs(filtered, basis, site_data, data_args)
    missing_sites = [site for site in setup.data_args["sites"] if site not in prepared.sites]
    if missing_sites:
        raise ValueError(
            "RHIME preparation could not produce required site input(s) "
            f"{missing_sites!r} for species {setup.data_args['species']!r} and period "
            f"{setup.data_args['start_date']} to {setup.data_args['end_date']}."
        )
    prepared_path = destination / "prepared-inputs.nc"
    prepared.save(prepared_path)
    manifest = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "prepare",
        "configuration_identity": configuration_identity(setup, model=model),
        "effective_configuration": effective_configuration(setup, model=model),
        "artifacts": {
            "merged_data": _artifact_path(merged_path),
            "prepared_inputs": _artifact_path(prepared_path),
        },
        "artifact_identities": {
            "merged_data": _file_identity(merged_path),
            "prepared_inputs": _file_identity(prepared_path),
        },
    }
    manifest_path = _write_json(destination / "prepare-manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_prepared(
    path: str | Path,
    *,
    setup: RhimeRunnerSetup,
    model: ModelKind,
    preparation_manifest: str | Path | None,
) -> tuple[RhimePreparedInputs, RhimeRunnerSetup]:
    if preparation_manifest is not None:
        manifest_path = Path(preparation_manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = configuration_identity(setup, model=model)
        if manifest.get("configuration_identity") != expected:
            raise ValueError(
                "Prepared inputs do not match the effective RHIME configuration: "
                f"manifest has {manifest.get('configuration_identity')!r}, current configuration has {expected!r}."
            )
    prepared = RhimePreparedInputs.load(Path(path).resolve())
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)
    return prepared, RhimeRunnerSetup(run_spec=run_spec, sampler=setup.sampler, data_args=setup.data_args)


def _build_prepared_model(
    prepared: RhimePreparedInputs,
    setup: RhimeRunnerSetup,
    *,
    model: ModelKind,
):
    if model == "multisector":
        names = multisector_model_input_names(prepared, setup.run_spec.model)
        build = build_multisector_rhime_model_result
    else:
        names = standard_model_input_names(prepared, setup.run_spec.model)
        build = build_standard_rhime_model_result
    inputs = materialize_pymc_inputs(prepared, variable_names=names)
    return build(prepared=prepared, model_inputs=inputs, run_spec=setup.run_spec)


def _check_result(
    *,
    name: str,
    status: str,
    measured_values: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    message: str,
    artifact_paths: list[str],
    stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECK_SCHEMA_VERSION,
        "name": name,
        "status": status,
        "producer": "openghg_inversions",
        "measured_values": dict(measured_values),
        "thresholds": dict(thresholds),
        "message": message,
        "artifact_paths": artifact_paths,
        "stage": stage,
    }


def prior_predictive_stage(
    *,
    setup: RhimeRunnerSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path | None = None,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Build the configured graph and check finite prior-predictive draws."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prior_path = destination / "prior-predictive.nc"
    try:
        prepared, resolved = _load_prepared(
            prepared_inputs,
            setup=setup,
            model=model,
            preparation_manifest=preparation_manifest,
        )
        built = _build_prepared_model(prepared, resolved, model=model)
        with built.model:
            prior = pm.sample_prior_predictive(draws, built.model)
        save_inferencedata(prior, prior_path)
        values = [np.asarray(prior[group][name].values) for group in prior.groups() for name in prior[group]]
        non_finite = sum(int(np.size(value) - np.isfinite(value).sum()) for value in values)
        status = "pass" if values and non_finite == 0 else "fail"
        message = (
            f"Prior predictive produced {draws} finite draws."
            if status == "pass"
            else f"Prior predictive contains {non_finite} non-finite values."
        )
        artifacts = [_artifact_path(prior_path)]
    except Exception as exc:
        non_finite = None
        status = "fail"
        message = f"Prior-predictive readiness failed: {type(exc).__name__}: {exc}"
        artifacts = []
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
    setup: RhimeRunnerSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Sample explicitly supplied prepared inputs without running preparation."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prepared, resolved = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    built = _build_prepared_model(prepared, resolved, model=model)
    idata = sample_rhime_model(built, resolved.sampler)
    trace_path = destination / "posterior.nc"
    save_inferencedata(idata, trace_path)
    manifest = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "sample",
        "configuration_identity": configuration_identity(resolved, model=model),
        "artifacts": {
            "posterior": _artifact_path(trace_path),
            "prepared_inputs": _artifact_path(Path(prepared_inputs)),
        },
        "artifact_identities": {
            "posterior": _file_identity(trace_path),
            "prepared_inputs": _file_identity(Path(prepared_inputs).resolve()),
        },
    }
    _write_json(destination / "sample-manifest.json", manifest)
    return manifest


def diagnose_rhime_stage(
    *,
    posterior: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    max_rhat: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_divergences: int = 0,
    stage: str = "posterior",
) -> dict[str, Any]:
    """Calculate the stable posterior convergence check."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    idata = load_inferencedata(Path(posterior).resolve())
    summary = az.summary(idata, kind="diagnostics", fmt="xarray")
    summary_path = destination / "posterior-diagnostics.nc"
    reset_serialisation_multiindexes(summary).to_netcdf(summary_path)

    def finite_extreme(name: str, operation: str) -> tuple[float | None, str | None]:
        candidates = []
        if name in summary:
            candidates.append((name, summary[name]))
        else:
            for variable, values in summary.data_vars.items():
                if "metric" in values.dims and name in values.coords["metric"]:
                    candidates.append((str(variable), values.sel(metric=name, drop=True)))
        best: tuple[float, str] | None = None
        for variable, values in candidates:
            array = np.asarray(values.values, dtype=float)
            finite = np.isfinite(array)
            if not finite.any():
                continue
            masked = np.where(finite, array, -np.inf if operation == "max" else np.inf)
            flat_index = int(masked.argmax() if operation == "max" else masked.argmin())
            index = np.unravel_index(flat_index, array.shape)
            labels = [
                f"{dim}={values.coords[dim].values[position]}"
                for dim, position in zip(values.dims, index)
                if dim in values.coords
            ]
            candidate = (float(array[index]), ",".join([variable, *labels]))
            if best is None or (candidate[0] > best[0] if operation == "max" else candidate[0] < best[0]):
                best = candidate
        return best if best is not None else (None, None)

    rhat, rhat_variable = finite_extreme("r_hat", "max")
    bulk_ess, bulk_ess_variable = finite_extreme("ess_bulk", "min")
    tail_ess, tail_ess_variable = finite_extreme("ess_tail", "min")
    posterior_group = getattr(idata, "posterior", None)
    sample_stats = getattr(idata, "sample_stats", None)
    divergences_by_chain = (
        np.asarray(sample_stats["diverging"].sum("draw").values, dtype=int).tolist()
        if sample_stats is not None and "diverging" in sample_stats
        else None
    )

    measured = {
        "chains": posterior_group.sizes.get("chain") if posterior_group is not None else None,
        "draws_per_chain": posterior_group.sizes.get("draw") if posterior_group is not None else None,
        "max_rhat": rhat,
        "max_rhat_variable": rhat_variable,
        "min_bulk_ess": bulk_ess,
        "min_bulk_ess_variable": bulk_ess_variable,
        "min_tail_ess": tail_ess,
        "min_tail_ess_variable": tail_ess_variable,
        "divergences": sum(divergences_by_chain) if divergences_by_chain is not None else None,
        "divergences_by_chain": divergences_by_chain,
    }
    thresholds = {
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "divergences": max_divergences,
    }
    assessed_names = ("max_rhat", "min_bulk_ess", "min_tail_ess", "divergences")
    missing = [name for name in assessed_names if measured[name] is None]
    failed = (
        (measured["max_rhat"] is not None and measured["max_rhat"] > max_rhat)
        or (measured["min_bulk_ess"] is not None and measured["min_bulk_ess"] < min_bulk_ess)
        or (measured["min_tail_ess"] is not None and measured["min_tail_ess"] < min_tail_ess)
        or (measured["divergences"] is not None and measured["divergences"] > max_divergences)
    )
    status = "fail" if failed else "unknown" if missing else "pass"
    message = (
        f"Convergence metrics unavailable: {', '.join(missing)}."
        if missing
        else "Posterior convergence thresholds were exceeded."
        if failed
        else "Posterior convergence thresholds were met."
    )
    result = _check_result(
        name=CONVERGENCE_CHECK_NAME,
        status=status,
        measured_values=measured,
        thresholds=thresholds,
        message=message,
        artifact_paths=[_artifact_path(summary_path)],
        stage=stage,
    )
    _write_json(_output_path(destination, check_output, "sampler-convergence.json"), result)
    return result


def postprocess_rhime_stage(
    *,
    setup: RhimeRunnerSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path | None = None,
) -> RhimeResult:
    """Build requested products from explicit prepared and posterior inputs."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prepared, resolved = _load_prepared(
        prepared_inputs,
        setup=setup,
        model=model,
        preparation_manifest=preparation_manifest,
    )
    configured_output = resolved.run_spec.output
    output_spec = replace(
        configured_output,
        output_path=str(destination),
        save_trace=False,
        save_inversion_output=bool(configured_output.save_inversion_output),
    )
    run_spec = replace(resolved.run_spec, output=output_spec)
    resolved = RhimeRunnerSetup(run_spec=run_spec, sampler=resolved.sampler, data_args=resolved.data_args)
    built = _build_prepared_model(prepared, resolved, model=model)
    idata = load_inferencedata(Path(posterior).resolve())
    if model == "multisector":
        result = make_multisector_rhime_result(
            prepared=prepared,
            run_spec=run_spec,
            sampler=resolved.sampler,
            model_build_result=built,
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
        basic_path = destination / "basic.nc"
        reset_serialisation_multiindexes(basic).to_netcdf(basic_path)
        artifacts["basic_path"] = str(basic_path)
    manifest = {
        "schema_version": 1,
        "producer": "openghg_inversions",
        "stage": "postprocess",
        "configuration_identity": configuration_identity(resolved, model=model),
        "artifacts": {name: _artifact_path(Path(path)) for name, path in artifacts.items()},
    }
    manifest_path = _write_json(destination / "postprocess-manifest.json", manifest)
    result.output_metadata["postprocess_manifest_path"] = str(manifest_path)
    return result
