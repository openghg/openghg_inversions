"""Configuration loading and explicit standard/multisector identity projection."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, cast
from .params import StandardRecipeConfig, params_from_config
from ._stage_artifacts import json_value as _json_value
from .specs import AdditiveSigmaSettings, PollutionEventSettings
ModelKind = Literal["standard", "multisector"]

_STAGE_PATH_OPTIONS = frozenset(
    {
        "basis_directory",
        "bc_basis_directory",
        "country_directory",
        "country_file",
        "merged_data_dir",
    }
)
# These fields define identity version 1; extending runtime records does not change it.
_PREPARATION_IDENTITY_FIELDS = (
    "averaging_error", "averaging_period", "basis_algorithm", "basis_directory",
    "bc_basis_case", "bc_basis_directory", "bc_freq", "bc_input", "bc_store",
    "calibration_scale", "country_directory", "domain", "emissions_domain",
    "emissions_store", "end_date", "filters", "fix_basis_outer_regions",
    "flux_non_finite_check", "flux_sources", "footprint_store", "fp_basis_case",
    "fp_height", "fp_model", "fp_species", "inlet", "instrument", "max_level",
    "met_model", "min_error", "min_error_options", "nbasis", "obs_data_level",
    "obs_store", "outer_regions_path", "platform", "sites", "species",
    "split_by_sectors", "start_date", "time_resolved", "use_bc", "use_tracer",
)
_MODEL_IDENTITY_FIELDS = (
    "species", "domain", "use_bc", "add_offset", "aggregation_error_mode",
    "bc_prior", "offset_prior", "offset_args",
)
_ACTIVITY_IDENTITY_FIELDS = ("active", "fixed_value", "fixed_groups", "group_coord")


def _activity_identity(activity: Any) -> dict[str, Any] | None:
    return None if activity is None else {
        name: getattr(activity, name) for name in _ACTIVITY_IDENTITY_FIELDS
    }


def _model_identity(model: Any) -> dict[str, Any]:
    """Encode named scientific choices independently of dataclass layout."""
    identity = {name: getattr(model, name) for name in _MODEL_IDENTITY_FIELDS}
    identity["sectors"] = [
        {
            "name": sector.name, "flux_source": sector.flux_source,
            "x_prior": sector.x_prior, "variable_suffix": sector.variable_suffix,
            "state_activity": _activity_identity(sector.state_activity),
        }
        for sector in model.sectors
    ]
    for name in ("bc_state_activity", "state_activity"):
        identity[name] = _activity_identity(getattr(model, name))
    likelihood = model.likelihood
    # Explicit setting names preserve the contract when runtime records grow.
    identity["likelihood"] = None if likelihood is None else {
        "kind": (
            "pollution_event" if isinstance(likelihood, PollutionEventSettings)
            else "additive_sigma" if isinstance(likelihood, AdditiveSigmaSettings)
            else "fixed_error"
        ),
        **{
            name: getattr(likelihood, name)
            for name in (
                "sigma_prior", "sigma_freq", "sigma_per_site", "sigma_freq_anchor",
                "pollution_events_from_obs", "power", "use_minimum_error_floor",
            )
            if hasattr(likelihood, name)
        },
    }
    return identity


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
        params = params_from_config(source_path, normalise=False)
    else:
        source_path = Path(cast(str | Path, params_file)).resolve()
        loaded = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"RHIME params file {source_path} must contain one JSON object.")
        params = loaded
    if overrides:
        params.update(overrides)
    return _resolve_stage_paths(params, base_dir=source_path.parent)


def effective_configuration(setup: StandardRecipeConfig, *, model: ModelKind) -> dict[str, Any]:
    """Return the resolved scientific configuration used by every stage."""
    return {
        "model": model,
        "run_spec": asdict(setup.run_spec),
        "sampler": setup.sampler_options.as_dict(),
        "preparation": setup.data_args,
    }


def configuration_identity(setup: StandardRecipeConfig, *, model: ModelKind) -> str:
    """Hash resolved data, period, model, and prior choices."""
    run_spec = setup.run_spec
    data_args = setup.data_args
    preparation = {name: data_args[name] for name in _PREPARATION_IDENTITY_FIELDS if name in data_args}
    if "sites" in preparation:
        preparation["sites"] = [str(site).upper() for site in preparation["sites"]]
    identity_configuration = {
        "model": model,
        "preparation": preparation,
        "run": {
            "start_date": run_spec.start_date,
            "end_date": run_spec.end_date,
            "sites": tuple(site.upper() for site in run_spec.sites),
            "averaging_period": run_spec.averaging_period,
            "model": _model_identity(run_spec.model),
            "split_by_sectors": run_spec.split_by_sectors,
        },
    }
    encoded = json.dumps(
        _json_value(identity_configuration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"

