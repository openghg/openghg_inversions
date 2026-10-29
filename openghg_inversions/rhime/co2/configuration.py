"""TOML loading and semantic configuration for the concrete CO2 recipes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from importlib.resources import files
from importlib.resources.abc import Traversable
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any
import tomllib


from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.priors import positive_prior_args
from openghg_inversions.models.scalar_sigma import add_scalar_sigma_eigen_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import normalise_optional_mapping

from .co2_cached_sigma_runner import validate_cached_co2_sampler
from .inference import uses_cached_ou


def _frozen(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _frozen(value) if isinstance(value, Mapping) else value for key, value in values.items()}
    )


def _table(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a table.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings.")
    return dict(value)


def _take(options: dict[str, object], key: str, path: str) -> object:
    try:
        return options.pop(key)
    except KeyError as exc:
        raise ValueError(f"Missing required option {path}.{key}.") from exc


def _reject_unknown(options: Mapping[str, object], path: str) -> None:
    if options:
        names = ", ".join(f"{path}.{name}" for name in sorted(options))
        raise ValueError(f"Unknown configuration option(s): {names}.")



# Names are checked for every phase; their values remain owned by the consuming parser.
_CO2_REQUEST_OPTION_NAMES = frozenset({
    "format_version", "recipe", "prepared_inputs", "model", "likelihood", "sampling", "outputs",
})
_PREPARATION_OPTION_NAMES = frozenset({"path", "reconstruction_path", "start_date", "end_date", "domain"})
_SAMPLING_OPTION_NAMES = frozenset({
    "draws", "burn", "tune", "chains", "nuts_sampler", "progressbar", "sample_prior_predictive",
    "sample_posterior_predictive", "target_accept", "random_seed",
})
_LIKELIHOOD_OPTION_NAMES = frozenset({
    "kind", "no_model_error", "sigma_prior", "fixed_model_mismatch", "fixed_site_amplitudes",
    "site_amplitude_prior", "tau_hours", "initial_site_amplitudes", "eigenbasis_path",
})
_PRIOR_OPTION_NAMES = frozenset({
    "pdf", "mu", "sigma", "lower", "upper", "nu", "alpha", "beta", "lam", "mean", "stdev", "reparameterise",
})
_MODEL_SECTION_OPTION_NAMES = {
    "boundary": frozenset({"enabled", "prior", "frequency", "anchor_time", "mean_shift_prior", "anomaly_scale"}),
    "boundary.mean_shift_prior": _PRIOR_OPTION_NAMES,
    "boundary.prior": _PRIOR_OPTION_NAMES,
    "offset": frozenset({"prior", "frequency", "per_site", "drop_first"}),
    "offset.prior": _PRIOR_OPTION_NAMES,
}


def _validate_section_option_names(
    requests: Iterable[Mapping[str, object]], sections: Mapping[str, frozenset[str]],
) -> None:
    """Check source/override names once without resolving any section values."""
    unknown: dict[str, object] = {}
    for config in requests:
        for path, allowed in sections.items():
            section: object = config
            for part in path.split(".") if path else ():
                section = section.get(part) if isinstance(section, Mapping) else None
            if isinstance(section, Mapping):
                prefix = f"{path}." if path else ""
                unknown.update({f"{prefix}{name}": value for name, value in section.items() if name not in allowed})
    _reject_unknown(unknown, "config")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value.strip()


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path} must be numeric.")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{path} must be {qualifier}.")
    return result


def _number_or_mapping(
    value: object,
    path: str,
    *,
    positive: bool,
    nonnegative: bool = False,
) -> float | Mapping[str, float]:
    def checked(item: object, item_path: str) -> float:
        result = _number(item, item_path, positive=positive)
        if nonnegative and result < 0.0:
            raise ValueError(f"{item_path} must be finite and non-negative.")
        return result

    if isinstance(value, Mapping):
        table = _table(value, path)
        if not table:
            raise ValueError(f"{path} must not be empty.")
        return MappingProxyType(
            {_string(key, f"{path} key"): checked(item, f"{path}.{key}") for key, item in table.items()}
        )
    return checked(value, path)


def _prior(value: object, path: str, *, positive: bool = False) -> Mapping[str, Any]:
    prior = _table(value, path)
    pdf = _string(_take(prior, "pdf", path), f"{path}.pdf").casefold().replace("-", "")
    required = {
        "normal": ("mu", "sigma"),
        "truncatednormal": ("mu", "sigma", "lower"),
        "halfnormal": ("sigma",),
        "halfstudentt": ("nu", "sigma"),
        "gamma": ("alpha", "beta"),
        "exponential": ("lam",),
        "uniform": ("lower", "upper"),
    }
    result: dict[str, Any] = {"pdf": pdf}
    if pdf == "lognormal":
        parameter_keys = {"mu", "sigma"}
        moment_keys = {"mean", "stdev"}
        if parameter_keys & prior.keys() and moment_keys & prior.keys():
            raise ValueError(f"{path} must use either mu/sigma or mean/stdev for a LogNormal prior.")
        names = ("mean", "stdev") if moment_keys & prior.keys() else ("mu", "sigma")
        for name in names:
            result[name] = _number(
                _take(prior, name, path),
                f"{path}.{name}",
                positive=name in {"mean", "stdev", "sigma"},
            )
        if "reparameterise" in prior:
            result["reparameterise"] = _bool(prior.pop("reparameterise"), f"{path}.reparameterise")
    elif pdf not in required:
        raise ValueError(f"{path}.pdf is not a supported configuration prior.")
    else:
        for name in required[pdf]:
            result[name] = _number(
                _take(prior, name, path),
                f"{path}.{name}",
                positive=name in {"sigma", "nu", "alpha", "beta", "lam"},
            )
    if pdf == "truncatednormal" and "upper" in prior:
        result["upper"] = _number(prior.pop("upper"), f"{path}.upper")
    _reject_unknown(prior, path)
    if pdf in {"uniform", "truncatednormal"} and "upper" in result:
        if result["lower"] >= result["upper"]:
            raise ValueError(f"{path}.lower must be less than {path}.upper.")
    if positive:
        result = positive_prior_args(result)
    return _frozen(result)


def validate_co2_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Validate the CO2 file envelope without resolving its sections.

    Missing headers select CO2 for phase-only requests. Component
    parsers own scientific values and resource paths. The request loader checks
    section names; this envelope check leaves inactive sections unresolved.
    """
    options = _table(config, "config")
    version = options.get("format_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("config.format_version must be the integer 1.")
    if options.get("recipe", "co2") != "co2":
        raise ValueError("config.recipe must be 'co2' for CO2 execution.")
    _reject_unknown({name: value for name, value in options.items() if name not in _CO2_REQUEST_OPTION_NAMES}, "config")
    return options


def _validate_co2_option_names(*requests: Mapping[str, object]) -> None:
    """Check request names independently of header values and winning overrides."""
    from .co2_outputs import Co2OutputOptions
    from openghg_inversions.postprocessing.paris_options import ParisOptions

    _validate_section_option_names(requests, {
        "": _CO2_REQUEST_OPTION_NAMES,
        "prepared_inputs": _PREPARATION_OPTION_NAMES,
        "model": frozenset({"boundary", "offset"}),
        **{f"model.{path}": names for path, names in _MODEL_SECTION_OPTION_NAMES.items()},
        "likelihood": _LIKELIHOOD_OPTION_NAMES,
        "likelihood.sigma_prior": _PRIOR_OPTION_NAMES,
        "likelihood.site_amplitude_prior": _PRIOR_OPTION_NAMES,
        "sampling": _SAMPLING_OPTION_NAMES | {"sigma_target_accept", "state_target_accept"},
        "outputs": frozenset(Co2OutputOptions.__dataclass_fields__) | {"source_to_sector"},
        "outputs.paris_postprocessing_kwargs": frozenset(ParisOptions.__dataclass_fields__),
    })


@dataclass(frozen=True)
class Co2PreparationConfig:
    """Artifact selection and original requested context, without scientific I/O.

    Paths resolve from the invocation directory. The affine companion is an
    association of the preparation, independent of current output choices.
    """

    prepared_path: Path | None = None
    affine_path: Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    domain: str | None = None

    @classmethod
    def from_params(cls, config: Mapping[str, object], *, require_path: bool = False) -> Co2PreparationConfig:
        """Parse only prepared-input options; optionally require an import path."""
        options = _table(config, "config")
        prepared = _table(options.get("prepared_inputs", {}), "prepared_inputs")
        result: dict[str, Any] = {
            "prepared_path": None, "affine_path": None,
            "start_date": None, "end_date": None, "domain": None,
        }
        if require_path and "path" not in prepared:
            raise ValueError("Missing required option prepared_inputs.path.")
        for name, argument in (("path", "prepared_path"), ("reconstruction_path", "affine_path")):
            if name in prepared:
                value = prepared.pop(name)
                if not isinstance(value, str | Path):
                    raise TypeError(f"prepared_inputs.{name} must be a filesystem path.")
                if isinstance(value, str):
                    value = _string(value, f"prepared_inputs.{name}")
                result[argument] = Path(value).expanduser().resolve()
        for name in ("start_date", "end_date", "domain"):
            if name in prepared:
                result[name] = _string(prepared.pop(name), f"prepared_inputs.{name}")
        _reject_unknown(prepared, "prepared_inputs")
        return cls(**result)

    def preparation_arguments(self) -> dict[str, Any]:
        """Return explicit keywords for the saved preparation boundary."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _random_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("sampling.random_seed must be a non-negative integer.")
    return value


def resolve_co2_prior_seed(config: Mapping[str, object]) -> int | None:
    """Resolve only the random seed consumed by prior predictive sampling.

    MCMC options are unused by prior prediction and remain unvalidated here.
    """
    options = _table(_table(config, "config").get("sampling", {}), "sampling")
    return _random_seed(options["random_seed"]) if "random_seed" in options else None


def _sampling(value: object) -> RhimeSampler:
    options = _table(value, "sampling")
    kwargs: dict[str, Any] = {}
    for name in ("draws", "burn", "tune", "chains"):
        if name not in options:
            continue
        raw = options.pop(name)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"sampling.{name} must be an integer.")
        minimum = 0 if name in ("burn", "tune") else 1
        if raw < minimum:
            raise ValueError(f"sampling.{name} must be >= {minimum}.")
        kwargs[name] = raw
    if "nuts_sampler" in options:
        kwargs["nuts_sampler"] = _string(options.pop("nuts_sampler"), "sampling.nuts_sampler")
    if "progressbar" in options:
        kwargs["progressbar"] = _bool(options.pop("progressbar"), "sampling.progressbar")
    if "sample_prior_predictive" in options:
        prior_predictive = options.pop("sample_prior_predictive")
        if not isinstance(prior_predictive, bool | int) or (
            isinstance(prior_predictive, int)
            and not isinstance(prior_predictive, bool)
            and prior_predictive < 0
        ):
            raise TypeError("sampling.sample_prior_predictive must be a boolean or non-negative integer.")
        kwargs["sample_prior_predictive"] = prior_predictive
    if "sample_posterior_predictive" in options:
        posterior = options.pop("sample_posterior_predictive")
        if isinstance(posterior, bool):
            kwargs["sample_posterior_predictive"] = posterior
        elif isinstance(posterior, Sequence) and not isinstance(posterior, str | bytes):
            kwargs["sample_posterior_predictive"] = tuple(
                _string(item, "sampling.sample_posterior_predictive item") for item in posterior
            )
        else:
            raise TypeError(
                "sampling.sample_posterior_predictive must be a boolean or list of variable names."
            )

    sample_kwargs: dict[str, Any] = {}
    if "target_accept" in options:
        target_accept = _number(options.pop("target_accept"), "sampling.target_accept")
        if not 0.0 < target_accept < 1.0:
            raise ValueError("sampling.target_accept must be between zero and one.")
        sample_kwargs["target_accept"] = target_accept
    posterior_kwargs: dict[str, Any] = {}
    if "random_seed" in options:
        random_seed = _random_seed(options.pop("random_seed"))
        sample_kwargs["random_seed"] = random_seed
        posterior_kwargs["random_seed"] = random_seed
    _reject_unknown(options, "sampling")
    if sample_kwargs:
        kwargs["sample_kwargs"] = sample_kwargs
    if posterior_kwargs:
        kwargs["posterior_predictive_kwargs"] = posterior_kwargs
    sampler = RhimeSampler(**kwargs)
    if sampler.burn >= sampler.draws:
        raise ValueError("sampling.burn must be less than sampling.draws.")
    return sampler


def _model(value: object) -> dict[str, Any]:
    options = _table(value, "model")
    result: dict[str, Any] = {}
    if "boundary" in options:
        boundary = _table(options.pop("boundary"), "model.boundary")
        enabled = _bool(boundary.pop("enabled", True), "model.boundary.enabled")
        prior = (
            _prior(boundary.pop("prior"), "model.boundary.prior")
            if "prior" in boundary
            else None
        )
        mean_shift_prior = (
            _prior(boundary.pop("mean_shift_prior"), "model.boundary.mean_shift_prior")
            if "mean_shift_prior" in boundary
            else None
        )
        anomaly_scale = (
            _number(boundary.pop("anomaly_scale"), "model.boundary.anomaly_scale", positive=True)
            if "anomaly_scale" in boundary else None
        )
        if (mean_shift_prior is None) != (anomaly_scale is None):
            raise ValueError("model.boundary.mean_shift_prior and anomaly_scale must be supplied together.")
        if mean_shift_prior is not None and prior is not None:
            raise ValueError("Centred boundary corrections cannot use model.boundary.prior.")
        periods = {runner: boundary.pop(name) for name, runner in
                   (("frequency", "bc_freq"), ("anchor_time", "bc_anchor_time")) if name in boundary}
        _reject_unknown(boundary, "model.boundary")
        if mean_shift_prior is not None:
            if not enabled:
                raise ValueError("model.boundary.mean_shift_prior requires model.boundary.enabled=true.")
            result["bc_mean_shift_prior"] = mean_shift_prior
            result["bc_anomaly_scale"] = anomaly_scale
        if not enabled and prior is not None:
            raise ValueError("model.boundary.prior requires model.boundary.enabled=true.")
        result["use_bc"] = enabled
        if prior is not None:
            result["bc_prior"] = prior
        if not enabled and any(value is not None for value in periods.values()):
            raise ValueError("Boundary frequency and anchor_time require enabled=true.")
        result.update(periods)
    if "offset" in options:
        offset = _table(options.pop("offset"), "model.offset")
        result["offset_prior"] = _prior(_take(offset, "prior", "model.offset"), "model.offset.prior")
        offset_args: dict[str, Any] = {}
        if "frequency" in offset:
            frequency = offset.pop("frequency")
            if frequency is not None:
                frequency = _string(frequency, "model.offset.frequency")
            offset_args["offset_freq"] = frequency
        for config_name, runner_name in (("per_site", "per_site"), ("drop_first", "drop_first")):
            if config_name in offset:
                offset_args[runner_name] = _bool(offset.pop(config_name), f"model.offset.{config_name}")
        _reject_unknown(offset, "model.offset")
        if offset_args.get("per_site") is False:
            if offset_args.get("offset_freq") is not None:
                raise ValueError("A global model.offset does not accept frequency.")
            if offset_args.get("drop_first") is True:
                raise ValueError("A global model.offset does not support drop_first=true.")
        result["offset_args"] = _frozen(offset_args)
    _reject_unknown(options, "model")
    return result


def _co2_likelihood(value: object, *, matched_ou: bool = False) -> dict[str, Any]:
    options = _table(value, "likelihood")
    kind = _string(_take(options, "kind", "likelihood"), "likelihood.kind")
    result: dict[str, Any] = {}
    if kind == "additive_sigma":
        no_model_error = _bool(options.pop("no_model_error", False), "likelihood.no_model_error")
        result["no_model_error"] = no_model_error
        if "sigma_prior" in options:
            if no_model_error:
                raise ValueError("likelihood.sigma_prior cannot be combined with no_model_error=true.")
            result["sigma_prior"] = _prior(
                options.pop("sigma_prior"), "likelihood.sigma_prior", positive=True
            )
        if "fixed_model_mismatch" in options:
            fixed = _number(
                options.pop("fixed_model_mismatch"),
                "likelihood.fixed_model_mismatch",
            )
            if fixed < 0.0:
                raise ValueError("likelihood.fixed_model_mismatch must be non-negative.")
            result["fixed_model_mismatch"] = fixed
    elif kind == "site_sigma":
        kwargs: dict[str, Any] = {}
        if "fixed_site_amplitudes" in options:
            fixed_site_amplitudes = options.pop("fixed_site_amplitudes")
            if not isinstance(fixed_site_amplitudes, Mapping):
                raise TypeError("likelihood.fixed_site_amplitudes must be a site mapping.")
            kwargs["fixed_site_amplitudes"] = _number_or_mapping(
                fixed_site_amplitudes,
                "likelihood.fixed_site_amplitudes",
                positive=False,
                nonnegative=True,
            )
        if "site_amplitude_prior" in options:
            kwargs["site_amplitude_prior"] = _prior(
                options.pop("site_amplitude_prior"),
                "likelihood.site_amplitude_prior",
                positive=True,
            )
        if len(kwargs) != 1:
            raise ValueError(
                "site_sigma requires exactly one of fixed_site_amplitudes or site_amplitude_prior."
            )
        result.update(
            likelihood_builder=add_site_sigma_gaussian_likelihood,
            likelihood_kwargs=_frozen(kwargs),
        )
    elif kind == "fixed_ou":
        kwargs = {
            "tau_hours": _number_or_mapping(
                _take(options, "tau_hours", "likelihood"),
                "likelihood.tau_hours",
                positive=True,
            )
        }
        if "fixed_site_amplitudes" in options:
            kwargs["fixed_site_amplitudes"] = _number_or_mapping(
                options.pop("fixed_site_amplitudes"),
                "likelihood.fixed_site_amplitudes",
                positive=False,
                nonnegative=True,
            )
        if "site_amplitude_prior" in options:
            kwargs["site_amplitude_prior"] = _prior(
                options.pop("site_amplitude_prior"),
                "likelihood.site_amplitude_prior",
                positive=True,
            )
        if len(kwargs) != 2:
            raise ValueError(
                "fixed_ou requires exactly one of fixed_site_amplitudes or site_amplitude_prior."
            )
        if matched_ou and "site_amplitude_prior" in kwargs and "initial_site_amplitudes" in options:
            kwargs["initial_site_amplitudes"] = _number_or_mapping(
                options.pop("initial_site_amplitudes"), "likelihood.initial_site_amplitudes", positive=True,
            )
        result.update(
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs=_frozen(kwargs),
        )
    elif kind == "scalar_sigma":
        result.update(
            likelihood_builder=add_scalar_sigma_eigen_likelihood,
            likelihood_kwargs=_frozen(
                {
                    "eigenbasis_path": Path(
                        _string(
                            _take(options, "eigenbasis_path", "likelihood"),
                            "likelihood.eigenbasis_path",
                        )
                    ),
                    "sigma_prior": _prior(
                        _take(options, "sigma_prior", "likelihood"),
                        "likelihood.sigma_prior",
                        positive=True,
                    ),
                }
            ),
        )
    else:
        raise ValueError(
            "likelihood.kind must be one of 'additive_sigma', 'site_sigma', 'fixed_ou', or 'scalar_sigma'."
        )
    _reject_unknown(options, "likelihood")
    return result


def load_co2_family_config(path: str | Path | Traversable) -> Mapping[str, object]:
    """Load one CO2-family TOML file without applying recipe semantics."""
    resource = Path(path) if isinstance(path, str | Path) else path
    with resource.open("rb") as config_file:
        return tomllib.load(config_file)


def co2_config_templates() -> Mapping[str, Traversable]:
    """Return installed CO2-family TOML templates keyed by file name."""
    directory = files("openghg_inversions.rhime").joinpath("config")
    return MappingProxyType(
        {name: directory.joinpath(name) for name in ("co2.toml", "co2_fixed_ou.toml", "co2_o2.toml")}
    )


def _validate_boundary_periods(frequency: str | None, anchor_time: str | None) -> None:
    """Validate model-owned period syntax without touching prepared arrays."""
    from openghg_inversions.boundary_sensitivity import validate_centered_boundary_periods

    if anchor_time is not None:
        _string(anchor_time, "model.boundary.anchor_time")
    validate_centered_boundary_periods(frequency, anchor_time)


@dataclass(frozen=True)
class Co2ModelConfig:
    """Scientific choices for ordinary CO2, independent of arrays and inference.

    Construction through :meth:`from_params` validates only model and likelihood
    sections. Ordinary containers are owned; opaque numerical values are borrowed.
    Defaults omitted here remain owned by the concrete scientific recipe.
    Boundary frequency and anchor choices apply to raw ordinary or centered operands
    and bind when inference consumes the preparation.
    """

    use_bc: bool | None = None
    bc_prior: Mapping[str, Any] | None = None
    bc_mean_shift_prior: Mapping[str, Any] | None = None
    bc_anomaly_scale: float | None = None
    offset_prior: Mapping[str, Any] | None = None
    offset_args: Mapping[str, Any] | None = None
    no_model_error: bool | None = None
    sigma_prior: Mapping[str, Any] | None = None
    fixed_model_mismatch: float | None = None
    likelihood_builder: Callable[..., Any] | None = None
    likelihood_kwargs: Mapping[str, Any] | None = None
    bc_freq: str | None = None
    bc_anchor_time: str | None = None

    def __post_init__(self) -> None:
        _validate_boundary_periods(self.bc_freq, self.bc_anchor_time)
        if self.use_bc is False and (self.bc_freq is not None or self.bc_anchor_time is not None):
            raise ValueError("Boundary frequency and anchor_time require enabled=true.")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                object.__setattr__(self, item.name, normalise_optional_mapping(value))

    @classmethod
    def from_params(cls, config: Mapping[str, object]) -> Co2ModelConfig:
        """Parse active science without opening artifacts or creating a sampler."""
        options = _table(config, "config")
        model = _model(options.get("model", {}))
        likelihood = _co2_likelihood(_take(options, "likelihood", "config"), matched_ou=True)
        kwargs = likelihood.get("likelihood_kwargs", {})
        if "eigenbasis_path" in kwargs:
            likelihood["likelihood_kwargs"] = _frozen({
                **kwargs, "eigenbasis_path": kwargs["eigenbasis_path"].expanduser().resolve(),
            })
        return cls(**model, **likelihood)

    def model_arguments(self) -> dict[str, Any]:
        """Bind these choices as keywords at the ordinary scientific boundary."""
        return {item.name: getattr(self, item.name) for item in fields(self) if getattr(self, item.name) is not None}


@dataclass(frozen=True)
class _Co2Sampling:
    """Resolved sampler and the matched OU transition tuning controls."""

    sampler: RhimeSampler
    sigma_target_accept: float = 0.8
    state_target_accept: float = 0.9


def _resolve_co2_sampling(config: Mapping[str, object], *, model: Mapping[str, Any]) -> _Co2Sampling:
    """Resolve sampling options after the scientific model has selected its graph.

    The configured backend is selected from resolved scientific choices.
    Additive sigma uses NumPyro; supported custom likelihoods use PyMC. Inferred
    OU amplitudes use the matched PyMC transition and its two explicit controls.

    Returns:
        object: Resolved sampler and matched OU tuning controls, with attributes
            ``sampler``, ``sigma_target_accept``, and ``state_target_accept``.
    """
    options = _table(config, "config")
    sampling = _table(options.get("sampling", {}), "sampling")
    if "nuts_sampler" in sampling:
        raise ValueError("Configured CO2 selects its sampler from the likelihood; remove sampling.nuts_sampler.")
    cached = uses_cached_ou(model)
    targets = {}
    for name, default in (("sigma_target_accept", 0.8), ("state_target_accept", 0.9)):
        if name in sampling and not cached:
            raise ValueError(f"sampling.{name} requires inferred fixed-OU amplitudes.")
        target = _number(sampling.pop(name, default), f"sampling.{name}")
        if not 0.0 < target < 1.0:
            raise ValueError(f"sampling.{name} must be between zero and one.")
        targets[name] = target
    sampling["nuts_sampler"] = "pymc" if model.get("likelihood_builder") is not None else "numpyro"
    sampler = _sampling(sampling)
    if cached:
        validate_cached_co2_sampler(sampler)
    return _Co2Sampling(
        sampler=sampler,
        sigma_target_accept=targets["sigma_target_accept"], state_target_accept=targets["state_target_accept"],
    )


__all__ = [
    "Co2PreparationConfig", "Co2ModelConfig", "co2_config_templates",
    "load_co2_family_config", "resolve_co2_prior_seed", "validate_co2_config",
]
