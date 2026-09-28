"""TOML loading and semantic configuration for the concrete CO2 recipes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
import tomllib

import numpy as np

from openghg_inversions.models import StateActivity
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.priors import positive_prior_args
from openghg_inversions.models.scalar_sigma import add_scalar_sigma_eigen_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.rhime.sampling import RhimeSampler

from .co2_cached_sigma_runner import run_rhime_co2_cached_sigma
from .co2_o2_cached_sigma_runner import run_rhime_co2_o2_cached_sigma_from_prepared_inputs
from .co2_o2_preparation import Co2O2PreparedInputs
from .co2_o2_runner import run_rhime_co2_o2_from_prepared_inputs
from .co2_preparation import Co2PreparedInputs
from .co2_runner import run_rhime_co2


Runner = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class Co2RunSetup:
    """Resolved execution setup for CO2 prepared-input replay.

    Attributes:
        preparation_kwargs: Artifact-location values resolved from the
            configuration. ``path`` is suitable for
            :meth:`Co2PreparedInputs.load`.
        runner: Selected public prepared-input runner.
        runner_kwargs: Validated scientific arguments for ``runner``.
        sampler: Validated sampling configuration for ``runner``.
    """

    preparation_kwargs: Mapping[str, object]
    runner: Runner
    runner_kwargs: Mapping[str, object]
    sampler: RhimeSampler

    def runner_arguments(self, prepared_inputs: Co2PreparedInputs) -> Mapping[str, object]:
        """Bind a prepared artifact to the selected runner.

        Args:
            prepared_inputs: In-memory CO2 artifact to replay.

        Returns:
            Exact keyword arguments for :attr:`runner`.
        """
        return _frozen(
            {
                "prepared_inputs": prepared_inputs,
                **self.runner_kwargs,
                "sampler": self.sampler,
            }
        )


@dataclass(frozen=True, slots=True)
class Co2O2RunSetup:
    """Resolved execution setup for linked CO2/O2 prepared-input replay.

    Attributes:
        preparation_kwargs: Validated ``co2_units`` and ``o2_units`` arguments
            for :func:`prepare_co2_o2_inputs`.
        runner: Selected public linked prepared-input runner.
        runner_kwargs: Validated channel error settings used when binding the
            prepared observations.
        sampler: Validated sampling configuration for ``runner``.
    """

    preparation_kwargs: Mapping[str, object]
    runner: Runner
    runner_kwargs: Mapping[str, object]
    sampler: RhimeSampler

    def runner_arguments(self, prepared_inputs: Co2O2PreparedInputs) -> Mapping[str, object]:
        """Bind configured channel errors to the joint observation axis.

        Args:
            prepared_inputs: In-memory linked artifact whose observations have
                observation-aligned ``species`` and ``observation_units``
                coordinates. Species must contain exactly ``co2`` and ``o2``;
                their unit labels must match the resolved configuration.

        Returns:
            Exact keyword arguments for :attr:`runner`, including a labelled
            ``independent_error_sd`` array expanded over the observation axis.

        Raises:
            ValueError: If the required coordinates are absent or misaligned,
                the species are not exactly CO2 and O2, or configured and
                prepared unit labels differ.
        """
        observations = prepared_inputs.observations
        if "species" not in observations.coords or observations["species"].dims != ("observation",):
            raise ValueError(
                "Linked prepared observations require an observation-aligned species coordinate."
            )
        if "observation_units" not in observations.coords or observations["observation_units"].dims != (
            "observation",
        ):
            raise ValueError("Linked prepared observations require observation-aligned observation_units.")
        configured_errors = self.runner_kwargs["independent_error_sd"]
        assert isinstance(configured_errors, Mapping)
        species = np.asarray(observations["species"].values).astype(str)
        observation_units = np.asarray(observations["observation_units"].values).astype(str)
        if set(species) != {"co2", "o2"}:
            raise ValueError("Linked prepared observations must contain exactly CO2 and O2 channels.")
        error_values = np.empty(observations.size, dtype=np.float64)
        for name in ("co2", "o2"):
            selected = species == name
            if not np.all(observation_units[selected] == self.preparation_kwargs[f"{name}_units"]):
                raise ValueError(f"Configured channels.{name}.units do not match prepared observation_units.")
            error_values[selected] = configured_errors[name]
        independent_error = observations.copy(data=error_values).rename("independent_error_sd")
        independent_error.attrs = {"units": "mixed; see observation_units coordinate"}
        return _frozen(
            {
                "prepared_inputs": prepared_inputs,
                **self.runner_kwargs,
                "independent_error_sd": independent_error,
                "sampler": self.sampler,
            }
        )


def _frozen(values: Mapping[str, object]) -> Mapping[str, object]:
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


def _prior(value: object, path: str, *, positive: bool = False) -> Mapping[str, object]:
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
    result: dict[str, object] = {"pdf": pdf}
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


def _prepared_inputs(options: dict[str, object]) -> Mapping[str, object]:
    prepared = _table(_take(options, "prepared_inputs", "config"), "prepared_inputs")
    path = Path(_string(_take(prepared, "path", "prepared_inputs"), "prepared_inputs.path"))
    _reject_unknown(prepared, "prepared_inputs")
    return _frozen({"path": path})


def _sampling(value: object, *, cached: bool) -> RhimeSampler:
    options = _table(value, "sampling")
    kwargs: dict[str, object] = {}
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

    sample_kwargs: dict[str, object] = {}
    if "target_accept" in options:
        if cached:
            raise ValueError(
                "sampling.target_accept is not valid for cached_fixed_ou; use "
                "likelihood.sigma_target_accept and likelihood.state_target_accept."
            )
        target_accept = _number(options.pop("target_accept"), "sampling.target_accept")
        if not 0.0 < target_accept < 1.0:
            raise ValueError("sampling.target_accept must be between zero and one.")
        sample_kwargs["target_accept"] = target_accept
    posterior_kwargs: dict[str, object] = {}
    if "random_seed" in options:
        random_seed = options.pop("random_seed")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("sampling.random_seed must be a non-negative integer.")
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
    if cached and sampler.nuts_sampler != "pymc":
        raise ValueError("The cached_fixed_ou variant requires sampling.nuts_sampler='pymc'.")
    posterior_predictive = sampler.sample_posterior_predictive
    if cached and not isinstance(posterior_predictive, bool):
        unsupported = set(posterior_predictive) - {"y", "concentration"}
        if unsupported:
            raise ValueError(
                "cached_fixed_ou sampling.sample_posterior_predictive supports only "
                f"'y' or 'concentration'; got {sorted(unsupported)!r}."
            )
    return sampler


def _model(value: object) -> dict[str, object]:
    options = _table(value, "model")
    result: dict[str, object] = {}
    if "boundary" in options:
        boundary = _table(options.pop("boundary"), "model.boundary")
        enabled = _bool(boundary.pop("enabled", True), "model.boundary.enabled")
        prior = _prior(boundary.pop("prior"), "model.boundary.prior") if "prior" in boundary else None
        _reject_unknown(boundary, "model.boundary")
        if not enabled and prior is not None:
            raise ValueError("model.boundary.prior requires model.boundary.enabled=true.")
        result["use_bc"] = enabled
        if prior is not None:
            result["bc_prior"] = prior
    if "offset" in options:
        offset = _table(options.pop("offset"), "model.offset")
        result["offset_prior"] = _prior(_take(offset, "prior", "model.offset"), "model.offset.prior")
        offset_args: dict[str, object] = {}
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


def _ordinary_likelihood(value: object) -> dict[str, object]:
    options = _table(value, "likelihood")
    kind = _string(_take(options, "kind", "likelihood"), "likelihood.kind")
    result: dict[str, object] = {}
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
        kwargs: dict[str, object] = {}
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


def _cached_likelihood(value: object) -> dict[str, object]:
    options = _table(value, "likelihood")
    kind = _string(_take(options, "kind", "likelihood"), "likelihood.kind")
    if kind != "fixed_ou":
        raise ValueError("The cached_fixed_ou variant requires likelihood.kind='fixed_ou'.")
    result: dict[str, object] = {
        "tau_hours": _number_or_mapping(
            _take(options, "tau_hours", "likelihood"),
            "likelihood.tau_hours",
            positive=True,
        ),
        "site_amplitude_prior_scale": _number(
            _take(options, "site_amplitude_prior_scale", "likelihood"),
            "likelihood.site_amplitude_prior_scale",
            positive=True,
        ),
    }
    if "initial_site_amplitudes" in options:
        result["initial_site_amplitudes"] = _number_or_mapping(
            options.pop("initial_site_amplitudes"),
            "likelihood.initial_site_amplitudes",
            positive=True,
        )
    for name in ("sigma_target_accept", "state_target_accept"):
        if name not in options:
            continue
        target = _number(options.pop(name), f"likelihood.{name}")
        if not 0.0 < target < 1.0:
            raise ValueError(f"likelihood.{name} must be between zero and one.")
        result[name] = target
    _reject_unknown(options, "likelihood")
    return result


def _resolve_co2(options: dict[str, object], variant: str) -> Co2RunSetup:
    if variant not in ("ordinary", "cached_fixed_ou"):
        raise ValueError("recipe='co2' requires variant='ordinary' or 'cached_fixed_ou'.")
    prepared = _prepared_inputs(options)
    model = _model(options.pop("model", {}))
    likelihood = _take(options, "likelihood", "config")
    cached = variant == "cached_fixed_ou"
    sampler = _sampling(options.pop("sampling", {}), cached=cached)
    _reject_unknown(options, "config")
    if cached:
        runner = run_rhime_co2_cached_sigma
        runner_kwargs = {**model, **_cached_likelihood(likelihood)}
    else:
        runner = run_rhime_co2
        runner_kwargs = {**model, **_ordinary_likelihood(likelihood)}
    return Co2RunSetup(prepared, runner, _frozen(runner_kwargs), sampler)


def _resolve_linked(options: dict[str, object], variant: str) -> Co2O2RunSetup:
    if variant not in ("linked", "cached_fixed_ou"):
        raise ValueError("recipe='co2_o2' requires variant='linked' or 'cached_fixed_ou'.")
    channels = _table(_take(options, "channels", "config"), "channels")
    errors: dict[str, object] = {}
    units: dict[str, str] = {}
    channel_model: dict[str, dict[str, object]] = {}
    for name in ("co2", "o2"):
        channel = _table(_take(channels, name, "channels"), f"channels.{name}")
        channel_units = _string(_take(channel, "units", f"channels.{name}"), f"channels.{name}.units")
        mole_fraction_unit_scale(channel_units, context=f"channels.{name}.units")
        error = _number(
            _take(channel, "independent_error_sd", f"channels.{name}"),
            f"channels.{name}.independent_error_sd",
            positive=True,
        )
        model_options = {key: channel.pop(key) for key in ("boundary", "offset") if key in channel}
        if "boundary" in model_options:
            boundary = _table(model_options["boundary"], f"channels.{name}.boundary")
            if "activity" in boundary:
                activity = _table(boundary.pop("activity"), f"channels.{name}.boundary.activity")
                active = _bool(activity.pop("active", True), f"channels.{name}.boundary.activity.active")
                fixed_value = _number(
                    activity.pop("fixed_value", 1.0), f"channels.{name}.boundary.activity.fixed_value"
                )
                _reject_unknown(activity, f"channels.{name}.boundary.activity")
                if boundary.get("enabled", True) is False:
                    raise ValueError("Boundary activity requires enabled=true.")
                channel_model.setdefault("bc_state_activity", {})[name] = StateActivity(
                    active=active, fixed_value=fixed_value
                )
            model_options["boundary"] = boundary
        for key, value in _model(model_options).items():
            channel_model.setdefault(key, {})[name] = value
        _reject_unknown(channel, f"channels.{name}")
        units[name] = channel_units
        errors[name] = error
    _reject_unknown(channels, "channels")
    if units["co2"] != units["o2"]:
        raise ValueError(
            "The linked configuration currently requires identical CO2 and O2 channel units."
        )
    cached = variant == "cached_fixed_ou"
    likelihood = options.pop("likelihood", None)
    runner = run_rhime_co2_o2_from_prepared_inputs
    likelihood_kwargs: dict[str, object] = {}
    if cached:
        runner = run_rhime_co2_o2_cached_sigma_from_prepared_inputs
        likelihood_kwargs = _cached_likelihood(likelihood)
    elif likelihood is not None:
        likelihood_options = _table(likelihood, "likelihood")
        if likelihood_options.get("kind") != "fixed_ou":
            raise ValueError("The linked recipe supports config.likelihood.kind='fixed_ou'.")
        likelihood_kwargs = dict(
            cast(Mapping[str, object], _ordinary_likelihood(likelihood_options)["likelihood_kwargs"])
        )
    sampling = _table(options.pop("sampling", {}), "sampling")
    sampling.setdefault("nuts_sampler", "pymc" if likelihood is not None or cached else "numpyro")
    if not cached:
        sampling.setdefault("target_accept", 0.95)
    sampler = _sampling(sampling, cached=cached)
    if likelihood is not None and sampler.nuts_sampler != "pymc":
        raise ValueError("The linked fixed_ou likelihood requires sampling.nuts_sampler='pymc'.")
    _reject_unknown(options, "config")
    preparation_kwargs = {
        "co2_units": units["co2"],
        "o2_units": units["o2"],
    }
    return Co2O2RunSetup(
        _frozen(preparation_kwargs),
        runner,
        _frozen({"independent_error_sd": _frozen(errors), **channel_model, **likelihood_kwargs}),
        sampler,
    )


def load_co2_family_config(path: str | Path | Traversable) -> Mapping[str, object]:
    """Load one CO2-family TOML file without applying recipe semantics."""
    resource = Path(path) if isinstance(path, str | Path) else path
    with resource.open("rb") as config_file:
        return tomllib.load(config_file)


def co2_config_templates() -> Mapping[str, Traversable]:
    """Return installed CO2-family TOML templates keyed by file name."""
    directory = files("openghg_inversions.rhime").joinpath("config")
    return MappingProxyType(
        {name: directory.joinpath(name) for name in ("co2.toml", "co2_cached_sigma.toml", "co2_o2.toml")}
    )


def resolve_co2_family_config(
    config: Mapping[str, object],
) -> Co2RunSetup | Co2O2RunSetup:
    """Resolve a format-neutral mapping into one concrete recipe setup."""
    options = _table(config, "config")
    version = _take(options, "format_version", "config")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("config.format_version must be the integer 1.")
    recipe = _string(_take(options, "recipe", "config"), "config.recipe")
    variant = _string(_take(options, "variant", "config"), "config.variant")
    if recipe == "co2":
        return _resolve_co2(options, variant)
    if recipe == "co2_o2":
        return _resolve_linked(options, variant)
    raise ValueError("config.recipe must be 'co2' or 'co2_o2'.")


__all__ = [
    "Co2O2RunSetup",
    "Co2RunSetup",
    "co2_config_templates",
    "load_co2_family_config",
    "resolve_co2_family_config",
]
