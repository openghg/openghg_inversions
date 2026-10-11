"""Deprecated fixedbasis and RHIME option adapters at entrypoint boundaries.

This module translates historical spellings and fixedbasis policy without
importing an executable runner or constructing scientific configuration.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ALIASES = {
    "outputpath": "output_path",
    "outputname": "output_name",
    "xprior": "x_prior",
    "bcprior": "bc_prior",
    "sigprior": "sigma_prior",
    "offsetprior": "offset_prior",
    "emissions_name": "flux_sources",
    "outer_region_definition_file": "outer_regions_path",
}
_OUTPUT_FORMAT_ALIASES = {
    "hbmcmc": "legacy",
    "hbmcmc_postprocessing": "legacy",
}


def normalise_param_aliases(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy config spellings to modern snake-case names."""
    normalized = dict(params)
    for old, new in _ALIASES.items():
        if old not in normalized:
            continue
        if new in normalized:
            warnings.warn(
                f"Ignoring deprecated RHIME parameter {old!r} because {new!r} was also supplied.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"RHIME parameter {old!r} is deprecated; use {new!r} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            normalized[new] = normalized[old]
        del normalized[old]

    if "use_tracer" in normalized:
        if normalized.pop("use_tracer"):
            raise ValueError("`use_tracer=True` is not supported; tracer inversions are not implemented.")
        warnings.warn(
            "RHIME parameter 'use_tracer' is obsolete and has been removed.",
            DeprecationWarning,
            stacklevel=3,
        )

    if "calculate_min_error" in normalized:
        raise ValueError("`calculate_min_error` is not supported by RHIME runners; use `min_error`.")
    if "reparameterise_log_normal" in normalized:
        raise ValueError(
            "`reparameterise_log_normal` is not supported by RHIME runners; "
            "set `reparameterise` in the relevant prior dictionary if needed."
        )
    if "mcmc_type" in normalized:
        raise ValueError("`mcmc_type` is not supported by RHIME runners; use `nuts_sampler` if needed.")

    return normalized


def normalise_output_format_alias(params: dict[str, Any]) -> None:
    """Normalize deprecated HBMCMC output format names in-place."""
    output_format = params.get("output_format")
    if output_format is None:
        return
    output_format = str(output_format).lower()
    alias = _OUTPUT_FORMAT_ALIASES.get(output_format)
    if alias is not None:
        warnings.warn(
            f"RHIME output_format {output_format!r} is deprecated; use {alias!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        output_format = alias
    params["output_format"] = output_format


def translate_rhime_aliases(params: Mapping[str, Any]) -> dict[str, Any]:
    """Translate deprecated request spellings before canonical construction.

    Args:
        params: Raw external options after the entrypoint applies overrides.

    Returns:
        A fresh shallow mapping using canonical RHIME names. When both an alias
        and its canonical spelling occur, the canonical value wins. Container
        payloads remain borrowed; this adapter does not resolve configuration.

    Raises:
        ValueError: If obsolete fixedbasis switches have no RHIME equivalent.

    Deprecated parameter and output-format spellings emit ``DeprecationWarning``.
    ``outer_region_definition_file`` is a deprecated RHIME spelling rather than
    a fixedbasis scientific switch and is translated at the same boundary.
    """
    translated = normalise_param_aliases(params)
    normalise_output_format_alias(translated)
    return translated


_RUN_HBMCMC_RHIME_ALIASES = {
    "nit": "draws",
    "nchain": "chains",
    "verbose": "progressbar",
    "sampler_kwargs": "sample_kwargs",
}
_LOGNORMAL_PRIOR_NAMES = ("xprior", "x_prior", "bcprior", "bc_prior")
_MIN_ERROR_METHODS = {"residual", "percentile"}


def _legacy_option_enabled(value: Any) -> bool:
    """Return whether a legacy option value should be treated as enabled."""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "none", "0"}
    return True


def _translate_legacy_aliases(params: dict[str, Any]) -> None:
    """Translate legacy run_hbmcmc parameter names to RHIME names in-place."""
    for old, new in _RUN_HBMCMC_RHIME_ALIASES.items():
        if old not in params:
            continue
        warnings.warn(
            f"run_hbmcmc parameter {old!r} is deprecated; use {new!r} instead."
            + (f" Ignoring {old!r} because {new!r} was also supplied." if new in params else ""),
            DeprecationWarning,
            stacklevel=3,
        )
        if new not in params:
            params[new] = params[old]
        del params[old]


def _normalise_legacy_output_format(params: dict[str, Any]) -> None:
    """Map old HBMCMC output names to the modern compatibility output."""
    paris_postprocessing = params.pop("paris_postprocessing", False)
    if _legacy_option_enabled(paris_postprocessing):
        params["output_format"] = "paris"
        return

    raw_output_format = params.get("output_format")
    if raw_output_format is None:
        params["output_format"] = "legacy"
        return

    params["output_format"] = str(raw_output_format).lower()


def _translate_calculate_min_error(params: dict[str, Any]) -> None:
    """Translate legacy ``calculate_min_error`` to the modern ``min_error`` option."""
    if "calculate_min_error" not in params:
        return

    value = params.pop("calculate_min_error")
    if not _legacy_option_enabled(value):
        return

    method = str(value).strip().lower()
    if method not in _MIN_ERROR_METHODS:
        raise ValueError(
            "`calculate_min_error` is deprecated and can only be translated when set to "
            f"one of {sorted(_MIN_ERROR_METHODS)!r}; use `min_error` instead."
        )

    warnings.warn(
        "`calculate_min_error` is deprecated. The run_hbmcmc compatibility shim is translating "
        "it to `min_error`.",
        DeprecationWarning,
        stacklevel=3,
    )
    params["min_error"] = method


def _translate_reparameterise_log_normal(params: dict[str, Any]) -> None:
    """Translate legacy lognormal reparameterisation flag into prior dictionaries."""
    value = params.pop("reparameterise_log_normal", False)
    if not _legacy_option_enabled(value):
        return

    warnings.warn(
        "`reparameterise_log_normal` is deprecated. The run_hbmcmc compatibility shim is setting "
        "`reparameterise=True` in lognormal emissions and BC prior dictionaries.",
        DeprecationWarning,
        stacklevel=3,
    )
    for name in _LOGNORMAL_PRIOR_NAMES:
        prior = params.get(name)
        if not isinstance(prior, dict):
            continue
        if str(prior.get("pdf", "")).lower() != "lognormal":
            continue
        prior = prior.copy()
        prior["reparameterise"] = True
        params[name] = prior


def _translate_legacy_options(params: dict[str, Any]) -> None:
    """Translate legacy fixedbasis options that have modern RHIME equivalents."""
    _translate_calculate_min_error(params)
    _translate_reparameterise_log_normal(params)


def fixedbasis_params_to_rhime(params: dict[str, Any]) -> dict[str, Any]:
    """Translate fixedbasis-style script/config parameters into RHIME arguments.

    Args:
        params: Decoded legacy options after command-line overrides. If supplied,
            ``mcmc_type`` must be ``fixed_basis``.

    Returns:
        A fresh mapping with canonical names and historical output policy.
        Values are validated and coerced later by ``RhimeConfig.from_params``.
        Prior mappings changed for lognormal reparameterisation are copied;
        other values remain borrowed.

    Raises:
        ValueError: If a removed MCMC route or unsupported legacy option is used.

    Deprecated names and translated scientific switches emit warnings.
    """
    translated = dict(params)
    translated.pop("likelihood", None)
    translated.pop("additive_sigma_prior", None)
    mcmc_type = translated.pop("mcmc_type", "fixed_basis")
    if mcmc_type != "fixed_basis":
        raise ValueError(f"Unsupported run_hbmcmc mcmc_type {mcmc_type!r}; expected 'fixed_basis'.")

    _translate_legacy_aliases(translated)
    _normalise_legacy_output_format(translated)
    _translate_legacy_options(translated)
    translated["output_filename_convention"] = "legacy"
    if "save_inversion_output" not in translated and translated["output_format"] != "inv_out":
        translated["save_inversion_output"] = False
    return translate_rhime_aliases(translated)


def params_from_config(
    config_file: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
    normalise: bool = True,
) -> dict[str, Any]:
    """Load file options through the deprecated dictionary-returning adapter.

    Args:
        config_file: Existing INI configuration file.
        start_date: Optional command-line start-date override.
        end_date: Optional command-line end-date override.
        output_path: Optional command-line output-path override.
        extra_kwargs: Final winning keyword overrides.
        normalise: Translate aliases, coerce scalars and validate structured
            values when true. False returns only decoded values and overrides.

    Returns:
        A fresh option mapping. This adapter does not resolve defaults or site
        shorthand and does not resolve runner setup.

    Raises:
        ValueError: If normalization finds unsupported legacy switches or
            malformed structured values.

    Emits ``DeprecationWarning``. Use ``read_rhime_ini`` for decoded values,
    apply overrides, then call ``RhimeConfig.from_params``.
    """
    from openghg_inversions.rhime.ini import read_rhime_ini

    warnings.warn(
        "params_from_config is deprecated; use read_rhime_ini, apply overrides, "
        "then RhimeConfig.from_params for canonical options.",
        DeprecationWarning,
        stacklevel=2,
    )
    params = read_rhime_ini(config_file)
    for name, value in (("start_date", start_date), ("end_date", end_date), ("output_path", output_path)):
        if value is not None:
            params[name] = value
    if extra_kwargs:
        params.update(extra_kwargs)
    if not normalise:
        return params

    from openghg_inversions.rhime.params import normalise_rhime_params

    return normalise_rhime_params(translate_rhime_aliases(params))
