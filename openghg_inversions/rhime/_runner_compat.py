"""Deprecated spellings accepted by runner frontends until 0.9."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

from openghg_inversions._flux_options import normalise_flux_aliases

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


def resource_migration_errors(names: Iterable[str]) -> list[str]:
    """Explain removed resource selectors at frontend validation boundaries."""
    names = set(names)
    errors = []
    if "country_directory" in names:
        errors.append(
            "`country_directory` was removed in 0.8; use `country_file` for reporting, "
            "`landsea_path` for land/sea classes, or `outer_regions_path` for outer regions."
        )
    historical = {"fp_basis_case", "basis_directory", "bc_basis_case", "bc_basis_directory"} & names
    if historical:
        errors.append(
            f"Historical resource selectors {sorted(historical)!r} belong to the temporary HBMCMC route. "
            "Use `basis_path`, `basis_partition_path`/`basis_partition_paths`, or `bc_basis_paths` "
            "in modern RHIME requests."
        )
    return errors


def normalise_param_aliases(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy config spellings to modern snake-case names."""
    normalized = dict(params)
    if errors := resource_migration_errors(normalized):
        raise ValueError(" ".join(errors))
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
        params: Flat or grouped external options after the entrypoint applies overrides.

    Returns:
        A fresh shallow mapping for the runner to pass to canonical configuration. When both an alias
        and its canonical spelling occur, the canonical value wins. Container
        payloads remain borrowed; this adapter does not resolve configuration.

    Raises:
        ValueError: If obsolete fixedbasis switches have no RHIME equivalent.

    Deprecated parameter and output-format spellings emit ``DeprecationWarning``.
    ``outer_region_definition_file`` is a deprecated RHIME spelling rather than
    a fixedbasis scientific switch and is translated at the same boundary.
    """
    from .params import rhime_option_groups

    translated = normalise_flux_aliases(normalise_param_aliases(params))
    for group in rhime_option_groups():
        if isinstance(translated.get(group), Mapping):
            translated[group] = translate_rhime_aliases(translated[group])
    normalise_output_format_alias(translated)
    return translated


