"""INI decoding and compatibility adapters for RHIME configuration.

Section flattening, first-occurrence precedence and literal decoding belong to
this frontend. Winning overrides precede shared semantic construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openghg_inversions.config import config

if TYPE_CHECKING:
    from .params import RhimeConfig


def _decode_rhime_ini(path: str | Path) -> dict[str, Any]:
    """Decode INI sections into bare keys using the existing first-key policy."""
    return dict(config.all_param(str(path), exclude_not_found=True, allow_new=True))


def read_rhime_ini(
    path: str | Path,
    *,
    overrides: Mapping[str, object] | None = None,
    multisector: bool = False,
) -> RhimeConfig:
    """Read an INI request and return complete resolved RHIME configuration.

    Args:
        path: Existing RHIME INI configuration file. Sections contribute bare
            option names; repeated names retain their first occurrence.
        overrides: Winning option values, applied before defaults and site
            shorthand are resolved.
        multisector: Whether to resolve the multisector recipe.

    Returns:
        Complete requested configuration, ready to inspect without data access
        or another semantic resolution pass.

    Raises:
        ValueError: If effective options cannot be resolved.
    """
    from .params import RhimeConfig

    params = _decode_rhime_ini(path)
    if overrides:
        params.update(overrides)
    return RhimeConfig.from_params(params, multisector=multisector)


def params_from_config(
    config_file: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: str | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
    normalise: bool = True,
) -> dict[str, Any]:
    """Load RHIME run parameters from an INI config file.

    Args:
        config_file: Path to an INI configuration file.
        start_date: Optional command-line start-date override.
        end_date: Optional command-line end-date override.
        output_path: Optional command-line output-path override.
        extra_kwargs: Optional keyword overrides, normally parsed from CLI JSON.
        normalise: Whether to normalize and validate the merged parameters.
            False returns decoded file options with overrides applied.

    Returns:
        RHIME options, normalized to snake-case names when ``normalise`` is
        true. This compatibility adapter retains a dictionary return.

    Raises:
        ValueError: If deprecated unsupported parameters are present or a
            structured RHIME option has an invalid type.
    """
    from .params import normalise_rhime_params

    params = _decode_rhime_ini(config_file)
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date
    if output_path is not None:
        params["output_path"] = output_path
    if extra_kwargs:
        params.update(extra_kwargs)
    return normalise_rhime_params(params) if normalise else params


