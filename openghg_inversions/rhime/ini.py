"""Decode RHIME INI syntax without resolving a scientific recipe.

Section flattening, first-occurrence precedence, literal decoding and the flat
spelling of nested option groups belong to this frontend. Callers combine overrides and consume recipe-specific options
before constructing canonical configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openghg_inversions.config import config
from openghg_inversions.rhime.params import DATA_OPTION_GROUPS, nest_flat_options


def read_rhime_ini(path: str | Path) -> dict[str, Any]:
    """Decode INI sections into option names and Python values.

    Args:
        path: Existing RHIME INI configuration file. Sections contribute bare
            option names; repeated names retain their first occurrence.

    INI files spell data option groups flat, for example ``bc_basis_algorithm``;
    the reader nests them as ``{"bc_basis": {"algorithm": ...}}``.

    Returns:
        A new dictionary of decoded file options. Other names and external
        shorthand are preserved, including recipe-specific options. The reader
        does not apply overrides, supply defaults or resolve a recipe.

    Raises:
        ValueError: If a file gives a data option group in both flat and nested form.
    """
    decoded = config.all_param(str(path), exclude_not_found=True, allow_new=True)
    return nest_flat_options(decoded, DATA_OPTION_GROUPS)
