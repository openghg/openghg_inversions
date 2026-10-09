"""Decode RHIME INI syntax without resolving a scientific recipe.

Section flattening, first-occurrence precedence and literal decoding belong to
this frontend. Callers combine overrides and consume recipe-specific options
before translating compatibility names and constructing configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openghg_inversions.config import config
from openghg_inversions.hbmcmc.compatibility import params_from_config as params_from_config


def read_rhime_ini(path: str | Path) -> dict[str, Any]:
    """Decode INI sections into option names and Python values.

    Args:
        path: Existing RHIME INI configuration file. Sections contribute bare
            option names; repeated names retain their first occurrence.

    Returns:
        A new dictionary of decoded file options. Names and external shorthand
        are preserved, including recipe-specific options. The reader does not
        apply overrides, translate aliases, supply defaults or resolve a recipe.
    """
    return dict(config.all_param(str(path), exclude_not_found=True, allow_new=True))
