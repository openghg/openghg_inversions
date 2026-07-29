"""Public RHIME API."""

from __future__ import annotations

from openghg_inversions.models.rhime import RhimeModelSpec, SectorSpec

from .params import params_from_config, resolve_flux_sources
from .runner import (
    RhimeResult,
    run_rhime,
    run_rhime_from_prepared_inputs,
    run_rhime_multisector,
)
from .sampling import RhimeSampler
from .specs import RhimeOutputSpec, RhimeRunSpec

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeRunSpec",
    "RhimeResult",
    "params_from_config",
    "resolve_flux_sources",
    "run_rhime",
    "run_rhime_from_prepared_inputs",
    "run_rhime_multisector",
]
