"""Public RHIME API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model_specs import RhimeModelSpec, SectorSpec

if TYPE_CHECKING:
    from .params import params_from_config, resolve_flux_sources
    from .runner import RhimeResult, run_rhime, run_rhime_multisector
    from .sampling import RhimeSampler, RhimeSamplingSpec
    from .specs import RhimeOutputSpec, RhimeRunSpec

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeSamplingSpec",
    "RhimeRunSpec",
    "RhimeResult",
    "params_from_config",
    "resolve_flux_sources",
    "run_rhime",
    "run_rhime_multisector",
]

_PARAM_EXPORTS = {"params_from_config", "resolve_flux_sources"}
_RUNNER_EXPORTS = {"RhimeResult", "run_rhime", "run_rhime_multisector"}
_SAMPLING_EXPORTS = {"RhimeSampler", "RhimeSamplingSpec"}
_SPEC_EXPORTS = {"RhimeOutputSpec", "RhimeRunSpec"}


def __getattr__(name: str) -> Any:
    """Load execution-layer exports on demand while keeping config/spec imports light."""
    if name in _PARAM_EXPORTS:
        from . import params

        value = getattr(params, name)
    elif name in _RUNNER_EXPORTS:
        from . import runner

        value = getattr(runner, name)
    elif name in _SAMPLING_EXPORTS:
        from . import sampling

        value = getattr(sampling, name)
    elif name in _SPEC_EXPORTS:
        from . import specs

        value = getattr(specs, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
