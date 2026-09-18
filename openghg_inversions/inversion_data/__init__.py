import warnings
from typing import Any

from .get_data import data_processing_surface_notracer
from .preparation import (
    RhimeMergedData,
    RhimePreparedInputs,
    prepare_rhime_inputs,
)
from .serialise import load_merged_data, _save_merged_data
from .xarray_adapter import prepare_rhime_inputs_from_xarray

__all__ = [
    "_save_merged_data",
    "FixedBasisPreparedData",
    "RhimeMergedData",
    "RhimePreparedInputs",
    "data_processing_surface_notracer",
    "load_merged_data",
    "prepare_fixedbasis_inversion_data",
    "prepare_rhime_inputs",
    "prepare_rhime_inputs_from_xarray",
]


def __getattr__(name: str) -> Any:
    """Provide warning-emitting aliases for former fixed-basis exports."""
    if name not in {"FixedBasisPreparedData", "prepare_fixedbasis_inversion_data"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"{__name__}.{name} has moved to openghg_inversions.hbmcmc.preparation; "
        "the old import path is deprecated.",
        FutureWarning,
        stacklevel=2,
    )
    from openghg_inversions.hbmcmc import preparation as fixedbasis_preparation

    return getattr(fixedbasis_preparation, name)
