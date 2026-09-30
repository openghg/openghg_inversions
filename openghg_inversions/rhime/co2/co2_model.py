"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.co2_model`."""

from openghg_inversions.recipes.co2.co2_model import (
    _fixed_mismatch_array as _fixed_mismatch_array,
    _normalise_offset_args as _normalise_offset_args,
    build_co2_model as build_co2_model,
)

__all__ = ['build_co2_model']
