"""Compatibility aliases for :mod:`openghg_inversions.model_components.coords`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.coords import (
    _coords_to_mapping as _coords_to_mapping,
    _coord_values as _coord_values,
    _coord_length as _coord_length,
    _coords_equal as _coords_equal,
    sanitize_coords_for_pymc as sanitize_coords_for_pymc,
    CoordRegistry as CoordRegistry,
    attach_coord_registry as attach_coord_registry,
    get_coord_registry as get_coord_registry,
    registered_model as registered_model,
    add_coords as add_coords,
    restore_inferencedata_coords as restore_inferencedata_coords,
)
