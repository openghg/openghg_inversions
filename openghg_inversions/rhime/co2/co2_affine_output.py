"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.co2_affine_output`."""

from openghg_inversions.recipes.co2.co2_affine_output import (
    BoundCo2AffineFluxMap as BoundCo2AffineFluxMap,
    _artifact as _artifact,
    _bind_affine_flux_map as _bind_affine_flux_map,
    _same_axis as _same_axis,
    _same_flux_units as _same_flux_units,
    import_explicit_affine_flux_map as import_explicit_affine_flux_map,
    load_and_bind_affine_flux_map as load_and_bind_affine_flux_map,
    prepared_inputs_content_id as prepared_inputs_content_id,
    produce_bucket_affine_flux_map as produce_bucket_affine_flux_map,
)

__all__ = ['BoundCo2AffineFluxMap', 'import_explicit_affine_flux_map', 'load_and_bind_affine_flux_map', 'prepared_inputs_content_id', 'produce_bucket_affine_flux_map']
