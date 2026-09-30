"""Compatibility imports; implementation lives in :mod:`openghg_inversions.inversion_data.prepared_inputs`."""

from openghg_inversions.inversion_data.prepared_inputs import (
    RHIME_PREPARED_INPUTS_SCHEMA as RHIME_PREPARED_INPUTS_SCHEMA,
    RHIME_PREPARED_INPUTS_SCHEMA_VERSION as RHIME_PREPARED_INPUTS_SCHEMA_VERSION,
    RhimePreparedInputs as RhimePreparedInputs,
    _SITE_AVERAGING_PERIOD as _SITE_AVERAGING_PERIOD,
    _canonicalize_rhime_inv_inputs as _canonicalize_rhime_inv_inputs,
    _make_site_metadata as _make_site_metadata,
    _multi_source_basis_labels as _multi_source_basis_labels,
    _normalize_site_metadata as _normalize_site_metadata,
    _site_metadata_for_serialisation as _site_metadata_for_serialisation,
    _site_metadata_from_serialisation as _site_metadata_from_serialisation,
)

__all__ = ['RHIME_PREPARED_INPUTS_SCHEMA', 'RHIME_PREPARED_INPUTS_SCHEMA_VERSION', 'RhimePreparedInputs']
