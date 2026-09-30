"""Compatibility aliases for :mod:`openghg_inversions.model_components.state_activity`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.state_activity import (
    to_dense as to_dense,
    ActivityValue as ActivityValue,
    FixedValue as FixedValue,
    StateActivity as StateActivity,
    PreparedLinearSensitivity as PreparedLinearSensitivity,
    prepare_linear_sensitivity as prepare_linear_sensitivity,
    ResolvedStateActivity as ResolvedStateActivity,
    _state_dim as _state_dim,
    detect_zero_sensitivity as detect_zero_sensitivity,
    _materialize_1d as _materialize_1d,
    _require_unique_state_coord as _require_unique_state_coord,
    _require_same_state_labels as _require_same_state_labels,
    _require_finite as _require_finite,
    _require_boolean as _require_boolean,
    _align_state_value as _align_state_value,
    resolve_state_activity as resolve_state_activity,
    active_prior_args as active_prior_args,
)
