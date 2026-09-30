"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.co2_preparation`."""

from openghg_inversions.recipes.co2.co2_preparation import (
    CO2_PREPARED_INPUTS_SCHEMA as CO2_PREPARED_INPUTS_SCHEMA,
    CO2_PREPARED_INPUTS_SCHEMA_VERSION as CO2_PREPARED_INPUTS_SCHEMA_VERSION,
    Co2AggregationErrorMode as Co2AggregationErrorMode,
    Co2PreparedInputs as Co2PreparedInputs,
    _AGGREGATION_PAYLOAD_NAMES as _AGGREGATION_PAYLOAD_NAMES,
    _AGGREGATION_REPRESENTATION_DIMS as _AGGREGATION_REPRESENTATION_DIMS,
    _borrow_without_axis_coordinates as _borrow_without_axis_coordinates,
    _json_mapping as _json_mapping,
    _renamed as _renamed,
    _require_axis as _require_axis,
    _require_equivalent_units as _require_equivalent_units,
    _require_same_axis as _require_same_axis,
    _validate_co2_dataset as _validate_co2_dataset,
    _without_aggregation_payload as _without_aggregation_payload,
    prepare_co2_inputs as prepare_co2_inputs,
)

__all__ = ['CO2_PREPARED_INPUTS_SCHEMA', 'CO2_PREPARED_INPUTS_SCHEMA_VERSION', 'Co2PreparedInputs', 'prepare_co2_inputs']
