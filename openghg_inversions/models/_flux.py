"""Compatibility aliases for :mod:`openghg_inversions.model_components._flux`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components._flux import (
    select_gathered_data_array as select_gathered_data_array,
    safe_pymc_name as safe_pymc_name,
    _prepared_sources as _prepared_sources,
    _validate_unpadded_sector_design as _validate_unpadded_sector_design,
    _select_sector_design as _select_sector_design,
    _namespace_sector_state_coords as _namespace_sector_state_coords,
)
