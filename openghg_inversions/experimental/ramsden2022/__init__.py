"""Historical Ramsden et al. (2022) methane/ethane inversion.

This experimental module preserves the paper's shared-state two-gas model
without porting the obsolete data loading, custom Metropolis-Hastings sampler,
or postprocessing from ``origin/adding_multi_gas_model``. Pass independently
prepared, canonical RHIME datasets to :func:`build_ramsden_model` to build a
PyMC graph or :func:`run_ramsden_from_prepared_inputs` to build and sample it.
The gases may use different observation axes, but coupled sector state
coordinates must match exactly.

Ratios may either be direct molar ratios applied to a ratio-free tracer design
or multipliers applied to a design that already includes a declared reference
ratio. Values must already share each channel's declared observation units;
the module validates supported unit declarations but does not convert values.
"""

from .model import (
    RamsdenChannelSpec,
    RamsdenModelSpec,
    RamsdenPreparedInputs,
    RamsdenResult,
    RamsdenSectorSpec,
    build_ramsden_model,
    run_ramsden_from_prepared_inputs,
)

__all__ = [
    "RamsdenChannelSpec",
    "RamsdenModelSpec",
    "RamsdenPreparedInputs",
    "RamsdenResult",
    "RamsdenSectorSpec",
    "build_ramsden_model",
    "run_ramsden_from_prepared_inputs",
]
