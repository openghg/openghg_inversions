r"""Model-owned likelihood with an additive model-data-mismatch scale.

For reported observation-error variance :math:`s_y^2`, optional fixed mismatch
:math:`s_{fixed}`, inferred mismatch scale :math:`\sigma`, and fixed
aggregation covariance :math:`C_{agg}`, this
component constructs the independent variance

.. math::

   v = s_y^2 + s_{fixed}^2 + \sigma^2

An optional error-floor policy may apply a lower bound to total marginal
standard deviation. ``add_additive_sigma_likelihood`` is the one direct model
component: RHIME-specific site/frequency translation lives outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable
from pydantic import StrictBool
from pydantic.dataclasses import dataclass

from openghg_inversions._record_schema import Record
from openghg_inversions.models._likelihood_settings import SETTINGS_CONFIG, SigmaSettings
from openghg_inversions.models.components import add_model_data, add_sigma_component
from openghg_inversions.models.priors import PriorArgs, PriorRecord
from openghg_inversions.models._gaussian_observation import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.observation_error import (
    AggregationError,
    validate_observation_error_arrays,
)
from openghg_inversions.sigma import SigmaAlignment


FIXED_MODEL_MISMATCH = "fixed_model_mismatch"
DEFAULT_ADDITIVE_SIGMA_PRIOR: PriorArgs = {"pdf": "halfnormal", "sigma": 1.0}


@dataclass(frozen=True, config=SETTINGS_CONFIG)
class AdditiveSigmaSettings(SigmaSettings):
    """Validated settings for additive model-data-mismatch error.

    Numerical prior values are borrowed; ``to_record`` persists only used,
    constructed prior metadata supplied by the caller.

    Args:
        sigma_prior: Prior for the additive model-error standard deviation.
        sigma_freq: Frequency of the latent model-error periods. ``None`` uses
            one period.
        sigma_per_site: Whether model error varies by observation site.
        sigma_freq_anchor: Optional anchor for fixed-duration periods.
        use_minimum_error_floor: Apply the prepared historical minimum total-
            error floor.
    """

    DEFAULT_SIGMA_PRIOR: ClassVar[PriorArgs] = DEFAULT_ADDITIVE_SIGMA_PRIOR

    use_minimum_error_floor: StrictBool = False
    kind: Literal["additive_sigma"] = field(default="additive_sigma", kw_only=True)

    @property
    def required_prepared_inputs(self) -> tuple[str, ...]:
        return ("min_error",) if self.use_minimum_error_floor else ()

    def to_record(self, *, priors: Mapping[str, PriorArgs]) -> AdditiveSigmaRecord:
        return AdditiveSigmaRecord.model_validate({
            "kind": self.kind,
            "options": {**self.recorded_options(), "sigma_prior": priors.get("sigma")},
        })


class AdditiveSigmaOptions(Record):
    sigma_prior: PriorRecord | None
    sigma_freq: str | None
    sigma_per_site: bool
    sigma_freq_anchor: str | None
    use_minimum_error_floor: bool


class AdditiveSigmaRecord(Record):
    kind: Literal["additive_sigma"]
    options: AdditiveSigmaOptions

    def to_runtime(self, *, trace: xr.DataTree | None = None) -> AdditiveSigmaSettings:
        return AdditiveSigmaSettings(
            **self.options.model_dump(exclude={"sigma_prior"}),
            sigma_prior=None if self.options.sigma_prior is None else self.options.sigma_prior.to_runtime(trace=trace),
        )


def add_additive_sigma_likelihood(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    minimum_error_floor: xr.DataArray | None = None,
    fixed_model_mismatch: xr.DataArray | None = None,
    additive_sigma_alignment: SigmaAlignment | None = None,
    additive_sigma_prior: PriorArgs | None = None,
    output_dim: str = "nmeasure",
    observation_error_name: str = "error",
) -> TensorVariable:
    """Add a Gaussian likelihood with additive mismatch variance.

    ``fixed_model_mismatch`` is a known observation-aligned standard deviation.
    The inferred absolute mismatch scale is observation-aligned through
    ``additive_sigma_alignment`` and enters as a separate variance term. When
    no alignment is supplied, no inferred scale is constructed. Fixed
    diagonal, dense, or low-rank aggregation error is included only when
    explicitly selected. If supplied, ``minimum_error_floor`` floors the total
    marginal standard deviation.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration aligned with
            ``output_dim``.
        minimum_error_floor: Optional minimum total-error standard deviations.
        fixed_model_mismatch: Optional known model-data mismatch standard
            deviation, in the same units and observation order as
            ``observations``.
        additive_sigma_alignment: Mapping from observations to absolute
            additive mismatch-scale parameters. Omit it for a fixed-error
            likelihood.
        additive_sigma_prior: Prior arguments used to construct the absolute
            additive mismatch scale.
        output_dim: Observation dimension used for named PyMC variables.
        observation_error_name: PyMC data name for the reported error.

    Returns:
        The observed Gaussian variable, named ``y``. The total marginal error
        scale is also recorded in the active model as ``epsilon``.

    Raises:
        ValueError: If the observation or aggregation-error inputs are
            inconsistent with ``output_dim``.
    """
    validate_observation_error_arrays(
        observations,
        observation_error,
        minimum_error_floor,
        owner="Additive-sigma likelihood",
        output_dim=output_dim,
    )
    reported_error = add_model_data(
        observation_error.transpose(output_dim),
        observation_error_name,
    )
    registered_aggregation_error = add_aggregation_error_data(
        aggregation_error,
        observations,
        output_dim=output_dim,
    )

    independent_variance = reported_error**2
    if fixed_model_mismatch is not None:
        if fixed_model_mismatch.dims != (output_dim,):
            raise ValueError(
                f"Additive-sigma likelihood input {FIXED_MODEL_MISMATCH!r} must "
                f"have dims ({output_dim!r},); got {fixed_model_mismatch.dims!r}."
            )
        fixed_values = np.asarray(fixed_model_mismatch.values)
        if not np.issubdtype(fixed_values.dtype, np.number):
            raise ValueError(f"{FIXED_MODEL_MISMATCH!r} must be numeric.")
        if not np.isfinite(fixed_values).all() or (fixed_values < 0).any():
            raise ValueError(f"{FIXED_MODEL_MISMATCH!r} must contain only finite, non-negative values.")
        fixed_mismatch_data = add_model_data(
            fixed_model_mismatch.transpose(output_dim),
            FIXED_MODEL_MISMATCH,
        )
        independent_variance = independent_variance + fixed_mismatch_data**2
    if additive_sigma_alignment is not None:
        if additive_sigma_prior is None:
            raise ValueError(
                "Additive-sigma likelihood requires `additive_sigma_prior` with "
                "`additive_sigma_alignment`."
            )
        additive_sigma = add_sigma_component(
            additive_sigma_alignment,
            prior_args=dict(additive_sigma_prior),
        )
        independent_variance = independent_variance + additive_sigma**2

    aggregation_marginal_variance = registered_aggregation_error.marginal_variance
    if minimum_error_floor is not None:
        minimum_error_data = add_model_data(
            minimum_error_floor.transpose(output_dim),
            "min_error",
        )
        floor_variance = cast(Any, pt.maximum)(
            minimum_error_data**2 - independent_variance - aggregation_marginal_variance,
            0.0,
        )
        independent_variance = independent_variance + floor_variance
    pm.Deterministic(
        "epsilon",
        pt.sqrt(independent_variance + aggregation_marginal_variance),
        dims=output_dim,
    )
    return add_gaussian_observation_likelihood(
        observed=pm.floatX(observations.transpose(output_dim).compute().values),
        mean=mean,
        independent_variance=independent_variance,
        aggregation_error=registered_aggregation_error,
        output_dim=output_dim,
    )
__all__ = [
    "AdditiveSigmaSettings", "AdditiveSigmaOptions", "AdditiveSigmaRecord",
    "DEFAULT_ADDITIVE_SIGMA_PRIOR",
    "FIXED_MODEL_MISMATCH",
    "add_additive_sigma_likelihood",
]
