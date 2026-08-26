r"""Model-owned likelihood with an additive model-data-mismatch scale.

For reported observation-error variance :math:`s_y^2`, optional fixed mismatch
:math:`s_{fixed}`, inferred mismatch scale :math:`\sigma`, and fixed
aggregation covariance :math:`C_{agg}`, this
component constructs the independent variance

.. math::

   v = s_y^2 + s_{fixed}^2 + \sigma^2

An optional ``min_error`` applies the same floor on total marginal standard
deviation as the historical likelihood.
``add_additive_sigma_gaussian_likelihood`` owns the complete observation model
for an explicitly supplied modelled concentration. The
``additive_sigma_likelihood_builder`` entry point supplies the same complete
PyMC construction through RHIME's ordinary likelihood-builder seam. Both
public functions require an active model context and therefore live in this
model module rather than under ``openghg_inversions.rhime``.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike, make_site_indicator, make_site_names
from openghg_inversions.models.components import add_model_data, add_sigma_component
from openghg_inversions.models.gaussian_likelihood import (
    add_aggregation_error_data,
    add_gaussian_observation_likelihood,
)
from openghg_inversions.observation_error import (
    AggregationError,
    validate_observation_error_arrays,
)
from openghg_inversions.sigma import SigmaAlignment


FIXED_MODEL_MISMATCH = "fixed_model_mismatch"
DEFAULT_ADDITIVE_SIGMA_PRIOR = {"pdf": "halfnormal", "sigma": 1.0}


def add_additive_sigma_gaussian_likelihood(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    minimum_error: xr.DataArray | None = None,
    fixed_model_mismatch: xr.DataArray | None = None,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: Mapping[str, Any] | None = None,
    output_dim: str = "nmeasure",
    observation_error_name: str = "error",
) -> TensorVariable:
    """Add a Gaussian likelihood with additive mismatch variance.

    ``fixed_model_mismatch`` is a known observation-aligned standard deviation.
    ``sigma`` is observation-aligned through ``sigma_alignment`` and enters as
    a separate inferred variance term. When no alignment is supplied, no
    ``sigma`` variable is constructed. Fixed diagonal, dense, or low-rank
    aggregation error is included only when explicitly selected. If supplied,
    ``minimum_error`` floors the total marginal standard deviation.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration aligned with
            ``output_dim``.
        minimum_error: Optional minimum total-error standard deviations.
        fixed_model_mismatch: Optional known model-data mismatch standard
            deviation, in the same units and observation order as
            ``observations``.
        sigma_alignment: Mapping from observations to mismatch-scale
            parameters. Omit it for a fixed-error likelihood.
        sigma_prior: Prior arguments used to construct ``sigma`` when model
            error is enabled.
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
        minimum_error,
        owner="Additive-sigma likelihood",
        output_dim=output_dim,
    )
    if sigma_alignment is not None and sigma_prior is None:
        raise ValueError("Additive-sigma likelihood requires `sigma_prior` with `sigma_alignment`.")
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
    if sigma_alignment is not None:
        assert sigma_prior is not None
        sigma = add_sigma_component(sigma_alignment, prior_args=dict(sigma_prior))
        independent_variance = independent_variance + sigma**2

    aggregation_marginal_variance = registered_aggregation_error.marginal_variance
    if minimum_error is not None:
        minimum_error_data = add_model_data(minimum_error.transpose(output_dim), "min_error")
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


def _resolve_site_sigma_prior(
    sigma_prior: Mapping[str, Any],
    site: xr.DataArray,
    *,
    per_site: bool,
) -> dict[str, Any]:
    """Align an optional site-keyed ``sigma`` parameter to latent site order."""
    resolved = dict(sigma_prior)
    sigma = resolved.get("sigma")
    if not isinstance(sigma, Mapping):
        return resolved
    if not per_site:
        raise ValueError("A site-keyed sigma prior requires `sigma_per_site=True`.")

    site_names = [str(value) for value in make_site_names(site).values]
    missing = [name for name in site_names if name not in sigma]
    if missing:
        raise ValueError(f"Site-keyed sigma prior is missing retained site(s): {missing!r}.")

    values = [sigma[name] for name in site_names]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("Site-keyed sigma prior values must be finite positive numbers.")
    scales = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Site-keyed sigma prior values must be finite positive numbers.")
    resolved["sigma"] = scales[:, None]
    return resolved


def additive_sigma_likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    output_dim: str,
    sigma_prior: Mapping[str, Any] | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    sigma_freq_anchor: DatetimeLike | None = None,
    no_model_error: bool = False,
) -> TensorVariable:
    """Build the additive-sigma Gaussian through RHIME's likelihood seam.

    Additive mismatch variance is independent of pollution-event terms. The
    builder derives sigma alignment from labelled observation ``site`` and
    ``time`` coordinates, then constructs the complete likelihood in the
    active PyMC model.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration.
        output_dim: Observation dimension used for named PyMC variables.
        sigma_prior: Optional prior arguments used to construct ``sigma``. The
            distribution's ``sigma`` parameter may map site names to scales
            when ``sigma_per_site`` is true.
        sigma_freq: Optional frequency for sigma periods.
        sigma_per_site: Whether sigma varies independently by site.
        sigma_freq_anchor: Optional anchor for fixed-duration sigma periods.
        no_model_error: Whether to omit inferred mismatch error.

    Returns:
        The observed Gaussian variable, named ``y``. The component also adds
        the canonical marginal error scale ``epsilon``.

    Raises:
        ValueError: If observation labels, sigma settings, or error inputs are
            invalid.
    """
    site = None
    sigma_alignment = None
    if not no_model_error:
        site = observations.coords.get("site")
        if site is None or site.dims != (output_dim,):
            raise ValueError(
                "The additive-sigma likelihood requires an observation-aligned "
                "'site' coordinate when model error is enabled."
            )
        sigma_alignment = SigmaAlignment.from_frequency(
            make_site_indicator(site),
            frequency=sigma_freq,
            per_site=sigma_per_site,
            anchor_time=sigma_freq_anchor,
        )
    resolved_sigma_prior = None
    if not no_model_error:
        assert site is not None
        resolved_sigma_prior = _resolve_site_sigma_prior(
            DEFAULT_ADDITIVE_SIGMA_PRIOR if sigma_prior is None else sigma_prior,
            site,
            per_site=sigma_per_site,
        )
    return add_additive_sigma_gaussian_likelihood(
        observations=observations,
        observation_error=observation_error,
        minimum_error=minimum_error,
        aggregation_error=aggregation_error,
        mean=mean,
        sigma_alignment=sigma_alignment,
        sigma_prior=resolved_sigma_prior,
        output_dim=output_dim,
    )


__all__ = [
    "DEFAULT_ADDITIVE_SIGMA_PRIOR",
    "FIXED_MODEL_MISMATCH",
    "add_additive_sigma_gaussian_likelihood",
    "additive_sigma_likelihood_builder",
]
