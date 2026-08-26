"""RHIME adapters for reusable observation likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike, make_site_indicator, make_site_names
from openghg_inversions.models.additive_sigma import (
    DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_gaussian_likelihood,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.sigma import SigmaAlignment


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
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    output_dim: str,
    sigma_prior: Mapping[str, Any] | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    sigma_freq_anchor: DatetimeLike | None = None,
    no_model_error: bool = False,
) -> TensorVariable:
    """Adapt the additive-sigma Gaussian to the RHIME likelihood seam.

    Additive mismatch variance is independent of the pollution enhancement.
    The adapter derives its sigma alignment from the labelled observation
    ``site`` and ``time`` coordinates. Its optional sigma settings are ordinary
    JSON-compatible ``likelihood_kwargs``, so this installed callable can be
    passed directly to ``run_rhime`` or ``run_rhime_multisector``.

    Args:
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration.
        pollution_mean: Modelled pollution contribution, unused by additive
            mismatch variance.
        pollution_event_baseline: Baseline used by pollution-event-scaled
            likelihoods, unused by additive mismatch variance.
        output_dim: Observation dimension used for named PyMC variables.
        sigma_prior: Optional prior arguments used to construct ``sigma``. The
            distribution's ``sigma`` parameter may map site names to scales
            when ``sigma_per_site`` is true.
        sigma_freq: Optional frequency for sigma periods.
        sigma_per_site: Whether sigma varies independently by site.
        sigma_freq_anchor: Optional anchor for fixed-duration sigma periods.
        no_model_error: Whether to omit inferred mismatch error.

    Returns:
        The observed Gaussian variable, named ``y``. The reusable component
        also adds the canonical marginal error scale ``epsilon``.

    Raises:
        ValueError: If observation labels, sigma settings, or error inputs are
            invalid.
    """
    del pollution_mean, pollution_event_baseline
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


__all__ = ["DEFAULT_ADDITIVE_SIGMA_PRIOR", "additive_sigma_likelihood_builder"]
