"""RHIME adapters for reusable observation likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import (
    DatetimeLike,
    make_site_indicator,
    make_site_names,
)
from openghg_inversions.models.components import add_model_data
from openghg_inversions.models.coords import add_coords
from openghg_inversions.models.additive_sigma import (
    DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_gaussian_likelihood,
)
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.models.likelihoods import add_aggregation_error_data
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.observation_error import (
    AggregationError,
    validate_observation_error_arrays,
)
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


DEFAULT_OU_SITE_AMPLITUDE_PRIOR: dict[str, Any] = {
    "pdf": "halfnormal",
    "sigma": 0.75,
}


def _aggregation_low_rank_factor(
    aggregation_error: AggregationError,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact factor-plus-diagonal representation of aggregation error."""
    n_observation = aggregation_error.marginal_variance.size
    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        covariance = np.asarray(aggregation_error.covariance.values, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
        tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if eigenvalues[0] < -tolerance:
            raise ValueError("Dense aggregation covariance must be positive semidefinite.")
        positive = eigenvalues > 0.0
        factor = eigenvectors[:, positive] * np.sqrt(np.maximum(eigenvalues[positive], 0.0))
        return factor, np.zeros(n_observation)
    if aggregation_error.mode == "low_rank":
        assert aggregation_error.factor is not None
        assert aggregation_error.diagonal_variance is not None
        return (
            np.asarray(aggregation_error.factor.values, dtype=float),
            np.asarray(aggregation_error.diagonal_variance.values, dtype=float),
        )
    if aggregation_error.mode == "diagonal":
        assert aggregation_error.diagonal_variance is not None
        return (
            np.empty((n_observation, 0)),
            np.asarray(aggregation_error.diagonal_variance.values, dtype=float),
        )
    return np.empty((n_observation, 0)), np.zeros(n_observation)


def _fixed_site_amplitudes(
    value: float | Mapping[str, float],
    site_labels: tuple[str, ...],
) -> np.ndarray:
    """Resolve a scalar or exact labelled fixed-amplitude mapping."""
    if isinstance(value, Mapping):
        if set(value) != set(site_labels):
            raise ValueError(
                "`fixed_site_amplitudes` mapping keys must exactly match the observation sites."
            )
        amplitudes = np.asarray([value[site] for site in site_labels], dtype=float)
    else:
        amplitudes = np.full(len(site_labels), value, dtype=float)
    if not np.isfinite(amplitudes).all() or (amplitudes < 0.0).any():
        raise ValueError("`fixed_site_amplitudes` must contain finite, non-negative values.")
    return amplitudes


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


def fixed_ou_likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    pollution_mean: TensorVariable,
    pollution_event_baseline: TensorVariable | None,
    output_dim: str,
    tau_hours: float | Mapping[str, float],
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> TensorVariable:
    """Add labelled fixed-tau within-site OU model-data mismatch.

    The complete covariance is the selected fixed aggregation covariance,
    reported observation-error variance, and one OU block per site. The
    ``minimum_error`` floor is applied to the fixed aggregation-plus-reported
    marginal before OU variance is added. Consequently OU variance can only
    increase the final marginal above that floor.

    ``tau_hours`` and fixed amplitudes may be scalar or exact mappings keyed by
    the observation sites. When amplitudes are inferred, their default prior
    is an independent ``HalfNormal(sigma=0.75)`` in observation units.

    Args:
        observations: Observed mole fractions carrying aligned ``site`` and
            ``time`` coordinates.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum fixed-base marginal standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        mean: Completed forward-model concentration.
        pollution_mean: Modelled pollution contribution, unused by this
            additive covariance component.
        pollution_event_baseline: Baseline used by pollution-event likelihoods,
            unused by this additive covariance component.
        output_dim: Observation dimension used for named PyMC variables.
        tau_hours: Fixed positive OU correlation time in hours, globally or by
            exact site-label mapping.
        fixed_site_amplitudes: Optional known non-negative OU standard
            deviations in observation units, globally or by site.
        site_amplitude_prior: Optional positive-support PyMC prior arguments
            for inferred site amplitudes. Mutually exclusive with fixed values.

    Returns:
        The observed Gaussian variable named ``y``. The component also creates
        canonical ``epsilon`` and labelled ``ou_site_amplitude`` variables.

    Raises:
        ValueError: If labels, configuration, or covariance inputs are invalid.
    """
    del pollution_mean, pollution_event_baseline
    if fixed_site_amplitudes is not None and site_amplitude_prior is not None:
        raise ValueError(
            "Pass either `fixed_site_amplitudes` or `site_amplitude_prior`, not both."
        )
    validate_observation_error_arrays(
        observations,
        observation_error,
        minimum_error,
        owner="Fixed-OU likelihood",
        output_dim=output_dim,
    )
    site = observations.coords.get("site")
    time = observations.coords.get("time")
    if site is None or site.dims != (output_dim,):
        raise ValueError(
            "The fixed-OU likelihood requires an observation-aligned 'site' coordinate."
        )
    if time is None or time.dims != (output_dim,):
        raise ValueError(
            "The fixed-OU likelihood requires an observation-aligned 'time' coordinate."
        )

    site_index = make_site_indicator(site)
    site_codes = np.asarray(site_index.values, dtype=np.int64)
    site_values = np.asarray(site.values)
    site_labels = tuple(
        str(site_values[np.flatnonzero(site_codes == index)[0]])
        for index in range(int(site_codes.max()) + 1)
    )
    add_coords({"ou_site": np.asarray(site_labels, dtype=object)})

    # Register the original labelled fixed inputs before crossing the eager
    # fixed-OU precomputation boundary.
    observed = add_model_data(observations.transpose(output_dim), "Y")
    add_model_data(observation_error.transpose(output_dim), "error")
    add_model_data(minimum_error.transpose(output_dim), "min_error")
    add_aggregation_error_data(aggregation_error, observations, output_dim=output_dim)
    add_model_data(site_index.rename("ou_site_index"), "ou_site_index")

    observation_variance = np.square(np.asarray(observation_error.values, dtype=float))
    minimum_variance = np.square(np.asarray(minimum_error.values, dtype=float))
    fixed_marginal = observation_variance + aggregation_error.marginal_variance
    floor_variance = np.maximum(minimum_variance - fixed_marginal, 0.0)
    factor, aggregation_diagonal = _aggregation_low_rank_factor(aggregation_error)
    prepared = prepare_fixed_ou_low_rank(
        factor,
        observation_variance + aggregation_diagonal + floor_variance,
        np.asarray(time.values),
        site_codes,
        tau_hours,
        site_labels=site_labels,
    )
    add_model_data(
        xr.DataArray(
            prepared.tau_hours_by_site,
            dims=("ou_site",),
            coords={"ou_site": np.asarray(site_labels, dtype=object)},
            name="ou_tau_hours",
        )
    )

    if fixed_site_amplitudes is None:
        amplitude = parse_prior(
            "ou_site_amplitude",
            dict(
                DEFAULT_OU_SITE_AMPLITUDE_PRIOR
                if site_amplitude_prior is None
                else site_amplitude_prior
            ),
            dims="ou_site",
        )
    else:
        fixed_amplitude_values = _fixed_site_amplitudes(
            fixed_site_amplitudes, site_labels
        )
        mode_variance = prepared.mode_eigenvalues + np.square(
            fixed_amplitude_values[prepared.mode_site_index]
        )
        if np.any(mode_variance <= 0.0):
            raise ValueError(
                "Fixed OU amplitudes and the fixed base must produce a positive-definite "
                "complete observation covariance."
            )
        amplitude = add_model_data(
            xr.DataArray(
                fixed_amplitude_values,
                dims=("ou_site",),
                coords={"ou_site": np.asarray(site_labels, dtype=object)},
                name="ou_site_amplitude",
            )
        )

    pm.Deterministic(
        "epsilon",
        pt.sqrt(prepared.marginal_variance(amplitude)),
        dims=output_dim,
    )
    return cast(
        TensorVariable,
        pm.CustomDist(
            "y",
            mean,
            amplitude,
            logp=prepared.logp,
            random=prepared.random,
            signature="(n),(s)->(n)",
            observed=observed,
            dims=output_dim,
        ),
    )


__all__ = [
    "DEFAULT_ADDITIVE_SIGMA_PRIOR",
    "DEFAULT_OU_SITE_AMPLITUDE_PRIOR",
    "additive_sigma_likelihood_builder",
    "fixed_ou_likelihood_builder",
]
