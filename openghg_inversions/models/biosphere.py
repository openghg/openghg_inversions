"""Retained flux-state prior: the correlated lognormal or a biosphere calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias
import json

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.components import (
    StateVectorResult,
    _add_prepared_correlated_lognormal_state_with_activity,
    prepare_active_correlated_lognormal_prior,
)
from openghg_inversions.models.coords import add_coords, add_model_data
from openghg_inversions.models.state_activity import ResolvedStateActivity


RetainedFluxPrior: TypeAlias = CorrelatedLognormalPrior | BiospherePrior
"""The prior a CO2 model samples for its retained flux state.

Either the prepared correlated lognormal prior, or a :class:`BiospherePrior`
already checked against it by :func:`resolve_retained_flux_prior`.
"""


@dataclass(frozen=True)
class PreparedGaussianBiosphere:
    """Eager factors for the signed Gaussian biosphere diagnostic.

    Protected states are lognormal; biosphere states are their linear
    regression on the protected states plus a Gaussian conditional residual.
    Arrays are small, dense and borrowed: do not mutate them.

    Attributes:
        mean: Labelled baseline arithmetic mean on the state dimension.
        protected: Positional indices of the protected lognormal states.
        biosphere: Positional indices of the biosphere states.
        protected_latent_mean: Latent Gaussian mean of the protected states.
        protected_latent_cholesky: Latent Cholesky factor of the protected states.
        regression: Biosphere-on-protected regression coefficients.
        cholesky: Cholesky factor of the biosphere conditional covariance.
    """

    mean: xr.DataArray
    protected: np.ndarray
    biosphere: np.ndarray
    protected_latent_mean: np.ndarray
    protected_latent_cholesky: np.ndarray
    regression: np.ndarray
    cholesky: np.ndarray


PreparedRetainedFluxPrior: TypeAlias = CorrelatedLognormalPrior | PreparedGaussianBiosphere | None
"""Eager active-state form of a :data:`RetainedFluxPrior`.

A correlated lognormal prior on the active states, the Gaussian diagnostic's
factors, or ``None`` when every state is fixed.
"""


def resolve_retained_flux_prior(
    retained_prior: CorrelatedLognormalPrior,
    calibration: BiospherePrior | None,
) -> RetainedFluxPrior:
    """Choose the flux prior a model samples, checking a calibration once.

    Runners call this where they first hold both the prepared retained prior
    and the requested calibration. Model builders then trust a
    :class:`BiospherePrior` as the model's prior.

    Args:
        retained_prior: The model's prepared arithmetic-moment prior.
        calibration: Optional calibration of that same prior.

    Returns:
        ``calibration`` when supplied, otherwise ``retained_prior``.

    Raises:
        ValueError: If the calibration's baseline labels or arithmetic moments
            differ from ``retained_prior``.
    """
    if calibration is None:
        return retained_prior
    retained_mean, calibrated_mean = xr.align(
        retained_prior.mean, calibration.baseline.mean, join="exact", copy=False,
    )
    if not (
        np.array_equal(retained_mean, calibrated_mean)
        and np.array_equal(
            retained_prior.arithmetic_covariance,
            calibration.baseline.arithmetic_covariance,
        )
    ):
        raise ValueError("Biosphere calibration must use this model's baseline arithmetic moments.")
    return calibration


def retained_flux_prior_family(prior: RetainedFluxPrior) -> str:
    """Return the ``prior`` model metadata recorded for a flux prior.

    Args:
        prior: The flux prior the model samples.

    Returns:
        The calibration family, or the correlated lognormal description.
    """
    if isinstance(prior, BiospherePrior):
        return str(prior.data.attrs["family"])
    return "correlated arithmetic-moment lognormal"


def prepare_retained_flux_prior(
    activity: ResolvedStateActivity,
    prior: RetainedFluxPrior,
    /,
    *,
    var_name: str,
) -> PreparedRetainedFluxPrior:
    """Do the eager work for one flux state before graph construction.

    The correlated lognormal prior, and the positive biosphere family, are
    subset to the active states. The Gaussian diagnostic computes its
    protected/biosphere regression and conditional Cholesky factor on the
    full state; inactive outputs are replaced after the joint draw.

    Args:
        activity: Resolved labelled active/fixed policy for the full flux state.
        prior: The flux prior the model samples.
        var_name: Full physical state name.

    Returns:
        The prepared prior for :func:`add_prepared_retained_flux_state`.

    Raises:
        ValueError: If the prior and activity labels differ.
    """
    if isinstance(prior, CorrelatedLognormalPrior):
        return prepare_active_correlated_lognormal_prior(activity, prior, var_name=var_name)
    if prior.positive_prior is not None:
        return prepare_active_correlated_lognormal_prior(activity, prior.positive_prior, var_name=var_name)
    if activity.n_active == 0:
        return prepare_active_correlated_lognormal_prior(activity, prior.baseline, var_name=var_name)
    return _prepare_gaussian_biosphere(activity, prior)


def _prepare_gaussian_biosphere(
    activity: ResolvedStateActivity, calibration: BiospherePrior,
) -> PreparedGaussianBiosphere:
    """Compute the Gaussian diagnostic's regression and conditional factors."""
    mean = calibration.data.baseline_mean
    xr.align(mean, activity.active, join="exact", copy=False)
    b = calibration.data.covariance.to_numpy()
    selector = calibration.data.biosphere.to_numpy()
    bio, other = np.flatnonzero(selector), np.flatnonzero(~selector)
    if other.size:
        protected = CorrelatedLognormalPrior(
            mean.isel({calibration.baseline.state_dim: other}),
            b[np.ix_(other, other)],
            covariance_dim=calibration.baseline.covariance_dim,
        )
        protected_latent_mean = protected.latent_mean.to_numpy()
        protected_latent_cholesky = protected.latent_cholesky.to_numpy()
        regression = np.linalg.solve(b[np.ix_(other, other)], b[np.ix_(other, bio)]).T
        conditional = b[np.ix_(bio, bio)] - regression @ b[np.ix_(other, bio)]
    else:
        protected_latent_mean = np.empty(0)
        protected_latent_cholesky = np.empty((0, 0))
        regression = np.empty((len(bio), 0))
        conditional = b[np.ix_(bio, bio)]
    return PreparedGaussianBiosphere(
        mean=mean,
        protected=other,
        biosphere=bio,
        protected_latent_mean=protected_latent_mean,
        protected_latent_cholesky=protected_latent_cholesky,
        regression=regression,
        cholesky=np.linalg.cholesky((conditional + conditional.T) / 2),
    )


def add_prepared_retained_flux_state(
    activity: ResolvedStateActivity,
    prepared: PreparedRetainedFluxPrior,
    /,
    *,
    var_name: str,
) -> StateVectorResult:
    """Wire a prepared flux prior into the active model.

    Lognormal priors use the standard correlated lognormal component. The
    Gaussian diagnostic uses one whitened Normal vector with protected
    lognormal marginal and linearly regressed Gaussian biosphere residuals,
    preserving nonzero biosphere/protected cross-covariances. It constructs
    the full joint draw before replacing inactive outputs, retaining the exact
    active marginal.

    Args:
        activity: The activity passed to :func:`prepare_retained_flux_prior`.
        prepared: Its result.
        var_name: Full physical state name; whitened sampled variables use a suffix.

    Returns:
        The physical flux coefficients, sampled latent vector if any, and activity.
    """
    if not isinstance(prepared, PreparedGaussianBiosphere):
        return _add_prepared_correlated_lognormal_state_with_activity(
            activity, prepared, var_name=var_name,
        )

    mean = prepared.mean
    dim = str(mean.dims[0])
    m = mean.to_numpy()
    bio, other = prepared.biosphere, prepared.protected
    add_coords(mean.coords, model_dims=(dim,))
    latent = pm.Normal(f"{var_name}_latent", 0, 1, dims=dim)
    state = pt.as_tensor_variable(m)
    if other.size:
        x_other = pt.exp(
            prepared.protected_latent_mean + prepared.protected_latent_cholesky @ latent[other]
        )
        state = pt.set_subtensor(state[other], x_other)
        x_bio = m[bio] + prepared.regression @ (x_other - m[other]) + prepared.cholesky @ latent[bio]
    else:
        x_bio = m[bio] + prepared.cholesky @ latent[bio]
    state = pt.set_subtensor(state[bio], x_bio)
    if activity.n_active != activity.n_state:
        add_model_data(activity.active.rename(f"{var_name}_is_active"))
        fixed = add_model_data(activity.fixed_value.rename(f"{var_name}_fixed_value"))
        state = pt.where(activity.active.to_numpy(), state, fixed)
    return StateVectorResult(
        latent=latent,
        state=pm.Deterministic(var_name, state, dims=dim),
        activity=activity,
    )


def add_retained_flux_state(
    activity: ResolvedStateActivity,
    prior: RetainedFluxPrior,
    /,
    *,
    var_name: str,
) -> StateVectorResult:
    """Prepare and add one shared flux state for its prior's type.

    Args:
        activity: Resolved labelled active/fixed policy for the full flux state.
        prior: The flux prior the model samples. A :class:`BiospherePrior` is
            trusted; :func:`resolve_retained_flux_prior` checks it once.
        var_name: Full physical state name; whitened sampled variables use a suffix.

    Returns:
        The physical flux coefficients, sampled latent vector if any, and activity.

    Raises:
        ValueError: If the prior and activity labels differ.
    """
    prepared = prepare_retained_flux_prior(activity, prior, var_name=var_name)
    return add_prepared_retained_flux_state(activity, prepared, var_name=var_name)


def annotate_biosphere_prior(trace: Any, calibration: BiospherePrior | None) -> Any:
    """Persist calibration, baseline covariance, support and report beside a trace.

    Works with both ArviZ InferenceData and the recipe's DataTree trace. This
    explicit output boundary should also be called after direct PyMC sampling.

    Args:
        trace: ArviZ InferenceData or DataTree trace, updated in place.
        calibration: Calibrated prior to record, or ``None`` to leave the trace
            unchanged.

    Returns:
        The same trace, with a ``biosphere_prior`` group, the calibration report
        attribute and updated ``rhime_model_metadata`` when ``calibration`` is
        given.
    """
    if calibration is not None:
        metadata = json.loads(trace.attrs.get("rhime_model_metadata", "{}"))
        metadata.update(
            prior=calibration.data.attrs["family"],
            biosphere_prior_scope="retained_state_prior_with_fixed_reduction_reconstruction_and_aggregation",
        )
        trace.attrs["rhime_model_metadata"] = json.dumps(metadata)
        if isinstance(trace, xr.DataTree):
            trace["biosphere_prior"] = xr.DataTree(calibration.to_dataset())
        else:
            trace.add_groups({"biosphere_prior": calibration.to_dataset()})
        trace.attrs["biosphere_prior"] = calibration.data.attrs["calibration_report"]
    return trace
