"""Ordinary state component for the calibrated retained biosphere prior."""

from __future__ import annotations

from typing import Any
import json

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.components import (
    StateVectorResult,
    add_correlated_lognormal_state_with_activity,
)
from openghg_inversions.models.coords import add_coords, add_model_data
from openghg_inversions.models.state_activity import ResolvedStateActivity


def add_biosphere_state(
    activity: ResolvedStateActivity,
    baseline: CorrelatedLognormalPrior,
    calibration: BiospherePrior,
    *,
    var_name: str = "flux_scaling",
) -> StateVectorResult:
    """Build the declared prior once for one shared flux state.

    The positive family reuses the standard correlated lognormal component.
    The diagnostic uses one whitened Normal vector with protected lognormal
    marginal and linearly regressed Gaussian biosphere residuals. It preserves
    nonzero biosphere/protected cross-covariances. Fixed output entries follow
    the supplied activity policy; diagnostic latent states are marginalized by
    retaining their full draw before substituting those fixed entries.
    """
    baseline_mean, calibrated_mean, _active = xr.align(
        baseline.mean,
        calibration.baseline.mean,
        activity.active,
        join="exact",
        copy=False,
    )
    if not (
        np.array_equal(baseline_mean, calibrated_mean)
        and np.array_equal(
            baseline.arithmetic_covariance,
            calibration.baseline.arithmetic_covariance,
        )
    ):
        raise ValueError("Biosphere calibration must use this model's baseline arithmetic moments.")
    model = pm.modelcontext(None)
    # The trace annotators persist this concrete scientific product as a group.
    model.biosphere_prior = calibration
    if calibration.positive_prior is not None:
        return add_correlated_lognormal_state_with_activity(
            activity, calibration.positive_prior, var_name=var_name
        )

    dim = baseline.state_dim
    mean = calibration.data.baseline_mean
    m = mean.to_numpy()
    b = calibration.data.covariance.to_numpy()
    bio = np.flatnonzero(calibration.data.biosphere.to_numpy())
    other = np.flatnonzero(~calibration.data.biosphere.to_numpy())
    if other.size:
        protected = CorrelatedLognormalPrior(
            mean.isel({dim: other}),
            b[np.ix_(other, other)],
            covariance_dim=baseline.covariance_dim,
        )
        regression = np.linalg.solve(b[np.ix_(other, other)], b[np.ix_(other, bio)]).T
        conditional = b[np.ix_(bio, bio)] - regression @ b[np.ix_(other, bio)]
    else:
        regression = np.empty((len(bio), 0))
        conditional = b[np.ix_(bio, bio)]
    cholesky = np.linalg.cholesky((conditional + conditional.T) / 2)
    add_coords(mean.coords, model_dims=(dim,))
    latent = pm.Normal(f"{var_name}_latent", 0, 1, dims=dim)
    state = pt.as_tensor_variable(m)
    if other.size:
        x_other = pt.exp(
            protected.latent_mean.to_numpy() + protected.latent_cholesky.to_numpy() @ latent[other]
        )
        state = pt.set_subtensor(state[other], x_other)
        x_bio = m[bio] + regression @ (x_other - m[other]) + cholesky @ latent[bio]
    else:
        x_bio = m[bio] + cholesky @ latent[bio]
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


def annotate_biosphere_prior(trace: Any, calibration: BiospherePrior | None) -> Any:
    """Persist calibration, baseline covariance, support and report beside a trace.

    Works with both ArviZ InferenceData and the recipe's DataTree trace. This
    explicit output boundary should also be called after direct PyMC sampling.
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
