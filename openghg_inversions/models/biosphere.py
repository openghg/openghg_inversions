"""Apply a biosphere calibration to a model's retained prior and record it."""

from __future__ import annotations

from typing import Any
import json

import numpy as np
import xarray as xr

from openghg_inversions.biosphere_prior import BIOSPHERE_PRIOR_SCOPE, BiospherePrior
from openghg_inversions.correlated_state import CorrelatedPrior


def resolve_retained_flux_prior(
    retained_prior: CorrelatedPrior,
    calibration: BiospherePrior | None,
) -> CorrelatedPrior:
    """Choose the flux prior a model samples, checking a calibration once.

    Runners call this where they first hold both the prepared retained prior
    and the requested calibration. The calibrated prior keeps the retained
    prior's mean and marginal families and takes the calibrated arithmetic
    covariance, so a biosphere state whose variance changes has a different
    marginal distribution in the same family.

    Args:
        retained_prior: The model's prepared prior.
        calibration: Optional calibration of that same prior.

    Returns:
        ``retained_prior`` without a calibration, otherwise its mean and
        marginal families with the calibrated covariance.

    Raises:
        ValueError: If the calibration's baseline labels or arithmetic moments
            differ from ``retained_prior``, or the calibrated moments are
            invalid for its marginal families.
    """
    if calibration is None:
        return retained_prior
    data = calibration.data
    retained_mean, calibrated_mean = xr.align(retained_prior.mean, data.baseline_mean, join="exact", copy=False)
    if not (
        np.array_equal(retained_mean, calibrated_mean)
        and np.array_equal(retained_prior.arithmetic_covariance, data.baseline_covariance)
    ):
        raise ValueError("Biosphere calibration must use this model's baseline arithmetic moments.")
    return CorrelatedPrior(
        retained_prior.mean,
        data.covariance,
        marginal=retained_prior.marginal,
        covariance_dim=retained_prior.covariance_dim,
    )


def annotate_biosphere_prior(trace: Any, calibration: BiospherePrior | None) -> Any:
    """Persist a calibration beside a trace.

    Works with both ArviZ InferenceData and the recipe's DataTree trace. This
    explicit output boundary should also be called after direct PyMC sampling.

    Args:
        trace: ArviZ InferenceData or DataTree trace, updated in place.
        calibration: Calibration to record, or ``None`` to leave the trace
            unchanged.

    Returns:
        The same trace. With a calibration it has a ``biosphere_prior`` group
        and ``rhime_model_metadata`` records its ``biosphere_prior_scope``.
    """
    if calibration is not None:
        metadata = json.loads(trace.attrs.get("rhime_model_metadata", "{}"))
        metadata["biosphere_prior_scope"] = BIOSPHERE_PRIOR_SCOPE
        trace.attrs["rhime_model_metadata"] = json.dumps(metadata)
        if isinstance(trace, xr.DataTree):
            trace["biosphere_prior"] = xr.DataTree(calibration.to_dataset())
        else:
            trace.add_groups({"biosphere_prior": calibration.to_dataset()})
    return trace
