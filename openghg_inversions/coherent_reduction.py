"""Exact labelled Gaussian reduction from one native model.

For ``x ~ N(m, B)`` and ``alpha = Pi x``, the reduced conditional
observation model is

``y | alpha ~ N(H m + H_alpha (alpha - Pi m), R + A)``,

where ``H_alpha = H B Pi.T C_alpha^-1`` and
``A = H B H.T - H_alpha C_alpha H_alpha.T``. The public function constructs
all linked covariance products from the same ``B/H/Pi`` inputs before applying
these equations.

:func:`marginalise_low_information_states` optionally applies a
``StateActivity.min_information`` threshold to one reduction. Retained states
whose prior-predicted information falls below the threshold are integrated out
exactly: their effective-operator columns become zero, so model construction
treats them as zero-sensitivity states, and their unresolved variance is added
to ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.array_ops import materialize_together
from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    RetainedProjectionStrategy,
    project_native_covariance,
)
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

if TYPE_CHECKING:
    from openghg_inversions.models.state_activity import StateActivity


@dataclass(frozen=True, slots=True, eq=False)
class CoherentGaussianReduction:
    """One exact, labelled native-to-retained Gaussian reduction.

    Attributes:
        retained_mean: Dimensionless ``Pi m`` on the retained-state axis.
        retained_covariance: Positive-definite ``C_alpha`` on distinct
            retained-state row and column axes.
        effective_observation_operator: ``H_alpha`` mapping retained-state
            perturbations to observations.
        native_observation_mean: ``H m`` on the observation axis.
        observation_intercept: Affine intercept ``H m - H_alpha Pi m``.
        unresolved_observation_covariance: ``A`` on distinct observation row
            and column axes. It is a positive-semidefinite covariance up to
            floating-point roundoff.
        projection_strategy: Scientific strategy that selected ``Pi``.
    """

    retained_mean: xr.DataArray
    retained_covariance: xr.DataArray
    effective_observation_operator: xr.DataArray
    native_observation_mean: xr.DataArray
    observation_intercept: xr.DataArray
    unresolved_observation_covariance: xr.DataArray
    projection_strategy: str


def reduce_native_gaussian(
    *,
    covariance: InvertibleNativeCovarianceAction,
    strategy: RetainedProjectionStrategy,
    state_dim: str,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_batch_size: int = 64,
) -> CoherentGaussianReduction:
    """Prepare the exact centred Gaussian model for one ``B/H/Pi/m`` set.

    This is a named eager numerical boundary. ``native_mean`` and
    ``native_sensitivity`` may be Dask-backed; their payloads are materialized
    together so a shared graph is executed once. The strategy then selects and
    materializes ``Pi`` from its own inputs. Inputs are borrowed and are not
    mutated.

    The inputs must use the covariance action's native dimensions and carry
    exactly matching indexed coordinates on shared dimensions. Unit conversion
    belongs to the upstream OpenGHG/pint-xarray preparation boundary: ``m`` and
    ``Pi`` describe dimensionless scaling, while ``H`` is already expressed
    in the desired observation units.

    Args:
        covariance: Invertible labelled native covariance action ``B``.
        strategy: Authoritative retained restriction strategy, owning the
            inputs that define ``Pi``.
        state_dim: Retained-state dimension of ``Pi``.
        native_mean: Dimensionless native scaling mean ``m``.
        native_sensitivity: Native sensitivity ``H`` in canonical observation
            units.
        observation_dim: Observation dimension in ``native_sensitivity``.
        observation_batch_size: Covariance right-hand-side batch size.

    Returns:
        The retained prior, centred effective forward model, and unresolved
        observation covariance.

    Raises:
        ValueError: If xarray cannot transpose or exactly align the labelled
            inputs, or if covariance-product construction fails.
    """
    native_dims = tuple(covariance.native_dims)
    mean = native_mean.transpose(*native_dims)
    sensitivity = native_sensitivity.transpose(observation_dim, *native_dims)

    # The covariance action checks the grids of H and Pi when it is applied,
    # but m meets only xr.dot, whose default inner join could silently discard
    # native cells. This index-only check executes no lazy payload.
    mean, sensitivity = xr.align(mean, sensitivity, join="exact", copy=False)
    mean, sensitivity = materialize_together(mean, sensitivity, purpose="native mean m and sensitivity H")

    products = project_native_covariance(
        covariance=covariance,
        strategy=strategy,
        state_dim=state_dim,
        native_sensitivity=sensitivity,
        observation_dim=observation_dim,
        observation_covariance="dense",
        observation_batch_size=observation_batch_size,
    )
    return _reduce_native_covariance_products(
        products=products,
        native_mean=mean,
        native_sensitivity=sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
        observation_dim=observation_dim,
    )


def _reduce_native_covariance_products(
    *,
    products: NativeCovarianceProducts,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    native_dims: tuple[str, ...],
    state_dim: str,
    observation_dim: str,
) -> CoherentGaussianReduction:
    """Apply the reduction equations to products constructed immediately above."""
    restriction = products.restriction
    state_covariance = products.state_covariance
    effective_operator = products.effective_observation_operator
    native_observation_covariance = products.native_observation_covariance

    retained_mean = xr.dot(restriction, native_mean, dim=list(native_dims)).transpose(state_dim)
    retained_mean = retained_mean.rename("retained_mean").assign_attrs(
        mathematical_name="Pi m",
        units="1",
    )
    native_observation_mean = xr.dot(
        native_sensitivity,
        native_mean,
        dim=list(native_dims),
    ).transpose(observation_dim)
    sensitivity_units = native_sensitivity.attrs.get("units")
    linear_units = {"units": sensitivity_units} if sensitivity_units is not None else {}
    native_observation_mean = native_observation_mean.rename("native_observation_mean").assign_attrs(
        mathematical_name="H m",
        **linear_units,
    )
    observation_intercept = (
        native_observation_mean
        - xr.dot(effective_operator, retained_mean, dim=state_dim)
    ).rename("observation_intercept")
    observation_intercept = observation_intercept.assign_attrs(
        mathematical_name="H m - H_alpha Pi m",
        **linear_units,
    )

    # A = H B H.T - H_alpha C_alpha H_alpha.T, formed positionally in NumPy.
    # H_alpha's state columns and both C_alpha axes follow Pi's state index,
    # so their orders already agree.
    unresolved_values = (
        np.asarray(native_observation_covariance.values, dtype=np.float64)
        - np.asarray(effective_operator.values, dtype=np.float64)
        @ np.asarray(state_covariance.values, dtype=np.float64)
        @ np.asarray(effective_operator.values, dtype=np.float64).T
    )
    # The two algebraically symmetric terms can differ at roundoff after the
    # retained solve. Symmetrization defines the numerical covariance returned
    # by this eager kernel; scientific PSD identities remain covered by tests.
    unresolved_values = (unresolved_values + unresolved_values.T) * 0.5
    unresolved_covariance = native_observation_covariance.copy(
        data=unresolved_values
    ).rename("unresolved_observation_covariance")
    unresolved_covariance = unresolved_covariance.assign_attrs(
        mathematical_name="A = H B_perp H.T",
        definition="H B H.T - H_alpha C_alpha H_alpha.T",
    )

    return CoherentGaussianReduction(
        retained_mean=retained_mean,
        retained_covariance=state_covariance.rename("retained_covariance"),
        effective_observation_operator=effective_operator,
        native_observation_mean=native_observation_mean,
        observation_intercept=observation_intercept,
        unresolved_observation_covariance=unresolved_covariance,
        projection_strategy=products.strategy,
    )


@dataclass(frozen=True, slots=True, eq=False)
class InformationSelection:
    """A coherent reduction after applying a prior-information threshold.

    Attributes:
        reduction: Reduction whose marginalised states have zero effective
            operator columns. It is the input reduction object itself when no
            state was marginalised.
        information: Prior-predicted information
            ``I_j = sum_o (H_alpha[o, j] s_j)**2 / sigma_o**2`` on the retained
            state axis, where ``s_j`` is the retained prior standard deviation.
        threshold: Threshold applied to each state; NaN where none applies.
        marginalised: Boolean mask of states moved into ``A``.
        unresolved_variance_increase: Change in ``diag A`` on the observation
            axis, in squared observation units.
    """

    reduction: CoherentGaussianReduction
    information: xr.DataArray
    threshold: xr.DataArray
    marginalised: xr.DataArray
    unresolved_variance_increase: xr.DataArray

    def to_dataset(self) -> xr.Dataset:
        """Return the per-state and per-observation diagnostics for saving."""
        return xr.Dataset(
            {
                "state_information": self.information,
                "state_information_threshold": self.threshold,
                "state_information_marginalised": self.marginalised,
                "unresolved_variance_increase": self.unresolved_variance_increase,
            }
        )

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the thresholds and marginalised states."""
        state_dim = str(self.information.dims[0])
        labels = self.information.indexes[state_dim]
        mask = np.asarray(self.marginalised.values, dtype=bool)
        information = np.asarray(self.information.values, dtype=float)
        threshold = np.asarray(self.threshold.values, dtype=float)
        increase = np.asarray(self.unresolved_variance_increase.values, dtype=float)
        return {
            "criterion": "I_j = sum_o (H_alpha[o, j] * s_j)**2 / sigma_o**2 < min_information_j",
            "n_states": int(mask.size),
            "n_marginalised": int(mask.sum()),
            "marginalised_states": [
                {
                    "state": _json_label(labels[i]),
                    "information": float(information[i]),
                    "threshold": float(threshold[i]),
                }
                for i in np.flatnonzero(mask)
            ],
            "min_retained_information": (
                float(information[~mask].min()) if (~mask).any() else None
            ),
            "max_unresolved_variance_increase": float(increase.max()) if increase.size else 0.0,
            "total_unresolved_variance_increase": float(increase.sum()),
        }


def _json_label(label: Any) -> Any:
    """Return a JSON-safe state label, keeping numbers numeric."""
    if isinstance(label, tuple):
        return [_json_label(item) for item in label]
    if isinstance(label, np.generic):
        label = label.item()
    return json.loads(json.dumps(label, default=str))


def retained_state_information(
    reduction: CoherentGaussianReduction,
    observation_variance: xr.DataArray,
) -> xr.DataArray:
    """Return each retained state's prior-predicted information.

    For retained prior standard deviation ``s_j = sqrt(C_alpha[j, j])``,
    effective operator ``H_alpha`` and prior error variance ``sigma_o**2``,
    ``I_j = sum_o (H_alpha[o, j] s_j)**2 / sigma_o**2``. It is the expected
    squared signal-to-noise ratio of one prior standard deviation of state
    ``j`` and uses only the prior and the operator, never observed values.

    Args:
        reduction: Coherent reduction supplying ``H_alpha`` and ``C_alpha``.
        observation_variance: Positive finite prior observation-error or
            model-mismatch variance on the reduction's observation dimension
            and exact labels, in squared observation units. Do not include
            ``A`` itself.

    Returns:
        Dimensionless information on the retained-state axis, carrying the
        reduction's state coordinates.

    Raises:
        ValueError: If the variance is not one-dimensional on the observation
            axis with matching labels, or is non-finite or non-positive.
    """
    operator = reduction.effective_observation_operator
    observation_dim = str(operator.dims[0])
    if observation_variance.dims != (observation_dim,):
        raise ValueError(
            f"observation_variance must have dimensions ({observation_dim!r},); "
            f"found {observation_variance.dims!r}."
        )
    variance, _ = xr.align(observation_variance, operator, join="exact", copy=False)
    variance_values = np.asarray(to_dense(variance).values, dtype=np.float64)
    if not np.isfinite(variance_values).all() or (variance_values <= 0).any():
        raise ValueError("observation_variance must be finite and positive.")
    operator_values = np.asarray(operator.values, dtype=np.float64)
    prior_variance = np.diag(np.asarray(reduction.retained_covariance.values, dtype=np.float64))
    information = prior_variance * np.sum(operator_values**2 / variance_values[:, None], axis=0)
    return reduction.retained_mean.copy(data=information).rename("state_information").assign_attrs(
        units="1",
        long_name="prior-predicted state information",
        mathematical_name="sum_o (H_alpha[o, j] s_j)**2 / sigma_o**2",
    )


def marginalise_retained_states(
    reduction: CoherentGaussianReduction,
    marginalised: xr.DataArray,
) -> CoherentGaussianReduction:
    """Integrate selected retained states out of a coherent reduction exactly.

    Split the retained states into kept ``K`` and marginalised ``D``. With
    ``G = C_DK C_KK^-1`` and ``S = C_DD - G C_KD``, the exact conditional
    model given ``alpha_K`` is

    ``y | alpha_K ~ N(H m + H'_K (alpha_K - Pi_K m), R + A')``,

    where ``H'_K = H_K + H_D G`` and ``A' = A + H_D S H_D.T``. This is the
    coherent reduction with restriction ``Pi_K``; it equals re-reducing with
    the smaller bucket basis whenever the marginalised states are prior
    uncorrelated with the kept ones.

    The returned reduction keeps the full retained-state axis so labels,
    prior moments and state-aligned metadata are unchanged. Marginalised
    states have zero effective-operator columns, so model construction removes
    them as zero-sensitivity states and holds them at their fixed values.

    Args:
        reduction: Coherent reduction to marginalise.
        marginalised: Boolean mask on the retained-state axis with exactly the
            reduction's state labels.

    Returns:
        ``reduction`` itself when no state is selected, otherwise a new
        reduction with the updated operator, intercept and unresolved
        covariance.

    Raises:
        ValueError: If the mask is not a boolean state-aligned array.
    """
    state_dim = str(reduction.retained_mean.dims[0])
    if marginalised.dims != (state_dim,):
        raise ValueError(f"marginalised must have dimensions ({state_dim!r},).")
    mask, _ = xr.align(marginalised, reduction.retained_mean, join="exact", copy=False)
    drop = np.asarray(mask.values)
    if drop.dtype != np.dtype(bool):
        raise ValueError("marginalised must contain only boolean values.")
    if not drop.any():
        return reduction
    keep = ~drop

    covariance = np.asarray(reduction.retained_covariance.values, dtype=np.float64)
    operator = np.asarray(reduction.effective_observation_operator.values, dtype=np.float64)
    operator_d = operator[:, drop]
    c_dd = covariance[np.ix_(drop, drop)]
    if keep.any():
        c_kd = covariance[np.ix_(keep, drop)]
        gain = cho_solve(cho_factor(covariance[np.ix_(keep, keep)]), c_kd).T
        schur = c_dd - gain @ c_kd
    else:
        gain = np.zeros((int(drop.sum()), 0))
        schur = c_dd
    schur = (schur + schur.T) * 0.5

    updated_operator = np.zeros_like(operator)
    updated_operator[:, keep] = operator[:, keep] + operator_d @ gain
    unresolved = np.asarray(reduction.unresolved_observation_covariance.values, dtype=np.float64)
    unresolved = unresolved + operator_d @ schur @ operator_d.T
    unresolved = (unresolved + unresolved.T) * 0.5

    effective = reduction.effective_observation_operator.copy(data=updated_operator)
    intercept = (
        reduction.native_observation_mean
        - xr.dot(effective, reduction.retained_mean, dim=state_dim)
    ).rename("observation_intercept")
    intercept = intercept.assign_attrs(reduction.observation_intercept.attrs)
    return CoherentGaussianReduction(
        retained_mean=reduction.retained_mean,
        retained_covariance=reduction.retained_covariance,
        effective_observation_operator=effective,
        native_observation_mean=reduction.native_observation_mean,
        observation_intercept=intercept,
        unresolved_observation_covariance=reduction.unresolved_observation_covariance.copy(
            data=unresolved
        ),
        projection_strategy=reduction.projection_strategy,
    )


def marginalise_low_information_states(
    reduction: CoherentGaussianReduction,
    *,
    observation_variance: xr.DataArray,
    policy: StateActivity | None,
) -> InformationSelection:
    """Apply ``policy.min_information`` to one coherent reduction.

    States with :func:`retained_state_information` below their threshold are
    integrated out by :func:`marginalise_retained_states`. Thresholds may
    apply to every state or only to labelled groups, see
    :class:`~openghg_inversions.models.StateActivity`. Only
    ``min_information`` and ``group_coord`` are read from the policy; its
    explicit activity mask, fixed values and fixed groups remain model-time
    choices.

    Marginalised states receive no posterior update. Retained-state
    reconstructions hold them at their fixed values, which should equal the
    retained prior mean, and uncertainty computed only from retained states
    omits their contribution.

    Args:
        reduction: Coherent reduction whose states are screened.
        observation_variance: Prior observation-error or model-mismatch
            variance per observation; see :func:`retained_state_information`.
        policy: Activity policy. ``None`` or ``min_information=None`` keeps
            every state and returns the input reduction object unchanged.

    Returns:
        The possibly marginalised reduction and its diagnostics.

    Raises:
        ValueError: If the variance or grouped thresholds are invalid.
    """
    from openghg_inversions.models.state_activity import information_thresholds

    information = retained_state_information(reduction, observation_variance)
    if policy is None or policy.min_information is None:
        threshold = xr.full_like(information, np.nan).rename("information_threshold")
    else:
        threshold = information_thresholds(reduction.retained_mean, policy)
    threshold = threshold.assign_attrs(units="1")
    marginalised = (information < threshold).rename("information_marginalised")
    selected = marginalise_retained_states(reduction, marginalised)
    before = reduction.unresolved_observation_covariance
    increase = np.diag(np.asarray(selected.unresolved_observation_covariance.values)) - np.diag(
        np.asarray(before.values)
    )
    observation_dim = str(before.dims[0])
    increase_array = xr.DataArray(
        increase,
        dims=(observation_dim,),
        coords={
            name: coord
            for name, coord in reduction.observation_intercept.coords.items()
            if set(coord.dims) <= {observation_dim}
        },
        name="unresolved_variance_increase",
        attrs={
            key: value for key, value in (("units", before.attrs.get("units")),) if value is not None
        },
    )
    return InformationSelection(
        reduction=selected,
        information=information,
        threshold=threshold,
        marginalised=marginalised,
        unresolved_variance_increase=increase_array,
    )
