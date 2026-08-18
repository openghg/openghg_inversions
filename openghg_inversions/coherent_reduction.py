"""Exact labelled Gaussian reduction from one native model.

For ``x ~ N(m, B)`` and ``alpha = Pi x``, the reduced conditional
observation model is

``y | alpha ~ N(H m + H_alpha (alpha - Pi m), R + A)``,

where ``H_alpha = H B Pi.T C_alpha^-1`` and
``A = H B H.T - H_alpha C_alpha H_alpha.T``. The public function constructs
all linked covariance products from the same ``B/H/Pi`` inputs before applying
these equations.
"""

from __future__ import annotations

from dataclasses import dataclass

from dask.base import compute
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    RetainedProjectionStrategy,
    project_native_covariance,
)
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction


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
    basis_prolongation: xr.DataArray,
    state_dim: str,
    native_mean: xr.DataArray,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_batch_size: int = 64,
    strategy: RetainedProjectionStrategy | None = None,
) -> CoherentGaussianReduction:
    """Prepare the exact centred Gaussian model for one ``B/H/Pi/m`` set.

    This is a named eager numerical boundary. ``native_mean`` and
    ``native_sensitivity`` may be Dask-backed; they are materialized together
    so a shared graph is executed once. Inputs are borrowed and are not mutated.

    The inputs must use the covariance action's native dimensions and carry
    exactly matching indexed coordinates on shared dimensions. Unit conversion
    belongs to the upstream OpenGHG/pint-xarray preparation boundary: ``m`` and
    the basis describe dimensionless scaling, while ``H`` is already expressed
    in the desired observation units.

    Args:
        covariance: Invertible labelled native covariance action ``B``.
        basis_prolongation: Canonical eager basis prolongation used by the
            retained projection strategy.
        state_dim: Retained-state dimension shared by the prolongation and
            restriction.
        native_mean: Dimensionless native scaling mean ``m``.
        native_sensitivity: Native sensitivity ``H`` in canonical observation
            units.
        observation_dim: Observation dimension in ``native_sensitivity``.
        observation_batch_size: Covariance right-hand-side batch size.
        strategy: Optional authoritative retained restriction strategy.

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
    prolongation = basis_prolongation.transpose(*native_dims, state_dim)

    # xr.dot uses an inner join by default, which could silently discard native
    # cells. Establish exact compatibility once before any eager computation.
    mean, sensitivity, prolongation = xr.align(
        mean,
        sensitivity,
        prolongation,
        join="exact",
        copy=False,
    )
    mean, sensitivity = compute(to_dense(mean), to_dense(sensitivity))

    products = project_native_covariance(
        covariance=covariance,
        basis_prolongation=prolongation,
        state_dim=state_dim,
        native_sensitivity=sensitivity,
        observation_dim=observation_dim,
        observation_covariance="dense",
        observation_batch_size=observation_batch_size,
        strategy=strategy,
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
