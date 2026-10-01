"""One common additive boundary correction and weighted-centred anomalies.

The additive mode replaces fitted multiplicative boundary scales. Its reference
contribution is H_bc @ 1, and its correction is G_bc @ (mean + anomaly).
See https://mc-stan.org/docs/stan-users-guide/regression.html#parameterizing-centered-vectors
for the non-redundant normal coordinates underlying the centring constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import dask
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import helmert
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.models.components import add_model_data, get_model_latent
from openghg_inversions.models.priors import PriorArgs, parse_prior
from openghg_inversions.models.state_activity import StateActivity


@dataclass(frozen=True)
class CenteredBoundary:
    """Eager labelled boundary field, centring basis, and affine design."""

    reference_sensitivity: xr.DataArray
    sensitivity: xr.DataArray
    weights: xr.DataArray
    basis: xr.DataArray
    design: np.ndarray


def prepare_centered_boundary(
    boundary_sensitivity: xr.DataArray | None,
    correction_sensitivity: xr.DataArray | None,
    weights: xr.DataArray | None,
    observations: xr.DataArray,
    mean_prior: PriorArgs | None,
    anomaly_scale: float | None,
    bc_prior: PriorArgs | None,
    bc_state_activity: StateActivity | None,
    *,
    output_dim: str = "nmeasure",
) -> CenteredBoundary | None:
    """Align and jointly materialize additive boundary inputs at model construction.

    Positive weights define the mean over all supplied curtain/period states,
    including states unobserved by the transport. The anomaly scale is the
    standard deviation before weighted centring, in concentration units.
    This is a new additive Gaussian prior, not a transformed truncated-normal
    scaling prior. It cannot be combined with scaling priors or activity masks.
    """
    if all(value is None for value in (mean_prior, anomaly_scale, correction_sensitivity, weights)):
        return None
    if any(
        value is None
        for value in (mean_prior, anomaly_scale, correction_sensitivity, weights, boundary_sensitivity)
    ):
        raise ValueError(
            "Centred boundaries require bc_mean_shift_prior, bc_anomaly_scale, "
            "boundary_correction_sensitivity, bc_centering_weights and boundary_sensitivity together."
        )
    if bc_prior is not None or bc_state_activity is not None:
        raise ValueError("Centred additive boundaries replace bc_prior and bc_state_activity; omit both.")
    assert anomaly_scale is not None
    if not np.isfinite(anomaly_scale) or anomaly_scale <= 0:
        raise ValueError("bc_anomaly_scale must be finite and strictly positive.")
    assert boundary_sensitivity is not None and correction_sensitivity is not None and weights is not None
    hbc = boundary_sensitivity.transpose(output_dim, "bc_region")
    gbc = correction_sensitivity.transpose(output_dim, "bc_region")
    weights = weights.transpose("bc_region")
    for array in (hbc, gbc, weights):
        for dim in array.dims:
            if dim not in array.indexes or not array.indexes[dim].is_unique:
                raise ValueError(f"Centred boundary inputs require unique indexed {dim!r} labels.")
    hbc, gbc, weights, _ = xr.align(hbc, gbc, weights, observations, join="exact", copy=False)
    for array in (gbc, weights):
        units = array.attrs.get("units")
        if units is None or not np.isclose(
            mole_fraction_unit_scale(str(units), context="centred boundary input"),
            1.0,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "Boundary correction sensitivity and centring weights require dimensionless units '1'."
            )
    hbc, gbc, weights = dask.compute(to_dense(hbc), to_dense(gbc), to_dense(weights))
    if not np.isfinite(hbc.values).all():
        raise ValueError("boundary_sensitivity must be finite.")
    if not np.isfinite(gbc.values).all() or (gbc.values < 0).any() or not (gbc.values > 0).any():
        raise ValueError("boundary_correction_sensitivity must be finite, non-negative and not all zero.")
    if not np.isfinite(weights.values).all() or not (weights.values > 0).all():
        raise ValueError("bc_centering_weights must be finite and strictly positive.")
    # Normalize by the maximum first to avoid overflow for large but valid weights.
    weights = weights / weights.max()
    weights = (weights / weights.sum()).assign_attrs(units="1")
    count = weights.size
    # Q spans the equal-sum-zero subspace. Applying P = I - 1 w.T gives
    # Cov(u) = scale**2 P P.T and preserves every pairwise contrast variance.
    q = helmert(count, full=False).T
    basis = q - np.ones((count, 1)) @ (weights.values @ q)[None, :]
    labelled_basis = xr.DataArray(
        basis,
        dims=("bc_region", "bc_contrast"),
        coords={"bc_region": weights.coords["bc_region"], "bc_contrast": np.arange(count - 1)},
    )
    design = np.column_stack((gbc.values.sum(axis=1), gbc.values @ basis))
    return CenteredBoundary(hbc, gbc, weights, labelled_basis, design)


def add_centered_boundary(
    prepared: CenteredBoundary,
    mean_prior: PriorArgs,
    anomaly_scale: float,
    *,
    output_dim: str = "nmeasure",
) -> tuple[TensorVariable, TensorVariable, tuple[TensorVariable, ...]]:
    """Build the total boundary signal and its non-redundant affine coefficients.

    Return the total observation contribution, coefficient vector (mean then
    K-1 normal contrasts), and sampler variables. For one boundary state the
    anomaly is identically zero. No multiplicative ``bc`` state is created.
    """
    hbc = add_model_data(prepared.reference_sensitivity, "hbc")
    gbc = add_model_data(prepared.sensitivity, "G_bc")
    add_model_data(prepared.weights, "bc_centering_weights")
    mean = parse_prior("bc_mean_shift", dict(mean_prior))
    if mean.ndim != 0:
        raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
    latents = (get_model_latent(mean, "bc_mean_shift"),)
    coefficients = pt.atleast_1d(mean)
    if prepared.basis.sizes["bc_contrast"]:
        basis = add_model_data(prepared.basis, "bc_anomaly_basis")
        contrasts = pm.Normal("bc_anomaly_latent", mu=0.0, sigma=anomaly_scale, dims="bc_contrast")
        anomaly_expression = pt.dot(basis, contrasts)
        coefficients = pt.concatenate((coefficients, contrasts))
        latents += (contrasts,)
    else:
        anomaly_expression = pt.zeros((1,))
    anomaly = pm.Deterministic("bc_anomaly", anomaly_expression, dims="bc_region")
    pm.Deterministic("bc_correction", mean + anomaly, dims="bc_region")
    reference = pm.Deterministic("mu_bc_reference", pt.sum(hbc, axis=1), dims=output_dim)
    mean_signal = pm.Deterministic("mu_bc_mean_shift", pt.sum(gbc, axis=1) * mean, dims=output_dim)
    anomaly_signal = pm.Deterministic("mu_bc_anomaly", pt.dot(gbc, anomaly), dims=output_dim)
    output = pm.Deterministic("mu_bc", reference + mean_signal + anomaly_signal, dims=output_dim)
    return output, coefficients, latents
