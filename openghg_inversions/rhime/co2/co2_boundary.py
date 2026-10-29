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
from openghg_inversions.models.coords import component_coords
from .centered_boundary import require_centered_precision
from openghg_inversions.models.state_activity import StateActivity


@dataclass(frozen=True)
class CenteredBoundary:
    """Eager labelled boundary field, centring basis, and affine design."""

    reference_sensitivity: xr.DataArray
    sensitivity: xr.DataArray
    weights: xr.DataArray
    basis: xr.DataArray
    design: np.ndarray
    reference_shift: xr.DataArray | None = None


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
    boundary_operands: xr.Dataset | None = None,
) -> CenteredBoundary | None:
    """Align and jointly materialize additive boundary inputs at model construction.

    Positive weights define the mean over all supplied curtain/period states,
    including states unobserved by the transport. The anomaly scale is the
    standard deviation before weighted centring, in concentration units.
    This is a new additive Gaussian prior, not a transformed truncated-normal
    scaling prior. It cannot be combined with scaling priors or activity masks.
    """
    if mean_prior is None and anomaly_scale is None:
        return None
    reference_shift = None
    if boundary_operands is not None:
        require_centered_precision()
        boundary_sensitivity = boundary_operands.H_bc_anomaly
        if correction_sensitivity is None:
            correction_sensitivity = boundary_operands.H_bc_mass
        reference_shift = boundary_operands.bc_reference * boundary_operands.H_bc_mass.sum("bc_region", skipna=False)
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
    hbc, gbc, weights, reference_shift = dask.compute(
        to_dense(hbc), to_dense(gbc), to_dense(weights),
        None if reference_shift is None else to_dense(reference_shift),
    )
    if reference_shift is not None and not np.isfinite(reference_shift.values).all():
        raise ValueError("Centered bc_reference and boundary reference shift must be finite.")
    if not np.isfinite(hbc.values).all():
        raise ValueError("boundary_sensitivity must be finite.")
    if not np.isfinite(gbc.values).all() or (gbc.values < 0).any() or not (gbc.values > 0).any():
        raise ValueError("boundary_correction_sensitivity must be finite, non-negative and not all zero.")
    if not np.isfinite(weights.values).all() or not (weights.values > 0).all():
        raise ValueError("bc_centering_weights must be finite and strictly positive.")
    count = weights.size
    if count == 0:
        raise ValueError("Additive boundaries require at least one boundary state.")
    # Normalize by the maximum first to avoid overflow for large but valid weights.
    weights = weights / weights.max()
    weights = (weights / weights.sum()).assign_attrs(units="1")
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
    return CenteredBoundary(hbc, gbc, weights, labelled_basis, design, reference_shift)


@dataclass(frozen=True)
class AdditiveBoundaryResult:
    """Absolute boundary output and the affine terms used by cached likelihoods.

    ``likelihood_contribution`` excludes ``reference_shift`` when numerical
    centering is used. Coefficients contain the scalar mean and K−1 Gaussian
    contrasts; latents are the underlying sampled PyMC variables.
    """

    output: TensorVariable
    likelihood_contribution: TensorVariable
    coefficients: TensorVariable
    latents: tuple[TensorVariable, ...]
    reference_shift: xr.DataArray | None


def add_centered_boundary(
    prepared: CenteredBoundary,
    mean_prior: PriorArgs,
    anomaly_scale: float,
    *,
    output_dim: str = "nmeasure",
    prefix: str = "",
    owner: str | None = None,
) -> AdditiveBoundaryResult:
    """Build additive boundary corrections on registered observation/state axes.

    The common mean and pre-centering anomaly scale use observation concentration
    units. With centered operands the likelihood uses A1 + G(mean + anomaly),
    while absolute outputs restore cM1 exactly once. The corrections are never
    multiplied by the numerical reference c. No multiplicative state is created.
    """
    def register(array: xr.DataArray, name: str) -> TensorVariable:
        return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

    hbc = register(prepared.reference_sensitivity, "hbc")
    gbc = register(prepared.sensitivity, "G_bc")
    register(prepared.weights, "bc_centering_weights")
    register(xr.DataArray(anomaly_scale), "bc_anomaly_scale")
    state_dim = str(component_coords(prepared.weights, owner).dims[0])
    mean = parse_prior(f"{prefix}bc_mean_shift", dict(mean_prior))
    if mean.ndim != 0:
        raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
    latents = (get_model_latent(mean, f"{prefix}bc_mean_shift"),)
    coefficients = pt.atleast_1d(mean)
    if prepared.basis.sizes["bc_contrast"]:
        basis = register(prepared.basis, "bc_anomaly_basis")
        contrast_dim = str(component_coords(prepared.basis, owner).dims[1])
        contrasts = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=anomaly_scale, dims=contrast_dim)
        anomaly_expression = pt.dot(basis, contrasts)
        coefficients = pt.concatenate((coefficients, contrasts))
        latents += (contrasts,)
    else:
        anomaly_expression = pt.zeros((1,))
    anomaly = pm.Deterministic(f"{prefix}bc_anomaly", anomaly_expression, dims=state_dim)
    pm.Deterministic(f"{prefix}bc_correction", mean + anomaly, dims=state_dim)
    reference_expression = pt.sum(hbc, axis=1)
    shift_data = None
    if prepared.reference_shift is not None:
        shift_data = register(prepared.reference_shift, "bc_reference_shift")
        reference_expression = reference_expression + shift_data
    reference = pm.Deterministic(f"{prefix}mu_bc_reference", reference_expression, dims=output_dim)
    mean_signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.sum(gbc, axis=1) * mean, dims=output_dim)
    anomaly_signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(gbc, anomaly), dims=output_dim)
    likelihood = pt.sum(hbc, axis=1) + mean_signal + anomaly_signal
    output = pm.Deterministic(f"{prefix}mu_bc", reference + mean_signal + anomaly_signal, dims=output_dim)
    if shift_data is not None:
        likelihood = pm.Deterministic(f"{prefix}mu_bc_centered", likelihood, dims=output_dim)
    return AdditiveBoundaryResult(output, likelihood, coefficients, latents, prepared.reference_shift)
