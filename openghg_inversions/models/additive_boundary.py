"""One common additive boundary correction and weighted-centred anomalies.

The additive mode replaces fitted multiplicative boundary scales. Its reference
contribution is A @ 1 + c, and its correction is M @ (mean + anomaly).
M is native dimensionless boundary transport; correction coefficients have
concentration units.
See https://mc-stan.org/docs/stan-users-guide/regression.html#parameterizing-centered-vectors
for the non-redundant normal coordinates underlying the centring constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from warnings import warn

import dask
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import helmert
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.boundary_sensitivity import _validate_boundary_weight_periods
from openghg.util import cf_ureg
from openghg_inversions.models.components import ConcentrationTerm, add_model_data, get_model_latent
from openghg_inversions.models.priors import PriorArgs, parse_prior
from openghg_inversions.models.coords import component_coords
from .boundary import require_centered_precision
from openghg_inversions.models.state_activity import StateActivity


@dataclass(frozen=True)
class AdditiveBoundary:
    """Eager labelled boundary field, centring basis, and affine design."""

    operands: xr.Dataset
    weights: xr.DataArray
    basis: xr.DataArray
    design: np.ndarray
    reference_shift: xr.DataArray


def prepare_additive_boundary(
    boundary_operands: xr.Dataset | None,
    weights: xr.DataArray | None,
    observations: xr.DataArray,
    mean_prior: PriorArgs | None,
    anomaly_scale: float | None,
    bc_prior: PriorArgs | None,
    bc_state_activity: StateActivity | None,
    *,
    output_dim: str = "nmeasure",
) -> AdditiveBoundary | None:
    """Align and jointly materialize additive boundary inputs at model construction.

    Positive weights define the mean over all supplied curtain/period states,
    including states unobserved by the transport. The anomaly scale is the
    standard deviation before weighted centring, in concentration units. Zero
    selects a common-level-only correction without sampled contrasts.
    The scalar mean keeps its supplied prior independently of Gaussian
    anomalies; this does not transform a multiplicative scaling prior.

    Args:
        boundary_operands: Native centered anomaly A and dimensionless transport
            M on (output_dim, bc_region), plus a scalar concentration reference.
            Labels must already describe the complete resolved region/period axis.
        weights: Positive dimensionless weights on that complete state axis,
            including unobserved states. Inputs are borrowed and may be lazy.
        observations: Labelled observation vector used for exact row alignment.
        mean_prior: Scalar concentration-unit prior; None with no anomaly scale
            disables the component without evaluating its passive inputs.
        anomaly_scale: Nonnegative concentration-unit scale before centering.
        bc_prior: Competing multiplicative prior, which must be omitted.
        bc_state_activity: Competing multiplicative activity, which must be omitted.
        output_dim: Observation dimension name.

    Returns:
        Jointly materialized native operands, normalized labelled weights, and
        the weighted basis and affine design; None when additive mode is disabled.

    Raises:
        ValueError: If options compete or are incomplete, labels or units differ,
            or numerical inputs are invalid. Double precision must be configured.
    """
    if mean_prior is None and anomaly_scale is None:
        return None
    if any(value is None for value in (mean_prior, anomaly_scale, weights, boundary_operands)):
        raise ValueError(
            "Additive boundaries require bc_mean_shift_prior, bc_anomaly_scale, "
            "bc_centering_weights and native centered boundary_operands together."
        )
    assert boundary_operands is not None
    require_centered_precision()
    boundary_sensitivity = boundary_operands.H_bc_anomaly
    correction_sensitivity = boundary_operands.H_bc_mass
    if bc_prior is not None or bc_state_activity is not None:
        raise ValueError("Centred additive boundaries replace bc_prior and bc_state_activity; omit both.")
    assert anomaly_scale is not None
    if not np.isfinite(anomaly_scale) or anomaly_scale < 0:
        raise ValueError("bc_anomaly_scale must be finite and non-negative.")
    assert weights is not None
    hbc = boundary_sensitivity.transpose(output_dim, "bc_region")
    gbc = correction_sensitivity.transpose(output_dim, "bc_region")
    weights = weights.transpose("bc_region")
    index = hbc.indexes.get("bc_region")
    if (
        isinstance(index, pd.MultiIndex)
        and len(index)
        and len(index.get_level_values("bc_period").unique()) == 1
        and index.get_level_values("bc_period")[0] == 0
        and not isinstance(weights.indexes.get("bc_region"), pd.MultiIndex)
        and weights.indexes.get("bc_region") is not None
        and weights.indexes["bc_region"].equals(index.get_level_values("bc_curtain"))
    ):
        # One whole-window state per curtain needs no inferred temporal policy.
        identity = hbc.bc_region.attrs.get("boundary_period_identity")
        if identity is not None and json.loads(identity)["frequency"] == "whole_window":
            weights = weights.assign_coords(hbc.bc_region.coords)
    for array in (hbc, gbc, weights):
        for dim in array.dims:
            if dim not in array.indexes or not array.indexes[dim].is_unique:
                raise ValueError(f"Centred boundary inputs require unique indexed {dim!r} labels.")
    _validate_boundary_weight_periods(boundary_operands, weights)
    hbc, gbc, weights, _ = xr.align(hbc, gbc, weights, observations, join="exact", copy=False)
    for array in (gbc, weights):
        units = array.attrs.get("units")
        try:
            scale = float(cf_ureg.Quantity(1, str(units)).to("1").magnitude)
        except Exception as exc:
            raise ValueError("Boundary correction sensitivity and centring weights require dimensionless units '1'.") from exc
        if units is None or not np.isclose(scale, 1.0, rtol=1e-12, atol=0.0):
            raise ValueError("Boundary correction sensitivity and centring weights require dimensionless units '1'.")
    hbc, gbc, weights, reference = dask.compute(
        to_dense(hbc),
        to_dense(gbc),
        to_dense(weights),
        to_dense(boundary_operands.bc_reference),
    )
    reference_shift = reference * gbc.sum("bc_region", skipna=False)
    if not np.isfinite(reference_shift.values).all():
        raise ValueError("Centered bc_reference and boundary reference shift must be finite.")
    if not np.isfinite(hbc.values).all():
        raise ValueError("boundary_sensitivity must be finite.")
    if not np.isfinite(gbc.values).all() or (gbc.values < 0).any() or not (gbc.values > 0).any():
        raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
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
    q = helmert(count, full=False).T if anomaly_scale > 0 else np.empty((count, 0))
    basis = q - np.ones((count, 1)) @ (weights.values @ q)[None, :]
    labelled_basis = xr.DataArray(
        basis,
        dims=("bc_region", "bc_contrast"),
        coords={"bc_region": weights.coords["bc_region"], "bc_contrast": np.arange(basis.shape[1])},
    )
    design = np.column_stack((gbc.values.sum(axis=1), gbc.values @ basis))
    return AdditiveBoundary(
        xr.Dataset({"H_bc_anomaly": hbc, "H_bc_mass": gbc, "bc_reference": reference}),
        weights, labelled_basis, design, reference_shift,
    )


@dataclass(frozen=True)
class AdditiveBoundaryResult:
    """Absolute boundary output and the affine terms used by cached likelihoods.

    ``term.likelihood_contribution`` excludes its observation shift when numerical
    centering is used. Coefficients contain the scalar mean and K−1 Gaussian
    contrasts when their scale is positive; latents are the underlying sampled
    PyMC variables.
    """

    term: ConcentrationTerm
    coefficients: TensorVariable
    latents: tuple[TensorVariable, ...]


def add_additive_boundary(
    prepared: AdditiveBoundary,
    mean_prior: PriorArgs,
    anomaly_scale: float,
    *,
    output_dim: str = "nmeasure",
    prefix: str = "",
    owner: str | None = None,
) -> AdditiveBoundaryResult:
    """Build additive boundary corrections on registered observation/state axes.

    The common mean and pre-centering anomaly scale use observation concentration
    units. With centered operands the likelihood uses A1 + M(mean + anomaly),
    where M is native dimensionless boundary transport, while absolute outputs
    restore cM1 exactly once. The corrections are never
    multiplied by the numerical reference c or the signed concentration anomaly
    A. A zero anomaly scale retains only the common mean as a sampled state.
    No multiplicative state is created.

    Args:
        prepared: Eager operands and weighted basis from prepare_additive_boundary.
        mean_prior: Arbitrary scalar prior for the whole-window concentration shift.
        anomaly_scale: Concentration-unit Gaussian scale used during preparation.
        output_dim: Observation dimension registered by the model.
        prefix: Prefix for graph variable names.
        owner: Namespace for labelled boundary-state coordinates.

    Returns:
        Absolute and centered concentration terms, their affine coefficients,
        and sampled latent variables for ordinary or cached likelihoods.

    Raises:
        ValueError: If the mean prior is not scalar or coordinates conflict.
            Call inside a registered_model context.
    """

    def register(array: xr.DataArray, name: str) -> TensorVariable:
        return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

    hbc = register(prepared.operands.H_bc_anomaly, "H_bc_anomaly")
    gbc = register(prepared.operands.H_bc_mass, "H_bc_mass")
    register(prepared.operands.bc_reference, "bc_reference")
    register(prepared.weights, "bc_centering_weights")
    register(xr.DataArray(anomaly_scale), "bc_anomaly_scale")
    state_dim = str(component_coords(prepared.weights, owner).dims[0])
    mean = parse_prior(f"{prefix}bc_mean_shift", dict(mean_prior))
    if mean.ndim != 0:
        raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
    latents: tuple[TensorVariable, ...] = (get_model_latent(mean, f"{prefix}bc_mean_shift"),)
    coefficients = pt.atleast_1d(mean)
    if prepared.basis.sizes["bc_contrast"]:
        basis = register(prepared.basis, "bc_anomaly_basis")
        contrast_dim = str(component_coords(prepared.basis, owner).dims[1])
        contrasts = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=anomaly_scale, dims=contrast_dim)
        anomaly_expression = pt.dot(basis, contrasts)
        coefficients = pt.concatenate((coefficients, contrasts))
        latents += (contrasts,)
    else:
        anomaly_expression = pt.zeros((prepared.weights.size,))
    anomaly = anomaly_expression
    if prepared.basis.sizes["bc_contrast"]:
        anomaly = pm.Deterministic(f"{prefix}bc_anomaly", anomaly, dims=state_dim)
    pm.Deterministic(f"{prefix}bc_correction", mean + anomaly, dims=state_dim)
    reference = register(
        (prepared.operands.H_bc_anomaly.sum("bc_region") + prepared.reference_shift).assign_attrs(
            prepared.operands.H_bc_anomaly.attrs
        ),
        "mu_bc_reference",
    )
    mean_signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.sum(gbc, axis=1) * mean, dims=output_dim)
    anomaly_signal = pt.dot(gbc, anomaly)
    if prepared.basis.sizes["bc_contrast"]:
        anomaly_signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", anomaly_signal, dims=output_dim)
    likelihood = pt.sum(hbc, axis=1) + mean_signal + anomaly_signal
    output = pm.Deterministic(f"{prefix}mu_bc", reference + mean_signal + anomaly_signal, dims=output_dim)
    return AdditiveBoundaryResult(
        ConcentrationTerm(output, likelihood, prepared.reference_shift), coefficients, latents
    )


def warn_boundary_offset_confounding(
    mean_response: np.ndarray,
    *,
    drop_first: bool,
    channel: str | None = None,
) -> None:
    """Warn about overlapping constant boundary and unrestricted offset terms.

    Call only when an observation offset is enabled, after jointly materializing
    the model inputs. ``mean_response`` is the eager channel-specific vector
    ``M @ ones``; select native channel rows before passing a padded design.
    Anchoring the first site's offset excludes this unrestricted constant mode.
    Unit response uses absolute tolerance 1e-12 with no relative tolerance.
    This check does not assess overlap with boundary anomalies or flux terms.

    Raises:
        TypeError: If the response is not an eager NumPy array; lazy inputs
            must be materialized by the caller's model boundary.
    """
    if not isinstance(mean_response, np.ndarray):
        raise TypeError("mean_response must be an eager NumPy array.")
    if drop_first:
        return
    if np.allclose(mean_response, 1.0, rtol=0.0, atol=1e-12):
        prefix = f"{channel}: " if channel is not None else ""
        warn(
            prefix + "Additive boundary mean and unrestricted observation offsets share a constant "
            "observation response; the likelihood identifies only their combined constant. "
            "Separating boundary-product bias from observation bias requires prior or external "
            "calibration information.",
            UserWarning,
            stacklevel=2,
        )
