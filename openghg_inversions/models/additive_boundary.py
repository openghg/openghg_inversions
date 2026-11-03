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
from warnings import warn

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import helmert
import xarray as xr

from openghg.util import cf_ureg
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, ConcentrationTerm, add_model_data, get_model_latent
from openghg_inversions.models.priors import PriorArgs, parse_prior
from openghg_inversions.models.coords import component_coords


@dataclass(frozen=True)
class AdditiveBoundary:
    """Borrowed labelled operands and resolved concentration-unit prior.

    The dataset contains native centered operands and full-state weights.
    Materialize it with the model inputs, then replace ``data`` explicitly.
    """

    data: xr.Dataset
    mean_prior: PriorArgs
    anomaly_scale: float


def prepare_additive_boundary(
    boundary_operands: xr.Dataset,
    weights: xr.DataArray,
    observations: xr.DataArray,
    mean_prior: PriorArgs,
    anomaly_scale: float,
    *,
    output_dim: str = "nmeasure",
) -> AdditiveBoundary:
    """Bind actual additive inputs without computing numerical payloads.

    Inputs already have bound periods and a resolved scalar concentration-unit
    mean prior and nonnegative anomaly scale. Mode selection/exclusion belongs
    solely to ``prepare_boundary``. Positive dimensionless weights span every
    supplied state, including states unobserved by transport. Curtain-only
    weights are allowed only for a genuine whole-window (no frequency) fit;
    temporal fits require explicit period-start labels even with one period.
    """
    from .boundary import _convert_units, require_centered_precision

    require_centered_precision()
    boundary_sensitivity = boundary_operands.H_bc_anomaly
    correction_sensitivity = boundary_operands.H_bc_mass
    reference = boundary_operands.bc_reference
    if reference.ndim != 0:
        raise ValueError("Centered bc_reference must be scalar.")
    units = observations.attrs.get("units") or reference.attrs.get("units")
    if units is None:
        raise ValueError("Centered boundary concentrations require units.")
    boundary_sensitivity = _convert_units(boundary_sensitivity, units)
    reference = _convert_units(reference, units)
    if not np.isfinite(anomaly_scale) or anomaly_scale < 0:
        raise ValueError("bc_anomaly_scale must be finite and non-negative.")
    hbc = boundary_sensitivity.transpose(output_dim, "bc_region")
    gbc = correction_sensitivity.transpose(output_dim, "bc_region")
    if set(weights.dims) == {"bc_curtain", "bc_period"}:
        weights = weights.stack(bc_region=("bc_curtain", "bc_period"))
    elif weights.dims == ("bc_curtain",):
        index = hbc.indexes.get("bc_region")
        if (hbc.attrs.get("bc_frequency") == "none" and isinstance(index, pd.MultiIndex)
                and len(index.get_level_values("bc_period").unique()) == 1):
            weights = weights.expand_dims(bc_period=index.get_level_values("bc_period").unique()).stack(
                bc_region=("bc_curtain", "bc_period")
            )
        else:
            raise ValueError("Temporal boundary weights require explicit period-start labels.")
    weights = weights.transpose("bc_region")
    for array in (hbc, gbc, weights):
        for dim in array.dims:
            if dim not in array.indexes or not array.indexes[dim].is_unique:
                raise ValueError(f"Centred boundary inputs require unique indexed {dim!r} labels.")
    hbc, gbc, weights, _ = xr.align(hbc, gbc, weights, observations, join="exact", copy=False)
    converted = []
    for array in (gbc, weights):
        try:
            scale = float(cf_ureg.Quantity(1, str(array.attrs.get("units"))).to("1").magnitude)
        except Exception as exc:
            raise ValueError("Boundary correction sensitivity and centring weights require dimensionless units '1'.") from exc
        normalized = array if scale == 1.0 else array * scale
        converted.append(normalized.assign_attrs(dict(array.attrs, units="1")))
    gbc, weights = converted
    data = xr.Dataset({
        "H_bc_anomaly": hbc, "H_bc_mass": gbc, "bc_reference": reference,
        "bc_centering_weights": weights,
    })
    return AdditiveBoundary(data, dict(mean_prior), float(anomaly_scale))


def add_additive_boundary(
    bound: AdditiveBoundary,
    *,
    output_dim: str = "nmeasure",
    prefix: str = "",
    owner: str | None = None,
) -> AffineConcentrationTerm:
    """Build additive boundary corrections on registered observation/state axes.

    The common mean and pre-centering anomaly scale use observation concentration
    units. With centered operands the likelihood uses A1 + M(mean + anomaly),
    where M is native dimensionless boundary transport, while absolute outputs
    restore cM1 exactly once. The corrections are never
    multiplied by the numerical reference c or the signed concentration anomaly
    A. A zero anomaly scale retains only the common mean as a sampled state.
    No multiplicative state is created.

    Args:
        bound: Jointly materialized operands and resolved scientific choices.
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

    from .boundary import require_centered_precision
    require_centered_precision()
    if any(hasattr(value.data, "__dask_graph__") for value in bound.data.data_vars.values()):
        raise ValueError("Materialize boundary.data jointly with model inputs before add_boundary.")
    operands = bound.data
    weights = operands.bc_centering_weights
    hbc = operands.H_bc_anomaly.transpose(output_dim, "bc_region")
    gbc = operands.H_bc_mass.transpose(output_dim, "bc_region")
    reference = operands.bc_reference
    if not np.isfinite(reference.values).all() or not np.isfinite(hbc.values).all():
        raise ValueError("Centered bc_reference and boundary anomaly must be finite.")
    if not np.isfinite(gbc.values).all() or (gbc.values < 0).any() or not (gbc.values > 0).any():
        raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
    if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
        raise ValueError("bc_centering_weights must be finite and strictly positive.")
    weights = weights / weights.max()
    weights = (weights / weights.sum()).assign_attrs(units="1")
    count = weights.size
    anomaly_scale = bound.anomaly_scale
    q = helmert(count, full=False).T if anomaly_scale > 0 else np.empty((count, 0))
    basis = q - np.ones((count, 1)) @ (weights.values @ q)[None, :]
    labelled_basis = xr.DataArray(
        basis, dims=("bc_region", "bc_contrast"),
        coords={"bc_region": weights.bc_region, "bc_contrast": np.arange(basis.shape[1])},
        attrs={"units": "1"},
    )
    reference_shift = (reference * gbc.sum("bc_region", skipna=False)).rename(
        f"{prefix}bc_reference_shift"
    ).assign_attrs(reference.attrs)
    if not np.isfinite(reference_shift.values).all():
        raise ValueError("Boundary reference shift must be finite.")
    fixed = hbc.sum("bc_region", skipna=False).assign_attrs(hbc.attrs)
    mean_design = gbc.sum("bc_region", skipna=False).expand_dims(bc_mean=[0], axis=1).assign_attrs(units="1")

    def register(array: xr.DataArray, name: str) -> TensorVariable:
        return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

    hbc = register(hbc, "H_bc_anomaly")
    gbc = register(gbc, "H_bc_mass")
    register(operands.bc_reference, "bc_reference")
    register(weights, "bc_centering_weights")
    register(xr.DataArray(anomaly_scale), "bc_anomaly_scale")
    state_dim = str(component_coords(weights, owner).dims[0])
    mean = parse_prior(f"{prefix}bc_mean_shift", dict(bound.mean_prior))
    if mean.ndim != 0:
        raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
    latents: tuple[TensorVariable, ...] = (get_model_latent(mean, f"{prefix}bc_mean_shift"),)
    if labelled_basis.sizes["bc_contrast"]:
        basis = register(labelled_basis, "bc_anomaly_basis")
        contrast_dim = str(component_coords(labelled_basis, owner).dims[1])
        contrasts = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=anomaly_scale, dims=contrast_dim)
        anomaly_expression = pt.dot(basis, contrasts)
        latents += (contrasts,)
    else:
        anomaly_expression = pt.zeros((weights.size,))
    anomaly = anomaly_expression
    if labelled_basis.sizes["bc_contrast"]:
        anomaly = pm.Deterministic(f"{prefix}bc_anomaly", anomaly, dims=state_dim)
    pm.Deterministic(f"{prefix}bc_correction", mean + anomaly, dims=state_dim)
    reference = register(
        (operands.H_bc_anomaly.sum("bc_region") + reference_shift).assign_attrs(
            operands.H_bc_anomaly.attrs
        ),
        "mu_bc_reference",
    )
    mean_signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.sum(gbc, axis=1) * mean, dims=output_dim)
    anomaly_signal = pt.dot(gbc, anomaly)
    if labelled_basis.sizes["bc_contrast"]:
        anomaly_signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", anomaly_signal, dims=output_dim)
    likelihood = pt.sum(hbc, axis=1) + mean_signal + anomaly_signal
    output = pm.Deterministic(f"{prefix}mu_bc", reference + mean_signal + anomaly_signal, dims=output_dim)
    blocks = [AffineBlock(mean_design, pt.atleast_1d(mean))]
    if labelled_basis.sizes["bc_contrast"]:
        blocks.append(AffineBlock(
            xr.dot(operands.H_bc_mass, labelled_basis, dim="bc_region").transpose(output_dim, "bc_contrast").assign_attrs(units="1"), contrasts,
        ))
    return AffineConcentrationTerm(
        ConcentrationTerm(output, likelihood, reference_shift), fixed, tuple(blocks), latents,
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
