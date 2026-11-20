"""Calibrate one retained-state gross-exchange uncertainty direction.

This is an explicit eager boundary for small, already-retained priors. The
transport reduction, affine reconstruction and unresolved covariance stay fixed.
Arrays in the returned scientific value are borrowed: do not mutate them.
"""

from __future__ import annotations

from dataclasses import dataclass

from dask import compute
import numpy as np
from openghg.util import cf_ureg
import xarray as xr
from scipy.linalg import qr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.correlated_state import CorrelatedPrior, _covariance_array, _matrix_tolerance


BIOSPHERE_PRIOR_SCHEMA = "openghg_inversions.biosphere_prior/v1"
"""Provisional schema of the saved calibration payload."""

BIOSPHERE_PRIOR_SCOPE = "retained_state_prior_with_fixed_reduction_reconstruction_and_aggregation"
"""What a calibration changes, recorded in ``rhime_model_metadata``."""

_POLICIES = ("residual_sd_multiplier", "target_residual_sd", "target_gross_sd")
_FUNCTIONALS = ("GPP", "TER", "net", "gross", "residual_gross")


@dataclass(frozen=True, eq=False)
class BiospherePrior:
    """Calibrated retained-prior covariance and the inputs that produced it.

    ``data`` is eager and borrowed: do not mutate it. It holds the inputs
    ``baseline_mean``, ``baseline_covariance``, ``biosphere``, ``weights`` and
    ``reference``, and the results ``covariance`` and ``residual_weights``.
    Its scalar attributes are the schema, ``covariance_dim``, the resolved
    ``policy``, the ``requested`` value, the reporting ``units`` and the
    ``residual_sd_multiplier``.

    A calibration fixes arithmetic moments only. The model samples them with
    the marginal families of the retained prior it calibrates; see
    :func:`~openghg_inversions.models.biosphere.resolve_retained_flux_prior`.
    """

    data: xr.Dataset

    def to_dataset(self) -> xr.Dataset:
        """Return the labelled serialization payload.

        Returns:
            The stored Dataset itself, not a copy, in the layout
            :meth:`from_dataset` reads.
        """
        return self.data

    @classmethod
    def from_dataset(cls, data: xr.Dataset) -> BiospherePrior:
        """Check a saved calibration without repeating it.

        Loading is an eager boundary for this small payload. Control selection
        and target resolution are not repeated.

        Args:
            data: Saved inputs, results and scalar attributes in the layout
                :func:`calibrate_biosphere_prior` returns.

        Returns:
            The calibration, with the saved covariance as its sampling target.

        Raises:
            ValueError: If the schema or policy is unsupported, an array has
                the wrong dimensions or non-finite values, the covariance
                column identity differs from the state labels, the GPP/TER labels,
                weight signs, selector or units are invalid, the multiplier is
                inconsistent with the policy, or the saved covariance is not the
                residual-direction change of the baseline that preserves the
                protected rows and the net-functional covariance.
        """
        if data.attrs.get("biosphere_prior_schema") != BIOSPHERE_PRIOR_SCHEMA:
            raise ValueError("Unsupported biosphere-prior schema; prepare again.")
        data = compute(data.map(to_dense, keep_attrs=True))[0]
        dim, cov_dim = str(data.baseline_mean.dims[0]), data.attrs["covariance_dim"]
        dimensions = {
            "baseline_mean": (dim,), "baseline_covariance": (dim, cov_dim), "covariance": (dim, cov_dim),
            "biosphere": (dim,), "residual_weights": (dim,), "weights": ("process", dim),
            "reference": ("process",),
        }
        for name, expected in dimensions.items():
            if data[name].dims != expected:
                raise ValueError(f"Stored {name} requires dimensions {expected}.")
            if not np.isfinite(data[name].values).all():
                raise ValueError(f"Stored {name} must be finite.")
        if f"{cov_dim}_label" not in data.coords:
            raise ValueError(f"Stored covariances require the {cov_dim}_label column identity.")
        for name in ("baseline_covariance", "covariance"):
            _covariance_array(data[name], mean=data.baseline_mean, state_dim=dim, covariance_dim=cov_dim)
        if set(data.get_index("process")) != {"GPP", "TER"} or data.sizes["process"] != 2:
            raise ValueError("Stored reporting weights require exactly GPP and TER labels.")
        selector = data.biosphere.values
        if selector.dtype != bool or not selector.any():
            raise ValueError("Stored biosphere selector must select labelled states.")
        units = data.attrs.get("units")
        if not units or data.weights.attrs.get("units") != units or not data.reference.attrs.get("units"):
            raise ValueError("Stored calibration requires declared flux units.")
        cf_ureg.Quantity(1, data.reference.attrs["units"]).to(units)
        weights = data.weights.sel(process=["GPP", "TER"]).values
        if (weights[0] > 0).any() or (weights[1] < 0).any():
            raise ValueError("Stored GPP/TER weights have invalid signs.")
        policy, requested = data.attrs.get("policy"), float(data.attrs["requested"])
        factor = float(data.attrs["residual_sd_multiplier"])
        if (policy not in _POLICIES or not np.isfinite(factor) or factor <= 0
                or not np.isfinite(requested) or requested < 0
                or (policy == "residual_sd_multiplier" and factor != requested)):
            raise ValueError("Stored multiplier is inconsistent with the resolved calibration policy.")
        b, changed = data.baseline_covariance.values, data.covariance.values
        if factor == 1:
            if not np.array_equal(changed, b):
                raise ValueError("Stored identity calibration must retain baseline covariance.")
            return cls(data)
        residual = data.residual_weights.values
        direction = b @ residual
        residual_variance = float(residual @ direction)
        if residual_variance <= 0:
            raise ValueError("Stored residual weights have no residual variance to calibrate.")
        expected_change = (np.square(factor) - 1) * np.outer(direction, direction) / residual_variance
        tolerance = _matrix_tolerance(b)
        if not np.allclose(changed - b, expected_change, rtol=1e-10, atol=tolerance):
            raise ValueError("Stored covariance must be the residual-direction change of the baseline.")
        if not np.allclose((changed - b)[~selector], 0, rtol=0, atol=tolerance):
            raise ValueError("Stored calibration must preserve protected covariance rows.")
        net = weights.sum(axis=0)
        net_scale = np.max(np.abs(net))
        if net_scale > 0 and not np.allclose(
            (net / net_scale) @ (changed - b), 0, rtol=0, atol=tolerance * np.abs(net / net_scale).sum(),
        ):
            raise ValueError("Stored calibration must preserve net-functional covariance.")
        return cls(data)

    def functionals(self) -> xr.Dataset:
        """Derive reporting-functional moments before and after calibration.

        The functionals are GPP, TER, net, gross and the residual gross
        exchange the calibration rescales. For any other linear functional
        ``l``, the variance is ``l.T @ covariance @ l``.

        Returns:
            ``functional_mean`` on ``functional``, and ``functional_sd`` and
            ``functional_correlation`` on ``prior`` (``baseline`` or
            ``calibrated``), all in the stored reporting units.
        """
        data = self.data
        weights = data.weights.sel(process=["GPP", "TER"]).values
        net, gross = weights[0] + weights[1], -weights[0] + weights[1]
        rows = np.vstack([weights, net, gross, data.residual_weights.values])
        covariance = np.stack([
            rows @ matrix @ rows.T for matrix in (data.baseline_covariance.values, data.covariance.values)
        ])
        sd = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        scale = sd[:, :, None] * sd[:, None, :]
        correlation = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)
        totals = data.reference.sel(process=["GPP", "TER"]).values
        means = [totals[0], totals[1], totals.sum(), -totals[0] + totals[1], 0.0]
        units = data.attrs["units"]
        return xr.Dataset(
            {
                "functional_mean": ("functional", means, {"units": units}),
                "functional_sd": (("prior", "functional"), sd, {"units": units}),
                "functional_correlation": (("prior", "functional", "functional_cov"), correlation),
            },
            coords={"prior": ["baseline", "calibrated"], "functional": list(_FUNCTIONALS),
                    "functional_cov": list(_FUNCTIONALS)},
        )


def calibrate_biosphere_prior(
    baseline: CorrelatedPrior,
    *,
    biosphere: xr.DataArray,
    weights: xr.DataArray,
    reference: xr.DataArray,
    residual_sd_multiplier: float | None = None,
    target_residual_sd: xr.DataArray | None = None,
    target_gross_sd: xr.DataArray | None = None,
) -> BiospherePrior:
    """Change one residual-gross SD while preserving all control covariances.

    Args:
        baseline: Existing labelled retained-state prior. The calibrated
            moments must be valid for its marginal families.
        biosphere: Boolean selector on the baseline state dimension. All other
            states are protected; no source names or positional pairing inferred.
        weights: Signed affine reporting weights on ``(process, state_dim)``
            with exactly ``GPP`` and ``TER`` process labels. Both have common
            declared flux units for dimensionless coefficients. GPP uptake is
            negative, TER respiration positive. Labels must match baseline order.
        reference: GPP and TER affine totals at the baseline mean, on ``process``.
            Declared units are converted to the weights' units with Pint.
        residual_sd_multiplier: Dimensionless nonnegative residual-SD factor.
            Exactly one calibration choice is required; one recovers baseline.
        target_residual_sd: Scalar SD with declared units, alternative to factor.
        target_gross_sd: Scalar *marginal* gross SD. Targets below variance
            explained by the protected controls are infeasible.

    Returns:
        Eager labelled inputs, calibrated covariance and residual gross weights.
        Redundant standardized controls are explicitly reduced by pivoted QR;
        zero residual variance rejects an intervention.

    Raises:
        ValueError: For ambiguous policies, invalid selectors/units/moments,
            infeasible targets, or absent residual uncertainty. Singular changed
            arithmetic covariance, or moments invalid for the baseline's
            marginal families, are rejected without adding jitter.
    """
    if (
        sum(
            value is not None
            for value in (
                residual_sd_multiplier,
                target_residual_sd,
                target_gross_sd,
            )
        )
        != 1
    ):
        raise ValueError(
            "Specify exactly one residual multiplier, residual SD target, or marginal gross SD target."
        )
    dim, cov_dim = baseline.state_dim, baseline.covariance_dim
    mean, covariance = baseline.mean, baseline.arithmetic_covariance
    weights = weights.transpose("process", dim)
    reference = reference.transpose("process")
    if set(weights.get_index("process")) != {"GPP", "TER"} or weights.sizes["process"] != 2:
        raise ValueError("Reporting weights require exactly GPP and TER labels.")
    weights = weights.sel(process=["GPP", "TER"])
    reference = reference.sel(process=["GPP", "TER"])
    mean, weights, biosphere, reference = xr.align(
        mean, weights, biosphere, reference, join="exact", copy=False
    )
    biosphere = biosphere.transpose(dim)
    if biosphere.dtype != bool:
        raise ValueError("biosphere must be an explicit boolean state selector.")
    units = weights.attrs.get("units")
    if not units or not reference.attrs.get("units"):
        raise ValueError("Reporting weights and reference require declared flux units.")
    reference = reference.pint.quantify().pint.to(units).pint.dequantify()
    target = target_residual_sd if target_residual_sd is not None else target_gross_sd
    if target is not None:
        if target.ndim != 0 or not target.attrs.get("units"):
            raise ValueError("SD target must be scalar with declared flux units.")
        target = target.pint.quantify().pint.to(units).pint.dequantify()
    weights, reference, biosphere, target = compute(
        to_dense(weights), to_dense(reference), to_dense(biosphere),
        to_dense(target) if target is not None else None,
    )
    w, totals, bio = weights.to_numpy(), reference.to_numpy(), biosphere.to_numpy()
    if not np.isfinite(w).all() or not np.isfinite(totals).all():
        raise ValueError("Reporting functionals must be finite.")
    if not bio.any():
        raise ValueError("Select at least one biosphere state.")
    if (w[0] > 0).any() or (w[1] < 0).any():
        raise ValueError("GPP weights must be negative uptake and TER positive respiration.")
    b = covariance.to_numpy()
    try:
        np.linalg.cholesky(b)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Baseline arithmetic covariance must be positive definite.") from exc
    net, gross = w[0] + w[1], -w[0] + w[1]
    controls = np.vstack([net, np.eye(mean.size)[~bio]])
    control_sd = np.sqrt(np.einsum("ij,jk,ik->i", controls, b, controls))
    nonzero = control_sd > 0
    standardized = controls[nonzero] / control_sd[nonzero, None]
    # QR identifies redundant rows before the well-scaled covariance solve.
    _, triangular, pivot = qr(standardized.T, pivoting=True, mode="economic")
    diagonal = np.abs(np.diag(triangular))
    tolerance = np.finfo(float).eps * max(standardized.shape) * (diagonal.max() if diagonal.size else 0)
    rank = int(np.sum(diagonal > tolerance))
    independent_controls = standardized[pivot[:rank]]
    # Remove gross exchange predicted by the controls in the prior covariance metric.
    residual = gross - independent_controls.T @ np.linalg.solve(
        independent_controls @ b @ independent_controls.T, independent_controls @ b @ gross
    )
    covariance_direction = b @ residual
    # controls @ covariance_direction = 0, so the update preserves all control covariances.
    residual_variance = float(residual @ covariance_direction)
    gross_variance = float(gross @ b @ gross)
    explained = max(0.0, gross_variance - residual_variance)
    if residual_variance <= np.finfo(float).eps * max(gross_variance, np.finfo(float).tiny) * mean.size:
        if residual_sd_multiplier != 1:
            raise ValueError(
                "Gross exchange has zero residual variance after the controls; no mode to calibrate."
            )
        residual = np.zeros_like(gross)
        covariance_direction = np.zeros_like(gross)
        residual_variance = 0.0
        explained = gross_variance

    policy = "residual_sd_multiplier"
    requested: float
    if target is not None:
        target_value = float(target.item())
        if not np.isfinite(target_value) or target_value < 0:
            raise ValueError("SD target must be finite and nonnegative.")
        requested = target_value
        if target_gross_sd is not None:
            policy = "target_gross_sd"
            with np.errstate(over="ignore"):
                target_variance = float(np.square(target_value))
            if not np.isfinite(target_variance):
                raise ValueError("Requested marginal gross variance is not finite.")
            if target_variance < explained:
                raise ValueError(
                    f"Marginal gross variance is below explained variance {explained:.12g} in ({units})^2."
                )
            factor = np.sqrt((target_variance - explained) / residual_variance)
        else:
            policy = "target_residual_sd"
            factor = target_value / np.sqrt(residual_variance)
    else:
        assert residual_sd_multiplier is not None
        factor = residual_sd_multiplier
        requested = factor
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Residual SD multiplier must be finite and nonnegative.")
    if factor == 0:
        raise ValueError(
            "Zero residual SD gives a singular arithmetic covariance; zero-SD targets are unsupported."
        )
    # The exact identity arm avoids even rounding the original covariance.
    with np.errstate(over="ignore", invalid="ignore"):
        changed = (
            b.copy()
            if factor == 1
            else b + (np.square(factor) - 1) * np.outer(
                covariance_direction, covariance_direction
            ) / residual_variance
        )
    if not np.isfinite(changed).all():
        raise ValueError("Requested uncertainty produces nonfinite arithmetic covariance.")
    changed = (changed + changed.T) / 2 if factor != 1 else changed
    try:
        np.linalg.cholesky(changed)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Changed arithmetic covariance must be positive definite; zero-SD singular targets are unsupported."
        ) from exc
    changed_array = covariance.copy(data=changed).rename("covariance")
    if factor != 1:
        try:
            CorrelatedPrior(mean, changed_array, marginal=baseline.marginal, covariance_dim=cov_dim)
        except ValueError as exc:
            raise ValueError(f"Requested biosphere calibration is invalid for the prior's marginal families: {exc}") from exc
    residual_weights = mean.copy(data=residual)
    residual_weights.attrs = {"units": units}
    data = xr.Dataset(
        {
            "baseline_mean": mean,
            "baseline_covariance": covariance,
            "covariance": changed_array,
            "biosphere": biosphere,
            "weights": weights,
            "reference": reference,
            "residual_weights": residual_weights,
        },
        attrs={
            "biosphere_prior_schema": BIOSPHERE_PRIOR_SCHEMA,
            "covariance_dim": cov_dim,
            "policy": policy,
            "requested": float(requested),
            "units": units,
            "residual_sd_multiplier": float(factor),
        },
    )
    return BiospherePrior(data)
