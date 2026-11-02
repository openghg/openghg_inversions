"""Calibrate one retained-state gross-exchange uncertainty direction.

This is an explicit eager boundary for small, already-retained priors. The
transport reduction, affine reconstruction and unresolved covariance stay fixed.
Arrays in the returned scientific value are borrowed: do not mutate them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from dask import compute
import numpy as np
from openghg.util import cf_ureg
import xarray as xr
from scipy.linalg import qr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.correlated_state import CorrelatedLognormalPrior


@dataclass(frozen=True, eq=False)
class BiospherePrior:
    """Baseline, calibrated moments and analytic audit for a retained prior.

    ``data`` is eagerly materialized by :func:`calibrate_biosphere_prior`.
    ``positive_prior`` is present only for the positive lognormal family. The
    Gaussian option is a signed-template diagnostic, conditionally Gaussian
    given the protected lognormal states, generally a mixture marginally.
    """

    baseline: CorrelatedLognormalPrior
    data: xr.Dataset
    positive_prior: CorrelatedLognormalPrior | None

    def to_dataset(self) -> xr.Dataset:
        """Return the labelled serialization payload, including baseline moments."""
        return self.data

    @classmethod
    def from_dataset(cls, data: xr.Dataset) -> BiospherePrior:
        """Recompute and check a saved calibration; never trust cached moments."""
        if data.attrs.get("biosphere_prior_schema") != "openghg_inversions.biosphere_prior/v1":
            raise ValueError("Unsupported biosphere-prior schema.")
        baseline = CorrelatedLognormalPrior(
            data.baseline_mean,
            data.baseline_covariance,
            covariance_dim=data.attrs["covariance_dim"],
        )
        report = json.loads(data.attrs["calibration_report"])
        policy = report["policy"]
        if policy not in ("residual_sd_multiplier", "target_residual_sd", "target_gross_sd"):
            raise ValueError("Unsupported stored calibration policy.")
        value = report["requested"]
        choice = {
            policy: value
            if policy == "residual_sd_multiplier"
            else xr.DataArray(value, attrs={"units": report["units"]})
        }
        result = calibrate_biosphere_prior(
            baseline,
            biosphere=data.biosphere,
            weights=data.weights,
            reference=data.reference,
            family=data.attrs["family"],
            **choice,
        )
        for name in (
            "covariance",
            "residual_weights",
            "functional_mean",
            "functional_sd",
            "functional_correlation",
        ):
            try:
                xr.testing.assert_allclose(result.data[name], data[name], rtol=1e-12, atol=1e-12)
            except AssertionError as exc:
                raise ValueError(f"Stored {name} is inconsistent with the calibration policy.") from exc
        if not np.isclose(
            result.data.attrs["residual_sd_multiplier"],
            data.attrs["residual_sd_multiplier"],
            rtol=1e-13,
            atol=0,
        ):
            raise ValueError("Stored multiplier is inconsistent with the requested calibration.")
        return result

    def lognormal_marginal(self, labels: xr.DataArray | list) -> CorrelatedLognormalPrior:
        """Select exact lognormal marginal moments, never a Schur complement.

        Calibrate the full prior first. This supports enhancement-only inputs
        whose retained operators and aggregation covariance were prepared
        separately. It does not regenerate a transport reduction.
        """
        if self.positive_prior is None:
            raise ValueError("This selection requires the lognormal family.")
        dim, cov_dim = self.baseline.state_dim, self.baseline.covariance_dim
        positions = self.data.get_index(dim).get_indexer(labels)
        if (positions < 0).any() or len(np.unique(positions)) != len(positions):
            raise ValueError("Marginal labels must be unique and belong to the full state.")
        selected = self.data.isel({dim: positions, cov_dim: positions})
        return CorrelatedLognormalPrior(
            selected.baseline_mean,
            selected.covariance,
            covariance_dim=cov_dim,
        )


def calibrate_biosphere_prior(
    baseline: CorrelatedLognormalPrior,
    *,
    biosphere: xr.DataArray,
    weights: xr.DataArray,
    reference: xr.DataArray,
    residual_sd_multiplier: float | None = None,
    target_residual_sd: xr.DataArray | None = None,
    target_gross_sd: xr.DataArray | None = None,
    family: Literal["lognormal", "gaussian"] = "lognormal",
) -> BiospherePrior:
    """Change one residual-gross SD while preserving all control covariances.

    Args:
        baseline: Existing positive labelled retained-state arithmetic moments.
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
        family: Positive correlated lognormal (primary) or signed-template
            conditional Gaussian biosphere diagnostic. No truncation or repair.

    Returns:
        Eager labelled baseline/result moments, functionals, SDs, correlations,
        rank and validity report. Redundant standardized controls are explicitly
        reduced by pivoted QR; zero residual variance rejects an intervention.

    Raises:
        ValueError: For ambiguous policies, invalid selectors/units/moments,
            infeasible targets, or absent residual uncertainty. Singular changed
            arithmetic or latent covariance is rejected, without adding jitter.
    """
    if family not in ("lognormal", "gaussian"):
        raise ValueError("family must be 'lognormal' or 'gaussian'.")
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
    weights, reference, biosphere = compute(to_dense(weights), to_dense(reference), to_dense(biosphere))
    w, totals, bio = weights.to_numpy(), reference.to_numpy(), biosphere.to_numpy()
    if not np.isfinite(w).all() or not np.isfinite(totals).all():
        raise ValueError("Reporting functionals must be finite.")
    if not bio.any():
        raise ValueError("Select at least one biosphere state.")
    if (w[0] > 0).any() or (w[1] < 0).any():
        raise ValueError("GPP weights must be negative uptake and TER positive respiration.")
    b = covariance.to_numpy()
    np.linalg.cholesky(b)
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
    c = standardized[pivot[:rank]]
    residual = gross - c.T @ np.linalg.solve(c @ b @ c.T, c @ b @ gross)
    direction = b @ residual
    residual_variance = float(residual @ direction)
    gross_variance = float(gross @ b @ gross)
    explained = max(0.0, gross_variance - residual_variance)
    if residual_variance <= np.finfo(float).eps * max(gross_variance, np.finfo(float).tiny) * mean.size:
        if residual_sd_multiplier != 1:
            raise ValueError(
                "Gross exchange has zero residual variance after the controls; no mode to calibrate."
            )
        residual = np.zeros_like(gross)
        direction = np.zeros_like(gross)
        residual_variance = 0.0
        explained = gross_variance

    policy = "residual_sd_multiplier"
    requested: float
    if target_residual_sd is not None or target_gross_sd is not None:
        target = target_residual_sd if target_residual_sd is not None else target_gross_sd
        assert target is not None
        if target.ndim != 0 or not target.attrs.get("units"):
            raise ValueError("SD target must be scalar with declared flux units.")
        target_value = float(cf_ureg.Quantity(target.item(), target.attrs["units"]).to(units).magnitude)
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
            else b + (np.square(factor) - 1) * np.outer(direction, direction) / residual_variance
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
    positive = None
    lognormal_valid = True
    lognormal_error = ""
    try:
        positive = (
            baseline if factor == 1 else CorrelatedLognormalPrior(mean, changed_array, covariance_dim=cov_dim)
        )
    except ValueError as exc:
        lognormal_valid, lognormal_error = False, str(exc)
        if family == "lognormal":
            raise ValueError(f"Requested biosphere calibration has invalid lognormal moments: {exc}") from exc
    if family == "gaussian":
        # Validate the protected marginal and conditional arithmetic covariance.
        other = np.flatnonzero(~bio)
        if other.size:
            CorrelatedLognormalPrior(
                mean.isel({dim: other}), changed[np.ix_(other, other)], covariance_dim=cov_dim
            )
        positive = None

    functionals = np.vstack([w, net, gross, residual])
    names = ["GPP", "TER", "net", "gross", "residual_gross"]
    functional_cov = np.stack([functionals @ matrix @ functionals.T for matrix in (b, changed)])
    sd = np.sqrt(np.diagonal(functional_cov, axis1=1, axis2=2))
    correlation = np.divide(
        functional_cov,
        sd[:, :, None] * sd[:, None, :],
        out=np.zeros_like(functional_cov),
        where=(sd[:, :, None] * sd[:, None, :]) > 0,
    )
    means = [totals[0], totals[1], totals.sum(), -totals[0] + totals[1], 0.0]
    protected_change = np.abs((changed - b)[~bio])
    report = {
        "family": family,
        "policy": policy,
        "requested": float(requested),
        "units": units,
        "residual_sd_multiplier": float(factor),
        "explained_gross_variance": explained,
        "control_rows": len(controls),
        "control_rank": rank,
        "redundant_control_rows": len(controls) - rank,
        "max_control_covariance_change": float(np.abs(controls @ (changed - b)).max()),
        "max_protected_covariance_change": float(protected_change.max()) if protected_change.size else 0.0,
        "baseline_arithmetic_positive_definite": True,
        "baseline_lognormal_valid": True,
        "arithmetic_positive_definite": True,
        "lognormal_valid": lognormal_valid,
        "lognormal_error": lognormal_error,
        "functional_mean": dict(zip(names, means)),
        "baseline_sd": dict(zip(names, sd[0].tolist())),
        "requested_sd": dict(zip(names, sd[1].tolist())),
        "scope": "retained_state_prior_with_fixed_reduction_reconstruction_and_aggregation",
    }
    data = xr.Dataset(
        {
            "baseline_mean": mean,
            "baseline_covariance": covariance,
            "covariance": changed_array,
            "biosphere": biosphere,
            "weights": weights,
            "reference": reference,
            "residual_weights": mean.copy(data=residual),
            "functional_mean": (("prior", "functional"), [means, means]),
            "functional_sd": (("prior", "functional"), sd),
            "functional_correlation": (("prior", "functional", "functional_cov"), correlation),
        },
        coords={"prior": ["baseline", "requested"], "functional": names, "functional_cov": names},
        attrs={
            "biosphere_prior_schema": "openghg_inversions.biosphere_prior/v1",
            "state_dim": dim,
            "covariance_dim": cov_dim,
            "family": family,
            "residual_sd_multiplier": float(factor),
            "calibration_report": json.dumps(report, allow_nan=False),
        },
    )
    for name in ("functional_mean", "functional_sd", "residual_weights"):
        data[name].attrs["units"] = units
    return BiospherePrior(baseline, data, positive)
