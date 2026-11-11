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
import pandas as pd
from openghg.util import cf_ureg
import xarray as xr
from scipy.linalg import qr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.correlated_state import CorrelatedLognormalPrior, _matrix_tolerance


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
        """Return the labelled serialization payload, including baseline moments.

        Returns:
            The stored Dataset itself, not a copy: baseline and calibrated
            moments, selectors, functionals and policy metadata, in the layout
            :meth:`from_dataset` reads.
        """
        return self.data

    @classmethod
    def from_dataset(cls, data: xr.Dataset) -> BiospherePrior:
        """Validate frozen moments, support and metadata without recalibrating a policy.

        Loading is an eager boundary for this small prior payload. The saved
        moments are checked for protected covariance rows and the net functional;
        no control selection or target resolution is repeated on load.

        Args:
            data: Saved labelled baseline/calibrated moments, selectors, functionals
                and resolved scalar policy metadata. Small arrays are computed here.

        Returns:
            Validated frozen prior with the saved covariance as its sampling target;
            no regression, calibration target, or transport preparation is repeated.

        Raises:
            ValueError: If the schema, family or policy is unsupported, a stored
                array has the wrong dimensions or non-finite values, the GPP/TER
                weights, selector or units are invalid, the stored metadata is
                inconsistent, or the saved moments do not preserve the protected
                covariance rows or the net-functional covariance.
        """
        if data.attrs.get("biosphere_prior_schema") != "openghg_inversions.biosphere_prior/v1":
            raise ValueError("Unsupported biosphere-prior schema; prepare again.")
        data = compute(data.map(to_dense, keep_attrs=True))[0]
        baseline = CorrelatedLognormalPrior(
            data.baseline_mean, data.baseline_covariance,
            covariance_dim=data.attrs["covariance_dim"],
        )
        family = data.attrs["family"]
        if family not in {"lognormal", "gaussian"}:
            raise ValueError("Unsupported stored biosphere-prior family.")
        report = json.loads(data.attrs["calibration_report"])
        policy = report["policy"]
        if policy not in {"residual_sd_multiplier", "target_residual_sd", "target_gross_sd"}:
            raise ValueError("Unsupported stored calibration policy.")
        dim, cov_dim = baseline.state_dim, baseline.covariance_dim
        dimensions = {
            "covariance": (dim, cov_dim), "biosphere": (dim,), "residual_weights": (dim,),
            "weights": ("process", dim), "reference": ("process",),
            "functional_mean": ("prior", "functional"), "functional_sd": ("prior", "functional"),
            "functional_correlation": ("prior", "functional", "functional_cov"),
        }
        for name, expected in dimensions.items():
            if data[name].dims != expected:
                raise ValueError(f"Stored {name} requires dimensions {expected}.")
            if not np.isfinite(data[name].values).all():
                raise ValueError(f"Stored {name} must be finite.")
        if set(data.get_index("process")) != {"GPP", "TER"} or data.sizes["process"] != 2:
            raise ValueError("Stored reporting weights require exactly GPP and TER labels.")
        selector = data.biosphere
        if selector.dtype != bool or not bool(selector.any()):
            raise ValueError("Stored biosphere selector must select labelled states.")
        units = data.weights.attrs.get("units")
        if not units or not data.reference.attrs.get("units"):
            raise ValueError("Stored calibration requires declared flux units.")
        cf_ureg.Quantity(1, data.reference.attrs["units"]).to(units)
        cf_ureg.Quantity(1, report["units"]).to(units)
        weights = data.weights.sel(process=["GPP", "TER"]).values
        if (weights[0] > 0).any() or (weights[1] < 0).any():
            raise ValueError("Stored GPP/TER weights have invalid signs.")
        factor = float(data.attrs["residual_sd_multiplier"])
        requested = report["requested"]
        if (not np.isfinite(factor) or factor <= 0 or not np.isfinite(requested) or requested < 0
                or not np.isclose(factor, report["residual_sd_multiplier"], rtol=1e-13, atol=0)
                or (policy == "residual_sd_multiplier" and factor != requested)):
            raise ValueError("Stored multiplier is inconsistent with the resolved calibration metadata.")
        if bool((data.functional_sd < 0).any()) or bool((abs(data.functional_correlation) > 1 + 1e-12).any()):
            raise ValueError("Stored functional SDs/correlations are outside their support.")
        if factor == 1 and not np.array_equal(data.covariance.values, data.baseline_covariance.values):
            raise ValueError("Stored identity calibration must retain baseline covariance.")
        positive = (CorrelatedLognormalPrior(baseline.mean, data.covariance, covariance_dim=cov_dim)
                    if family == "lognormal" else None)
        if family == "gaussian":
            covariance = data.covariance.values
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=_matrix_tolerance(covariance)):
                raise ValueError("Stored calibrated covariance must be symmetric.")
            covariance = (covariance + covariance.T) / 2
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Stored calibrated covariance must be positive definite.") from exc
            positions = np.flatnonzero(~selector.values)
            if positions.size:
                CorrelatedLognormalPrior(baseline.mean.isel({dim: positions}),
                    covariance[np.ix_(positions, positions)])
        baseline_covariance = data.baseline_covariance.values
        covariance_change = data.covariance.values - baseline_covariance
        tolerance = _matrix_tolerance(baseline_covariance)
        if not np.allclose(covariance_change[~selector.values], 0, rtol=0, atol=tolerance):
            raise ValueError("Stored calibration must preserve protected covariance rows.")
        net = weights.sum(axis=0)
        net_scale = np.max(np.abs(net))
        if net_scale > 0:
            net = net / net_scale
            if not np.allclose(net @ covariance_change, 0, rtol=0,
                               atol=tolerance * np.abs(net).sum()):
                raise ValueError("Stored calibration must preserve net-functional covariance.")
        return cls(baseline, data, positive)

    def lognormal_marginal(self, labels: xr.DataArray | list) -> CorrelatedLognormalPrior:
        """Select exact lognormal marginal moments, never a Schur complement.

        Calibrate the full prior first. This supports enhancement-only inputs
        whose retained operators and aggregation covariance were prepared
        separately. It does not regenerate a transport reduction.

        Args:
            labels: State labels to select, in the order wanted.

        Returns:
            Lognormal prior with the calibrated arithmetic mean and covariance
            restricted to ``labels``.

        Raises:
            ValueError: If the prior is Gaussian, or the labels are duplicated
                or not in the full state.
        """
        if self.positive_prior is None:
            raise ValueError("This selection requires the lognormal family.")
        dim, cov_dim = self.baseline.state_dim, self.baseline.covariance_dim
        positions = self.data.get_index(dim).get_indexer(
            pd.Index(labels.values if isinstance(labels, xr.DataArray) else labels)
        )
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
