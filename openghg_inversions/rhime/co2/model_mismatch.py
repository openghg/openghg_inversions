r"""Fixed model-mismatch choices for the exact Gaussian CO2 estimator.

``likelihood.fixed_model_mismatch`` is a known concentration standard
deviation :math:`\sigma` added in quadrature to ``mf_error``. With
``sampling.method = "gaussian"`` it may be given as:

- a scalar, applied to every observation;
- a per-site mapping ``{site = ppm}``, mapped onto observations through their
  ``site`` coordinate; every retained site needs a value and unknown sites are
  refused;
- a per-measurement ``xarray.DataArray`` on ``nmeasure`` (Python API);
- an **empirical Bayes** choice ``{empirical_bayes = "prior_residual_moment",
  floor = 0.1}``, estimated per site from the prior residuals. ``floor`` is
  required; a declared mismatch or the median sigma of a previous NUTS run is a
  sensible value.

Precedence is unchanged: an explicit value (in any of these forms) wins over a
prepared ``fixed_model_mismatch`` field, which wins over no mismatch.

Empirical Bayes uses the prior residual :math:`r = y - \bar y`, where
:math:`\bar y` is the affine likelihood mean at the prior mean of every state
(intercept, retained flux, boundary and offset prior means). Two modes exist:

- ``"prior_residual_moment"`` (default): per site,
  :math:`\sigma_s^2 = \max(0, \overline{r^2}_s - \overline{v}_s
  - \overline{A}_s - \overline{e^2}_s)`, where :math:`v` is the diagonal of
  the prior-predictive state covariance (:math:`H B H^T` plus the boundary and
  offset prior variance), :math:`A` the aggregation-error variance and
  :math:`e` the reported ``mf_error``. :math:`\sigma_s` is then floored.
- ``"prior_residual_rms"``: :math:`\sigma_s = \sqrt{\overline{r^2}_s}`, floored.
  This counts the prior flux spread, the boundary and offset spread, the
  aggregation error and the reported error as model mismatch too, so it
  overstates :math:`\sigma`.
- ``"type2_ml"``: per-site type-II maximum likelihood. It maximises the exact
  Gaussian marginal likelihood
  :math:`\log N(r; 0, K + \mathrm{diag}(e^2 + \sigma_{s(i)}^2))`, with
  :math:`K = J\,\mathrm{diag}(\text{prior sd}^2)J^T + A` from the same
  graph-derived affine map (retained flux, boundary and offset prior spread),
  jointly over :math:`\log\sigma_s` by L-BFGS-B with the analytic gradient,
  bounded below by the floor. Unlike the moment mode it accounts for the
  correlated, shared-mode prior spread. ``aggregation = "low_rank"`` (default)
  uses Woodbury with the prepared low-rank aggregation factor (or a
  rank-``rank`` approximation of a dense one); ``aggregation = "exact"`` factors
  the dense covariance at every evaluation. With a dense prepared covariance
  the default therefore fits sigma under the low-rank approximation (its
  relative Frobenius error is recorded) while the Paige posterior uses the
  exact covariance. The two likelihoods differ most at small sigma, where the
  approximation's diagonal tail dominates: on E-r07 the log marginal
  likelihood at the floor was -1.6e6 with the rank-512 approximation and
  -1.05e7 exactly. Choose ``aggregation = "exact"`` when sigma is small
  relative to the aggregation error. A non-converged optimisation is
  kept and logged with a loud warning; its flag is recorded. It is still
  empirical Bayes: the observations are used twice and the posterior excludes
  sigma uncertainty, and under a prior equal to the truth it measures only
  observation noise plus aggregation error, not real model error.

Both are point estimates: the posterior treats :math:`\sigma` as known, so
it carries no :math:`\sigma` uncertainty, and the observations are used twice,
to set :math:`\sigma` and to fit. Known limitations of the moment mode:

- When the prior equals the truth (for example the UOB control), the prior
  residual is pure observation noise plus aggregation error. Its expectation
  lacks the whole prior state spread that the moment mode subtracts, so the
  estimate is biased low by the mean of :math:`\mathrm{diag}(H B H^T)` (and
  the boundary and offset spread), not merely noisy, and usually returns the
  floor. An empirical-Bayes :math:`\sigma` there is no evidence about model
  error.
- When the prior-predictive variance sits in a few modes shared across
  observations, a per-site mean of :math:`r^2` has only a handful of effective
  degrees of freedom, so the subtraction is dominated by sampling error. In
  the E-r07 configuration the four shared boundary scalings contribute about
  271 ppm² of the average subtracted variance and the retained flux about
  4.7 ppm², about 98 % from the boundary.

The ``"type2_ml"`` mode, which uses the full correlated prior-predictive
covariance, is the remedy for the second limitation (not the first).

The floor is required. When every site sits at its floor and the resulting
mean :math:`e^2 + \sigma^2` is below ``1e-6`` times the mean aggregation
variance, the estimate is refused unless ``allow_zero = true``: such a
mismatch would make the posterior overconfident.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve, lapack as _lapack, solve_triangular
from scipy.optimize import minimize
import xarray as xr

# The LAPACK wrappers are generated at build time and are untyped.
lapack: Any = _lapack

EmpiricalBayesMode = Literal["prior_residual_moment", "prior_residual_rms", "type2_ml"]
Type2Aggregation = Literal["low_rank", "exact"]
EMPIRICAL_BAYES_MODES: tuple[str, ...] = get_args(EmpiricalBayesMode)


@dataclass(frozen=True, kw_only=True)
class EmpiricalBayesMismatch:
    """Per-site empirical-Bayes model mismatch, estimated from prior residuals.

    Attributes:
        floor: Required minimum per-site standard deviation in observation
            units, for example a declared mismatch or a NUTS median.
        mode: ``"prior_residual_moment"`` (default), ``"prior_residual_rms"``
            or ``"type2_ml"``; see the module documentation.
        allow_zero: Accept a result where every site is at its floor and the
            total independent variance is negligible against the aggregation
            variance.
        aggregation: ``type2_ml`` only. ``"low_rank"`` (default) evaluates the
            marginal likelihood by Woodbury with the prepared low-rank
            aggregation factor, or a rank-``rank`` approximation of a dense
            aggregation covariance; ``"exact"`` uses the dense covariance and
            one dense Cholesky per evaluation.
        rank: ``type2_ml`` with ``aggregation = "low_rank"`` and a dense
            prepared covariance: approximation rank.
        max_iterations: ``type2_ml`` only: L-BFGS-B iteration limit.
    """

    floor: float
    mode: EmpiricalBayesMode = "prior_residual_moment"
    allow_zero: bool = False
    aggregation: Type2Aggregation = "low_rank"
    rank: int = 512
    max_iterations: int = 200

    def __post_init__(self) -> None:
        """Validate the mode and the floor."""
        if self.mode not in EMPIRICAL_BAYES_MODES:
            raise ValueError(f"empirical_bayes must be one of {list(EMPIRICAL_BAYES_MODES)!r}, not {self.mode!r}.")
        if isinstance(self.floor, bool) or not isfinite(float(self.floor)) or float(self.floor) < 0:
            raise ValueError("The empirical-Bayes floor must be finite and non-negative.")
        if not isinstance(self.allow_zero, bool):
            raise TypeError("allow_zero must be a boolean.")
        if self.aggregation not in get_args(Type2Aggregation):
            raise ValueError(f"aggregation must be 'low_rank' or 'exact', not {self.aggregation!r}.")
        for name in ("rank", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


FixedMismatch = float | Mapping[str, float] | EmpiricalBayesMismatch | xr.DataArray


def parse_fixed_model_mismatch(value: object, path: str) -> float | Mapping[str, float] | EmpiricalBayesMismatch:
    """Parse a configured fixed mismatch: a scalar, a site table or an empirical-Bayes table.

    Args:
        value: Configured value.
        path: Configuration path used in messages.

    Returns:
        A non-negative scalar, a read-only site mapping or the empirical-Bayes
        choice.

    Raises:
        TypeError: If the value or an entry has the wrong type.
        ValueError: If a value is negative or non-finite, a table is empty, or
            an empirical-Bayes table has unknown options.
    """

    def number(item: object, item_path: str) -> float:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"{item_path} must be numeric.")
        result = float(item)
        if not isfinite(result) or result < 0:
            raise ValueError(f"{item_path} must be finite and non-negative.")
        return result

    if not isinstance(value, Mapping):
        return number(value, path)
    table = dict(value)
    if "empirical_bayes" in table:
        mode = table.pop("empirical_bayes")
        if not isinstance(mode, str):
            raise TypeError(f"{path}.empirical_bayes must be a mode name.")
        if "floor" not in table:
            raise ValueError(
                f"{path}.floor is required with empirical_bayes: set a minimum sigma in observation units, "
                "for example the declared mismatch or the median sigma of a previous NUTS run."
            )
        floor = number(table.pop("floor"), f"{path}.floor")
        allow_zero = table.pop("allow_zero", False)
        if not isinstance(allow_zero, bool):
            raise TypeError(f"{path}.allow_zero must be a boolean.")
        type2 = {name: table.pop(name) for name in ("aggregation", "rank", "max_iterations") if name in table}
        if type2 and mode != "type2_ml":
            raise ValueError(f"{', '.join(f'{path}.{name}' for name in type2)} require empirical_bayes = \"type2_ml\".")
        if "rank" in type2 and type2.get("aggregation") == "exact":
            raise ValueError(f"{path}.rank applies only to aggregation = \"low_rank\".")
        if table:
            raise ValueError(f"Unknown empirical-Bayes option(s): {', '.join(f'{path}.{name}' for name in table)}.")
        return EmpiricalBayesMismatch(mode=mode, floor=floor, allow_zero=allow_zero, **type2)  # type: ignore[arg-type]
    if not table:
        raise ValueError(f"{path} must not be an empty site table.")
    if any(not isinstance(site, str) or not site.strip() for site in table):
        raise TypeError(f"{path} site names must be non-empty strings.")
    return MappingProxyType({site.strip(): number(item, f"{path}.{site}") for site, item in table.items()})


def observation_sites(observations: xr.DataArray) -> NDArray[np.str_]:
    """Return the per-observation site labels.

    Raises:
        ValueError: If ``observations`` has no observation-aligned ``site`` coordinate.
    """
    site = observations.coords.get("site")
    if site is None or site.dims != observations.dims or observations.ndim != 1:
        raise ValueError("Per-site model mismatch requires an observation-aligned 'site' coordinate.")
    return np.asarray(site.values).astype(str)


def per_site_mismatch(sigma_by_site: Mapping[str, float], observations: xr.DataArray) -> xr.DataArray:
    """Map a per-site standard deviation onto the observations.

    Args:
        sigma_by_site: Standard deviation for every retained site.
        observations: Observations with a ``site`` coordinate.

    Returns:
        Labelled per-observation standard deviations named ``fixed_model_mismatch``.

    Raises:
        ValueError: If a configured site is not observed or an observed site
            has no value.
    """
    sites = observation_sites(observations)
    observed = list(dict.fromkeys(sites.tolist()))
    unknown = sorted(set(sigma_by_site) - set(observed))
    missing = [site for site in observed if site not in sigma_by_site]
    if unknown or missing:
        raise ValueError(
            "Per-site fixed_model_mismatch must name exactly the observed sites; "
            f"unknown: {unknown!r}, missing: {missing!r}."
        )
    values = np.array([sigma_by_site[site] for site in sites], dtype=np.float64)
    return xr.zeros_like(observations, dtype=np.float64).copy(data=values).rename("fixed_model_mismatch")


def empirical_bayes_mismatch(
    choice: EmpiricalBayesMismatch,
    *,
    sites: NDArray[np.str_],
    residual: NDArray[np.float64],
    prior_state_variance: NDArray[np.float64],
    aggregation_variance: NDArray[np.float64],
    error_variance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Estimate a per-site empirical-Bayes mismatch from prior residuals.

    Args:
        choice: Mode and floor.
        sites: Per-observation site labels.
        residual: ``y`` minus the affine likelihood mean at the prior state.
        prior_state_variance: Diagonal of the prior-predictive state
            covariance, including boundary and offset prior variance.
        aggregation_variance: Diagonal of the aggregation-error covariance.
        error_variance: ``mf_error ** 2``.

    Returns:
        Per-observation standard deviations and a JSON-safe record with, per
        site, the observation count, every averaged term, the raw estimate,
        the final value and whether the floor applied.

    Raises:
        ValueError: If every site is at its floor and the mean of
            ``error_variance + sigma**2`` is below ``1e-6`` times the mean
            ``aggregation_variance``, unless ``choice.allow_zero``.
    """
    if choice.mode == "type2_ml":
        raise ValueError("Use type2_ml_mismatch for the type2_ml mode.")
    sigma = np.empty_like(residual, dtype=np.float64)
    by_site: dict[str, dict[str, Any]] = {}
    for site in dict.fromkeys(sites.tolist()):
        rows = sites == site
        terms = {
            "n_observations": int(rows.sum()),
            "mean_residual_squared": float(np.mean(residual[rows] ** 2)),
            "mean_prior_state_variance": float(np.mean(prior_state_variance[rows])),
            "mean_aggregation_variance": float(np.mean(aggregation_variance[rows])),
            "mean_mf_error_variance": float(np.mean(error_variance[rows])),
        }
        if choice.mode == "prior_residual_rms":
            raw_variance = terms["mean_residual_squared"]
        else:
            raw_variance = (terms["mean_residual_squared"] - terms["mean_prior_state_variance"]
                            - terms["mean_aggregation_variance"] - terms["mean_mf_error_variance"])
        raw = float(np.sqrt(max(raw_variance, 0.0)))
        value = max(raw, choice.floor)
        sigma[rows] = value
        by_site[site] = {**terms, "raw_variance": float(raw_variance), "raw_sigma": raw, "sigma": value,
                         "floor_applied": bool(choice.floor > raw), "at_floor": bool(raw <= choice.floor)}
    independent = _check_not_negligible(choice, by_site, sigma, error_variance, aggregation_variance)
    return sigma, {"label": "empirical Bayes", **_choice_record(choice), "by_site": by_site,
                   "mean_independent_variance": independent}


def _choice_record(choice: EmpiricalBayesMismatch) -> dict[str, Any]:
    """Return the choice fields that apply to its mode."""
    record = asdict(choice)
    if choice.mode != "type2_ml":
        for name in ("aggregation", "rank", "max_iterations"):
            record.pop(name)
    elif choice.aggregation == "exact":
        record.pop("rank")
    return record


def _check_not_negligible(
    choice: EmpiricalBayesMismatch, by_site: Mapping[str, Mapping[str, Any]], sigma: NDArray[np.float64],
    error_variance: NDArray[np.float64], aggregation_variance: NDArray[np.float64],
) -> float:
    """Refuse an all-floor result with negligible independent variance unless allowed; return that variance."""
    independent = float(np.mean(error_variance + sigma**2))
    negligible = independent < 1e-6 * float(np.mean(aggregation_variance))
    if negligible and all(item["at_floor"] for item in by_site.values()) and not choice.allow_zero:
        raise ValueError(
            f"Empirical-Bayes sigma ({choice.mode}) is at the floor {choice.floor:g} at every site and the mean "
            f"independent variance {independent:.3g} is negligible against the aggregation variance, which "
            "would make the posterior overconfident. Set a larger floor (for example the declared mismatch or a "
            "NUTS median sigma), or allow_zero = true to accept it."
        )
    return independent


class GaussianMarginal:
    r"""Log marginal likelihood of prior residuals and its gradient in per-site log sigma.

    The residual ``r`` is modelled as :math:`N(0, S)` with
    :math:`S = U U^T + \mathrm{diag}(d + \sigma_{s(i)}^2) + A_{dense}`. ``U`` stacks the
    prior state factor (``J diag(prior_sd)``) and, for the low-rank route, the
    aggregation factor, with ``d`` the aggregation diagonal plus ``mf_error**2``.
    The low-rank route evaluates log det and solves by Woodbury with a Cholesky
    of the ``k x k`` capacitance matrix, ``O(n k^2)`` per evaluation; the exact
    route adds a dense aggregation covariance and factors ``S`` (``O(n^3)``).

    Args:
        residual: Prior residual ``r``.
        codes: Site index of each observation.
        factor: ``U``.
        diagonal: ``d``.
        dense: Optional dense matrix added to ``S``; selects the exact route.
            It is consumed: the factor product is added to it in place, so
            pass an owned (preferably Fortran-ordered) copy.
    """

    def __init__(
        self, residual: NDArray[np.float64], codes: NDArray[np.intp], factor: NDArray[np.float64],
        diagonal: NDArray[np.float64], dense: NDArray[np.float64] | None = None,
    ) -> None:
        self.residual = residual
        self.codes = codes
        self.n_sites = int(codes.max()) + 1 if codes.size else 0
        self.factor = np.asfortranarray(factor)
        self.diagonal = diagonal
        self.base: NDArray[np.float64] | None = None
        if dense is not None:
            self.base = np.asfortranarray(dense)
            self.base += factor @ factor.T
        self.evaluations = 0
        self.seconds = 0.0

    def __call__(self, log_sigma: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the log marginal likelihood and its gradient in ``log_sigma``."""
        start = perf_counter()
        sigma2 = np.exp(2.0 * np.asarray(log_sigma, dtype=np.float64))
        total = self.diagonal + sigma2[self.codes]
        r = self.residual
        n = r.size
        if self.base is None:
            inverse = 1.0 / total
            weighted = self.factor * np.sqrt(inverse)[:, None]
            capacitance = weighted.T @ weighted
            capacitance[np.diag_indices_from(capacitance)] += 1.0
            lower = np.linalg.cholesky(capacitance)
            rd = inverse * r
            v = cho_solve((lower, True), self.factor.T @ rd, check_finite=False)
            alpha = rd - inverse * (self.factor @ v)
            logdet = float(np.sum(np.log(total)) + 2.0 * np.sum(np.log(np.diag(lower))))
            projected = solve_triangular(lower, weighted.T, lower=True, check_finite=False)
            diagonal_inverse = inverse - inverse * np.einsum("ij,ij->j", projected, projected)
        else:
            matrix = np.array(self.base, order="F")
            matrix[np.diag_indices_from(matrix)] += total
            lower, info = lapack.dpotrf(matrix, lower=1, overwrite_a=1)
            if info != 0:
                raise np.linalg.LinAlgError("The marginal covariance is not positive definite.")
            del matrix
            alpha = cho_solve((lower, True), r, check_finite=False)
            logdet = float(2.0 * np.sum(np.log(np.diag(lower))))
            inverse_matrix, info = lapack.dpotri(lower, lower=1, overwrite_c=1)
            if info != 0:
                raise np.linalg.LinAlgError("The marginal covariance inverse failed.")
            diagonal_inverse = np.diag(inverse_matrix).copy()
            del inverse_matrix, lower
        value = -0.5 * (float(r @ alpha) + logdet + n * np.log(2.0 * np.pi))
        per_site = np.bincount(self.codes, weights=alpha**2 - diagonal_inverse, minlength=self.n_sites)
        self.evaluations += 1
        self.seconds += perf_counter() - start
        return value, np.exp(2.0 * np.asarray(log_sigma)) * per_site


def type2_ml_mismatch(
    choice: EmpiricalBayesMismatch,
    *,
    sites: NDArray[np.str_],
    residual: NDArray[np.float64],
    state_factor: NDArray[np.float64],
    aggregation_factor: NDArray[np.float64],
    aggregation_diagonal: NDArray[np.float64],
    aggregation_dense: NDArray[np.float64] | None,
    aggregation_variance: NDArray[np.float64],
    error_variance: NDArray[np.float64],
    aggregation_representation: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    r"""Estimate per-site sigma by type-II maximum likelihood (empirical Bayes).

    Maximises :math:`\log N(r; 0, U_x U_x^T + A + \mathrm{diag}(e^2 + \sigma_{s(i)}^2))`
    jointly over :math:`\log\sigma_s`, with L-BFGS-B and the analytic gradient
    :math:`\partial/\partial\log\sigma_s = \sigma_s^2 \sum_{i\in s}(\alpha_i^2 - (S^{-1})_{ii})`,
    :math:`\alpha = S^{-1} r`. The lower bound is
    ``max(floor, 1e-6 * sqrt(mean diag(K + A + e^2)))`` and each site's upper
    bound ``max(10 * rms_site, 10 * lower, sqrt(mean diag(K + A + e^2)))``; a
    site at its upper bound is flagged ``at_upper``. A failed likelihood
    evaluation at the start or the lower bound is recorded as ``nan`` with
    its reason.

    Args:
        choice: Mode ``type2_ml`` with its floor and options.
        sites: Per-observation site labels.
        residual: Prior residual.
        state_factor: ``J diag(prior_sd)``, the prior-predictive state factor.
        aggregation_factor: Low-rank aggregation factor (may have no columns).
        aggregation_diagonal: Aggregation diagonal for the low-rank route.
        aggregation_dense: Dense aggregation covariance for the exact route, else ``None``.
        aggregation_variance: Exact aggregation marginal variance, for the floor check.
        error_variance: ``mf_error ** 2``.
        aggregation_representation: Description recorded with the result.

    Returns:
        Per-observation sigma and a JSON-safe record: per-site start, final
        sigma and floor flags; the optimiser, iterations, evaluations,
        convergence, message, log marginal likelihoods at the start, the floor
        and the optimum, and timing.

    Raises:
        ValueError: For an all-floor result with negligible independent
            variance, unless ``allow_zero``.
    """
    labels = list(dict.fromkeys(sites.tolist()))
    positions = {label: index for index, label in enumerate(labels)}
    codes = np.fromiter((positions[site] for site in sites.tolist()), dtype=np.intp, count=sites.size)
    exact = aggregation_dense is not None
    factor = state_factor if exact else np.hstack([state_factor, aggregation_factor])
    diagonal = error_variance + (0.0 if exact else aggregation_diagonal)
    marginal = GaussianMarginal(residual, codes, factor, diagonal, aggregation_dense)
    # Bounds relative to the problem scale: sqrt(mean diag(K + A + e^2)).
    scale = float(np.sqrt(np.mean(np.einsum("ij,ij->i", state_factor, state_factor)
                                  + aggregation_variance + error_variance))) or 1.0
    lower_sigma = max(choice.floor, 1e-6 * scale)
    rms = np.array([np.sqrt(np.mean(residual[codes == index] ** 2)) for index in range(len(labels))])
    upper_sigma = np.maximum(np.maximum(10.0 * rms, 10.0 * lower_sigma), scale)
    start = np.clip(0.1 * rms, lower_sigma, upper_sigma)
    bounds = [(float(np.log(lower_sigma)), float(np.log(upper))) for upper in upper_sigma]

    def guarded(log_sigma: NDArray[np.float64]) -> tuple[float, str | None]:
        try:
            return marginal(log_sigma)[0], None
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            return float("nan"), f"{type(error).__name__}: {error}"

    start_value, start_failure = guarded(np.log(start))
    floor_value, floor_failure = guarded(np.full(len(labels), np.log(lower_sigma)))

    def objective(log_sigma: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value, gradient = marginal(log_sigma)
        return -value, -gradient

    began = perf_counter()
    result = minimize(objective, np.log(start), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": choice.max_iterations})
    elapsed = perf_counter() - began
    site_sigma = np.maximum(np.exp(result.x), choice.floor)
    sigma = site_sigma[codes]
    at_bound = result.x <= np.log(lower_sigma) + 1e-6
    at_upper = result.x >= np.log(upper_sigma) - 1e-6
    by_site = {
        label: {"n_observations": int(np.sum(codes == index)), "start_sigma": float(start[index]),
                "sigma": float(site_sigma[index]), "floor_applied": bool(at_bound[index] and choice.floor > 0),
                "at_floor": bool(at_bound[index]), "at_upper": bool(at_upper[index]),
                "upper_bound": float(upper_sigma[index]), "rms_residual": float(rms[index])}
        for index, label in enumerate(labels)
    }
    independent = _check_not_negligible(choice, by_site, sigma, error_variance, aggregation_variance)
    record = {
        "label": "empirical Bayes", **_choice_record(choice), "by_site": by_site,
        "mean_independent_variance": independent,
        "optimiser": "scipy L-BFGS-B in log sigma with the analytic gradient",
        "aggregation_representation": aggregation_representation,
        "converged": bool(result.success), "message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "log_marginal_likelihood": float(-result.fun), "log_marginal_likelihood_start": float(start_value),
        "log_marginal_likelihood_floor": float(floor_value),
        "log_marginal_likelihood_start_failure": start_failure, "log_marginal_likelihood_floor_failure": floor_failure,
        "sigma_lower_bound": float(lower_sigma),
        "sigma_lower_bound_rule": "max(floor, 1e-6 * sqrt(mean diag(K + A + mf_error^2)))",
        "sigma_upper_bound_rule": "per site max(10 * rms_site, 10 * lower bound, sqrt(mean diag(K + A + mf_error^2)))",
        "problem_scale": scale, "sites_at_upper_bound": [label for label, item in by_site.items() if item["at_upper"]],
        "seconds_total": elapsed, "seconds_per_evaluation": marginal.seconds / max(marginal.evaluations, 1),
        "evaluations_total": marginal.evaluations,
    }
    return sigma, record


__all__ = [
    "EMPIRICAL_BAYES_MODES", "EmpiricalBayesMismatch", "FixedMismatch", "GaussianMarginal",
    "empirical_bayes_mismatch", "observation_sites", "parse_fixed_model_mismatch", "per_site_mismatch",
    "type2_ml_mismatch",
]
