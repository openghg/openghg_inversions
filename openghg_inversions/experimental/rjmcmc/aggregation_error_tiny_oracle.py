"""Support-aware exact references for corrected tiny root-NLE experiments.

This module owns the public tiny-case definitions used by the corrected
score-regularized exploration.  It deliberately does not import an executable
screen or its private checkerboard state grid.

For a root state, allocation is marginalized with endpoint-aware
Gauss--Jacobi quadrature.  The Gamma total is integrated adaptively in
``z = log(T)`` rather than with a fixed generalized-Laguerre rule.  This is
important for a boundary-heavy prior whose likelihood concentrates the
posterior between the moving Laguerre nodes.

The boundary-heavy two-cell evidence can also be evaluated independently in
the native coordinates ``(log(X_1), log(X_2))``.  Agreement between the two
coordinate systems is an oracle diagnostic, never structural information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Literal, TypeAlias, cast
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special, stats

from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    FourCellAggregationOracle,
    TwoCellAggregationOracle,
)

FloatArray: TypeAlias = NDArray[np.float64]
Family = Literal["two_cell", "four_cell"]

SCHEMA = "rjmcmc-score-nle-support-aware-tiny-root-oracle-v1"
BOUNDARY_CASE_ID = "boundary_heavy__two_cell__root"
PRIMARY_LOG_EVIDENCE_TOLERANCE_NAT = 0.0025
INDEPENDENT_LOG_EVIDENCE_TOLERANCE_NAT = 0.0025
POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD = 0.005
POSTERIOR_SD_RELATIVE_TOLERANCE = 0.002


@dataclass(frozen=True, slots=True)
class TinyRootCase:
    """One frozen native Gamma model and synthetic observation."""

    case_id: str
    family: Family
    shapes: tuple[float, ...]
    gamma_rate: float
    design: tuple[tuple[float, ...], ...]
    observation: tuple[float, ...]
    noise_sd: tuple[float, ...]

    def arrays(
        self,
    ) -> tuple[FloatArray, float, FloatArray, FloatArray, FloatArray]:
        """Return owned numerical arrays for this case."""
        return (
            np.asarray(self.shapes, dtype=np.float64),
            self.gamma_rate,
            np.asarray(self.design, dtype=np.float64),
            np.asarray(self.observation, dtype=np.float64),
            np.asarray(self.noise_sd, dtype=np.float64),
        )


CASES = (
    TinyRootCase(
        case_id="near_gaussian__two_cell__root",
        family="two_cell",
        shapes=(45.0, 55.0),
        gamma_rate=100.0,
        design=((1.0, 0.7), (0.2, 1.1), (-0.5, 0.3)),
        observation=(0.93, 0.71, -0.08),
        noise_sd=(0.42, 0.55, 0.48),
    ),
    TinyRootCase(
        case_id="near_gaussian__four_cell__root",
        family="four_cell",
        shapes=(40.0, 35.0, 45.0, 30.0),
        gamma_rate=150.0,
        design=(
            (1.00, 0.82, 0.45, 0.30),
            (0.15, 0.42, 0.90, 1.10),
            (-0.50, -0.10, 0.35, 0.55),
            (0.70, 0.62, 0.78, 0.85),
        ),
        observation=(0.72, 0.64, 0.04, 0.79),
        noise_sd=(0.40, 0.48, 0.45, 0.52),
    ),
    TinyRootCase(
        case_id="skewed__two_cell__root",
        family="two_cell",
        shapes=(0.35, 4.0),
        gamma_rate=4.35,
        design=((1.8, 0.1), (-0.4, 1.2), (0.8, -0.3)),
        observation=(0.44, 0.91, -0.08),
        noise_sd=(0.25, 0.32, 0.28),
    ),
    TinyRootCase(
        case_id="skewed__four_cell__root",
        family="four_cell",
        shapes=(0.35, 4.0, 1.2, 8.0),
        gamma_rate=13.55,
        design=(
            (1.80, 0.10, 0.50, -0.20),
            (-0.40, 1.20, 0.20, 0.85),
            (0.80, -0.30, 1.45, 0.10),
            (0.20, 0.35, -0.15, 1.60),
        ),
        observation=(0.23, 0.83, 0.36, 1.12),
        noise_sd=(0.22, 0.30, 0.26, 0.34),
    ),
    TinyRootCase(
        case_id=BOUNDARY_CASE_ID,
        family="two_cell",
        shapes=(0.12, 0.18),
        gamma_rate=0.30,
        design=((2.0, 0.0), (0.0, 1.7), (1.0, -1.0)),
        observation=(1.75, 0.08, 0.94),
        noise_sd=(0.12, 0.14, 0.13),
    ),
    TinyRootCase(
        case_id="boundary_heavy__four_cell__root",
        family="four_cell",
        shapes=(0.15, 0.18, 0.20, 0.12),
        gamma_rate=0.65,
        design=(
            (2.00, 0.00, 0.10, 0.00),
            (0.00, 1.70, 0.00, 0.10),
            (0.05, 0.00, 1.90, 0.00),
            (0.00, 0.10, 0.00, 2.10),
        ),
        observation=(1.62, 0.08, 0.13, 0.06),
        noise_sd=(0.12, 0.14, 0.13, 0.15),
    ),
)
CASE_IDS = tuple(case.case_id for case in CASES)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def definitions_sha256() -> str:
    """Return the strict identity of all corrected tiny case definitions."""
    return _sha256_json([asdict(case) for case in CASES])


def tiny_root_case(case_id: str) -> TinyRootCase:
    """Return one frozen root case or fail closed."""
    if not isinstance(case_id, str):
        raise TypeError("case_id must be a string.")
    for case in CASES:
        if case.case_id == case_id:
            return case
    raise ValueError("case_id is not one of the six corrected tiny root cases.")


def _positive_order(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _adaptive_quad(
    function: Any,
    lower: float,
    upper: float,
    **kwargs: Any,
) -> tuple[float, float]:
    """Call SciPy quad behind one stable scalar typing boundary."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", integrate.IntegrationWarning)
        result = cast(Any, integrate.quad)(
            function,
            lower,
            upper,
            **kwargs,
        )
    return float(result[0]), float(result[1])


def _logsumexp_scalar(values: FloatArray) -> float:
    """Return SciPy log-sum-exp behind one stable scalar typing boundary."""
    return float(cast(Any, special.logsumexp)(values))


def _conditional_log_likelihood_function(
    case: TinyRootCase,
    *,
    fraction_order: int,
) -> Any:
    shapes, rate, design, observation, noise = case.arrays()
    order = _positive_order(fraction_order, name="fraction_order")
    if case.family == "two_cell":
        oracle = TwoCellAggregationOracle(
            gamma_shape=float(shapes.sum()),
            gamma_rate=rate,
            beta_first_shape=float(shapes[0]),
            beta_second_shape=float(shapes[1]),
            total_order=1,
            fraction_order=order,
        )

        def evaluate(total: ArrayLike) -> float | FloatArray:
            return oracle.coarse_conditional_log_likelihood(
                total,
                observation,
                design,
                noise,
            )

        return evaluate
    oracle4 = FourCellAggregationOracle(
        native_shapes=shapes,
        gamma_rate=rate,
        total_order=1,
        fraction_order=order,
    )

    def evaluate_four(total: ArrayLike) -> float | FloatArray:
        return oracle4.conditional_log_likelihood(
            total,
            observation,
            design,
            noise,
            tiling="root",
        )

    return evaluate_four


def root_conditional_log_likelihood(
    case_id: str,
    total_mass: ArrayLike,
    *,
    fraction_order: int = 64,
) -> float | FloatArray:
    """Evaluate the exact allocation-marginalized likelihood at fixed total."""
    case = tiny_root_case(case_id)
    return _conditional_log_likelihood_function(
        case,
        fraction_order=fraction_order,
    )(total_mass)


@dataclass(frozen=True, slots=True)
class AdaptiveRootSummary:
    """One adaptive log-total reference and its numerical diagnostics."""

    schema: str
    case_id: str
    definitions_sha256: str
    method: str
    fraction_order: int
    lower_log_total: float
    upper_log_total: float
    epsabs: float
    epsrel: float
    posterior_mode_total: float
    log_evidence: float
    posterior_mean_total: float
    posterior_sd_total: float
    posterior_lower_0_025: float
    posterior_median: float
    posterior_upper_0_975: float
    scaled_quadrature_error: float
    represented_prior_mass: float
    represented_posterior_mass: float
    posterior_mass_accounting: str
    mode_included: bool
    sha256: str

    def payload(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_sha256:
            result.pop("sha256")
        return cast(dict[str, object], result)

    def verify(self) -> None:
        if _sha256_json(self.payload(include_sha256=False)) != self.sha256:
            raise ValueError("adaptive root summary SHA-256 does not replay.")


def adaptive_log_total_summary(
    case_id: str,
    *,
    fraction_order: int,
    epsabs: float = 1.0e-10,
    epsrel: float = 1.0e-10,
    prior_tail_probability: float = 1.0e-15,
) -> AdaptiveRootSummary:
    """Integrate one root evidence and total posterior adaptively in log mass."""
    case = tiny_root_case(case_id)
    order = _positive_order(fraction_order, name="fraction_order")
    if not 0.0 < epsabs < 1.0 or not 0.0 < epsrel < 1.0:
        raise ValueError("epsabs and epsrel must lie strictly between zero and one.")
    if not 0.0 < prior_tail_probability < 0.5:
        raise ValueError("prior_tail_probability must lie strictly between zero and one half.")
    shapes, rate, _, _, noise = case.arrays()
    gamma_shape = float(shapes.sum())
    lower_total = float(
        stats.gamma.ppf(
            prior_tail_probability,
            a=gamma_shape,
            scale=1.0 / rate,
        )
    )
    upper_total = float(
        stats.gamma.ppf(
            1.0 - prior_tail_probability,
            a=gamma_shape,
            scale=1.0 / rate,
        )
    )
    lower_z = math.log(max(lower_total, np.finfo(np.float64).tiny))
    upper_z = math.log(upper_total)
    conditional = _conditional_log_likelihood_function(
        case,
        fraction_order=order,
    )
    log_gamma_normalizer = gamma_shape * math.log(rate) - float(special.gammaln(gamma_shape))

    def log_integrand(z: float) -> float:
        if z < math.log(np.finfo(np.float64).tiny) or z > math.log(np.finfo(np.float64).max):
            return -math.inf
        total = math.exp(z)
        log_prior_with_jacobian = log_gamma_normalizer + gamma_shape * z - rate * total
        return log_prior_with_jacobian + float(conditional(total))

    optimization = cast(Any, optimize.minimize_scalar)(
        lambda value: -log_integrand(float(value)),
        bounds=(lower_z, upper_z),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not optimization.success:
        raise RuntimeError("posterior log-total mode optimization failed.")
    mode_z = float(optimization.x)
    log_scale = log_integrand(mode_z)

    def scaled_density(z: float) -> float:
        log_value = log_integrand(z) - log_scale
        return 0.0 if log_value < -745.0 else math.exp(log_value)

    normalizer, normalizer_error = _adaptive_quad(
        scaled_density,
        lower_z,
        upper_z,
        epsabs=epsabs,
        epsrel=epsrel,
        points=(mode_z,),
        limit=400,
    )
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("adaptive log-total normalizer is invalid.")

    def posterior_moment(order_value: int) -> float:
        value, _ = _adaptive_quad(
            lambda z: scaled_density(z) * math.exp(order_value * z),
            lower_z,
            upper_z,
            epsabs=epsabs,
            epsrel=epsrel,
            points=(mode_z,),
            limit=400,
        )
        return value / normalizer

    mean = posterior_moment(1)
    second = posterior_moment(2)
    sd = math.sqrt(max(second - mean * mean, 0.0))

    def posterior_cdf(z: float) -> float:
        value, _ = _adaptive_quad(
            scaled_density,
            lower_z,
            z,
            epsabs=epsabs,
            epsrel=epsrel,
            points=(mode_z,) if z > mode_z else None,
            limit=400,
        )
        return value / normalizer

    def quantile(probability: float) -> float:
        root = float(
            cast(Any, optimize.brentq)(
                lambda z: posterior_cdf(float(z)) - probability,
                lower_z,
                upper_z,
                xtol=1.0e-12,
                rtol=np.float64(1.0e-12),
            )
        )
        return math.exp(root)

    prior_retained = float(
        stats.gamma.cdf(upper_total, a=gamma_shape, scale=1.0 / rate)
        - stats.gamma.cdf(lower_total, a=gamma_shape, scale=1.0 / rate)
    )
    log_evidence = math.log(normalizer) + log_scale
    omitted_prior_mass = max(1.0 - prior_retained, 0.0)
    gaussian_log_density_upper_bound = -0.5 * (
        noise.size * math.log(2.0 * math.pi) + 2.0 * float(np.log(noise).sum())
    )
    if omitted_prior_mass == 0.0:
        posterior_retained_lower_bound = 1.0
    else:
        log_omitted_evidence_upper_bound = math.log(omitted_prior_mass) + gaussian_log_density_upper_bound
        posterior_retained_lower_bound = 1.0 / (
            1.0 + math.exp(log_omitted_evidence_upper_bound - log_evidence)
        )
    without_sha: dict[str, object] = {
        "schema": SCHEMA,
        "case_id": case_id,
        "definitions_sha256": definitions_sha256(),
        "method": "adaptive_log_total_with_gauss_jacobi_allocation",
        "fraction_order": order,
        "lower_log_total": lower_z,
        "upper_log_total": upper_z,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "posterior_mode_total": math.exp(mode_z),
        "log_evidence": log_evidence,
        "posterior_mean_total": mean,
        "posterior_sd_total": sd,
        "posterior_lower_0_025": quantile(0.025),
        "posterior_median": quantile(0.5),
        "posterior_upper_0_975": quantile(0.975),
        "scaled_quadrature_error": normalizer_error / normalizer,
        "represented_prior_mass": prior_retained,
        "represented_posterior_mass": posterior_retained_lower_bound,
        "posterior_mass_accounting": (
            "conservative lower bound from omitted Gamma prior mass times "
            "the global normalized-Gaussian density upper bound"
        ),
        "mode_included": bool(lower_z < mode_z < upper_z),
    }
    summary = AdaptiveRootSummary(
        **cast(Any, without_sha),
        sha256=_sha256_json(without_sha),
    )
    summary.verify()
    return summary


@dataclass(frozen=True, slots=True)
class NativeLogMassSummary:
    """Independent adaptive evidence/moments in native log-mass coordinates."""

    schema: str
    case_id: str
    definitions_sha256: str
    method: str
    lower_log_mass: float
    upper_log_mass: float
    epsabs: float
    epsrel: float
    log_evidence: float
    posterior_mean_total: float
    posterior_sd_total: float
    scaled_quadrature_error: float
    maximum_inner_scaled_quadrature_error: float
    sha256: str

    def payload(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_sha256:
            result.pop("sha256")
        return cast(dict[str, object], result)

    def verify(self) -> None:
        if _sha256_json(self.payload(include_sha256=False)) != self.sha256:
            raise ValueError("native log-mass summary SHA-256 does not replay.")


@dataclass(frozen=True, slots=True)
class SupportAudit:
    """Mass accounting for any proposed pointwise evaluation subset."""

    retained_prior_mass: float
    omitted_prior_mass: float
    retained_posterior_mass: float
    omitted_posterior_mass: float
    posterior_mode_included: bool
    posterior_weighted_metric_valid: bool
    conditional_renormalization_permitted: bool


def audit_evaluation_support(
    log_prior_weights: ArrayLike,
    exact_log_likelihood: ArrayLike,
    mask: ArrayLike,
    *,
    minimum_posterior_mass: float = 1.0 - 1.0e-6,
) -> SupportAudit:
    """Report unnormalized retained masses before any subset renormalization."""
    log_prior = np.asarray(log_prior_weights, dtype=np.float64)
    log_likelihood = np.asarray(exact_log_likelihood, dtype=np.float64)
    selected = np.asarray(mask)
    if (
        log_prior.ndim != 1
        or log_likelihood.shape != log_prior.shape
        or selected.shape != log_prior.shape
        or selected.dtype != np.bool_
        or log_prior.size == 0
    ):
        raise ValueError("support inputs must be aligned non-empty vectors and a Boolean mask.")
    if not np.all(np.isfinite(log_prior)) or not np.all(np.isfinite(log_likelihood)) or not np.any(selected):
        raise ValueError("support inputs must be finite and select at least one state.")
    if not 0.0 < minimum_posterior_mass <= 1.0:
        raise ValueError("minimum_posterior_mass must lie in (0, 1].")
    log_prior_total = _logsumexp_scalar(log_prior)
    prior_retained = math.exp(_logsumexp_scalar(log_prior[selected]) - log_prior_total)
    log_joint = log_prior + log_likelihood
    log_evidence = _logsumexp_scalar(log_joint)
    posterior_retained = math.exp(_logsumexp_scalar(log_joint[selected]) - log_evidence)
    mode_included = bool(selected[int(np.argmax(log_joint))])
    valid = bool(posterior_retained >= minimum_posterior_mass and mode_included)
    return SupportAudit(
        retained_prior_mass=prior_retained,
        omitted_prior_mass=max(1.0 - prior_retained, 0.0),
        retained_posterior_mass=posterior_retained,
        omitted_posterior_mass=max(1.0 - posterior_retained, 0.0),
        posterior_mode_included=mode_included,
        posterior_weighted_metric_valid=valid,
        conditional_renormalization_permitted=valid,
    )


def native_log_mass_summary(
    case_id: str = BOUNDARY_CASE_ID,
    *,
    lower_log_mass: float,
    epsabs: float = 2.0e-8,
    epsrel: float = 2.0e-8,
) -> NativeLogMassSummary:
    """Independently integrate the two-cell model in native log masses."""
    case = tiny_root_case(case_id)
    if case.family != "two_cell":
        raise ValueError("native log-mass cross-check currently supports two cells.")
    if not math.isfinite(lower_log_mass) or lower_log_mass >= -1.0:
        raise ValueError("lower_log_mass must be a finite negative tail bound.")
    shapes, rate, design, observation, noise = case.arrays()
    upper_mass = max(
        float(stats.gamma.ppf(1.0 - 1.0e-14, a=float(shape), scale=1.0 / rate)) for shape in shapes
    )
    upper_z = math.log(upper_mass)
    gaussian_constant = -0.5 * (observation.size * math.log(2.0 * math.pi) + 2.0 * float(np.log(noise).sum()))
    gamma_constants = shapes * math.log(rate) - special.gammaln(shapes)

    def log_joint(z: FloatArray) -> float:
        masses = np.exp(z)
        residual = (observation - design @ masses) / noise
        log_prior_with_jacobian = float(np.sum(gamma_constants + shapes * z - rate * masses))
        return log_prior_with_jacobian + gaussian_constant - 0.5 * float(residual @ residual)

    mode = cast(Any, optimize.minimize)(
        lambda z: -log_joint(np.asarray(z, dtype=np.float64)),
        x0=np.log(shapes / rate),
        method="L-BFGS-B",
        bounds=((lower_log_mass, upper_z), (lower_log_mass, upper_z)),
        options={"ftol": 1.0e-15, "gtol": 1.0e-10, "maxiter": 2_000},
    )
    if not mode.success:
        raise RuntimeError("native log-mass mode optimization failed.")
    mode_z = np.asarray(mode.x, dtype=np.float64)
    log_scale = log_joint(mode_z)

    def integrate_moment(moment_order: int) -> tuple[float, float, float]:
        maximum_inner_error = 0.0

        def outer(z_first: float) -> float:
            nonlocal maximum_inner_error

            def inner(z_second: float) -> float:
                z = np.asarray((z_first, z_second), dtype=np.float64)
                log_value = log_joint(z) - log_scale
                if log_value < -745.0:
                    return 0.0
                return math.exp(log_value) * (math.exp(z_first) + math.exp(z_second)) ** moment_order

            value, error = _adaptive_quad(
                inner,
                lower_log_mass,
                upper_z,
                epsabs=epsabs,
                epsrel=epsrel,
                points=(float(mode_z[1]),),
                limit=400,
            )
            maximum_inner_error = max(maximum_inner_error, error)
            return value

        value, error = _adaptive_quad(
            outer,
            lower_log_mass,
            upper_z,
            epsabs=epsabs,
            epsrel=epsrel,
            points=(float(mode_z[0]),),
            limit=400,
        )
        return value, error, maximum_inner_error

    with warnings.catch_warnings():
        warnings.simplefilter("error", integrate.IntegrationWarning)
        normalizer, normalizer_error, maximum_inner_error = integrate_moment(0)
        first, _, _ = integrate_moment(1)
        second, _, _ = integrate_moment(2)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("native log-mass normalizer is invalid.")
    mean = first / normalizer
    variance = max(second / normalizer - mean * mean, 0.0)
    without_sha: dict[str, object] = {
        "schema": SCHEMA,
        "case_id": case_id,
        "definitions_sha256": definitions_sha256(),
        "method": "adaptive_native_two_dimensional_log_masses",
        "lower_log_mass": lower_log_mass,
        "upper_log_mass": upper_z,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "log_evidence": math.log(normalizer) + log_scale,
        "posterior_mean_total": mean,
        "posterior_sd_total": math.sqrt(variance),
        "scaled_quadrature_error": normalizer_error / normalizer,
        "maximum_inner_scaled_quadrature_error": (maximum_inner_error / normalizer),
    }
    summary = NativeLogMassSummary(
        **cast(Any, without_sha),
        sha256=_sha256_json(without_sha),
    )
    summary.verify()
    return summary


def boundary_oracle_certificate() -> dict[str, Any]:
    """Return the independently converged boundary-heavy oracle certificate."""
    primary = [
        adaptive_log_total_summary(
            BOUNDARY_CASE_ID,
            fraction_order=order,
        )
        for order in (16, 32, 64)
    ]
    independent = [
        native_log_mass_summary(
            BOUNDARY_CASE_ID,
            lower_log_mass=lower,
        )
        for lower in (-40.0, -80.0, -120.0)
    ]
    reference = primary[-1]
    previous = primary[-2]
    independent_reference = independent[-1]
    independent_previous = independent[-2]
    primary_log_delta = abs(reference.log_evidence - previous.log_evidence)
    independent_log_delta = abs(reference.log_evidence - independent_reference.log_evidence)
    location_delta = (
        abs(reference.posterior_mean_total - independent_reference.posterior_mean_total)
        / reference.posterior_sd_total
    )
    sd_relative_delta = (
        abs(reference.posterior_sd_total - independent_reference.posterior_sd_total)
        / reference.posterior_sd_total
    )
    primary_mean_delta = (
        abs(reference.posterior_mean_total - previous.posterior_mean_total) / reference.posterior_sd_total
    )
    primary_sd_delta = (
        abs(reference.posterior_sd_total - previous.posterior_sd_total) / reference.posterior_sd_total
    )
    primary_endpoint_delta = (
        max(
            abs(reference.posterior_lower_0_025 - previous.posterior_lower_0_025),
            abs(reference.posterior_upper_0_975 - previous.posterior_upper_0_975),
            abs(reference.posterior_median - previous.posterior_median),
        )
        / reference.posterior_sd_total
    )
    independent_tail_log_delta = abs(independent_reference.log_evidence - independent_previous.log_evidence)
    independent_tail_mean_delta = (
        abs(independent_reference.posterior_mean_total - independent_previous.posterior_mean_total)
        / reference.posterior_sd_total
    )
    independent_tail_sd_delta = (
        abs(independent_reference.posterior_sd_total - independent_previous.posterior_sd_total)
        / reference.posterior_sd_total
    )
    checks = {
        "primary_log_evidence_converged": (primary_log_delta <= PRIMARY_LOG_EVIDENCE_TOLERANCE_NAT),
        "independent_log_evidence_agrees": (independent_log_delta <= INDEPENDENT_LOG_EVIDENCE_TOLERANCE_NAT),
        "independent_posterior_mean_agrees": (location_delta <= POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD),
        "independent_posterior_sd_agrees": (sd_relative_delta <= POSTERIOR_SD_RELATIVE_TOLERANCE),
        "primary_posterior_mean_converged": (primary_mean_delta <= POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD),
        "primary_posterior_sd_converged": (primary_sd_delta <= POSTERIOR_SD_RELATIVE_TOLERANCE),
        "primary_posterior_endpoints_converged": (
            primary_endpoint_delta <= POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "independent_tail_log_evidence_converged": (
            independent_tail_log_delta <= INDEPENDENT_LOG_EVIDENCE_TOLERANCE_NAT
        ),
        "independent_tail_posterior_mean_converged": (
            independent_tail_mean_delta <= POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "independent_tail_posterior_sd_converged": (
            independent_tail_sd_delta <= POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "primary_scaled_quadrature_error_small": (reference.scaled_quadrature_error <= 1.0e-6),
        "independent_outer_scaled_quadrature_error_small": (
            independent_reference.scaled_quadrature_error <= 1.0e-6
        ),
        "independent_inner_scaled_quadrature_error_small": (
            independent_reference.maximum_inner_scaled_quadrature_error <= 1.0e-6
        ),
        "support_retains_prior_mass": reference.represented_prior_mass >= 1.0 - 1.0e-12,
        "support_retains_posterior_mass": (reference.represented_posterior_mass >= 1.0 - 1.0e-6),
        "posterior_mode_included": reference.mode_included,
    }
    without_sha: dict[str, Any] = {
        "schema": SCHEMA,
        "case_id": BOUNDARY_CASE_ID,
        "definitions_sha256": definitions_sha256(),
        "primary_order_ladder": [summary.payload() for summary in primary],
        "independent_tail_ladder": [summary.payload() for summary in independent],
        "diagnostics": {
            "primary_log_evidence_delta_nat": primary_log_delta,
            "independent_log_evidence_delta_nat": independent_log_delta,
            "independent_posterior_mean_delta_reference_sd": location_delta,
            "independent_posterior_sd_relative_delta": sd_relative_delta,
            "primary_posterior_mean_delta_reference_sd": (primary_mean_delta),
            "primary_posterior_sd_relative_delta": primary_sd_delta,
            "primary_posterior_endpoint_delta_reference_sd": (primary_endpoint_delta),
            "independent_tail_log_evidence_delta_nat": (independent_tail_log_delta),
            "independent_tail_posterior_mean_delta_reference_sd": (independent_tail_mean_delta),
            "independent_tail_posterior_sd_relative_delta": (independent_tail_sd_delta),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    return {
        **without_sha,
        "sha256": _sha256_json(without_sha),
    }


__all__ = [
    "AdaptiveRootSummary",
    "BOUNDARY_CASE_ID",
    "CASES",
    "CASE_IDS",
    "NativeLogMassSummary",
    "SCHEMA",
    "SupportAudit",
    "TinyRootCase",
    "adaptive_log_total_summary",
    "audit_evaluation_support",
    "boundary_oracle_certificate",
    "definitions_sha256",
    "native_log_mass_summary",
    "root_conditional_log_likelihood",
    "tiny_root_case",
]
