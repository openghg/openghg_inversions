"""Experimental PyMC compound kernel for mobile fixed-``K`` full tilings.

This module composes one existing full-tiling structural Metropolis--Hastings
proposal with one topology-conditioned PyMC Hamiltonian Monte Carlo trajectory.  The
continuous HMC chart is symmetric in the active leaves: ``x_i = log(m_i)``
for leaf masses and ``y_j = log(c_j)`` for always-active coefficients.
The PyMC model uses flat computational variables plus one explicitly
normalized potential for the scientific ``(T, shares, c)`` target, including
the chart Jacobians.

Topology-dependent design columns, Dirichlet shapes, and the Dirichlet log
normalizer are ``pm.Data`` containers. A custom ``BlockedStep`` owns a harmless
one-state discrete token, draws an exact structural involution in the
authoritative log-coordinate chart, updates all three topology-data containers
before HMC, and
reseeds the PyMC HMC step from the sole NumPy PCG64 stream on every sweep.
Thus accepted, rejected, and invalid structural attempts are all followed by
exactly one HMC trajectory.

The HMC kernel is deliberately non-adapting: no tuning or step-size
randomization is allowed.  Before every trajectory, including after rejected
and invalid structural attempts, it installs the deterministic
Gamma--Dirichlet--lognormal prior plus likelihood Gauss--Newton reference
precision for the selected topology.  In-memory checkpoints retain the
scientific state, exact unconstrained coordinates, immutable metric identities,
resolved precision hash, sweep coordinate, and master PCG64 state.
Reconstructing and reseeding PyMC at every continuation boundary makes split
execution exactly replayable without persisting mutable backend sampler state.
Fresh initializer states instead pass through
:func:`canonicalize_full_tiling_pymc_hmc_fresh_state` before draw zero so their
physical values decode exactly from the authoritative log coordinates. This
explicitly experimental module performs no durable file I/O.

The principal public entry points build the topology precision and model with
:func:`build_full_tiling_pymc_hmc_topology_precision` and
:func:`build_full_tiling_pymc_hmc_model`, then run fresh or continued segments
with :func:`sample_full_tiling_pymc_hmc` and
:func:`continue_full_tiling_pymc_hmc`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib import import_module
import math
from numbers import Integral
import platform
import sys
import time
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import lognormal_mu_sigma
from .full_tiling import Axis, SplitChoice, TilingState, merge_choices
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    PosteriorTransitionTerms,
    accept_or_reject,
    build_full_tiling_posterior_state,
    propose_posterior_edge_flip,
    propose_posterior_resolution_relocation,
)
from .sampling import PCG64State

if TYPE_CHECKING:
    from pymc import Model

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]
UIntArray: TypeAlias = NDArray[np.uint64]

FULL_TILING_PYMC_HMC_SCHEDULE_ID = (
    "full_tiling_1_exact_log_mass_involution_1_topology_conditioned_pymc_hmc_v5"
)
"""Versioned identity of the structural-then-HMC sweep."""

FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID = "symmetric_log_leaf_mass_then_log_fixed_coefficient_v1"
"""Versioned ordering and interpretation of the HMC value coordinates."""

FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID = "pymc_scaling_topology_reference_precision_is_cov_false_v3"
"""Versioned meaning of the full matrix passed to PyMC with ``is_cov=False``."""

FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID = "gamma_dirichlet_lognormal_gauss_newton_reference_precision_v1"
"""Versioned arithmetic and validation used to construct each precision."""

FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID = "gamma_mean_nominal_dirichlet_shares_fixed_arithmetic_means_v1"
"""Versioned topology-dependent continuous reference coordinates."""


@dataclass(frozen=True, slots=True, eq=False)
class _LogMassInvolutionTransitionTerms(PosteriorTransitionTerms):
    """Valid structural terms scored in the authoritative PyMC ``x`` chart.

    ``PosteriorTransitionTerms`` decomposes the scientific-coordinate target.
    This specialization retains those component deltas but makes
    ``log_target_delta`` the exact transformed-coordinate target difference
    used for MH. The extra fields expose the exact scientific difference,
    chart change, and component-summation roundoff rather than hiding those
    distinctions in a proposal Jacobian or one scientific component.

    Args:
        exact_scientific_log_target_delta: Candidate minus source
            ``state.log_target`` evaluated as one binary64 subtraction.
        log_mass_chart_delta: Candidate minus source normalized log-mass chart
            log-Jacobian.
        exact_transformed_log_target_delta: Candidate minus source
            ``state.log_target + chart_log_jacobian`` evaluated in the same
            grouping used by the PyMC target comparison.
    """

    exact_scientific_log_target_delta: float = math.nan
    log_mass_chart_delta: float = math.nan
    exact_transformed_log_target_delta: float = math.nan
    component_roundoff_correction: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate and install exact transformed-coordinate MH aggregates."""
        super(_LogMassInvolutionTransitionTerms, self).__post_init__()
        values = (
            float(self.exact_scientific_log_target_delta),
            float(self.log_mass_chart_delta),
            float(self.exact_transformed_log_target_delta),
        )
        if any(math.isnan(value) for value in values):
            raise ValueError("log-involution target terms cannot be NaN.")
        if not self.valid:
            raise ValueError("log-involution transition terms must be valid.")
        if (
            self.log_q_forward_auxiliary != 0.0
            or self.log_q_reverse_auxiliary != 0.0
            or self.log_jacobian != 0.0
        ):
            raise ValueError("log-involution proposals have no auxiliary or proposal Jacobian.")
        component_delta = self.log_target_delta
        correction = 0.0 if values[0] == component_delta else values[0] - component_delta
        acceptance = values[2] + self.log_q_reverse_selection - self.log_q_forward_selection
        if math.isnan(acceptance):
            raise ValueError("log-involution acceptance ratio cannot be NaN.")
        object.__setattr__(self, "exact_scientific_log_target_delta", values[0])
        object.__setattr__(self, "log_mass_chart_delta", values[1])
        object.__setattr__(self, "exact_transformed_log_target_delta", values[2])
        object.__setattr__(self, "component_roundoff_correction", correction)
        object.__setattr__(self, "log_target_delta", values[2])
        object.__setattr__(self, "log_acceptance_ratio", float(acceptance))


__all__ = [
    "FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID",
    "FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID",
    "FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID",
    "FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID",
    "FULL_TILING_PYMC_HMC_SCHEDULE_ID",
    "FullTilingPyMCHMCCheckpoint",
    "FullTilingPyMCHMCConfig",
    "FullTilingPyMCHMCKernelSettings",
    "FullTilingPyMCHMCRuntimeIdentity",
    "FullTilingPyMCHMCSamplingResult",
    "FullTilingPyMCHMCTrace",
    "build_full_tiling_pymc_hmc_topology_precision",
    "build_full_tiling_pymc_hmc_model",
    "canonicalize_full_tiling_pymc_hmc_fresh_state",
    "continue_full_tiling_pymc_hmc",
    "full_tiling_pymc_hmc_runtime_identity",
    "sample_full_tiling_pymc_hmc",
]


def _import_runtime() -> tuple[Any, Any]:
    """Import PyMC and PyTensor with one actionable optional-runtime error."""
    try:
        pm = import_module("pymc")
        pt = import_module("pytensor.tensor")
    except ImportError as error:
        raise ImportError(
            "The experimental full-tiling compound HMC kernel requires PyMC "
            "and PyTensor from the repository environment."
        ) from error
    return pm, pt


def _require_float64() -> None:
    """Require the process-global PyTensor default to be binary64."""
    try:
        pytensor = import_module("pytensor")
    except ImportError as error:
        raise ImportError("The experimental full-tiling compound HMC kernel requires PyTensor.") from error
    float_x = str(pytensor.config.floatX)
    if float_x != "float64":
        raise RuntimeError(
            f"PyTensor floatX must be float64 for the full-tiling compound HMC kernel; found {float_x!r}."
        )


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCRuntimeIdentity:
    """Backend and coordinate identity required for exact continuation.

    Attributes:
        python_minor: Python major/minor version.
        platform_system: Operating-system family.
        platform_machine: Hardware architecture.
        numpy_version: NumPy version.
        pymc_version: PyMC version.
        pytensor_version: PyTensor version.
        pytensor_float_x: Required PyTensor floating-point default.
        coordinate_layout_id: Versioned unconstrained-coordinate ordering.
        metric_semantics_id: Versioned interpretation of the topology
            precision passed to PyMC.
    """

    python_minor: str
    platform_system: str
    platform_machine: str
    numpy_version: str
    pymc_version: str
    pytensor_version: str
    pytensor_float_x: str
    coordinate_layout_id: str
    metric_semantics_id: str


def full_tiling_pymc_hmc_runtime_identity() -> FullTilingPyMCHMCRuntimeIdentity:
    """Return the exact backend identity used by a compound HMC checkpoint.

    Returns:
        Immutable versions, precision, architecture, coordinate layout, and
        metric semantics.

    Raises:
        ImportError: If PyMC or PyTensor is unavailable.
        RuntimeError: If PyTensor is not configured for float64.
    """
    _require_float64()
    pm, _ = _import_runtime()
    pytensor = import_module("pytensor")
    return FullTilingPyMCHMCRuntimeIdentity(
        python_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        numpy_version=str(np.__version__),
        pymc_version=str(pm.__version__),
        pytensor_version=str(pytensor.__version__),
        pytensor_float_x=str(pytensor.config.floatX),
        coordinate_layout_id=FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID,
        metric_semantics_id=FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID,
    )


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite strictly positive scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    """Return one finite non-negative scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _positive_integer(value: object, *, name: str, allow_zero: bool = False) -> int:
    """Return one validated built-in integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        adjective = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {adjective}.")
    return result


def _readonly_array(
    values: object,
    *,
    dtype: np.dtype[np.generic] | type[np.generic],
    ndim: int,
    name: str,
) -> np.ndarray:
    """Return an owned read-only array of the requested rank."""
    result = np.array(values, dtype=dtype, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    result.setflags(write=False)
    return result


def _rectangle_bounds(state: FullTilingPosteriorState) -> IntArray:
    """Return canonical rectangle bounds for one fixed-``K`` state."""
    return np.asarray(
        [
            (
                leaf.row_start,
                leaf.row_stop,
                leaf.col_start,
                leaf.col_stop,
            )
            for leaf in state.tiling_state.tiling.leaves
        ],
        dtype=np.int64,
    )


def _topology_arrays(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> tuple[FloatArray, FloatArray, float]:
    """Return design, Dirichlet shapes, and its exact normalized constant."""
    design = np.column_stack([problem.design_column(leaf) for leaf in state.tiling_state.tiling.leaves])
    alpha = problem.allocation_prior.leaf_alphas(state.tiling_state.tiling)
    log_normalizer = float(
        math.lgamma(problem.allocation_prior.concentration)
        - sum(math.lgamma(float(value)) for value in alpha)
    )
    return (
        np.asarray(design, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        log_normalizer,
    )


def _fixed_log_parameters(problem: FullTilingProblem) -> tuple[FloatArray, FloatArray]:
    """Return normal-space parameters of arithmetic-moment lognormal priors."""
    block = problem.base.fixed_block
    if block is None:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    pairs = tuple(
        lognormal_mu_sigma(float(mean), float(sd))
        for mean, sd in zip(
            block.coefficient_prior_mean,
            block.coefficient_prior_sd,
            strict=True,
        )
    )
    mu, sigma = zip(*pairs, strict=True)
    return (
        np.asarray(mu, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )


def _validate_topology_precision(values: ArrayLike) -> FloatArray:
    """Return one owned, symmetric, read-only positive-definite precision.

    Args:
        values: Candidate square dense precision matrix.

    Returns:
        Owned read-only ``float64`` matrix.

    Raises:
        ValueError: If the matrix is not square, finite, exactly symmetric in
            binary64, or strictly positive definite.
    """
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or result.shape[0] < 1:
        raise ValueError("topology precision must be a non-empty square matrix.")
    if np.any(~np.isfinite(result)):
        raise ValueError("topology precision must contain only finite values.")
    if not np.array_equal(result, result.T):
        raise ValueError("topology precision must be exactly symmetric in binary64.")
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError("topology precision must be strictly positive definite.") from error
    result.setflags(write=False)
    return result


def _assemble_topology_reference_precision(
    *,
    dynamic_design: ArrayLike,
    alpha: ArrayLike,
    root_shape: float,
    root_rate: float,
    fixed_design: ArrayLike,
    fixed_reference: ArrayLike,
    fixed_log_sigma: ArrayLike,
    observation_sd: ArrayLike,
    likelihood_power: float,
) -> FloatArray:
    """Assemble the exact topology-reference prior plus Gauss--Newton precision.

    This pure array seam also defines permutation equivariance: applying the
    same leaf permutation to ``dynamic_design`` and ``alpha`` permutes the
    corresponding rows and columns of the returned precision, while the fixed
    block remains in its declared order.

    Args:
        dynamic_design: Observation-by-leaf design matrix in leaf order.
        alpha: Positive Dirichlet shapes in the same leaf order.
        root_shape: Positive Gamma root shape.
        root_rate: Positive Gamma root rate.
        fixed_design: Observation-by-fixed-coefficient design matrix.
        fixed_reference: Positive arithmetic prior means for the fixed block.
        fixed_log_sigma: Positive lognormal log-space standard deviations.
        observation_sd: Positive observation standard deviations.
        likelihood_power: Non-negative likelihood multiplier.

    Returns:
        Owned read-only dense reference precision in leaf-then-fixed order.

    Raises:
        ValueError: If input dimensions or support are inconsistent, or the
            assembled precision fails finite, symmetry, or positive-definite
            validation.
    """
    design = np.asarray(dynamic_design, dtype=np.float64)
    shapes = np.asarray(alpha, dtype=np.float64)
    fixed = np.asarray(fixed_design, dtype=np.float64)
    fixed_mean = np.asarray(fixed_reference, dtype=np.float64)
    fixed_sigma = np.asarray(fixed_log_sigma, dtype=np.float64)
    sd = np.asarray(observation_sd, dtype=np.float64)
    if design.ndim != 2 or shapes.ndim != 1 or design.shape[1] != shapes.size or shapes.size < 1:
        raise ValueError("dynamic design and alpha must define one or more aligned leaves.")
    if fixed.ndim != 2 or fixed.shape[0] != design.shape[0]:
        raise ValueError("fixed design must be two-dimensional and observation-aligned.")
    if fixed.shape[1] != fixed_mean.size or fixed_mean.shape != fixed_sigma.shape:
        raise ValueError("fixed reference arrays must align with the fixed design.")
    if sd.shape != (design.shape[0],):
        raise ValueError("observation_sd must contain one value per design row.")
    scalars = np.asarray([root_shape, root_rate, likelihood_power], dtype=np.float64)
    if (
        np.any(~np.isfinite(design))
        or np.any(~np.isfinite(shapes))
        or np.any(shapes <= 0.0)
        or np.any(~np.isfinite(fixed))
        or np.any(~np.isfinite(fixed_mean))
        or np.any(fixed_mean <= 0.0)
        or np.any(~np.isfinite(fixed_sigma))
        or np.any(fixed_sigma <= 0.0)
        or np.any(~np.isfinite(sd))
        or np.any(sd <= 0.0)
        or np.any(~np.isfinite(scalars))
        or root_shape <= 0.0
        or root_rate <= 0.0
        or likelihood_power < 0.0
    ):
        raise ValueError("topology-reference precision inputs violate finite support.")

    kappa = float(shapes.sum())
    shares = shapes / kappa
    root_reference = root_shape / root_rate
    share_covariance = np.diag(shares) - np.outer(shares, shares)
    share_outer = np.outer(shares, shares)
    leaf_prior = (
        kappa - root_shape + root_rate * root_reference
    ) * share_covariance + root_rate * root_reference * share_outer
    dimension = shapes.size + fixed_mean.size
    prior = np.zeros((dimension, dimension), dtype=np.float64)
    prior[: shapes.size, : shapes.size] = leaf_prior
    if fixed_mean.size:
        fixed_indices = np.arange(shapes.size, dimension)
        prior[fixed_indices, fixed_indices] = 1.0 / np.square(fixed_sigma)

    if likelihood_power == 0.0:
        likelihood_precision = np.zeros_like(prior)
    else:
        leaf_reference = root_reference * shares
        jacobian = np.concatenate(
            (
                design * leaf_reference[np.newaxis, :],
                fixed * fixed_mean[np.newaxis, :],
            ),
            axis=1,
        )
        whitened_jacobian = jacobian / sd[:, np.newaxis]
        with np.errstate(over="ignore", invalid="ignore"):
            likelihood_precision = likelihood_power * (whitened_jacobian.T @ whitened_jacobian)
    return _validate_topology_precision(prior + likelihood_precision)


def build_full_tiling_pymc_hmc_topology_precision(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> FloatArray:
    """Build the deterministic HMC precision for one selected topology.

    The reference root total is the Gamma prior mean ``shape / rate``. Leaf
    shares are the topology's Dirichlet shapes normalized by their sum, and
    fixed coefficients use their arithmetic lognormal prior means. The exact
    transformed Gamma--Dirichlet and lognormal prior negative Hessian is added
    to ``likelihood_power * J.T @ J`` after row whitening by
    ``observation_sd``. The current continuous values in ``state`` are never
    consulted.

    Args:
        problem: Frozen full-tiling likelihood and prior inputs.
        state: State supplying the selected topology and canonical leaf order.

    Returns:
        Owned read-only finite symmetric positive-definite ``float64`` matrix
        in log-leaf-mass then log-fixed-coefficient order.

    Raises:
        TypeError: If either argument has an incompatible public type.
        ValueError: If the state belongs to another problem or the reference
            precision is non-finite, asymmetric, or not positive definite.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(state, FullTilingPosteriorState):
        raise TypeError("state must be a FullTilingPosteriorState.")
    if state.problem is not problem:
        raise ValueError("state must belong to the exact supplied problem.")
    dynamic_design, alpha, _ = _topology_arrays(problem, state)
    fixed_block = problem.base.fixed_block
    if fixed_block is None:
        fixed_design = np.empty((problem.observations.size, 0), dtype=np.float64)
        fixed_reference = np.empty(0, dtype=np.float64)
    else:
        fixed_design = fixed_block.design
        fixed_reference = fixed_block.coefficient_prior_mean
    _, fixed_log_sigma = _fixed_log_parameters(problem)
    prior = problem.base.prior
    return _assemble_topology_reference_precision(
        dynamic_design=dynamic_design,
        alpha=alpha,
        root_shape=prior.root_shape,
        root_rate=prior.root_rate,
        fixed_design=fixed_design,
        fixed_reference=fixed_reference,
        fixed_log_sigma=fixed_log_sigma,
        observation_sd=problem.observation_sd,
        likelihood_power=problem.base.likelihood_power,
    )


def _topology_precision_sha256(precision: FloatArray) -> str:
    """Hash a precision using the canonical checkpoint byte encoding.

    Args:
        precision: Dense precision matrix.

    Returns:
        Lowercase SHA-256 hex digest of little-endian int64 shape bytes
        followed by C-order little-endian float64 matrix bytes.
    """
    digest = hashlib.sha256()
    digest.update(np.asarray(precision.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(precision, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCConfig:
    """Configuration for a fresh compound structural plus HMC segment.

    Args:
        iterations: Positive number of structural-then-HMC sweeps.
        step_size: Requested unscaled leapfrog step size. PyMC's internal
            ``exp(log(epsilon))`` representation may move the effective value
            reported in the trace by one binary64 ULP.
        leapfrog_steps: Exact positive number of leapfrog steps per sweep.
        seed: Optional non-negative seed for the sole NumPy PCG64 stream.

    Raises:
        TypeError: If integer or scalar settings have invalid types.
        ValueError: If settings lie outside their supported ranges.
    """

    iterations: int
    step_size: float
    leapfrog_steps: int
    seed: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Normalize and validate problem-independent settings."""
        object.__setattr__(
            self,
            "iterations",
            _positive_integer(self.iterations, name="iterations"),
        )
        object.__setattr__(
            self,
            "step_size",
            _positive_float(self.step_size, name="step_size"),
        )
        object.__setattr__(
            self,
            "leapfrog_steps",
            _positive_integer(self.leapfrog_steps, name="leapfrog_steps"),
        )
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                _positive_integer(self.seed, name="seed", allow_zero=True),
            )


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCKernelSettings:
    """Immutable problem-resolved topology-conditioned HMC settings.

    Args:
        fixed_k: Positive leaf count preserved by the structural kernel.
        step_size: Requested unscaled leapfrog step size. The exact effective
            PyMC value is recorded for every sweep and may differ by one
            binary64 ULP.
        leapfrog_steps: Exact number of leapfrog steps.
        metric_builder_id: Exact precision arithmetic and validation identity.
        metric_reference_id: Exact topology-reference coordinate identity.

    Raises:
        TypeError: If an integer setting has an invalid type.
        ValueError: If a trajectory control or metric identity is
            incompatible.
    """

    fixed_k: int
    step_size: float
    leapfrog_steps: int
    metric_builder_id: str = FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID
    metric_reference_id: str = FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID

    def __post_init__(self) -> None:
        """Validate all resolved settings and finite trajectory length."""
        object.__setattr__(
            self,
            "fixed_k",
            _positive_integer(self.fixed_k, name="fixed_k"),
        )
        step_size = _positive_float(self.step_size, name="step_size")
        steps = _positive_integer(self.leapfrog_steps, name="leapfrog_steps")
        if not math.isfinite(step_size * steps):
            raise ValueError("step_size times leapfrog_steps must be finite.")
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "leapfrog_steps", steps)
        if self.metric_builder_id != FULL_TILING_PYMC_HMC_METRIC_BUILDER_ID:
            raise ValueError("metric_builder_id is incompatible.")
        if self.metric_reference_id != FULL_TILING_PYMC_HMC_METRIC_REFERENCE_ID:
            raise ValueError("metric_reference_id is incompatible.")


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCTrace:
    """Every-sweep structural/HMC diagnostics and visited scientific states.

    State arrays include the segment boundary followed by every post-HMC
    state. Diagnostic arrays contain one row per completed sweep.

    Args:
        state_sweep: Global sweep coordinates of retained states.
        rectangle_bounds: Canonical bounds with shape ``(draw, K, 4)``.
        leaf_masses: Canonical positive masses with shape ``(draw, K)``.
        fixed_coefficients: Positive fixed coefficients with shape
            ``(draw, n_fixed)``.
        log_leaf_mass: Authoritative PyMC ``x`` coordinates with shape
            ``(draw, K)``.
        log_fixed_coefficient: Authoritative PyMC ``y`` coordinates with
            shape ``(draw, n_fixed)``.
        log_target: Exact scientific target value by draw.
        global_sweep: Consecutive one-based diagnostic coordinates.
        structural_move: Existing structural proposal name.
        structural_valid: Whether the structural proposal reached MH.
        structural_accepted: Whether the topology/masses changed.
        structural_log_acceptance_ratio: Untruncated structural log ratio
            after authoritative-boundary target correction for valid moves.
        structural_invalid_reason: Empty for valid proposals.
        hmc_start_log_leaf_mass: Post-structure, pre-HMC leaf log masses in the
            same canonical order as the corresponding post-HMC state.
        hmc_start_log_fixed_coefficient: Post-structure, pre-HMC fixed
            log-coefficient coordinates.
        hmc_accepted: Whether PyMC accepted the HMC endpoint.
        hmc_acceptance_probability: HMC Metropolis acceptance probability.
        hmc_diverging: Whether PyMC flagged the trajectory as divergent.
        hmc_energy: Endpoint Hamiltonian reported by PyMC. A divergent,
            rejected trajectory may report a non-finite diagnostic.
        hmc_energy_error: Endpoint-minus-start Hamiltonian error. A divergent,
            rejected trajectory may report a non-finite diagnostic.
        hmc_step_size: Actual step size reported by PyMC.
        hmc_n_steps: Leapfrog step count reported by PyMC.
        hmc_seed: Per-sweep uint64 seed drawn from the master PCG64 stream and
            used to reseed the complete PyMC HMC step.

    Raises:
        ValueError: If state arrays do not share the boundary-inclusive draw
            coordinates, diagnostic arrays do not share the sweep coordinate,
            or support and transition invariants are violated.
    """

    state_sweep: IntArray
    rectangle_bounds: IntArray
    leaf_masses: FloatArray
    fixed_coefficients: FloatArray
    log_leaf_mass: FloatArray
    log_fixed_coefficient: FloatArray
    log_target: FloatArray
    global_sweep: IntArray
    structural_move: StringArray
    structural_valid: BoolArray
    structural_accepted: BoolArray
    structural_log_acceptance_ratio: FloatArray
    structural_invalid_reason: StringArray
    hmc_start_log_leaf_mass: FloatArray
    hmc_start_log_fixed_coefficient: FloatArray
    hmc_accepted: BoolArray
    hmc_acceptance_probability: FloatArray
    hmc_diverging: BoolArray
    hmc_energy: FloatArray
    hmc_energy_error: FloatArray
    hmc_step_size: FloatArray
    hmc_n_steps: IntArray
    hmc_seed: UIntArray

    def __post_init__(self) -> None:
        """Copy arrays read-only and validate their fixed-width contracts."""
        state_specs = {
            "state_sweep": (np.int64, 1),
            "rectangle_bounds": (np.int64, 3),
            "leaf_masses": (np.float64, 2),
            "fixed_coefficients": (np.float64, 2),
            "log_leaf_mass": (np.float64, 2),
            "log_fixed_coefficient": (np.float64, 2),
            "log_target": (np.float64, 1),
        }
        diagnostic_specs = {
            "global_sweep": (np.int64, 1),
            "structural_move": (np.dtype("U24"), 1),
            "structural_valid": (np.bool_, 1),
            "structural_accepted": (np.bool_, 1),
            "structural_log_acceptance_ratio": (np.float64, 1),
            "structural_invalid_reason": (np.dtype("U96"), 1),
            "hmc_accepted": (np.bool_, 1),
            "hmc_acceptance_probability": (np.float64, 1),
            "hmc_diverging": (np.bool_, 1),
            "hmc_energy": (np.float64, 1),
            "hmc_energy_error": (np.float64, 1),
            "hmc_step_size": (np.float64, 1),
            "hmc_n_steps": (np.int64, 1),
            "hmc_seed": (np.uint64, 1),
        }
        for name, (dtype, ndim) in state_specs.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    dtype=dtype,
                    ndim=ndim,
                    name=name,
                ),
            )
        for name, (dtype, ndim) in diagnostic_specs.items():
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    dtype=dtype,
                    ndim=ndim,
                    name=name,
                ),
            )

        draws = self.state_sweep.size
        if self.rectangle_bounds.shape[0] != draws or self.rectangle_bounds.shape[2:] != (4,):
            raise ValueError("rectangle_bounds must have shape (draw, K, 4).")
        fixed_k = self.rectangle_bounds.shape[1]
        if self.leaf_masses.shape != (draws, fixed_k):
            raise ValueError("leaf_masses must have shape (draw, K).")
        if self.fixed_coefficients.shape[0] != draws:
            raise ValueError("fixed_coefficients must have one row per state.")
        if self.log_leaf_mass.shape != (draws, fixed_k):
            raise ValueError("log_leaf_mass must have shape (draw, K).")
        if self.log_fixed_coefficient.shape != self.fixed_coefficients.shape:
            raise ValueError("log_fixed_coefficient must match the fixed_coefficients shape.")
        if self.log_target.shape != (draws,):
            raise ValueError("log_target must have one entry per state.")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            decoded_leaf_mass = np.exp(self.log_leaf_mass)
            decoded_fixed = np.exp(self.log_fixed_coefficient)
        if draws and (
            np.any(np.diff(self.state_sweep) != 1)
            or np.any(self.state_sweep < 0)
            or np.any(~np.isfinite(self.leaf_masses))
            or np.any(self.leaf_masses <= 0.0)
            or np.any(~np.isfinite(self.fixed_coefficients))
            or np.any(self.fixed_coefficients <= 0.0)
            or np.any(~np.isfinite(self.log_leaf_mass))
            or np.any(~np.isfinite(self.log_fixed_coefficient))
            or not np.array_equal(decoded_leaf_mass, self.leaf_masses)
            or not np.array_equal(decoded_fixed, self.fixed_coefficients)
            or np.any(~np.isfinite(self.log_target))
        ):
            raise ValueError("retained states violate support or sweep ordering.")

        sweeps = self.global_sweep.size
        for name in diagnostic_specs:
            if getattr(self, name).shape != (sweeps,):
                raise ValueError(f"{name} must have one entry per sweep.")
        object.__setattr__(
            self,
            "hmc_start_log_leaf_mass",
            _readonly_array(
                self.hmc_start_log_leaf_mass,
                dtype=np.float64,
                ndim=2,
                name="hmc_start_log_leaf_mass",
            ),
        )
        object.__setattr__(
            self,
            "hmc_start_log_fixed_coefficient",
            _readonly_array(
                self.hmc_start_log_fixed_coefficient,
                dtype=np.float64,
                ndim=2,
                name="hmc_start_log_fixed_coefficient",
            ),
        )
        if self.hmc_start_log_leaf_mass.shape != (sweeps, fixed_k):
            raise ValueError("hmc_start_log_leaf_mass must have shape (sweep, K).")
        if self.hmc_start_log_fixed_coefficient.shape != (sweeps, self.fixed_coefficients.shape[1]):
            raise ValueError("hmc_start_log_fixed_coefficient must have shape (sweep, n_fixed).")
        if draws != sweeps + 1 or not np.array_equal(
            self.state_sweep[1:],
            self.global_sweep,
        ):
            raise ValueError(
                "state coordinates must contain one boundary followed by every diagnostic sweep."
            )
        if sweeps and (
            np.any(np.diff(self.global_sweep) != 1)
            or np.any(self.global_sweep < 1)
            or np.any(
                ~np.isin(
                    self.structural_move,
                    ("edge_flip", "resolution_relocation"),
                )
            )
            or np.any(self.structural_accepted & ~self.structural_valid)
            or np.any(np.isnan(self.structural_log_acceptance_ratio))
            or np.any(self.structural_log_acceptance_ratio == np.inf)
            or np.any(self.structural_valid & (self.structural_invalid_reason != ""))
            or np.any(~self.structural_valid & (self.structural_invalid_reason == ""))
            or np.any(~np.isfinite(self.hmc_start_log_leaf_mass))
            or np.any(~np.isfinite(self.hmc_start_log_fixed_coefficient))
            or np.any(~np.isfinite(self.hmc_acceptance_probability))
            or np.any((self.hmc_acceptance_probability < 0.0) | (self.hmc_acceptance_probability > 1.0))
            or np.any(self.hmc_accepted & self.hmc_diverging)
            or np.any(
                ~self.hmc_diverging & (~np.isfinite(self.hmc_energy) | ~np.isfinite(self.hmc_energy_error))
            )
            or np.any(~np.isfinite(self.hmc_step_size))
            or np.any(self.hmc_step_size <= 0.0)
            or np.any(self.hmc_n_steps < 1)
        ):
            raise ValueError("per-sweep diagnostics violate kernel invariants.")

    @property
    def k(self) -> int:
        """Return the fixed leaf count."""
        return int(self.rectangle_bounds.shape[1])


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCCheckpoint:
    """Exact in-memory continuation boundary for the compound HMC kernel.

    Args:
        problem: Exact in-memory problem object.
        state: Post-HMC scientific state at the boundary.
        log_leaf_mass: Exact authoritative PyMC ``x`` coordinate.
        log_fixed_coefficient: Exact authoritative PyMC ``y`` coordinate.
        rng_state: Exact state of the sole master PCG64 stream.
        sweeps_completed: Global number of completed compound sweeps.
        kernel_settings: Complete non-adapting HMC and metric settings.
        runtime_identity: Backend, precision, layout, and metric identity.
        topology_precision_sha256: Lowercase SHA-256 of little-endian int64
            shape bytes followed by C-order little-endian float64 bytes for
            the resolved precision. Validation recomputes it from ``state``.
        schedule_id: Exact compatible schedule identifier.

    Raises:
        TypeError: If fields have invalid public types.
        ValueError: If identity, dimension, support, or schedule invariants
            are inconsistent.
        ImportError: If PyMC or PyTensor is unavailable while validating
            runtime identity.
        RuntimeError: If PyTensor is not configured for float64.
    """

    problem: FullTilingProblem
    state: FullTilingPosteriorState
    log_leaf_mass: FloatArray
    log_fixed_coefficient: FloatArray
    rng_state: PCG64State
    sweeps_completed: int
    kernel_settings: FullTilingPyMCHMCKernelSettings
    runtime_identity: FullTilingPyMCHMCRuntimeIdentity
    topology_precision_sha256: str
    schedule_id: str = FULL_TILING_PYMC_HMC_SCHEDULE_ID

    def __post_init__(self) -> None:
        """Validate the complete in-memory continuation contract."""
        if not isinstance(self.problem, FullTilingProblem):
            raise TypeError("problem must be a FullTilingProblem.")
        if not isinstance(self.state, FullTilingPosteriorState):
            raise TypeError("state must be a FullTilingPosteriorState.")
        if self.state.problem is not self.problem:
            raise ValueError("checkpoint state must belong to checkpoint problem.")
        log_leaf_mass = _readonly_array(
            self.log_leaf_mass,
            dtype=np.float64,
            ndim=1,
            name="log_leaf_mass",
        )
        log_fixed = _readonly_array(
            self.log_fixed_coefficient,
            dtype=np.float64,
            ndim=1,
            name="log_fixed_coefficient",
        )
        if log_leaf_mass.shape != (self.state.k,) or np.any(~np.isfinite(log_leaf_mass)):
            raise ValueError("log_leaf_mass must contain one finite value per leaf.")
        if log_fixed.shape != self.state.fixed_coefficients.shape or np.any(~np.isfinite(log_fixed)):
            raise ValueError("log_fixed_coefficient must match the finite fixed block.")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            decoded_leaf_mass = np.exp(log_leaf_mass)
            decoded_fixed = np.exp(log_fixed)
        if not np.array_equal(decoded_leaf_mass, self.state.leaf_masses):
            raise ValueError("log_leaf_mass must exactly encode checkpoint state.")
        if not np.array_equal(decoded_fixed, self.state.fixed_coefficients):
            raise ValueError("log_fixed_coefficient must exactly encode checkpoint state.")
        if not isinstance(self.rng_state, PCG64State):
            raise TypeError("rng_state must be a PCG64State.")
        completed = _positive_integer(
            self.sweeps_completed,
            name="sweeps_completed",
            allow_zero=True,
        )
        if not isinstance(
            self.kernel_settings,
            FullTilingPyMCHMCKernelSettings,
        ):
            raise TypeError("kernel_settings must be FullTilingPyMCHMCKernelSettings.")
        if not isinstance(
            self.runtime_identity,
            FullTilingPyMCHMCRuntimeIdentity,
        ):
            raise TypeError("runtime_identity must be a FullTilingPyMCHMCRuntimeIdentity.")
        if self.state.k != self.kernel_settings.fixed_k:
            raise ValueError("checkpoint state K must match fixed kernel K.")
        if self.schedule_id != FULL_TILING_PYMC_HMC_SCHEDULE_ID:
            raise ValueError("checkpoint schedule is incompatible.")
        if self.runtime_identity != full_tiling_pymc_hmc_runtime_identity():
            raise ValueError("checkpoint runtime identity is incompatible.")
        if not math.isfinite(self.state.log_target):
            raise ValueError("checkpoint state must have finite target support.")
        expected_precision_hash = _topology_precision_sha256(
            build_full_tiling_pymc_hmc_topology_precision(
                self.problem,
                self.state,
            )
        )
        if self.topology_precision_sha256 != expected_precision_hash:
            raise ValueError("checkpoint topology precision hash is incompatible.")
        object.__setattr__(self, "log_leaf_mass", log_leaf_mass)
        object.__setattr__(self, "log_fixed_coefficient", log_fixed)
        object.__setattr__(self, "sweeps_completed", completed)


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCSamplingResult:
    """One compound HMC segment and its exact continuation boundary.

    Attributes:
        trace: Boundary-inclusive visited states and every-sweep diagnostics.
        final_state: Scientific oracle state after the final HMC step.
        checkpoint: Exact next in-memory continuation boundary.
        kernel_setup_seconds: Non-authoritative elapsed time spent building
            the PyMC model and compound step for this segment.
        transition_seconds: Non-authoritative elapsed time spent executing
            this segment's compound sweeps.

    Raises:
        TypeError: If either elapsed time is not a real number.
        ValueError: If either elapsed time is non-finite or negative.
    """

    trace: FullTilingPyMCHMCTrace
    final_state: FullTilingPosteriorState
    checkpoint: FullTilingPyMCHMCCheckpoint
    kernel_setup_seconds: float
    transition_seconds: float

    def __post_init__(self) -> None:
        """Validate and normalize the non-authoritative elapsed timings."""
        for name in ("kernel_setup_seconds", "transition_seconds"):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name=name),
            )


def build_full_tiling_pymc_hmc_model(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
) -> Model:
    """Build the exact dynamic-topology PyMC model in unconstrained charts.

    ``x`` contains log leaf masses and ``y`` contains log fixed
    coefficients. For the current topology ``tau``, the model density is
    ``log_target(T(x), shares(x), c(y), tau) + sum(x)
    - (K - 1) * logsumexp(x) + sum(y)``. At the encoded initial state this is
    ``initial_state.log_target`` plus the two chart Jacobians. The final sum
    is absent when there is no fixed block.

    Args:
        problem: Full-tiling observation model and normalized priors.
        initial_state: Valid state supplying fixed ``K``, topology, and
            computational initial values.

    Returns:
        PyMC model containing mutable same-shaped ``dynamic_design`` and
        ``dirichlet_alpha`` data, the mutable exact Dirichlet log normalizer,
        flat ``x``/``y`` variables, a one-state ``topology_token``, scientific
        deterministics, and normalized target potential.

    Raises:
        TypeError: If arguments have invalid public types.
        ValueError: If the state belongs to another problem or lacks finite
            target support.
        ImportError: If PyMC or PyTensor is unavailable.
        RuntimeError: If PyTensor is not configured for float64.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(initial_state, FullTilingPosteriorState):
        raise TypeError("initial_state must be a FullTilingPosteriorState.")
    if initial_state.problem is not problem:
        raise ValueError("initial_state must belong to the exact supplied problem.")
    if not math.isfinite(initial_state.log_target):
        raise ValueError("initial_state must have finite target support.")

    _require_float64()
    pm, pt = _import_runtime()
    dynamic_design, alpha, dirichlet_log_normalizer = _topology_arrays(
        problem,
        initial_state,
    )
    fixed_block = problem.base.fixed_block
    n_fixed = initial_state.fixed_coefficients.size
    fixed_mu, fixed_sigma = _fixed_log_parameters(problem)
    fixed_offset = problem.base.fixed_offset
    if fixed_offset is None:
        raise RuntimeError("validated full-tiling problem has no fixed offset.")

    with pm.Model() as model:
        dynamic_data = pm.Data(
            "dynamic_design",
            dynamic_design,
        )
        alpha_data = pm.Data(
            "dirichlet_alpha",
            alpha,
        )
        dirichlet_log_normalizer_data = pm.Data(
            "dirichlet_log_normalizer",
            np.float64(dirichlet_log_normalizer),
        )
        x = pm.Flat(
            "x",
            shape=initial_state.k,
            initval=np.log(initial_state.leaf_masses),
            dtype="float64",
        )
        if n_fixed:
            if fixed_block is None:
                raise RuntimeError("fixed coefficients require a fixed design block.")
            y = pm.Flat(
                "y",
                shape=n_fixed,
                initval=np.log(initial_state.fixed_coefficients),
                dtype="float64",
            )
            coefficient = pt.exp(y)
            fixed_prediction = pt.dot(
                pt.as_tensor_variable(fixed_block.design),
                coefficient,
            )
        else:
            y = pt.zeros((0,), dtype="float64")
            coefficient = y
            fixed_prediction = pt.zeros_like(
                pt.as_tensor_variable(fixed_offset),
                dtype="float64",
            )
        # This RV gives the structural BlockedStep a harmless discrete owner.
        pm.Categorical(
            "topology_token",
            p=np.asarray([1.0], dtype=np.float64),
            initval=0,
        )

        log_total = pt.logsumexp(x)
        root_total = pt.exp(log_total)
        log_share = x - log_total
        leaf_mass = pt.exp(x)
        mean_observation = (
            pt.as_tensor_variable(fixed_offset) + pt.dot(dynamic_data, leaf_mass) + fixed_prediction
        )
        residual = (mean_observation - pt.as_tensor_variable(problem.observations)) / (
            pt.as_tensor_variable(problem.observation_sd)
        )

        prior = problem.base.prior
        log_gamma = (
            np.float64(prior.root_shape * math.log(prior.root_rate) - math.lgamma(prior.root_shape))
            + np.float64(prior.root_shape - 1.0) * log_total
            - np.float64(prior.root_rate) * root_total
        )
        log_dirichlet = dirichlet_log_normalizer_data + pt.dot(alpha_data - 1.0, log_share)
        n_observations = problem.observations.size
        log_gaussian = (
            -0.5 * pt.dot(residual, residual)
            - np.float64(np.log(problem.observation_sd).sum())
            - np.float64(0.5 * n_observations * math.log(2.0 * math.pi))
        )
        likelihood_power = problem.base.likelihood_power
        log_likelihood = (
            np.float64(0.0) if likelihood_power == 0.0 else np.float64(likelihood_power) * log_gaussian
        )
        if n_fixed:
            # This is normalized LogNormal(c | mu, sigma) plus dc/dy.
            log_fixed = pt.sum(
                -0.5 * ((y - fixed_mu) / fixed_sigma) ** 2
                - np.log(fixed_sigma)
                - np.float64(0.5 * math.log(2.0 * math.pi))
            )
        else:
            log_fixed = np.float64(0.0)
        log_x_jacobian = pt.sum(x) - np.float64(initial_state.k - 1) * log_total
        largest_float = np.float64(np.finfo(np.float64).max)
        representable = pt.all((leaf_mass > 0.0) & (leaf_mass <= largest_float))
        representable = representable & (root_total <= largest_float)
        if n_fixed:
            representable = representable & pt.all((coefficient > 0.0) & (coefficient <= largest_float))
        scientific_log_target = log_likelihood + log_gamma + log_dirichlet + log_fixed + log_x_jacobian
        pm.Potential(
            "scientific_target",
            pt.switch(
                representable,
                scientific_log_target,
                np.float64(-math.inf),
            ),
        )
        pm.Deterministic("root_total", root_total)
        pm.Deterministic("leaf_share", pt.exp(log_share))
        pm.Deterministic("leaf_mass", leaf_mass)
        pm.Deterministic("fixed_coefficient", coefficient)
        pm.Deterministic("mean_observation", mean_observation)
    return model


def _set_topology_data_atomically(
    model: Any,
    dynamic_design: FloatArray,
    alpha: FloatArray,
    dirichlet_log_normalizer: float,
) -> None:
    """Replace all topology-dependent PyMC data with rollback on failure.

    Args:
        model: PyMC model containing the ``dynamic_design``,
            ``dirichlet_alpha``, and ``dirichlet_log_normalizer`` mutable-data
            containers.
        dynamic_design: New observation-by-leaf design matrix. Its shape must
            exactly match the existing container.
        alpha: New length-``K`` Dirichlet parameter vector. Its shape must
            exactly match the existing container.
        dirichlet_log_normalizer: Scalar log normalizer associated with
            ``alpha``.

    Raises:
        KeyError: If a required mutable-data container is absent.
        ValueError: If either array update would change its container shape.
        Exception: Propagates an assignment or rollback failure.

    Notes:
        Assignments occur sequentially. On failure, the helper attempts to
        restore all saved values before re-raising; a rollback failure
        propagates and may leave partial state.
    """
    dynamic_data = model["dynamic_design"]
    alpha_data = model["dirichlet_alpha"]
    normalizer_data = model["dirichlet_log_normalizer"]
    old_dynamic = np.array(dynamic_data.get_value(borrow=False), copy=True)
    old_alpha = np.array(alpha_data.get_value(borrow=False), copy=True)
    old_normalizer = np.array(
        normalizer_data.get_value(borrow=False),
        copy=True,
    )
    if dynamic_design.shape != old_dynamic.shape or alpha.shape != old_alpha.shape:
        raise ValueError("topology updates must preserve dynamic-design and alpha shapes.")
    try:
        dynamic_data.set_value(dynamic_design, borrow=False)
        alpha_data.set_value(alpha, borrow=False)
        normalizer_data.set_value(
            np.asarray(dirichlet_log_normalizer, dtype=np.float64),
            borrow=False,
        )
    except Exception:
        dynamic_data.set_value(old_dynamic, borrow=False)
        alpha_data.set_value(old_alpha, borrow=False)
        normalizer_data.set_value(old_normalizer, borrow=False)
        raise


def _build_topology_hmc_objects(
    hmc: Any,
    precision: FloatArray,
) -> tuple[Any, Any]:
    """Build a dense inverse potential and matching CPU integrator.

    The fixed local generator only satisfies the new potential's construction
    contract. It is never derived from, and cannot advance, the sampler's
    authoritative PCG64 stream. ``BaseHMC.set_rng`` replaces it after atomic
    installation and before momentum generation.

    Args:
        hmc: Existing PyMC HMC step supplying the compiled logp/gradient
            function.
        precision: Validated topology-reference precision.

    Returns:
        A new ``QuadPotentialFullInv`` and a new
        ``CpuLeapfrogIntegrator`` that references it.

    Raises:
        ImportError: If the required PyMC HMC internals are unavailable.
    """
    quadpotential = import_module("pymc.step_methods.hmc.quadpotential")
    integration = import_module("pymc.step_methods.hmc.integration")
    potential = quadpotential.QuadPotentialFullInv(
        np.array(precision, dtype=np.float64, copy=True),
        rng=np.random.Generator(np.random.PCG64(0)),
    )
    integrator = integration.CpuLeapfrogIntegrator(
        potential,
        hmc._logp_dlogp_func,
    )
    return potential, integrator


def _install_topology_kernel_atomically(
    model: Any,
    hmc: Any,
    *,
    dynamic_design: FloatArray,
    alpha: FloatArray,
    dirichlet_log_normalizer: float,
    precision: FloatArray,
) -> tuple[Any, Any]:
    """Install matched topology data, inverse potential, and integrator.

    All replacement objects are constructed before the first mutation. If a
    data or sampler-reference assignment fails, the complete old topology
    payload, potential, and integrator are restored before the original error
    is re-raised.

    Args:
        model: Dynamic-topology PyMC model.
        hmc: Live non-adapting PyMC HMC step.
        dynamic_design: Selected topology's observation-by-leaf design.
        alpha: Selected topology's aligned Dirichlet shapes.
        dirichlet_log_normalizer: Exact normalized Dirichlet constant.
        precision: Selected topology's validated reference precision.

    Returns:
        Newly installed potential and integrator.

    Raises:
        Exception: Propagates construction or installation failures. A
            ``RuntimeError`` is raised instead if restoring the old kernel
            also fails.
    """
    new_potential, new_integrator = _build_topology_hmc_objects(
        hmc,
        precision,
    )
    dynamic_data = model["dynamic_design"]
    alpha_data = model["dirichlet_alpha"]
    normalizer_data = model["dirichlet_log_normalizer"]
    old_dynamic = np.array(dynamic_data.get_value(borrow=False), copy=True)
    old_alpha = np.array(alpha_data.get_value(borrow=False), copy=True)
    old_normalizer = np.array(
        normalizer_data.get_value(borrow=False),
        copy=True,
    )
    old_potential = hmc.potential
    old_integrator = hmc.integrator
    try:
        _set_topology_data_atomically(
            model,
            dynamic_design,
            alpha,
            dirichlet_log_normalizer,
        )
        hmc.potential = new_potential
        hmc.integrator = new_integrator
    except Exception:
        rollback_errors: list[Exception] = []
        try:
            _set_topology_data_atomically(
                model,
                old_dynamic,
                old_alpha,
                float(old_normalizer.item()),
            )
        except Exception as error:
            rollback_errors.append(error)
        try:
            hmc.potential = old_potential
        except Exception as error:
            rollback_errors.append(error)
        try:
            hmc.integrator = old_integrator
        except Exception as error:
            rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError("topology-kernel installation and rollback both failed.") from rollback_errors[
                0
            ]
        raise
    return new_potential, new_integrator


def _rebuild_structural_transition_at_hmc_boundary(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    transition: PosteriorTransitionTerms,
    *,
    current_log_leaf_mass: ArrayLike,
    current_log_fixed_coefficient: ArrayLike,
) -> tuple[_LogMassInvolutionTransitionTerms, FloatArray]:
    """Apply one exact structural involution to authoritative log masses.

    An edge flip maps the two removed friend-child coordinates, in canonical
    child order, to the two new perpendicular children. A resolution
    relocation maps the old destination-leaf coordinate to the merged parent
    and maps the two old merge-child coordinates to the two new destination
    children. Every unchanged leaf retains its exact coordinate bits. These
    maps are their own reverse when used with the transition's unique reverse
    merge and split choices.

    The scientific candidate is rebuilt from ``exp(candidate_x)``. Structural
    MH is scored directly in the normalized PyMC log-mass chart,
    ``log_target + sum(x) - (K - 1) * logsumexp(x)``. The fixed-coordinate
    Jacobian cancels because structure does not alter ``y``. Only the
    reverse-minus-forward discrete selection probability is added. The
    target difference is retained explicitly alongside its scientific and
    chart components. The proposal permutation itself has unit Jacobian:
    there is no Beta auxiliary density and no physical mass-map Jacobian.

    Args:
        problem: Scientific problem shared by source and candidate.
        source: Current scientific state exactly decoded from the authoritative
            HMC coordinates.
        transition: Valid edge-flip or resolution-relocation geometry and
            discrete-selection proposal. Its raw candidate masses are ignored.
        current_log_leaf_mass: Authoritative source log masses in canonical
            source-leaf order.
        current_log_fixed_coefficient: Authoritative fixed-coefficient logs.

    Returns:
        A rebuilt transition and the exact candidate log-mass vector used to
        rebuild its candidate.

    Raises:
        TypeError: If ``transition`` has the wrong type.
        ValueError: If the transition is invalid, non-structural, belongs to a
            different problem, or authoritative arrays have the wrong shape.
        RuntimeError: If source coordinates do not exactly decode ``source``,
            the involution metadata do not cover the candidate topology
            exactly, or the rebuilt reduced ratio is NaN.
    """
    if not isinstance(transition, PosteriorTransitionTerms):
        raise TypeError("transition must be a PosteriorTransitionTerms.")
    if not transition.valid:
        raise ValueError("only valid structural transitions can be rebuilt at the HMC boundary.")
    if transition.move not in ("edge_flip", "resolution_relocation"):
        raise ValueError("only structural transitions can be rebuilt at the HMC boundary.")
    geometry_candidate = transition.candidate
    if source.problem is not problem or geometry_candidate.problem is not problem:
        raise ValueError("source and transition candidate must belong to problem.")

    source_x = np.asarray(current_log_leaf_mass, dtype=np.float64)
    if source_x.shape != (source.k,) or np.any(~np.isfinite(source_x)):
        raise ValueError("current_log_leaf_mass must contain one finite value per source leaf.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        decoded_source_masses = np.exp(source_x)
    if not np.array_equal(decoded_source_masses, source.leaf_masses):
        raise RuntimeError("authoritative source log masses do not exactly decode the source state.")

    source_x_by_leaf = {
        leaf: float(coordinate)
        for leaf, coordinate in zip(
            source.tiling_state.tiling.leaves,
            source_x,
            strict=True,
        )
    }
    reverse_merge = transition.reverse_merge_choice
    reverse_split = transition.reverse_split_choice
    if reverse_merge is None or reverse_split is None:
        raise RuntimeError("a valid structural transition lacks its reverse choices.")

    if transition.move == "edge_flip":
        source_children = reverse_split.leaf.midpoint_children(reverse_split.axis)
        candidate_children = reverse_merge.children
        removed = frozenset(source_children)
        coordinate_map = {
            candidate_leaf: source_x_by_leaf[source_leaf]
            for source_leaf, candidate_leaf in zip(
                source_children,
                candidate_children,
                strict=True,
            )
        }
    else:
        source_children = reverse_split.leaf.midpoint_children(reverse_split.axis)
        source_destination = reverse_merge.parent
        candidate_parent = reverse_split.leaf
        candidate_children = reverse_merge.children
        removed = frozenset((*source_children, source_destination))
        coordinate_map = {
            candidate_parent: source_x_by_leaf[source_destination],
            candidate_children[0]: source_x_by_leaf[source_children[0]],
            candidate_children[1]: source_x_by_leaf[source_children[1]],
        }

    candidate_x = np.empty(geometry_candidate.k, dtype=np.float64)
    for position, leaf in enumerate(geometry_candidate.tiling_state.tiling.leaves):
        mapped_coordinate = coordinate_map.get(leaf)
        if mapped_coordinate is not None:
            candidate_x[position] = mapped_coordinate
            continue
        if leaf in removed or leaf not in source_x_by_leaf:
            raise RuntimeError("the log-mass involution does not cover the candidate topology.")
        candidate_x[position] = source_x_by_leaf[leaf]

    fixed_y = np.asarray(current_log_fixed_coefficient, dtype=np.float64)
    if fixed_y.shape != source.fixed_coefficients.shape or np.any(~np.isfinite(fixed_y)):
        raise ValueError("current_log_fixed_coefficient must match the finite source fixed block.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        candidate_masses = np.exp(candidate_x)
        fixed_coefficients = np.exp(fixed_y)
    if not np.array_equal(fixed_coefficients, source.fixed_coefficients):
        raise RuntimeError("authoritative fixed logs do not exactly decode the source state.")
    if not np.array_equal(geometry_candidate.fixed_coefficients, source.fixed_coefficients):
        raise RuntimeError("a structural proposal unexpectedly changed fixed coefficients.")

    rebuilt_candidate = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            geometry_candidate.tiling_state.tiling,
            candidate_masses,
        ),
        fixed_coefficients=fixed_coefficients,
    )
    source_chart_log_jacobian = _log_mass_chart_jacobian(source_x)
    candidate_chart_log_jacobian = _log_mass_chart_jacobian(candidate_x)
    exact_scientific_delta = rebuilt_candidate.log_target - source.log_target
    exact_transformed_delta = (rebuilt_candidate.log_target + candidate_chart_log_jacobian) - (
        source.log_target + source_chart_log_jacobian
    )
    rebuilt_transition = _LogMassInvolutionTransitionTerms(
        candidate=rebuilt_candidate,
        move=transition.move,
        delta_log_likelihood=rebuilt_candidate.log_likelihood - source.log_likelihood,
        delta_log_root_prior=rebuilt_candidate.log_root_prior - source.log_root_prior,
        delta_log_allocation_prior=(rebuilt_candidate.log_allocation_prior - source.log_allocation_prior),
        delta_log_fixed_coefficient_prior=(
            rebuilt_candidate.log_fixed_coefficient_prior - source.log_fixed_coefficient_prior
        ),
        log_q_forward_selection=transition.log_q_forward_selection,
        log_q_forward_auxiliary=0.0,
        log_q_reverse_selection=transition.log_q_reverse_selection,
        log_q_reverse_auxiliary=0.0,
        log_jacobian=0.0,
        reverse_merge_choice=transition.reverse_merge_choice,
        reverse_split_choice=transition.reverse_split_choice,
        exact_scientific_log_target_delta=exact_scientific_delta,
        log_mass_chart_delta=(candidate_chart_log_jacobian - source_chart_log_jacobian),
        exact_transformed_log_target_delta=exact_transformed_delta,
    )
    candidate_x.setflags(write=False)
    return rebuilt_transition, candidate_x


def _log_mass_chart_jacobian(log_leaf_mass: ArrayLike) -> float:
    """Return the normalized log-mass chart log-Jacobian.

    Args:
        log_leaf_mass: Non-empty finite authoritative log masses.

    Returns:
        ``sum(x) - (K - 1) * logsumexp(x)``.

    Raises:
        ValueError: If the coordinate vector is empty, not one-dimensional,
            or contains non-finite values.
    """
    coordinates = np.asarray(log_leaf_mass, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 1 or np.any(~np.isfinite(coordinates)):
        raise ValueError("log_leaf_mass must be a non-empty finite vector.")
    log_total = float(np.logaddexp.reduce(coordinates))
    return float(np.sum(coordinates) - (coordinates.size - 1) * log_total)


def _transformed_log_mass_target(
    state: FullTilingPosteriorState,
    log_leaf_mass: ArrayLike,
) -> float:
    """Return the scientific target density in the normalized log-mass chart.

    Args:
        state: Scientific state decoded from ``log_leaf_mass``.
        log_leaf_mass: One finite authoritative coordinate per canonical leaf.

    Returns:
        Scientific log target plus the normalized log-mass chart Jacobian.

    Raises:
        ValueError: If the coordinate vector has wrong shape or non-finite
            values.
        RuntimeError: If exponentiation does not exactly reproduce the state.
    """
    coordinates = np.asarray(log_leaf_mass, dtype=np.float64)
    if coordinates.shape != (state.k,) or np.any(~np.isfinite(coordinates)):
        raise ValueError("log_leaf_mass must contain one finite value per state leaf.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        decoded = np.exp(coordinates)
    if not np.array_equal(decoded, state.leaf_masses):
        raise RuntimeError("log_leaf_mass does not exactly decode the scientific state.")
    return float(state.log_target + _log_mass_chart_jacobian(coordinates))


def _draw_log_involution_structural_transition(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    current_log_leaf_mass: ArrayLike,
    current_log_fixed_coefficient: ArrayLike,
    rng: np.random.Generator,
) -> tuple[PosteriorTransitionTerms, FloatArray | None]:
    """Draw the fixed-catalogue structural move and apply its log involution.

    Component choice remains an unconditional half-and-half draw. A source
    merge is uniform, and relocation selects uniformly from the fixed
    ``leaf × axis`` intermediate catalogue. Invalid choices remain explicit
    self-attempts. The placeholder physical fraction is used only to construct
    and validate geometry; no Beta random variate is drawn and all continuous
    proposal terms are discarded by the log-coordinate rebuild.

    Args:
        problem: Fixed-``K`` posterior target.
        source: Current scientific state.
        current_log_leaf_mass: Authoritative source log masses.
        current_log_fixed_coefficient: Authoritative fixed logs.
        rng: Sole compound-chain generator.

    Returns:
        Transition terms and candidate logs for a valid proposal, or the
        invalid transition and ``None``.
    """
    choose_edge_flip = float(rng.random()) < 0.5
    merges = merge_choices(source.tiling_state.tiling)
    move = "edge_flip" if choose_edge_flip else "resolution_relocation"
    if not merges:
        return (
            PosteriorTransitionTerms(
                candidate=source,
                move=move,
                delta_log_likelihood=0.0,
                valid=False,
                reason="selected merge is unavailable",
            ),
            None,
        )

    merge = merges[int(rng.integers(len(merges)))]
    if choose_edge_flip:
        geometry = propose_posterior_edge_flip(
            problem,
            source,
            merge_choice=merge,
            new_fraction=0.5,
        )
    else:
        intermediate = source.tiling_state.tiling.merge(merge)
        axes: tuple[Axis, Axis] = ("horizontal", "vertical")
        catalogue = tuple(SplitChoice(leaf, axis) for leaf in intermediate.leaves for axis in axes)
        expected_size = 2 * (source.k - 1)
        if len(catalogue) != expected_size:
            raise RuntimeError("relocation catalogue does not have fixed size 2 * (K - 1).")
        split = catalogue[int(rng.integers(len(catalogue)))]
        geometry = propose_posterior_resolution_relocation(
            problem,
            source,
            merge_choice=merge,
            split_choice=split,
            new_fraction=0.5,
        )
        if (
            not geometry.valid
            and geometry.reason == "proposed child masses are outside representable support"
        ):
            # The smallest positive binary64 destination mass cannot itself
            # be halved into two positive placeholder masses. Geometry in the
            # log involution does not conserve or split that physical mass, so
            # retry only the geometry oracle at the smallest splittable mass.
            # The final candidate is still rebuilt exclusively from source x.
            placeholder_masses = np.array(source.leaf_masses, copy=True)
            destination_position = source.tiling_state.tiling.leaves.index(split.leaf)
            smallest_positive = np.nextafter(np.float64(0.0), np.float64(1.0))
            placeholder_masses[destination_position] = 2.0 * smallest_positive
            placeholder_source = build_full_tiling_posterior_state(
                problem,
                allocation=TilingState(
                    source.tiling_state.tiling,
                    placeholder_masses,
                ),
                fixed_coefficients=source.fixed_coefficients,
            )
            geometry = propose_posterior_resolution_relocation(
                problem,
                placeholder_source,
                merge_choice=merge,
                split_choice=split,
                new_fraction=0.5,
            )
            if not geometry.valid:
                raise RuntimeError("representable placeholder relocation did not recover valid geometry.")
    if not geometry.valid:
        return geometry, None
    return _rebuild_structural_transition_at_hmc_boundary(
        problem,
        source,
        geometry,
        current_log_leaf_mass=current_log_leaf_mass,
        current_log_fixed_coefficient=current_log_fixed_coefficient,
    )


def _build_compound_kernel(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
    settings: FullTilingPyMCHMCKernelSettings,
    rng: np.random.Generator,
    *,
    log_leaf_mass: ArrayLike | None = None,
    log_fixed_coefficient: ArrayLike | None = None,
) -> tuple[Any, Any, Any, dict[str, np.ndarray]]:
    """Build a stopped structural-then-HMC kernel and exact initial point.

    Args:
        problem: Fixed-``K`` scientific posterior problem.
        state: Scientific initial state belonging to ``problem``.
        settings: Resolved non-adapting HMC controls and metric identities.
        rng: Authoritative compound-chain generator. The returned structural
            step closes over this object and advances it for proposals,
            acceptance uniforms, and per-sweep HMC seeds.
        log_leaf_mass: Optional authoritative length-``K`` log-mass
            coordinates. When provided, exponentiation must reproduce
            ``state.leaf_masses`` exactly, enabling bitwise checkpoint replay.
        log_fixed_coefficient: Optional authoritative log coordinates with
            the same shape as ``state.fixed_coefficients``. It must be empty
            when the problem has no fixed block and must exactly encode the
            fixed coefficients otherwise.

    Returns:
        A tuple containing the PyMC model, stopped compound step, structural
        substep, and copied initial point mapping. The point contains ``x`` of
        shape ``(K,)`` and, when present, ``y`` of shape ``(n_fixed,)``.

    Raises:
        ImportError: If the optional PyMC/PyTensor runtime is unavailable.
        KeyError: If the generated model does not expose an expected variable.
        ValueError: If authoritative coordinates have incompatible shapes,
            contain non-finite values, or do not exactly encode ``state``.
        RuntimeError: If the constructed compound kernel is not fully
            non-adapting.

    Notes:
        Construction itself does not advance ``rng``. Each later structural
        step advances it and resets the HMC substep to a freshly drawn PCG64
        seed, preserving single-stream replay semantics.
    """
    pm, _ = _import_runtime()
    model = build_full_tiling_pymc_hmc_model(problem, state)
    point = model.initial_point()
    if log_leaf_mass is not None:
        x = np.asarray(log_leaf_mass, dtype=np.float64)
        if x.shape != (settings.fixed_k,) or np.any(~np.isfinite(x)):
            raise ValueError("log_leaf_mass must contain one finite value per leaf.")
        if not np.array_equal(np.exp(x), state.leaf_masses):
            raise ValueError("log_leaf_mass must exactly encode the initial state.")
        point["x"] = np.array(x, copy=True)
    if state.fixed_coefficients.size:
        if log_fixed_coefficient is not None:
            y = np.asarray(log_fixed_coefficient, dtype=np.float64)
            if y.shape != state.fixed_coefficients.shape or np.any(~np.isfinite(y)):
                raise ValueError("log_fixed_coefficient must match the finite fixed block.")
            if not np.array_equal(np.exp(y), state.fixed_coefficients):
                raise ValueError("log_fixed_coefficient must exactly encode the initial state.")
            point["y"] = np.array(y, copy=True)
    elif log_fixed_coefficient is not None:
        y = np.asarray(log_fixed_coefficient, dtype=np.float64)
        if y.shape != (0,):
            raise ValueError("log_fixed_coefficient must be empty without a fixed block.")
    continuous_rvs = [model["x"]]
    if state.fixed_coefficients.size:
        continuous_rvs.append(model["y"])
    dimension = settings.fixed_k + state.fixed_coefficients.size
    topology_precision = build_full_tiling_pymc_hmc_topology_precision(
        problem,
        state,
    )
    # BaseHMC divides step_scale by dimension**0.25. This inverse supplies
    # the caller's actual requested step size to the integrator.
    step_scale = settings.step_size * dimension**0.25
    hmc = pm.HamiltonianMC(
        vars=continuous_rvs,
        model=model,
        scaling=np.array(topology_precision, copy=True),
        is_cov=False,
        step_scale=step_scale,
        path_length=settings.step_size * settings.leapfrog_steps,
        max_steps=settings.leapfrog_steps,
        adapt_step_size=False,
        step_rand=None,
        rng=np.random.Generator(np.random.PCG64(0)),
        initial_point=point,
    )
    hmc.stop_tuning()
    # DualAverageAdaptation represents even a non-adapting step through an
    # exp(log(epsilon)) round trip, which can move the requested value by one
    # binary64 ULP. Derive the path from that actual value so integer
    # truncation inside PyMC always gives the requested leapfrog count.
    effective_step_size = float(hmc.step_adapt.current(False))
    hmc.path_length = float(
        np.nextafter(
            effective_step_size * settings.leapfrog_steps,
            math.inf,
        )
    )

    class _StructuralTilingStep(pm.BlockedStep):
        """Custom one-token step that runs structural MH then reseeds HMC."""

        name = "full_tiling_structure"
        default_blocked = True
        stats_dtypes_shapes = {
            "move": (object, []),
            "valid": (bool, []),
            "accepted": (bool, []),
            "log_acceptance_ratio": (np.float64, []),
            "invalid_reason": (object, []),
            "hmc_seed": (np.uint64, []),
        }

        def __init__(
            self,
            vars: list[Any],
            *,
            model: Any,
            blocked: bool = True,
            **_: Any,
        ) -> None:
            """Initialize the token owner around closed-over kernel state."""
            self.vars = vars
            self.model = model
            self.blocked = blocked
            self.state = state
            self.last_transition: Any = None
            self.topology_precision = topology_precision
            self.hmc_start_log_leaf_mass: FloatArray | None = None
            self.hmc_start_log_fixed_coefficient: FloatArray | None = None

        def step(
            self,
            current_point: dict[str, np.ndarray],
        ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
            """Apply structural MH, update data/masses, and seed the next HMC."""
            source = self.state
            transition, candidate_x = _draw_log_involution_structural_transition(
                problem,
                source,
                current_log_leaf_mass=current_point["x"],
                current_log_fixed_coefficient=(
                    current_point["y"] if source.fixed_coefficients.size else np.empty(0, dtype=np.float64)
                ),
                rng=rng,
            )
            uniform = float(rng.random())
            log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
            next_state = accept_or_reject(
                source,
                transition,
                log_uniform=log_uniform,
            )
            accepted = transition.valid and next_state is transition.candidate
            (
                dynamic_design,
                alpha,
                dirichlet_log_normalizer,
            ) = _topology_arrays(problem, next_state)
            next_precision = build_full_tiling_pymc_hmc_topology_precision(
                problem,
                next_state,
            )
            _install_topology_kernel_atomically(
                model,
                hmc,
                dynamic_design=dynamic_design,
                alpha=alpha,
                dirichlet_log_normalizer=dirichlet_log_normalizer,
                precision=next_precision,
            )
            next_point = dict(current_point)
            if accepted:
                if candidate_x is None:
                    raise RuntimeError("an accepted structural proposal has no authoritative candidate logs.")
                next_point["x"] = np.array(candidate_x, copy=True)
            self.hmc_start_log_leaf_mass = np.array(
                next_point["x"],
                dtype=np.float64,
                copy=True,
            )
            self.hmc_start_log_fixed_coefficient = (
                np.array(
                    next_point["y"],
                    dtype=np.float64,
                    copy=True,
                )
                if next_state.fixed_coefficients.size
                else np.empty(0, dtype=np.float64)
            )
            self.state = next_state
            self.last_transition = transition
            self.topology_precision = next_precision
            hmc_seed = int(
                rng.integers(
                    np.iinfo(np.uint64).max,
                    dtype=np.uint64,
                )
            )
            hmc.set_rng(np.random.Generator(np.random.PCG64(hmc_seed)))
            return next_point, [
                {
                    "move": transition.move,
                    "valid": transition.valid,
                    "accepted": accepted,
                    "log_acceptance_ratio": transition.log_acceptance_ratio,
                    "invalid_reason": ("" if transition.reason is None else transition.reason),
                    "hmc_seed": np.uint64(hmc_seed),
                }
            ]

        @staticmethod
        def competence(var: Any, has_grad: bool) -> Any:
            """Declare compatibility only for the deliberately discrete token."""
            del var, has_grad
            return pm.step_methods.compound.Competence.COMPATIBLE

    structural = _StructuralTilingStep(
        vars=[model["topology_token"]],
        model=model,
    )
    compound = pm.CompoundStep([structural, hmc])
    compound.stop_tuning()
    if compound.tune:
        # PyMC 5.25 leaves CompoundStep.tune as an informational attribute;
        # substeps are authoritative, but keeping it false avoids ambiguity.
        compound.tune = False
    if hmc.tune or hmc.adapt_step_size or hmc._step_rand is not None:
        raise RuntimeError("compound HMC must be fully non-adapting before sampling.")
    return model, compound, structural, point


def _state_from_point(
    problem: FullTilingProblem,
    structural_state: FullTilingPosteriorState,
    point: dict[str, np.ndarray],
) -> FullTilingPosteriorState:
    """Decode a PyMC endpoint through the complete scientific-state oracle.

    Args:
        problem: Scientific problem used to evaluate the decoded endpoint.
        structural_state: Post-structural state supplying the authoritative
            current tiling and fixed-block shape.
        point: PyMC point containing length-``K`` log masses under ``x`` and,
            when a fixed block exists, log fixed coefficients under ``y``.

    Returns:
        A newly rebuilt posterior state on ``structural_state``'s tiling, with
        all predictions, residuals, and target components recomputed by the
        scientific oracle.

    Raises:
        KeyError: If ``point`` omits a required coordinate.
        TypeError: If decoded arrays cannot construct a scientific state.
        ValueError: If decoded masses or coefficients have invalid shape,
            support, or non-finite target terms.

    Notes:
        Exponentiation deliberately permits NumPy overflow or underflow; the
        scientific-state constructor is the authoritative support check.
    """
    x = np.asarray(point["x"], dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        masses = np.exp(x)
    if structural_state.fixed_coefficients.size:
        y = np.asarray(point["y"], dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            fixed = np.exp(y)
    else:
        fixed = np.empty(0, dtype=np.float64)
    allocation = TilingState(
        structural_state.tiling_state.tiling,
        masses,
    )
    return build_full_tiling_posterior_state(
        problem,
        allocation=allocation,
        fixed_coefficients=fixed,
    )


def canonicalize_full_tiling_pymc_hmc_fresh_state(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> tuple[FullTilingPosteriorState, FloatArray, FloatArray]:
    """Canonicalize a fresh scientific state in the PyMC log chart.

    Fresh initializers construct scientific masses and fixed coefficients
    directly, but the HMC kernel's authoritative boundary coordinates are
    their elementwise binary64 natural logarithms. This function repeatedly
    maps those positive values through ``log -> exp``, performs a complete
    scientific-oracle rebuild, and stops only at an exact log/exp fixed point.
    The returned arrays therefore decode bit-for-bit to the returned state,
    and applying this function again preserves the state and coordinate bits.
    Neither input object is mutated.

    This public entry point supports fresh initializer states only.
    :func:`continue_full_tiling_pymc_hmc` bypasses it because a durable
    checkpoint's stored scientific state and log coordinates are already
    authoritative replay inputs and must not be rebuilt.

    Args:
        problem: Exact full-tiling posterior problem that owns ``state``.
        state: Supported fresh scientific state to canonicalize.

    Returns:
        A tuple containing the fully rebuilt canonical scientific state,
        read-only authoritative log leaf masses with shape ``(K,)``, and
        read-only authoritative log fixed coefficients with shape
        ``(n_fixed,)``.

    Raises:
        TypeError: If either argument has an incompatible public type.
        ValueError: If ``state`` does not belong to ``problem`` or a decoded
            state violates scientific support.
        RuntimeError: If the binary64 log/exp mapping does not reach an exact
            fixed point within the bounded canonicalization audit.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(state, FullTilingPosteriorState):
        raise TypeError("state must be a FullTilingPosteriorState.")
    if state.problem is not problem:
        raise ValueError("state must belong to the exact supplied problem.")

    current = state
    for _ in range(8):
        log_leaf_mass = np.log(current.leaf_masses)
        log_fixed_coefficient = np.log(current.fixed_coefficients)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            leaf_masses = np.exp(log_leaf_mass)
            fixed_coefficients = np.exp(log_fixed_coefficient)
        canonical = build_full_tiling_posterior_state(
            problem,
            allocation=TilingState(
                current.tiling_state.tiling,
                leaf_masses,
            ),
            fixed_coefficients=fixed_coefficients,
        )
        if np.array_equal(np.log(canonical.leaf_masses), log_leaf_mass) and np.array_equal(
            np.log(canonical.fixed_coefficients),
            log_fixed_coefficient,
        ):
            authoritative_leaf = _readonly_array(
                log_leaf_mass,
                dtype=np.float64,
                ndim=1,
                name="log_leaf_mass",
            )
            authoritative_fixed = _readonly_array(
                log_fixed_coefficient,
                dtype=np.float64,
                ndim=1,
                name="log_fixed_coefficient",
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                decoded_leaf = np.exp(authoritative_leaf)
                decoded_fixed = np.exp(authoritative_fixed)
            if not np.array_equal(decoded_leaf, canonical.leaf_masses) or not np.array_equal(
                decoded_fixed,
                canonical.fixed_coefficients,
            ):
                raise RuntimeError("fresh-state canonicalization produced non-authoritative log coordinates.")
            return canonical, authoritative_leaf, authoritative_fixed
        current = canonical
    raise RuntimeError("fresh-state canonicalization did not reach an exact binary64 log/exp fixed point.")


def _run_segment(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
    *,
    iterations: int,
    sweeps_completed: int,
    settings: FullTilingPyMCHMCKernelSettings,
    rng: np.random.Generator,
    log_leaf_mass: ArrayLike | None = None,
    log_fixed_coefficient: ArrayLike | None = None,
) -> FullTilingPyMCHMCSamplingResult:
    """Run one exact, boundary-inclusive compound-HMC segment.

    Args:
        problem: Fixed-``K`` posterior problem that owns ``initial_state``.
        initial_state: Scientific state at the segment boundary.
        iterations: Positive number of complete structural-then-HMC sweeps.
        sweeps_completed: Non-negative global sweep offset preceding this
            segment.
        settings: Frozen kernel controls and metric identities compatible with
            ``initial_state``.
        rng: Authoritative PCG64-backed generator. It is advanced in place for
            every structural proposal, decision, and HMC seed; its final state
            is stored in the returned checkpoint.
        log_leaf_mass: Optional authoritative boundary log-mass coordinates
            for exact restart replay.
        log_fixed_coefficient: Optional authoritative boundary log fixed
            coordinates for exact restart replay.

    Returns:
        Boundary-inclusive trace, final scientific state, exact continuation
        checkpoint, and separate kernel-setup and transition timings. State
        arrays have ``iterations + 1`` rows, while per-sweep diagnostics have
        ``iterations`` rows.

    Raises:
        ImportError: If the optional PyMC/PyTensor runtime is unavailable.
        TypeError: If a state, coordinate, or reconstructed endpoint has an
            incompatible type.
        ValueError: If problem identity, fixed ``K``, metric identity, target
            support, or authoritative restart coordinates are invalid.
        RuntimeError: If compound-step statistics are malformed, PyMC executes
            a different leapfrog count, or an endpoint cannot be rebuilt.

    Notes:
        The input state is not mutated. The supplied RNG is intentionally
        mutated, and replay requires restoring its exact checkpoint state
        together with both authoritative log-coordinate arrays.
    """
    if initial_state.problem is not problem:
        raise ValueError("initial_state must belong to the exact supplied problem.")
    if initial_state.k != settings.fixed_k:
        raise ValueError("initial_state K must match fixed kernel K.")
    if not math.isfinite(initial_state.log_target):
        raise ValueError("initial_state must have finite target support.")

    setup_start = time.perf_counter()
    _, compound, structural, point = _build_compound_kernel(
        problem,
        initial_state,
        settings,
        rng,
        log_leaf_mass=log_leaf_mass,
        log_fixed_coefficient=log_fixed_coefficient,
    )
    kernel_setup_seconds = time.perf_counter() - setup_start
    states = [initial_state]
    log_leaf_mass_coordinates = [np.array(point["x"], dtype=np.float64, copy=True)]
    log_fixed_coefficient_coordinates = [
        (
            np.array(point["y"], dtype=np.float64, copy=True)
            if initial_state.fixed_coefficients.size
            else np.empty(0, dtype=np.float64)
        )
    ]
    structural_move: list[str] = []
    structural_valid: list[bool] = []
    structural_accepted: list[bool] = []
    structural_log_ratio: list[float] = []
    structural_reason: list[str] = []
    hmc_start_log_leaf_mass: list[FloatArray] = []
    hmc_start_log_fixed_coefficient: list[FloatArray] = []
    hmc_accepted: list[bool] = []
    hmc_accept: list[float] = []
    hmc_diverging: list[bool] = []
    hmc_energy: list[float] = []
    hmc_energy_error: list[float] = []
    hmc_step_size: list[float] = []
    hmc_n_steps: list[int] = []
    hmc_seed: list[np.uint64] = []

    state = initial_state
    transition_start = time.perf_counter()
    for _ in range(iterations):
        point, stats = compound.step(point)
        if len(stats) != 2:
            raise RuntimeError("compound sweep must return structural and HMC stats.")
        structural_stats, hmc_stats = stats
        state = _state_from_point(problem, structural.state, point)
        structural.state = state
        states.append(state)
        log_leaf_mass_coordinates.append(np.array(point["x"], dtype=np.float64, copy=True))
        log_fixed_coefficient_coordinates.append(
            (
                np.array(point["y"], dtype=np.float64, copy=True)
                if state.fixed_coefficients.size
                else np.empty(0, dtype=np.float64)
            )
        )

        structural_move.append(str(structural_stats["move"]))
        structural_valid.append(bool(structural_stats["valid"]))
        structural_accepted.append(bool(structural_stats["accepted"]))
        structural_log_ratio.append(float(structural_stats["log_acceptance_ratio"]))
        structural_reason.append(str(structural_stats["invalid_reason"]))
        if structural.hmc_start_log_leaf_mass is None or structural.hmc_start_log_fixed_coefficient is None:
            raise RuntimeError("structural step did not retain the pre-HMC coordinates.")
        hmc_start_log_leaf_mass.append(
            np.array(structural.hmc_start_log_leaf_mass, dtype=np.float64, copy=True)
        )
        hmc_start_log_fixed_coefficient.append(
            np.array(
                structural.hmc_start_log_fixed_coefficient,
                dtype=np.float64,
                copy=True,
            )
        )
        hmc_accepted.append(bool(hmc_stats["accepted"]))
        hmc_accept.append(float(hmc_stats["accept"]))
        hmc_diverging.append(bool(hmc_stats["diverging"]))
        hmc_energy.append(float(hmc_stats["energy"]))
        hmc_energy_error.append(float(hmc_stats["energy_error"]))
        hmc_step_size.append(float(hmc_stats["step_size"]))
        completed_hmc_steps = int(hmc_stats["n_steps"])
        if completed_hmc_steps != settings.leapfrog_steps:
            raise RuntimeError("PyMC HMC did not execute the configured leapfrog count.")
        hmc_n_steps.append(completed_hmc_steps)
        hmc_seed.append(np.uint64(structural_stats["hmc_seed"]))
    transition_seconds = time.perf_counter() - transition_start

    state_sweep = np.arange(
        sweeps_completed,
        sweeps_completed + iterations + 1,
        dtype=np.int64,
    )
    global_sweep = np.arange(
        sweeps_completed + 1,
        sweeps_completed + iterations + 1,
        dtype=np.int64,
    )
    trace = FullTilingPyMCHMCTrace(
        state_sweep=state_sweep,
        rectangle_bounds=np.stack([_rectangle_bounds(item) for item in states]),
        leaf_masses=np.stack([item.leaf_masses for item in states]),
        fixed_coefficients=np.stack([item.fixed_coefficients for item in states]),
        log_leaf_mass=np.stack(log_leaf_mass_coordinates),
        log_fixed_coefficient=np.stack(log_fixed_coefficient_coordinates),
        log_target=np.asarray([item.log_target for item in states]),
        global_sweep=global_sweep,
        structural_move=np.asarray(structural_move, dtype="U24"),
        structural_valid=np.asarray(structural_valid, dtype=np.bool_),
        structural_accepted=np.asarray(structural_accepted, dtype=np.bool_),
        structural_log_acceptance_ratio=np.asarray(
            structural_log_ratio,
            dtype=np.float64,
        ),
        structural_invalid_reason=np.asarray(structural_reason, dtype="U96"),
        hmc_start_log_leaf_mass=np.stack(hmc_start_log_leaf_mass),
        hmc_start_log_fixed_coefficient=np.stack(
            hmc_start_log_fixed_coefficient,
        ),
        hmc_accepted=np.asarray(hmc_accepted, dtype=np.bool_),
        hmc_acceptance_probability=np.asarray(hmc_accept, dtype=np.float64),
        hmc_diverging=np.asarray(hmc_diverging, dtype=np.bool_),
        hmc_energy=np.asarray(hmc_energy, dtype=np.float64),
        hmc_energy_error=np.asarray(hmc_energy_error, dtype=np.float64),
        hmc_step_size=np.asarray(hmc_step_size, dtype=np.float64),
        hmc_n_steps=np.asarray(hmc_n_steps, dtype=np.int64),
        hmc_seed=np.asarray(hmc_seed, dtype=np.uint64),
    )
    checkpoint = FullTilingPyMCHMCCheckpoint(
        problem=problem,
        state=state,
        log_leaf_mass=np.asarray(point["x"], dtype=np.float64),
        log_fixed_coefficient=(
            np.asarray(point["y"], dtype=np.float64)
            if state.fixed_coefficients.size
            else np.empty(0, dtype=np.float64)
        ),
        rng_state=PCG64State.from_generator(rng),
        sweeps_completed=sweeps_completed + iterations,
        kernel_settings=settings,
        runtime_identity=full_tiling_pymc_hmc_runtime_identity(),
        topology_precision_sha256=_topology_precision_sha256(
            structural.topology_precision,
        ),
    )
    return FullTilingPyMCHMCSamplingResult(
        trace=trace,
        final_state=state,
        checkpoint=checkpoint,
        kernel_setup_seconds=kernel_setup_seconds,
        transition_seconds=transition_seconds,
    )


def sample_full_tiling_pymc_hmc(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
    config: FullTilingPyMCHMCConfig,
) -> FullTilingPyMCHMCSamplingResult:
    """Run a fresh mobile full-tiling compound HMC segment.

    Before draw zero, the supplied scientific state is rebuilt at an exact
    binary64 natural-log/exp fixed point. The caller's state is not mutated,
    but the retained boundary may differ from it by a few ULPs.

    Args:
        problem: Fixed-``K`` full-tiling posterior problem.
        initial_state: Fresh initializer state built for the exact problem
            object.
        config: Fresh-chain sweep count, non-adapting HMC controls, and
            optional master PCG64 seed. Exact replay from chain start requires
            an explicit seed.

    Returns:
        Boundary-inclusive trace, final oracle state, and exact checkpoint.

    Raises:
        TypeError: If arguments have invalid public types.
        ValueError: If problem identity, support, or resolved metric
            construction is incompatible.
        ImportError: If the optional PyMC runtime is unavailable.
        RuntimeError: If fresh-state canonicalization does not converge,
            PyTensor is not float64, or PyMC does not execute the configured
            leapfrog count.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(initial_state, FullTilingPosteriorState):
        raise TypeError("initial_state must be a FullTilingPosteriorState.")
    if not isinstance(config, FullTilingPyMCHMCConfig):
        raise TypeError("config must be a FullTilingPyMCHMCConfig.")
    settings = FullTilingPyMCHMCKernelSettings(
        fixed_k=initial_state.k,
        step_size=config.step_size,
        leapfrog_steps=config.leapfrog_steps,
    )
    canonical_state, log_leaf_mass, log_fixed_coefficient = canonicalize_full_tiling_pymc_hmc_fresh_state(
        problem,
        initial_state,
    )
    return _run_segment(
        problem,
        canonical_state,
        iterations=config.iterations,
        sweeps_completed=0,
        settings=settings,
        rng=np.random.Generator(np.random.PCG64(config.seed)),
        log_leaf_mass=log_leaf_mass,
        log_fixed_coefficient=log_fixed_coefficient,
    )


def continue_full_tiling_pymc_hmc(
    problem: FullTilingProblem,
    checkpoint: FullTilingPyMCHMCCheckpoint,
    *,
    iterations: int,
) -> FullTilingPyMCHMCSamplingResult:
    """Continue exactly from an in-memory compound HMC checkpoint.

    The checkpoint's scientific state and log-coordinate arrays are joint
    authoritative replay inputs. They are passed through unchanged and are
    not fresh-state canonicalized or rebuilt at the segment boundary.

    Args:
        problem: Exact problem object retained by the checkpoint.
        checkpoint: Compatible post-HMC continuation boundary.
        iterations: Positive number of additional compound sweeps.

    Returns:
        Boundary-inclusive continuation trace, final state, and next
        checkpoint.

    Raises:
        TypeError: If arguments have invalid public types.
        ValueError: If problem identity, schedule, dimensions, or support are
            incompatible.
        ImportError: If the optional PyMC runtime is unavailable.
        RuntimeError: If PyTensor is not float64 or PyMC does not execute the
            configured leapfrog count.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(checkpoint, FullTilingPyMCHMCCheckpoint):
        raise TypeError("checkpoint must be a FullTilingPyMCHMCCheckpoint.")
    count = _positive_integer(iterations, name="iterations")
    if checkpoint.problem is not problem:
        raise ValueError("continuation requires the exact checkpoint problem.")
    if checkpoint.schedule_id != FULL_TILING_PYMC_HMC_SCHEDULE_ID:
        raise ValueError("checkpoint schedule is incompatible.")
    if checkpoint.runtime_identity != full_tiling_pymc_hmc_runtime_identity():
        raise ValueError("checkpoint runtime identity is incompatible.")
    return _run_segment(
        problem,
        checkpoint.state,
        iterations=count,
        sweeps_completed=checkpoint.sweeps_completed,
        settings=checkpoint.kernel_settings,
        rng=checkpoint.rng_state.generator(),
        log_leaf_mass=checkpoint.log_leaf_mass,
        log_fixed_coefficient=checkpoint.log_fixed_coefficient,
    )
