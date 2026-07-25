"""Experimental PyMC compound kernel for mobile fixed-``K`` full tilings.

This module composes one existing full-tiling structural Metropolis--Hastings
proposal with one static PyMC Hamiltonian Monte Carlo trajectory.  The
continuous HMC chart is symmetric in the active leaves: ``x_i = log(m_i)``
for leaf masses and ``y_j = log(c_j)`` for always-active coefficients.
The PyMC model uses flat computational variables plus one explicitly
normalized potential for the scientific ``(T, shares, c)`` target, including
the chart Jacobians.

Topology-dependent design columns and Dirichlet shapes are same-shaped
``pm.Data`` containers.  A custom ``BlockedStep`` owns a harmless one-state
discrete token, calls the existing structural draw and acceptance functions
unchanged in scientific coordinates, updates both data containers before HMC,
and reseeds the PyMC HMC step from the sole NumPy PCG64 stream on every sweep.
Thus accepted, rejected, and invalid structural attempts are all followed by
exactly one HMC trajectory.

The HMC kernel is deliberately static: no tuning, step-size randomization, or
topology-dependent metric is allowed. Its leaf block has separate static
position-covariance eigenscales for normalized-common total motion and
orthogonal log-mass contrasts; fixed coefficients retain an ordered diagonal
block. In-memory checkpoints retain the scientific state, exact unconstrained
coordinates, immutable settings, sweep coordinate, and master PCG64 state.
Reconstructing and reseeding PyMC at every continuation boundary makes split
execution exactly replayable without persisting mutable backend sampler state.
Fresh initializer states instead pass through
:func:`canonicalize_full_tiling_pymc_hmc_fresh_state` before draw zero so their
physical values decode exactly from the authoritative log coordinates. This
explicitly experimental module performs no durable file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from .full_tiling import TilingState
from .full_tiling_compound_sampling import _draw_structural_transition
from .full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    accept_or_reject,
    build_full_tiling_posterior_state,
)
from .sampling import PCG64State

if TYPE_CHECKING:
    from pymc import Model

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]
UIntArray: TypeAlias = NDArray[np.uint64]

FULL_TILING_PYMC_HMC_SCHEDULE_ID = "full_tiling_1_structure_1_static_pymc_hmc_v2"
"""Versioned identity of the structural-then-HMC sweep."""

FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID = "symmetric_log_leaf_mass_then_log_fixed_coefficient_v1"
"""Versioned ordering and interpretation of the HMC value coordinates."""

FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID = (
    "pymc_position_covariance_momentum_precision_normalized_common_contrast_projector_v2"
)
"""Versioned meaning of the full matrix passed to PyMC with ``is_cov=True``."""

__all__ = [
    "FULL_TILING_PYMC_HMC_COORDINATE_LAYOUT_ID",
    "FULL_TILING_PYMC_HMC_METRIC_SEMANTICS_ID",
    "FULL_TILING_PYMC_HMC_SCHEDULE_ID",
    "FullTilingPyMCHMCCheckpoint",
    "FullTilingPyMCHMCConfig",
    "FullTilingPyMCHMCKernelSettings",
    "FullTilingPyMCHMCRuntimeIdentity",
    "FullTilingPyMCHMCSamplingResult",
    "FullTilingPyMCHMCTrace",
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
        metric_semantics_id: Versioned interpretation of the PyMC position
            covariance, equivalently momentum precision.
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


def _normalize_position_scale(
    value: float | ArrayLike,
    *,
    name: str,
) -> float | tuple[float, ...]:
    """Normalize a positive scalar or one-dimensional position-scale block."""
    array = np.asarray(value)
    if array.ndim == 0:
        return _positive_float(array.item(), name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional.")
    return tuple(_positive_float(item, name=name) for item in array.tolist())


def _resolve_fixed_position_scale(
    value: float | tuple[float, ...],
    *,
    n_fixed: int,
) -> tuple[float, ...]:
    """Resolve one shared or position-specific PyMC position scale."""
    if isinstance(value, tuple):
        if len(value) != n_fixed:
            raise ValueError("fixed_coefficient_position_scale must have one entry per fixed coefficient.")
        return value
    return (value,) * n_fixed


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


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCConfig:
    """Configuration for a fresh compound structural plus HMC segment.

    Args:
        iterations: Positive number of structural-then-HMC sweeps.
        step_size: Requested unscaled leapfrog step size. PyMC's internal
            ``exp(log(epsilon))`` representation may move the effective value
            reported in the trace by one binary64 ULP.
        leapfrog_steps: Exact positive number of leapfrog steps per sweep.
        leaf_contrast_position_scale: Positive PyMC position-covariance scale
            for leaf log-mass contrasts, equivalently momentum precision on
            the contrast subspace.
        fixed_coefficient_position_scale: Shared positive PyMC
            position-covariance scale or a positive vector in deterministic
            fixed-coefficient order, equivalently momentum precision.
        seed: Optional non-negative seed for the sole NumPy PCG64 stream.
        leaf_total_position_scale: Optional positive PyMC
            position-covariance scale for the normalized common leaf
            log-mass direction, equivalently momentum precision on that
            direction. ``None`` resolves to
            ``leaf_contrast_position_scale``. This new v2 setting is
            keyword-only so legacy positional calls retain their meaning.

    Raises:
        TypeError: If integer or scalar settings have invalid types.
        ValueError: If settings lie outside their supported ranges.
    """

    iterations: int
    step_size: float
    leapfrog_steps: int
    leaf_contrast_position_scale: float = 1.0
    fixed_coefficient_position_scale: float | tuple[float, ...] = 1.0
    seed: int | None = None
    leaf_total_position_scale: float | None = field(default=None, kw_only=True)

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
        object.__setattr__(
            self,
            "leaf_contrast_position_scale",
            _positive_float(
                self.leaf_contrast_position_scale,
                name="leaf_contrast_position_scale",
            ),
        )
        if self.leaf_total_position_scale is not None:
            object.__setattr__(
                self,
                "leaf_total_position_scale",
                _positive_float(
                    self.leaf_total_position_scale,
                    name="leaf_total_position_scale",
                ),
            )
        object.__setattr__(
            self,
            "fixed_coefficient_position_scale",
            _normalize_position_scale(
                self.fixed_coefficient_position_scale,
                name="fixed_coefficient_position_scale",
            ),
        )
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                _positive_integer(self.seed, name="seed", allow_zero=True),
            )


@dataclass(frozen=True, slots=True)
class FullTilingPyMCHMCKernelSettings:
    """Immutable problem-resolved static HMC kernel settings.

    Args:
        fixed_k: Positive leaf count preserved by the structural kernel.
        step_size: Requested unscaled leapfrog step size. The exact effective
            PyMC value is recorded for every sweep and may differ by one
            binary64 ULP.
        leapfrog_steps: Exact number of leapfrog steps.
        leaf_contrast_position_scale: PyMC position-covariance eigenvalue on the
            leaf log-mass contrast subspace, equivalently momentum precision.
        fixed_coefficient_position_scale: Resolved per-coefficient PyMC
            position-covariance diagonal, equivalently momentum precision.
        leaf_total_position_scale: PyMC position-covariance eigenvalue on the
            normalized common leaf log-mass direction, equivalently momentum
            precision.

    Raises:
        TypeError: If an integer setting has an invalid type.
        ValueError: If a scale, trajectory length, dimension, or count lies
            outside supported ranges.
    """

    fixed_k: int
    step_size: float
    leapfrog_steps: int
    leaf_contrast_position_scale: float
    fixed_coefficient_position_scale: tuple[float, ...]
    leaf_total_position_scale: float

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
        object.__setattr__(
            self,
            "leaf_contrast_position_scale",
            _positive_float(
                self.leaf_contrast_position_scale,
                name="leaf_contrast_position_scale",
            ),
        )
        total_position_scale = _positive_float(
            self.leaf_total_position_scale,
            name="leaf_total_position_scale",
        )
        object.__setattr__(self, "leaf_total_position_scale", total_position_scale)
        masses = tuple(
            _positive_float(
                value,
                name="fixed_coefficient_position_scale",
            )
            for value in self.fixed_coefficient_position_scale
        )
        object.__setattr__(
            self,
            "fixed_coefficient_position_scale",
            masses,
        )
        dimension_factor = (self.fixed_k + len(self.fixed_coefficient_position_scale)) ** 0.25
        if not math.isfinite(self.step_size * dimension_factor):
            raise ValueError("dimension-adjusted PyMC step scale must be finite.")
        _build_position_scale_matrix(self)

    @property
    def position_scale_matrix(self) -> FloatArray:
        """Return the permutation-invariant PyMC position covariance.

        The leaf block is
        ``g_contrast * (I - 11' / K) + g_total * (11' / K)``. It is followed
        by the ordered fixed-coefficient diagonal, with an exactly zero
        cross-block.

        Returns:
            Owned read-only symmetric positive-definite ``float64`` array of
            shape ``(fixed_k + n_fixed, fixed_k + n_fixed)``.
        """
        return _build_position_scale_matrix(self)

    @property
    def position_scale_diagonal(self) -> FloatArray:
        """Return the diagonal of the PyMC position-covariance matrix.

        This diagnostic projection does not contain the off-diagonal leaf
        covariances when the total and contrast scales differ. It cannot
        reconstruct the full matrix supplied to PyMC.

        Returns:
            Read-only float64 array of shape ``(fixed_k + n_fixed,)``, with
            the leaf-block diagonal followed by fixed-coefficient order.
        """
        result = np.diag(self.position_scale_matrix).copy()
        result.setflags(write=False)
        return result


def _build_position_scale_matrix(
    settings: FullTilingPyMCHMCKernelSettings,
) -> FloatArray:
    """Build and validate the full topology-neutral position covariance.

    Args:
        settings: Resolved positive leaf eigenscales and fixed diagonal.

    Returns:
        Owned read-only finite symmetric positive-definite matrix.

    Raises:
        ValueError: If binary64 construction does not produce a finite,
            symmetric positive-definite matrix.
    """
    fixed_k = settings.fixed_k
    dimension = fixed_k + len(settings.fixed_coefficient_position_scale)
    common = np.full(
        (fixed_k, fixed_k),
        1.0 / fixed_k,
        dtype=np.float64,
    )
    leaf_block = (
        settings.leaf_contrast_position_scale * np.eye(fixed_k, dtype=np.float64)
        + (settings.leaf_total_position_scale - settings.leaf_contrast_position_scale) * common
    )
    result = np.zeros((dimension, dimension), dtype=np.float64)
    result[:fixed_k, :fixed_k] = leaf_block
    if settings.fixed_coefficient_position_scale:
        fixed_indices = np.arange(fixed_k, dimension)
        result[fixed_indices, fixed_indices] = settings.fixed_coefficient_position_scale
    if np.any(~np.isfinite(result)) or not np.array_equal(result, result.T):
        raise ValueError("position scale matrix must be finite and symmetric.")
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError("position scale matrix must be positive definite.") from error
    result.setflags(write=False)
    return result


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
        structural_log_acceptance_ratio: Raw untruncated structural log ratio.
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
        kernel_settings: Complete static HMC settings.
        runtime_identity: Backend, precision, layout, and metric identity.
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
        if self.state.fixed_coefficients.size != len(self.kernel_settings.fixed_coefficient_position_scale):
            raise ValueError("checkpoint fixed block must match resolved position scales.")
        if self.schedule_id != FULL_TILING_PYMC_HMC_SCHEDULE_ID:
            raise ValueError("checkpoint schedule is incompatible.")
        if self.runtime_identity != full_tiling_pymc_hmc_runtime_identity():
            raise ValueError("checkpoint runtime identity is incompatible.")
        if not math.isfinite(self.state.log_target):
            raise ValueError("checkpoint state must have finite target support.")
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
        Exception: Propagates a container update failure after restoring all
            three original values.

    Notes:
        This function mutates the model only when all three assignments
        succeed. A failed assignment restores copies of the complete prior
        topology payload before re-raising.
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
        settings: Resolved static-HMC settings, including the exact leaf and
            fixed-coordinate position-scale matrix.
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
        RuntimeError: If the constructed compound kernel is not fully static.

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
    dimension = settings.fixed_k + len(settings.fixed_coefficient_position_scale)
    # BaseHMC divides step_scale by dimension**0.25. This inverse supplies
    # the caller's actual requested step size to the integrator.
    step_scale = settings.step_size * dimension**0.25
    hmc = pm.HamiltonianMC(
        vars=continuous_rvs,
        model=model,
        scaling=np.array(settings.position_scale_matrix, copy=True),
        is_cov=True,
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
            self.hmc_start_log_leaf_mass: FloatArray | None = None
            self.hmc_start_log_fixed_coefficient: FloatArray | None = None

        def step(
            self,
            current_point: dict[str, np.ndarray],
        ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
            """Apply structural MH, update data/masses, and seed the next HMC."""
            source = self.state
            transition, _ = _draw_structural_transition(
                problem,
                source,
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
            _set_topology_data_atomically(
                model,
                dynamic_design,
                alpha,
                dirichlet_log_normalizer,
            )
            next_point = dict(current_point)
            if accepted:
                next_point["x"] = np.log(next_state.leaf_masses)
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
        raise RuntimeError("compound HMC must be fully static before sampling.")
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
        settings: Frozen kernel settings compatible with ``initial_state`` and
            its fixed block.
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
        ValueError: If problem identity, fixed ``K``, position-scale width,
            target support, or authoritative restart coordinates are invalid.
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
    if initial_state.fixed_coefficients.size != len(settings.fixed_coefficient_position_scale):
        raise ValueError("resolved fixed position scales must match the problem fixed block.")
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
        config: Fresh-chain sweep count, static HMC controls, position scales,
            and optional master PCG64 seed. Exact replay from chain start
            requires an explicit seed.

    Returns:
        Boundary-inclusive trace, final oracle state, and exact checkpoint.

    Raises:
        TypeError: If arguments have invalid public types.
        ValueError: If problem identity, support, or resolved fixed position
            scales are incompatible.
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
    fixed_position_scale = _resolve_fixed_position_scale(
        config.fixed_coefficient_position_scale,
        n_fixed=initial_state.fixed_coefficients.size,
    )
    settings = FullTilingPyMCHMCKernelSettings(
        fixed_k=initial_state.k,
        step_size=config.step_size,
        leapfrog_steps=config.leapfrog_steps,
        leaf_contrast_position_scale=config.leaf_contrast_position_scale,
        leaf_total_position_scale=(
            config.leaf_contrast_position_scale
            if config.leaf_total_position_scale is None
            else config.leaf_total_position_scale
        ),
        fixed_coefficient_position_scale=fixed_position_scale,
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
