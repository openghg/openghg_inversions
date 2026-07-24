"""Likelihood-aware fixed-``K`` posterior operations for full leaf tilings.

This module joins the construction-history-free geometry in
:mod:`openghg_inversions.experimental.rjmcmc.full_tiling` to the immutable
observation model supplied by the Gamma--Beta RHIME bridge.  The scientific
coordinates are a positive root total and Dirichlet allocation shares on the
active rectangles. Rectangle design columns are computed lazily from direct
spatial slices of the base sensitivity matrix and cached by immutable
:class:`~openghg_inversions.experimental.rjmcmc.full_tiling.Rectangle`
instances. The complete tiling state space and exhaustive oracle path
catalogues are never enumerated; proposals construct only current-state merge
choices and fixed local destination catalogues.

All proposal constructors are deterministic.  Their discrete choices,
continuous draws, and accept/reject log-uniform value are supplied explicitly,
which makes incremental prediction updates directly testable against
:func:`build_full_tiling_posterior_state`.  The only random operation is the
separately seeded random-recursive initializer; the module contains no sampler,
persistence, or input/output layer.

Positive allocation ratios are evaluated directly in log-mass coordinates,
and matched additive-Dirichlet/Beta terms use their algebraically reduced MH
ratios. These operations target the declared continuous mass model and avoid
binary64 endpoint failures; they do not assert exact balance between finite
floating-point rounding bins.

The public entry points are :class:`FullTilingProblem`,
:class:`FullTilingPosteriorState`, :class:`PosteriorTransitionTerms`,
:func:`full_tiling_problem_from_gamma_beta_adapter`,
:func:`initialize_full_tiling_posterior_state`,
:func:`initialize_random_full_tiling_posterior_state`,
:func:`build_full_tiling_posterior_state`,
:func:`log_root_total_slice_density`,
:func:`rescale_full_tiling_root_total`, the five ``propose_*`` functions, and
:func:`accept_or_reject`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import lognormal_coefficient_log_prior_numpy
from .full_tiling import (
    AdditiveAlphaPrior,
    LeafTiling,
    MergeChoice,
    Rectangle,
    SplitChoice,
    TilingState,
    _log_beta_density_from_masses,
    _log_normalized_positive_masses,
    _representable_split_masses,
    is_recursive_bisection_tiling,
    merge_choices,
    split_choices,
)
from .gamma_beta_adapter import GammaBetaRHIMEAdapterResult
from .gamma_beta_tree import GammaBetaTreeProblem

FloatArray: TypeAlias = NDArray[np.float64]
PosteriorMove: TypeAlias = Literal[
    "edge_flip",
    "resolution_relocation",
    "pair_allocation_refresh",
    "root_total_refresh",
    "fixed_coefficient",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


def _readonly_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Return an owned finite read-only ``float64`` array.

    Args:
        values: Candidate numeric values.
        name: Field name used in validation errors.

    Returns:
        Owned read-only array retaining the input shape.

    Raises:
        ValueError: If conversion fails or a value is non-finite.
    """
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _positive_finite(value: float, *, name: str) -> float:
    """Return one finite positive scalar.

    Args:
        value: Candidate scalar.
        name: Field name used in validation errors.

    Returns:
        Positive built-in float.

    Raises:
        TypeError: If ``value`` is Boolean or not float-convertible.
        ValueError: If the converted value is non-finite or non-positive.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class FullTilingProblem:
    """Lazy full-tiling view of an existing Gamma--Beta tree problem.

    The base problem remains the sole owner of observations, observation
    errors, native-cell sensitivities, fixed design arrays, and fixed offsets.
    This wrapper therefore does not copy those potentially large arrays.
    Nominal cell masses are normalized over the native grid before constructing
    the globally additive allocation prior.

    Args:
        base: Existing immutable Gamma--Beta observation problem.
        concentration: Explicit positive total concentration for the
            full-tiling Dirichlet allocation.

    Attributes:
        allocation_prior: Additive alpha measure built from normalized nominal
            native-cell masses.

    Raises:
        TypeError: If ``base`` or ``concentration`` has the wrong type.
        ValueError: If concentration or nominal masses are malformed.
    """

    base: GammaBetaTreeProblem
    concentration: float
    allocation_prior: AdditiveAlphaPrior = field(init=False)
    _normalized_nominal_mass: FloatArray = field(init=False, repr=False)
    _design_cache: dict[Rectangle, FloatArray] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate the bridge and initialize its small lazy caches."""
        if not isinstance(self.base, GammaBetaTreeProblem):
            raise TypeError("base must be a GammaBetaTreeProblem.")
        concentration = _positive_finite(self.concentration, name="concentration")
        nominal = np.asarray(self.base.prior.nominal_cell_mass, dtype=np.float64)
        total = float(nominal.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("base nominal cell masses must have a positive finite sum.")
        normalized = np.array(
            nominal.reshape(self.base.tree.shape) / total,
            dtype=np.float64,
            copy=True,
        )
        normalized.setflags(write=False)
        allocation_prior = AdditiveAlphaPrior(normalized, concentration)
        object.__setattr__(self, "concentration", concentration)
        object.__setattr__(self, "_normalized_nominal_mass", normalized)
        object.__setattr__(self, "allocation_prior", allocation_prior)
        object.__setattr__(self, "_design_cache", {})

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native spatial ``(rows, columns)`` shape.

        Returns:
            Shape shared by the base tree and all full tilings.
        """
        return self.base.tree.shape

    @property
    def base_problem(self) -> GammaBetaTreeProblem:
        """Return the wrapped Gamma--Beta problem without copying.

        Returns:
            Exact object supplied at construction.
        """
        return self.base

    @property
    def observations(self) -> FloatArray:
        """Return the base read-only observation vector without copying.

        Returns:
            Base array with shape ``(n_observations,)``.
        """
        return self.base.observations

    @property
    def observation_sd(self) -> FloatArray:
        """Return the base read-only Gaussian standard deviations.

        Returns:
            Base array aligned with :attr:`observations`.
        """
        return self.base.observation_sd

    @property
    def normalized_nominal_mass(self) -> FloatArray:
        """Return normalized nominal cell masses in spatial shape.

        Returns:
            Read-only ``float64`` array summing to one.
        """
        return self._normalized_nominal_mass

    @property
    def n_fixed_coefficients(self) -> int:
        """Return the number of always-active fixed coefficients.

        Returns:
            Zero when the base problem has no fixed design block.
        """
        return self.base.n_fixed_coefficients

    def rectangle_nominal_mass(self, rectangle: Rectangle) -> float:
        """Return normalized nominal mass contained in one rectangle.

        Args:
            rectangle: In-domain half-open native-grid block.

        Returns:
            Strictly positive normalized nominal mass.

        Raises:
            TypeError: If ``rectangle`` has the wrong type.
            ValueError: If the rectangle lies outside the native grid.
        """
        self._validate_rectangle(rectangle)
        return float(
            self._normalized_nominal_mass[
                rectangle.row_start : rectangle.row_stop,
                rectangle.col_start : rectangle.col_stop,
            ].sum()
        )

    def design_column(self, rectangle: Rectangle) -> FloatArray:
        """Return the cached response per unit total mass in a rectangle.

        The column is the nominal-mass-weighted average of native-cell
        response-per-unit-mass columns.  It is computed from direct
        ``(row, column)`` slices of the base sensitivity matrix, copied only
        into the one-dimensional cached result.

        Args:
            rectangle: In-domain native-grid block.

        Returns:
            Read-only vector with one value per observation.

        Raises:
            TypeError: If ``rectangle`` has the wrong type.
            ValueError: If the rectangle lies outside the native grid.
        """
        self._validate_rectangle(rectangle)
        cached = self._design_cache.get(rectangle)
        if cached is not None:
            return cached
        spatial_sensitivity = self.base.sensitivity.reshape(
            self.base.observations.size,
            self.shape[0],
            self.shape[1],
        )
        mass_slice = self._normalized_nominal_mass[
            rectangle.row_start : rectangle.row_stop,
            rectangle.col_start : rectangle.col_stop,
        ]
        sensitivity_slice = spatial_sensitivity[
            :,
            rectangle.row_start : rectangle.row_stop,
            rectangle.col_start : rectangle.col_stop,
        ]
        nominal_mass = float(mass_slice.sum())
        column = np.zeros(self.base.observations.size, dtype=np.float64)
        for row_offset in range(rectangle.height):
            column += sensitivity_slice[:, row_offset, :] @ mass_slice[row_offset, :]
        column /= nominal_mass
        result = _readonly_float_array(column, name="rectangle design column")
        self._design_cache[rectangle] = result
        return result

    def rectangle_design_column(self, rectangle: Rectangle) -> FloatArray:
        """Return the cached per-unit-total column for one rectangle.

        Args:
            rectangle: In-domain native-grid block.

        Returns:
            Same read-only object as :meth:`design_column`.
        """
        return self.design_column(rectangle)

    def _validate_rectangle(self, rectangle: Rectangle) -> None:
        """Require one rectangle to lie inside this problem's grid."""
        if not isinstance(rectangle, Rectangle):
            raise TypeError("rectangle must be a Rectangle.")
        if (
            rectangle.row_start < 0
            or rectangle.col_start < 0
            or rectangle.row_stop > self.shape[0]
            or rectangle.col_stop > self.shape[1]
        ):
            raise ValueError("rectangle must lie within the problem grid.")


@dataclass(frozen=True, slots=True, eq=False)
class FullTilingPosteriorState:
    """Full-tiling allocation and complete posterior caches.

    The continuous chart is ``(T, shares)``: ``allocation.total_mass`` is the
    positive root total and normalized leaf masses are the simplex shares.
    Consequently :attr:`log_allocation_prior` is the normalized Dirichlet
    density with respect to shares and contains no ``T**(-(K-1))`` physical
    mass-coordinate factor.

    Attributes:
        problem: Exact full-tiling problem used for all caches.
        allocation: Canonical geometry and positive aligned leaf masses.
        fixed_coefficients: Positive always-active coefficients.
        dynamic_prediction: Tiling contribution in observation space.
        fixed_prediction: Fixed offset plus fixed-design contribution.
        prediction: Sum of dynamic and fixed predictions.
        residual: ``prediction - observations``.
        log_gaussian_likelihood: Raw normalized independent-Gaussian density.
        log_likelihood: Likelihood-power-scaled Gaussian target component.
        log_root_prior: Normalized Gamma density of total mass.
        log_allocation_prior: Normalized Dirichlet share density.
        log_fixed_coefficient_prior: Normalized independent lognormal density.

    Note:
        Use :func:`build_full_tiling_posterior_state` rather than direct
        construction. States returned by the public builders contain
        validated read-only caches; direct dataclass construction does not
        enforce those invariants.
    """

    problem: FullTilingProblem
    allocation: TilingState
    fixed_coefficients: FloatArray
    dynamic_prediction: FloatArray
    fixed_prediction: FloatArray
    prediction: FloatArray
    residual: FloatArray
    log_gaussian_likelihood: float
    log_likelihood: float
    log_root_prior: float
    log_allocation_prior: float
    log_fixed_coefficient_prior: float

    @property
    def k(self) -> int:
        """Return the active rectangle count.

        Returns:
            Positive fixed dimension of the tiling.
        """
        return self.allocation.tiling.k

    @property
    def root_total(self) -> float:
        """Return the positive total allocation coordinate.

        Returns:
            Sum of all active leaf allocations in nominal-weight units.
        """
        return self.allocation.total_mass

    @property
    def tiling_state(self) -> TilingState:
        """Return the canonical geometry and leaf-mass allocation.

        Returns:
            Exact immutable allocation object.
        """
        return self.allocation

    @property
    def total_prediction(self) -> FloatArray:
        """Return the cached total observation-space prediction.

        Returns:
            Same read-only object as :attr:`prediction`.
        """
        return self.prediction

    @property
    def leaf_masses(self) -> FloatArray:
        """Return aligned read-only active leaf masses.

        Returns:
            Array in canonical rectangle order.
        """
        return self.allocation.leaf_masses

    @property
    def log_target(self) -> float:
        """Return the powered likelihood plus continuous-prior target.

        Returns:
            Log density in root-total plus allocation-share coordinates,
            omitting the constant structural normalizer of the reachable
            fixed-``K`` component.
        """
        return float(
            self.log_likelihood
            + self.log_root_prior
            + self.log_allocation_prior
            + self.log_fixed_coefficient_prior
        )


@dataclass(frozen=True, slots=True, eq=False)
class PosteriorTransitionTerms:
    """Candidate state and decomposed deterministic MH accounting.

    Attributes:
        candidate: Proposed posterior state; invalid attempts retain the source.
        move: Stable proposal name.
        delta_log_likelihood: Powered likelihood change.
        delta_log_root_prior: Normalized Gamma-prior change.
        delta_log_allocation_prior: Normalized Dirichlet-share prior change.
        delta_log_fixed_coefficient_prior: Independent lognormal-prior change.
        log_q_forward_selection: Forward discrete-choice log probability.
        log_q_forward_auxiliary: Forward continuous auxiliary log density.
        log_q_reverse_selection: Reverse discrete-choice log probability.
        log_q_reverse_auxiliary: Reverse continuous auxiliary log density.
        log_jacobian: Log absolute augmented-coordinate Jacobian.
        reverse_merge_choice: Unique reverse merge for structural proposals.
        reverse_split_choice: Unique reverse split for structural proposals.
        valid: Whether the candidate enters an MH decision.
        reason: Invalid self-transition explanation.
        log_target_delta: Sum of all candidate-minus-source target components.
        log_q_forward: Sum of forward selection and auxiliary terms.
        log_q_reverse: Sum of reverse selection and auxiliary terms.
        log_acceptance_ratio: Complete untruncated MH log ratio.
            Matched additive-Dirichlet/Beta proposals use their algebraically
            reduced ratio to avoid floating-point cancellation.

    Raises:
        TypeError: If candidate or validity metadata have the wrong type.
        ValueError: If move, reverse metadata, or logarithmic terms are
            inconsistent.
    """

    candidate: FullTilingPosteriorState
    move: PosteriorMove
    delta_log_likelihood: float
    delta_log_root_prior: float = 0.0
    delta_log_allocation_prior: float = 0.0
    delta_log_fixed_coefficient_prior: float = 0.0
    log_q_forward_selection: float = 0.0
    log_q_forward_auxiliary: float = 0.0
    log_q_reverse_selection: float = 0.0
    log_q_reverse_auxiliary: float = 0.0
    log_jacobian: float = 0.0
    reverse_merge_choice: MergeChoice | None = None
    reverse_split_choice: SplitChoice | None = None
    valid: bool = True
    reason: str | None = None
    log_target_delta: float = field(init=False)
    log_q_forward: float = field(init=False)
    log_q_reverse: float = field(init=False)
    log_acceptance_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate decomposed terms and calculate aggregate MH quantities."""
        if not isinstance(self.candidate, FullTilingPosteriorState):
            raise TypeError("candidate must be a FullTilingPosteriorState.")
        if self.move not in (
            "edge_flip",
            "resolution_relocation",
            "pair_allocation_refresh",
            "root_total_refresh",
            "fixed_coefficient",
        ):
            raise ValueError("move must name a supported full-tiling posterior proposal.")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be Boolean.")
        if self.valid and self.reason is not None:
            raise ValueError("a valid transition cannot have an invalidity reason.")
        if not self.valid and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("an invalid transition must provide a non-empty reason.")
        if self.valid and self.move in ("edge_flip", "resolution_relocation"):
            if self.reverse_merge_choice is None or self.reverse_split_choice is None:
                raise ValueError("valid structural transitions require unique reverse choices.")
        names = (
            "delta_log_likelihood",
            "delta_log_root_prior",
            "delta_log_allocation_prior",
            "delta_log_fixed_coefficient_prior",
            "log_q_forward_selection",
            "log_q_forward_auxiliary",
            "log_q_reverse_selection",
            "log_q_reverse_auxiliary",
            "log_jacobian",
        )
        values = tuple(float(getattr(self, name)) for name in names)
        if any(math.isnan(value) for value in values):
            raise ValueError("transition logarithmic terms cannot be NaN.")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)
        target_delta = sum(values[:4])
        q_forward = values[4] + values[5]
        q_reverse = values[6] + values[7]
        if self.valid:
            acceptance = target_delta + q_reverse - q_forward + values[8]
        else:
            acceptance = -math.inf
        if math.isnan(acceptance):
            raise ValueError("the calculated log acceptance ratio cannot be NaN.")
        object.__setattr__(self, "log_target_delta", float(target_delta))
        object.__setattr__(self, "log_q_forward", float(q_forward))
        object.__setattr__(self, "log_q_reverse", float(q_reverse))
        object.__setattr__(self, "log_acceptance_ratio", float(acceptance))


def full_tiling_problem_from_gamma_beta_adapter(
    source: GammaBetaRHIMEAdapterResult | GammaBetaTreeProblem,
    *,
    concentration: float,
) -> FullTilingProblem:
    """Bridge an adapter result or numerical tree problem to full tilings.

    Args:
        source: RHIME adapter result or its public numerical problem.
        concentration: Explicit positive global Dirichlet concentration.

    Returns:
        Lazy full-tiling problem retaining the exact base problem instance.

    Raises:
        TypeError: If ``source`` is neither supported bridge type.
        ValueError: If concentration or nominal masses are malformed.
    """
    if isinstance(source, GammaBetaRHIMEAdapterResult):
        base = source.problem
    elif isinstance(source, GammaBetaTreeProblem):
        base = source
    else:
        raise TypeError("source must be a GammaBetaRHIMEAdapterResult or GammaBetaTreeProblem.")
    return FullTilingProblem(base=base, concentration=concentration)


def initialize_full_tiling_posterior_state(
    problem: FullTilingProblem,
    *,
    k: int,
) -> FullTilingPosteriorState:
    """Build a deterministic prior-mean state at one requested fixed ``K``.

    Starting at the canonical tree root, the active node having the largest
    nominal mass is split at each step.  Stable node ID breaks ties.  The
    resulting node bounds become construction-history-free rectangles.  Leaf
    masses equal the Gamma root-prior mean times nominal leaf shares, and fixed
    coefficients equal their independent arithmetic prior means.

    Args:
        problem: Full-tiling observation model.
        k: Requested active rectangle count.

    Returns:
        Fully rebuilt immutable initial posterior state.

    Raises:
        TypeError: If ``problem`` or ``k`` has the wrong type.
        ValueError: If ``k`` lies outside the native-cell range.
    """
    k = _validate_initial_tiling_request(problem, k)
    active = {problem.base.tree.root_id}
    while len(active) < k:
        splittable = [node_id for node_id in active if problem.base.tree.children(node_id)]
        if not splittable:
            raise RuntimeError("canonical tree cannot reach the requested active count.")
        selected = min(
            splittable,
            key=lambda node_id: (
                -float(problem.base.node_nominal_mass[node_id]),
                node_id,
            ),
        )
        active.remove(selected)
        active.update(problem.base.tree.children(selected))
    rectangles = tuple(Rectangle(*problem.base.tree.node(node_id).bounds) for node_id in active)
    tiling = LeafTiling(problem.shape, rectangles)
    return _build_prior_mean_full_tiling_posterior_state(problem, tiling)


def initialize_random_full_tiling_posterior_state(
    problem: FullTilingProblem,
    *,
    k: int,
    seed: int,
) -> FullTilingPosteriorState:
    """Build a seeded random-recursive prior-mean state at fixed ``K``.

    A fresh NumPy PCG64 stream selects uniformly from the canonical
    :func:`~openghg_inversions.experimental.rjmcmc.full_tiling.split_choices`
    catalogue at every recursive bisection.  This initialization-only stream
    is separate from any sampler stream. Stepwise uniform selection is
    generally path-biased because different final tilings can have different
    construction-path multiplicities. It is an initializer, not a uniform
    draw from the final tilings and not the structural prior or target. Leaf
    masses and fixed coefficients use the same prior-mean construction as
    :func:`initialize_full_tiling_posterior_state`.

    Args:
        problem: Full-tiling observation model.
        k: Requested active rectangle count.
        seed: Non-negative integer seed for the initialization-only PCG64
            stream.

    Returns:
        Fully rebuilt immutable initial posterior state.

    Raises:
        TypeError: If ``problem``, ``k``, or ``seed`` has the wrong type.
        ValueError: If ``k`` lies outside the native-cell range or ``seed`` is
            negative.
        RuntimeError: If recursive bisection cannot reach ``k``.
    """
    k = _validate_initial_tiling_request(problem, k)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-negative integer.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    generator = np.random.Generator(np.random.PCG64(seed))
    tiling = LeafTiling.root(problem.shape)
    while tiling.k < k:
        choices = split_choices(tiling)
        if not choices:
            raise RuntimeError("recursive bisection cannot reach the requested active count.")
        tiling = tiling.split(choices[int(generator.integers(len(choices)))])
    return _build_prior_mean_full_tiling_posterior_state(problem, tiling)


def _validate_initial_tiling_request(problem: FullTilingProblem, k: int) -> int:
    """Validate and normalize a requested initial tiling size.

    Args:
        problem: Full-tiling observation model.
        k: Requested active rectangle count.

    Returns:
        Requested count as a built-in integer.

    Raises:
        TypeError: If ``problem`` or ``k`` has the wrong type.
        ValueError: If ``k`` lies outside the native-cell range.
    """
    _require_problem(problem)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise TypeError("k must be an integer.")
    normalized_k = int(k)
    maximum = len(problem.base.tree.leaf_ids)
    if normalized_k < 1 or normalized_k > maximum:
        raise ValueError(f"k must lie between one and {maximum}.")
    return normalized_k


def _build_prior_mean_full_tiling_posterior_state(
    problem: FullTilingProblem,
    tiling: LeafTiling,
) -> FullTilingPosteriorState:
    """Build prior-mean posterior coordinates on a validated full tiling.

    Args:
        problem: Full-tiling observation model.
        tiling: Validated construction-history-free leaf tiling.

    Returns:
        Fully rebuilt immutable state with masses proportional to nominal
        emissions and fixed coefficients at their arithmetic prior means.
    """
    root_mean = problem.base.prior.root_shape / problem.base.prior.root_rate
    nominal = np.asarray(
        [problem.rectangle_nominal_mass(leaf) for leaf in tiling.leaves],
        dtype=np.float64,
    )
    masses = root_mean * nominal / float(nominal.sum())
    allocation = TilingState(tiling, masses)
    fixed_coefficients: ArrayLike
    if problem.base.fixed_block is None:
        fixed_coefficients = np.empty(0, dtype=np.float64)
    else:
        fixed_coefficients = problem.base.fixed_block.coefficient_prior_mean
    return build_full_tiling_posterior_state(
        problem,
        allocation=allocation,
        fixed_coefficients=fixed_coefficients,
    )


def build_full_tiling_posterior_state(
    problem: FullTilingProblem,
    *,
    allocation: TilingState,
    fixed_coefficients: ArrayLike | None = None,
) -> FullTilingPosteriorState:
    """Fully rebuild one fixed-``K`` posterior state from direct columns.

    This complete builder is the validation oracle for all incremental
    proposal paths.

    Args:
        problem: Full-tiling observation model and priors.
        allocation: Canonical active tiling with aligned positive masses.
        fixed_coefficients: Positive coefficients for the configured fixed
            block.  Omit only when no fixed block exists.

    Returns:
        Immutable state with exact prediction closure and normalized target
        components.

    Raises:
        TypeError: If ``problem`` or ``allocation`` has the wrong type.
        ValueError: If shapes, grids, fixed coefficients, or derived caches are
            inconsistent.
    """
    _validate_problem_allocation(problem, allocation)
    coefficients = _prepare_fixed_coefficients(problem, fixed_coefficients)
    dynamic = np.zeros(problem.observations.shape, dtype=np.float64)
    for leaf, mass in zip(
        allocation.tiling.leaves,
        allocation.leaf_masses,
        strict=True,
    ):
        dynamic += float(mass) * problem.design_column(leaf)
    fixed = np.array(problem.base.fixed_offset, dtype=np.float64, copy=True)
    if problem.base.fixed_block is not None:
        fixed += problem.base.fixed_block.design @ coefficients
    return _assemble_state(
        problem,
        allocation=allocation,
        fixed_coefficients=coefficients,
        dynamic_prediction=dynamic,
        fixed_prediction=fixed,
    )


def log_root_total_slice_density(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    log_root_total: float,
) -> float:
    """Return the exact conditional log density for ``z = log(T)``.

    Geometry, allocation shares, and fixed coefficients are held at
    ``source``. The result contains the powered normalized Gaussian likelihood,
    the normalized Gamma density in the scientific ``T`` chart, and the
    ``+z`` Jacobian for the computational log-total chart. Terms constant in
    ``z`` from allocation shares and fixed coefficients are omitted.

    The Gamma contribution is evaluated directly as a function of ``z``.
    Consequently a finite ``z`` remains supported when ``exp(z)`` underflows
    to zero. Overflowing exponentials and non-finite log totals have log
    density negative infinity.

    Args:
        problem: Exact problem associated with ``source``.
        source: State supplying fixed geometry, shares, and predictions.
        log_root_total: Candidate log root total ``z``.

    Returns:
        Conditional log density in the computational ``z`` chart, or negative
        infinity outside representable upper support.

    Raises:
        TypeError: If object arguments have the wrong type or
            ``log_root_total`` is not float-convertible.
        ValueError: If ``source`` belongs to a different problem or
            ``log_root_total`` cannot be converted to a float.
    """
    _validate_problem_state(problem, source)
    if isinstance(log_root_total, (bool, np.bool_)):
        return -math.inf
    z = float(log_root_total)
    if not math.isfinite(z):
        return -math.inf
    try:
        root_total = math.exp(z)
    except OverflowError:
        return -math.inf

    likelihood_power = problem.base.likelihood_power
    log_likelihood_constant = 0.0
    likelihood_quadratic = 0.0
    likelihood_linear = 0.0
    if likelihood_power != 0.0:
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_dynamic = (source.dynamic_prediction / source.root_total) / problem.observation_sd
            weighted_fixed_residual = (
                source.fixed_prediction - problem.observations
            ) / problem.observation_sd
            dynamic_squared_norm = float(np.dot(weighted_dynamic, weighted_dynamic))
            dynamic_fixed_inner_product = float(np.dot(weighted_dynamic, weighted_fixed_residual))
            fixed_squared_norm = float(np.dot(weighted_fixed_residual, weighted_fixed_residual))
        if not all(
            math.isfinite(value)
            for value in (
                dynamic_squared_norm,
                dynamic_fixed_inner_product,
                fixed_squared_norm,
            )
        ):
            return -math.inf
        log_likelihood_constant = float(
            likelihood_power
            * (
                -0.5 * fixed_squared_norm
                - np.log(problem.observation_sd).sum()
                - 0.5 * problem.observations.size * _LOG_TWO_PI
            )
        )
        likelihood_quadratic = float(0.5 * likelihood_power * dynamic_squared_norm)
        likelihood_linear = float(likelihood_power * dynamic_fixed_inner_product)

    prior = problem.base.prior
    root_squared = root_total * root_total
    if likelihood_quadratic > 0.0 and not math.isfinite(root_squared):
        return -math.inf
    quadratic_penalty = 0.0 if likelihood_quadratic == 0.0 else likelihood_quadratic * root_squared
    if not math.isfinite(quadratic_penalty):
        return -math.inf
    linear_penalty = (prior.root_rate + likelihood_linear) * root_total
    result = float(
        log_likelihood_constant
        + prior.root_shape * math.log(prior.root_rate)
        - math.lgamma(prior.root_shape)
        + prior.root_shape * z
        - quadratic_penalty
        - linear_penalty
    )
    return -math.inf if math.isnan(result) else result


def rescale_full_tiling_root_total(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    new_root_total: float,
) -> FullTilingPosteriorState:
    """Construct a posterior state with only its root total rescaled.

    The tiling, allocation shares, and fixed coefficients are unchanged.
    Dynamic prediction is rescaled linearly, and all dependent posterior
    caches are reconstructed through the same validated assembly path used by
    the independence proposal.

    Args:
        problem: Exact problem associated with ``source``.
        source: State whose scientific root total is replaced.
        new_root_total: Finite strictly positive replacement total.

    Returns:
        Fully assembled state at the replacement total.

    Raises:
        TypeError: If object arguments or ``new_root_total`` have the wrong
            type.
        ValueError: If ``source`` belongs to a different problem or the new
            total is not finite and strictly positive.
    """
    _validate_problem_state(problem, source)
    root_total = _positive_finite(new_root_total, name="new_root_total")
    shares = source.allocation.leaf_masses / source.root_total
    allocation = TilingState(
        source.allocation.tiling,
        shares * root_total,
    )
    dynamic = (source.dynamic_prediction / source.root_total) * root_total
    return _assemble_state(
        problem,
        allocation=allocation,
        fixed_coefficients=source.fixed_coefficients,
        dynamic_prediction=dynamic,
        fixed_prediction=source.fixed_prediction,
    )


def propose_posterior_edge_flip(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    merge_choice: MergeChoice,
    new_fraction: float,
) -> PosteriorTransitionTerms:
    """Construct one likelihood-aware selected-attempt edge flip.

    The caller selects one merge uniformly from ``merge_choices(source)``.
    The merged parent is then bisected on the perpendicular axis.  Unavailable
    merges, unavailable perpendicular axes, nonrecursive candidates, and
    fractions outside ``(0, 1)`` are explicit invalid self-transitions.

    Args:
        problem: Exact problem associated with ``source``.
        source: Source posterior state.
        merge_choice: Caller-selected source friend-pair merge.
        new_fraction: Proposed first-child share of the conserved pair total.

    Returns:
        Candidate and complete likelihood, allocation, Beta, selection, and
        Jacobian terms with a unique reverse choice.

    Raises:
        TypeError: If an object argument has the wrong type.
        ValueError: If ``source`` belongs to a different problem.
    """
    _validate_problem_state(problem, source)
    if not isinstance(merge_choice, MergeChoice):
        raise TypeError("merge_choice must be a MergeChoice.")
    move: PosteriorMove = "edge_flip"
    merges = merge_choices(source.allocation.tiling)
    if merge_choice not in merges:
        return _invalid_transition(source, move, "selected merge is unavailable")
    fraction = _proposal_fraction(new_fraction)
    if fraction is None:
        return _invalid_transition(source, move, "new fraction lies outside support")
    target_axis = "vertical" if merge_choice.axis == "horizontal" else "horizontal"
    if target_axis not in merge_choice.parent.admissible_axes:
        return _invalid_transition(source, move, "perpendicular split is unavailable")
    intermediate = source.allocation.tiling.merge(merge_choice)
    target_split = SplitChoice(merge_choice.parent, target_axis)
    candidate_tiling = LeafTiling(
        problem.shape,
        tuple(leaf for leaf in intermediate.leaves if leaf != merge_choice.parent)
        + merge_choice.parent.midpoint_children(target_axis),
    )
    if not is_recursive_bisection_tiling(candidate_tiling):
        return _invalid_transition(source, move, "edge-flip candidate is nonrecursive")

    source_children = merge_choice.children
    first_source_mass = source.allocation.mass(source_children[0])
    second_source_mass = source.allocation.mass(source_children[1])
    old_total = first_source_mass + second_source_mass
    target_children = target_split.leaf.midpoint_children(target_split.axis)
    proposed_masses = _representable_split_masses(old_total, fraction)
    if proposed_masses is None:
        return _invalid_transition(
            source,
            move,
            "proposed child masses are outside representable support",
        )
    masses = {
        leaf: source.allocation.mass(leaf)
        for leaf in source.allocation.tiling.leaves
        if leaf not in source_children
    }
    masses[target_children[0]], masses[target_children[1]] = proposed_masses
    allocation = _allocation_from_mass_map(candidate_tiling, masses)
    candidate = _incremental_allocation_state(
        problem,
        source,
        allocation,
        removed=source_children,
        added=target_children,
    )
    reverse_merge = MergeChoice(merge_choice.parent, target_axis)
    reverse_split = SplitChoice(merge_choice.parent, merge_choice.axis)
    reverse_merges = merge_choices(candidate_tiling)
    if reverse_merge not in reverse_merges:
        raise RuntimeError("constructed edge flip has no unique reverse merge.")
    return _valid_transition(
        source,
        candidate,
        move=move,
        delta_log_allocation_prior=(candidate.log_allocation_prior - source.log_allocation_prior),
        log_q_forward_selection=-math.log(len(merges)),
        log_q_forward_auxiliary=problem.allocation_prior.log_beta_density(
            target_children,
            fraction,
        ),
        log_q_reverse_selection=-math.log(len(reverse_merges)),
        log_q_reverse_auxiliary=_log_beta_density_from_masses(
            problem.allocation_prior.alpha(source_children[0]),
            problem.allocation_prior.alpha(source_children[1]),
            first_source_mass,
            second_source_mass,
        ),
        log_jacobian=0.0,
        reverse_merge_choice=reverse_merge,
        reverse_split_choice=reverse_split,
        reduced_log_acceptance_ratio=(
            candidate.log_likelihood
            - source.log_likelihood
            - math.log(len(reverse_merges))
            + math.log(len(merges))
        ),
    )


def propose_posterior_resolution_relocation(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    merge_choice: MergeChoice,
    split_choice: SplitChoice,
    new_fraction: float,
) -> PosteriorTransitionTerms:
    """Construct one likelihood-aware selected-attempt resolution relocation.

    After the uniformly selected merge, the caller-selected destination is
    interpreted in a fixed catalogue containing both axes for each of the
    ``K-1`` intermediate leaves.  Thus that catalogue always has
    ``2 * (K - 1)`` entries, including invalid attempts, and cancels between
    forward and reverse directions.  Splitting the merged parent, an absent
    leaf, an inadmissible axis, or a nonrecursive candidate yields an explicit
    self-transition.

    Args:
        problem: Exact problem associated with ``source``.
        source: Source posterior state.
        merge_choice: Caller-selected source friend-pair merge.
        split_choice: Caller-selected intermediate leaf and axis.
        new_fraction: Proposed first-child share of the destination total.

    Returns:
        Candidate and complete likelihood, allocation, Beta, merge-selection,
        fixed-catalogue, and physical augmented-map Jacobian terms.

    Raises:
        TypeError: If an object argument has the wrong type.
        ValueError: If ``source`` belongs to a different problem.
    """
    _validate_problem_state(problem, source)
    if not isinstance(merge_choice, MergeChoice):
        raise TypeError("merge_choice must be a MergeChoice.")
    if not isinstance(split_choice, SplitChoice):
        raise TypeError("split_choice must be a SplitChoice.")
    move: PosteriorMove = "resolution_relocation"
    merges = merge_choices(source.allocation.tiling)
    if merge_choice not in merges:
        return _invalid_transition(source, move, "selected merge is unavailable")
    fraction = _proposal_fraction(new_fraction)
    if fraction is None:
        return _invalid_transition(source, move, "new fraction lies outside support")
    intermediate = source.allocation.tiling.merge(merge_choice)
    if split_choice.leaf == merge_choice.parent:
        return _invalid_transition(source, move, "relocation cannot split the merged parent")
    if split_choice.leaf not in intermediate.leaves:
        return _invalid_transition(source, move, "selected destination leaf is unavailable")
    if split_choice.axis not in split_choice.leaf.admissible_axes:
        return _invalid_transition(source, move, "selected destination axis is inadmissible")
    destination_children = split_choice.leaf.midpoint_children(split_choice.axis)
    candidate_tiling = LeafTiling(
        problem.shape,
        tuple(leaf for leaf in intermediate.leaves if leaf != split_choice.leaf) + destination_children,
    )
    if not is_recursive_bisection_tiling(candidate_tiling):
        return _invalid_transition(source, move, "relocation candidate is nonrecursive")

    source_children = merge_choice.children
    first_source_mass = source.allocation.mass(source_children[0])
    second_source_mass = source.allocation.mass(source_children[1])
    source_total = first_source_mass + second_source_mass
    destination_total = source.allocation.mass(split_choice.leaf)
    proposed_masses = _representable_split_masses(
        destination_total,
        fraction,
    )
    if proposed_masses is None:
        return _invalid_transition(
            source,
            move,
            "proposed child masses are outside representable support",
        )
    removed = (*source_children, split_choice.leaf)
    added = (merge_choice.parent, *destination_children)
    masses = {
        leaf: source.allocation.mass(leaf) for leaf in source.allocation.tiling.leaves if leaf not in removed
    }
    masses[merge_choice.parent] = source_total
    masses[destination_children[0]], masses[destination_children[1]] = proposed_masses
    allocation = _allocation_from_mass_map(candidate_tiling, masses)
    candidate = _incremental_allocation_state(
        problem,
        source,
        allocation,
        removed=removed,
        added=added,
    )
    reverse_merge = MergeChoice(split_choice.leaf, split_choice.axis)
    reverse_split = SplitChoice(merge_choice.parent, merge_choice.axis)
    reverse_merges = merge_choices(candidate_tiling)
    if reverse_merge not in reverse_merges:
        raise RuntimeError("constructed relocation has no unique reverse merge.")
    destination_catalogue_size = 2 * (source.k - 1)
    log_forward_selection = -math.log(len(merges)) - math.log(destination_catalogue_size)
    log_reverse_selection = -math.log(len(reverse_merges)) - math.log(destination_catalogue_size)
    return _valid_transition(
        source,
        candidate,
        move=move,
        delta_log_allocation_prior=(candidate.log_allocation_prior - source.log_allocation_prior),
        log_q_forward_selection=log_forward_selection,
        log_q_forward_auxiliary=problem.allocation_prior.log_beta_density(
            destination_children,
            fraction,
        ),
        log_q_reverse_selection=log_reverse_selection,
        log_q_reverse_auxiliary=_log_beta_density_from_masses(
            problem.allocation_prior.alpha(source_children[0]),
            problem.allocation_prior.alpha(source_children[1]),
            first_source_mass,
            second_source_mass,
        ),
        log_jacobian=math.log(destination_total) - math.log(source_total),
        reverse_merge_choice=reverse_merge,
        reverse_split_choice=reverse_split,
        reduced_log_acceptance_ratio=(
            candidate.log_likelihood - source.log_likelihood + log_reverse_selection - log_forward_selection
        ),
    )


def propose_pair_allocation_refresh(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    first_leaf: Rectangle,
    second_leaf: Rectangle,
    new_fraction: float,
) -> PosteriorTransitionTerms:
    """Refresh any unordered active leaf pair while preserving its total.

    The caller selects uniformly from the fixed ``K choose 2`` catalogue of
    canonical unordered active-leaf pairs. ``first_leaf`` must precede
    ``second_leaf`` in canonical rectangle order. The continuous proposal is
    the normalized additive-alpha Beta conditional for that ordered pair;
    adjacency or a shared recursive parent is not required.

    Args:
        problem: Exact problem associated with ``source``.
        source: Source posterior state.
        first_leaf: Canonically first caller-selected active rectangle.
        second_leaf: Canonically second caller-selected active rectangle.
        new_fraction: Proposed first-leaf share of the pair total.

    Returns:
        Candidate and complete likelihood, Dirichlet, selection, and normalized
        Beta proposal terms.

    Raises:
        TypeError: If an object argument has the wrong type.
        ValueError: If ``source`` belongs to a different problem.
    """
    _validate_problem_state(problem, source)
    if not isinstance(first_leaf, Rectangle) or not isinstance(second_leaf, Rectangle):
        raise TypeError("first_leaf and second_leaf must be Rectangle objects.")
    move: PosteriorMove = "pair_allocation_refresh"
    leaves = source.allocation.tiling.leaves
    if first_leaf not in leaves or second_leaf not in leaves:
        return _invalid_transition(source, move, "selected pair contains an inactive leaf")
    if first_leaf >= second_leaf:
        return _invalid_transition(
            source,
            move,
            "selected pair must use distinct leaves in canonical order",
        )
    fraction = _proposal_fraction(new_fraction)
    if fraction is None:
        return _invalid_transition(source, move, "new fraction lies outside support")
    first_source_mass = source.allocation.mass(first_leaf)
    second_source_mass = source.allocation.mass(second_leaf)
    pair_total = first_source_mass + second_source_mass
    proposed_masses = _representable_split_masses(pair_total, fraction)
    if proposed_masses is None:
        return _invalid_transition(
            source,
            move,
            "proposed pair masses are outside representable support",
        )
    masses = {leaf: source.allocation.mass(leaf) for leaf in source.allocation.tiling.leaves}
    masses[first_leaf], masses[second_leaf] = proposed_masses
    allocation = _allocation_from_mass_map(source.allocation.tiling, masses)
    candidate = _incremental_allocation_state(
        problem,
        source,
        allocation,
        removed=(first_leaf, second_leaf),
        added=(first_leaf, second_leaf),
    )
    pair_count = source.k * (source.k - 1) // 2
    return _valid_transition(
        source,
        candidate,
        move=move,
        delta_log_allocation_prior=(candidate.log_allocation_prior - source.log_allocation_prior),
        log_q_forward_selection=-math.log(pair_count),
        log_q_forward_auxiliary=_log_pair_beta_density(
            problem,
            first_leaf,
            second_leaf,
            fraction,
        ),
        log_q_reverse_selection=-math.log(pair_count),
        log_q_reverse_auxiliary=_log_pair_beta_density_from_masses(
            problem,
            first_leaf,
            second_leaf,
            first_source_mass,
            second_source_mass,
        ),
        reduced_log_acceptance_ratio=(candidate.log_likelihood - source.log_likelihood),
    )


def propose_root_total_refresh(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    new_root_total: float,
) -> PosteriorTransitionTerms:
    """Scale every leaf mass using an independent Gamma-prior root proposal.

    Allocation shares remain unchanged.  Both the target and independence
    proposal use the same normalized Gamma density, so their root terms cancel
    exactly in the MH ratio.  The declared chart is ``(T, shares)`` and
    therefore has no root-scaling Jacobian.  An equivalent physical
    ``K``-mass density decomposition would instead pair its allocation-density
    change with ``(K-1) * log(T_new / T_old)``.

    Args:
        problem: Exact problem associated with ``source``.
        source: Source posterior state.
        new_root_total: Explicit proposed positive root total.

    Returns:
        Candidate and likelihood plus cancelling Gamma target/proposal terms.

    Raises:
        TypeError: If an object argument has the wrong type.
        ValueError: If ``source`` belongs to a different problem.
    """
    _validate_problem_state(problem, source)
    move: PosteriorMove = "root_total_refresh"
    try:
        _positive_finite(new_root_total, name="new_root_total")
    except (TypeError, ValueError):
        return _invalid_transition(source, move, "new root total lies outside support")
    candidate = rescale_full_tiling_root_total(
        problem,
        source,
        new_root_total=new_root_total,
    )
    return _valid_transition(
        source,
        candidate,
        move=move,
        delta_log_root_prior=candidate.log_root_prior - source.log_root_prior,
        log_q_forward_auxiliary=candidate.log_root_prior,
        log_q_reverse_auxiliary=source.log_root_prior,
    )


def propose_fixed_coefficient(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    proposal_stdev: float | None = None,
) -> PosteriorTransitionTerms:
    """Propose one fixed coefficient with a symmetric Gaussian random walk.

    Equal forward and reverse Gaussian densities cancel. When
    ``proposal_stdev`` is supplied, both normalized densities are reported
    explicitly; omitting it records their already-cancelled value as zero.
    The target retains the independently calibrated arithmetic-moment
    lognormal prior for the selected coefficient.

    Args:
        problem: Exact problem associated with ``source``.
        source: Source posterior state.
        coefficient_position: Zero-based fixed-design column position.
        proposed_coefficient: Explicit proposed positive coefficient.
        proposal_stdev: Optional finite positive symmetric random-walk
            standard deviation.

    Returns:
        Candidate and likelihood plus independent lognormal-prior change.

    Raises:
        TypeError: If object arguments or the position have the wrong type.
        ValueError: If ``source`` belongs to a different problem, the position
            is outside the fixed block, or ``proposal_stdev`` is invalid.
    """
    _validate_problem_state(problem, source)
    if isinstance(coefficient_position, (bool, np.bool_)) or not isinstance(
        coefficient_position,
        Integral,
    ):
        raise TypeError("coefficient_position must be an integer.")
    position = int(coefficient_position)
    if not 0 <= position < problem.n_fixed_coefficients:
        raise ValueError("coefficient_position must select a configured fixed column.")
    if proposal_stdev is None:
        stdev = None
    else:
        stdev = _positive_finite(proposal_stdev, name="proposal_stdev")
    move: PosteriorMove = "fixed_coefficient"
    try:
        coefficient = _positive_finite(
            proposed_coefficient,
            name="proposed_coefficient",
        )
    except (TypeError, ValueError):
        return _invalid_transition(source, move, "proposed coefficient lies outside support")
    coefficients = np.array(source.fixed_coefficients, dtype=np.float64, copy=True)
    change = coefficient - float(coefficients[position])
    coefficients[position] = coefficient
    fixed_block = problem.base.fixed_block
    if fixed_block is None:
        raise RuntimeError("a selected fixed coefficient requires a fixed block.")
    fixed_prediction = source.fixed_prediction + fixed_block.design[:, position] * change
    candidate = _assemble_state(
        problem,
        allocation=source.allocation,
        fixed_coefficients=coefficients,
        dynamic_prediction=source.dynamic_prediction,
        fixed_prediction=fixed_prediction,
    )
    log_q = (
        0.0
        if stdev is None
        else _normal_log_density(
            coefficient,
            mean=float(source.fixed_coefficients[position]),
            stdev=stdev,
        )
    )
    return _valid_transition(
        source,
        candidate,
        move=move,
        delta_log_fixed_coefficient_prior=(
            candidate.log_fixed_coefficient_prior - source.log_fixed_coefficient_prior
        ),
        log_q_forward_auxiliary=log_q,
        log_q_reverse_auxiliary=log_q,
    )


def accept_or_reject(
    source: FullTilingPosteriorState,
    transition: PosteriorTransitionTerms,
    *,
    log_uniform: float,
) -> FullTilingPosteriorState:
    """Apply the strict truncated-log MH decision to explicit terms.

    Args:
        source: State retained on rejection.
        transition: Deterministic candidate and MH accounting.
        log_uniform: Logarithm of a supplied draw in ``(0, 1]``; negative
            infinity is accepted as the limiting value.

    Returns:
        Candidate below the strict acceptance threshold, otherwise ``source``.

    Raises:
        TypeError: If source or transition has the wrong type.
        ValueError: If the candidate uses another problem, or ``log_uniform``
            is NaN or positive.
    """
    if not isinstance(source, FullTilingPosteriorState):
        raise TypeError("source must be a FullTilingPosteriorState.")
    if not isinstance(transition, PosteriorTransitionTerms):
        raise TypeError("transition must be a PosteriorTransitionTerms.")
    if transition.candidate.problem is not source.problem:
        raise ValueError("transition candidate and source must use the same problem.")
    log_uniform = float(log_uniform)
    if math.isnan(log_uniform) or log_uniform > 0.0:
        raise ValueError("log_uniform must be non-positive and cannot be NaN.")
    threshold = min(0.0, transition.log_acceptance_ratio)
    if transition.valid and log_uniform < threshold:
        return transition.candidate
    return source


def _require_problem(problem: FullTilingProblem) -> None:
    """Require the public full-tiling problem type."""
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")


def _validate_problem_allocation(
    problem: FullTilingProblem,
    allocation: TilingState,
) -> None:
    """Require a recursive allocation on the problem's exact grid."""
    _require_problem(problem)
    if not isinstance(allocation, TilingState):
        raise TypeError("allocation must be a TilingState.")
    if allocation.tiling.shape != problem.shape:
        raise ValueError("allocation and problem grid shapes must match.")


def _validate_problem_state(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> None:
    """Require a state built for the exact supplied problem instance."""
    _require_problem(problem)
    if not isinstance(state, FullTilingPosteriorState):
        raise TypeError("source must be a FullTilingPosteriorState.")
    if state.problem is not problem:
        raise ValueError("source must have been built for this exact problem.")


def _prepare_fixed_coefficients(
    problem: FullTilingProblem,
    fixed_coefficients: ArrayLike | None,
) -> FloatArray:
    """Own and validate coefficients against the optional fixed block."""
    if problem.base.fixed_block is None:
        if fixed_coefficients is None:
            return _readonly_float_array(
                np.empty(0, dtype=np.float64),
                name="fixed_coefficients",
            )
        result = _readonly_float_array(
            fixed_coefficients,
            name="fixed_coefficients",
        )
        if result.shape != (0,):
            raise ValueError("fixed_coefficients require a configured fixed block.")
        return result
    if fixed_coefficients is None:
        raise ValueError("fixed_coefficients are required for the configured fixed block.")
    result = _readonly_float_array(fixed_coefficients, name="fixed_coefficients")
    expected = (problem.base.fixed_block.n_coefficients,)
    if result.shape != expected:
        raise ValueError(f"fixed_coefficients must have shape {expected}.")
    if np.any(result <= 0.0):
        raise ValueError("fixed_coefficients must be strictly positive.")
    return result


def _log_allocation_share_prior(
    problem: FullTilingProblem,
    allocation: TilingState,
) -> float:
    """Return the normalized active Dirichlet density in share coordinates."""
    alphas = problem.allocation_prior.leaf_alphas(allocation.tiling)
    log_shares = _log_normalized_positive_masses(allocation.leaf_masses)
    return float(
        math.lgamma(problem.allocation_prior.concentration)
        - sum(math.lgamma(float(alpha)) for alpha in alphas)
        + np.dot(alphas - 1.0, log_shares)
    )


def _log_pair_beta_density(
    problem: FullTilingProblem,
    first_leaf: Rectangle,
    second_leaf: Rectangle,
    fraction: float,
) -> float:
    """Return the normalized additive-alpha Beta density for any leaf pair.

    Args:
        problem: Full-tiling problem supplying the additive alpha measure.
        first_leaf: Rectangle receiving ``fraction`` of the pair total.
        second_leaf: Rectangle receiving the complementary pair fraction.
        fraction: Open-unit first-leaf fraction.

    Returns:
        Normalized Beta log density.
    """
    first_alpha = problem.allocation_prior.alpha(first_leaf)
    second_alpha = problem.allocation_prior.alpha(second_leaf)
    return float(
        math.lgamma(first_alpha + second_alpha)
        - math.lgamma(first_alpha)
        - math.lgamma(second_alpha)
        + (first_alpha - 1.0) * math.log(fraction)
        + (second_alpha - 1.0) * math.log1p(-fraction)
    )


def _log_pair_beta_density_from_masses(
    problem: FullTilingProblem,
    first_leaf: Rectangle,
    second_leaf: Rectangle,
    first_mass: float,
    second_mass: float,
) -> float:
    """Return an arbitrary-pair Beta density from stable log masses.

    Args:
        problem: Full-tiling problem supplying the additive alpha measure.
        first_leaf: Rectangle associated with ``first_mass``.
        second_leaf: Rectangle associated with ``second_mass``.
        first_mass: Finite strictly positive first mass.
        second_mass: Finite strictly positive second mass.

    Returns:
        Normalized Beta log density without materializing either mass fraction.

    Raises:
        ValueError: If either mass is non-finite or not strictly positive.
    """
    return _log_beta_density_from_masses(
        problem.allocation_prior.alpha(first_leaf),
        problem.allocation_prior.alpha(second_leaf),
        first_mass,
        second_mass,
    )


def _log_fixed_coefficient_prior(
    problem: FullTilingProblem,
    coefficients: FloatArray,
) -> float:
    """Return normalized independent arithmetic-moment lognormal priors."""
    block = problem.base.fixed_block
    if block is None:
        return 0.0
    return float(
        sum(
            lognormal_coefficient_log_prior_numpy(
                coefficients[position : position + 1],
                1,
                float(mean),
                float(sd),
            )
            for position, (mean, sd) in enumerate(
                zip(
                    block.coefficient_prior_mean,
                    block.coefficient_prior_sd,
                    strict=True,
                )
            )
        )
    )


def _assemble_state(
    problem: FullTilingProblem,
    *,
    allocation: TilingState,
    fixed_coefficients: ArrayLike,
    dynamic_prediction: ArrayLike,
    fixed_prediction: ArrayLike,
) -> FullTilingPosteriorState:
    """Validate supplied prediction caches and assemble all posterior terms."""
    _validate_problem_allocation(problem, allocation)
    coefficients = _prepare_fixed_coefficients(problem, fixed_coefficients)
    dynamic = _readonly_float_array(
        dynamic_prediction,
        name="dynamic_prediction",
    )
    fixed = _readonly_float_array(fixed_prediction, name="fixed_prediction")
    expected = problem.observations.shape
    if dynamic.shape != expected or fixed.shape != expected:
        raise ValueError("prediction components must have the observation shape.")
    prediction = _readonly_float_array(dynamic + fixed, name="prediction")
    residual = _readonly_float_array(
        prediction - problem.observations,
        name="residual",
    )
    standardized = residual / problem.observation_sd
    log_gaussian = float(
        -0.5 * np.dot(standardized, standardized)
        - np.log(problem.observation_sd).sum()
        - 0.5 * problem.observations.size * _LOG_TWO_PI
    )
    likelihood_power = problem.base.likelihood_power
    log_likelihood = 0.0 if likelihood_power == 0.0 else float(likelihood_power * log_gaussian)
    return FullTilingPosteriorState(
        problem=problem,
        allocation=allocation,
        fixed_coefficients=coefficients,
        dynamic_prediction=dynamic,
        fixed_prediction=fixed,
        prediction=prediction,
        residual=residual,
        log_gaussian_likelihood=log_gaussian,
        log_likelihood=log_likelihood,
        log_root_prior=problem.base.prior.log_root_density(allocation.total_mass),
        log_allocation_prior=_log_allocation_share_prior(problem, allocation),
        log_fixed_coefficient_prior=_log_fixed_coefficient_prior(
            problem,
            coefficients,
        ),
    )


def _allocation_from_mass_map(
    tiling: LeafTiling,
    masses: dict[Rectangle, float],
) -> TilingState:
    """Build a canonical allocation from a complete rectangle-to-mass map."""
    return TilingState(
        tiling,
        np.asarray([masses[leaf] for leaf in tiling.leaves], dtype=np.float64),
    )


def _incremental_allocation_state(
    problem: FullTilingProblem,
    source: FullTilingPosteriorState,
    allocation: TilingState,
    *,
    removed: tuple[Rectangle, ...],
    added: tuple[Rectangle, ...],
) -> FullTilingPosteriorState:
    """Update the dynamic prediction using only explicitly changed rectangles."""
    dynamic = np.array(source.dynamic_prediction, dtype=np.float64, copy=True)
    for rectangle in removed:
        dynamic -= source.allocation.mass(rectangle) * problem.design_column(rectangle)
    for rectangle in added:
        dynamic += allocation.mass(rectangle) * problem.design_column(rectangle)
    return _assemble_state(
        problem,
        allocation=allocation,
        fixed_coefficients=source.fixed_coefficients,
        dynamic_prediction=dynamic,
        fixed_prediction=source.fixed_prediction,
    )


def _proposal_fraction(value: float) -> float | None:
    """Return a finite open-unit fraction, or ``None`` outside support."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        return None
    return fraction


def _normal_log_density(value: float, *, mean: float, stdev: float) -> float:
    """Return one normalized Gaussian log density."""
    standardized = (value - mean) / stdev
    return float(-0.5 * standardized * standardized - math.log(stdev) - 0.5 * _LOG_TWO_PI)


def _invalid_transition(
    source: FullTilingPosteriorState,
    move: PosteriorMove,
    reason: str,
) -> PosteriorTransitionTerms:
    """Return an invalid self-transition with neutral decomposed terms."""
    return PosteriorTransitionTerms(
        candidate=source,
        move=move,
        delta_log_likelihood=0.0,
        valid=False,
        reason=reason,
    )


def _with_reduced_acceptance(
    transition: PosteriorTransitionTerms,
    log_acceptance_ratio: float,
) -> PosteriorTransitionTerms:
    """Apply a builder-proved algebraic reduction to valid transition terms.

    Args:
        transition: Generic validated transition accounting.
        log_acceptance_ratio: Complete reduced log MH ratio.

    Returns:
        The same transition object with its derived aggregate ratio replaced.

    Raises:
        ValueError: If the transition is invalid or the ratio is NaN.
    """
    ratio = float(log_acceptance_ratio)
    if not transition.valid:
        raise ValueError("only valid transition terms can use a reduced ratio.")
    if math.isnan(ratio):
        raise ValueError("reduced log acceptance ratio cannot be NaN.")
    object.__setattr__(transition, "log_acceptance_ratio", ratio)
    return transition


def _valid_transition(
    source: FullTilingPosteriorState,
    candidate: FullTilingPosteriorState,
    *,
    move: PosteriorMove,
    delta_log_root_prior: float = 0.0,
    delta_log_allocation_prior: float = 0.0,
    delta_log_fixed_coefficient_prior: float = 0.0,
    log_q_forward_selection: float = 0.0,
    log_q_forward_auxiliary: float = 0.0,
    log_q_reverse_selection: float = 0.0,
    log_q_reverse_auxiliary: float = 0.0,
    log_jacobian: float = 0.0,
    reverse_merge_choice: MergeChoice | None = None,
    reverse_split_choice: SplitChoice | None = None,
    reduced_log_acceptance_ratio: float | None = None,
) -> PosteriorTransitionTerms:
    """Return complete terms for one valid deterministic candidate.

    Args:
        source: Source posterior state.
        candidate: Valid proposed posterior state.
        move: Concrete posterior proposal name.
        delta_log_root_prior: Candidate-minus-source root-prior density.
        delta_log_allocation_prior: Candidate-minus-source allocation-prior
            density.
        delta_log_fixed_coefficient_prior: Candidate-minus-source fixed-block
            prior density.
        log_q_forward_selection: Forward discrete-choice log probability.
        log_q_forward_auxiliary: Forward continuous-auxiliary log density.
        log_q_reverse_selection: Reverse discrete-choice log probability.
        log_q_reverse_auxiliary: Reverse continuous-auxiliary log density.
        log_jacobian: Log absolute augmented-coordinate Jacobian.
        reverse_merge_choice: Unique reverse merge for a structural proposal.
        reverse_split_choice: Unique reverse split for a structural proposal.
        reduced_log_acceptance_ratio: Builder-proved complete reduced log MH
            ratio, or ``None`` to retain generic decomposed accounting.
    Returns:
        Validated decomposed transition accounting.
    """
    transition = PosteriorTransitionTerms(
        candidate=candidate,
        move=move,
        delta_log_likelihood=candidate.log_likelihood - source.log_likelihood,
        delta_log_root_prior=delta_log_root_prior,
        delta_log_allocation_prior=delta_log_allocation_prior,
        delta_log_fixed_coefficient_prior=delta_log_fixed_coefficient_prior,
        log_q_forward_selection=log_q_forward_selection,
        log_q_forward_auxiliary=log_q_forward_auxiliary,
        log_q_reverse_selection=log_q_reverse_selection,
        log_q_reverse_auxiliary=log_q_reverse_auxiliary,
        log_jacobian=log_jacobian,
        reverse_merge_choice=reverse_merge_choice,
        reverse_split_choice=reverse_split_choice,
    )
    if reduced_log_acceptance_ratio is None:
        return transition
    return _with_reduced_acceptance(
        transition,
        reduced_log_acceptance_ratio,
    )


__all__ = [
    "FullTilingPosteriorState",
    "FullTilingProblem",
    "PosteriorTransitionTerms",
    "accept_or_reject",
    "build_full_tiling_posterior_state",
    "full_tiling_problem_from_gamma_beta_adapter",
    "initialize_full_tiling_posterior_state",
    "initialize_random_full_tiling_posterior_state",
    "log_root_total_slice_density",
    "propose_fixed_coefficient",
    "propose_pair_allocation_refresh",
    "propose_posterior_edge_flip",
    "propose_posterior_resolution_relocation",
    "propose_root_total_refresh",
    "rescale_full_tiling_root_total",
]
