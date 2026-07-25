"""PyMC/NumPyro NUTS reference model for one fixed full-tiling basis.

This module provides an independent continuous-sampler reference for the
experimental full-tiling target.  It freezes one canonical
:class:`~openghg_inversions.experimental.rjmcmc.full_tiling.LeafTiling` and
represents its allocation by a Gamma root total and Dirichlet leaf shares.
The resulting leaf masses drive the same direct rectangle design columns as
the NumPy full-tiling posterior.  Optional always-active coefficients retain
their arithmetic-moment lognormal priors.

PyMC, PyTensor, JAX, NumPyro, and ArviZ are imported only inside public
functions.  Importing this module therefore does not initialize either
computational backend.  Sampling deliberately requires PyTensor ``floatX`` to
be ``float64`` and JAX 64-bit mode to have been enabled by the caller's
environment.  The preflight reports, but never mutates, those process-global
settings.

This is a fixed-topology diagnostic model, not a reversible-jump kernel.  Its
posterior density is defined in ``(root_total, leaf_share,
fixed_coefficient)`` coordinates and excludes a structural probability that
is constant on the singleton topology.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
import math
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import lognormal_mu_sigma
from .full_tiling import LeafTiling
from .full_tiling_posterior import FullTilingPosteriorState, FullTilingProblem

if TYPE_CHECKING:
    from arviz import InferenceData
    from pymc import Model

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ConstrainedInitvals: TypeAlias = Mapping[str, Any] | Sequence[Mapping[str, Any] | None]
ChainMethod: TypeAlias = Literal["parallel", "vectorized"]

__all__ = [
    "FixedBasisNUTSData",
    "build_fixed_basis_pymc_model",
    "fixed_basis_nuts_initvals",
    "preflight_fixed_basis_nuts",
    "prepare_fixed_basis_nuts",
    "require_fixed_basis_nuts_float64",
    "sample_fixed_basis_nuts",
]


def _readonly_float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
) -> FloatArray:
    """Return an owned finite read-only ``float64`` array."""
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _readonly_int_array(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...],
) -> IntArray:
    """Return an owned read-only integer array with one required shape."""
    try:
        numeric = np.asarray(values, dtype=np.float64)
        result = np.array(values, dtype=np.int64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer values.") from error
    if not np.all(np.isfinite(numeric)) or not np.array_equal(
        numeric,
        result.astype(np.float64),
    ):
        raise ValueError(f"{name} must contain exact integer values.")
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    result.setflags(write=False)
    return result


def _positive_float(value: object, *, name: str) -> float:
    """Return one finite, strictly positive built-in float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class FixedBasisNUTSData:
    """Immutable fixed-basis data and prior bridge for PyMC.

    Arrays are owned, converted to ``float64`` (or ``int64`` for rectangle
    bounds), and made read-only.  Leaves, design columns, prior parameters,
    and initial coordinates all use the canonical order of ``tiling.leaves``.

    Args:
        tiling: Frozen canonical recursive-bisection tiling.
        rectangle_bounds: Canonical ``(K, 4)`` half-open rectangle bounds.
        observations: Observation vector in the model's concentration units.
        observation_sd: Positive independent Gaussian standard deviations in
            the same concentration units.
        dynamic_design: Concentration response per unit normalized leaf mass,
            with shape ``(n_observations, K)``.
        nominal_leaf_share: Positive dimensionless nominal mass shares used to
            report leaf scaling factors.
        dirichlet_alpha: Positive Dirichlet shapes for canonical leaf shares.
        root_shape: Gamma shape for the dimensionless root scaling.
        root_rate: Gamma rate, not scale, for the dimensionless root scaling.
        fixed_design: Concentration response per unit fixed-coefficient
            scaling, with shape ``(n_observations, n_fixed)``.
        fixed_offset: Coefficient-independent contribution in concentration
            units.
        fixed_coefficient_prior_mean: Positive arithmetic lognormal means.
        fixed_coefficient_prior_sd: Positive arithmetic lognormal standard
            deviations.
        initial_root_total: Constrained initial dimensionless root scaling.
        initial_leaf_share: Constrained initial canonical simplex shares.
        initial_fixed_coefficient: Constrained initial dimensionless fixed
            coefficients.
        likelihood_power: Gaussian likelihood multiplier.  Only exactly one
            is currently supported.

    Raises:
        TypeError: If ``tiling`` or a scalar has the wrong type.
        ValueError: If shapes, supports, ordering metadata, or initial
            coordinates are inconsistent.
    """

    tiling: LeafTiling
    rectangle_bounds: IntArray
    observations: FloatArray
    observation_sd: FloatArray
    dynamic_design: FloatArray
    nominal_leaf_share: FloatArray
    dirichlet_alpha: FloatArray
    root_shape: float
    root_rate: float
    fixed_design: FloatArray
    fixed_offset: FloatArray
    fixed_coefficient_prior_mean: FloatArray
    fixed_coefficient_prior_sd: FloatArray
    initial_root_total: float
    initial_leaf_share: FloatArray
    initial_fixed_coefficient: FloatArray
    likelihood_power: float

    def __post_init__(self) -> None:
        """Own and validate the complete fixed-topology model description."""
        if not isinstance(self.tiling, LeafTiling):
            raise TypeError("tiling must be a LeafTiling.")
        k = self.tiling.k
        bounds = _readonly_int_array(
            self.rectangle_bounds,
            name="rectangle_bounds",
            shape=(k, 4),
        )
        expected_bounds = np.asarray(
            [
                (
                    leaf.row_start,
                    leaf.row_stop,
                    leaf.col_start,
                    leaf.col_stop,
                )
                for leaf in self.tiling.leaves
            ],
            dtype=np.int64,
        )
        if not np.array_equal(bounds, expected_bounds):
            raise ValueError("rectangle_bounds must follow canonical tiling leaf order.")

        observations = _readonly_float_array(self.observations, name="observations", ndim=1)
        observation_sd = _readonly_float_array(
            self.observation_sd,
            name="observation_sd",
            ndim=1,
        )
        dynamic_design = _readonly_float_array(
            self.dynamic_design,
            name="dynamic_design",
            ndim=2,
        )
        nominal_share = _readonly_float_array(
            self.nominal_leaf_share,
            name="nominal_leaf_share",
            ndim=1,
        )
        alpha = _readonly_float_array(
            self.dirichlet_alpha,
            name="dirichlet_alpha",
            ndim=1,
        )
        fixed_design = _readonly_float_array(
            self.fixed_design,
            name="fixed_design",
            ndim=2,
        )
        fixed_offset = _readonly_float_array(
            self.fixed_offset,
            name="fixed_offset",
            ndim=1,
        )
        fixed_mean = _readonly_float_array(
            self.fixed_coefficient_prior_mean,
            name="fixed_coefficient_prior_mean",
            ndim=1,
        )
        fixed_sd = _readonly_float_array(
            self.fixed_coefficient_prior_sd,
            name="fixed_coefficient_prior_sd",
            ndim=1,
        )
        initial_share = _readonly_float_array(
            self.initial_leaf_share,
            name="initial_leaf_share",
            ndim=1,
        )
        initial_fixed = _readonly_float_array(
            self.initial_fixed_coefficient,
            name="initial_fixed_coefficient",
            ndim=1,
        )

        n_observations = observations.size
        n_fixed = fixed_design.shape[1]
        if n_observations < 1:
            raise ValueError("observations cannot be empty.")
        if observation_sd.shape != observations.shape or np.any(observation_sd <= 0.0):
            raise ValueError("observation_sd must be positive with the observation shape.")
        if dynamic_design.shape != (n_observations, k):
            raise ValueError("dynamic_design must have shape (n_observations, K).")
        if nominal_share.shape != (k,) or np.any(nominal_share <= 0.0):
            raise ValueError("nominal_leaf_share must contain one positive value per leaf.")
        if not np.isclose(float(nominal_share.sum()), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("nominal_leaf_share must sum to one.")
        if alpha.shape != (k,) or np.any(alpha <= 0.0):
            raise ValueError("dirichlet_alpha must contain one positive shape per leaf.")
        if fixed_design.shape[0] != n_observations:
            raise ValueError("fixed_design must have one row per observation.")
        if fixed_offset.shape != observations.shape:
            raise ValueError("fixed_offset must have the observation shape.")
        if fixed_mean.shape != (n_fixed,) or fixed_sd.shape != (n_fixed,):
            raise ValueError("fixed coefficient prior moments must match fixed_design.")
        if np.any(fixed_mean <= 0.0) or np.any(fixed_sd <= 0.0):
            raise ValueError("fixed coefficient prior moments must be strictly positive.")
        if initial_share.shape != (k,) or np.any(initial_share <= 0.0):
            raise ValueError("initial_leaf_share must contain one positive value per leaf.")
        if not np.isclose(float(initial_share.sum()), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("initial_leaf_share must sum to one.")
        if initial_fixed.shape != (n_fixed,) or np.any(initial_fixed <= 0.0):
            raise ValueError("initial_fixed_coefficient must be positive and match fixed_design.")

        root_shape = _positive_float(self.root_shape, name="root_shape")
        root_rate = _positive_float(self.root_rate, name="root_rate")
        initial_root = _positive_float(self.initial_root_total, name="initial_root_total")
        likelihood_power = float(self.likelihood_power)
        if likelihood_power != 1.0:
            raise ValueError("fixed-basis NUTS currently requires likelihood_power == 1.0.")

        for name, value in (
            ("rectangle_bounds", bounds),
            ("observations", observations),
            ("observation_sd", observation_sd),
            ("dynamic_design", dynamic_design),
            ("nominal_leaf_share", nominal_share),
            ("dirichlet_alpha", alpha),
            ("fixed_design", fixed_design),
            ("fixed_offset", fixed_offset),
            ("fixed_coefficient_prior_mean", fixed_mean),
            ("fixed_coefficient_prior_sd", fixed_sd),
            ("initial_leaf_share", initial_share),
            ("initial_fixed_coefficient", initial_fixed),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "root_shape", root_shape)
        object.__setattr__(self, "root_rate", root_rate)
        object.__setattr__(self, "initial_root_total", initial_root)
        object.__setattr__(self, "likelihood_power", likelihood_power)

    @property
    def k(self) -> int:
        """Return the fixed number of canonical leaves."""
        return self.tiling.k

    @property
    def n_fixed_coefficients(self) -> int:
        """Return the number of always-active coefficients."""
        return int(self.fixed_design.shape[1])

    @property
    def leaf_labels(self) -> tuple[str, ...]:
        """Return stable human-readable labels in canonical leaf order."""
        return tuple(
            f"r{row_start}:{row_stop}_c{col_start}:{col_stop}"
            for row_start, row_stop, col_start, col_stop in self.rectangle_bounds
        )

    @property
    def fixed_lognormal_mu_sigma(self) -> tuple[FloatArray, FloatArray]:
        """Return normal-space parameters converted from arithmetic moments."""
        pairs = tuple(
            lognormal_mu_sigma(float(mean), float(sd))
            for mean, sd in zip(
                self.fixed_coefficient_prior_mean,
                self.fixed_coefficient_prior_sd,
                strict=True,
            )
        )
        if not pairs:
            empty = np.empty(0, dtype=np.float64)
            empty.setflags(write=False)
            return empty, empty
        mu, sigma = (np.asarray(values, dtype=np.float64) for values in zip(*pairs, strict=True))
        mu.setflags(write=False)
        sigma.setflags(write=False)
        return mu, sigma


def prepare_fixed_basis_nuts(
    problem: FullTilingProblem,
    initial_state: FullTilingPosteriorState,
) -> FixedBasisNUTSData:
    """Freeze one full-tiling posterior state into a NUTS data bridge.

    The exact problem identity is required; accepting a state constructed for
    an equal-looking problem could silently pair its topology with different
    observation arrays.  Dynamic columns are generated directly for every
    canonical rectangle and copied into one dense matrix suitable for JAX.

    Args:
        problem: Full-tiling problem supplying data and priors.
        initial_state: Valid state supplying the fixed topology and constrained
            initial coordinates.

    Returns:
        Immutable, backend-independent fixed-basis model data.

    Raises:
        TypeError: If either argument has the wrong public type.
        ValueError: If the state belongs to another problem, the likelihood is
            powered, or model arrays are inconsistent.
        RuntimeError: If a validated base problem unexpectedly lacks its
            normalized fixed offset.
    """
    if not isinstance(problem, FullTilingProblem):
        raise TypeError("problem must be a FullTilingProblem.")
    if not isinstance(initial_state, FullTilingPosteriorState):
        raise TypeError("initial_state must be a FullTilingPosteriorState.")
    if initial_state.problem is not problem:
        raise ValueError("initial_state must have been built for this exact problem.")
    if problem.base.likelihood_power != 1.0:
        raise ValueError("fixed-basis NUTS currently requires likelihood_power == 1.0.")

    tiling = initial_state.allocation.tiling
    leaves = tiling.leaves
    dynamic_design = np.column_stack([problem.design_column(rectangle) for rectangle in leaves])
    nominal_share = np.asarray(
        [problem.rectangle_nominal_mass(rectangle) for rectangle in leaves],
        dtype=np.float64,
    )
    alpha = problem.allocation_prior.leaf_alphas(tiling)
    rectangle_bounds = np.asarray(
        [
            (
                rectangle.row_start,
                rectangle.row_stop,
                rectangle.col_start,
                rectangle.col_stop,
            )
            for rectangle in leaves
        ],
        dtype=np.int64,
    )
    fixed_block = problem.base.fixed_block
    if fixed_block is None:
        fixed_design = np.empty((problem.observations.size, 0), dtype=np.float64)
        fixed_mean = np.empty(0, dtype=np.float64)
        fixed_sd = np.empty(0, dtype=np.float64)
    else:
        fixed_design = fixed_block.design
        fixed_mean = fixed_block.coefficient_prior_mean
        fixed_sd = fixed_block.coefficient_prior_sd

    initial_root = initial_state.root_total
    initial_share = np.asarray(
        initial_state.leaf_masses / initial_root,
        dtype=np.float64,
    )
    fixed_offset = problem.base.fixed_offset
    if fixed_offset is None:
        raise RuntimeError("validated GammaBetaTreeProblem has no fixed offset.")
    return FixedBasisNUTSData(
        tiling=tiling,
        rectangle_bounds=rectangle_bounds,
        observations=problem.observations,
        observation_sd=problem.observation_sd,
        dynamic_design=dynamic_design,
        nominal_leaf_share=nominal_share,
        dirichlet_alpha=alpha,
        root_shape=problem.base.prior.root_shape,
        root_rate=problem.base.prior.root_rate,
        fixed_design=fixed_design,
        fixed_offset=fixed_offset,
        fixed_coefficient_prior_mean=fixed_mean,
        fixed_coefficient_prior_sd=fixed_sd,
        initial_root_total=initial_root,
        initial_leaf_share=initial_share,
        initial_fixed_coefficient=initial_state.fixed_coefficients,
        likelihood_power=problem.base.likelihood_power,
    )


def fixed_basis_nuts_initvals(data: FixedBasisNUTSData) -> dict[str, object]:
    """Return owned constrained PyMC initial values from the bridge state.

    Args:
        data: Validated fixed-basis bridge.

    Returns:
        Mapping for all free random variables.  ``fixed_coefficient`` is
        omitted when the model has no fixed design block.

    Raises:
        TypeError: If ``data`` is not :class:`FixedBasisNUTSData`.
    """
    if not isinstance(data, FixedBasisNUTSData):
        raise TypeError("data must be FixedBasisNUTSData.")
    result: dict[str, object] = {
        "root_total": float(data.initial_root_total),
        "leaf_share": np.array(data.initial_leaf_share, copy=True),
    }
    if data.n_fixed_coefficients:
        result["fixed_coefficient"] = np.array(
            data.initial_fixed_coefficient,
            copy=True,
        )
    return result


def _import_runtime() -> tuple[Any, Any, Any, Any, Any]:
    """Import the optional computational stack with one actionable failure."""
    try:
        import arviz as az
        import jax
        import numpyro
        import pymc as pm
        import pytensor
    except ImportError as error:
        raise ImportError(
            "Fixed-basis NUTS requires PyMC, PyTensor, JAX, NumPyro, and ArviZ; "
            "use the repository Pixi environment."
        ) from error
    return pm, pytensor, jax, numpyro, az


def require_fixed_basis_nuts_float64() -> dict[str, str | bool]:
    """Require an already configured all-``float64`` PyMC/JAX runtime.

    This function is intentionally read-only.  Configure JAX before process
    startup, for example with ``JAX_ENABLE_X64=1``; changing it after JAX has
    initialized can invalidate compiled-cache and reproducibility assumptions.

    Returns:
        Version and precision metadata for provenance.

    Raises:
        ImportError: If any required computational dependency is unavailable.
        RuntimeError: If PyTensor ``floatX`` is not ``float64``, JAX 64-bit
            mode is disabled, or JAX did not select the CPU backend.
    """
    pm, pytensor, jax, numpyro, az = _import_runtime()
    pytensor_float_x = str(pytensor.config.floatX)
    jax_x64 = bool(jax.config.jax_enable_x64)
    jax_backend = str(jax.default_backend())
    if pytensor_float_x != "float64":
        raise RuntimeError(
            f"PyTensor floatX must be float64 for the fixed-basis NUTS reference; found {pytensor_float_x!r}."
        )
    if not jax_x64:
        raise RuntimeError(
            "JAX 64-bit mode must be enabled before import for fixed-basis NUTS; "
            "set JAX_ENABLE_X64=1 in the launch environment."
        )
    if jax_backend != "cpu":
        raise RuntimeError(
            f"The fixed-basis NUTS reference currently requires the JAX CPU backend; found {jax_backend!r}."
        )
    return {
        "pymc_version": str(pm.__version__),
        "pytensor_version": str(pytensor.__version__),
        "jax_version": str(jax.__version__),
        "numpyro_version": str(numpyro.__version__),
        "arviz_version": str(az.__version__),
        "pytensor_floatX": pytensor_float_x,
        "jax_enable_x64": jax_x64,
        "jax_backend": jax_backend,
    }


def build_fixed_basis_pymc_model(data: FixedBasisNUTSData) -> Model:
    """Build the exact fixed-topology Gamma--Dirichlet PyMC model.

    Args:
        data: Immutable model data in canonical leaf order.

    Returns:
        PyMC model with constrained scientific variables ``root_total``,
        ``leaf_share``, ``leaf_mass``, ``leaf_scaling``,
        ``fixed_coefficient``, ``mean_observation``, and ``observed``.

    Raises:
        TypeError: If ``data`` has the wrong type.
        ImportError: If the PyMC/JAX runtime is unavailable.
        RuntimeError: If the runtime is not configured for ``float64``.
    """
    if not isinstance(data, FixedBasisNUTSData):
        raise TypeError("data must be FixedBasisNUTSData.")
    require_fixed_basis_nuts_float64()
    pm, _, _, _, _ = _import_runtime()
    # PyTensor exposes these runtime operations from ``pytensor.tensor``, but
    # its current typing metadata does not re-export them consistently.
    pt: Any = import_module("pytensor.tensor")

    coords = {
        "observation": np.arange(data.observations.size, dtype=np.int64),
        "leaf": data.leaf_labels,
        "fixed": tuple(f"fixed_{position}" for position in range(data.n_fixed_coefficients)),
    }
    initvals = fixed_basis_nuts_initvals(data)
    fixed_mu, fixed_sigma = data.fixed_lognormal_mu_sigma
    with pm.Model(coords=coords) as model:
        root_total = pm.Gamma(
            "root_total",
            alpha=np.float64(data.root_shape),
            beta=np.float64(data.root_rate),
            initval=initvals["root_total"],
            dtype="float64",
        )
        leaf_share = pm.Dirichlet(
            "leaf_share",
            a=np.asarray(data.dirichlet_alpha, dtype=np.float64),
            dims="leaf",
            initval=initvals["leaf_share"],
            dtype="float64",
        )
        leaf_mass = pm.Deterministic(
            "leaf_mass",
            root_total * leaf_share,
            dims="leaf",
        )
        pm.Deterministic(
            "leaf_scaling",
            leaf_mass / pt.as_tensor_variable(data.nominal_leaf_share),
            dims="leaf",
        )
        if data.n_fixed_coefficients:
            fixed_coefficient = pm.LogNormal(
                "fixed_coefficient",
                mu=np.asarray(fixed_mu, dtype=np.float64),
                sigma=np.asarray(fixed_sigma, dtype=np.float64),
                dims="fixed",
                initval=initvals["fixed_coefficient"],
                dtype="float64",
            )
            fixed_prediction = pt.dot(
                pt.as_tensor_variable(data.fixed_design),
                fixed_coefficient,
            )
        else:
            fixed_coefficient = pm.Deterministic(
                "fixed_coefficient",
                pt.zeros((0,), dtype="float64"),
                dims="fixed",
            )
            fixed_prediction = pt.zeros_like(
                pt.as_tensor_variable(data.fixed_offset),
                dtype="float64",
            )
        mean_observation = pm.Deterministic(
            "mean_observation",
            pt.as_tensor_variable(data.fixed_offset)
            + pt.dot(pt.as_tensor_variable(data.dynamic_design), leaf_mass)
            + fixed_prediction,
            dims="observation",
        )
        pm.Normal(
            "observed",
            mu=mean_observation,
            sigma=pt.as_tensor_variable(data.observation_sd),
            observed=pt.as_tensor_variable(data.observations),
            dims="observation",
            dtype="float64",
        )
    return model


def preflight_fixed_basis_nuts(
    data: FixedBasisNUTSData,
    model: Model,
    *,
    initvals: ConstrainedInitvals | None = None,
    expected_log_target: float | None = None,
) -> dict[str, float | int | str | bool]:
    """Validate precision, constrained initialization, and optional target parity.

    ``model.compile_logp(jacobian=False)`` evaluates the scientific constrained
    density after PyMC has transformed the supplied constrained initial
    values.  It can therefore be compared with
    :attr:`FullTilingPosteriorState.log_target`; transformed-chart sampler
    statistics such as ``sample_stats.lp`` cannot.

    Args:
        data: Immutable bridge used to build ``model``.
        model: PyMC model returned by :func:`build_fixed_basis_pymc_model`.
        initvals: Optional constrained initial-value mapping.  For a per-chain
            sequence, the first mapping is used for this scalar preflight.
        expected_log_target: Optional independent constrained-coordinate log
            target to check.

    Returns:
        Precision/version metadata plus the compiled constrained log target
        and, when supplied, its difference from ``expected_log_target``.

    Raises:
        TypeError: If arguments have incompatible public types.
        ValueError: If initial values or expected target are malformed, or
            target parity fails a strict numerical tolerance.
        ImportError: If the computational runtime is unavailable.
        RuntimeError: If runtime precision is not ``float64``.
    """
    if not isinstance(data, FixedBasisNUTSData):
        raise TypeError("data must be FixedBasisNUTSData.")
    metadata: dict[str, float | int | str | bool] = dict(require_fixed_basis_nuts_float64())
    pm, _, _, _, _ = _import_runtime()
    if not isinstance(model, pm.Model):
        raise TypeError("model must be a PyMC Model.")
    non_float64 = tuple(
        f"{variable.name}:{variable.dtype}"
        for variable in model.value_vars
        if str(variable.dtype) != "float64"
    )
    if non_float64:
        raise RuntimeError("Every PyMC value variable must be float64; found " + ", ".join(non_float64) + ".")
    metadata["model_value_variables_float64"] = True
    metadata["model_value_variable_count"] = len(model.value_vars)

    selected_initvals: Mapping[str, Any]
    supplied = fixed_basis_nuts_initvals(data) if initvals is None else initvals
    if isinstance(supplied, Mapping):
        selected_initvals = supplied
    elif isinstance(supplied, Sequence) and supplied:
        selected = supplied[0]
        if selected is None or not isinstance(selected, Mapping):
            raise ValueError("the first per-chain initvals entry must be a mapping.")
        selected_initvals = selected
    else:
        raise TypeError("initvals must be a mapping or non-empty sequence of mappings.")

    point_fn = pm.initial_point.make_initial_point_fn(
        model=model,
        overrides=dict(selected_initvals),
        jitter_rvs=set(),
        default_strategy="support_point",
        return_transformed=True,
    )
    point = point_fn(0)
    compiled_logp = float(model.compile_logp(jacobian=False)(point))
    if not math.isfinite(compiled_logp):
        raise ValueError("the constrained initial values have a non-finite model log target.")
    metadata["constrained_log_target"] = compiled_logp

    if expected_log_target is not None:
        expected = float(expected_log_target)
        if not math.isfinite(expected):
            raise ValueError("expected_log_target must be finite.")
        difference = compiled_logp - expected
        tolerance = 5.0e-10 * max(1.0, abs(expected))
        metadata["expected_log_target"] = expected
        metadata["log_target_difference"] = difference
        metadata["log_target_absolute_tolerance"] = tolerance
        if abs(difference) > tolerance:
            raise ValueError(
                "PyMC constrained log target does not match the independent "
                f"target: difference={difference:.17g}, tolerance={tolerance:.17g}."
            )
    return metadata


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def sample_fixed_basis_nuts(
    model: Model,
    data: FixedBasisNUTSData,
    *,
    draws: int,
    tune: int,
    seed: int,
    target_accept: float,
    chains: int,
    cores: int,
    chain_method: ChainMethod,
    progressbar: bool,
    max_tree_depth: int = 10,
    dense_mass: bool = False,
    initvals: ConstrainedInitvals | None = None,
) -> InferenceData:
    """Run NumPyro NUTS for the fixed-basis PyMC model.

    The primary HPC usage is one chain per array task with
    ``chains=cores=1`` and ``chain_method='parallel'``.  Jitter is disabled
    so supplied constrained starts are reproducible and auditable.

    Args:
        model: PyMC model built from ``data``.
        data: Immutable bridge associated with ``model``.
        draws: Positive retained post-warmup draw count.
        tune: Positive warmup/adaptation draw count.
        seed: Non-negative integer random seed passed directly to PyMC.
        target_accept: Open-unit NUTS target acceptance probability.
        chains: Positive number of chains sampled in this process.
        cores: Positive PyMC worker count.
        chain_method: NumPyro chain execution strategy.
        progressbar: Whether PyMC/NumPyro should show progress.
        max_tree_depth: Positive NumPyro NUTS tree-depth limit.
        dense_mass: Whether NumPyro should adapt a dense mass matrix.
        initvals: Optional constrained mapping or per-chain mappings.  Defaults
            to the bridge state's constrained coordinates.

    Returns:
        ArviZ ``InferenceData`` containing posterior, sample statistics,
        observed data, and pointwise log likelihood.

    Raises:
        TypeError: If arguments have incompatible types.
        ValueError: If numeric controls lie outside support.
        ImportError: If the PyMC/NumPyro runtime is unavailable.
        RuntimeError: If runtime precision is not ``float64`` or PyMC does not
            return ``InferenceData``.
    """
    if not isinstance(data, FixedBasisNUTSData):
        raise TypeError("data must be FixedBasisNUTSData.")
    require_fixed_basis_nuts_float64()
    pm, _, _, _, az = _import_runtime()
    if not isinstance(model, pm.Model):
        raise TypeError("model must be a PyMC Model.")
    draws = _positive_integer(draws, name="draws")
    tune = _positive_integer(tune, name="tune")
    chains = _positive_integer(chains, name="chains")
    cores = _positive_integer(cores, name="cores")
    max_tree_depth = _positive_integer(max_tree_depth, name="max_tree_depth")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    target_accept = float(target_accept)
    if not math.isfinite(target_accept) or not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must lie strictly between zero and one.")
    if chain_method not in ("parallel", "vectorized"):
        raise ValueError("chain_method must be 'parallel' or 'vectorized'.")
    if not isinstance(progressbar, bool):
        raise TypeError("progressbar must be Boolean.")
    if not isinstance(dense_mass, bool):
        raise TypeError("dense_mass must be Boolean.")

    selected_initvals = fixed_basis_nuts_initvals(data) if initvals is None else initvals
    with model:
        result = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=progressbar,
            nuts_sampler="numpyro",
            initvals=selected_initvals,
            target_accept=target_accept,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            nuts_sampler_kwargs={
                "jitter": False,
                "chain_method": chain_method,
                "nuts_kwargs": {
                    "max_tree_depth": max_tree_depth,
                    "dense_mass": dense_mass,
                },
            },
        )
    if not isinstance(result, az.InferenceData):
        raise RuntimeError("PyMC did not return an ArviZ InferenceData result.")
    return result
