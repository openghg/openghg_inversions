"""Accepted-sigma cached sampling for the fixed-OU CO2 likelihood.

The compound sweep is deliberately ordered ``sigma -> state``.  A stock PyMC
NUTS step samples site amplitudes against the exact conditional likelihood,
then refreshes the accepted state quadratic consumed by the following stock
state NUTS step.  State leapfrogs therefore do not refactor the observation
covariance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.blocking import PointType, StatsType
from pymc.model import modelcontext
from pymc.step_methods.compound import BlockedStep, Competence, StepMethodState
from pymc.step_methods.hmc.quadpotential import QuadPotentialDiagAdapt
from pymc.step_methods.state import dataclass_state
from pymc.util import RandomGenerator, get_random_generator, get_value_vars_from_user_vars
from pytensor.gradient import DisconnectedType, grad_not_implemented
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op

from openghg_inversions.models.cached_sigma import (
    FixedOuCachedSigmaTarget,
    MarginalQuadraticCache,
)


FloatArray = NDArray[np.float64]

CACHED_SIGMA_SAMPLER_METADATA: Mapping[str, Any] = {
    "sampler": "accepted_sigma_cached_compound",
    "step_order": ("sigma_site_nuts", "state_nuts"),
    "sigma_step": "stock_pymc_nuts_exact_conditional",
    "state_step": "stock_pymc_nuts_cached_quadratic",
    "verification_games_source": (
        "src/verification_games/rhime_calibration/cached_sigma_step.py"
        "@51aaeb101a5d8daf57b1dcf43ea1716c850fc21c"
    ),
}


class PytensorMarginalQuadraticCache:
    """Mutable float32 coefficients read by the state-only PyTensor graph."""

    def __init__(self, initial: MarginalQuadraticCache) -> None:
        self.constant = pytensor.shared(
            np.asarray(initial.constant, dtype=np.float32),
            name="cached_marginal_constant",
        )
        self.linear = pytensor.shared(
            np.asarray(initial.linear, dtype=np.float32),
            name="cached_marginal_linear",
        )
        self.precision = pytensor.shared(
            np.asarray(initial.precision, dtype=np.float32),
            name="cached_marginal_precision",
        )

    @property
    def n_state(self) -> int:
        """Return the represented physical-state dimension."""
        return int(self.linear.get_value(borrow=True).size)

    def log_likelihood(self, state: Any) -> Any:
        """Return the normalized cached quadratic as a PyTensor scalar."""
        return (
            self.constant
            + pt.dot(self.linear, state)
            - np.float32(0.5) * pt.dot(state, pt.dot(self.precision, state))
        )

    def update(self, cache: MarginalQuadraticCache) -> None:
        """Install one complete accepted-sigma cache between compound steps."""
        if cache.n_state != self.n_state:
            raise ValueError(
                f"Cache has {cache.n_state} states, expected {self.n_state}."
            )
        self.constant.set_value(np.asarray(cache.constant, dtype=np.float32))
        self.linear.set_value(np.asarray(cache.linear, dtype=np.float32))
        self.precision.set_value(np.asarray(cache.precision, dtype=np.float32))


@dataclass(frozen=True)
class SigmaLikelihoodDiagnostics:
    """Exact conditional evaluations accumulated during one NUTS transition."""

    likelihood_evaluations: int
    gradient_evaluations: int
    factor_cholesky_operations: int
    factor_cholesky_seconds: float
    evaluation_seconds: float


def _rejection_gradient_sigma(sigma: FloatArray) -> FloatArray:
    """Return a transform-safe physical gradient for an invalid proposal."""
    gradient = np.full(sigma.shape, -1.0, dtype=np.float64)
    finite_large = np.isfinite(sigma) & (sigma > 1.0)
    gradient[finite_large] = -np.reciprocal(sigma[finite_large])
    return gradient


class _PytensorSigmaLikelihoodOp(Op):
    """Exact sigma-conditional likelihood with a chain-local fixed residual."""

    def __init__(self, target: FixedOuCachedSigmaTarget) -> None:
        self.target = target
        self.residual = pytensor.shared(
            np.zeros(target.n_obs, dtype=np.float64),
            name="sigma_conditional_residual",
        )
        self.reset_diagnostics()

    def make_node(self, sigma: Any, residual: Any) -> Apply:
        sigma_variable = pt.as_tensor_variable(sigma)
        residual_variable = pt.as_tensor_variable(residual)
        if sigma_variable.ndim != 1:
            raise TypeError("sigma must be a one-dimensional PyTensor variable.")
        if residual_variable.ndim != 1:
            raise TypeError("residual must be a one-dimensional PyTensor variable.")
        return Apply(
            self,
            [sigma_variable, residual_variable],
            [pt.dscalar(), pt.dvector()],
        )

    def perform(
        self,
        node: Apply,
        inputs: list[NDArray[Any]],
        output_storage: list[list[NDArray[Any] | None]],
    ) -> None:
        del node
        started = time.perf_counter()
        sigma = np.asarray(inputs[0], dtype=np.float64)
        residual = np.asarray(inputs[1], dtype=np.float64)
        self.likelihood_evaluations += 1
        self.gradient_evaluations += 1
        if sigma.shape != (self.target.n_group,):
            raise ValueError(
                f"sigma has shape {sigma.shape}, expected {(self.target.n_group,)}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            sigma_squared = np.square(sigma)
        invalid = bool(
            not np.isfinite(sigma).all()
            or np.any(sigma < 0.0)
            or not np.isfinite(sigma_squared).all()
        )
        if invalid:
            output_storage[0][0] = np.asarray(-np.inf, dtype=np.float64)
            output_storage[1][0] = _rejection_gradient_sigma(sigma)
        else:
            evaluation = self.target.evaluate_from_residual(residual, sigma)
            output_storage[0][0] = np.asarray(
                evaluation.log_likelihood,
                dtype=np.float64,
            )
            output_storage[1][0] = np.asarray(
                evaluation.gradient_site_amplitude,
                dtype=np.float64,
            )
            self.factor_cholesky_operations += int(
                self.target.prepared.rank > 0
                and np.isfinite(evaluation.log_likelihood)
            )
            self.factor_cholesky_seconds += evaluation.factor_cholesky_seconds
        self.evaluation_seconds += time.perf_counter() - started

    def L_op(
        self,
        inputs: list[Variable],
        outputs: list[Variable],
        output_grads: list[Variable],
    ) -> list[Variable]:
        if not isinstance(output_grads[1].type, DisconnectedType):
            return [
                grad_not_implemented(self, 0, inputs[0]),
                grad_not_implemented(self, 1, inputs[1]),
            ]
        return [
            output_grads[0] * outputs[1],
            grad_not_implemented(self, 1, inputs[1]),
        ]

    def infer_shape(
        self,
        fgraph: Any,
        node: Apply,
        input_shapes: list[tuple[Any, ...]],
    ) -> list[tuple[Any, ...]]:
        del fgraph, node
        return [(), input_shapes[0]]

    def install_residual(self, residual: ArrayLike) -> None:
        """Set the exact residual used for the next conditional trajectory."""
        value = np.asarray(residual, dtype=np.float64)
        if value.shape != (self.target.n_obs,):
            raise ValueError(
                f"residual has shape {value.shape}, expected {(self.target.n_obs,)}."
            )
        if not np.isfinite(value).all():
            raise ValueError("residual must contain only finite values.")
        self.residual.set_value(value.copy())

    def reset_diagnostics(self) -> None:
        """Reset counters immediately before one delegated NUTS transition."""
        self.likelihood_evaluations = 0
        self.gradient_evaluations = 0
        self.factor_cholesky_operations = 0
        self.factor_cholesky_seconds = 0.0
        self.evaluation_seconds = 0.0

    @property
    def diagnostics(self) -> SigmaLikelihoodDiagnostics:
        """Return an immutable snapshot of current counters."""
        return SigmaLikelihoodDiagnostics(
            likelihood_evaluations=self.likelihood_evaluations,
            gradient_evaluations=self.gradient_evaluations,
            factor_cholesky_operations=self.factor_cholesky_operations,
            factor_cholesky_seconds=self.factor_cholesky_seconds,
            evaluation_seconds=self.evaluation_seconds,
        )


@dataclass_state
class _CachedSigmaNutsState(StepMethodState):
    """Chain-local state including the delegated stock PyMC NUTS state."""

    tune: bool
    nuts_step: Any


class PymcCachedSigmaNutsStep(BlockedStep):
    """Sample site amplitudes exactly conditional on the current flux state."""

    name = "cached_sigma_nuts"
    default_blocked = True
    stats_dtypes_shapes = {
        "tune": (bool, []),
        "sigma_nuts_tree_steps": (int, []),
        "sigma_nuts_tree_depth": (int, []),
        "sigma_nuts_diverging": (bool, []),
        "sigma_nuts_divergences": (int, []),
        "sigma_nuts_energy_error": (float, []),
        "sigma_nuts_max_energy_error": (float, []),
        "sigma_nuts_reached_max_treedepth": (bool, []),
        "sigma_nuts_index_in_trajectory": (int, []),
        "sigma_likelihood_evaluations": (int, []),
        "sigma_gradient_evaluations": (int, []),
        "factor_cholesky_operations": (int, []),
        "accepted_sigma_block": (int, []),
        "sigma_accept_probability": (float, []),
        "sigma_proposal_scale": (float, []),
        "cache_refreshes": (int, []),
        "sigma_block_seconds": (float, []),
        "sigma_likelihood_evaluation_seconds": (float, []),
        "quadratic_refresh_seconds": (float, []),
        "quadratic_factor_cholesky_seconds": (float, []),
        "sigma_factor_cholesky_seconds": (float, []),
    }
    _state_class = _CachedSigmaNutsState

    def __init__(  # noqa: PLR0913
        self,
        vars=None,
        *,
        target: FixedOuCachedSigmaTarget,
        shared_cache: PytensorMarginalQuadraticCache,
        state_value_name: str,
        state_location: ArrayLike,
        state_cholesky: ArrayLike,
        prior_scale: ArrayLike | float,
        state_link: Literal["identity", "exp"] = "exp",
        target_accept: float = 0.8,
        max_treedepth: int = 10,
        early_max_treedepth: int = 8,
        step_scale: float = 0.25,
        compile_kwargs: dict[str, Any] | None = None,
        initial_point: PointType | None = None,
        model=None,
        blocked: bool = True,
        rng: RandomGenerator = None,
    ) -> None:
        outer_model = modelcontext(model)
        if vars is None:
            raise ValueError("The grouped sigma variable must be supplied explicitly.")
        value_vars = get_value_vars_from_user_vars(vars, outer_model)
        if len(value_vars) != 1:
            raise ValueError("Exactly one grouped sigma variable is required.")
        sigma_value_name = cast(str | None, value_vars[0].name)
        if not sigma_value_name or not state_value_name:
            raise ValueError("State and transformed-sigma point names must be non-empty.")
        if shared_cache.n_state != target.n_state:
            raise ValueError("Shared cache and marginalized target dimensions differ.")
        if state_link not in {"identity", "exp"}:
            raise ValueError("state_link must be 'identity' or 'exp'.")
        if not np.isfinite(target_accept) or not 0.0 < target_accept < 1.0:
            raise ValueError("target_accept must lie strictly between zero and one.")
        if max_treedepth <= 0:
            raise ValueError("max_treedepth must be positive.")
        if early_max_treedepth <= 0 or early_max_treedepth > max_treedepth:
            raise ValueError(
                "early_max_treedepth must be positive and no larger than max_treedepth."
            )
        if not np.isfinite(step_scale) or step_scale <= 0.0:
            raise ValueError("step_scale must be finite and positive.")

        point = outer_model.initial_point() if initial_point is None else initial_point
        try:
            state_point = np.asarray(point[state_value_name])
            initial_eta = np.asarray(point[sigma_value_name])
        except KeyError as exc:
            raise KeyError(f"Initial point is missing value {exc.args[0]!r}.") from exc
        location = np.asarray(state_location, dtype=state_point.dtype)
        cholesky = np.asarray(state_cholesky, dtype=state_point.dtype)
        if location.shape != state_point.shape:
            raise ValueError(
                f"state_location has shape {location.shape}, expected {state_point.shape}."
            )
        if cholesky.shape != (location.size, location.size):
            raise ValueError("state_cholesky must be square with one row per state.")
        if target.n_state != location.size:
            raise ValueError("Target design and state transform dimensions differ.")
        if not np.isfinite(location).all() or not np.isfinite(cholesky).all():
            raise ValueError("State transform must contain only finite values.")
        if initial_eta.shape != (target.n_group,):
            raise ValueError(
                f"Initial transformed sigma has shape {initial_eta.shape}, "
                f"expected {(target.n_group,)}."
            )
        scale = np.asarray(prior_scale, dtype=np.float64)
        if scale.ndim > 1 or (
            scale.ndim == 1 and scale.shape != (target.n_group,)
        ):
            raise ValueError("prior_scale must be scalar or contain one value per site.")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("prior_scale must contain only finite positive values.")

        self.vars = value_vars
        self.blocked = blocked
        self.rng = get_random_generator(rng)
        self.target = target
        self.shared_cache = shared_cache
        self.state_value_name = state_value_name
        self.sigma_value_name = sigma_value_name
        self.state_location = location.copy()
        self.state_cholesky = cholesky.copy()
        self.state_link = state_link
        self.tune = True
        self.current_cache = target.refresh(np.exp(initial_eta.astype(np.float64)))
        shared_cache.update(self.current_cache)

        point_dtype = initial_eta.dtype
        outer_sigma_rv = outer_model.values_to_rvs[value_vars[0]]
        sigma_rv_name = cast(str | None, outer_sigma_rv.name)
        if not sigma_rv_name:
            raise ValueError("The grouped sigma random variable must have a name.")
        self.likelihood_op = _PytensorSigmaLikelihoodOp(target)
        with pm.Model(model=None) as conditional_model:
            conditional_sigma = pm.HalfNormal(
                sigma_rv_name,
                sigma=scale,
                shape=target.n_group,
                dtype=point_dtype.name,
            )
            likelihood, _ = self.likelihood_op(
                conditional_sigma,
                self.likelihood_op.residual,
            )
            pm.Potential("sigma_conditional_likelihood", likelihood)
        conditional_value_name = cast(
            str | None,
            conditional_model.rvs_to_values[conditional_sigma].name,
        )
        if conditional_value_name != sigma_value_name:
            raise ValueError(
                "Conditional and outer sigma transformed names differ: "
                f"{conditional_value_name!r} and {sigma_value_name!r}."
            )
        conditional_initial = conditional_model.initial_point()
        conditional_initial[sigma_value_name] = initial_eta.copy()
        nuts_rng, potential_rng = self.rng.spawn(2)
        potential = QuadPotentialDiagAdapt(
            target.n_group,
            np.zeros(target.n_group, dtype=point_dtype),
            np.ones(target.n_group, dtype=point_dtype),
            10,
            dtype=point_dtype.name,
            rng=potential_rng,
        )
        nuts_kwargs: dict[str, Any] = {
            "vars": [conditional_sigma],
            "target_accept": target_accept,
            "max_treedepth": int(max_treedepth),
            "early_max_treedepth": int(early_max_treedepth),
            "step_scale": float(step_scale),
            "model": conditional_model,
            "initial_point": conditional_initial,
            "rng": nuts_rng,
            "dtype": point_dtype.name,
            "potential": potential,
        }
        if compile_kwargs is not None:
            nuts_kwargs["compile_kwargs"] = compile_kwargs
        self.conditional_model = conditional_model
        self.conditional_sigma = conditional_sigma
        self.nuts_step = pm.NUTS(**nuts_kwargs)

    def _point_values(self, point: PointType) -> tuple[FloatArray, FloatArray]:
        try:
            state_white = np.asarray(point[self.state_value_name])
            eta = np.asarray(point[self.sigma_value_name])
        except KeyError as exc:
            raise KeyError(
                f"Compound point is missing required value {exc.args[0]!r}."
            ) from exc
        if state_white.shape != self.state_location.shape:
            raise ValueError(
                f"Whitened state has shape {state_white.shape}, "
                f"expected {self.state_location.shape}."
            )
        if eta.shape != (self.target.n_group,):
            raise ValueError(
                f"Transformed sigma has shape {eta.shape}, "
                f"expected {(self.target.n_group,)}."
            )
        linear_state = self.state_location + self.state_cholesky @ state_white
        physical_state = (
            np.exp(linear_state, dtype=linear_state.dtype)
            if self.state_link == "exp"
            else linear_state
        )
        if not np.isfinite(physical_state).all():
            raise FloatingPointError("Physical state is non-finite.")
        residual = self.target.observations - (
            self.target.fixed_contribution
            + self.target.design @ np.asarray(physical_state, dtype=np.float64)
        )
        return eta.astype(np.float64), residual

    def _ensure_cache_matches(self, eta: FloatArray) -> int:
        sigma = np.exp(eta)
        if np.array_equal(sigma, self.current_cache.sigma):
            return 0
        self.current_cache = self.target.refresh(sigma)
        self.shared_cache.update(self.current_cache)
        return 1

    def step(self, point: PointType) -> tuple[PointType, StatsType]:
        """Delegate the conditional trajectory and refresh only when changed."""
        started = time.perf_counter()
        eta_initial, residual = self._point_values(point)
        pre_refreshes = self._ensure_cache_matches(eta_initial)
        pre_refresh_factorizations = (
            self.current_cache.factor_cholesky_operations if pre_refreshes else 0
        )
        pre_refresh_cholesky_seconds = (
            self.current_cache.factor_cholesky_seconds if pre_refreshes else 0.0
        )
        self.likelihood_op.install_residual(residual)
        self.likelihood_op.reset_diagnostics()
        conditional_point = {
            self.sigma_value_name: np.asarray(point[self.sigma_value_name]).copy()
        }
        conditional_updated, stats = self.nuts_step.step(conditional_point)
        if len(stats) != 1:
            raise AssertionError("Stock sigma NUTS returned an unexpected stats block.")
        nuts_stats = stats[0]
        eta_next_point = np.asarray(
            conditional_updated[self.sigma_value_name],
            dtype=np.asarray(point[self.sigma_value_name]).dtype,
        )
        eta_next = eta_next_point.astype(np.float64)
        changed = not np.array_equal(eta_next, eta_initial)
        cache_refreshes = pre_refreshes
        refresh_seconds = 0.0
        refresh_factorizations = 0
        refresh_cholesky_seconds = 0.0
        if changed:
            cache = self.target.refresh(np.exp(eta_next))
            self.current_cache = cache
            self.shared_cache.update(cache)
            cache_refreshes += 1
            refresh_seconds = cache.refresh_seconds
            refresh_factorizations = cache.factor_cholesky_operations
            refresh_cholesky_seconds = cache.factor_cholesky_seconds
        diagnostics = self.likelihood_op.diagnostics
        point_new = point.copy()
        point_new[self.sigma_value_name] = eta_next_point
        return point_new, [
            {
                "tune": self.tune,
                "sigma_nuts_tree_steps": int(nuts_stats["tree_size"]),
                "sigma_nuts_tree_depth": int(nuts_stats["depth"]),
                "sigma_nuts_diverging": bool(nuts_stats["diverging"]),
                "sigma_nuts_divergences": int(nuts_stats["divergences"]),
                "sigma_nuts_energy_error": float(nuts_stats["energy_error"]),
                "sigma_nuts_max_energy_error": float(nuts_stats["max_energy_error"]),
                "sigma_nuts_reached_max_treedepth": bool(
                    nuts_stats["reached_max_treedepth"]
                ),
                "sigma_nuts_index_in_trajectory": int(
                    nuts_stats["index_in_trajectory"]
                ),
                "sigma_likelihood_evaluations": diagnostics.likelihood_evaluations,
                "sigma_gradient_evaluations": diagnostics.gradient_evaluations,
                "factor_cholesky_operations": (
                    pre_refresh_factorizations
                    + diagnostics.factor_cholesky_operations
                    + refresh_factorizations
                ),
                "accepted_sigma_block": int(changed),
                "sigma_accept_probability": float(nuts_stats["mean_tree_accept"]),
                "sigma_proposal_scale": float(nuts_stats["step_size"]),
                "cache_refreshes": cache_refreshes,
                "sigma_block_seconds": time.perf_counter() - started,
                "sigma_likelihood_evaluation_seconds": diagnostics.evaluation_seconds,
                "quadratic_refresh_seconds": refresh_seconds,
                "quadratic_factor_cholesky_seconds": (
                    pre_refresh_cholesky_seconds + refresh_cholesky_seconds
                ),
                "sigma_factor_cholesky_seconds": diagnostics.factor_cholesky_seconds,
            }
        ]

    def set_rng(self, rng: RandomGenerator) -> None:
        """Propagate PyMC's per-chain RNG into the delegated stock NUTS step."""
        super().set_rng(rng)
        self.nuts_step.set_rng(self.rng.spawn(1)[0])

    def stop_tuning(self) -> None:
        """Freeze delegated step-size and mass-matrix adaptation."""
        self.nuts_step.stop_tuning()
        self.tune = False

    def reset_tuning(self) -> None:
        """Reset delegated NUTS adaptation for a fresh chain."""
        self.nuts_step.reset_tuning()
        self.tune = True

    @staticmethod
    def competence(var: Any, has_grad: bool) -> Literal[Competence.INCOMPATIBLE]:
        """Never claim variables through PyMC's automatic assignment."""
        return Competence.INCOMPATIBLE


def make_cached_sigma_compound_step(  # noqa: PLR0913
    *,
    model: pm.Model,
    sigma: Any,
    state: Any,
    target: FixedOuCachedSigmaTarget,
    shared_cache: PytensorMarginalQuadraticCache,
    state_value_name: str,
    state_location: ArrayLike,
    state_cholesky: ArrayLike,
    prior_scale: ArrayLike | float,
    state_link: Literal["identity", "exp"] = "exp",
    initial_point: PointType | None = None,
    sigma_target_accept: float = 0.8,
    state_target_accept: float = 0.9,
    rng: RandomGenerator = None,
) -> pm.CompoundStep:
    """Construct the required sigma-then-state stock PyMC compound sweep."""
    root_rng = get_random_generator(rng)
    sigma_rng, state_rng = root_rng.spawn(2)
    sigma_step = PymcCachedSigmaNutsStep(
        [sigma],
        target=target,
        shared_cache=shared_cache,
        state_value_name=state_value_name,
        state_location=state_location,
        state_cholesky=state_cholesky,
        prior_scale=prior_scale,
        state_link=state_link,
        target_accept=sigma_target_accept,
        initial_point=initial_point,
        model=model,
        rng=sigma_rng,
    )
    with model:
        state_step = pm.NUTS(
            vars=[state],
            target_accept=state_target_accept,
            initial_point=initial_point,
            model=model,
            rng=state_rng,
        )
    return pm.CompoundStep([sigma_step, state_step])


__all__ = [
    "CACHED_SIGMA_SAMPLER_METADATA",
    "PymcCachedSigmaNutsStep",
    "PytensorMarginalQuadraticCache",
    "make_cached_sigma_compound_step",
]
