"""Cache the fixed-OU state likelihood between ordered sigma and state updates.

The compound sweep is deliberately ordered ``site amplitude -> state``. A
stock PyMC NUTS step first updates site amplitudes using their exact
conditional likelihood. If that transition changes the amplitudes, the
sampler rebuilds the state-likelihood quadratic for the returned values. The
following stock state NUTS step holds that quadratic fixed, so its leapfrog
evaluations do not refactor the observation covariance.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.cached_sigma import (
    FixedOuCachedSigmaTarget,
    MarginalQuadraticCache,
)


FloatArray = NDArray[np.float64]


class PytensorMarginalQuadraticCache:
    """Mutable coefficients read by the state-only PyTensor graph."""

    def __init__(self, initial: MarginalQuadraticCache) -> None:
        dtype = pytensor.config.floatX
        self.constant = pytensor.shared(
            np.asarray(initial.constant, dtype=dtype),
            name="cached_marginal_constant",
        )
        self.linear = pytensor.shared(
            np.asarray(initial.linear, dtype=dtype),
            name="cached_marginal_linear",
        )
        self.precision = pytensor.shared(
            np.asarray(initial.precision, dtype=dtype),
            name="cached_marginal_precision",
        )

    @property
    def n_state(self) -> int:
        """Return the represented affine-coefficient dimension."""
        return int(self.linear.get_value(borrow=True).size)

    def log_likelihood(self, coefficients: TensorVariable) -> TensorVariable:
        """Return the normalized cached quadratic as a PyTensor scalar."""
        return cast(
            TensorVariable,
            self.constant
            + pt.dot(self.linear, coefficients)
            - np.asarray(0.5, dtype=self.constant.dtype)
            * pt.dot(coefficients, pt.dot(self.precision, coefficients)),
        )

    def update(self, cache: MarginalQuadraticCache) -> None:
        """Install the supplied state-likelihood quadratic between compound steps."""
        self.constant.set_value(np.asarray(cache.constant, dtype=self.constant.dtype))
        self.linear.set_value(np.asarray(cache.linear, dtype=self.linear.dtype))
        self.precision.set_value(np.asarray(cache.precision, dtype=self.precision.dtype))


class _PytensorSigmaLikelihoodOp(Op):
    """Exact sigma-conditional likelihood with a chain-local fixed residual."""

    def __init__(self, target: FixedOuCachedSigmaTarget) -> None:
        self.target = target
        self.residual = pytensor.shared(
            np.zeros(target.n_obs, dtype=np.float64),
            name="sigma_conditional_residual",
        )

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
        sigma = np.asarray(inputs[0], dtype=np.float64)
        residual = np.asarray(inputs[1], dtype=np.float64)
        evaluation = self.target.evaluate_from_residual(residual, sigma)
        output_storage[0][0] = np.asarray(
            evaluation.log_likelihood,
            dtype=np.float64,
        )
        output_storage[1][0] = np.asarray(
            evaluation.gradient_site_amplitude,
            dtype=np.float64,
        )

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
        self.residual.set_value(np.asarray(residual, dtype=np.float64))


@dataclass_state
class _CachedSigmaNutsState(StepMethodState):
    """Chain-local state including the delegated stock PyMC NUTS state."""

    tune: bool
    nuts_step: Any


class PymcCachedSigmaNutsStep(BlockedStep):
    """Sample site amplitudes conditional on the current completed model mean."""

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
        "sigma_accept_probability": (float, []),
        "sigma_proposal_scale": (float, []),
        "cache_refreshes": (int, []),
    }
    _state_class = _CachedSigmaNutsState

    def __init__(  # noqa: PLR0913
        self,
        sigma: TensorVariable,
        *,
        target: FixedOuCachedSigmaTarget,
        shared_cache: PytensorMarginalQuadraticCache,
        initial_cache: MarginalQuadraticCache,
        modelled_mean: TensorVariable,
        prior_scale: ArrayLike | float,
        target_accept: float = 0.8,
        max_treedepth: int = 10,
        early_max_treedepth: int = 8,
        step_scale: float = 0.25,
        initial_point: PointType | None = None,
        model: pm.Model | None = None,
        blocked: bool = True,
        rng: RandomGenerator = None,
    ) -> None:
        outer_model = modelcontext(model)
        value_vars = get_value_vars_from_user_vars([sigma], outer_model)
        if len(value_vars) != 1:
            raise ValueError("Exactly one grouped sigma variable is required.")
        sigma_value_name = cast(str | None, value_vars[0].name)
        if not sigma_value_name:
            raise ValueError("The transformed-sigma point name must be non-empty.")
        if shared_cache.n_state != target.n_state:
            raise ValueError("Shared cache and marginalized target dimensions differ.")
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
            initial_eta = np.asarray(point[sigma_value_name])
        except KeyError as exc:
            raise KeyError(f"Initial point is missing value {exc.args[0]!r}.") from exc
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
        self.sigma_value_name = sigma_value_name
        modelled_mean_value = outer_model.replace_rvs_by_values([modelled_mean])[0]
        self.modelled_mean_fn = outer_model.compile_fn(
            modelled_mean_value,
            inputs=outer_model.value_vars,
            on_unused_input="ignore",
        )
        initial_mean = np.asarray(self.modelled_mean_fn(point), dtype=np.float64)
        if initial_mean.shape != (target.n_obs,):
            raise ValueError(
                "Modelled mean has shape "
                f"{initial_mean.shape}, expected {(target.n_obs,)}."
            )
        self.tune = True
        initial_sigma = np.exp(initial_eta.astype(np.float64))
        tolerance = 8.0 * np.finfo(initial_eta.dtype).eps
        if not np.allclose(initial_cache.sigma, initial_sigma, rtol=tolerance, atol=0.0):
            raise ValueError("Initial cache and transformed sigma point do not match.")
        self.current_cache = initial_cache
        self.current_eta = initial_eta.astype(np.float64)

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
        self.conditional_model = conditional_model
        self.conditional_sigma = conditional_sigma
        self.nuts_step = pm.NUTS(
            vars=[conditional_sigma],
            target_accept=target_accept,
            max_treedepth=int(max_treedepth),
            early_max_treedepth=int(early_max_treedepth),
            step_scale=float(step_scale),
            model=conditional_model,
            initial_point=conditional_initial,
            rng=nuts_rng,
            dtype=point_dtype.name,
            potential=potential,
        )

    def _point_values(self, point: PointType) -> tuple[FloatArray, FloatArray]:
        eta = np.asarray(point[self.sigma_value_name])
        modelled_mean = np.asarray(
            self.modelled_mean_fn(point),
            dtype=np.float64,
        )
        residual = self.target.observations - modelled_mean
        return eta.astype(np.float64), residual

    def _ensure_cache_matches(self, eta: FloatArray) -> int:
        if np.array_equal(eta, self.current_eta):
            return 0
        self.current_cache = self.target.refresh(np.exp(eta))
        self.shared_cache.update(self.current_cache)
        self.current_eta = eta.copy()
        return 1

    def step(self, point: PointType) -> tuple[PointType, StatsType]:
        """Delegate the conditional trajectory and refresh only when changed."""
        eta_initial, residual = self._point_values(point)
        cache_refreshes = self._ensure_cache_matches(eta_initial)
        self.likelihood_op.install_residual(residual)
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
        if changed:
            cache = self.target.refresh(np.exp(eta_next))
            self.current_cache = cache
            self.shared_cache.update(cache)
            self.current_eta = eta_next.copy()
            cache_refreshes += 1
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
                "sigma_accept_probability": float(nuts_stats["mean_tree_accept"]),
                "sigma_proposal_scale": float(nuts_stats["step_size"]),
                "cache_refreshes": cache_refreshes,
            }
        ]

    def setup_chain(self, rng: RandomGenerator, tune: int, draws: int) -> None:
        """Propagate PyMC's per-chain setup into the delegated stock NUTS step.

        Args:
            rng: Random generator for this chain.
            tune: Number of tuning iterations.
            draws: Number of posterior draws.
        """
        super().setup_chain(rng, tune, draws)
        self.nuts_step.setup_chain(self.rng.spawn(1)[0], tune, draws)

    def set_rng(self, rng: RandomGenerator) -> None:
        """Reseed this step and its delegated NUTS step.

        This compatibility method preserves the pre-PyMC 6 public API. PyMC 6
        calls :meth:`setup_chain` directly during sampling.

        Args:
            rng: Random generator for this chain.
        """
        self.setup_chain(rng, tune=0, draws=0)

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
    sigma: TensorVariable,
    states: Sequence[TensorVariable],
    modelled_mean: TensorVariable,
    target: FixedOuCachedSigmaTarget,
    shared_cache: PytensorMarginalQuadraticCache,
    initial_cache: MarginalQuadraticCache,
    prior_scale: ArrayLike | float,
    initial_point: PointType | None = None,
    sigma_target_accept: float = 0.8,
    state_target_accept: float = 0.9,
    rng: RandomGenerator = None,
) -> pm.CompoundStep:
    """Construct the required sigma-then-state stock PyMC compound sweep.

    At the start of each sweep, the sigma step evaluates ``modelled_mean`` at
    the current state point. It then updates sigma and refreshes the quadratic
    for the value returned by that transition before the following stock NUTS
    step jointly updates ``states``.
    """
    root_rng = get_random_generator(rng)
    sigma_rng, state_rng = root_rng.spawn(2)
    sigma_step = PymcCachedSigmaNutsStep(
        sigma,
        target=target,
        shared_cache=shared_cache,
        initial_cache=initial_cache,
        modelled_mean=modelled_mean,
        prior_scale=prior_scale,
        target_accept=sigma_target_accept,
        initial_point=initial_point,
        model=model,
        rng=sigma_rng,
    )
    with model:
        state_step = pm.NUTS(
            vars=list(states),
            target_accept=state_target_accept,
            initial_point=initial_point,
            model=model,
            rng=state_rng,
        )
    return pm.CompoundStep([sigma_step, state_step])


__all__ = [
    "PymcCachedSigmaNutsStep",
    "PytensorMarginalQuadraticCache",
    "make_cached_sigma_compound_step",
]
