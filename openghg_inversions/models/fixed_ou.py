"""Fixed within-site OU covariance under a diagonal-plus-low-rank base."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pymc as pm
import pytensor.tensor as pt
from pytensor.gradient import DisconnectedType, grad_not_implemented
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import cho_solve, eigh, solve_triangular
from scipy.sparse import csr_matrix


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_HOURS_PER_NANOSECOND = 1.0 / 3.6e12


@dataclass(frozen=True)
class FixedOuSiteBlock:
    """Precomputed generalized eigenbasis for one site's fixed OU template."""

    site: int
    observation_indices: IntArray
    mode_slice: slice
    correlation: FloatArray
    correlation_cholesky: FloatArray
    transform: FloatArray


@dataclass(frozen=True)
class FixedOuLikelihoodEvaluation:
    """One exact likelihood value and its physical-space gradients."""

    log_likelihood: float
    gradient_residual: FloatArray
    gradient_site_amplitude: FloatArray


def _rejection_gradient_amplitude(amplitude: FloatArray) -> FloatArray:
    """Return a transform-safe gradient for an invalid scale proposal."""
    gradient = np.full(amplitude.shape, -1.0, dtype=np.float64)
    finite_large = np.isfinite(amplitude) & (amplitude > 1.0)
    gradient[finite_large] = -np.reciprocal(amplitude[finite_large])
    return gradient


class _FixedOuLogpOp(Op):
    """Instance-local PyTensor bridge to one prepared NumPy/SciPy target."""

    def __init__(self, target: FixedOuLowRank) -> None:
        self.target = target

    def make_node(self, residual: Any, site_amplitude: Any) -> Apply:
        residual_variable = pt.as_tensor_variable(residual)
        amplitude_variable = pt.as_tensor_variable(site_amplitude)
        if residual_variable.ndim != 1:
            raise TypeError("residual must be a one-dimensional PyTensor variable.")
        if amplitude_variable.ndim != 1:
            raise TypeError("site_amplitude must be a one-dimensional PyTensor variable.")
        return Apply(
            self,
            [residual_variable, amplitude_variable],
            [pt.dscalar(), pt.dvector(), pt.dvector()],
        )

    def perform(
        self,
        node: Apply,
        inputs: list[NDArray[Any]],
        output_storage: list[list[NDArray[Any] | None]],
    ) -> None:
        del node
        evaluation = self.target.evaluate(inputs[0], inputs[1])
        output_storage[0][0] = np.asarray(evaluation.log_likelihood, dtype=np.float64)
        output_storage[1][0] = np.asarray(evaluation.gradient_residual, dtype=np.float64)
        output_storage[2][0] = np.asarray(
            evaluation.gradient_site_amplitude, dtype=np.float64
        )

    def L_op(
        self,
        inputs: list[Variable],
        outputs: list[Variable],
        output_grads: list[Variable],
    ) -> list[Variable]:
        """Reuse analytic outputs when differentiating the likelihood value."""
        if not isinstance(output_grads[1].type, DisconnectedType) or not isinstance(
            output_grads[2].type, DisconnectedType
        ):
            return [
                grad_not_implemented(self, 0, inputs[0]),
                grad_not_implemented(self, 1, inputs[1]),
            ]
        return [output_grads[0] * outputs[1], output_grads[0] * outputs[2]]

    def infer_shape(
        self,
        fgraph: Any,
        node: Apply,
        input_shapes: list[tuple[Any, ...]],
    ) -> list[tuple[Any, ...]]:
        del fgraph, node
        return [(), input_shapes[0], input_shapes[1]]


@dataclass(frozen=True)
class FixedOuLowRank:
    """Exact fixed-OU target with a diagonal-plus-low-rank fixed covariance.

    The eager constructor precomputes one generalized eigenbasis per site.
    :meth:`logp` then evaluates changing site amplitudes through an ``r x r``
    Woodbury system, where ``r`` is the aggregation-covariance rank.
    """

    factor: FloatArray
    diagonal_variance: FloatArray
    observation_time_hours: FloatArray
    site_index: IntArray
    site_labels: tuple[str, ...]
    tau_hours_by_site: FloatArray
    site_blocks: tuple[FixedOuSiteBlock, ...]
    mode_eigenvalues: FloatArray
    mode_site_index: IntArray
    transformed_factor: FloatArray
    mode_transform: csr_matrix
    base_logdet: float
    _logp_op: _FixedOuLogpOp = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_logp_op", _FixedOuLogpOp(self))

    @property
    def n_observation(self) -> int:
        """Number of observation rows."""
        return int(self.diagonal_variance.size)

    @property
    def n_site(self) -> int:
        """Number of independently scaled sites."""
        return int(self.tau_hours_by_site.size)

    @property
    def rank(self) -> int:
        """Rank of the fixed coherent covariance factor."""
        return int(self.factor.shape[1])

    def logp(
        self,
        value: TensorVariable,
        mean: TensorVariable,
        site_amplitude: TensorVariable,
    ) -> TensorVariable:
        """Return the normalized Gaussian log density as a differentiable Op."""
        residual = pt.cast(value - mean, "float64")
        amplitude = pt.cast(pt.atleast_1d(site_amplitude), "float64")
        return cast(TensorVariable, self._logp_op(residual, amplitude)[0])

    def evaluate(
        self,
        residual: ArrayLike,
        site_amplitude: ArrayLike | float,
    ) -> FixedOuLikelihoodEvaluation:
        """Evaluate the exact log density and analytic physical gradients."""
        residual_value = np.asarray(residual, dtype=np.float64)
        amplitude = np.atleast_1d(np.asarray(site_amplitude, dtype=np.float64))
        if residual_value.shape != (self.n_observation,):
            raise ValueError(
                f"residual has shape {residual_value.shape}, "
                f"expected {(self.n_observation,)}."
            )
        if amplitude.shape != (self.n_site,):
            raise ValueError(
                f"site_amplitude has shape {amplitude.shape}, expected {(self.n_site,)}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            amplitude_squared = np.square(amplitude)
        invalid_amplitude = bool(
            not np.isfinite(amplitude).all()
            or np.any(amplitude < 0.0)
            or not np.isfinite(amplitude_squared).all()
        )
        if invalid_amplitude:
            return FixedOuLikelihoodEvaluation(
                log_likelihood=-np.inf,
                gradient_residual=np.zeros(self.n_observation, dtype=np.float64),
                gradient_site_amplitude=_rejection_gradient_amplitude(amplitude),
            )
        if not np.isfinite(residual_value).all():
            gradient_residual = np.zeros(self.n_observation, dtype=np.float64)
            infinite = np.isinf(residual_value)
            gradient_residual[infinite] = -np.sign(residual_value[infinite])
            return FixedOuLikelihoodEvaluation(
                log_likelihood=-np.inf,
                gradient_residual=gradient_residual,
                gradient_site_amplitude=np.zeros(self.n_site, dtype=np.float64),
            )

        mode_variance = self.mode_eigenvalues + amplitude_squared[self.mode_site_index]
        if not np.isfinite(mode_variance).all() or np.any(mode_variance <= 0.0):
            return FixedOuLikelihoodEvaluation(
                log_likelihood=-np.inf,
                gradient_residual=np.zeros(self.n_observation, dtype=np.float64),
                gradient_site_amplitude=_rejection_gradient_amplitude(amplitude),
            )
        weights = np.reciprocal(mode_variance)
        transformed_residual = cast(FloatArray, self.mode_transform @ residual_value)
        square_root_weights = np.sqrt(weights)
        whitened_residual = transformed_residual * square_root_weights

        if self.rank:
            whitened_factor = self.transformed_factor * square_root_weights[:, None]
            core = np.eye(self.rank, dtype=np.float64) + (
                whitened_factor.T @ whitened_factor
            )
            core = (core + core.T) * 0.5
            cholesky = np.linalg.cholesky(core)
            projected = whitened_factor.T @ whitened_residual
            latent_mode = cho_solve((cholesky, True), projected, check_finite=False)
            conditional_whitened = whitened_residual - whitened_factor @ latent_mode
            quadratic = float(
                conditional_whitened @ conditional_whitened
                + latent_mode @ latent_mode
            )
            logdet_core = float(2.0 * np.log(np.diag(cholesky)).sum())
            conditional_residual = (
                transformed_residual - self.transformed_factor @ latent_mode
            )
            transformed = solve_triangular(
                cholesky,
                self.transformed_factor.T,
                lower=True,
                check_finite=False,
            )
            leverage = np.square(transformed).sum(axis=0)
        else:
            quadratic = float(whitened_residual @ whitened_residual)
            logdet_core = 0.0
            conditional_residual = transformed_residual
            leverage = np.zeros(self.n_observation, dtype=np.float64)

        logdet = self.base_logdet + float(np.log(mode_variance).sum()) + logdet_core
        log_likelihood = -0.5 * (
            self.n_observation * math.log(2.0 * math.pi) + logdet + quadratic
        )
        mode_score = (
            np.square(weights) * (np.square(conditional_residual) + leverage) - weights
        )
        gradient_amplitude = amplitude * np.bincount(
            self.mode_site_index,
            weights=mode_score,
            minlength=self.n_site,
        )
        solved_transformed = weights * conditional_residual
        gradient_residual = -(self.mode_transform.T @ solved_transformed)
        return FixedOuLikelihoodEvaluation(
            log_likelihood=float(log_likelihood),
            gradient_residual=cast(FloatArray, np.asarray(gradient_residual)),
            gradient_site_amplitude=cast(FloatArray, gradient_amplitude),
        )

    def marginal_variance(self, site_amplitude: TensorVariable) -> TensorVariable:
        """Return the observation-aligned covariance diagonal."""
        amplitude = pt.atleast_1d(site_amplitude)
        site_index = pt.as_tensor_variable(self.site_index)
        return (
            pm.floatX(self.diagonal_variance)
            + pm.floatX(np.square(self.factor).sum(axis=1))
            + pt.square(amplitude[site_index])
        )

    def covariance_dense(self, site_amplitude: ArrayLike | float) -> FloatArray:
        """Materialize the represented covariance for small reference checks."""
        amplitude = _site_values(site_amplitude, self.n_site, "site_amplitude", positive=False)
        covariance = self.factor @ self.factor.T + np.diag(self.diagonal_variance)
        for block in self.site_blocks:
            indices = block.observation_indices
            covariance[np.ix_(indices, indices)] += (
                amplitude[block.site] ** 2 * block.correlation
            )
        return cast(FloatArray, (covariance + covariance.T) * 0.5)

    def random(
        self,
        mean: np.ndarray,
        site_amplitude: np.ndarray,
        rng: np.random.Generator | None = None,
        size: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Draw from the represented Gaussian for ``pm.CustomDist``."""
        rng = np.random.default_rng() if rng is None else rng
        sample_shape = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        amplitude = _site_values(site_amplitude, self.n_site, "site_amplitude", positive=False)
        result = np.broadcast_to(np.asarray(mean), (*sample_shape, self.n_observation)).copy()
        if self.rank:
            factor_noise = rng.normal(size=(*sample_shape, self.rank))
            result += np.einsum("...r,nr->...n", factor_noise, self.factor)
        result += rng.normal(size=result.shape) * np.sqrt(self.diagonal_variance)
        for block in self.site_blocks:
            site_noise = rng.normal(size=(*sample_shape, block.observation_indices.size))
            result[..., block.observation_indices] += amplitude[block.site] * (
                site_noise @ block.correlation_cholesky.T
            )
        return result


def prepare_fixed_ou_low_rank(
    factor: ArrayLike,
    diagonal_variance: ArrayLike,
    observation_times: ArrayLike,
    site_index: ArrayLike,
    tau_hours: ArrayLike | float | Mapping[str, float],
    *,
    site_labels: Sequence[str] | None = None,
) -> FixedOuLowRank:
    """Precompute a fixed within-site OU target at an explicit eager boundary.

    Observation rows may be interleaved and their times need not be sorted.
    ``site_index`` must use contiguous integer identifiers from zero. A mapping
    of tau values is accepted only with matching, ordered ``site_labels``.
    """
    diagonal = _vector(diagonal_variance, "diagonal_variance")
    n_observation = diagonal.size
    if n_observation == 0:
        raise ValueError("At least one observation is required.")
    if np.any(diagonal < 0.0):
        raise ValueError("diagonal_variance must contain non-negative values.")
    factor_value = _matrix(factor, "factor")
    if factor_value.shape[0] != n_observation:
        raise ValueError("factor must have one row per observation.")
    time_hours = _time_hours(observation_times)
    if time_hours.shape != (n_observation,):
        raise ValueError("observation_times must have one value per observation.")
    group_index = _group_index(site_index, n_observation)
    n_site = int(group_index.max()) + 1
    labels = _site_labels(site_labels, n_site)
    tau_by_site = _tau_values(tau_hours, labels)

    blocks: list[FixedOuSiteBlock] = []
    eigenvalues: list[FloatArray] = []
    mode_sites: list[IntArray] = []
    transform_rows: list[IntArray] = []
    transform_columns: list[IntArray] = []
    transform_values: list[FloatArray] = []
    transformed_factor = np.empty_like(factor_value)
    mode_start = 0
    base_logdet = 0.0
    for site in range(n_site):
        indices = np.flatnonzero(group_index == site).astype(np.int64)
        times = time_hours[indices]
        if np.unique(times).size != times.size:
            raise ValueError(
                f"Observation times must be unique within site {labels[site]!r}."
            )
        lag_hours = np.abs(times[:, None] - times[None, :])
        correlation = np.exp(-lag_hours / tau_by_site[site])
        correlation = cast(FloatArray, (correlation + correlation.T) * 0.5)
        try:
            correlation_cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"OU correlation template for site {labels[site]!r} must be positive definite."
            ) from error
        base_logdet += float(2.0 * np.log(np.diag(correlation_cholesky)).sum())

        values, vectors = eigh(
            np.diag(diagonal[indices]), correlation, check_finite=False
        )
        scale = max(1.0, float(np.max(np.abs(values))))
        if values[0] < -1.0e-10 * scale:
            raise np.linalg.LinAlgError(
                f"OU template for site {labels[site]!r} is not positive semidefinite."
            )
        values = cast(FloatArray, np.maximum(values, 0.0))
        transform = cast(FloatArray, vectors.T)
        mode_stop = mode_start + indices.size
        transformed_factor[mode_start:mode_stop] = transform @ factor_value[indices]
        transform_rows.append(
            np.repeat(np.arange(mode_start, mode_stop, dtype=np.int64), indices.size)
        )
        transform_columns.append(np.tile(indices, indices.size))
        transform_values.append(transform.ravel())
        blocks.append(
            FixedOuSiteBlock(
                site=site,
                observation_indices=indices,
                mode_slice=slice(mode_start, mode_stop),
                correlation=correlation,
                correlation_cholesky=cast(FloatArray, correlation_cholesky),
                transform=transform,
            )
        )
        eigenvalues.append(values)
        mode_sites.append(np.full(indices.size, site, dtype=np.int64))
        mode_start = mode_stop

    return FixedOuLowRank(
        factor=factor_value,
        diagonal_variance=diagonal,
        observation_time_hours=time_hours,
        site_index=group_index,
        site_labels=labels,
        tau_hours_by_site=tau_by_site,
        site_blocks=tuple(blocks),
        mode_eigenvalues=cast(FloatArray, np.concatenate(eigenvalues)),
        mode_site_index=cast(IntArray, np.concatenate(mode_sites)),
        transformed_factor=cast(FloatArray, transformed_factor),
        mode_transform=csr_matrix(
            (
                np.concatenate(transform_values),
                (np.concatenate(transform_rows), np.concatenate(transform_columns)),
            ),
            shape=(n_observation, n_observation),
        ),
        base_logdet=base_logdet,
    )


def _time_hours(value: ArrayLike) -> FloatArray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"observation_times must be one-dimensional, got {array.shape}.")
    if np.issubdtype(array.dtype, np.datetime64):
        nanoseconds = array.astype("datetime64[ns]").astype(np.int64)
        if np.any(nanoseconds == np.iinfo(np.int64).min):
            raise ValueError("observation_times must not contain NaT.")
        hours = (nanoseconds - nanoseconds.min()).astype(np.float64)
        hours *= _HOURS_PER_NANOSECOND
    else:
        hours = np.asarray(value, dtype=np.float64)
    if not np.isfinite(hours).all():
        raise ValueError("observation_times must contain only finite values.")
    return cast(FloatArray, hours.copy())


def _vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return cast(FloatArray, array.copy())


def _matrix(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return cast(FloatArray, array.copy())


def _group_index(value: ArrayLike, n_observation: int) -> IntArray:
    array = np.asarray(value)
    if array.shape != (n_observation,):
        raise ValueError(
            f"site_index has shape {array.shape}, expected {(n_observation,)}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("site_index must contain integer identifiers.")
    index = np.asarray(array, dtype=np.int64)
    if np.any(index < 0):
        raise ValueError("site_index must be non-negative.")
    expected = np.arange(int(index.max()) + 1, dtype=np.int64)
    if not np.array_equal(np.unique(index), expected):
        raise ValueError("site_index identifiers must be contiguous from zero.")
    return cast(IntArray, index.copy())


def _site_labels(labels: Sequence[str] | None, n_site: int) -> tuple[str, ...]:
    result = tuple(str(site) for site in (range(n_site) if labels is None else labels))
    if len(result) != n_site or len(set(result)) != n_site:
        raise ValueError("site_labels must contain one unique label per site.")
    return result


def _tau_values(
    tau_hours: ArrayLike | float | Mapping[str, float], labels: tuple[str, ...]
) -> FloatArray:
    if isinstance(tau_hours, Mapping):
        if set(tau_hours) != set(labels):
            raise ValueError("tau_hours mapping keys must exactly match site_labels.")
        value: ArrayLike | float = [tau_hours[label] for label in labels]
    else:
        value = tau_hours
    return _site_values(value, len(labels), "tau_hours", positive=True)


def _site_values(
    value: ArrayLike | float,
    n_site: int,
    name: str,
    *,
    positive: bool,
) -> FloatArray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(n_site, values.item(), dtype=np.float64)
    elif values.shape != (n_site,):
        raise ValueError(f"{name} must be scalar or have shape {(n_site,)}, got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values <= 0.0 if positive else values < 0.0):
        qualifier = "strictly positive" if positive else "non-negative"
        raise ValueError(f"{name} must contain finite, {qualifier} values.")
    return cast(FloatArray, values.copy())
