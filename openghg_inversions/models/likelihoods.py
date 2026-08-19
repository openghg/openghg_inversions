"""Observation likelihoods for diagonal and structured covariance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.observation_error import AggregationError


def _low_rank_gaussian_logp(
    value: TensorVariable,
    mean: TensorVariable,
    factor: TensorVariable,
    diagonal_variance: TensorVariable,
) -> TensorVariable:
    """Evaluate a normalized low-rank-plus-diagonal Gaussian log density.

    Args:
        value: Observation vector.
        mean: Modelled mean vector.
        factor: Low-rank covariance factor with observation rows.
        diagonal_variance: Positive independent variance for each observation.

    Returns:
        Scalar Gaussian log density evaluated through the Woodbury identity.
    """
    inverse_sqrt_diagonal = pt.reciprocal(pt.sqrt(diagonal_variance))
    whitened_residual = (value - mean) * inverse_sqrt_diagonal
    whitened_factor = factor * inverse_sqrt_diagonal[:, None]
    core = pt.eye(factor.shape[1], dtype=whitened_factor.dtype) + pt.dot(
        whitened_factor.T,
        whitened_factor,
    )
    cholesky = pt.linalg.cholesky(core)
    projected = pt.dot(whitened_factor.T, whitened_residual)
    latent_mode = pt.linalg.solve(
        cholesky.T,
        pt.linalg.solve(cholesky, projected),
    )
    # This augmented-residual form avoids subtracting two large, nearly equal
    # quadratic terms when the retained low-rank variance is large.
    conditional_residual = whitened_residual - pt.dot(whitened_factor, latent_mode)
    quadratic = pt.dot(conditional_residual, conditional_residual) + pt.dot(
        latent_mode,
        latent_mode,
    )
    logdet = pt.sum(pt.log(diagonal_variance)) + pm.floatX(2.0) * pt.sum(
        pt.log(pt.diagonal(cholesky))
    )
    normalizing_constant = value.shape[0] * pm.floatX(np.log(2.0 * np.pi))
    return -pm.floatX(0.5) * (normalizing_constant + logdet + quadratic)


def _low_rank_gaussian_random(
    mean: np.ndarray,
    factor: np.ndarray,
    diagonal_variance: np.ndarray,
    rng: np.random.Generator | None = None,
    size: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Draw from a low-rank-plus-diagonal Gaussian.

    Args:
        mean: Mean vector over observations.
        factor: Low-rank covariance factor with observation rows.
        diagonal_variance: Positive independent variance for each observation.
        rng: Optional NumPy random-number generator.
        size: Optional leading sample shape.

    Returns:
        Gaussian draws with the requested sample shape followed by the
        observation dimension.
    """
    rng = np.random.default_rng() if rng is None else rng
    sample_shape = () if size is None else (size,) if isinstance(size, int) else tuple(size)
    rank_noise = rng.normal(size=(*sample_shape, factor.shape[1]))
    diagonal_noise = rng.normal(size=(*sample_shape, mean.shape[-1]))
    correlated = np.einsum("...r,nr->...n", rank_noise, factor)
    return mean + correlated + np.sqrt(diagonal_variance) * diagonal_noise


def add_gaussian_observation_likelihood(
    *,
    observed: TensorVariable,
    mean: TensorVariable,
    independent_variance: TensorVariable,
    aggregation_error: AggregationError,
    output_dim: str,
) -> TensorVariable:
    """Add the canonical Gaussian observation variable ``y``.

    Args:
        observed: Observed concentration vector.
        mean: Completed forward-model concentration.
        independent_variance: Observation-aligned variance independent of the
            fixed aggregation error.
        aggregation_error: Selected diagonal, dense, or low-rank fixed
            aggregation covariance.
        output_dim: Named observation dimension for the PyMC variable.

    Returns:
        Observed PyMC variable named ``y``.
    """
    if aggregation_error.mode in ("none", "diagonal"):
        variance = independent_variance
        if aggregation_error.mode == "diagonal":
            assert aggregation_error.diagonal_variance is not None
            variance = variance + pt.as_tensor_variable(
                pm.floatX(np.asarray(aggregation_error.diagonal_variance.values))
            )
        return cast(
            TensorVariable,
            pm.Normal("y", mu=mean, sigma=pt.sqrt(variance), observed=observed, dims=output_dim),
        )

    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        covariance = pt.as_tensor_variable(pm.floatX(np.asarray(aggregation_error.covariance.values)))
        return cast(
            TensorVariable,
            pm.MvNormal(
                "y",
                mu=mean,
                cov=covariance + pt.diag(independent_variance),
                observed=observed,
                dims=output_dim,
            ),
        )

    assert aggregation_error.factor is not None
    assert aggregation_error.diagonal_variance is not None
    factor = pt.as_tensor_variable(pm.floatX(np.asarray(aggregation_error.factor.values)))
    diagonal = independent_variance + pt.as_tensor_variable(
        pm.floatX(np.asarray(aggregation_error.diagonal_variance.values))
    )
    return cast(
        TensorVariable,
        pm.CustomDist(
            "y",
            mean,
            factor,
            diagonal,
            logp=_low_rank_gaussian_logp,
            random=_low_rank_gaussian_random,
            signature="(n),(n,r),(n)->(n)",
            observed=observed,
            dims=output_dim,
        ),
    )
