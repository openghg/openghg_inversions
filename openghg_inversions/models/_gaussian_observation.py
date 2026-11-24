"""Private PyMC mechanics for Gaussian observation distributions.

This module owns the PyMC graph mechanics shared by concrete Gaussian
likelihood components. Error models such as additive sigma and pollution-event
scaling prepare their independent variance and fixed aggregation covariance,
then call :func:`add_gaussian_observation_likelihood` inside the active model.
Concrete model components own the scientific covariance and call these private
helpers only after that choice has been made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pymc as pm
from pymc.pytensorf import floatX
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import add_model_data
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    DIAGONAL_RESIDUAL_VARIANCE,
    LOW_RANK_FACTOR,
    AggregationError,
)

if TYPE_CHECKING:
    from openghg_inversions.models.paired_error import RegisteredPairedError


@dataclass(frozen=True)
class RegisteredAggregationError:
    """Aggregation-error tensors registered with the active PyMC model."""

    mode: str
    marginal_variance: TensorVariable
    covariance: TensorVariable | None = None
    factor: TensorVariable | None = None
    diagonal_variance: TensorVariable | None = None


def add_aggregation_error_data(
    aggregation_error: AggregationError | None,
    observations: xr.DataArray,
    *,
    output_dim: str,
) -> RegisteredAggregationError:
    """Register labelled fixed aggregation-error data with the active model.

    Args:
        aggregation_error: Validated backend-neutral covariance representation,
            or ``None`` for no fixed aggregation covariance.
        observations: Observation vector supplying labels for the marginal
            variance.
        output_dim: Observation dimension.

    Returns:
        Registered tensors used by the error scale and likelihood graph.
    """
    if aggregation_error is None:
        return RegisteredAggregationError(mode="none", marginal_variance=pt.constant(floatX(0.0)))
    marginal = xr.DataArray(
        floatX(aggregation_error.marginal_variance),
        dims=(output_dim,),
        coords={output_dim: observations.coords[output_dim]},
        name="aggregation_error_marginal_variance",
    )
    covariance = (
        add_model_data(aggregation_error.covariance, AGGREGATION_ERROR_COVARIANCE)
        if aggregation_error.covariance is not None
        else None
    )
    factor = (
        add_model_data(aggregation_error.factor, LOW_RANK_FACTOR)
        if aggregation_error.factor is not None
        else None
    )
    diagonal_variance = (
        add_model_data(
            aggregation_error.diagonal_variance,
            DIAGONAL_RESIDUAL_VARIANCE,
        )
        if aggregation_error.diagonal_variance is not None
        else None
    )
    return RegisteredAggregationError(
        mode=aggregation_error.mode,
        marginal_variance=add_model_data(marginal),
        covariance=covariance,
        factor=factor,
        diagonal_variance=diagonal_variance,
    )


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
    return _whitened_low_rank_logp(value - mean, factor, diagonal_variance, lambda rows: rows)


def _whitened_low_rank_logp(residual, factor, diagonal_variance, decorrelate) -> TensorVariable:
    """Woodbury log density for ``L diag(d) L^T + F F^T`` given ``decorrelate = L^{-1}`` on rows."""
    inverse_sqrt_diagonal = pt.reciprocal(pt.sqrt(diagonal_variance))
    whitened_residual = decorrelate(residual) * inverse_sqrt_diagonal
    whitened_factor = decorrelate(factor) * inverse_sqrt_diagonal[:, None]
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
    logdet = pt.sum(pt.log(diagonal_variance)) + floatX(2.0) * pt.sum(
        pt.log(pt.diagonal(cholesky))
    )
    normalizing_constant = residual.shape[0] * floatX(np.log(2.0 * np.pi))
    return -floatX(0.5) * (normalizing_constant + logdet + quadratic)


def _decorrelate_pairs(first: np.ndarray, second: np.ndarray, coefficient: TensorVariable):
    """Return ``L^{-1}`` on rows: subtract each pair's second-row regression on its first row."""

    def decorrelate(rows: TensorVariable) -> TensorVariable:
        scale = coefficient[second]
        if rows.ndim == 2:
            scale = scale[:, None]
        return pt.inc_subtensor(rows[second], -scale * rows[first])

    return decorrelate


def _paired_gaussian_distribution(first: np.ndarray, second: np.ndarray, *, low_rank: bool):
    """Log density and sampler for 2x2 pair blocks, optionally plus a low-rank factor.

    The covariance is ``L diag(conditional) L^T`` (plus ``F F^T``), where ``L``
    adds ``coefficient[q]`` times row ``p`` to row ``q`` of each pair.
    """

    def logp(value, mean, *params):
        *factor, conditional, coefficient = params
        decorrelate = _decorrelate_pairs(first, second, coefficient)
        if low_rank:
            return _whitened_low_rank_logp(value - mean, factor[0], conditional, decorrelate)
        whitened = decorrelate(value - mean) / pt.sqrt(conditional)
        normalizing_constant = value.shape[0] * floatX(np.log(2.0 * np.pi))
        return -floatX(0.5) * (
            normalizing_constant + pt.sum(pt.log(conditional)) + pt.dot(whitened, whitened)
        )

    def random(mean, *params, rng=None, size=None):
        *factor, conditional, coefficient = params
        rng = np.random.default_rng() if rng is None else rng
        sample_shape = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        draw = np.sqrt(conditional) * rng.normal(size=(*sample_shape, mean.shape[-1]))
        draw[..., second] += coefficient[second] * draw[..., first]
        if low_rank:
            noise = rng.normal(size=(*sample_shape, factor[0].shape[1]))
            draw = draw + np.einsum("...r,nr->...n", noise, factor[0])
        return mean + draw

    return logp, random


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
    aggregation_error: RegisteredAggregationError,
    output_dim: str,
    paired_error: RegisteredPairedError | None = None,
    paired_error_sd: TensorVariable | None = None,
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
        paired_error: Optional fixed correlations between pairs of rows.
            Their cross covariance is ``rho * sd_a * sd_b`` with
            ``paired_error_sd``. Pairs with ``rho = 0`` leave the likelihood on
            its unpaired path. Otherwise, without aggregation error or with
            diagonal or low-rank aggregation error, each correlated pair is an
            exact 2x2 block of the diagonal term, so the cost stays linear in
            the pairs and the low-rank width is unchanged; dense aggregation
            error adds the cross terms to its covariance in place (see
            :mod:`openghg_inversions.models.paired_error`).
        paired_error_sd: Standard deviations correlated by ``paired_error``.
            Their variance must be part of ``independent_variance``.

    Returns:
        Observed PyMC variable named ``y``.
    """
    if paired_error is not None and paired_error_sd is None:
        raise ValueError("Paired error correlation requires the paired standard deviations.")
    if paired_error is not None and paired_error.correlated:
        return _add_paired_gaussian_likelihood(
            observed=observed, mean=mean, independent_variance=independent_variance,
            aggregation_error=aggregation_error, output_dim=output_dim,
            paired_error=paired_error, paired_error_sd=paired_error_sd,
        )
    if aggregation_error.mode in ("none", "diagonal"):
        variance = independent_variance
        if aggregation_error.mode == "diagonal":
            assert aggregation_error.diagonal_variance is not None
            variance = variance + aggregation_error.diagonal_variance
        return cast(
            TensorVariable,
            pm.Normal("y", mu=mean, sigma=pt.sqrt(variance), observed=observed, dims=output_dim),
        )

    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        return cast(
            TensorVariable,
            pm.MvNormal(
                "y",
                mu=mean,
                cov=aggregation_error.covariance + pt.diag(independent_variance),
                observed=observed,
                dims=output_dim,
            ),
        )

    assert aggregation_error.factor is not None
    assert aggregation_error.diagonal_variance is not None
    diagonal = independent_variance + aggregation_error.diagonal_variance
    return cast(
        TensorVariable,
        pm.CustomDist(
            "y",
            mean,
            aggregation_error.factor,
            diagonal,
            logp=_low_rank_gaussian_logp,
            random=_low_rank_gaussian_random,
            signature="(n),(n,r),(n)->(n)",
            observed=observed,
            dims=output_dim,
        ),
    )


def _add_paired_gaussian_likelihood(
    *,
    observed: TensorVariable,
    mean: TensorVariable,
    independent_variance: TensorVariable,
    aggregation_error: RegisteredAggregationError,
    output_dim: str,
    paired_error: RegisteredPairedError,
    paired_error_sd: TensorVariable,
) -> TensorVariable:
    """Add ``y`` with correlated pairs; see :func:`add_gaussian_observation_likelihood`."""
    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        rows, columns, values = paired_error.cross(paired_error_sd)
        covariance = aggregation_error.covariance + pt.diag(independent_variance)
        covariance = pt.inc_subtensor(covariance[rows, columns], values)
        return cast(
            TensorVariable,
            pm.MvNormal("y", mu=mean, cov=covariance, observed=observed, dims=output_dim),
        )
    variance = independent_variance
    if aggregation_error.mode in ("diagonal", "low_rank"):
        assert aggregation_error.diagonal_variance is not None
        variance = variance + aggregation_error.diagonal_variance
    variance = pt.broadcast_to(variance, mean.shape)
    first, second, conditional, coefficient = paired_error.conditional(paired_error_sd, variance)
    low_rank = aggregation_error.mode == "low_rank"
    logp, random = _paired_gaussian_distribution(first, second, low_rank=low_rank)
    if low_rank:
        assert aggregation_error.factor is not None
        params, signature = (aggregation_error.factor, conditional, coefficient), "(n),(n,r),(n),(n)->(n)"
    else:
        params, signature = (conditional, coefficient), "(n),(n),(n)->(n)"
    return cast(
        TensorVariable,
        pm.CustomDist(
            "y",
            mean,
            *params,
            logp=logp,
            random=random,
            signature=signature,
            observed=observed,
            dims=output_dim,
        ),
    )


