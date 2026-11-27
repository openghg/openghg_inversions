"""Local posterior response to ``mu(lambda) = mu - lambda * e``.

For a Gaussian likelihood with perturbation-independent covariance, the score
is ``e.T @ R**-1 @ (mu-y)`` and the derivative of a posterior expectation
of a perturbation-independent quantity q is ``Cov(q, score)``. These are local
derivatives, conditional on the fitted model and priors, not finite shifts or
prior-independent information measures.

The score functions are explicit eager NumPy boundaries, evaluated separately
for each draw. Callers supply a matrix-free ``solve(rhs)`` bound to that draw's
covariance R, accepting an observation vector or a matrix of right-hand sides.
No fixed-precision assumption is made across draws. Array inputs are borrowed.
"""

from collections.abc import Callable

import numpy as np
import xarray as xr


def _check_covariance(covariance_depends_on_perturbation: bool) -> None:
    if covariance_depends_on_perturbation:
        raise ValueError("Perturbation-dependent covariance requires additional score terms.")


def _check_observations(residual: np.ndarray, directions: np.ndarray) -> None:
    if residual.ndim != 1 or directions.ndim != 2 or directions.shape[0] != residual.size:
        raise ValueError("Require residual (observation,) and directions (observation, direction).")


def _gaussian_prior_precision(
    design: np.ndarray, mean: np.ndarray, covariance: np.ndarray, nobs: int
) -> np.ndarray:
    if mean.ndim != 1 or design.shape != (nobs, mean.size) or covariance.shape != (mean.size, mean.size):
        raise ValueError("Gaussian prior mean, covariance and nuisance design shapes do not match.")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
        raise ValueError("Gaussian prior covariance must be symmetric positive definite.")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Gaussian prior covariance must be symmetric positive definite.") from error
    return np.linalg.solve(covariance, np.eye(mean.size))


def likelihood_score(
    mu_minus_y: np.ndarray,
    directions: np.ndarray,
    solve: Callable[[np.ndarray], np.ndarray],
    *,
    covariance_depends_on_perturbation: bool,
) -> np.ndarray:
    """Evaluate ``e.T R^-1 (mu-y)`` for directions shaped (observation, direction).

    The residual has shape (observation,). The caller must acknowledge whether
    the perturbation changes R; that unsupported case raises ``ValueError``.
    R may depend on the posterior draw, including on sampled error amplitudes.
    """
    _check_covariance(covariance_depends_on_perturbation)
    _check_observations(mu_minus_y, directions)
    return directions.T @ solve(mu_minus_y)


def gaussian_conditional_score(
    mu_without_nuisance_minus_y: np.ndarray,
    directions: np.ndarray,
    nuisance_design: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    solve: Callable[[np.ndarray], np.ndarray],
    *,
    covariance_depends_on_perturbation: bool,
) -> np.ndarray:
    """Integrate a Gaussian nuisance beta out of the likelihood score.

    Assume ``mu = mu_without_nuisance + D beta`` and an untruncated Gaussian
    prior ``beta ~ N(prior_mean, prior_covariance)``, independent of the retained
    state. R, D and directions must be independent of beta conditional on that
    state. D has shape (observation, nuisance), and the covariance is positive definite.
    This equals the score under covariance ``R + D C D.T`` without forming that
    observation-space matrix. Use with a q independent of beta conditional on
    the retained state; otherwise averaging the score alone is insufficient.

    Applying this formula to truncated or non-Gaussian priors is an explicit
    Gaussian approximation, not an exact identity for the original posterior.
    Inputs and the per-draw solve follow :func:`likelihood_score`.
    """
    _check_covariance(covariance_depends_on_perturbation)
    _check_observations(mu_without_nuisance_minus_y, directions)
    prior_precision = _gaussian_prior_precision(
        nuisance_design, prior_mean, prior_covariance, mu_without_nuisance_minus_y.size
    )
    residual = mu_without_nuisance_minus_y + nuisance_design @ prior_mean
    solved = solve(np.column_stack((residual, nuisance_design)))
    solved_residual, solved_design = solved[:, 0], solved[:, 1:]
    conditional_precision = prior_precision + nuisance_design.T @ solved_design
    correction = np.linalg.solve(conditional_precision, nuisance_design.T @ solved_residual)
    return directions.T @ (solved_residual - solved_design @ correction)


def gaussian_prior_shift_score(
    mu_minus_y: np.ndarray,
    directions: np.ndarray,
    nuisance: np.ndarray,
    nuisance_design: np.ndarray,
    translations: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    solve: Callable[[np.ndarray], np.ndarray],
    *,
    covariance_depends_on_perturbation: bool,
) -> np.ndarray:
    """Combine the exact residual likelihood score and Gaussian prior score.

    ``translations`` is a fixed (nuisance, direction) matrix d chosen by the
    caller. The result is ``(e-Dd).T R^-1 (mu-y) - d.T C^-1 (beta-m)``.
    The residual e-Dd is retained even for an imperfect projection. Require an
    untruncated Gaussian nuisance prior N(m,C), independent of retained state,
    and R independent of nuisance conditional on that state. A response based
    on this score alone requires q independent of the translated nuisance;
    otherwise its explicit translation derivative must also be included.
    Non-Gaussian or truncated priors make this a Gaussian approximation.
    """
    _check_covariance(covariance_depends_on_perturbation)
    _check_observations(mu_minus_y, directions)
    prior_precision = _gaussian_prior_precision(
        nuisance_design, prior_mean, prior_covariance, mu_minus_y.size
    )
    if nuisance.shape != prior_mean.shape or translations.shape != (prior_mean.size, directions.shape[1]):
        raise ValueError("Nuisance and fixed translation shapes do not match the prior and directions.")
    residual_directions = directions - nuisance_design @ translations
    return residual_directions.T @ solve(mu_minus_y) - translations.T @ prior_precision @ (
        nuisance - prior_mean
    )


def _align_samples(*arrays: xr.DataArray) -> tuple[xr.DataArray, ...]:
    for array in arrays:
        if not {"chain", "draw"}.issubset(array.dims):
            raise ValueError("Every input must retain chain and draw dimensions.")
        for dim in ("chain", "draw"):
            if dim not in array.indexes:
                raise ValueError(f"Every input must have explicit indexed {dim} coordinates.")
    return xr.align(*arrays, join="exact", copy=False)


def response_summary(q: xr.DataArray, score: xr.DataArray) -> xr.Dataset:
    """Construct a lazy, globally centered covariance estimate with divisor N.

    q and score retain explicit, identical chain/draw coordinates. Distinct
    dimensions (e.g. functional and direction) broadcast to response axes;
    shared dimensions must align exactly. ``chain_contribution`` is each
    chain's mean globally centered product, not a within-chain covariance.
    ``influence`` is the centered product minus its pooled gain, suitable for
    chain-aware MCSE diagnostics. Missing values propagate rather than silently
    changing the normalization. No array data are materialized here.
    """
    q, score = _align_samples(q, score)
    sample_dims = ("chain", "draw")
    product = (q - q.mean(sample_dims, skipna=False)) * (score - score.mean(sample_dims, skipna=False))
    gain = product.mean(sample_dims, skipna=False)
    return xr.Dataset(
        {
            "q": q,
            "score": score,
            "centered_product": product,
            "influence": product - gain,
            "gain": gain,
            "chain_contribution": product.mean("draw", skipna=False),
        },
        attrs={
            "intervention": "mu(lambda) = mu - lambda * e",
            "covariance_normalization": "1/N; global chain/draw centering",
        },
    )


def paired_response_summary(q: xr.DataArray, score_a: xr.DataArray, score_b: xr.DataArray) -> xr.Dataset:
    """Compare two scores on the same draws using their paired influence.

    The gain is response(a) minus response(b). Its influence subtracts the two
    same-draw influences, preserving their MCMC dependence for MCSE estimation.
    """
    q, score_a, score_b = _align_samples(q, score_a, score_b)
    if set(score_a.dims) != set(score_b.dims):
        raise ValueError("Paired scores must have the same dimensions.")
    return response_summary(q, score_a - score_b)


def response_diagnostics(summary: xr.Dataset) -> xr.Dataset:
    """Densify and materialize related samples together for ArviZ diagnostics.

    This explicit eager boundary returns the existing posterior-summary format:
    q, score and influence variables with a metric axis including MCSE of the
    mean, bulk ESS and R-hat. Influence MCSE estimates uncertainty of the gain
    (or paired difference), preserving the original chain/draw axes. Multiple
    chains and sufficient draws are needed for meaningful diagnostics; these
    diagnostics alone do not establish posterior accuracy or variance reduction.
    """
    from openghg_inversions.array_ops import to_dense
    from openghg_inversions.inference.diagnostics import posterior_summary

    samples = xr.Dataset({name: to_dense(summary[name]) for name in ("q", "score", "influence")}).compute()
    return posterior_summary(samples)
