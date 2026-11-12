"""Condition the unresolved part of native linear functionals on each draw.

A coherent reduction splits a native functional as
``Q x = Q m + D (alpha - Pi m) + Q x_perp``. Retained-state draws give the
first two terms. Given one posterior draw of ``alpha``, the other model
parameters and the noise parameters, ``Q x_perp`` is Gaussian with mean
``Q B_perp H.T (A + R)^-1 r`` and covariance
``Q B_perp Q.T - Q B_perp H.T (A + R)^-1 H B_perp Q.T``, where ``r`` is that
draw's observation residual and ``A + R`` its observation covariance. Adding
the mean and one draw from that covariance to each retained draw gives draws
of the complete functional. Under the prior, ``Q x_perp`` is independent of
``alpha`` with covariance ``Q B_perp Q.T``.

The observation covariance enters only through a caller-supplied solve, so
the same reconstruction works with a dense ``A + R``, with the fit's
low-rank-plus-diagonal ``A`` and fixed-OU noise (:func:`fixed_ou_solve`), or
with any other representation the caller has factorized.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.array_ops import same_index

if TYPE_CHECKING:
    from openghg_inversions.coherent_reduction import NativeFunctionalProducts
    from openghg_inversions.models.fixed_ou import FixedOuLowRank


__all__ = [
    "CovarianceSolve",
    "UnresolvedFunctionalConditioning",
    "complete_functional_draws",
    "dense_solve",
    "fixed_ou_solve",
]

FloatArray = NDArray[np.float64]
CovarianceSolve = Callable[[FloatArray, int, int], FloatArray]
"""Return ``(A + R)^-1 rhs`` for the draw at positional ``(chain, draw)``."""

SAMPLE_DIMS = ("chain", "draw")


@dataclass(frozen=True, slots=True, eq=False)
class UnresolvedFunctionalConditioning:
    """Inputs that condition unresolved native functionals on posterior draws.

    Attributes:
        products: Functional products of the reduction the fit used.
        residual: Posterior draws of the observation residual
            ``y - E[y | alpha, beta]`` on ``(chain, draw, observation_dim)``,
            from the same draws as the retained states.
        solve: ``(A + R)^-1`` for each posterior draw, with the ``A`` and
            noise model of the fit's likelihood.
        description: Human-readable account of ``A`` and ``R`` recorded in
            output metadata, for example
            ``"rank-512 LRPD A + fixed-OU noise at each draw's amplitudes"``.
        noise_varies_by_draw: Whether ``solve`` depends on the draw. When it
            does not, the conditional covariance is factorized once.
        seed: Seed for conditional-covariance draws.
    """

    products: NativeFunctionalProducts
    residual: xr.DataArray
    solve: CovarianceSolve
    description: str
    noise_varies_by_draw: bool = True
    seed: int = 0


def dense_solve(covariance: xr.DataArray | FloatArray) -> CovarianceSolve:
    """Return a draw-independent solve with one dense observation covariance.

    Args:
        covariance: Positive-definite ``A + R`` ordered like the residual's
            observation axis.

    Returns:
        A solve that factorizes ``covariance`` once.

    Raises:
        numpy.linalg.LinAlgError: If ``covariance`` is not positive definite.
    """
    factor = cho_factor(np.asarray(covariance, dtype=np.float64), lower=True)

    def solve(rhs: FloatArray, chain: int, draw: int) -> FloatArray:
        """Solve with the shared factor."""
        return np.asarray(cho_solve(factor, rhs), dtype=np.float64)

    return solve


def fixed_ou_solve(
    covariance: FixedOuLowRank,
    site_amplitude: xr.DataArray,
    *,
    site_dim: str,
) -> CovarianceSolve:
    """Return a per-draw solve with the fit's fixed-OU covariance.

    ``covariance`` already holds the fit's low-rank-plus-diagonal ``A`` and
    fixed noise variance; each draw adds its own OU site amplitudes through
    :meth:`FixedOuLowRank.solve`. To condition on an exact dense ``A``
    instead, prepare ``covariance`` with a full-width factor of that ``A``.

    Args:
        covariance: The prepared fixed-OU covariance of the fit.
        site_amplitude: Posterior site amplitudes on
            ``(chain, draw, site_dim)``, labelled by the covariance's sites.
        site_dim: Site dimension of ``site_amplitude``.

    Returns:
        A solve indexed by positional chain and draw.

    Raises:
        ValueError: If the amplitude sites differ from the covariance's.
    """
    labels = [str(label) for label in site_amplitude[site_dim].values]
    if sorted(labels) != sorted(covariance.site_labels):
        raise ValueError("Site amplitudes must use the fixed-OU covariance's site labels.")
    amplitudes = np.asarray(
        site_amplitude.sel({site_dim: list(covariance.site_labels)}).transpose(*SAMPLE_DIMS, site_dim).values,
        dtype=np.float64,
    )

    def solve(rhs: FloatArray, chain: int, draw: int) -> FloatArray:
        """Solve with this draw's site amplitudes."""
        return covariance.solve(rhs, amplitudes[chain, draw]).solution

    return solve


def complete_functional_draws(
    centred_state: xr.DataArray,
    products: NativeFunctionalProducts,
    *,
    residual: xr.DataArray | None = None,
    solve: CovarianceSolve | None = None,
    noise_varies_by_draw: bool = True,
    rng: np.random.Generator | None = None,
) -> xr.DataArray:
    """Reconstruct complete native functional draws from retained-state draws.

    This is an eager numerical boundary with one solve per draw. Without a
    residual the draws are prior draws: the unresolved part is independent of
    the retained state. With a residual, each draw is conditioned on the data.
    Inputs are borrowed.

    Args:
        centred_state: Retained-state draws ``alpha - Pi m`` on
            ``(chain, draw, state_dim)``, with the products' state labels.
        products: Functional products of the reduction the fit used.
        residual: Posterior residual draws ``y - E[y | alpha, beta]`` on
            ``(chain, draw, observation_dim)``, or ``None`` for prior draws.
        solve: ``(A + R)^-1`` per draw; required with ``residual``.
        noise_varies_by_draw: Whether ``solve`` depends on the draw.
        rng: Generator for one draw from each conditional covariance. With
            ``None`` only conditional means are added, so the result's spread
            omits the unresolved conditional variance.

    Returns:
        Draws of ``Q x`` on ``(chain, draw, functional_dim)`` in the
        functional's units. ``attrs["conditional_covariance_min_eigenvalue"]``
        records the most negative eigenvalue clipped before drawing,
        relative to the largest; values well below zero mean the solve's
        ``A`` is not the ``A`` of the products.

    Raises:
        ValueError: If labels or sample axes differ, or ``residual`` is given
            without ``solve``.
    """
    functional_dim = products.functional_dim
    state_dim = products.state_dim
    response = products.retained_response
    state = centred_state.transpose(*SAMPLE_DIMS, state_dim)
    if not same_index(state.indexes[state_dim], response.indexes[state_dim]):
        raise ValueError("Retained-state draws and functional products have different state labels.")
    retained = products.functional_mean + xr.dot(response, state.astype("float64"), dim=state_dim)
    retained = retained.transpose(*SAMPLE_DIMS, functional_dim)
    values = np.array(retained.values, dtype=np.float64)
    chains, draws, count = values.shape
    unresolved = np.asarray(products.unresolved_covariance.values, dtype=np.float64)
    min_eigenvalue = 0.0

    if residual is None:
        if rng is not None:
            root, min_eigenvalue = _covariance_root(unresolved)
            values += rng.standard_normal(values.shape) @ root.T
    else:
        if solve is None:
            raise ValueError("Conditioning on a residual requires an observation-covariance solve.")
        observation_dim = products.observation_dim
        residual = residual.transpose(*SAMPLE_DIMS, observation_dim)
        cross = products.unresolved_observation_covariance
        if not same_index(residual.indexes[observation_dim], cross.indexes[observation_dim]):
            raise ValueError("Residual and functional products have different observation labels.")
        for dim in SAMPLE_DIMS:
            if residual.sizes[dim] != state.sizes[dim] or not np.array_equal(residual[dim], state[dim]):
                raise ValueError(f"Residual and retained-state draws must share the {dim!r} axis.")
        residual_values = np.asarray(residual.values, dtype=np.float64)
        cross_values = np.asarray(cross.values, dtype=np.float64)
        root = None
        for chain in range(chains):
            for draw in range(draws):
                draw_residual = residual_values[chain, draw][:, None]
                need_covariance = rng is not None and (root is None or noise_varies_by_draw)
                rhs = np.column_stack([draw_residual, cross_values.T]) if need_covariance else draw_residual
                solution = solve(rhs, chain, draw)
                values[chain, draw] += cross_values @ solution[:, 0]
                if need_covariance:
                    conditional = unresolved - cross_values @ solution[:, 1:]
                    root, eigenvalue = _covariance_root(conditional)
                    min_eigenvalue = min(min_eigenvalue, eigenvalue)
                if rng is not None and root is not None:
                    values[chain, draw] += root @ rng.standard_normal(count)

    return (
        retained.copy(data=values)
        .rename("complete_functional")
        .assign_attrs(
            products.functional_mean.attrs,
            mathematical_name="Q x",
            conditional_covariance_min_eigenvalue=min_eigenvalue,
        )
    )


def _covariance_root(covariance: FloatArray) -> tuple[FloatArray, float]:
    """Return a square root of a symmetric covariance, clipping negative eigenvalues.

    Returns:
        ``S`` with ``S S.T`` the clipped covariance, and the most negative
        eigenvalue relative to the largest absolute eigenvalue (zero if none).
    """
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    relative = float(min(0.0, eigenvalues.min() / scale)) if scale > 0.0 else 0.0
    return vectors * np.sqrt(np.clip(eigenvalues, 0.0, None)), relative
