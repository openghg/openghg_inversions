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

The observation covariance enters only through a labelled solve, so the same
reconstruction works with a dense ``A + R`` (:func:`dense_solve`), with the
fit's low-rank-plus-diagonal ``A`` and fixed-OU noise
(:func:`fixed_ou_solve`), or with any other representation the caller has
factorized.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from openghg.util import cf_ureg  # pyright: ignore[reportPrivateImportUsage]
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.array_ops import require_unique_index, same_index

if TYPE_CHECKING:
    from openghg_inversions.coherent_reduction import NativeFunctionalProducts
    from openghg_inversions.models.fixed_ou import FixedOuLowRank


__all__ = [
    "ObservationCovarianceSolve",
    "UnresolvedFunctionalConditioning",
    "complete_functional_draws",
    "dense_solve",
    "fixed_ou_solve",
]

FloatArray = NDArray[np.float64]
SAMPLE_DIMS = ("chain", "draw")


@dataclass(frozen=True, slots=True, eq=False)
class ObservationCovarianceSolve:
    """``(A + R)^-1`` for each posterior draw, on labelled observations.

    Attributes:
        observation_index: Observation labels in the row order ``solve``
            expects; residuals are checked against it.
        solve: Returns ``(A + R)^-1 rhs`` for the draw with the given chain
            and draw labels.
        varies_by_draw: Whether the covariance depends on the draw. When it
            does not, the conditional covariance is factorized once.
        description: Account of ``A`` and ``R`` recorded in output metadata.
    """

    observation_index: pd.Index
    solve: Callable[[FloatArray, Hashable, Hashable], FloatArray]
    varies_by_draw: bool
    description: str


@dataclass(frozen=True, slots=True, eq=False)
class UnresolvedFunctionalConditioning:
    """Inputs that condition unresolved native functionals on posterior draws.

    Attributes:
        products: Full-restriction functional products of the reduction the
            fit used, before any marginalisation.
        residual: Posterior draws of the observation residual
            ``y - E[y | alpha, beta]`` on ``(chain, draw, observation_dim)``,
            from the same draws as the retained states, in the units of
            ``H``.
        solve: ``(A + R)^-1`` for each posterior draw, with the ``A`` and
            noise model of the fit's likelihood.
        marginalised: Boolean state mask of states the fit integrated into
            ``A`` and holds fixed, or ``None`` when the fit samples every
            state. It must equal the fit's inactive states.
        seed: Seed for conditional-covariance draws.
    """

    products: NativeFunctionalProducts
    residual: xr.DataArray
    solve: ObservationCovarianceSolve
    marginalised: xr.DataArray | None = None
    seed: int = 0


def dense_solve(covariance: xr.DataArray, *, description: str = "dense A + R") -> ObservationCovarianceSolve:
    """Return a draw-independent solve with one dense observation covariance.

    Args:
        covariance: Positive-definite ``A + R`` whose first dimension is the
            labelled observation axis.
        description: Account of the covariance for output metadata.

    Returns:
        A solve that factorizes ``covariance`` once.

    Raises:
        numpy.linalg.LinAlgError: If ``covariance`` is not positive definite.
        ValueError: If the observation axis is unlabelled or not unique.
    """
    observation_dim = str(covariance.dims[0])
    index = require_unique_index(covariance, observation_dim, name="Observation covariance")
    factor = cho_factor(np.asarray(covariance.values, dtype=np.float64), lower=True)

    def solve(rhs: FloatArray, chain: Hashable, draw: Hashable) -> FloatArray:
        """Solve with the shared factor."""
        return np.asarray(cho_solve(factor, rhs), dtype=np.float64)

    return ObservationCovarianceSolve(index, solve, varies_by_draw=False, description=description)


def fixed_ou_solve(
    covariance: FixedOuLowRank,
    site_amplitude: xr.DataArray,
    *,
    site_dim: str,
    observation_index: pd.Index,
    description: str = "low-rank-plus-diagonal A + fixed-OU noise at each draw's amplitudes",
) -> ObservationCovarianceSolve:
    """Return a per-draw solve with the fit's fixed-OU covariance.

    ``covariance`` already holds the fit's low-rank-plus-diagonal ``A`` and
    fixed noise variance; each draw adds its own OU site amplitudes through
    :meth:`FixedOuLowRank.solve`. To condition on an exact dense ``A``
    instead, prepare ``covariance`` with a full-width factor of that ``A``.

    Args:
        covariance: The prepared fixed-OU covariance of the fit.
        site_amplitude: Posterior site amplitudes on
            ``(chain, draw, site_dim)``, labelled by the covariance's sites.
            Draws are selected by chain and draw label.
        site_dim: Site dimension of ``site_amplitude``.
        observation_index: Labels of the observation rows ``covariance`` was
            prepared with, in its row order.
        description: Account of the covariance for output metadata.

    Returns:
        A per-draw solve.

    Raises:
        ValueError: If the sites or the number of observations differ from
            the covariance's.
    """
    labels = [str(label) for label in site_amplitude[site_dim].values]
    if sorted(labels) != sorted(covariance.site_labels):
        raise ValueError("Site amplitudes must use the fixed-OU covariance's site labels.")
    if len(observation_index) != covariance.n_observation or not observation_index.is_unique:
        raise ValueError("observation_index must label each fixed-OU covariance row once.")
    ordered = site_amplitude.sel({site_dim: list(covariance.site_labels)}).transpose(*SAMPLE_DIMS, site_dim)
    amplitudes = np.asarray(ordered.values, dtype=np.float64)
    chains = {label: position for position, label in enumerate(ordered["chain"].values.tolist())}
    draws = {label: position for position, label in enumerate(ordered["draw"].values.tolist())}

    def solve(rhs: FloatArray, chain: Hashable, draw: Hashable) -> FloatArray:
        """Solve with this draw's site amplitudes."""
        try:
            amplitude = amplitudes[chains[chain], draws[draw]]
        except KeyError as error:
            raise ValueError(f"No site amplitude for chain {chain!r}, draw {draw!r}.") from error
        return covariance.solve(rhs, amplitude).solution

    return ObservationCovarianceSolve(observation_index, solve, varies_by_draw=True, description=description)


def complete_functional_draws(
    centred_state: xr.DataArray,
    products: NativeFunctionalProducts,
    *,
    residual: xr.DataArray | None = None,
    solve: ObservationCovarianceSolve | None = None,
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
        products: Functional products of the reduction the fit used, already
            marginalised if the fit marginalised states.
        residual: Posterior residual draws ``y - E[y | alpha, beta]`` on
            ``(chain, draw, observation_dim)``, or ``None`` for prior draws.
            When ``H_alpha`` records units, the residual must carry the same
            units.
        solve: ``(A + R)^-1`` per draw; required with ``residual``.
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
        ValueError: If labels, units or sample axes differ, or ``residual``
            is given without ``solve``.
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
        _require_residual_units(residual, products.effective_observation_operator)
        cross = products.unresolved_observation_covariance
        observation_index = cross.indexes[observation_dim]
        if not same_index(residual.indexes[observation_dim], observation_index):
            raise ValueError("Residual and functional products have different observation labels.")
        if not same_index(solve.observation_index, observation_index):
            raise ValueError("The observation-covariance solve uses different observation labels or order.")
        for dim in SAMPLE_DIMS:
            if not np.array_equal(residual[dim].values, state[dim].values):
                raise ValueError(f"Residual and retained-state draws must share the {dim!r} labels.")
        residual_values = np.asarray(residual.values, dtype=np.float64)
        cross_values = np.asarray(cross.values, dtype=np.float64)
        chain_labels = state["chain"].values.tolist()
        draw_labels = state["draw"].values.tolist()
        root = None
        for chain in range(chains):
            for draw in range(draws):
                draw_residual = residual_values[chain, draw][:, None]
                need_covariance = rng is not None and (root is None or solve.varies_by_draw)
                rhs = np.column_stack([draw_residual, cross_values.T]) if need_covariance else draw_residual
                solution = solve.solve(rhs, chain_labels[chain], draw_labels[draw])
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


def _require_residual_units(residual: xr.DataArray, effective_operator: xr.DataArray) -> None:
    """Require residual units equal to the recorded ``H`` units at the same numeric scale."""
    expected = effective_operator.attrs.get("units")
    if expected is None:
        return
    actual = residual.attrs.get("units")
    if actual is None:
        raise ValueError(f"Residual draws must carry units compatible with H ({expected!r}).")
    try:
        factor = cf_ureg.Quantity(1.0, actual).to(expected).magnitude
    except Exception as error:
        raise ValueError(f"Residual units {actual!r} are incompatible with H units {expected!r}.") from error
    if not np.isclose(factor, 1.0, rtol=1e-12, atol=0.0):
        raise ValueError(f"Residual units {actual!r} differ in scale from H units {expected!r}.")


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
