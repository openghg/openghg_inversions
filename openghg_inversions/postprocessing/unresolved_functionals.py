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
from numpy.typing import ArrayLike, NDArray
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
        reduction_identity: The
            :func:`~openghg_inversions.coherent_reduction.reduction_identity`
            of the reduction the fit was prepared from, as recorded at
            preparation. It must equal the products' identity.
        marginalised: Boolean state mask of states the fit integrated into
            ``A`` and holds fixed, or ``None`` when the fit samples every
            state. It must equal the fit's inactive states.
        seed: Seed for conditional-covariance draws.
    """

    products: NativeFunctionalProducts
    residual: xr.DataArray
    solve: ObservationCovarianceSolve
    reduction_identity: str
    marginalised: xr.DataArray | None = None
    seed: int = 0


def dense_solve(covariance: xr.DataArray, *, description: str = "dense A + R") -> ObservationCovarianceSolve:
    """Return a draw-independent solve with one dense observation covariance.

    Args:
        covariance: Positive-definite ``A + R`` on a labelled observation row
            axis and a distinct column axis carrying the same labels, in any
            order. Columns are aligned to the row order before factorizing.
        description: Account of the covariance for output metadata.

    Returns:
        A solve that factorizes ``covariance`` once.

    Raises:
        numpy.linalg.LinAlgError: If ``covariance`` is not positive definite.
        ValueError: If either axis is unlabelled or not unique, the column
            labels are not the row labels, or the aligned matrix is not
            symmetric.
    """
    if covariance.ndim != 2:
        raise ValueError("The observation covariance must be a two-dimensional matrix.")
    observation_dim, column_dim = (str(dim) for dim in covariance.dims)
    index = require_unique_index(covariance, observation_dim, name="Observation covariance rows")
    columns = require_unique_index(covariance, column_dim, name="Observation covariance columns")
    order = columns.get_indexer(index)
    if len(columns) != len(index) or np.any(order < 0):
        raise ValueError("Observation covariance columns must carry exactly the row labels.")
    values = np.asarray(covariance.isel({column_dim: order}).values, dtype=np.float64)
    scale = float(np.max(np.abs(values), initial=0.0))
    if np.max(np.abs(values - values.T), initial=0.0) > 1e-10 * scale:
        raise ValueError(
            "The observation covariance must be symmetric after aligning its columns to its rows."
        )
    factor = cho_factor((values + values.T) * 0.5, lower=True)

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
    site_index: ArrayLike | None = None,
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
        site_index: Integer site of each row of ``observation_index``,
            numbered like ``covariance.site_labels``, for example the fit's
            ``ou_site_index`` constant data. Defaults to the codes of the
            index's ``site`` level when it is a MultiIndex.
        description: Account of the covariance for output metadata.

    Returns:
        A per-draw solve.

    Raises:
        ValueError: If the sites or the number of observations differ from
            the covariance's, or the per-row sites differ from the
            covariance's row sites. Within-site row order cannot be checked
            because the covariance does not keep observation times.
    """
    labels = [str(label) for label in site_amplitude[site_dim].values]
    if sorted(labels) != sorted(covariance.site_labels):
        raise ValueError("Site amplitudes must use the fixed-OU covariance's site labels.")
    if len(observation_index) != covariance.n_observation or not observation_index.is_unique:
        raise ValueError("observation_index must label each fixed-OU covariance row once.")
    if site_index is not None:
        rows = np.asarray(site_index)
    elif isinstance(observation_index, pd.MultiIndex) and "site" in observation_index.names:
        codes = {label: code for code, label in enumerate(covariance.site_labels)}
        try:
            rows = np.array([codes[str(site)] for site in observation_index.get_level_values("site")])
        except KeyError as error:
            raise ValueError(
                f"Observation site {error.args[0]!r} is not a fixed-OU covariance site."
            ) from error
    else:
        raise ValueError("Pass site_index, or an observation_index with a 'site' level.")
    if rows.shape != covariance.site_index.shape or not np.array_equal(rows, covariance.site_index):
        raise ValueError("The observation rows' sites differ from the fixed-OU covariance's row sites.")
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
    clipping_tolerance: float = 1e-6,
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
        clipping_tolerance: Largest magnitude of a negative eigenvalue,
            relative to the largest absolute eigenvalue of the same
            covariance, that is clipped to zero before drawing. Roundoff is
            far smaller; a larger negative eigenvalue means the solve's ``A``
            is not the ``A`` of the products, and raises.

    Returns:
        Draws of ``Q x`` on ``(chain, draw, functional_dim)`` in the
        functional's units. ``attrs["conditional_covariance_min_eigenvalue"]``
        records the most negative relative eigenvalue clipped before drawing
        (zero when none was negative).

    Raises:
        ValueError: If labels, units or sample axes differ, ``residual`` is
            given without ``solve``, or a covariance to draw from has a
            relative negative eigenvalue beyond ``clipping_tolerance``.
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
            root, min_eigenvalue = _covariance_root(unresolved, clipping_tolerance, "Q B_perp Q.T")
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
                    root, eigenvalue = _covariance_root(
                        conditional,
                        clipping_tolerance,
                        f"the conditional covariance of draw {draw_labels[draw]!r}",
                    )
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


def _covariance_root(covariance: FloatArray, tolerance: float, name: str) -> tuple[FloatArray, float]:
    """Return a square root of a symmetric covariance, clipping roundoff-sized negative eigenvalues.

    Returns:
        ``S`` with ``S S.T`` the clipped covariance, and the most negative
        eigenvalue relative to the largest absolute eigenvalue (zero if none).

    Raises:
        ValueError: If a relative negative eigenvalue exceeds ``tolerance``.
    """
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    relative = float(min(0.0, eigenvalues.min() / scale)) if scale > 0.0 else 0.0
    if relative < -tolerance:
        raise ValueError(
            f"{name} has a negative eigenvalue of {relative:.3e} relative to its largest, beyond the "
            f"clipping tolerance {tolerance:.1e}; the observation-covariance solve probably uses a "
            "different A from the functional products."
        )
    return vectors * np.sqrt(np.clip(eigenvalues, 0.0, None)), relative
