"""Fixed correlation between the model errors of paired observation rows.

Some observations come in pairs whose model errors are correlated, for example
same-hour CO2 and delta(O2/N2) rows at one site, where one transport or flux
error moves both channels. The residual covariance is split into named parts::

    Sigma = A + D_obs + Sigma_model(rho) [+ OU]

``A`` is the fixed aggregation covariance and ``D_obs`` the diagonal
measurement (observation) variance; neither is touched by ``rho``. The fixed
model error has a known standard deviation ``m`` per row (the
``fixed_model_mismatch`` term), and for a pair ``(a, b)`` its block is::

    [[m_a**2,          rho * m_a * m_b],
     [rho * m_a * m_b, m_b**2         ]]

The cross term ``rho * m_a * m_b`` is in the product of the two rows' units, so
pairs may mix units (for example ppm and per meg). Fixed-OU likelihoods use
the exact split::

    block = (1 - |rho|) * diag(m_a**2, m_b**2) + |rho| * v v^T,
    v = (m_a, sign(rho) * m_b)

a positive diagonal plus one rank-one column per correlated pair. Generic
Gaussian likelihoods keep each correlated pair as an exact 2x2 block: the
second row is regressed on the first (see
:meth:`RegisteredPairedError.conditional`), so independent and low-rank
likelihoods stay O(n) in the pairs and dense likelihoods add the cross terms
in place. Pairs with ``rho = 0`` take no part in either form, so they leave
the original numerical path unchanged. Recorded standard deviations and
aggregation covariance stay unchanged.

:func:`pair_correlation_diagnostic` reports, pair by pair, the correlation
that ``A`` alone implies and the total correlation of the whole covariance, so
a ``rho`` estimated from residuals can be checked against what ``A`` already
carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymc.pytensorf import floatX
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import add_model_data

PAIRED_ERROR_CORRELATION = "paired_error_correlation"
PAIRED_ERROR_PARTNER = "paired_error_partner"
FIXED_MODEL_MISMATCH = "fixed_model_mismatch"


def validate_fixed_model_mismatch(
    fixed_model_mismatch: xr.DataArray,
    *,
    output_dim: str,
    owner: str,
) -> np.ndarray:
    """Check a known model-error standard deviation and return its float64 values.

    Raises:
        ValueError: If it is not a finite, non-negative numeric vector on ``output_dim``.
    """
    if fixed_model_mismatch.dims != (output_dim,):
        raise ValueError(
            f"{owner} input {FIXED_MODEL_MISMATCH!r} must have dims ({output_dim!r},); "
            f"got {fixed_model_mismatch.dims!r}."
        )
    values = np.asarray(fixed_model_mismatch.values)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{FIXED_MODEL_MISMATCH!r} must be numeric.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{FIXED_MODEL_MISMATCH!r} must contain only finite, non-negative values.")
    return values.astype(np.float64)


def require_model_error_for_pairs(
    pairs: PairedErrorCorrelation | None,
    fixed_model_mismatch: object | None,
    *,
    owner: str,
) -> None:
    """Reject pair correlations without the model error they correlate."""
    if pairs is not None and fixed_model_mismatch is None:
        raise ValueError(
            f"{owner}: paired error correlations act only on the fixed model error, "
            f"so they require {FIXED_MODEL_MISMATCH!r}; measurement and aggregation "
            "errors stay uncorrelated."
        )


@dataclass(frozen=True)
class PairedErrorCorrelation:
    """Fixed correlations between disjoint pairs of observation rows.

    Attributes:
        first: Row position of the first member of each pair.
        second: Row position of the second member of each pair.
        correlation: Correlation of each pair's model errors, strictly
            between -1 and 1.
    """

    first: np.ndarray
    second: np.ndarray
    correlation: np.ndarray

    def __post_init__(self) -> None:
        first = np.asarray(self.first)
        second = np.asarray(self.second)
        correlation = np.asarray(self.correlation, dtype=np.float64)
        if first.ndim != 1 or first.shape != second.shape or first.shape != correlation.shape:
            raise ValueError("Paired error rows and correlations must be equal-length vectors.")
        if first.size and not (
            np.issubdtype(first.dtype, np.integer) and np.issubdtype(second.dtype, np.integer)
        ):
            raise ValueError("Paired error rows must be integer positions.")
        rows = np.concatenate([first, second]).astype(np.int64)
        if np.any(rows < 0) or np.unique(rows).size != rows.size:
            raise ValueError("Each observation row may belong to at most one error pair.")
        if not np.all(np.isfinite(correlation)) or np.any(np.abs(correlation) >= 1.0):
            raise ValueError("Paired error correlations must be finite and strictly between -1 and 1.")
        object.__setattr__(self, "first", first.astype(np.int64))
        object.__setattr__(self, "second", second.astype(np.int64))
        object.__setattr__(self, "correlation", correlation)

    def _check_size(self, size: int) -> None:
        if self.first.size and max(self.first.max(), self.second.max()) >= size:
            raise ValueError("Paired error rows exceed the observation vector.")

    def row_correlation(self, size: int) -> np.ndarray:
        """Return each row's pair correlation, zero for unpaired rows."""
        self._check_size(size)
        result = np.zeros(size, dtype=np.float64)
        result[self.first] = self.correlation
        result[self.second] = self.correlation
        return result

    def partner(self, size: int) -> np.ndarray:
        """Return each row's partner position, or -1 for unpaired rows."""
        self._check_size(size)
        result = np.full(size, -1, dtype=np.int64)
        result[self.first] = self.second
        result[self.second] = self.first
        return result

    def covariance(self, sd: np.ndarray) -> np.ndarray:
        """Return the dense zero-diagonal cross covariance for standard deviations ``sd``."""
        sd = np.asarray(sd, dtype=np.float64)
        self._check_size(sd.size)
        result = np.zeros((sd.size, sd.size), dtype=np.float64)
        values = self.correlation * sd[self.first] * sd[self.second]
        result[self.first, self.second] = values
        result[self.second, self.first] = values
        return result

    def correlated(self) -> np.ndarray:
        """Return the positions of pairs with nonzero correlation, the only ones that need a factor column."""
        return np.flatnonzero(self.correlation != 0.0)

    def split(self, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split the paired block into a variance reduction and rank-one factor columns.

        Returns:
            ``reduction`` (``|rho| * sd**2`` on paired rows, zero elsewhere) and
            ``factor`` with one column per pair of nonzero correlation, so that
            ``diag(sd**2) + covariance(sd) == diag(sd**2 - reduction) + factor @ factor.T``.
        """
        sd = np.asarray(sd, dtype=np.float64)
        self._check_size(sd.size)
        keep = self.correlated()
        first, second, rho = self.first[keep], self.second[keep], self.correlation[keep]
        magnitude = np.abs(rho)
        reduction = np.zeros(sd.size, dtype=np.float64)
        reduction[first] = magnitude * sd[first] ** 2
        reduction[second] = magnitude * sd[second] ** 2
        factor = np.zeros((sd.size, keep.size), dtype=np.float64)
        columns = np.arange(keep.size)
        scale = np.sqrt(magnitude)
        factor[first, columns] = scale * sd[first]
        factor[second, columns] = scale * np.sign(rho) * sd[second]
        return reduction, factor


@dataclass(frozen=True)
class RegisteredPairedError:
    """Paired-error tensors registered with the active model."""

    pairs: PairedErrorCorrelation
    correlation: TensorVariable

    @property
    def correlated(self) -> bool:
        """Whether any pair has a nonzero correlation."""
        return bool(self.pairs.correlated().size)

    def _rows(self) -> tuple[np.ndarray, np.ndarray]:
        keep = self.pairs.correlated()
        return self.pairs.first[keep], self.pairs.second[keep]

    def cross(self, sd: TensorVariable) -> tuple[np.ndarray, np.ndarray, TensorVariable]:
        """Return rows, columns and values of the nonzero symmetric cross covariance for tensor ``sd``."""
        first, second = self._rows()
        rows = np.concatenate([first, second])
        columns = np.concatenate([second, first])
        return rows, columns, self.correlation[rows] * sd[rows] * sd[columns]

    def conditional(
        self, sd: TensorVariable, variance: TensorVariable
    ) -> tuple[np.ndarray, np.ndarray, TensorVariable, TensorVariable]:
        """Factor ``diag(variance) + cross covariance`` into exact 2x2 blocks.

        For a correlated pair ``(p, q)`` with cross covariance ``c``, row ``q``
        is regressed on row ``p``: ``coefficient[q] = c / variance[p]`` and the
        conditional variance of ``q`` is ``variance[q] - c**2 / variance[p]``.
        Every other row keeps ``variance`` and a zero coefficient. Then the
        block covariance is ``L diag(conditional) L^T`` with ``L`` the identity
        plus ``coefficient[q]`` at ``(q, p)``, and its log determinant is
        ``sum(log(conditional))``.

        Returns:
            First and second rows of the correlated pairs, the conditional
            variance and the regression coefficient.
        """
        first, second = self._rows()
        cross = self.correlation[first] * sd[first] * sd[second]
        ratio = cross / variance[first]
        conditional = pt.set_subtensor(variance[second], variance[second] - ratio * cross)
        coefficient = pt.set_subtensor(pt.zeros_like(variance)[second], ratio)
        return first, second, conditional, coefficient


def pair_correlation_diagnostic(
    pairs: PairedErrorCorrelation,
    *,
    aggregation_cross: np.ndarray,
    aggregation_variance: np.ndarray,
    observation_sd: np.ndarray,
    model_error_sd: np.ndarray,
    other_variance: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return the per-pair correlations implied by each part of the covariance.

    For a pair ``(p, q)`` with aggregation covariance ``A``, observation
    standard deviations ``e``, model-error standard deviations ``m`` and
    correlation ``rho``::

        aggregation = A_pq / sqrt(A_pp * A_qq)
        total = (A_pq + rho m_p m_q)
                / sqrt((A_pp + e_p**2 + m_p**2 + o_p) * (A_qq + e_q**2 + m_q**2 + o_q))

    where ``o`` is any other variance on the rows' diagonals (for example the
    OU marginal variance, whose blocks are independent between channels). A
    pair whose ``A_pp`` or ``A_qq`` is zero has a NaN aggregation correlation.

    Args:
        pairs: The paired rows and their model-error correlations.
        aggregation_cross: ``A_pq`` for each pair, in pair order.
        aggregation_variance: The row-aligned diagonal of ``A``.
        observation_sd: Row-aligned observation (measurement) standard deviations.
        model_error_sd: Row-aligned model-error standard deviations.
        other_variance: Optional extra row-aligned variance.

    Returns:
        ``aggregation``, ``model`` (``rho``) and ``total`` correlations per pair.
    """
    first, second = pairs.first, pairs.second
    variance = (
        np.asarray(aggregation_variance, dtype=np.float64)
        + np.square(np.asarray(observation_sd, dtype=np.float64))
        + np.square(np.asarray(model_error_sd, dtype=np.float64))
    )
    if other_variance is not None:
        variance = variance + np.asarray(other_variance, dtype=np.float64)
    model = np.asarray(model_error_sd, dtype=np.float64)
    cross = np.asarray(aggregation_cross, dtype=np.float64)
    aggregation_variance = np.asarray(aggregation_variance, dtype=np.float64)
    scale = np.sqrt(aggregation_variance[first] * aggregation_variance[second])
    with np.errstate(divide="ignore", invalid="ignore"):
        aggregation = np.where(scale > 0.0, cross / scale, np.nan)
    total = (cross + pairs.correlation * model[first] * model[second]) / np.sqrt(
        variance[first] * variance[second]
    )
    return {"aggregation": aggregation, "model": pairs.correlation.copy(), "total": total}


def add_paired_error_data(
    pairs: PairedErrorCorrelation | None,
    observations: xr.DataArray,
    *,
    output_dim: str,
) -> RegisteredPairedError | None:
    """Register row-aligned pair correlations and partners with the active model.

    Returns:
        The registered pairs, or ``None`` when ``pairs`` is ``None`` or empty.
    """
    if pairs is None or pairs.first.size == 0:
        return None
    size = observations.sizes[output_dim]
    coords = {output_dim: observations.coords[output_dim]}
    working = floatX(pairs.row_correlation(size))
    if np.any(np.abs(working) >= 1.0):
        raise ValueError(
            f"Paired error correlations round to +/-1 at the working precision ({working.dtype}); "
            "use a smaller |rho| or floatX='float64'."
        )
    correlation = xr.DataArray(working, dims=(output_dim,), coords=coords, name=PAIRED_ERROR_CORRELATION)
    partner = xr.DataArray(pairs.partner(size), dims=(output_dim,), coords=coords, name=PAIRED_ERROR_PARTNER)
    registered = add_model_data(correlation)
    add_model_data(partner)
    return RegisteredPairedError(pairs=pairs, correlation=registered)


__all__ = [
    "FIXED_MODEL_MISMATCH",
    "PAIRED_ERROR_CORRELATION",
    "PAIRED_ERROR_PARTNER",
    "PairedErrorCorrelation",
    "RegisteredPairedError",
    "add_paired_error_data",
    "pair_correlation_diagnostic",
    "require_model_error_for_pairs",
    "validate_fixed_model_mismatch",
]
