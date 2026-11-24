"""Fixed correlation between the independent errors of paired observation rows.

Some observations come in pairs whose independent errors are correlated, for
example same-hour CO2 and delta(O2/N2) rows at one site. For a pair ``(a, b)``
with standard deviations ``s_a`` and ``s_b`` and correlation ``rho``, the
independent covariance block becomes::

    [[s_a**2,          rho * s_a * s_b],
     [rho * s_a * s_b, s_b**2         ]]

The cross term ``rho * s_a * s_b`` is in the product of the two rows' units, so
pairs may mix units. Likelihoods use the exact split::

    block = (1 - |rho|) * diag(s_a**2, s_b**2) + |rho| * v v^T,
    v = (s_a, sign(rho) * s_b)

so the paired term is a positive diagonal plus one rank-one column per pair.
This keeps dense, low-rank and fixed-OU likelihoods exact while recorded
standard deviations and aggregation covariance stay unchanged.
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


@dataclass(frozen=True)
class PairedErrorCorrelation:
    """Fixed correlations between disjoint pairs of observation rows.

    Attributes:
        first: Row position of the first member of each pair.
        second: Row position of the second member of each pair.
        correlation: Correlation of each pair's independent errors, strictly
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

    def reduction(self, sd: TensorVariable) -> TensorVariable:
        """Return ``|rho| * sd**2`` on paired rows, zero elsewhere."""
        return pt.abs(self.correlation) * sd**2

    def covariance(self, sd: TensorVariable) -> TensorVariable:
        """Return the dense zero-diagonal cross covariance for tensor ``sd``."""
        size = sd.shape[0]
        rows = np.concatenate([self.pairs.first, self.pairs.second])
        columns = np.concatenate([self.pairs.second, self.pairs.first])
        values = self.correlation[rows] * sd[rows] * sd[columns]
        return pt.set_subtensor(pt.zeros((size, size), dtype=values.dtype)[rows, columns], values)

    def factor(self, sd: TensorVariable) -> TensorVariable:
        """Return one rank-one column per pair of nonzero correlation for tensor ``sd``."""
        size = sd.shape[0]
        keep = self.pairs.correlated()
        first, second = self.pairs.first[keep], self.pairs.second[keep]
        columns = np.arange(first.size)
        rho = self.correlation[first]
        scale = pt.sqrt(pt.abs(rho))
        result = pt.zeros((size, first.size), dtype=sd.dtype)
        result = pt.set_subtensor(result[first, columns], scale * sd[first])
        return pt.set_subtensor(result[second, columns], scale * pt.sgn(rho) * sd[second])


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
    correlation = xr.DataArray(
        floatX(pairs.row_correlation(size)), dims=(output_dim,), coords=coords, name=PAIRED_ERROR_CORRELATION
    )
    partner = xr.DataArray(pairs.partner(size), dims=(output_dim,), coords=coords, name=PAIRED_ERROR_PARTNER)
    registered = add_model_data(correlation)
    add_model_data(partner)
    return RegisteredPairedError(pairs=pairs, correlation=registered)


__all__ = [
    "PAIRED_ERROR_CORRELATION",
    "PAIRED_ERROR_PARTNER",
    "PairedErrorCorrelation",
    "RegisteredPairedError",
    "add_paired_error_data",
]
