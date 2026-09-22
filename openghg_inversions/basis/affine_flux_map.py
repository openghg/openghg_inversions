"""Affine retained-state reconstruction of native scaling and flux fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd
import xarray as xr
from openghg.util import cf_ureg  # pyright: ignore[reportPrivateImportUsage]

from .operators import BucketBasisOperator, MultiSourceBucketBasisOperator


RETAINED_STATE_CONDITIONAL = "retained_state_conditional"
BucketProlongation: TypeAlias = BucketBasisOperator | MultiSourceBucketBasisOperator
Prolongation: TypeAlias = BucketProlongation | xr.DataArray


def _require_axis(array: xr.DataArray, dim: str, *, name: str) -> pd.Index:
    """Return a unique indexed dimension owned by an independent input."""
    if dim not in array.dims or dim not in array.indexes:
        raise ValueError(f"{name} requires a labelled {dim!r} dimension.")
    index = array.indexes[dim]
    if not index.is_unique:
        raise ValueError(f"{name} {dim!r} labels must be unique.")
    return index


def _same_index(left: pd.Index, right: pd.Index) -> bool:
    """Compare labels and MultiIndex level names without positional fallback."""
    if not left.equals(right):
        return False
    if isinstance(left, pd.MultiIndex) or isinstance(right, pd.MultiIndex):
        return (
            isinstance(left, pd.MultiIndex)
            and isinstance(right, pd.MultiIndex)
            and left.names == right.names
        )
    return True


def _require_same_axis(
    array: xr.DataArray,
    dim: str,
    expected: pd.Index,
    *,
    name: str,
) -> None:
    """Require the exact ordered labels of a canonical axis."""
    if not _same_index(_require_axis(array, dim, name=name), expected):
        raise ValueError(f"{name} {dim!r} labels must exactly match the prolongation.")


def _require_unit_scale(array: xr.DataArray, expected: str, *, name: str) -> None:
    """Require units compatible with ``expected`` at the same numeric scale."""
    actual = array.attrs.get("units")
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError(f"{name} requires non-empty units.")
    try:
        actual_quantity = cf_ureg.Quantity(cf_ureg.parse_expression(actual))
        expected_quantity = cf_ureg.Quantity(cf_ureg.parse_expression(expected))
        scale = float(actual_quantity.to(expected_quantity.units).magnitude)
        expected_scale = float(expected_quantity.magnitude)
    except Exception as exc:
        raise ValueError(f"{name} units {actual!r} are incompatible with {expected!r}.") from exc
    if scale != expected_scale:
        raise ValueError(f"{name} units {actual!r} do not have the same numeric scale as {expected!r}.")


@dataclass(frozen=True, slots=True, eq=False)
class AffineFluxMap:
    """Reconstruct retained-state-conditional native scaling and flux means.

    The value borrows its xarray objects. Construction and ordinary access do
    not copy, compute, persist, densify, or rechunk their payloads. A bucket
    prolongation remains a :class:`BasisOperator`; supplied restrictions use
    an explicit labelled native-by-state :class:`xarray.DataArray`.

    ``state_to_native`` evaluates ``m + U* (alpha - alpha_ref)`` and
    ``state_to_flux`` additionally multiplies the result by signed reference
    flux ``F``. The caller supplies the authoritative ``alpha_ref`` explicitly.

    Args:
        native_mean: Dimensionless native scaling mean ``m``.
        flux: Signed reference flux ``F`` containing every native dimension.
        prolongation: Bucket operator or explicit dimensionless ``U*``.
        native_dims: Ordered dimensions of the native scaling state.
        state_dim: Retained-state dimension.
        uncertainty_scope: Fixed machine-readable scope of reconstructed data.
    """

    native_mean: xr.DataArray
    flux: xr.DataArray
    prolongation: Prolongation
    native_dims: tuple[str, ...]
    state_dim: str
    uncertainty_scope: Literal["retained_state_conditional"] = RETAINED_STATE_CONDITIONAL

    def __post_init__(self) -> None:
        if not self.native_dims or len(set(self.native_dims)) != len(self.native_dims):
            raise ValueError("native_dims must contain unique dimension names.")
        if self.state_dim in self.native_dims:
            raise ValueError("state_dim must be distinct from native_dims.")
        if self.uncertainty_scope != RETAINED_STATE_CONDITIONAL:
            raise ValueError(
                f"AffineFluxMap uncertainty_scope must be {RETAINED_STATE_CONDITIONAL!r}."
            )
        if self.native_mean.dims != self.native_dims:
            raise ValueError(
                f"native_mean must have dimensions {self.native_dims!r}; "
                f"got {self.native_mean.dims!r}."
            )
        for dim in self.native_dims:
            native_index = _require_axis(self.native_mean, dim, name="native_mean")
            _require_same_axis(self.flux, dim, native_index, name="flux")
        _require_unit_scale(self.native_mean, "1", name="native_mean")
        flux_units = self.flux.attrs.get("units")
        if not isinstance(flux_units, str) or not flux_units.strip():
            raise ValueError("flux requires non-empty units.")

        prolongation = self._prolongation_array()
        if prolongation.dims != (*self.native_dims, self.state_dim):
            raise ValueError(
                "prolongation must have ordered dimensions "
                f"{(*self.native_dims, self.state_dim)!r}; got {prolongation.dims!r}."
            )
        for dim in self.native_dims:
            _require_same_axis(
                prolongation,
                dim,
                self.native_mean.indexes[dim],
                name="prolongation",
            )
        _require_axis(prolongation, self.state_dim, name="prolongation")
        _require_unit_scale(prolongation, "1", name="prolongation")

    @property
    def representation(self) -> Literal["bucket", "explicit"]:
        """Return the closed prolongation representation kind."""
        return "explicit" if isinstance(self.prolongation, xr.DataArray) else "bucket"

    def _prolongation_array(self) -> xr.DataArray:
        """Expose labelled ``U*`` without caching or changing its backend."""
        if isinstance(self.prolongation, xr.DataArray):
            return self.prolongation
        if isinstance(
            self.prolongation,
            (BucketBasisOperator, MultiSourceBucketBasisOperator),
        ):
            if self.prolongation.meta.state_dim != self.state_dim:
                raise ValueError(
                    "Bucket prolongation state dimension must match AffineFluxMap.state_dim."
                )
            result = self.prolongation.native_prolongation(
                self.native_mean,
                native_dims=self.native_dims,
            )
            attrs = dict(result.attrs)
            attrs.setdefault("units", "1")
            return result.assign_attrs(attrs)
        raise TypeError(
            "prolongation must be a BucketBasisOperator, "
            "MultiSourceBucketBasisOperator, or labelled DataArray."
        )

    def _centred_state(
        self,
        state: xr.DataArray,
        reference_state: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validate independent state inputs and return centred state with ``U*``."""
        prolongation = self._prolongation_array()
        state_index = prolongation.indexes[self.state_dim]
        _require_same_axis(state, self.state_dim, state_index, name="state")
        if reference_state.dims != (self.state_dim,):
            raise ValueError(
                f"reference_state must have dimensions {(self.state_dim,)!r}; "
                f"got {reference_state.dims!r}."
            )
        _require_same_axis(
            reference_state,
            self.state_dim,
            state_index,
            name="reference_state",
        )
        _require_unit_scale(state, "1", name="state")
        _require_unit_scale(reference_state, "1", name="reference_state")
        return state - reference_state, prolongation

    def state_to_native(
        self,
        state: xr.DataArray,
        *,
        reference_state: xr.DataArray,
    ) -> xr.DataArray:
        """Return ``m + U* (alpha - alpha_ref)`` with sample dimensions preserved."""
        centred, prolongation = self._centred_state(state, reference_state)
        reconstructed = self.native_mean + xr.dot(
            prolongation,
            centred,
            dim=self.state_dim,
        )
        return reconstructed.rename("native_scaling").assign_attrs(
            units="1",
            uncertainty_scope=self.uncertainty_scope,
        )

    def state_to_flux(
        self,
        state: xr.DataArray,
        *,
        reference_state: xr.DataArray,
    ) -> xr.DataArray:
        """Return ``F [m + U* (alpha - alpha_ref)]`` at the grid product boundary."""
        native = self.state_to_native(state, reference_state=reference_state)
        native, flux = xr.align(native, self.flux, join="exact", copy=False)
        return (flux * native).rename("flux").assign_attrs(
            units=str(self.flux.attrs["units"]),
            uncertainty_scope=self.uncertainty_scope,
        )


__all__ = [
    "AffineFluxMap",
    "BucketProlongation",
    "Prolongation",
    "RETAINED_STATE_CONDITIONAL",
]
