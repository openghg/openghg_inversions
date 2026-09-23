"""Affine retained-state reconstruction of native scaling and flux fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import xarray as xr
from openghg.util import cf_ureg  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import require_unique_index, same_index

from .operators import BucketBasisOperator, MultiSourceBucketBasisOperator


RETAINED_STATE_CONDITIONAL = "retained_state_conditional"


def _require_same_axis(
    array: xr.DataArray,
    dim: str,
    expected: pd.Index,
    *,
    name: str,
) -> None:
    """Require the exact ordered labels of a canonical axis."""
    if not same_index(require_unique_index(array, dim, name=name), expected):
        raise ValueError(f"{name} {dim!r} labels must exactly match the prolongation.")


def _dimensionless_scale(array: xr.DataArray, *, name: str) -> float:
    """Return the Pint conversion factor from input units to dimensionless."""
    actual = array.attrs.get("units")
    if not isinstance(actual, str) or not actual.strip():
        raise ValueError(f"{name} requires non-empty units.")
    try:
        return float(cf_ureg.Quantity(cf_ureg.parse_expression(actual)).to("dimensionless").magnitude)
    except Exception as exc:
        raise ValueError(f"{name} units {actual!r} are incompatible with dimensionless scaling.") from exc


def _in_dimensionless_units(array: xr.DataArray, *, name: str) -> xr.DataArray:
    """Convert compatible scaling values lazily while borrowing unit-one arrays."""
    scale = _dimensionless_scale(array, name=name)
    return array if scale == 1.0 else array * scale


@dataclass(frozen=True, slots=True, eq=False)
class AffineFluxMap:
    """Reconstruct retained-state-conditional native scaling and flux means.

    ``state_to_native`` evaluates ``m + U* (alpha - alpha_ref)`` and
    ``state_to_flux`` additionally multiplies the result by signed reference
    flux ``F``. The caller supplies the authoritative ``alpha_ref`` explicitly.

    The value borrows its inputs and does not materialize their payloads on
    construction. A bucket prolongation remains a :class:`BasisOperator`;
    for a supplied restriction, the resulting exact ``U*`` is a labelled
    native-by-state :class:`xarray.DataArray`.

    Args:
        native_mean: Dimensionless native scaling mean ``m``.
        flux: Signed reference flux ``F`` containing every native dimension.
        prolongation: Bucket operator or explicit dimensionless ``U*``.
        state_dim: Retained-state dimension.
        uncertainty_scope: Fixed machine-readable scope of reconstructed data.
    """

    native_mean: xr.DataArray
    flux: xr.DataArray
    prolongation: BucketBasisOperator | MultiSourceBucketBasisOperator | xr.DataArray
    state_dim: str
    uncertainty_scope: Literal["retained_state_conditional"] = RETAINED_STATE_CONDITIONAL

    @property
    def native_dims(self) -> tuple[str, ...]:
        """Return the ordered native dimensions labelled by ``native_mean``."""
        return self.native_mean.dims

    def __post_init__(self) -> None:
        if not self.native_dims or len(set(self.native_dims)) != len(self.native_dims):
            raise ValueError("native_dims must contain unique dimension names.")
        if self.state_dim in self.native_dims:
            raise ValueError("state_dim must be distinct from native_dims.")
        if self.state_dim in self.flux.dims:
            raise ValueError("flux must not use the retained-state dimension.")
        if self.uncertainty_scope != RETAINED_STATE_CONDITIONAL:
            raise ValueError(
                f"AffineFluxMap uncertainty_scope must be {RETAINED_STATE_CONDITIONAL!r}."
            )
        for dim in self.native_dims:
            native_index = require_unique_index(self.native_mean, dim, name="native_mean")
            _require_same_axis(self.flux, dim, native_index, name="flux")
        _dimensionless_scale(self.native_mean, name="native_mean")
        flux_units = self.flux.attrs.get("units")
        if not isinstance(flux_units, str) or not flux_units.strip():
            raise ValueError("flux requires non-empty units.")
        try:
            cf_ureg.parse_expression(flux_units)
        except Exception as exc:
            raise ValueError(f"flux units {flux_units!r} are invalid.") from exc

        if isinstance(self.prolongation, xr.DataArray):
            prolongation = self.prolongation
            if prolongation.dims != (*self.native_dims, self.state_dim):
                raise ValueError(
                    "prolongation must have ordered dimensions "
                    f"{(*self.native_dims, self.state_dim)!r}; got {prolongation.dims!r}."
                )
            _dimensionless_scale(prolongation, name="prolongation")
        elif isinstance(self.prolongation, (BucketBasisOperator, MultiSourceBucketBasisOperator)):
            operator = self.prolongation
            if operator.meta.state_dim != self.state_dim:
                raise ValueError("Bucket prolongation state dimension must match AffineFluxMap.state_dim.")
            if isinstance(operator, MultiSourceBucketBasisOperator):
                if self.native_dims[1:] != operator.meta.grid_dims or self.native_dims[0] == operator.source_dim:
                    raise ValueError("Multisource bucket prolongation requires distinct native source and grid axes.")
                native_source = self.native_dims[0]
                if native_source in operator.basis_matrix.coords:
                    raise ValueError("Native source dimension collides with a retained-state coordinate.")
                if list(self.native_mean.indexes[native_source]) != list(operator.source_labels):
                    raise ValueError("Native source order must exactly match bucket prolongation sources.")
            elif self.native_dims != operator.meta.grid_dims:
                raise ValueError("Bucket prolongation native_dims must match its grid dimensions.")
            prolongation = operator.basis_matrix
            if "units" in prolongation.attrs:
                _dimensionless_scale(prolongation, name="prolongation")
        else:
            raise TypeError(
                "prolongation must be a BucketBasisOperator, "
                "MultiSourceBucketBasisOperator, or labelled DataArray."
            )
        for dim in prolongation.dims:
            require_unique_index(prolongation, dim, name="prolongation")
        for dim in self.native_dims:
            if dim in prolongation.dims:
                _require_same_axis(prolongation, dim, self.native_mean.indexes[dim], name="prolongation")

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
            result = self.prolongation.native_prolongation(
                self.native_mean,
                native_dims=self.native_dims,
            )
            attrs = dict(result.attrs)
            attrs.setdefault("units", "1")
            return result.assign_attrs(attrs)
        raise AssertionError("Constructor validates the closed prolongation representations.")

    def _centred_state(
        self,
        state: xr.DataArray,
        reference_state: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validate independent state inputs and return centred state with ``U*``."""
        retained = (
            self.prolongation
            if isinstance(self.prolongation, xr.DataArray)
            else self.prolongation.basis_matrix
        )
        state_index = retained.indexes[self.state_dim]
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
        state_scale = _dimensionless_scale(state, name="state")
        reference_scale = _dimensionless_scale(reference_state, name="reference_state")
        occupied: set[str] = set()
        for array in (self.native_mean, self.flux, retained, state, reference_state):
            occupied.update(array.dims)
            occupied.update(array.coords)
        renames: dict[str, str] = {}
        for dim in state.dims:
            if dim == self.state_dim or (dim not in self.native_dims and dim not in self.flux.dims):
                continue
            base = dim if dim.startswith("state_") else f"state_{dim}"
            candidate = base
            suffix = 2
            while candidate in occupied:
                candidate = f"{base}_{suffix}"
                suffix += 1
            renames[dim] = candidate
            occupied.add(candidate)
        centred_state = state.rename(renames)
        if state_scale != 1.0:
            centred_state = centred_state * state_scale
        centred_reference = reference_state if reference_scale == 1.0 else reference_state * reference_scale
        prolongation = self._prolongation_array()
        return centred_state - centred_reference, _in_dimensionless_units(prolongation, name="prolongation")

    def state_to_native(
        self,
        state: xr.DataArray,
        *,
        reference_state: xr.DataArray,
    ) -> xr.DataArray:
        """Reconstruct the retained-state-conditional native scaling mean.

        Args:
            state: Dimensionless retained state ``alpha`` with the map's exact
                state labels. Other dimensions are state-input axes.
            reference_state: Dimensionless authoritative ``alpha_ref`` with
                only the retained-state dimension and the same state labels.

        Returns:
            Native scaling ``m + U* (alpha - alpha_ref)`` in dimensionless
            units, preserving all non-state axes and carrying
            ``retained_state_conditional`` scope. A state-input axis whose
            name collides with a native or flux dimension is renamed to
            ``state_<axis>`` or a numbered variant if that name is taken.

        Raises:
            ValueError: If state labels or units are incompatible, or the
                reference state has incompatible dimensions or labels.
        """
        centred, prolongation = self._centred_state(state, reference_state)
        reconstructed = _in_dimensionless_units(self.native_mean, name="native_mean") + xr.dot(
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
        """Reconstruct the retained-state-conditional signed flux mean.

        Args:
            state: Dimensionless retained state ``alpha`` with the map's exact
                state labels. Other dimensions are state-input axes.
            reference_state: Dimensionless authoritative ``alpha_ref`` with
                only the retained-state dimension and the same state labels.

        Returns:
            Signed flux ``F [m + U* (alpha - alpha_ref)]`` in the reference
            flux units, preserving all non-state axes and carrying
            ``retained_state_conditional`` scope. A state-input axis whose
            name collides with a native or flux dimension is renamed to
            ``state_<axis>`` or a numbered variant if that name is taken.

        Raises:
            ValueError: If state labels or units are incompatible, or the
                reference state has incompatible dimensions or labels.
        """
        native = self.state_to_native(state, reference_state=reference_state)
        native, flux = xr.align(native, self.flux, join="exact", copy=False)
        return (flux * native).rename("flux").assign_attrs(
            units=str(self.flux.attrs["units"]),
            uncertainty_scope=self.uncertainty_scope,
        )


__all__ = [
    "AffineFluxMap",
    "RETAINED_STATE_CONDITIONAL",
]
