"""Helpers for applying a consistent non-finite flux policy.

OpenGHG flux data can contain ``NaN`` or infinite values even though those
values do not have a scientific interpretation for flux weighting. This module
normalises those values to zero early enough to protect generated and retained
basis workflows.

The sanitation policy is stored in one namespaced xarray attribute as compact
JSON. xarray attrs are the metadata channel that survives NetCDF-like
serialisation, but they should contain simple serialisable values rather than
Python objects. Keeping the metadata behind a small Pydantic model gives callers a
typed interface without spreading many string attribute names through the code.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from math import isclose
from typing import Annotated, Any, Literal, TypeVar
import warnings

import numpy as np
import xarray as xr
from pydantic import Field, model_validator

from openghg_inversions._record_schema import JsonFloat, Record, SchemaVersion
from openghg_inversions.array_ops import to_dense

FluxNonFiniteCheck = Literal["lazy", "count"]
_XarrayObjectT = TypeVar("_XarrayObjectT", xr.DataArray, xr.Dataset)

# xarray persists attrs to NetCDF/Zarr metadata, so this key is namespaced to
# avoid collisions with CF/OpenGHG attrs while keeping the value JSON-serialisable.
NONFINITE_METADATA_ATTR = "openghg_inversions:non_finite_flux"

NONFINITE_POLICY_ZERO_FILL = "zero_fill"
NONFINITE_CHECKED_NOT_COUNTED = "not_counted"
NONFINITE_CHECKED_COMPUTED = "computed"


class NonFiniteFluxWarning(UserWarning):
    """Warning emitted when non-finite flux values are replaced."""


class FluxNonFiniteMetadata(Record):
    """Machine-readable metadata describing non-finite flux handling.

    The same Pydantic model validates in-memory and persisted metadata. Its
    serialised form is a JSON object stored under ``NONFINITE_METADATA_ATTR`` so
    the metadata can survive NetCDF/Zarr roundtrips without placing a Python
    object in ``DataArray.attrs``.

    Computed evidence requires count, total and fraction, with count no larger
    than total and fraction equal to their ratio within relative tolerance
    ``1e-12``. Empty data has count and fraction zero. Uncounted evidence carries
    none of these values. These relationships are validated without reading flux
    data; generated JSON Schema describes the individual fields only.

    Attributes:
        schema_version: JSON metadata schema version.
        policy: Non-finite handling policy. Currently only ``"zero_fill"`` is
            written locally.
        fill_value: Value used to replace non-finite cells.
        checked: Whether the array was lazily sanitised without counting values
            or audited with an exact count.
        context: Human-readable processing context where the policy was
            applied.
        source: Optional flux source label.
        count: Exact non-finite count when ``checked`` is ``"computed"``.
        total: Exact total cell count when ``checked`` is ``"computed"``.
        fraction: Exact non-finite fraction when ``checked`` is ``"computed"``.
    """

    schema_version: SchemaVersion
    policy: str = NONFINITE_POLICY_ZERO_FILL
    fill_value: JsonFloat = 0.0
    checked: Literal["not_counted", "computed"] = NONFINITE_CHECKED_NOT_COUNTED
    context: str | None = None
    source: str | None = None
    count: Annotated[int, Field(ge=0)] | None = None
    total: Annotated[int, Field(ge=0)] | None = None
    fraction: Annotated[JsonFloat, Field(ge=0, le=1)] | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "FluxNonFiniteMetadata":
        """Keep counted evidence coherent without inspecting the flux array."""
        if self.checked == NONFINITE_CHECKED_NOT_COUNTED:
            if any(value is not None for value in (self.count, self.total, self.fraction)):
                raise ValueError("Uncounted flux metadata cannot contain count, total or fraction.")
            return self
        if self.count is None or self.total is None or self.fraction is None:
            raise ValueError("Computed flux metadata requires count, total and fraction.")
        if self.count > self.total:
            raise ValueError("Non-finite count cannot exceed the total cell count.")
        expected = self.count / self.total if self.total else 0.0
        if not isclose(self.fraction, expected, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("Non-finite fraction must equal count / total (zero for empty data).")
        return self

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> "FluxNonFiniteMetadata | None":
        """Parse local non-finite metadata from xarray attrs.

        Args:
            attrs: Attribute mapping from a DataArray or Dataset.

        Returns:
            Parsed metadata when local JSON metadata is present, otherwise
            ``None``.

        Raises:
            ValueError: If the attribute is present but malformed, including
                a null value or missing/unsupported schema version.
        """
        return cls.from_json(attrs[NONFINITE_METADATA_ATTR]) if NONFINITE_METADATA_ATTR in attrs else None

    @classmethod
    def from_json(cls, value: object) -> "FluxNonFiniteMetadata":
        """Parse metadata from a JSON string or mapping.

        Args:
            value: JSON string, bytes, or mapping stored in xarray attrs.

        Returns:
            Parsed metadata.

        Raises:
            ValueError: If the stored value is not valid metadata.
        """
        return cls.model_validate_json(value) if isinstance(value, str | bytes) else cls.model_validate(value)

    def declares_zero_fill(self) -> bool:
        """Return whether this metadata declares a zero-fill policy.

        Returns:
            ``True`` when the policy and fill value describe replacement with
            zero.
        """
        return self.policy == NONFINITE_POLICY_ZERO_FILL and float(self.fill_value) == 0.0

    def to_json_dict(self) -> dict[str, str | int | float | None]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary containing only primitive or null values suitable for
            JSON encoding in xarray attrs.
        """
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Return compact JSON suitable for storing in xarray attrs.

        Returns:
            JSON object encoded as a compact string.
        """
        return self.model_dump_json()


def _history_entry(
    *,
    context: str,
    source: str | None,
    checked: str,
    count: int | None,
    total: int | None,
) -> str:
    """Build a compact CF-style history entry for flux sanitation.

    Args:
        context: Processing context where sanitation was applied.
        source: Optional flux source label.
        checked: Check mode recorded in metadata.
        count: Exact non-finite count when ``checked`` is ``"computed"``.
        total: Exact total cell count when ``checked`` is ``"computed"``.

    Returns:
        A UTC timestamped history line describing either lazy replacement or
        exact counted replacement.
    """
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    source_part = f" source={source}" if source is not None else ""
    if checked == NONFINITE_CHECKED_COMPUTED and count is not None and total is not None:
        count_part = f" replaced {count}/{total} non-finite values"
    else:
        count_part = " applied lazy non-finite replacement without counting values"
    return (
        f"{timestamp} OpenGHG Inversions:{source_part}{count_part}; "
        f"policy={NONFINITE_POLICY_ZERO_FILL}; fill_value=0.0; context={context}."
    )


def _nonfinite_count(finite: xr.DataArray, total: int) -> tuple[int, int, float]:
    """Compute non-finite count metadata from a Boolean finite mask.

    Args:
        finite: Boolean mask with ``True`` where flux values are finite.
        total: Total number of cells represented by ``finite``.

    Returns:
        Tuple of ``(count, total, fraction)`` for non-finite cells.

    Notes:
        This function calls ``compute()`` on the mask reduction, so callers
        should only use it for explicit audit/count mode.
    """
    count_value = (~finite).sum().compute()
    count = int(to_dense(count_value).item())
    fraction = float(count / total) if total else 0.0
    return count, total, fraction


def _with_nonfinite_attrs(
    flux: xr.DataArray,
    *,
    context: str,
    source: str | None,
    checked: str,
    count: int | None = None,
    total: int | None = None,
    fraction: float | None = None,
) -> xr.DataArray:
    """Return a shallow copy with non-finite policy attrs attached.

    Args:
        flux: Flux DataArray receiving metadata. Data values are not changed by
            this helper.
        context: Processing context where sanitation was applied.
        source: Optional flux source label.
        checked: Check mode recorded in metadata.
        count: Exact non-finite count when ``checked`` is ``"computed"``.
        total: Exact total cell count when ``checked`` is ``"computed"``.
        fraction: Exact non-finite fraction when ``checked`` is ``"computed"``.

    Returns:
        A shallow copy of ``flux`` with JSON metadata attached under
        ``NONFINITE_METADATA_ATTR`` and a new CF-style ``history`` entry
        appended.
    """
    result = flux.copy(deep=False)
    attrs = dict(flux.attrs)
    metadata = FluxNonFiniteMetadata(
        schema_version=1, checked=checked,
        context=context,
        source=source,
        count=count,
        total=total,
        fraction=fraction,
    )
    attrs[NONFINITE_METADATA_ATTR] = metadata.to_json()

    history_entry = _history_entry(
        context=context,
        source=source,
        checked=checked,
        count=count,
        total=total,
    )
    existing_history = attrs.get("history")
    attrs["history"] = f"{existing_history}\n{history_entry}" if existing_history else history_entry
    result.attrs = attrs
    return result


def sanitize_flux_nonfinite(
    flux: xr.DataArray,
    *,
    context: str,
    source: str | None = None,
    check: FluxNonFiniteCheck = "lazy",
    trust_attrs: bool = True,
    warn: bool = False,
) -> xr.DataArray:
    """Replace non-finite flux values with zero and record the policy.

    The default ``check="lazy"`` creates a lazy xarray/dask graph and does not
    compute a count. Use ``check="count"`` when exact non-finite counts are
    needed for audit logs, accepting that this computes the finite mask.

    Args:
        flux: Flux field to sanitize.
        context: Human-readable processing context used in metadata and
            warnings.
        source: Optional flux source label to record in metadata.
        check: ``"lazy"`` to avoid computing counts, or ``"count"`` to compute
            exact count/total/fraction metadata.
        trust_attrs: If ``True``, return flux unchanged when local attrs already
            declare a zero-fill policy. A previous lazy replacement cannot be
            upgraded to an exact count because the original values are gone.
        warn: Emit a warning when values are replaced or a lazy fallback is
            applied.

    Returns:
        Sanitized flux with non-finite values replaced by ``0.0`` and local
        JSON policy metadata attached.

    Raises:
        ValueError: If ``check`` is not ``"lazy"`` or ``"count"``.

    Warns:
        NonFiniteFluxWarning: If values cannot be counted because a lazy
            replacement already occurred, or if ``warn`` is ``True`` and values
            are replaced or a lazy fallback is applied.
    """
    if check not in ("lazy", "count"):
        raise ValueError("`check` must be either 'lazy' or 'count'.")

    metadata = FluxNonFiniteMetadata.from_attrs(flux.attrs)
    if trust_attrs and metadata is not None and metadata.declares_zero_fill():
        if check == "count" and metadata.checked != NONFINITE_CHECKED_COMPUTED:
            warnings.warn(
                (
                    f"Flux {source or '<unknown>'} was already zero-filled without counting; "
                    "the original non-finite count cannot be recovered."
                ),
                NonFiniteFluxWarning,
                stacklevel=2,
            )
        return flux

    finite = xr.apply_ufunc(np.isfinite, flux, dask="allowed")
    count: int | None = None
    total: int | None = None
    fraction: float | None = None
    checked = NONFINITE_CHECKED_NOT_COUNTED
    if check == "count":
        total = int(flux.size)
        count, total, fraction = _nonfinite_count(finite, total)
        checked = NONFINITE_CHECKED_COMPUTED

    sanitized = flux.where(finite, 0.0)
    if np.issubdtype(flux.dtype, np.floating):
        sanitized = sanitized.astype(flux.dtype)
    sanitized = _with_nonfinite_attrs(
        sanitized,
        context=context,
        source=source,
        checked=checked,
        count=count,
        total=total,
        fraction=fraction,
    )

    if warn:
        if check == "count" and count:
            warnings.warn(
                (
                    f"Flux {source or '<unknown>'} contains {count} non-finite values "
                    f"out of {total}; replacing them with 0.0 for {context}."
                ),
                NonFiniteFluxWarning,
                stacklevel=2,
            )
        elif check == "lazy":
            warnings.warn(
                (
                    f"Flux {source or '<unknown>'} has not declared a non-finite policy; "
                    f"applying a lazy zero-fill guard for {context}."
                ),
                NonFiniteFluxWarning,
                stacklevel=2,
            )

    return sanitized


def copy_flux_nonfinite_attrs(target: _XarrayObjectT, flux: xr.Dataset | xr.DataArray) -> _XarrayObjectT:
    """Copy machine-readable non-finite flux metadata onto an output object.

    Args:
        target: Dataset or DataArray receiving metadata.
        flux: Dataset or DataArray whose attrs may contain local non-finite flux
            metadata.

    Returns:
        A shallow copy of ``target`` with local JSON non-finite metadata copied
        from ``flux`` when present.
    """
    metadata = FluxNonFiniteMetadata.from_attrs(flux.attrs)
    if metadata is None:
        return target

    result = target.copy(deep=False)
    result.attrs = dict(target.attrs)
    result.attrs[NONFINITE_METADATA_ATTR] = metadata.to_json()
    return result
