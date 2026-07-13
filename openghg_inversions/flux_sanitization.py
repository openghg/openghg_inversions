"""Helpers for applying a consistent non-finite flux policy.

OpenGHG flux data can contain ``NaN`` or infinite values even though those
values do not have a scientific interpretation for flux weighting. This module
normalises those values to zero early enough to protect generated and retained
basis workflows.

The sanitation policy is stored in one namespaced xarray attribute as compact
JSON. xarray attrs are the metadata channel that survives NetCDF-like
serialisation, but they should contain simple serialisable values rather than
Python objects. Keeping the metadata behind a small dataclass gives callers a
typed interface without spreading many string attribute names through the code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from typing import Any, Literal, TypeVar
import warnings

import numpy as np
import xarray as xr

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


@dataclass(frozen=True, slots=True)
class FluxNonFiniteMetadata:
    """Machine-readable metadata describing non-finite flux handling.

    The dataclass is the in-memory representation used by this module. Its
    serialised form is a JSON object stored under ``NONFINITE_METADATA_ATTR`` so
    the metadata can survive NetCDF/Zarr roundtrips without placing a Python
    object in ``DataArray.attrs``.

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

    schema_version: int = 1
    policy: str = NONFINITE_POLICY_ZERO_FILL
    fill_value: float = 0.0
    checked: str = NONFINITE_CHECKED_NOT_COUNTED
    context: str | None = None
    source: str | None = None
    count: int | None = None
    total: int | None = None
    fraction: float | None = None

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> "FluxNonFiniteMetadata | None":
        """Parse local non-finite metadata from xarray attrs.

        Args:
            attrs: Attribute mapping from a DataArray or Dataset.

        Returns:
            Parsed metadata when local JSON metadata is present, otherwise
            ``None``.
        """
        raw_metadata = attrs.get(NONFINITE_METADATA_ATTR)
        if raw_metadata is not None:
            return cls.from_json(raw_metadata)
        return None

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
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            try:
                data = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid non-finite flux metadata JSON: {value!r}") from exc
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise ValueError(f"Unsupported non-finite flux metadata value: {value!r}")

        if not isinstance(data, Mapping):
            raise ValueError(f"Non-finite flux metadata must be a JSON object: {value!r}")

        context = data.get("context")
        source = data.get("source")
        count = data.get("count")
        total = data.get("total")
        fraction = data.get("fraction")
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            policy=str(data.get("policy", NONFINITE_POLICY_ZERO_FILL)),
            fill_value=float(data.get("fill_value", 0.0)),
            checked=str(data.get("checked", NONFINITE_CHECKED_NOT_COUNTED)),
            context=None if context is None else str(context),
            source=None if source is None else str(source),
            count=None if count is None else int(str(count)),
            total=None if total is None else int(str(total)),
            fraction=None if fraction is None else float(str(fraction)),
        )

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
        return asdict(self)

    def to_json(self) -> str:
        """Return compact JSON suitable for storing in xarray attrs.

        Returns:
            JSON object encoded as a compact string.
        """
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))


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
    count = int(count_value.item())
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
        checked=checked,
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
