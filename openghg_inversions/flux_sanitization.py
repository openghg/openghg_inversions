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
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Literal
import warnings

import numpy as np
import xarray as xr

FluxNonFiniteCheck = Literal["lazy", "count"]

# xarray persists attrs to NetCDF/Zarr metadata, so this key is namespaced to
# avoid collisions with CF/OpenGHG attrs while keeping the value JSON-serialisable.
NONFINITE_METADATA_ATTR = "openghg_inversions:non_finite_flux"

NONFINITE_POLICY_ZERO_FILL = "zero_fill"
NONFINITE_CHECKED_NOT_COUNTED = "not_counted"
NONFINITE_CHECKED_COMPUTED = "computed"

_ZERO_FILL_POLICY_VALUES = {"zero_fill", "fill_zero", "replace_with_zero"}
_UPSTREAM_POLICY_ATTRS = (
    "openghg:missing_value_policy",
    "openghg:non_finite_policy",
)


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

        return cls(
            schema_version=int(data.get("schema_version", 1)),
            policy=str(data.get("policy", NONFINITE_POLICY_ZERO_FILL)),
            fill_value=float(data.get("fill_value", 0.0)),
            checked=str(data.get("checked", NONFINITE_CHECKED_NOT_COUNTED)),
            context=_optional_str(data.get("context")),
            source=_optional_str(data.get("source")),
            count=_optional_int(data.get("count")),
            total=_optional_int(data.get("total")),
            fraction=_optional_float(data.get("fraction")),
        )

    def declares_zero_fill(self) -> bool:
        """Return whether this metadata declares a zero-fill policy."""
        return _normalise_policy(self.policy) in _ZERO_FILL_POLICY_VALUES and float(self.fill_value) == 0.0

    def to_json_dict(self) -> dict[str, str | int | float | None]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary containing only primitive or null values suitable for
            JSON encoding in xarray attrs.
        """
        data: dict[str, str | int | float | None] = {
            "schema_version": int(self.schema_version),
            "policy": self.policy,
            "fill_value": float(self.fill_value),
            "checked": self.checked,
            "context": self.context,
            "source": self.source,
            "count": None,
            "total": None,
            "fraction": None,
        }
        if self.checked == NONFINITE_CHECKED_COMPUTED:
            data["count"] = int(self.count or 0)
            data["total"] = int(self.total or 0)
            data["fraction"] = float(self.fraction or 0.0)
        return data

    def to_json(self) -> str:
        """Return compact JSON suitable for storing in xarray attrs."""
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))


def _optional_str(value: object) -> str | None:
    """Coerce optional attr values to strings."""
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    """Coerce optional attr values to integers."""
    return None if value is None else int(str(value))


def _optional_float(value: object) -> float | None:
    """Coerce optional attr values to floats."""
    return None if value is None else float(str(value))


def _normalise_policy(value: object) -> str:
    """Normalize policy text for attribute comparisons."""
    return str(value).strip().lower().replace("-", "_")


def attrs_declare_zero_filled_nonfinite(attrs: Mapping[Any, Any]) -> bool:
    """Return true if attrs declare a trusted zero-fill non-finite policy.

    Args:
        attrs: Attribute mapping from a DataArray or Dataset.

    Returns:
        ``True`` when local JSON metadata or recognised upstream OpenGHG
        metadata declares that non-finite flux values have already been
        zero-filled.
    """
    local_metadata = FluxNonFiniteMetadata.from_attrs(attrs)
    if local_metadata is not None and local_metadata.declares_zero_fill():
        return True

    for attr in _UPSTREAM_POLICY_ATTRS:
        upstream_policy = attrs.get(attr)
        if upstream_policy is not None and _normalise_policy(upstream_policy) in _ZERO_FILL_POLICY_VALUES:
            return True

    return False


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


def _append_history(attrs: dict[Any, Any], entry: str) -> None:
    """Append a history entry while preserving any existing value."""
    existing = attrs.get("history")
    attrs["history"] = f"{existing}\n{entry}" if existing else entry


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

    _append_history(
        attrs,
        _history_entry(
            context=context,
            source=source,
            checked=checked,
            count=count,
            total=total,
        ),
    )
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
        trust_attrs: If ``True``, return flux unchanged when attrs already
            declare a zero-fill policy. ``check="count"`` still audits local
            metadata that has not already recorded computed counts.
        warn: Emit a warning when values are replaced or a lazy fallback is
            applied.

    Returns:
        Sanitized flux with non-finite values replaced by ``0.0`` and local
        JSON policy metadata attached.

    Raises:
        ValueError: If ``check`` is not ``"lazy"`` or ``"count"``.

    Warns:
        NonFiniteFluxWarning: If ``warn`` is ``True`` and values are counted as
        non-finite in count mode, or when lazy fallback sanitation is applied.
    """
    if check not in ("lazy", "count"):
        raise ValueError("`check` must be either 'lazy' or 'count'.")

    metadata = FluxNonFiniteMetadata.from_attrs(flux.attrs)
    if (
        trust_attrs
        and attrs_declare_zero_filled_nonfinite(flux.attrs)
        and (check == "lazy" or (metadata is not None and metadata.checked == NONFINITE_CHECKED_COMPUTED))
    ):
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
                    f"lazily replacing any non-finite values with 0.0 for {context}."
                ),
                NonFiniteFluxWarning,
                stacklevel=2,
            )

    return sanitized


def sanitize_fp_all_fluxes(
    fp_all: dict,
    *,
    context: str,
    check: FluxNonFiniteCheck = "lazy",
    trust_attrs: bool = True,
    warn: bool = False,
) -> dict:
    """Sanitize all flux entries in an ``fp_all`` mapping.

    Args:
        fp_all: Legacy footprint/flux mapping. Flux entries are expected under
            ``fp_all[".flux"]`` as objects with a ``data`` dataset containing a
            ``"flux"`` variable.
        context: Human-readable processing context used in metadata and
            warnings.
        check: ``"lazy"`` to avoid computing counts, or ``"count"`` to compute
            exact count/total/fraction metadata.
        trust_attrs: If ``True``, skip entries that already declare a zero-fill
            policy.
        warn: Emit warnings for replacements or lazy fallbacks.

    Returns:
        The input mapping, modified in place where flux entries are found.
    """
    flux_entries = fp_all.get(".flux")
    if not isinstance(flux_entries, Mapping):
        return fp_all

    for source, flux_data in flux_entries.items():
        data = getattr(flux_data, "data", None)
        if not isinstance(data, xr.Dataset) or "flux" not in data:
            continue
        data["flux"] = sanitize_flux_nonfinite(
            data["flux"],
            context=context,
            source=str(source),
            check=check,
            trust_attrs=trust_attrs,
            warn=warn,
        )

    return fp_all


def copy_flux_nonfinite_attrs(
    target: xr.Dataset | xr.DataArray, flux: xr.Dataset | xr.DataArray
) -> xr.Dataset | xr.DataArray:
    """Copy machine-readable non-finite flux metadata onto an output object.

    Args:
        target: Dataset or DataArray receiving metadata.
        flux: Dataset or DataArray whose attrs may contain local non-finite flux
            metadata.

    Returns:
        A shallow copy of ``target`` with local JSON non-finite metadata copied
        from ``flux`` when present.
    """
    result = target.copy(deep=False)
    result.attrs = dict(target.attrs)
    metadata = FluxNonFiniteMetadata.from_attrs(flux.attrs)
    if metadata is not None:
        result.attrs[NONFINITE_METADATA_ATTR] = metadata.to_json()
    return result
