"""Helpers for applying a consistent non-finite flux policy."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal
import warnings

import numpy as np
import xarray as xr

FluxNonFiniteCheck = Literal["lazy", "count"]

NONFINITE_POLICY_ATTR = "openghg_inversions:non_finite_policy"
NONFINITE_FILL_VALUE_ATTR = "openghg_inversions:non_finite_fill_value"
NONFINITE_CHECKED_ATTR = "openghg_inversions:non_finite_checked"
NONFINITE_COUNT_ATTR = "openghg_inversions:non_finite_count"
NONFINITE_TOTAL_ATTR = "openghg_inversions:non_finite_total"
NONFINITE_FRACTION_ATTR = "openghg_inversions:non_finite_fraction"
NONFINITE_CONTEXT_ATTR = "openghg_inversions:non_finite_context"
NONFINITE_SOURCE_ATTR = "openghg_inversions:non_finite_source"

NONFINITE_POLICY_ZERO_FILL = "zero_fill"
NONFINITE_CHECKED_NOT_COUNTED = "not_counted"
NONFINITE_CHECKED_COMPUTED = "computed"

_ZERO_FILL_POLICY_VALUES = {"zero_fill", "fill_zero", "replace_with_zero"}
_UPSTREAM_POLICY_ATTRS = (
    "openghg:missing_value_policy",
    "openghg:non_finite_policy",
)
_LOCAL_ATTRS = (
    NONFINITE_POLICY_ATTR,
    NONFINITE_FILL_VALUE_ATTR,
    NONFINITE_CHECKED_ATTR,
    NONFINITE_COUNT_ATTR,
    NONFINITE_TOTAL_ATTR,
    NONFINITE_FRACTION_ATTR,
    NONFINITE_CONTEXT_ATTR,
    NONFINITE_SOURCE_ATTR,
)


class NonFiniteFluxWarning(UserWarning):
    """Warning emitted when non-finite flux values are replaced."""


def _normalise_policy(value: object) -> str:
    """Normalize policy text for attribute comparisons."""
    return str(value).strip().lower().replace("-", "_")


def attrs_declare_zero_filled_nonfinite(attrs: Mapping[Any, Any]) -> bool:
    """Return true if attrs declare a trusted zero-fill non-finite policy."""
    local_policy = attrs.get(NONFINITE_POLICY_ATTR)
    if local_policy is not None and _normalise_policy(local_policy) in _ZERO_FILL_POLICY_VALUES:
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
    """Build a compact CF-style history entry for flux sanitation."""
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
    """Compute non-finite count metadata from a Boolean finite mask."""
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
    """Return a shallow copy with non-finite policy attrs attached."""
    result = flux.copy(deep=False)
    attrs = dict(flux.attrs)
    attrs[NONFINITE_POLICY_ATTR] = NONFINITE_POLICY_ZERO_FILL
    attrs[NONFINITE_FILL_VALUE_ATTR] = 0.0
    attrs[NONFINITE_CHECKED_ATTR] = checked
    attrs[NONFINITE_CONTEXT_ATTR] = context
    if source is not None:
        attrs[NONFINITE_SOURCE_ATTR] = source
    if checked == NONFINITE_CHECKED_COMPUTED:
        attrs[NONFINITE_COUNT_ATTR] = int(count or 0)
        attrs[NONFINITE_TOTAL_ATTR] = int(total or 0)
        attrs[NONFINITE_FRACTION_ATTR] = float(fraction or 0.0)
    else:
        for attr in (NONFINITE_COUNT_ATTR, NONFINITE_TOTAL_ATTR, NONFINITE_FRACTION_ATTR):
            attrs.pop(attr, None)

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
    """
    if check not in ("lazy", "count"):
        raise ValueError("`check` must be either 'lazy' or 'count'.")

    if (
        trust_attrs
        and attrs_declare_zero_filled_nonfinite(flux.attrs)
        and (check == "lazy" or flux.attrs.get(NONFINITE_CHECKED_ATTR) == NONFINITE_CHECKED_COMPUTED)
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
    """Sanitize every ``fp_all['.flux']`` entry in place and return ``fp_all``."""
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
    """Copy machine-readable non-finite flux attrs onto an output object."""
    result = target.copy(deep=False)
    result.attrs = dict(target.attrs)
    for attr in _LOCAL_ATTRS:
        if attr in flux.attrs:
            result.attrs[attr] = flux.attrs[attr]
    return result
