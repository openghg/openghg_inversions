"""Shared normalization helpers for options aligned to requested sites."""

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from numbers import Integral
from typing import Any, cast


def expand_site_option(
    value: Iterable[Any] | str | slice | int | Integral | None,
    *,
    nsites: int,
    name: str,
) -> tuple[Any, ...]:
    """Broadcast a scalar or copy an iterable to one value per site.

    Args:
        value: A string, integer, slice, or ``None`` scalar to broadcast, or
            an iterable containing one value per site.
        nsites: Number of requested sites.
        name: Option name used in validation errors.

    Returns:
        An immutable tuple containing exactly ``nsites`` values.

    Raises:
        ValueError: If ``nsites`` is negative, ``value`` is unsupported, or
            an iterable does not contain exactly ``nsites`` values.
    """
    if nsites < 0:
        raise ValueError(f"`nsites` must be non-negative, got {nsites}.")

    if value is None or isinstance(value, str | slice):
        return (value,) * nsites
    if isinstance(value, Integral) and not isinstance(value, bool):
        return (int(value),) * nsites
    if isinstance(value, bool | bytes | Mapping | AbstractSet):
        raise ValueError(f"`{name}` must be a scalar string/integer/slice, a site-aligned iterable, or None.")

    try:
        values = tuple(cast(Iterable[Any], value))
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be a scalar string/integer/slice, a site-aligned iterable, or None."
        ) from exc

    if len(values) != nsites:
        raise ValueError(f"List {name} does not have specified length: {len(values)} != {nsites}.")
    return values


def expand_site_boolean_option(
    value: Iterable[bool | None] | bool | None,
    *,
    nsites: int,
    name: str,
) -> tuple[bool | None, ...]:
    """Broadcast an optional boolean or validate one value per site.

    Boolean site options are kept separate from :func:`expand_site_option`,
    whose general scalar contract deliberately rejects booleans as ambiguous
    integer-like values.
    """
    if nsites < 0:
        raise ValueError(f"`nsites` must be non-negative, got {nsites}.")

    if value is None or isinstance(value, bool):
        return (value,) * nsites
    if isinstance(value, str | bytes | Mapping | AbstractSet):
        raise ValueError(f"`{name}` must be a boolean, a site-aligned iterable of booleans, or None.")

    try:
        values = tuple(cast(Iterable[Any], value))
    except TypeError as exc:
        raise ValueError(
            f"`{name}` must be a boolean, a site-aligned iterable of booleans, or None."
        ) from exc

    if len(values) != nsites:
        raise ValueError(f"List {name} does not have specified length: {len(values)} != {nsites}.")
    invalid = [item for item in values if item is not None and not isinstance(item, bool)]
    if invalid:
        raise ValueError(f"`{name}` entries must be booleans or None. Invalid value(s): {invalid!r}.")
    return cast(tuple[bool | None, ...], values)


def is_column_observation(inlet: object, platform: object) -> bool:
    """Return whether one inlet/platform pair explicitly selects column data."""
    return isinstance(inlet, str) and inlet.lower() == "column" or is_column_platform(platform)


def is_column_platform(platform: object) -> bool:
    """Return whether a platform name selects satellite or site-column data."""
    return isinstance(platform, str) and platform.lower() in {"satellite", "site-column"}


def is_satellite_platform(platform: object) -> bool:
    """Return whether a platform name selects satellite data."""
    return isinstance(platform, str) and platform.lower() == "satellite"
