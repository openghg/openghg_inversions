"""Complete site-aligned selectors and their external shorthand normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
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
    return (isinstance(inlet, str) and inlet.lower() == "column") or is_column_platform(platform)


def is_column_platform(platform: object) -> bool:
    """Return whether a platform name selects satellite or site-column data."""
    return isinstance(platform, str) and platform.lower() in {"satellite", "site-column"}


def is_satellite_platform(platform: object) -> bool:
    """Return whether a platform name selects satellite data."""
    return isinstance(platform, str) and platform.lower() == "satellite"


SiteStringOption = Sequence[str | None] | str | None
SiteInletOption = Sequence[str | slice | None] | str | None
SiteIntegerOption = Sequence[int | None] | int | None
SiteBooleanOption = Sequence[bool | None] | bool | None


def _normalise_site_strings(
    value: Sequence[str | None] | str | None,
    *,
    length: int,
    name: str,
) -> list[str | None]:
    """Normalize and validate one optional-string value per requested site."""
    normalized = list(expand_site_option(value, nsites=length, name=name))
    invalid = [item for item in normalized if item is not None and not isinstance(item, str)]
    if invalid:
        raise ValueError(f"`{name}` entries must be strings or None. Invalid value(s): {invalid!r}.")
    return normalized


def _normalise_site_integers(
    value: Sequence[int | None] | int | None,
    *,
    length: int,
    name: str,
) -> list[int | None]:
    """Normalize and validate one optional integer value per requested site."""
    normalized = list(expand_site_option(value, nsites=length, name=name))

    invalid = [
        item
        for item in normalized
        if item is not None and (not isinstance(item, Integral) or isinstance(item, bool))
    ]
    if invalid:
        raise ValueError(f"`{name}` entries must be integers or None. Invalid value(s): {invalid!r}.")
    return [None if item is None else int(item) for item in normalized]


def _normalise_site_inlets(
    value: Sequence[str | slice | None] | str | None,
    *,
    length: int,
) -> list[str | slice | None]:
    """Normalize inlet selectors, including legacy per-site slice selectors."""
    normalized = list(expand_site_option(value, nsites=length, name="inlet"))
    invalid = [item for item in normalized if item is not None and not isinstance(item, str | slice)]
    if invalid:
        raise ValueError(f"`inlet` entries must be strings, slices, or None. Invalid value(s): {invalid!r}.")
    return normalized


def _normalise_site_booleans(
    value: SiteBooleanOption,
    *,
    length: int,
    name: str,
) -> list[bool | None]:
    """Normalize one optional boolean selector per requested site."""
    return list(expand_site_boolean_option(value, nsites=length, name=name))


@dataclass(frozen=True)
class SiteOptions:
    """Complete selectors sharing one site order.

    Every field has the same length and ordering. Selection always creates a
    new complete record so no option can drift independently from its site.

    Direct construction accepts resolved uppercase site labels and aligned
    sequences. It freezes sequences as tuples and checks nonempty unique sites
    and common lengths; it does not normalize labels or validate selector
    entries. Use :meth:`from_inputs` to resolve external shorthand.

    Args:
        sites: Nonempty unique uppercase observation site labels.
        averaging_period: Observation averaging periods in site order;
            ``None`` leaves an individual site's averaging unspecified.
        inlet: Observation inlet selectors, including legacy slices.
        fp_height: Footprint inlet-height selectors.
        instrument: Observation instrument selectors.
        platform: Observation platforms, such as surface or satellite.
        obs_data_level: Observation data-level selectors.
        met_model: Footprint meteorological-model selectors.
        max_level: Maximum column levels, using integers or ``None``.
        time_resolved: Footprint selectors: ``True`` for high-frequency,
            ``False`` for integrated, and ``None`` for unspecified selection.

    Raises:
        ValueError: If sites are empty or duplicated, or field lengths differ.
    """

    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    inlet: tuple[str | slice | None, ...]
    fp_height: tuple[str | None, ...]
    instrument: tuple[str | None, ...]
    platform: tuple[str | None, ...]
    obs_data_level: tuple[str | None, ...]
    met_model: tuple[str | None, ...]
    max_level: tuple[int | None, ...]
    time_resolved: tuple[bool | None, ...]

    def __post_init__(self) -> None:
        """Freeze supplied sequences and enforce the common-length invariant."""
        field_names = (
            "sites",
            "averaging_period",
            "inlet",
            "fp_height",
            "instrument",
            "platform",
            "obs_data_level",
            "met_model",
            "max_level",
            "time_resolved",
        )
        for name in field_names:
            object.__setattr__(self, name, tuple(getattr(self, name)))

        if not self.sites:
            raise ValueError("At least one site must be specified for inversion data preparation.")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError(f"Site names must be unique: {self.sites!r}.")

        expected_length = len(self.sites)
        misaligned = {
            name: len(getattr(self, name))
            for name in field_names[1:]
            if len(getattr(self, name)) != expected_length
        }
        if misaligned:
            raise ValueError(
                "Every site-aligned option must have the same length as `sites`; "
                f"expected {expected_length}, got {misaligned!r}."
            )

    @classmethod
    def from_inputs(
        cls,
        *,
        sites: Sequence[str],
        averaging_period: Sequence[str | None] | str | None,
        inlet: Sequence[str | slice | None] | str | None = None,
        fp_height: Sequence[str | None] | str | None = None,
        instrument: Sequence[str | None] | str | None = None,
        platform: Sequence[str | None] | str | None = None,
        obs_data_level: Sequence[str | None] | str | None = None,
        met_model: Sequence[str | None] | str | None = None,
        max_level: Sequence[int | None] | int | None = None,
        time_resolved: SiteBooleanOption = None,
    ) -> SiteOptions:
        """Normalize all site options and validate their common length.

        Site names are uppercased. Scalar option values are broadcast, while
        sequences must match the number of sites. Inlets also support legacy
        ``slice`` selectors; maximum levels reject booleans.

        Args:
            sites: Requested site-label sequence, with no case-insensitive
                duplicates. A single site must still be supplied as a sequence.
            averaging_period: Observation averaging period, either one value
                for all sites or a site-aligned sequence; ``None`` is unspecified.
            inlet: Scalar or aligned observation inlet selectors. Strings,
                legacy slices and ``None`` are supported.
            fp_height: Scalar or aligned footprint inlet-height selectors.
            instrument: Scalar or aligned observation instrument selectors.
            platform: Scalar or aligned observation platforms.
            obs_data_level: Scalar or aligned observation data levels.
            met_model: Scalar or aligned footprint meteorological models.
            max_level: Scalar or aligned maximum column levels; entries must
                be integers or ``None``, not booleans.
            time_resolved: Scalar or aligned footprint time-resolution choices.
                ``True`` selects high-frequency, ``False`` integrated, and
                ``None`` leaves store selection unspecified.

        Returns:
            Complete uppercase labels and site-aligned selector tuples, with
            caller sequences left unchanged.

        Raises:
            ValueError: If no sites are supplied, site names are duplicated,
                an option has the wrong length, or an entry has an invalid
                type.
        """
        normalized_sites = [site.upper() for site in sites]
        if not normalized_sites:
            raise ValueError("At least one site must be specified for inversion data preparation.")
        if len(set(normalized_sites)) != len(normalized_sites):
            raise ValueError(f"Site names must be unique: {normalized_sites!r}.")
        nsites = len(normalized_sites)
        return cls(
            sites=tuple(normalized_sites),
            averaging_period=tuple(
                _normalise_site_strings(averaging_period, length=nsites, name="averaging_period")
            ),
            inlet=tuple(_normalise_site_inlets(inlet, length=nsites)),
            fp_height=tuple(_normalise_site_strings(fp_height, length=nsites, name="fp_height")),
            instrument=tuple(_normalise_site_strings(instrument, length=nsites, name="instrument")),
            platform=tuple(_normalise_site_strings(platform, length=nsites, name="platform")),
            obs_data_level=tuple(
                _normalise_site_strings(obs_data_level, length=nsites, name="obs_data_level")
            ),
            met_model=tuple(_normalise_site_strings(met_model, length=nsites, name="met_model")),
            max_level=tuple(_normalise_site_integers(max_level, length=nsites, name="max_level")),
            time_resolved=tuple(_normalise_site_booleans(time_resolved, length=nsites, name="time_resolved")),
        )

    def select_indices(self, indices: Sequence[int]) -> SiteOptions:
        """Return a new complete option record in the supplied index order.

        Args:
            indices: Positions in the current site order, using ordinary
                Python sequence indexing.

        Returns:
            A record selecting every field together without changing this one.

        Raises:
            IndexError: If an index is outside the current site sequence.
            ValueError: If selection is empty or repeats a site.
        """

        def select(values: Sequence[Any]) -> tuple[Any, ...]:
            return tuple(values[index] for index in indices)

        return SiteOptions(
            sites=select(self.sites),
            averaging_period=select(self.averaging_period),
            inlet=select(self.inlet),
            fp_height=select(self.fp_height),
            instrument=select(self.instrument),
            platform=select(self.platform),
            obs_data_level=select(self.obs_data_level),
            met_model=select(self.met_model),
            max_level=select(self.max_level),
            time_resolved=select(self.time_resolved),
        )

    @property
    def is_column(self) -> bool:
        """Whether any retained site uses a supported column-data selector."""
        return any(
            is_column_observation(inlet, platform)
            for inlet, platform in zip(self.inlet, self.platform, strict=True)
        )

    def retain_sites(self, retained_sites: Sequence[str], *, context: str) -> SiteOptions:
        """Return options for retained sites in their supplied order.

        Args:
            retained_sites: Nonempty retained labels; matching is case-insensitive
                against this record's resolved uppercase site labels.
            context: Boundary name included in alignment errors.

        Returns:
            A new complete record in retained-site order, leaving this one intact.

        Raises:
            ValueError: If retained names are empty or duplicated, or a retained
                name was not in the original request.
        """
        normalized_retained = [site.upper() for site in retained_sites]
        index_by_site = {site: index for index, site in enumerate(self.sites)}
        if len(index_by_site) != len(self.sites):
            raise ValueError(f"{context} cannot align duplicate requested site names: {self.sites!r}.")

        missing_sites = [site for site in normalized_retained if site not in index_by_site]
        if missing_sites:
            raise ValueError(f"{context} returned site(s) that were not requested: {missing_sites!r}.")
        if len(set(normalized_retained)) != len(normalized_retained):
            raise ValueError(f"{context} returned duplicate site names: {normalized_retained!r}.")

        return self.select_indices([index_by_site[site] for site in normalized_retained])


def convert_to_list(
    x: Iterable[Any] | str | slice | int | Integral | None,
    length: int,
    name: str | None = None,
) -> list[Any]:
    """Convert a scalar or sequence to a list of the expected size.

    Args:
        x: Scalar string/integer/slice/``None`` to broadcast, or an iterable
            to copy.
        length: Required output length.
        name: Optional argument name used in error messages.

    Returns:
        A new list of the requested length.

    Raises:
        ValueError: If an iterable has the wrong length, or if ``x`` is neither
            a supported scalar nor an iterable.
    """
    return list(expand_site_option(x, nsites=length, name=name or "value"))
