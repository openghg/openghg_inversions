"""Country totals that are exact sums of whole basis regions.

When every basis region with flux in a country lies inside that country, for
every source, the country total is the sum of those regions' totals. This
module checks which countries a basis preserves in that sense and reports
their totals as region sums, optionally restricted to those countries.

"Country" means any row of a membership matrix with dimensions
``(country, lat, lon)``, such as :attr:`Countries.matrix`, or the classes a
region-constrained basis was built with (see :func:`region_class_membership`).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator

from .co2_flux_outputs import co2_country_flux_outputs
from .countries import Countries

if TYPE_CHECKING:
    from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap


__all__ = [
    "EXACT_REGION_SUM",
    "RETAINED_STATE_FUNCTIONAL",
    "BorderCheck",
    "RegionLeak",
    "check_border_preserving_countries",
    "co2_region_sum_country_outputs",
    "region_class_membership",
    "region_sum_membership",
]

EXACT_REGION_SUM = "exact_region_sum"
RETAINED_STATE_FUNCTIONAL = "retained_state_functional"
# Rounding slack, so that tolerance 0 accepts regions contained up to floating-point error.
_ROUNDING = 1e-12


@dataclass(frozen=True)
class RegionLeak:
    """A basis region that has support in a country but is not contained in it.

    Attributes:
        source: Source (sector) whose labels define the region.
        region: Basis label of the region.
        leak_fraction: Share of the region's support weight outside the country.
        country_fraction: Share of the country's support weight (for this
            source) that lies in this region.
    """

    source: str
    region: int
    leak_fraction: float
    country_fraction: float


@dataclass(frozen=True)
class BorderCheck:
    """Which countries a per-source basis preserves, and why others fail.

    Attributes:
        tolerance: Tolerance used for both region leaks and the country share
            left in crossing regions.
        qualifying: Countries whose total is a sum of whole regions, in
            membership order.
        regions: For every country, the regions contained in it per source.
            Sources without support in the country map to an empty tuple.
        failures: For each non-qualifying country, the crossing regions of
            the sources that fail, largest country share first.
        exempt_sources: For every country, the sources with no support in it.
            Their labels cannot affect the country and are not checked.
    """

    tolerance: float
    qualifying: tuple[str, ...]
    regions: Mapping[str, Mapping[str, tuple[int, ...]]]
    failures: Mapping[str, tuple[RegionLeak, ...]] = field(default_factory=dict)
    exempt_sources: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def non_qualifying(self) -> tuple[str, ...]:
        """Countries whose border at least one source's regions cross."""
        return tuple(country for country in self.regions if country not in self.qualifying)

    def failing_sources(self, country: str) -> tuple[str, ...]:
        """Return the sources whose regions cross ``country``'s border, in check order."""
        return tuple(dict.fromkeys(leak.source for leak in self.failures.get(country, ())))

    def to_attrs(self) -> dict[str, str]:
        """Return NetCDF-safe output metadata describing this check.

        Returns:
            ``border_check_tolerance``, ``border_preserving_countries`` and
            ``non_border_preserving_countries`` (JSON lists), and
            ``border_check_failures``: a JSON object mapping each
            non-qualifying country to its failing regions.
        """
        failures = {
            country: [
                {
                    "source": leak.source,
                    "region": leak.region,
                    "leak_fraction": leak.leak_fraction,
                    "country_fraction": leak.country_fraction,
                }
                for leak in leaks
            ]
            for country, leaks in self.failures.items()
        }
        return {
            "border_check_tolerance": repr(self.tolerance),
            "border_preserving_countries": json.dumps(list(self.qualifying)),
            "non_border_preserving_countries": json.dumps(list(self.non_qualifying)),
            "border_check_failures": json.dumps(failures, sort_keys=True),
        }


def _grid_values(array: xr.DataArray, reference: xr.DataArray, *, name: str) -> np.ndarray:
    """Return eager 2-D values of ``array`` on the exact ``(lat, lon)`` grid of ``reference``."""
    if set(array.dims) != {"lat", "lon"}:
        raise ValueError(f"{name} must have exactly lat/lon dimensions; got {array.dims!r}.")
    array, _ = xr.align(array, reference, join="exact", copy=False)
    return np.asarray(to_dense(array.transpose("lat", "lon")).values)


def check_border_preserving_countries(
    labels: Mapping[str, xr.DataArray],
    membership: xr.DataArray,
    support: Mapping[str, xr.DataArray] | None = None,
    *,
    tolerance: float = 1e-6,
) -> BorderCheck:
    """Find the countries whose total is a sum of whole basis regions for every source.

    For source ``s`` with non-negative support weight ``w`` (for example
    ``|flux| * area``) and country fractions ``f_c``, region ``r`` overlaps
    country ``c`` by ``O = sum_{i in r} f_c(i) w(i)``. It is *contained* in
    ``c`` when its leak ``1 - O / sum_{i in r} w(i)`` is at most
    ``tolerance``. A country qualifies for ``s`` when the share of its support
    that lies in non-contained regions is at most ``tolerance``. It qualifies
    overall when it qualifies for every source.

    A source with no support in a country, such as an ocean source whose
    regions hold only sea cells, or whose flux is zero on the country's cells,
    is exempt for that country. The exemption follows from labels, support and
    membership only; source names play no part.

    Args:
        labels: Integer basis labels per source on ``(lat, lon)``. Every label
            value, including ``0``, is a region. Sources may use different
            labels.
        membership: Country fractions in ``[0, 1]`` with dimensions
            ``(country, lat, lon)``, for example :attr:`Countries.matrix`. Its
            ``lat``/``lon`` coordinates define the grid.
        support: Optional non-negative weight per source on ``(lat, lon)``.
            ``None`` gives every cell weight one, which checks geometry only and
            does not exempt sources whose regions are mixed but whose flux is
            zero on land. Prefer ``|flux| * area`` summed over time.
        tolerance: Allowed leak per region and allowed share of the country
            left in crossing regions. With an exact categorical membership,
            ``0`` certifies equality; fractional border cells need a positive
            tolerance.

    Returns:
        BorderCheck: Qualifying countries, contained regions per country and
        source, failing regions with their leak fractions, and exempt sources.

    Raises:
        ValueError: If inputs are not on one exact grid, support or membership
            is negative, labels are not integers, or the tolerance is outside
            ``[0, 1)``.

    Notes:
        The region-sum total includes the outside part of contained regions
        and omits the part of the country in non-contained regions. Both are at
        most about ``tolerance`` of the country's support weight. Inputs are
        computed eagerly, one country row at a time.
    """
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must lie in [0, 1).")
    if set(membership.dims) != {"country", "lat", "lon"}:
        raise ValueError("membership must have dimensions country, lat and lon.")
    if not labels:
        raise ValueError("labels must contain at least one source.")
    grid = membership.isel(country=0, drop=True)
    countries = [str(country) for country in membership["country"].values]

    sources: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for source, source_labels in labels.items():
        values = _grid_values(source_labels, grid, name=f"labels[{source!r}]")
        if not np.issubdtype(values.dtype, np.integer):
            if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
                raise ValueError(f"labels[{source!r}] must hold finite integer labels.")
            values = values.astype(np.int64)
        if support is None:
            weight = np.ones(values.shape)
        else:
            weight = _grid_values(support[source], grid, name=f"support[{source!r}]").astype(float)
            if np.any(~np.isfinite(weight)) or np.any(weight < 0):
                raise ValueError(f"support[{source!r}] must be finite and non-negative.")
        region_ids, codes = np.unique(values.ravel(), return_inverse=True)
        weight = weight.ravel()
        region_weight = np.bincount(codes, weights=weight, minlength=len(region_ids))
        sources[str(source)] = (region_ids, codes, weight, region_weight)

    qualifying = []
    regions: dict[str, dict[str, tuple[int, ...]]] = {}
    failures: dict[str, tuple[RegionLeak, ...]] = {}
    exempt: dict[str, tuple[str, ...]] = {}
    for index, country in enumerate(countries):
        fraction = np.asarray(to_dense(membership.isel(country=index)).transpose("lat", "lon").values, float)
        if np.any(fraction < 0) or np.any(fraction > 1 + 1e-6):
            raise ValueError(f"membership of {country!r} must lie in [0, 1].")
        fraction = fraction.ravel()
        regions[country] = {}
        leaks: list[RegionLeak] = []
        exempt_sources = []
        for source, (region_ids, codes, weight, region_weight) in sources.items():
            overlap = np.bincount(codes, weights=fraction * weight, minlength=len(region_ids))
            country_weight = overlap.sum()
            if country_weight <= 0:
                exempt_sources.append(source)
                regions[country][source] = ()
                continue
            touched = overlap > 0
            leak = np.zeros_like(overlap)
            leak[touched] = 1 - overlap[touched] / region_weight[touched]
            contained = touched & (leak <= tolerance + _ROUNDING)
            crossing = touched & ~contained
            regions[country][source] = tuple(int(region) for region in region_ids[contained])
            if overlap[crossing].sum() / country_weight > tolerance + _ROUNDING:
                order = np.argsort(-overlap[crossing], kind="stable")
                leaks.extend(
                    RegionLeak(
                        source=source,
                        region=int(region),
                        leak_fraction=float(region_leak),
                        country_fraction=float(region_overlap / country_weight),
                    )
                    for region, region_leak, region_overlap in zip(
                        region_ids[crossing][order], leak[crossing][order], overlap[crossing][order]
                    )
                )
        exempt[country] = tuple(exempt_sources)
        if leaks:
            failures[country] = tuple(leaks)
        else:
            qualifying.append(country)
    return BorderCheck(
        tolerance=float(tolerance),
        qualifying=tuple(qualifying),
        regions=regions,
        failures=failures,
        exempt_sources=exempt,
    )


def region_sum_membership(
    check: BorderCheck,
    labels: Mapping[str, xr.DataArray],
    *,
    countries: list[str] | tuple[str, ...] | None = None,
    source_dim: str = "source",
) -> xr.DataArray:
    """Return each country's whole-region membership per source.

    The row of country ``c`` and source ``s`` is one on the cells of every
    region of ``s`` contained in ``c`` and zero elsewhere. Applying it as a
    country functional gives the sum of those whole-region totals, so any
    reconstruction (bucket or explicit) gives the region-sum country total.

    Args:
        check: Result of :func:`check_border_preserving_countries` for ``labels``.
        labels: The same per-source basis labels.
        countries: Countries to include. ``None`` selects ``check.qualifying``.
        source_dim: Name of the source dimension, whose labels are the keys
            of ``labels`` in mapping order.

    Returns:
        xarray.DataArray: Float membership on ``(country, source_dim, lat, lon)``.

    Raises:
        ValueError: If a country was not checked.
    """
    countries = list(check.qualifying if countries is None else countries)
    missing = [country for country in countries if country not in check.regions]
    if missing:
        raise ValueError(f"Countries {missing!r} are not in the border check.")
    per_source = []
    for source, source_labels in labels.items():
        rows = [
            source_labels.isin(list(check.regions[country][str(source)])).astype("float64")
            for country in countries
        ]
        per_source.append(xr.concat(rows, dim=pd.Index(countries, name="country")))
    result = xr.concat(per_source, dim=pd.Index([str(source) for source in labels], name=source_dim))
    return result.transpose("country", source_dim, "lat", "lon").rename("region_sum_membership")


def region_class_membership(
    region_classes: xr.DataArray,
    *,
    unmapped_values: tuple[Hashable, ...] = (),
) -> xr.DataArray:
    """Return a membership matrix with one row per class of a region-class map.

    Use it in place of a country matrix to report totals for exactly the
    classes a region-constrained basis was built with.

    Args:
        region_classes: Class field on ``(lat, lon)``, as passed to
            :func:`~openghg_inversions.basis.algorithms.region_constrained_basis`.
            Null cells and ``unmapped_values`` belong to no class.
        unmapped_values: Additional class values without a row.

    Returns:
        xarray.DataArray: Zero-one membership on ``(country, lat, lon)``. The
        ``country`` coordinate holds ``str(class)`` in sorted order.
    """
    values = region_classes.transpose("lat", "lon").values
    unmapped = set(unmapped_values)
    text = np.empty(values.shape, dtype=object)
    for index, value in np.ndenumerate(values):
        missing = not isinstance(value, tuple) and bool(pd.isna(value))
        text[index] = "" if missing or value in unmapped else str(value)
    classes = sorted({name for name in text.ravel() if name})
    rows = np.stack([(text == name).astype("float64") for name in classes])
    return xr.DataArray(
        rows,
        dims=("country", "lat", "lon"),
        coords={"country": classes, "lat": region_classes["lat"], "lon": region_classes["lon"]},
        name="region_class_membership",
    )


def _affine_labels(bound: BoundCo2AffineFluxMap, source_dim: str | None) -> dict[str, xr.DataArray]:
    """Return per-source labels from a bucket prolongation."""
    prolongation = bound.affine_map.prolongation
    if isinstance(prolongation, MultiSourceBucketBasisOperator):
        return {str(source): bf for source, bf in prolongation.basis_flat.items()}
    if isinstance(prolongation, BucketBasisOperator):  # AffineFluxMap allows it only without a source axis.
        return {"total": prolongation.basis_flat}
    raise ValueError("Explicit affine maps need the restriction region labels passed as `labels`.")


def _flux_support(
    bound: BoundCo2AffineFluxMap, area: xr.DataArray, source_dim: str | None
) -> dict[str, xr.DataArray]:
    """Return ``|F| * area`` summed over non-native axes, per source."""
    flux = bound.affine_map.flux
    extra = [dim for dim in flux.dims if dim not in ("lat", "lon", source_dim)]
    magnitude = abs(to_dense(flux)).sum(extra).astype("float64") * area
    magnitude = magnitude.compute()
    if source_dim is None:
        return {"total": magnitude}
    return {str(source): magnitude.sel({source_dim: source}, drop=True) for source in flux[source_dim].values}


def co2_region_sum_country_outputs(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    labels: Mapping[str, xr.DataArray] | None = None,
    *,
    membership: xr.DataArray | None = None,
    support: Mapping[str, xr.DataArray] | None = None,
    tolerance: float = 1e-6,
    country_outputs: Literal["constrained_only", "all"] = "constrained_only",
) -> xr.Dataset:
    """Summarize CO2 country totals as sums of whole-region totals where the basis allows.

    Countries are first checked with :func:`check_border_preserving_countries`
    across every source. Each qualifying country's total, per source and
    summed over sources per draw, is the sum of its whole regions' totals.

    Args:
        trace (xarray.DataTree): CO2-only annotated trace, as for
            :func:`~openghg_inversions.postprocessing.co2_flux_outputs.co2_country_flux_outputs`.
        bound (openghg_inversions.rhime.co2.co2_affine_output.BoundCo2AffineFluxMap):
            Reconstruction bound to its prepared reference.
        countries: Country membership and cell areas on the map's ``lat, lon`` grid.
        labels: Region labels per native source, keyed by the map's native
            source labels (one key, any name, for a map without a source axis).
            ``None`` reads them from a bucket prolongation; an explicit map must
            pass the labels of the regions its restriction averages over.
        membership: Optional ``(country, lat, lon)`` fractions replacing
            ``countries.matrix``, for example :func:`region_class_membership`
            of the classes the basis was built with. ``countries`` then supplies
            only the area grid.
        support: Optional support weight per source. ``None`` uses
            ``|flux| * area`` summed over non-native axes, so a source with no
            flux in a country is exempt for it.
        tolerance: Border tolerance, see :func:`check_border_preserving_countries`.
        country_outputs: ``"constrained_only"`` writes only qualifying
            countries. ``"all"`` also writes the others, from their ordinary
            retained-state country functional.

    Returns:
        xarray.Dataset: The variables of ``co2_country_flux_outputs`` with a
        ``country_total_method`` coordinate on ``country`` (``"exact_region_sum"``
        or ``"retained_state_functional"``). Global attributes add the check
        from :meth:`BorderCheck.to_attrs`; with ``"constrained_only"`` the global
        ``country_total_method`` is ``"exact_region_sum"`` and
        ``non_border_preserving_countries`` lists the countries left out.

    Raises:
        ValueError: If labels are missing or do not match the native sources,
            or ``"constrained_only"`` leaves no country.

    Notes:
        Totals are evaluated in one pass with each country's per-source union
        of contained regions, which is the same linear functional as summing
        the region totals. The uncertainty scope stays
        ``retained_state_conditional``. With ``tolerance=0`` and categorical
        membership these totals equal the ordinary country functional exactly.
    """
    if country_outputs not in ("constrained_only", "all"):
        raise ValueError("country_outputs must be 'constrained_only' or 'all'.")
    native_dims = bound.affine_map.native_dims
    source_dim = native_dims[0] if len(native_dims) == 3 else None
    if labels is None:
        labels = _affine_labels(bound, source_dim)
    expected = (
        [str(source) for source in bound.affine_map.native_mean.indexes[source_dim]] if source_dim else None
    )
    if expected is not None and list(map(str, labels)) != expected:
        raise ValueError(f"labels must be keyed by the native sources {expected!r} in order.")
    if expected is None and len(labels) != 1:
        raise ValueError("A map without a source axis takes labels for exactly one source.")
    matrix = countries.matrix if membership is None else membership
    if support is None:
        support = _flux_support(bound, countries.area_grid, source_dim)
    check = check_border_preserving_countries(labels, matrix, support, tolerance=tolerance)
    if country_outputs == "constrained_only" and not check.qualifying:
        raise ValueError(
            "No country is a sum of whole basis regions for every source; failures: "
            + check.to_attrs()["border_check_failures"]
        )

    parts = []
    if check.qualifying:
        exact = region_sum_membership(check, labels, source_dim=source_dim or "source")
        if source_dim is None:
            exact = exact.isel(source=0, drop=True)
        parts.append(_with_membership(trace, bound, countries, exact, EXACT_REGION_SUM))
    others = list(check.non_qualifying)
    if country_outputs == "all" and others:
        native = matrix.sel(country=others).transpose("country", "lat", "lon")
        parts.append(_with_membership(trace, bound, countries, native, RETAINED_STATE_FUNCTIONAL))
    result = xr.concat(parts, dim="country") if len(parts) > 1 else parts[0]
    result.attrs = {**parts[0].attrs, **check.to_attrs()}
    if country_outputs == "constrained_only":
        result.attrs["country_total_method"] = EXACT_REGION_SUM
    return result


def _with_membership(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    matrix: xr.DataArray,
    method: str,
) -> xr.Dataset:
    """Run the ordinary CO2 country summaries with another membership matrix."""
    substitute = copy.copy(countries)
    substitute.matrix = matrix.astype("float64")
    result = co2_country_flux_outputs(trace, bound, substitute)
    method_coord = xr.DataArray([method] * result.sizes["country"], dims="country")
    return result.assign_coords(country_total_method=method_coord)
