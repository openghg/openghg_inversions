"""Country totals as sums of whole basis regions, labelled by what they can claim.

A country whose border no source's basis regions cross has a total that is the
sum of whole-region totals. :func:`check_border_preserving_countries` finds
those countries from labels, a membership matrix and a support weight.

Preserving borders does not by itself make a retained-state country total
complete. In a coherent reduction the native scaling is
``x = m + U* (alpha - Pi m) + x_perp``, and a country functional ``Q`` loses
``Q x_perp`` unless ``Q`` lies in the row space of the restriction ``Pi``. For
an absolute-flux-weighted regional mean that holds for a whole region only
when the signed flux keeps one sign and one spatial pattern over time.
:func:`co2_region_sum_country_outputs` therefore labels a country
``exact_region_sum`` only when its border check has no leak and its
row-space residual is negligible (or the map is a bucket reconstruction,
which has no unresolved part). Other border-preserving countries are labelled
``region_sum_retained_conditional``.

"Country" means any row of a membership matrix with dimensions
``(country, lat, lon)``, such as :attr:`Countries.matrix`, or the classes a
region-constrained basis was built with (see :func:`region_class_membership`).
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import sparse
import xarray as xr
from openghg.util import molar_mass  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import sparse_xr_dot, to_dense
from openghg_inversions.basis.affine_flux_map import _in_dimensionless_units
from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator

# Reuse the CO2 country-summary helpers; the source-varying membership used
# here is handled locally so that co2_flux_outputs stays unchanged.
from .co2_flux_outputs import _flux_unit_scale, _output_inputs, _summarize, _with_metadata
from .countries import Countries, _bounded_projection

if TYPE_CHECKING:
    from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap


__all__ = [
    "COUNTRY_TOTAL_METHODS",
    "EXACT_REGION_SUM",
    "REGION_SUM_RETAINED_CONDITIONAL",
    "RETAINED_STATE_FUNCTIONAL",
    "BorderCheck",
    "RegionLeak",
    "SourceBorder",
    "REGION_SUM_COORDS",
    "check_border_preserving_countries",
    "co2_border_check",
    "co2_region_sum_country_outputs",
    "region_class_membership",
    "region_sum_membership",
    "restriction_row_space_residual",
]

EXACT_REGION_SUM = "exact_region_sum"
REGION_SUM_RETAINED_CONDITIONAL = "region_sum_retained_conditional"
RETAINED_STATE_FUNCTIONAL = "retained_state_functional"
COUNTRY_TOTAL_METHODS = {
    EXACT_REGION_SUM: (
        "Sum of whole-region totals with no border leak, and the country functional lies in the "
        "row space of the restriction (or the reconstruction is bucket), so no unresolved native "
        "variance is omitted."
    ),
    REGION_SUM_RETAINED_CONDITIONAL: (
        "Sum of whole-region totals within the border tolerance; retained-state conditional, so "
        "unresolved native variance may be omitted and any border leak is recorded."
    ),
    RETAINED_STATE_FUNCTIONAL: (
        "Ordinary retained-state country functional of a country whose border basis regions cross; "
        "retained-state conditional."
    ),
}
# Rounding slack, so that tolerance 0 accepts regions contained up to floating-point error.
_ROUNDING = 1e-12
CountryOutputs = Literal["border_preserving_only", "all"]
# Country coordinates added by co2_region_sum_country_outputs.
REGION_SUM_COORDS = ("country_total_method", "border_leak_fraction", "restriction_residual")


@dataclass(frozen=True)
class RegionLeak:
    """A basis region with support in a country that is not contained in it.

    Attributes:
        source: Source (sector) whose labels define the region.
        region: Basis label of the region (never an inactive, negative label).
        leak_fraction: Share of the region's support weight outside the country.
        country_fraction: Share of the source's support in the country that
            lies in this region.
    """

    source: str
    region: int
    leak_fraction: float
    country_fraction: float


@dataclass(frozen=True)
class SourceBorder:
    """How one source's regions meet one country's border.

    Fractions are relative to the source's support weight in the country.

    Attributes:
        status: ``"preserved"`` (within tolerance), ``"exempt"`` (no support
            in the country), ``"negligible"`` (fails, but the source's share
            is within the opt-in threshold) or ``"fails"``.
        share: The source's share of the country's support summed over
            sources. Only comparable when supports share units, such as
            ``|flux| * area``.
        excess_fraction: Support of contained regions lying outside the
            country, which the region sum includes.
        missed_fraction: Country support in crossing regions, which the
            region sum omits.
        inactive_fraction: Country support on inactive cells (negative
            labels), which no region carries.
    """

    status: Literal["preserved", "exempt", "negligible", "fails"]
    share: float = 0.0
    excess_fraction: float = 0.0
    missed_fraction: float = 0.0
    inactive_fraction: float = 0.0

    @property
    def leak_fraction(self) -> float:
        """Return the total border leak of this source relative to its own country support."""
        return self.excess_fraction + self.missed_fraction + self.inactive_fraction


@dataclass(frozen=True)
class BorderCheck:
    """Which countries a per-source basis preserves, and why others fail.

    Attributes:
        tolerance: Allowed region leak and allowed country share in crossing
            regions and inactive cells, per source.
        negligible_source_share: Opt-in threshold below which a failing
            source does not disqualify a country. ``0`` disables it.
        qualifying: Countries whose total is a sum of whole regions, in
            membership order.
        regions: For every country, the contained regions per source.
        sources: For every country, the :class:`SourceBorder` of each source.
        failures: For each country with a failing or negligible source, the
            crossing regions of those sources, largest country fraction first
            per source. A failure caused only by inactive cells has no regions
            here; see ``sources``.
    """

    tolerance: float
    negligible_source_share: float
    qualifying: tuple[str, ...]
    regions: Mapping[str, Mapping[str, tuple[int, ...]]]
    sources: Mapping[str, Mapping[str, SourceBorder]]
    failures: Mapping[str, tuple[RegionLeak, ...]] = field(default_factory=dict)

    @property
    def non_qualifying(self) -> tuple[str, ...]:
        """Countries whose border at least one non-negligible source's regions cross."""
        return tuple(country for country in self.regions if country not in self.qualifying)

    @property
    def exempt_sources(self) -> dict[str, tuple[str, ...]]:
        """Sources with no support in each country."""
        return {
            country: tuple(source for source, border in borders.items() if border.status == "exempt")
            for country, borders in self.sources.items()
        }

    def failing_sources(self, country: str) -> tuple[str, ...]:
        """Return the sources that disqualify ``country``, in check order."""
        return tuple(source for source, border in self.sources[country].items() if border.status == "fails")

    def negligible_sources(self, country: str) -> tuple[str, ...]:
        """Return the failing sources ignored under ``negligible_source_share``."""
        return tuple(
            source for source, border in self.sources[country].items() if border.status == "negligible"
        )

    def leak_fraction(self, country: str) -> float:
        """Return the country's border leak, as a share of its support summed over sources."""
        return float(sum(border.share * border.leak_fraction for border in self.sources[country].values()))

    def to_attrs(self) -> dict[str, str]:
        """Return NetCDF-safe output metadata describing this check.

        Returns:
            Tolerances, qualifying and non-qualifying countries (JSON lists),
            ``border_leak_fractions`` (JSON object per qualifying country),
            and ``border_check_details``: for each country that fails, leaks
            or has a negligible source, the record of each non-exempt source
            and its crossing regions.
        """
        details = {}
        for country, borders in self.sources.items():
            if country in self.qualifying and self.leak_fraction(country) <= _ROUNDING:
                continue
            regions_by_source: dict[str, list[dict[str, float | int]]] = {}
            for leak in self.failures.get(country, ()):
                regions_by_source.setdefault(leak.source, []).append(
                    {
                        "region": leak.region,
                        "leak_fraction": leak.leak_fraction,
                        "country_fraction": leak.country_fraction,
                    }
                )
            details[country] = {
                source: {
                    "status": border.status,
                    "share": border.share,
                    "excess_fraction": border.excess_fraction,
                    "missed_fraction": border.missed_fraction,
                    "inactive_fraction": border.inactive_fraction,
                    "crossing_regions": regions_by_source.get(source, []),
                }
                for source, border in borders.items()
                if border.status != "exempt"
            }
        return {
            "border_check_tolerance": repr(self.tolerance),
            "border_check_negligible_source_share": repr(self.negligible_source_share),
            "border_preserving_countries": json.dumps(list(self.qualifying)),
            "non_border_preserving_countries": json.dumps(list(self.non_qualifying)),
            "border_leak_fractions": json.dumps(
                {country: self.leak_fraction(country) for country in self.qualifying}
            ),
            "border_check_details": json.dumps(details, sort_keys=True),
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
    tolerance: float = 0.0,
    negligible_source_share: float = 0.0,
) -> BorderCheck:
    """Find the countries whose total is a sum of whole basis regions for every source.

    For source ``s`` with non-negative support weight ``w`` (for example
    ``|flux| * area``) and country fractions ``f_c``, region ``r`` overlaps
    country ``c`` by ``O = sum_{i in r} f_c(i) w(i)``. It is *contained* in
    ``c`` when its leak ``1 - O / sum_{i in r} w(i)`` is at most
    ``tolerance``. A source preserves ``c`` when the share of its support in
    ``c`` lying in non-contained regions or on inactive cells is at most
    ``tolerance``. A country qualifies when every source preserves it, is
    exempt, or is negligible.

    A source with no support in a country, such as an ocean source whose
    regions hold only sea cells, or whose flux is zero on the country's cells,
    is exempt for that country. The exemption follows from labels, support and
    membership only; source names play no part.

    Args:
        labels: Integer basis labels per source on ``(lat, lon)``. Zero and
            positive labels are regions; negative labels mark inactive cells,
            as in :func:`~openghg_inversions.basis.covariance_products.absolute_weighted_regional_mean_restriction`.
            Sources may use different labels.
        membership: Country fractions in ``[0, 1]`` with dimensions
            ``(country, lat, lon)``, for example :attr:`Countries.matrix`. Its
            ``lat``/``lon`` coordinates define the grid.
        support: Optional non-negative weight per source on ``(lat, lon)``.
            ``None`` gives every cell weight one, which checks geometry only and
            does not exempt sources whose regions are mixed but whose flux is
            zero on land. Prefer ``|flux| * area`` summed over time.
        tolerance: Allowed leak per region and allowed share of the country
            in crossing regions and inactive cells. The default ``0`` is
            strict; fractional border cells need a positive tolerance.
        negligible_source_share: Opt-in. A source that fails for a country
            but carries at most this share of the country's support summed
            over sources is reported as ``"negligible"`` and does not
            disqualify it. Its leak still counts in
            :meth:`BorderCheck.leak_fraction`.

    Returns:
        BorderCheck: Qualifying countries, contained regions, per-source
        leaks and statuses, and the crossing regions of failing sources.

    Raises:
        ValueError: If inputs are not on one exact grid, support or membership
            is negative, labels are not integers, or a tolerance is outside
            ``[0, 1)``.

    Notes:
        Inputs are computed eagerly, one country row at a time.
    """
    if not 0 <= tolerance < 1 or not 0 <= negligible_source_share < 1:
        raise ValueError("tolerance and negligible_source_share must lie in [0, 1).")
    if set(membership.dims) != {"country", "lat", "lon"}:
        raise ValueError("membership must have dimensions country, lat and lon.")
    if not labels:
        raise ValueError("labels must contain at least one source.")
    grid = membership.isel(country=0, drop=True)
    countries = [str(country) for country in membership["country"].values]

    prepared = {}
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
        values, weight = values.ravel(), weight.ravel()
        active = values >= 0
        region_ids, codes = np.unique(values[active], return_inverse=True)
        region_weight = np.bincount(codes, weights=weight[active], minlength=len(region_ids))
        prepared[str(source)] = (active, region_ids, codes, weight, region_weight)

    qualifying = []
    regions: dict[str, dict[str, tuple[int, ...]]] = {}
    sources: dict[str, dict[str, SourceBorder]] = {}
    failures: dict[str, tuple[RegionLeak, ...]] = {}
    for index, country in enumerate(countries):
        fraction = np.asarray(to_dense(membership.isel(country=index)).transpose("lat", "lon").values, float)
        if np.any(fraction < 0) or np.any(fraction > 1 + 1e-6):
            raise ValueError(f"membership of {country!r} must lie in [0, 1].")
        fraction = fraction.ravel()
        regions[country] = {}
        records: dict[str, tuple[float, SourceBorder, list[RegionLeak]]] = {}
        for source, (active, region_ids, codes, weight, region_weight) in prepared.items():
            weighted = fraction * weight
            country_weight = float(weighted.sum())
            if country_weight <= 0:
                regions[country][source] = ()
                records[source] = (0.0, SourceBorder("exempt"), [])
                continue
            overlap = np.bincount(codes, weights=weighted[active], minlength=len(region_ids))
            touched = overlap > 0
            leak = np.zeros_like(overlap)
            leak[touched] = 1 - overlap[touched] / region_weight[touched]
            contained = touched & (leak <= tolerance + _ROUNDING)
            crossing = touched & ~contained
            regions[country][source] = tuple(int(region) for region in region_ids[contained])
            missed = float(overlap[crossing].sum()) / country_weight
            inactive = float(weighted[~active].sum()) / country_weight
            excess = float((region_weight[contained] - overlap[contained]).sum()) / country_weight
            fails = missed + inactive > tolerance + _ROUNDING
            order = np.argsort(-overlap[crossing], kind="stable")
            leaks = [
                RegionLeak(source, int(region), float(region_leak), float(region_overlap / country_weight))
                for region, region_leak, region_overlap in zip(
                    region_ids[crossing][order], leak[crossing][order], overlap[crossing][order]
                )
            ]
            border = SourceBorder("fails" if fails else "preserved", 0.0, excess, missed, inactive)
            records[source] = (country_weight, border, leaks)

        total = sum(country_weight for country_weight, _, _ in records.values())
        sources[country] = {}
        country_failures: list[RegionLeak] = []
        for source, (country_weight, border, leaks) in records.items():
            share = country_weight / total if total > 0 else 0.0
            status = border.status
            if status == "fails" and share <= negligible_source_share:
                status = "negligible"
            sources[country][source] = SourceBorder(
                status, share, border.excess_fraction, border.missed_fraction, border.inactive_fraction
            )
            if status in ("fails", "negligible"):
                country_failures.extend(leaks)
        if country_failures:
            failures[country] = tuple(country_failures)
        if not any(border.status == "fails" for border in sources[country].values()):
            qualifying.append(country)
    return BorderCheck(
        tolerance=float(tolerance),
        negligible_source_share=float(negligible_source_share),
        qualifying=tuple(qualifying),
        regions=regions,
        sources=sources,
        failures=failures,
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
    region of ``s`` contained in ``c`` and zero elsewhere, including inactive
    cells. As a country functional it gives the sum of those whole-region
    totals for any reconstruction, bucket or explicit.

    Args:
        check: Result of :func:`check_border_preserving_countries` for ``labels``.
        labels: The same per-source basis labels.
        countries: Countries to include. ``None`` selects ``check.qualifying``.
        source_dim: Name of the source dimension, whose labels are the keys
            of ``labels`` in mapping order.

    Returns:
        xarray.DataArray: Float membership on ``(country, source_dim, lat, lon)``.

    Raises:
        ValueError: If a country was not checked or no country is selected.
    """
    countries = list(check.qualifying if countries is None else countries)
    if not countries:
        raise ValueError("region_sum_membership needs at least one country.")
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


def _restriction_matrix(restriction: xr.DataArray, state_dim: str) -> scipy.sparse.csr_matrix:
    """Return ``Pi`` as a sparse ``(state, native)`` matrix, materializing it once."""
    data = restriction.data
    if hasattr(data, "compute"):
        data = data.compute()
    shape = (restriction.sizes[state_dim], -1)
    if isinstance(data, sparse.SparseArray):
        return sparse.COO(data).reshape(shape).to_scipy_sparse().tocsr()
    return scipy.sparse.csr_matrix(np.asarray(data, dtype=float).reshape(shape))


def restriction_row_space_residual(
    restriction: xr.DataArray,
    functional: xr.DataArray,
    *,
    state_dim: str,
) -> xr.DataArray:
    """Return how far functionals lie outside the row space of a restriction.

    For a functional ``Q`` on the native axes, the relative residual is
    ``||Q - P Q|| / ||Q||``, where ``P`` projects onto the row space of
    ``Pi``. It is zero exactly when ``Q`` is a linear combination of
    restriction rows, so that ``Q x_perp = 0`` for every unresolved native
    component and the retained-state total of ``Q`` is complete.

    Args:
        restriction: Dimensionless ``Pi`` on ``(state_dim, *native_dims)``,
            NumPy, ``sparse`` or Dask-backed. It must have full row rank.
        functional: Functionals on the native axes; every other dimension
            (country, time, ...) indexes separate functionals.
        state_dim: Retained-state dimension of ``restriction``.

    Returns:
        xarray.DataArray: Relative residuals over the non-native dimensions of
        ``functional``; ``NaN`` where ``Q`` is zero.

    Raises:
        ValueError: If native coordinates differ or ``Pi`` is rank deficient.
    """
    native_dims = tuple(str(dim) for dim in restriction.dims if dim != state_dim)
    restriction = restriction.transpose(state_dim, *native_dims)
    functional, _ = xr.align(functional, restriction, join="exact", copy=False, exclude=[state_dim])
    outer = [dim for dim in functional.dims if dim not in native_dims]
    native_size = int(np.prod([restriction.sizes[dim] for dim in native_dims]))
    values = np.asarray(to_dense(functional.transpose(*outer, *native_dims)).values, dtype=float)
    values = values.reshape(-1, native_size)
    pi = _restriction_matrix(restriction, state_dim)
    gram = (pi @ pi.T).toarray()
    try:
        factor = scipy.linalg.cho_factor(gram)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The restriction must have full row rank.") from exc
    coefficients = scipy.linalg.cho_solve(factor, np.asarray(pi @ values.T))
    projected = np.asarray((pi.T @ coefficients).T)
    norms = np.linalg.norm(values, axis=1)
    residual = np.linalg.norm(values - projected, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(norms > 0, residual / norms, np.nan)
    shape = [functional.sizes[dim] for dim in outer]
    coords = {dim: functional[dim] for dim in outer if dim in functional.coords}
    return xr.DataArray(relative.reshape(shape), dims=outer, coords=coords, name="restriction_residual")


def _affine_labels(bound: BoundCo2AffineFluxMap) -> dict[str, xr.DataArray]:
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
    magnitude = (abs(to_dense(flux)).sum(extra).astype("float64") * area).compute()
    if source_dim is None:
        return {"total": magnitude}
    return {str(source): magnitude.sel({source_dim: source}, drop=True) for source in flux[source_dim].values}


def _country_summaries(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    matrix: xr.DataArray,
    area: xr.DataArray,
) -> xr.Dataset:
    """Summarize CO2 country draws for a membership that may vary by native source.

    This follows :func:`~openghg_inversions.postprocessing.co2_flux_outputs.co2_country_flux_outputs`:
    country reference ``A F m`` and response ``A F U*`` are formed before
    applying sample axes.
    """
    samples, source_dim, attrs = _output_inputs(trace, bound)
    affine = bound.affine_map
    flux, matrix, area = xr.align(affine.flux, matrix, area, join="exact", copy=False)
    weighted_flux = (
        flux.astype("float64")
        * matrix
        * area
        * _flux_unit_scale(bound)
        * (365 * 24 * 3600)
        * molar_mass("co2")
    )
    native_mean = _in_dimensionless_units(affine.native_mean, name="native_mean")
    reference = to_dense(sparse_xr_dot(weighted_flux, native_mean, dim=["lat", "lon"]))
    outputs = []
    response = None
    for group, state in samples.items():
        centred, prolongation = affine._centred_state(state, bound.reference_state)
        if response is None:
            response = to_dense(sparse_xr_dot(weighted_flux, prolongation, dim=["lat", "lon"]))
            response = _bounded_projection(response, [affine.state_dim])
            reference = _bounded_projection(reference, [])
        draws = reference + xr.dot(response, centred.astype("float64"), dim=affine.state_dim)
        outputs.append(_summarize(draws, prefix="country", group=group, source_dim=source_dim))
    attrs["annualization"] = "365-day year; grams of CO2, not grams of carbon"
    return _with_metadata(xr.merge(outputs), attrs, units="g yr-1")


def _region_sum_summaries(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    matrix: xr.DataArray,
    labels: Mapping[str, xr.DataArray],
    check: BorderCheck,
    country_outputs: CountryOutputs,
) -> xr.Dataset:
    """Summarize qualifying countries as region sums and, for ``"all"``, the others as usual."""
    native_dims = bound.affine_map.native_dims
    source_dim = native_dims[0] if len(native_dims) == 3 else None
    parts = []
    if check.qualifying:
        exact = region_sum_membership(check, labels, source_dim=source_dim or "source")
        if source_dim is None:
            exact = exact.isel(source=0, drop=True)
        parts.append(_country_summaries(trace, bound, exact, countries.area_grid))
    others = list(check.non_qualifying)
    if country_outputs == "all" and others:
        native = matrix.sel(country=others).transpose("country", "lat", "lon").astype("float64")
        parts.append(_country_summaries(trace, bound, native, countries.area_grid))
    return xr.concat(parts, dim="country", combine_attrs="override") if len(parts) > 1 else parts[0]


def co2_border_check(
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    labels: Mapping[str, xr.DataArray] | None = None,
    *,
    membership: xr.DataArray | None = None,
    support: Mapping[str, xr.DataArray] | None = None,
    tolerance: float = 0.0,
    negligible_source_share: float = 0.0,
) -> tuple[dict[str, xr.DataArray], BorderCheck]:
    """Check which countries a CO2 affine map's per-source regions preserve.

    Args:
        bound (openghg_inversions.rhime.co2.co2_affine_output.BoundCo2AffineFluxMap):
            Reconstruction bound to its prepared reference.
        countries: Country membership and cell areas on the map's grid.
        labels: Region labels per native source, as for
            :func:`co2_region_sum_country_outputs`; ``None`` reads a bucket
            prolongation.
        membership: Optional membership replacing ``countries.matrix``.
        support: Optional support per source; ``None`` uses ``|flux| * area``.
        tolerance: See :func:`check_border_preserving_countries`.
        negligible_source_share: See :func:`check_border_preserving_countries`.

    Returns:
        The resolved labels, keyed by native source, and the border check.

    Raises:
        ValueError: If labels are missing or not keyed by the native sources.
    """
    native_dims = bound.affine_map.native_dims
    source_dim = native_dims[0] if len(native_dims) == 3 else None
    resolved = dict(_affine_labels(bound) if labels is None else labels)
    if source_dim is not None:
        expected = [str(source) for source in bound.affine_map.native_mean.indexes[source_dim]]
        if list(map(str, resolved)) != expected:
            raise ValueError(f"labels must be keyed by the native sources {expected!r} in order.")
    elif len(resolved) != 1:
        raise ValueError("A map without a source axis takes labels for exactly one source.")
    if support is None:
        support = _flux_support(bound, countries.area_grid, source_dim)
    matrix = countries.matrix if membership is None else membership
    check = check_border_preserving_countries(
        resolved, matrix, support, tolerance=tolerance, negligible_source_share=negligible_source_share
    )
    return resolved, check


def co2_region_sum_country_outputs(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    labels: Mapping[str, xr.DataArray] | None = None,
    *,
    restriction: xr.DataArray | None = None,
    membership: xr.DataArray | None = None,
    support: Mapping[str, xr.DataArray] | None = None,
    tolerance: float = 0.0,
    negligible_source_share: float = 0.0,
    residual_tolerance: float = 1e-8,
    country_outputs: CountryOutputs = "border_preserving_only",
    check: BorderCheck | None = None,
) -> xr.Dataset:
    """Summarize CO2 country totals as sums of whole-region totals where the basis allows.

    Countries are checked with :func:`check_border_preserving_countries`
    across every source. Each qualifying country's total, per source and
    summed over sources per draw, is the sum of its whole regions' totals.
    Each country is labelled with one of :data:`COUNTRY_TOTAL_METHODS`:

    * ``exact_region_sum``: no border leak, and the region-sum functional of
      every source and time lies in the row space of the restriction
      (relative residual at most ``residual_tolerance``), or the map is a
      bucket reconstruction;
    * ``region_sum_retained_conditional``: qualifies otherwise, for example
      with a mixed-sign or time-varying flux under an
      absolute-flux-weighted restriction, without ``restriction``, or with a
      leak within ``tolerance``;
    * ``retained_state_functional``: a non-qualifying country, written only
      with ``country_outputs="all"``.

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
            Negative labels are inactive cells.
        restriction: The restriction ``Pi`` of an explicit map, on
            ``(state, *native_dims)``. Without it an explicit map's countries
            are never labelled exact.
        membership: Optional ``(country, lat, lon)`` fractions replacing
            ``countries.matrix``, for example :func:`region_class_membership`
            of the classes the basis was built with. ``countries`` then supplies
            only the area grid.
        support: Optional support weight per source. ``None`` uses
            ``|flux| * area`` summed over non-native axes.
        tolerance: Border tolerance, see :func:`check_border_preserving_countries`.
        negligible_source_share: Opt-in negligible-source threshold, see
            :func:`check_border_preserving_countries`.
        residual_tolerance: Largest relative row-space residual labelled exact.
        country_outputs: ``"border_preserving_only"`` writes only qualifying
            countries. ``"all"`` also writes the others from their ordinary
            retained-state country functional.
        check: Optional precomputed :func:`co2_border_check` for these labels
            and membership, for example to report sector subsets with the full
            map's countries. ``support`` and the tolerances are then unused.

    Returns:
        xarray.Dataset: The variables of ``co2_country_flux_outputs``, with
        country coordinates ``country_total_method``, ``border_leak_fraction``
        and ``restriction_residual`` (largest over sources and times; ``0`` for
        bucket maps, ``NaN`` when not computed). Global attributes add the check
        (:meth:`BorderCheck.to_attrs`), ``country_total_method_definitions``,
        ``country_outputs`` and per-source residuals. The uncertainty scope
        stays ``retained_state_conditional``; only ``exact_region_sum`` rows
        omit no unresolved native variance.

    Raises:
        ValueError: If labels are missing or do not match the native sources,
            or ``"border_preserving_only"`` leaves no country.
    """
    if country_outputs not in ("border_preserving_only", "all"):
        raise ValueError("country_outputs must be 'border_preserving_only' or 'all'.")
    source_dim = bound.affine_map.native_dims[0] if len(bound.affine_map.native_dims) == 3 else None
    matrix = countries.matrix if membership is None else membership
    if check is None:
        labels, check = co2_border_check(
            bound,
            countries,
            labels,
            membership=membership,
            support=support,
            tolerance=tolerance,
            negligible_source_share=negligible_source_share,
        )
    elif labels is None:
        labels = _affine_labels(bound)
    if country_outputs == "border_preserving_only" and not check.qualifying:
        raise ValueError(
            "No country is a sum of whole basis regions for every source; details: "
            + check.to_attrs()["border_check_details"]
        )

    residuals = _residuals(bound, labels, check, countries.area_grid, restriction, source_dim)
    result = _region_sum_summaries(trace, bound, countries, matrix, labels, check, country_outputs)
    methods, leaks, worst = [], [], []
    for country in (str(country) for country in result["country"].values):
        if country not in check.qualifying:
            methods.append(RETAINED_STATE_FUNCTIONAL)
            leaks.append(np.nan)
            worst.append(np.nan)
            continue
        leak = check.leak_fraction(country)
        residual = max(residuals[country].values(), default=0.0) if country in residuals else np.nan
        exact = leak <= _ROUNDING and bool(residual <= residual_tolerance)
        methods.append(EXACT_REGION_SUM if exact else REGION_SUM_RETAINED_CONDITIONAL)
        leaks.append(leak)
        worst.append(residual)
    result = result.assign_coords(
        dict(zip(REGION_SUM_COORDS, (("country", values) for values in (methods, leaks, worst))))
    )
    result.attrs.update(check.to_attrs())
    result.attrs["country_outputs"] = country_outputs
    result.attrs["country_total_method_definitions"] = json.dumps(COUNTRY_TOTAL_METHODS, sort_keys=True)
    result.attrs["restriction_residual_tolerance"] = repr(residual_tolerance)
    result.attrs["restriction_residuals"] = json.dumps(residuals, sort_keys=True)
    if isinstance(bound.affine_map.prolongation, xr.DataArray):
        basis = (
            "supplied restriction" if restriction is not None else "not computed (no restriction supplied)"
        )
    else:
        basis = "bucket reconstruction (no unresolved component)"
    result.attrs["restriction_residual_basis"] = basis
    return result


def _residuals(
    bound: BoundCo2AffineFluxMap,
    labels: Mapping[str, xr.DataArray],
    check: BorderCheck,
    area: xr.DataArray,
    restriction: xr.DataArray | None,
    source_dim: str | None,
) -> dict[str, dict[str, float]]:
    """Return, per qualifying country and source, the largest row-space residual over time.

    Bucket maps have no unresolved component, so their residual is zero. An
    explicit map without a restriction has no entry. A source without support
    in a country has residual zero.
    """
    if not check.qualifying:
        return {}
    if not isinstance(bound.affine_map.prolongation, xr.DataArray):
        return {country: {str(source): 0.0 for source in labels} for country in check.qualifying}
    if restriction is None:
        return {}
    state_dim = bound.affine_map.state_dim
    union = region_sum_membership(check, labels, source_dim=source_dim or "source")
    if source_dim is None:
        union = union.isel(source=0, drop=True)
    functional = to_dense(bound.affine_map.flux).astype("float64") * union * area
    if source_dim is not None:
        # One functional per reported source: zero every other native source.
        selector = xr.DataArray(
            np.eye(len(labels)),
            dims=("reported_source", source_dim),
            coords={"reported_source": list(map(str, labels)), source_dim: union[source_dim].values},
        )
        functional = functional * selector
    residual = restriction_row_space_residual(restriction, functional, state_dim=state_dim)
    if source_dim is None:
        residual = residual.expand_dims(reported_source=[str(next(iter(labels)))])
    worst = residual.max(
        [dim for dim in residual.dims if dim not in ("country", "reported_source")], skipna=True
    )
    worst = worst.fillna(0.0)
    return {
        str(country): {
            str(source): float(worst.sel(country=country, reported_source=source))
            for source in worst["reported_source"].values
        }
        for country in check.qualifying
    }
