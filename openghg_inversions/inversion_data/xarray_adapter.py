"""Adapt source-neutral xarray data to the canonical RHIME input contract.

The public :func:`prepare_rhime_inputs_from_xarray` adapter starts after data
acquisition. It accepts site data already held in xarray objects, applies a
retained :class:`~openghg_inversions.basis.basis_functions.BasisFunctions`
object when only footprint-times-flux caches are available, and returns the
same :class:`~openghg_inversions.inversion_data.preparation.RhimePreparedInputs`
contract used by the OpenGHG-backed preparation path.

Each site-local input dataset must contain a ``time`` dimension and the
observation variables required by
:func:`openghg_inversions.inversion_inputs.make_inv_inputs`:
``mf``, ``mf_error``, ``mf_repeatability``, and ``mf_variability``. They must
also contain either canonical ``H`` or one of ``fp_x_flux`` and
``fp_x_flux_sectoral``. When both cache variables are present, the explicitly
sector-resolved ``fp_x_flux_sectoral`` is used. Optional ``H_bc`` retains the
existing sampled boundary-condition contribution. Optional
``fixed_baseline(time)`` is retained as a fixed, observation-aligned model
contribution in the same units as ``mf``; it may be supplied alone or together
with ``H_bc``.

Site data must be supplied as an ordered site-to-Dataset mapping or a root
DataTree with one direct child node per site, each holding a site-local Dataset
and no nested children. Direct Dataset, dense
``Dataset(site, time)``, and pre-stacked ``nmeasure`` layouts are deliberately
outside this adapter's contract.

Canonical ``H`` uses dimensions ``(region, time)`` for single-source and
source-specific gathered state, or ``(region, time, source)`` for a shared
rectangular state. A gathered source-specific ``region`` coordinate is a unique
MultiIndex over ``(source, region_in_source)`` and must exactly match the
retained operator state. Multisector inputs additionally require
source-resolved retained prior flux with exactly the same source names and
order. The adapter rejects broadcasting one total prior flux across multiple
sectors because that would corrupt flux reconstruction.

The adapter is pure: it never mutates the supplied xarray objects or retained
basis functions. Known footprint-times-flux caches and non-time-dependent data
variables are excluded from the returned canonical inputs; other
observation-aligned extension variables are retained.

Every supplied row is active. Each site therefore needs an explicit, nonempty,
unique ``datetime64`` time coordinate without ``NaT``. Observation, projected
sensitivity, and optional baseline values must be finite. Required observation
and error fields, selected ``H`` or cache, and optional ``H_bc`` and
``fixed_baseline`` must declare the same exact nonempty unit string; the
adapter performs no conversion. Labels on explicit ``source`` dimensions must
be nonempty, unique Python or NumPy strings in their intended order, and
non-string labels are not coerced. Repeated source values inside a gathered
``(source, region_in_source)`` state MultiIndex are valid.

Use :meth:`RhimePreparedInputs.save
<openghg_inversions.inversion_data.preparation.RhimePreparedInputs.save>` and
:meth:`~openghg_inversions.inversion_data.preparation.RhimePreparedInputs.load`
for durable canonical artifacts. Serialized or otherwise pre-stacked
``nmeasure`` data should not be passed back through this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data.preparation import RhimePreparedInputs
from openghg_inversions.inversion_inputs import DatetimeLike, make_inv_inputs

_CACHE_VARIABLES = ("fp_x_flux", "fp_x_flux_sectoral")
_REQUIRED_OBSERVATION_VARIABLES = ("mf", "mf_error", "mf_repeatability", "mf_variability")


def _ordered_unique_strings(values: np.ndarray) -> list[str]:
    """Return string labels in first-occurrence order.

    Args:
        values: String labels that may repeat.

    Returns:
        Unique labels ordered by their first input position.
    """
    unique, first_indices = np.unique(values, return_index=True)
    return [str(value) for value in unique[np.argsort(first_indices)]]


def _validated_source_labels(
    coordinate: xr.DataArray,
    *,
    context: str,
    require_unique: bool,
) -> list[str]:
    """Return exact source labels after validating the narrow string contract.

    Args:
        coordinate: Explicit one-dimensional source coordinate.
        context: Human-readable input location for validation errors.
        require_unique: Whether repeated labels are invalid. Operator state
            metadata may repeat a source once per region, while source
            dimensions must be unique.

    Returns:
        Source labels without coercing non-string labels or sorting.

    Raises:
        ValueError: If a label is not a nonempty Python/NumPy string, or if
            uniqueness is required and labels are repeated.
    """
    values = np.asarray(coordinate.values)
    labels: list[str] = []
    for value in values:
        if not isinstance(value, (str, np.str_)) or not value:
            raise ValueError(
                f"{context} source labels must be nonempty Python/NumPy strings; found {value!r}."
            )
        labels.append(str(value))
    if require_unique and len(set(labels)) != len(labels):
        raise ValueError(f"{context} source labels must be unique; found {labels!r}.")
    return labels


def _require_dim_coordinate(
    data: xr.DataArray | xr.Dataset,
    dim: str,
    *,
    context: str,
) -> xr.DataArray:
    """Return an explicit one-dimensional coordinate for a required dimension.

    Args:
        data: Array or Dataset carrying the dimension.
        dim: Required dimension and coordinate name.
        context: Human-readable input location for validation errors.

    Returns:
        The explicit one-dimensional dimension coordinate.

    Raises:
        ValueError: If the dimension has no matching one-dimensional coordinate.
    """
    if dim not in data.coords or data.coords[dim].dims != (dim,):
        raise ValueError(f"{context} must provide an explicit ordered {dim!r} dimension coordinate.")
    return data.coords[dim]


def _require_concentration_units(data: xr.DataArray, *, context: str) -> str:
    """Return a nonempty concentration unit string without normalization.

    Args:
        data: Concentration-valued array to inspect.
        context: Human-readable input location for validation errors.

    Returns:
        The exact supplied unit string.

    Raises:
        ValueError: If ``units`` is missing, empty, or not a string.
    """
    units = data.attrs.get("units")
    if not isinstance(units, (str, np.str_)) or not units:
        raise ValueError(f"{context} must provide a nonempty 'units' string.")
    return str(units)


def _require_finite(data: xr.DataArray, *, context: str) -> None:
    """Require all values in a concentration-valued array to be finite.

    Args:
        data: Numeric array to validate.
        context: Human-readable input location for validation errors.

    Raises:
        ValueError: If the values are non-numeric or contain non-finite entries.
    """
    try:
        finite = np.isfinite(np.asarray(data.values))
    except TypeError as error:
        raise ValueError(f"{context} must contain numeric finite values.") from error
    if not finite.all():
        raise ValueError(f"{context} must contain only finite values.")


def _validate_time_coordinate(dataset: xr.Dataset, *, site: str) -> None:
    """Validate an explicit, nonempty, unique datetime64 site time coordinate.

    Args:
        dataset: Site-local dataset to validate.
        site: Site name used in validation errors.

    Raises:
        ValueError: If ``time`` is missing, empty, duplicated, non-datetime, or
            contains ``NaT``.
    """
    time = _require_dim_coordinate(dataset, "time", context=f"Site {site!r} dataset")
    values = np.asarray(time.values)
    if values.size == 0:
        raise ValueError(f"Site {site!r} time coordinate must be nonempty.")
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError(f"Site {site!r} time coordinate must have datetime64 dtype.")
    if np.isnat(values).any():
        raise ValueError(f"Site {site!r} time coordinate must not contain NaT.")
    if np.unique(values).size != values.size:
        raise ValueError(f"Site {site!r} time coordinate values must be unique.")


def _operator_source_order(basis_functions: BasisFunctions) -> list[str] | None:
    """Return ordered source semantics declared by the retained operator.

    Args:
        basis_functions: Retained basis whose operator metadata is inspected.

    Returns:
        Source labels in first-occurrence order, or ``None`` for an operator
        without source-specific state.

    Raises:
        ValueError: If operator source metadata contains invalid labels.
    """
    basis_matrix = basis_functions.operator.basis_matrix
    if "source" not in basis_matrix.coords:
        return None
    source_coord = basis_matrix.coords["source"]
    labels = _validated_source_labels(
        source_coord,
        context="Retained basis operator",
        require_unique=False,
    )
    return _ordered_unique_strings(np.asarray(labels))


def _validate_state_and_source_compatibility(
    sensitivity: xr.DataArray,
    basis_functions: BasisFunctions,
    *,
    site: str,
) -> None:
    """Validate sensitivity state/source semantics against retained basis data.

    Region labels must exactly match the retained operator state. Single-source
    and shared-basis multisource sensitivities use a plain region coordinate;
    source-specific sensitivities gather their ragged state into a unique
    MultiIndex with levels ``(source, region_in_source)``. A single-source
    sensitivity cannot be paired with source-resolved flux or operator
    metadata. Every multisource sensitivity requires source-resolved retained
    flux with labels in the operator/sensitivity order.

    Args:
        sensitivity: Canonical ``H(region, time[, source])`` to validate.
        basis_functions: Retained operator and prior flux used for projection
            and posterior reconstruction.
        site: Site name used in validation errors.

    Raises:
        ValueError: If region labels, source layout/order, or retained flux
            are incompatible.
    """
    state_coord = _require_dim_coordinate(sensitivity, "region", context=f"Site {site!r} H")
    operator = basis_functions.operator
    basis_matrix = operator.basis_matrix
    state_dim = operator.meta.state_dim
    operator_sources = _operator_source_order(basis_functions)
    state_index = sensitivity.indexes.get("region")
    flux = basis_functions.flux
    flux_sources: list[str] | None = None
    if "source" in flux.dims:
        flux_source_coord = _require_dim_coordinate(flux, "source", context="Retained prior flux")
        flux_sources = _validated_source_labels(
            flux_source_coord,
            context="Retained prior flux",
            require_unique=True,
        )

    if operator_sources is not None:
        if "source" in sensitivity.dims:
            raise ValueError(
                f"Site {site!r} H uses a rectangular source dimension, but the retained "
                "source-specific basis operator requires a gathered region MultiIndex."
            )
        if flux_sources is None:
            raise ValueError(
                f"Site {site!r} H uses a source-specific basis operator, but retained prior "
                "flux has no source dimension."
            )
        if flux_sources != operator_sources:
            raise ValueError(
                f"Retained prior flux sources/order {flux_sources!r} do not match retained "
                f"basis operator sources/order {operator_sources!r}."
            )
        region_in_source = getattr(operator, "region_in_source_dim", "region_in_source")
        expected_names = ("source", region_in_source)
        if not isinstance(state_index, pd.MultiIndex) or tuple(state_index.names) != expected_names:
            raise ValueError(
                f"Site {site!r} H region coordinate must be a MultiIndex with levels "
                f"{expected_names!r} for the retained source-specific basis operator."
            )
        if not state_index.is_unique:
            raise ValueError(f"Site {site!r} H gathered region MultiIndex values must be unique.")
        source_values = state_index.get_level_values("source").to_numpy()
        _validated_source_labels(
            xr.DataArray(source_values, dims="region"),
            context=f"Site {site!r} H gathered region",
            require_unique=False,
        )
        expected_state = _require_dim_coordinate(
            basis_matrix,
            state_dim,
            context="Retained basis operator",
        )
        expected_index = basis_matrix.indexes.get(state_dim)
        if not isinstance(expected_index, pd.MultiIndex) or not state_index.equals(expected_index):
            raise ValueError(
                f"Site {site!r} H gathered region MultiIndex does not match the retained "
                f"basis operator state coordinate: found {state_coord.values.tolist()!r}, "
                f"expected {expected_state.values.tolist()!r}."
            )
        return

    if isinstance(state_index, pd.MultiIndex):
        raise ValueError(
            f"Site {site!r} H uses a gathered region MultiIndex, but the retained basis "
            "operator is not source-specific."
        )

    if "source" not in sensitivity.dims:
        if flux_sources is not None:
            raise ValueError(
                f"Site {site!r} H has no source dimension, but retained prior flux is "
                "source-resolved. Select a single-source BasisFunctions object first."
            )
        expected_state = _require_dim_coordinate(
            basis_matrix,
            state_dim,
            context="Retained basis operator",
        )
        if not np.array_equal(state_coord.values, expected_state.values):
            raise ValueError(
                f"Site {site!r} H region coordinate does not match the retained basis operator "
                f"state coordinate: found {state_coord.values.tolist()!r}, "
                f"expected {expected_state.values.tolist()!r}."
            )
        return

    source_coord = _require_dim_coordinate(sensitivity, "source", context=f"Site {site!r} H")
    sources = _validated_source_labels(
        source_coord,
        context=f"Site {site!r} H",
        require_unique=True,
    )
    if flux_sources is None:
        raise ValueError(
            f"Site {site!r} H is multisector with sources {sources!r}, but retained prior flux "
            "has no source dimension; a total flux cannot be reused for each sector."
        )
    if flux_sources != sources:
        raise ValueError(
            f"Site {site!r} H sources {sources!r} do not match retained prior flux sources/order "
            f"{flux_sources!r}."
        )
    expected_state = _require_dim_coordinate(
        basis_matrix,
        state_dim,
        context="Retained shared-basis operator",
    )
    if not np.array_equal(state_coord.values, expected_state.values):
        raise ValueError(
            f"Site {site!r} H region coordinate does not match the retained shared-basis "
            f"state coordinate: found {state_coord.values.tolist()!r}, "
            f"expected {expected_state.values.tolist()!r}."
        )


def _reject_unsupported_site_layout(data: xr.Dataset, *, context: str) -> None:
    """Reject dense multi-site and pre-stacked layouts outside the MVP contract.

    Args:
        data: Dataset whose dimensions should be checked.
        context: Human-readable input location for validation errors.

    Raises:
        ValueError: If ``data`` has a ``site`` or ``nmeasure`` dimension.
    """
    if "site" in data.dims:
        raise ValueError(
            f"{context} contains a 'site' dimension. Dense Dataset(site, time) inputs are "
            "unsupported; supply an ordered mapping/DataTree of site-local Datasets or load "
            "an established artifact with RhimePreparedInputs.load."
        )
    if "nmeasure" in data.dims:
        raise ValueError(
            f"{context} contains an 'nmeasure' dimension. Pre-stacked inputs are unsupported; "
            "supply an ordered mapping/DataTree of site-local time-indexed Datasets or load "
            "an established artifact with RhimePreparedInputs.load."
        )


def _validated_site_labels(values: Sequence[str], *, context: str) -> list[str]:
    """Return nonempty unique site labels without coercing other types.

    Args:
        values: Site labels in their semantic order.
        context: Human-readable input location for validation errors.

    Returns:
        Validated site labels as ordinary Python strings.

    Raises:
        ValueError: If a label is empty, non-string, or repeated.
    """
    labels: list[str] = []
    for value in values:
        if not isinstance(value, (str, np.str_)) or not value:
            raise ValueError(f"{context} site labels must be nonempty Python/NumPy strings; found {value!r}.")
        labels.append(str(value))
    if len(set(labels)) != len(labels):
        raise ValueError(f"{context} site labels must be unique; found {labels!r}.")
    return labels


def _sites_from_container(
    data: xr.DataTree | Mapping[str, xr.Dataset],
    sites: Sequence[str] | None,
) -> dict[str, xr.Dataset]:
    """Normalize supported xarray containers to an ordered site mapping.

    Args:
        data: Direct-child DataTree or ordered per-site Dataset mapping.
        sites: Optional site selection and explicit order.

    Returns:
        Ordered mapping of requested site names to shallow Dataset copies.

    Raises:
        TypeError: If the container or a mapping value is unsupported.
        ValueError: If the DataTree is empty or nested, a requested site is
            absent, or a selected Dataset has a dense or stacked layout.
    """
    if isinstance(data, xr.DataTree):
        tree_sites = _validated_site_labels(list(data.children), context="DataTree child")
        nested_sites = [site for site, node in data.children.items() if node.children]
        if nested_sites:
            raise ValueError(
                "An input DataTree must contain one direct child Dataset per site; "
                f"nested child nodes were found under {nested_sites!r}."
            )
        supplied = {
            site: data.children[original_site].to_dataset()
            for site, original_site in zip(tree_sites, data.children, strict=True)
        }
        if not supplied:
            raise ValueError("An input DataTree must contain one child dataset per site.")
    elif isinstance(data, Mapping):
        supplied = {}
        for site, site_data in data.items():
            validated_site = _validated_site_labels([site], context="Mapping key")[0]
            if validated_site in supplied:
                raise ValueError(
                    "Mapping key site labels must be unique after preserving string values; "
                    f"found repeated label {validated_site!r}."
                )
            if not isinstance(site_data, xr.Dataset):
                raise TypeError(
                    "Per-site xarray mappings must contain Dataset values; "
                    f"site {site!r} contains {type(site_data).__name__}."
                )
            supplied[validated_site] = site_data
    else:
        raise TypeError(
            "`data` must be an xarray DataTree or mapping of site names to Datasets; "
            "direct Dataset inputs are unsupported."
        )

    if sites is None:
        site_order = list(supplied)
    else:
        if isinstance(sites, (str, np.str_)):
            raise ValueError("`sites` must be a sequence of nonempty unique string site labels.")
        site_order = _validated_site_labels(sites, context="`sites`")
    if not site_order:
        raise ValueError("At least one site dataset is required.")
    missing = [site for site in site_order if site not in supplied]
    if missing:
        raise ValueError(f"Requested site(s) are absent from the supplied xarray data: {missing!r}.")
    selected: dict[str, xr.Dataset] = {}
    for site in site_order:
        dataset = supplied[site]
        _reject_unsupported_site_layout(dataset, context=f"Site {site!r} dataset")
        selected[site] = dataset.copy(deep=False)
    return selected


def _selected_sensitivity_name(site_data: xr.Dataset) -> str:
    """Return the sensitivity field selected for a site.

    Canonical ``H`` takes precedence, followed by ``fp_x_flux_sectoral`` and
    then ``fp_x_flux``.

    Args:
        site_data: Site-local input dataset.

    Returns:
        Selected variable name.

    Raises:
        ValueError: If no supported sensitivity field is present.
    """
    if "H" in site_data:
        return "H"
    cache_name = next((name for name in reversed(_CACHE_VARIABLES) if name in site_data), None)
    if cache_name is None:
        raise ValueError(
            "Each site dataset must contain canonical 'H' or a footprint-times-flux cache named "
            "'fp_x_flux' or 'fp_x_flux_sectoral'."
        )
    return cache_name


def _project_sensitivity(
    site_data: xr.Dataset,
    basis_functions: BasisFunctions,
    *,
    site: str,
) -> xr.DataArray:
    """Return canonical site sensitivity, projecting a cache when needed.

    Args:
        site_data: Validated site-local dataset containing ``H`` or a cache.
        basis_functions: Retained operator used for cache projection.
        site: Site name used in validation errors.

    Returns:
        Canonical ``H`` with cache units and ordered source labels retained.

    Raises:
        ValueError: If a cache is absent, source labels are invalid, or the
            projected state dimension cannot be identified.
    """
    selected_name = _selected_sensitivity_name(site_data)
    if selected_name == "H":
        return site_data["H"]

    cache = site_data[selected_name]
    source_order: list[str] | None = None
    if "source" in cache.dims:
        source_coord = _require_dim_coordinate(
            cache,
            "source",
            context=f"Site {site!r} {selected_name}",
        )
        source_order = _validated_source_labels(
            source_coord,
            context=f"Site {site!r} {selected_name}",
            require_unique=True,
        )
    operator_sources = _operator_source_order(basis_functions)
    flux_sources: list[str] | None = None
    if "source" in basis_functions.flux.dims:
        flux_sources = _validated_source_labels(
            _require_dim_coordinate(
                basis_functions.flux,
                "source",
                context="Retained prior flux",
            ),
            context="Retained prior flux",
            require_unique=True,
        )
    for context, retained_sources in (
        ("retained prior flux", flux_sources),
        ("retained basis operator", operator_sources),
    ):
        if retained_sources is not None and source_order != retained_sources:
            raise ValueError(
                f"Site {site!r} {selected_name} sources/order {source_order!r} do not match "
                f"{context} sources/order {retained_sources!r}."
            )
    sensitivity = basis_functions.sensitivity(cache)
    state_dim = basis_functions.operator.meta.state_dim
    if state_dim in sensitivity.dims and state_dim != "region":
        sensitivity = sensitivity.rename({state_dim: "region"})
    elif "region" in sensitivity.dims:
        state_dim = "region"
    else:
        extra_dims = [dim for dim in sensitivity.dims if dim not in cache.dims]
        if len(extra_dims) != 1:
            raise ValueError(
                "Could not identify the sensitivity state dimension from "
                f"sensitivity dims {sensitivity.dims!r} and cache dims {cache.dims!r}."
            )
        projected_state_dim = cast(str, extra_dims[0])
        sensitivity = sensitivity.rename({projected_state_dim: "region"})

    if source_order is not None and "source" in sensitivity.dims:
        sensitivity = sensitivity.reindex(source=source_order)
    return sensitivity.rename("H").assign_attrs(units=cache.attrs["units"])


def _normalise_release_coordinates(dataset: xr.Dataset, *, site: str) -> xr.Dataset:
    """Return a dataset with paired release coordinates aligned to ``time``.

    Scalar and singleton latitude/longitude pairs are stationary shorthand and
    are broadcast across the site's observations. Otherwise both arrays must
    have exactly dimension ``("time",)``. Legacy ``sitelats``/``sitelons``
    variables are deliberately ignored because they are not mobile-observation
    coordinates.

    Args:
        dataset: Site-local dataset with a validated ``time`` dimension.
        site: Site name used in validation errors.

    Returns:
        A shallow dataset copy with any release coordinate pair represented as
        time-indexed data variables.

    Raises:
        ValueError: If only one coordinate is present, the pair mixes
            stationary and time-varying layouts, a non-singleton array is not
            exactly time-indexed, or values are non-finite.
    """
    has_lat = "release_lat" in dataset
    has_lon = "release_lon" in dataset
    if has_lat != has_lon:
        raise ValueError(f"Site {site!r} release_lat and release_lon must be supplied together.")
    if not has_lat:
        return dataset

    latitude = dataset["release_lat"]
    longitude = dataset["release_lon"]
    latitude_stationary = latitude.size == 1
    longitude_stationary = longitude.size == 1
    if latitude_stationary != longitude_stationary:
        raise ValueError(
            f"Site {site!r} release_lat and release_lon must use matching stationary "
            "singleton or exact ('time',) layouts."
        )

    for name, coordinate in (("release_lat", latitude), ("release_lon", longitude)):
        if not latitude_stationary and coordinate.dims != ("time",):
            raise ValueError(
                f"Site {site!r} {name} must be scalar/singleton stationary shorthand "
                "or have exactly dimension ('time',)."
            )
        _require_finite(coordinate, context=f"Site {site!r} {name}")

    result = dataset.drop_vars(("release_lat", "release_lon")).copy(deep=False)
    if latitude_stationary:
        time = dataset.coords["time"]
        latitude = xr.DataArray(
            np.repeat(np.asarray(latitude.values).reshape(-1), dataset.sizes["time"]),
            dims=("time",),
            coords={"time": time},
            attrs=latitude.attrs,
        )
        longitude = xr.DataArray(
            np.repeat(np.asarray(longitude.values).reshape(-1), dataset.sizes["time"]),
            dims=("time",),
            coords={"time": time},
            attrs=longitude.attrs,
        )
    result["release_lat"] = latitude
    result["release_lon"] = longitude
    return result


def _validate_and_prepare_sites(
    site_data: Mapping[str, xr.Dataset],
    basis_functions: BasisFunctions,
) -> dict[str, xr.Dataset]:
    """Validate site contracts, project caches, and remove non-canonical caches.

    Args:
        site_data: Ordered site-local datasets to validate.
        basis_functions: Retained basis, prior flux, and operator metadata.

    Returns:
        Ordered canonical site datasets containing projected ``H`` and no
        footprint-times-flux caches.

    Raises:
        ValueError: If time, dimensions, units, source ordering, finite values,
            state compatibility, or cross-site layouts violate the contract.
    """
    prepared: dict[str, xr.Dataset] = {}
    release_sites = [
        site for site, dataset in site_data.items() if "release_lat" in dataset or "release_lon" in dataset
    ]
    if release_sites and len(release_sites) != len(site_data):
        missing = [site for site in site_data if site not in release_sites]
        raise ValueError(
            "Release coordinates must be supplied as a pair for every retained site; "
            f"missing both release_lat and release_lon for {missing!r}."
        )
    fixed_baseline_sites = [site for site, dataset in site_data.items() if "fixed_baseline" in dataset]
    if fixed_baseline_sites and len(fixed_baseline_sites) != len(site_data):
        missing = [site for site in site_data if site not in fixed_baseline_sites]
        raise ValueError(
            f"Fixed baseline data must be supplied for every retained site; missing {missing!r}."
        )

    expected_sources: list[str] | None = None
    expected_units: str | None = None
    source_layout_set = False
    for site, dataset in site_data.items():
        if "time" not in dataset.dims:
            raise ValueError(f"Site {site!r} dataset must contain a 'time' dimension.")
        _validate_time_coordinate(dataset, site=site)
        dataset = _normalise_release_coordinates(dataset, site=site)
        missing_vars = [name for name in _REQUIRED_OBSERVATION_VARIABLES if name not in dataset]
        if missing_vars:
            raise ValueError(f"Site {site!r} dataset is missing required variable(s): {missing_vars!r}.")
        invalid_observation_dims = [
            name for name in _REQUIRED_OBSERVATION_VARIABLES if dataset[name].dims != ("time",)
        ]
        if invalid_observation_dims:
            raise ValueError(
                f"Site {site!r} observation variable(s) must have exactly dimension ('time',): "
                f"{invalid_observation_dims!r}."
            )
        for name in _REQUIRED_OBSERVATION_VARIABLES:
            _require_finite(dataset[name], context=f"Site {site!r} {name}")

        selected_sensitivity_name = _selected_sensitivity_name(dataset)
        concentration_names = [
            *_REQUIRED_OBSERVATION_VARIABLES,
            selected_sensitivity_name,
            *(name for name in ("H_bc", "fixed_baseline") if name in dataset),
        ]
        mf_units = _require_concentration_units(
            dataset["mf"],
            context=f"Site {site!r} mf",
        )
        if expected_units is None:
            expected_units = mf_units
        elif mf_units != expected_units:
            raise ValueError(
                f"Site {site!r} mf units {mf_units!r} do not exactly match the retained "
                f"concentration units {expected_units!r}; the adapter does not convert units."
            )
        for name in concentration_names[1:]:
            units = _require_concentration_units(
                dataset[name],
                context=f"Site {site!r} {name}",
            )
            if units != expected_units:
                raise ValueError(
                    f"Site {site!r} {name} units {units!r} do not match mf units exactly "
                    f"{expected_units!r}; the adapter does not convert units."
                )

        if "fixed_baseline" in dataset:
            fixed_baseline = dataset["fixed_baseline"]
            if fixed_baseline.dims != ("time",):
                raise ValueError(
                    f"Site {site!r} fixed_baseline must have exactly the observation dimension ('time',)."
                )
            _require_finite(fixed_baseline, context=f"Site {site!r} fixed_baseline")
        if "H_bc" in dataset and dataset["H_bc"].dims not in {
            ("time", "bc_region"),
            ("bc_region", "time"),
        }:
            raise ValueError(
                f"Site {site!r} H_bc must have exactly dimensions 'time' and 'bc_region' in either order."
            )
        if "H_bc" in dataset:
            _require_finite(dataset["H_bc"], context=f"Site {site!r} H_bc")

        sensitivity = _project_sensitivity(dataset, basis_functions, site=site)
        expected_h_dims = ("region", "time", "source") if "source" in sensitivity.dims else ("region", "time")
        if sensitivity.dims != expected_h_dims:
            raise ValueError(
                f"Site {site!r} H must have exactly dimensions {expected_h_dims!r}; "
                f"found {sensitivity.dims!r}."
            )
        _require_finite(sensitivity, context=f"Site {site!r} projected H")
        _validate_state_and_source_compatibility(sensitivity, basis_functions, site=site)
        sources = (
            _validated_source_labels(
                sensitivity.coords["source"],
                context=f"Site {site!r} H",
                require_unique=True,
            )
            if "source" in sensitivity.dims
            else None
        )
        if not source_layout_set:
            expected_sources = sources
            source_layout_set = True
        elif sources != expected_sources:
            raise ValueError(
                "All site sensitivities must use the same source layout and order; "
                f"site {site!r} has {sources!r}, expected {expected_sources!r}."
            )

        non_observation_variables = [
            name for name, variable in dataset.data_vars.items() if "time" not in variable.dims
        ]
        canonical = dataset.drop_vars(
            (*_CACHE_VARIABLES, *non_observation_variables),
            errors="ignore",
        ).copy(deep=False)
        unused_dims = [
            dim
            for dim in canonical.dims
            if all(dim not in variable.dims for variable in canonical.data_vars.values())
        ]
        if unused_dims:
            canonical = canonical.drop_dims(unused_dims)
        if selected_sensitivity_name != "H":
            canonical["H"] = sensitivity
        prepared[site] = canonical
    return prepared


def _normalise_averaging_period(
    averaging_period: str | Sequence[str | None] | Mapping[str, str | None] | None,
    sites: Sequence[str],
) -> tuple[str | None, ...]:
    """Return averaging periods aligned to the retained site order.

    Args:
        averaging_period: One common value, a site-keyed mapping, a sequence
            aligned with ``sites``, or ``None``.
        sites: Retained site names in semantic order.

    Returns:
        One averaging-period value per retained site.

    Raises:
        ValueError: If a mapping omits a site or a sequence has the wrong
            length.
    """
    if averaging_period is None or isinstance(averaging_period, str):
        return tuple(averaging_period for _ in sites)
    if isinstance(averaging_period, Mapping):
        missing = [site for site in sites if site not in averaging_period]
        if missing:
            raise ValueError(f"`averaging_period` is missing retained site(s): {missing!r}.")
        return tuple(averaging_period[site] for site in sites)

    periods = tuple(averaging_period)
    if len(periods) != len(sites):
        raise ValueError(
            "`averaging_period` must contain one value per retained site; "
            f"received {len(periods)} for {len(sites)} sites."
        )
    return periods


def _validate_retained_site_order(inv_inputs: xr.Dataset, requested_sites: Sequence[str]) -> None:
    """Require gathered observations to retain every requested site in order.

    Args:
        inv_inputs: Canonical gathered inputs returned by ``make_inv_inputs``.
        requested_sites: Required site names in semantic input order.

    Raises:
        ValueError: If any site has been emptied or the retained order changed.
    """
    retained_sites = (
        _ordered_unique_strings(np.asarray(inv_inputs["site"].values)) if "site" in inv_inputs.coords else []
    )
    requested = list(requested_sites)
    if retained_sites == requested:
        return

    emptied = [site for site in requested if site not in retained_sites]
    if emptied:
        raise ValueError(
            "Input gathering removed every active observation for site(s) "
            f"{emptied!r}; each requested site must retain at least one row."
        )
    raise ValueError(
        "Input gathering changed the requested site order: "
        f"retained {retained_sites!r}, requested {requested!r}."
    )


def prepare_rhime_inputs_from_xarray(
    data: xr.DataTree | Mapping[str, xr.Dataset],
    *,
    basis_functions: BasisFunctions,
    sites: Sequence[str] | None = None,
    averaging_period: str | Sequence[str | None] | Mapping[str, str | None] | None = None,
    bc_freq: str | None = None,
    min_error: str | dict[str, float] | float = 0.0,
    min_error_per_site: bool = False,
    start_date: DatetimeLike | None = None,
) -> RhimePreparedInputs:
    """Create canonical RHIME inputs from source-neutral xarray site data.

    Input must be an ordered mapping or a DataTree with one direct child
    Dataset per site. Mapping keys and DataTree child names define site order
    unless ``sites`` explicitly selects and orders them. Direct Dataset, dense
    ``Dataset(site, time)``, and pre-stacked ``nmeasure`` layouts are rejected.

    Each site dataset must follow the module-level variable contract. Cached
    ``fp_x_flux`` or ``fp_x_flux_sectoral`` is projected through
    ``basis_functions.sensitivity`` before site-time observations are gathered
    to ``nmeasure``. Existing ``H`` is accepted directly. The retained
    operator and prior flux are preserved; artifact provenance may be added to
    a new ``BasisFunctions`` value in the returned prepared inputs.

    Every row is active. Per-site ``time`` coordinates must be explicit,
    nonempty, unique ``datetime64`` values without ``NaT``; valid
    non-monotonic order is preserved. Observation variables must have exactly
    dimension ``("time",)``. Required observations, projected ``H``, and
    optional baseline fields must be finite. Required observation and error
    fields, selected ``H`` or cache, and optional ``H_bc`` and
    ``fixed_baseline`` must have the same exact nonempty unit string as ``mf``
    across all sites; no conversion is performed. Release-coordinate pairs and
    ``fixed_baseline`` are each all-or-none across retained sites. Labels on an
    explicit ``source`` dimension are nonempty, unique Python or NumPy strings
    and are never coerced from bytes or numbers. Repeated source values in a
    gathered source-specific state MultiIndex remain valid.

    Persist and reopen canonical artifacts with
    :meth:`RhimePreparedInputs.save
    <openghg_inversions.inversion_data.preparation.RhimePreparedInputs.save>`
    and
    :meth:`~openghg_inversions.inversion_data.preparation.RhimePreparedInputs.load`;
    do not pass serialized ``nmeasure`` data back to this adapter.

    Args:
        data: Site data as a DataTree or site-to-Dataset mapping.
        basis_functions: Self-contained retained basis object, including the
            prior/reference flux used for posterior reconstruction. Multisector
            data require matching ordered ``source`` coordinates on retained
            flux and any source-specific operator.
        sites: Optional site selection and order.
        averaging_period: One common period, a site-aligned sequence or
            mapping, or ``None``.
        bc_freq: Frequency used to expand sampled ``H_bc`` contributions.
        min_error: Minimum model error accepted by ``make_inv_inputs``.
        min_error_per_site: Whether calculated minimum errors vary by site.
            Defaults to ``False``, matching the OpenGHG-backed RHIME
            preparation route.
        start_date: Optional frequency anchor passed to ``make_inv_inputs``.

    Returns:
        Canonical prepared inputs accepted by
        :func:`openghg_inversions.rhime.run_rhime_from_prepared_inputs`.

    Raises:
        TypeError: If the input container or mapping values are unsupported,
            or ``basis_functions`` is not a ``BasisFunctions`` object.
        ValueError: If site data violate the documented variable, dimension,
            source-order, or site-alignment contract.
    """
    if not isinstance(basis_functions, BasisFunctions):
        raise TypeError("`basis_functions` must be a self-contained BasisFunctions object.")

    per_site = _sites_from_container(data, sites)
    prepared_sites = _validate_and_prepare_sites(per_site, basis_functions)
    site_order = list(prepared_sites)
    inv_inputs = make_inv_inputs(
        prepared_sites,
        sites=site_order,
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_per_site=min_error_per_site,
        start_date=start_date,
        missing_data_vars="error",
    )
    _validate_retained_site_order(inv_inputs, site_order)

    return RhimePreparedInputs.from_legacy_inputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        sites=tuple(site_order),
        averaging_period=_normalise_averaging_period(averaging_period, site_order),
        basis_artifact_source=basis_functions.basis_artifact_source or "supplied",
        basis_artifact_path=basis_functions.basis_artifact_path,
    )


__all__ = ["prepare_rhime_inputs_from_xarray"]
