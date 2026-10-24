"""Nested-domain preparation and execution for modern RHIME.

Nested domains are independent spatial resolutions of the same emissions
source.  They are deliberately not represented as RHIME emissions sectors:
the outer and inner basis operators may have different grids, state labels,
and region counts.  This module preserves those objects separately, masks the
outer prior response over the inner footprint extent before either basis is
built, and combines only the two observation-aligned sensitivity matrices.
"""

from __future__ import annotations

from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions._provenance import installed_ogi_provenance

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.basis import prepare_basis
from openghg_inversions.rhime.preparation import resolve_basis_inputs
from openghg_inversions.rhime._resources import resolve_resource_paths

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral
import warnings
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from dask import compute as dask_compute, is_dask_collection
from dask.base import unpack_collections
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from openghg.util import align_lat_lon, get_domain_info

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.array_ops import force_align, to_dense
from ._domain_support import rectangular_extent_mask, remove_domain_overlap
from openghg_inversions.inversion_data import MergedData, PreparedInputs, SiteOptions
from openghg_inversions.models.components import add_linear_component, add_offset_component
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.model_error import MinErrorConfig
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity
from openghg_inversions.observation_error import (
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput

from ._model_building import ForwardModelTerms, add_rhime_likelihood
from .builders import RhimeLikelihoodBuilder, RhimeModelBuildResult
from .materialization import _materialize_pymc_inputs_and_priors
from .outputs import (
    make_model_output, validate_nested_output_resources, annotate_sample_trace, save_inversion_output,
    _define_derived_output_filename,
)
from .ini import read_rhime_ini
from .params import DataConfig, RhimeConfig
from .preparation import (
    acquire_merged_data,
    assemble_inputs,
    build_sensitivities,
    export_basis,
    filter_observations,
    prepare_observation_errors,
)
from openghg_inversions.inference.sampling import resolve_sampler
from .sampling import sample_rhime_model
from .specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    RhimeModelSpec,
    RhimeOutputSpec,
)
from .standard import standard_model_input_names, _standard_prior_parameters

__all__ = [
    "NestedRhimePreparedInputs",
    "NestedOutput",
    "inner_data_config",
    "build_nested_rhime_model",
    "align_inner_merged_to_outer_observations",
    "combine_nested_rhime_inputs",
    "make_nested_inversion_outputs",
    "mask_outer_merged_for_inner_domain",
    "nested_model_input_names",
    "prepare_nested_rhime_inputs",
    "run_nested",
    "materialize_nested_inputs",
]


_PREPARED_OVERLAP_MASK_POLICY = "caller_supplied_pre_masked_outer_sensitivity"
_RETRIEVAL_OVERLAP_MASK_POLICY = (
    "outer_footprint_and_flux_zeroed_within_shared_inner_grid_rectangular_extent_before_basis_projection"
)


@dataclass(frozen=True)
class NestedRhimePreparedInputs:
    """Prepared outer, inner, and combined nested-domain inputs.

    ``outer`` and ``inner`` retain their native basis operators and flux grids.
    ``combined`` retains the outer basis for the ordinary RHIME validation
    contract and adds the aligned inner sensitivity as ``H_inner``.  The inner
    basis remains available explicitly on ``inner`` and is never coerced onto
    the outer grid. ``outer_overlap_mask_policy`` records how that support was
    removed so saved per-domain artifacts remain scientifically interpretable.

    Args:
        outer: Prepared outer-domain inputs with the inner-domain spatial
            response removed before basis projection.
        inner: Prepared inner-domain inputs on their native grid.
        combined: Observation-aligned inputs containing outer ``H`` and inner
            ``H_inner`` sensitivities.
        time_tolerance: Maximum time difference allowed when aligning inner
            observations to the outer observation index.
        inner_state_dim: State dimension assigned to the inner sensitivity.
        inner_domain_label: Optional label retained for the inner domain.
        outer_overlap_masked: Whether the caller confirms that the outer
            response was masked before basis projection.
        outer_overlap_mask_policy: Machine-readable description of how the
            overlapping outer response was removed.
        outer_overlap_mask_extent: Applied inclusive rectangle as latitude minimum,
            latitude maximum, longitude minimum and longitude maximum. Omission
            leaves the actual mask bounds unknown, including for supplied inputs.
    """

    outer: PreparedInputs
    inner: PreparedInputs
    combined: PreparedInputs
    time_tolerance: str | pd.Timedelta | None = None
    inner_state_dim: str = "inner_region"
    inner_domain_label: str | None = None
    outer_overlap_masked: bool = False
    outer_overlap_mask_policy: str = _PREPARED_OVERLAP_MASK_POLICY
    outer_overlap_mask_extent: tuple[float, float, float, float] | None = None

    def validated(self) -> NestedRhimePreparedInputs:
        """Revalidate both native preparations and rebuild their combination."""
        return combine_nested_rhime_inputs(
            self.outer,
            self.inner,
            time_tolerance=self.time_tolerance,
            inner_state_dim=self.inner_state_dim,
            inner_domain_label=self.inner_domain_label,
            outer_overlap_masked=self.outer_overlap_masked,
            outer_overlap_mask_policy=self.outer_overlap_mask_policy,
            outer_overlap_mask_extent=self.outer_overlap_mask_extent,
        )


@dataclass(frozen=True)
class NestedOutput:
    """Native outer and inner scientific views borrowing one sampled trace.

    Attributes:
        outer: Masked outer-grid flux view, also owning shared concentrations.
        inner: Native inner-grid flux view with its own state-axis mapping.
        products: Computed PARIS datasets, when requested.
        paths: Written product and artifact destinations.
    """

    outer: InversionOutput
    inner: InversionOutput
    products: dict[str, xr.Dataset] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


def _domain_variable_roles(all_roles: Mapping[str, str], *, tag: str) -> dict[str, str]:
    """Return one nested domain's role mapping without its tag suffix."""
    roles: dict[str, str] = {}
    for role, name in all_roles.items():
        if ":" in role:
            base_role, role_tag = role.split(":", 1)
            if role_tag == tag:
                roles[base_role] = name
        else:
            roles[role] = name
    return roles


def _grid_identity(prepared: PreparedInputs) -> dict[str, Any]:
    """Return descriptive spatial sizes and extents for one domain."""
    flux = prepared.basis_functions.flux
    missing = [name for name in ("lat", "lon") if name not in flux.coords]
    if missing:
        raise ValueError(f"Nested basis flux is missing spatial coordinate(s): {missing!r}.")

    identity: dict[str, Any] = {}
    for name in ("lat", "lon"):
        values = np.asarray(flux[name].values)
        identity[name] = {
            "size": int(values.size),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return identity


def _nested_domain_artifact_metadata(
    prepared: NestedRhimePreparedInputs,
    *,
    outer_domain: str,
    view: str,
    outer_grid: dict[str, Any],
    inner_grid: dict[str, Any],
) -> dict[str, Any]:
    """Describe nested identities, grids, and masking in one durable view."""
    inner_domain = prepared.inner_domain_label or f"{outer_domain}-inner"
    extent = prepared.outer_overlap_mask_extent
    extent_source = "unknown" if extent is None else (
        "inner_footprint_grid" if prepared.outer_overlap_mask_policy == _RETRIEVAL_OVERLAP_MASK_POLICY
        else "caller_supplied"
    )
    return {
        "view": view,
        "view_domain": outer_domain if view == "outer" else inner_domain,
        "outer_domain": outer_domain,
        "inner_domain": inner_domain,
        "outer_grid": outer_grid,
        "inner_grid": inner_grid,
        "outer_support": {
            "overlap_masked": prepared.outer_overlap_masked,
            "mask_policy": prepared.outer_overlap_mask_policy,
            "mask_extent_source": extent_source,
            "inner_extent": None if extent is None else {
                "latitude": list(extent[:2]),
                "longitude": list(extent[2:]),
            },
        },
    }


def make_nested_inversion_outputs(
    *,
    prepared: NestedRhimePreparedInputs,
    model_spec: RhimeModelSpec,
    posterior: xr.DataTree,
    output_contract: OutputContract,
    start_date: str,
    end_date: str,
    actual_inference: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    inner_state_activity: StateActivity | None = None,
) -> NestedOutput:
    """Bind one posterior to each native grid without computing or writing arrays.

    Args:
        prepared: Borrowed outer, inner and combined scientific preparations.
        model_spec: Resolved scientific choices; its domain remains the reporting
            domain for both views. The actual inner label lives in nested metadata.
        posterior: Shared trace with construction-owned prior metadata.
        output_contract: Built model roles and native state dimensions.
        start_date: Requested inversion start.
        end_date: Requested inversion end.
        actual_inference: Description of inference that ran.
        provenance: Package provenance; omission records the installed package.
        inner_state_activity: Activity policy used for the inner state.

    Returns:
        Two views borrowing their native arrays and the same posterior.
    """
    outer_grid = _grid_identity(prepared.outer)
    inner_grid = _grid_identity(prepared.inner)
    outputs = []
    for tag, native in (("outer", prepared.outer), ("inner", prepared.inner)):
        contract = replace(
            output_contract,
            variable_roles=_domain_variable_roles(output_contract.variable_roles, tag=tag),
            state_dimension_mapping={
                "trace": str(output_contract.metadata[f"{tag}_state_dimension"]),
                "basis": native.basis_functions.operator.meta.state_dim,
            },
        )
        native_model_spec = model_spec if tag == "outer" else replace(
            model_spec, state_activity=inner_state_activity,
            sectors=(replace(model_spec.sectors[0], state_activity=inner_state_activity),),
        )
        output = make_model_output(
            prepared=native, model_spec=native_model_spec, posterior=posterior,
            output_contract=contract, start_date=start_date, end_date=end_date,
            actual_inference=actual_inference, provenance=provenance, split_by_sectors=False,
        )
        output.output_metadata["nested_domain"] = _nested_domain_artifact_metadata(
            prepared, outer_domain=model_spec.domain, view=tag,
            outer_grid=outer_grid, inner_grid=inner_grid,
        )
        outputs.append(output)
    return NestedOutput(outer=outputs[0], inner=outputs[1])


def _state_dimension(sensitivity: xr.DataArray, *, label: str) -> str:
    """Return the single non-measurement state dimension."""
    if "nmeasure" not in sensitivity.dims:
        raise ValueError(f"{label} sensitivity must contain the `nmeasure` dimension.")
    state_dims = [str(dim) for dim in sensitivity.dims if dim != "nmeasure"]
    if len(state_dims) != 1:
        raise ValueError(f"{label} sensitivity must have exactly one state dimension; found {state_dims!r}.")
    return state_dims[0]


def _measurement_index(prepared: PreparedInputs, *, label: str) -> pd.MultiIndex:
    """Return the canonical site/time measurement index."""
    index = prepared.inv_inputs.indexes.get("nmeasure")
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != ("site", "time"):
        raise ValueError(f"{label} inputs require an nmeasure MultiIndex with levels ('site', 'time').")
    return index


def _validate_one_to_one_inner_time_matches(
    outer_times: pd.DatetimeIndex,
    inner_times: pd.DatetimeIndex,
    inner_indexer: np.ndarray,
    *,
    label: str,
) -> None:
    """Reject time alignment that would reuse one inner footprint."""
    matched_positions = inner_indexer[inner_indexer >= 0]
    positions, counts = np.unique(matched_positions, return_counts=True)
    reused_positions = positions[counts > 1]
    if reused_positions.size == 0:
        return

    examples = []
    for position in reused_positions[:5]:
        matched_outer_times = outer_times[inner_indexer == position]
        outer_preview = ", ".join(str(time) for time in matched_outer_times)
        examples.append(f"{inner_times[int(position)]} for outer observations {outer_preview}")
    suffix = "" if reused_positions.size <= 5 else f", ... ({reused_positions.size} reused times total)"
    raise ValueError(
        f"{label} cannot align one-to-one because nearest-time matching would reuse "
        f"an inner footprint: {'; '.join(examples)}{suffix}."
    )


def _site_time_indexer(
    outer_times: pd.DatetimeIndex,
    inner_times: pd.DatetimeIndex,
    *,
    site: str,
    time_tolerance: str | pd.Timedelta | None,
) -> np.ndarray:
    """Select native inner positions once per site, without reusing footprints."""
    if outer_times.has_duplicates or inner_times.has_duplicates:
        raise ValueError(f"Nested inputs for site {site!r} contain duplicate timestamps.")
    tolerance = None if time_tolerance is None else pd.Timedelta(time_tolerance)
    order = inner_times.argsort()
    sorted_times = inner_times.take(order)
    if tolerance is None:
        indexer = sorted_times.get_indexer(outer_times)
    else:
        indexer = sorted_times.get_indexer(outer_times, method="nearest", tolerance=tolerance)
    missing = indexer < 0
    if bool(missing.any()):
        missing_times = outer_times[missing]
        preview = ", ".join(str(time) for time in missing_times[:5])
        suffix = "" if len(missing_times) <= 5 else f", ... ({len(missing_times)} total)"
        alignment = "exact timestamps" if tolerance is None else f"nearest timestamps within {tolerance}"
        raise ValueError(
            f"Inner nested inputs for site {site!r} cannot be aligned to every outer observation "
            f"using {alignment}: {preview}{suffix}."
        )
    _validate_one_to_one_inner_time_matches(
        outer_times, sorted_times, indexer, label=f"Inner nested inputs for site {site!r}",
    )
    return order[indexer]


def _inner_measurement_positions(
    outer_index: pd.MultiIndex,
    inner_index: pd.MultiIndex,
    *,
    time_tolerance: str | pd.Timedelta | None,
) -> list[int]:
    """Map each outer site/time observation to an inner observation position."""
    positions = np.empty(len(outer_index), dtype=int)
    outer_sites = outer_index.get_level_values("site")
    inner_sites = inner_index.get_level_values("site")
    for site in dict.fromkeys(outer_sites):
        outer_positions = np.flatnonzero(outer_sites == site)
        inner_positions = np.flatnonzero(inner_sites == site)
        indexer = _site_time_indexer(
            pd.DatetimeIndex(outer_index.get_level_values("time").take(outer_positions)),
            pd.DatetimeIndex(inner_index.get_level_values("time").take(inner_positions)),
            site=str(site), time_tolerance=time_tolerance,
        )
        positions[outer_positions] = inner_positions[indexer]
    return positions.tolist()


def combine_nested_rhime_inputs(
    outer: PreparedInputs,
    inner: PreparedInputs,
    *,
    time_tolerance: str | pd.Timedelta | None = None,
    inner_state_dim: str = "inner_region",
    inner_domain_label: str | None = None,
    outer_overlap_masked: bool = False,
    outer_overlap_mask_policy: str = _PREPARED_OVERLAP_MASK_POLICY,
    outer_overlap_mask_extent: tuple[float, float, float, float] | None = None,
) -> NestedRhimePreparedInputs:
    """Combine independently prepared native grids at the sensitivity boundary.

    Exact site/time alignment is required by default. Set ``time_tolerance``
    to opt into nearest-time alignment for footprints whose timestamps differ
    slightly from the observations. Unmatched observations are rejected;
    they are never silently duplicated or converted to zero sensitivity.

    Prepared sensitivity matrices cannot be spatially masked after basis
    projection. Callers must therefore remove the inner-domain contribution
    from the outer merged data before preparing ``outer`` and explicitly set
    ``outer_overlap_masked=True``. The retrieval-backed nested runner performs
    that masking itself. Requiring the acknowledgement here prevents ordinary,
    overlapping prepared inputs from being combined silently. Prepared-input
    callers may also provide a more specific ``outer_overlap_mask_policy``;
    otherwise the durable metadata records that the mask was caller supplied.
    ``outer_overlap_mask_extent`` records the actual applied rectangle, when
    known; it is never inferred from inner basis or flux coordinates.
    """
    if outer_overlap_masked is not True:
        raise ValueError(
            "Nested prepared inputs require an outer sensitivity with the inner-domain overlap "
            "already removed; pass outer_overlap_masked=True only after masking the outer merged data."
        )
    if not isinstance(outer_overlap_mask_policy, str) or not outer_overlap_mask_policy.strip():
        raise ValueError("`outer_overlap_mask_policy` must describe how outer support was removed.")
    if outer_overlap_mask_extent is not None:
        if (
            len(outer_overlap_mask_extent) != 4
            or not np.isfinite(outer_overlap_mask_extent).all()
            or outer_overlap_mask_extent[0] > outer_overlap_mask_extent[1]
            or outer_overlap_mask_extent[2] > outer_overlap_mask_extent[3]
        ):
            raise ValueError("`outer_overlap_mask_extent` must contain four finite, ordered rectangle bounds.")
    outer = outer.validated()
    inner = inner.validated()
    if outer.sites != inner.sites:
        raise ValueError(
            "Nested RHIME outer and inner prepared inputs must retain the same sites in the same order; "
            f"outer={outer.sites!r}, inner={inner.sites!r}."
        )
    if outer.averaging_period != inner.averaging_period:
        raise ValueError(
            "Nested RHIME outer and inner averaging-period metadata must match; "
            f"outer={outer.averaging_period!r}, inner={inner.averaging_period!r}."
        )

    outer_h = outer.inv_inputs["H"]
    inner_h = inner.inv_inputs["H"]
    outer_state_dim = _state_dimension(outer_h, label="Outer")
    source_coord = outer_h.coords.get("source")
    inner_source_coord = inner_h.coords.get("source")
    if "source" in outer_h.dims or "source" in inner_h.dims:
        raise ValueError(
            "Nested-domain preparation currently supports one emissions source; use ordinary "
            "run_multisector for same-grid sector separation."
        )
    if source_coord is not None and inner_source_coord is not None:
        if str(source_coord.item()) != str(inner_source_coord.item()):
            raise ValueError(
                "Nested RHIME outer and inner sensitivities must describe the same emissions source."
            )
    if inner_state_dim == outer_state_dim:
        raise ValueError(f"`inner_state_dim` must differ from the outer state dimension {outer_state_dim!r}.")
    if not inner_state_dim or inner_state_dim in outer.inv_inputs.dims:
        raise ValueError(
            "`inner_state_dim` must be a non-empty dimension name absent from the outer inputs; "
            f"got {inner_state_dim!r}."
        )

    outer_index = _measurement_index(outer, label="Outer")
    inner_index = _measurement_index(inner, label="Inner")
    if outer_index.has_duplicates:
        raise ValueError("Outer nested inputs contain duplicate site/time observations.")
    positions = _inner_measurement_positions(
        outer_index,
        inner_index,
        time_tolerance=time_tolerance,
    )
    native_inner_state_dim = _state_dimension(inner_h, label="Inner")
    selected_inner = (
        inner_h.isel(nmeasure=positions)
        .transpose(native_inner_state_dim, "nmeasure")
        .rename({native_inner_state_dim: inner_state_dim})
    )
    inner_state_coordinate = selected_inner.coords.get(inner_state_dim)
    combined_dataset = outer.inv_inputs.copy(deep=False)
    combined_dataset["H_inner"] = xr.Variable(
        (inner_state_dim, "nmeasure"),
        selected_inner.data,
        attrs=dict(selected_inner.attrs),
    )
    if inner_state_coordinate is not None:
        combined_dataset = combined_dataset.assign_coords({inner_state_dim: inner_state_coordinate.variable})
    combined = PreparedInputs(
        inv_inputs=combined_dataset,
        basis_functions=outer.basis_functions,
        site_metadata=outer.site_metadata,
    )
    return NestedRhimePreparedInputs(
        outer=outer,
        inner=inner,
        combined=combined,
        time_tolerance=time_tolerance,
        inner_state_dim=inner_state_dim,
        inner_domain_label=inner_domain_label,
        outer_overlap_masked=True,
        outer_overlap_mask_policy=outer_overlap_mask_policy,
        outer_overlap_mask_extent=outer_overlap_mask_extent,
    )


def _mask_spatial_variables(dataset: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """Return a shallow dataset copy with prior-response arrays lazily masked."""
    result = dataset.copy(deep=False)
    for name in ("fp", "fp_x_flux", "fp_x_flux_sectoral"):
        if name in result and set(mask.dims).issubset(result[name].dims):
            result[name] = remove_domain_overlap(result[name], mask)
    return result


@dataclass(frozen=True)
class _NestedDomainData:
    """Borrowed preparation intermediates, separate from the acquisition handoff."""

    site_data: Mapping[str, xr.Dataset]
    flux_data: Mapping[str, xr.Dataset]
    site_options: SiteOptions

    @property
    def sites(self) -> tuple[str, ...]:
        """Site order owned by the retained selectors."""
        return self.site_options.sites

    def with_site_data(
        self, site_data: Mapping[str, xr.Dataset], *, context: str,
    ) -> _NestedDomainData:
        """Retain aligned selectors and report sites dropped during preparation."""
        return replace(
            self, site_data=site_data,
            site_options=self.site_options.retain_sites(tuple(site_data), context=context),
        )


_DomainData = TypeVar("_DomainData", MergedData, _NestedDomainData)


def _select_merged_sites(merged: _DomainData, sites: Sequence[str]) -> _DomainData:
    """Return a borrowed merged-data view restricted to a common site set."""
    return merged.with_site_data(
        {str(site): merged.site_data[str(site)] for site in sites},
        context="Nested-domain alignment",
    )


def _retain_common_sites(
    outer: _DomainData,
    inner: _DomainData,
) -> tuple[_DomainData, _DomainData]:
    """Select sites present in both domains, preserving outer order."""
    common_sites = tuple(site for site in outer.sites if site in set(inner.sites))
    if not common_sites:
        raise ValueError("No common sites remain across outer and inner nested-domain data preparation.")
    return _select_merged_sites(outer, common_sites), _select_merged_sites(inner, common_sites)


def align_inner_merged_to_outer_observations(
    outer: MergedData | _NestedDomainData,
    inner: _DomainData,
    *,
    time_tolerance: str | pd.Timedelta | None = None,
) -> _DomainData:
    """Select inner scenarios at the filtered outer observation timestamps.

    Observation filters are evaluated once on the canonical outer scenario.
    This function then applies that retained observation selection to every
    inner-domain array. Exact matching is the default; nearest selection is
    site-local and requires an explicit tolerance.
    """
    if outer.sites != inner.sites:
        raise ValueError("Nested time alignment requires identical outer and inner site order.")
    site_data = dict(inner.site_data)
    for site in outer.sites:
        outer_dataset = outer.site_data[site]
        inner_dataset = inner.site_data[site]
        if not isinstance(outer_dataset, xr.Dataset) or not isinstance(inner_dataset, xr.Dataset):
            raise TypeError("Modern nested time alignment requires per-site xarray Datasets.")
        if "time" not in outer_dataset.indexes or "time" not in inner_dataset.indexes:
            raise ValueError(f"Nested datasets for site {site!r} require indexed time coordinates.")
        outer_times = pd.DatetimeIndex(outer_dataset.get_index("time"))
        inner_times = pd.DatetimeIndex(inner_dataset.get_index("time"))
        native_positions = _site_time_indexer(
            outer_times, inner_times, site=site, time_tolerance=time_tolerance,
        )
        aligned = inner_dataset.isel(time=native_positions).assign_coords(time=outer_dataset["time"].variable)
        site_data[site] = aligned
    return inner.with_site_data(site_data, context="Nested-domain time alignment")


def _align_domain_sites(
    data: MergedData | _NestedDomainData, *, domain: str | None, label: str,
) -> _NestedDomainData:
    """Normalize site and corresponding flux labels to one domain reference."""
    if not data.sites:
        raise ValueError("Nested-domain masking requires at least one retained site.")
    known = domain is not None and domain.upper() in get_domain_info()
    reference_site = data.sites[0]
    reference = data.site_data[reference_site]
    if domain is not None and not known:
        warnings.warn(
            f"Nested domain {domain!r} is not in OpenGHG domain definitions; "
            f"using retained site {reference_site!r} as its spatial reference.",
            UserWarning, stacklevel=2,
        )
    site_data = {}
    for site in data.sites:
        dataset = data.site_data[site]
        for name in ("lat", "lon"):
            if name not in dataset.indexes or dataset[name].dims != (name,):
                raise ValueError(f"{label} dataset for site {site!r} requires indexed {name!r} coordinates.")
            if dataset.get_index(name).empty:
                raise ValueError(f"{label} dataset for site {site!r} has empty {name!r} coordinates.")
        try:
            aligned = align_lat_lon(dataset, domain) if known and domain is not None else force_align(
                dataset, reference, dims=("lat", "lon"),
            )
        except ValueError as exc:
            raise ValueError(f"{label} nested-domain sites must share the same indexed lat/lon grid.") from exc
        # OpenGHG replaces coordinate labels without retaining their metadata.
        aligned = aligned.assign_coords({
            name: aligned[name].assign_attrs(dataset[name].attrs) for name in ("lat", "lon")
        })
        site_data[site] = aligned
    flux_data = dict(data.flux_data)
    for source, dataset in flux_data.items():
        try:
            aligned = align_lat_lon(dataset, domain) if known and domain is not None else force_align(
                dataset, site_data[reference_site], dims=("lat", "lon"),
            )
        except ValueError:
            # Inventories on genuinely different grids retain their native labels.
            continue
        flux_data[source] = aligned.assign_coords({
            name: aligned[name].assign_attrs(dataset[name].attrs) for name in ("lat", "lon")
        })
    return _NestedDomainData(site_data, flux_data, data.site_options)


def mask_outer_merged_for_inner_domain(
    outer: MergedData | _NestedDomainData,
    inner: MergedData | _NestedDomainData,
    *,
    outer_domain: str | None = None,
    inner_domain: str | None = None,
) -> _NestedDomainData:
    """Mask outer footprint response and prior flux over the inner extent.

    The returned preparation record holds shallow dataset copies and
    ``where`` preserves Dask laziness. Acquired ``MergedData`` inputs remain
    borrowed and unmodified; masked values are never wrapped in that type.
    All retained sites must share one grid per domain. The shared inner
    rectangle is also removed from each retained outer flux grid. Optional
    ``outer_domain`` and ``inner_domain`` names select OpenGHG domain definitions;
    unknown names use the first retained site as the coordinate reference.
    """
    if outer.sites != inner.sites:
        raise ValueError("Masking requires outer and inner merged data aligned to identical sites.")

    outer = _align_domain_sites(outer, domain=outer_domain, label="Outer")
    inner = _align_domain_sites(inner, domain=inner_domain, label="Inner")
    return _mask_outer_for_aligned_inner_domain(outer, inner)


def _mask_outer_for_aligned_inner_domain(
    outer: _NestedDomainData, inner: _NestedDomainData,
) -> _NestedDomainData:
    """Remove inner support from already aligned nested-domain inputs."""
    outer_grid = outer.site_data[outer.sites[0]]
    inner_grid = inner.site_data[inner.sites[0]]
    site_mask = rectangular_extent_mask(
        inner_grid, target_lat=outer_grid["lat"], target_lon=outer_grid["lon"],
    )
    site_data = {
        site: _mask_spatial_variables(outer.site_data[site], site_mask)
        for site in outer.sites
    }

    flux_entries = outer.flux_data
    if not isinstance(flux_entries, Mapping) or not flux_entries:
        raise ValueError("Outer nested-domain merged data requires source-labelled flux datasets.")
    masked_flux_entries: dict[str, Any] = {}
    for source, flux_dataset in flux_entries.items():
        if not isinstance(flux_dataset, xr.Dataset) or "flux" not in flux_dataset:
            raise TypeError(f"Outer flux entry {source!r} does not contain an xarray `flux` field.")
        flux = flux_dataset["flux"]
        if "lat" not in flux.coords or "lon" not in flux.coords:
            raise ValueError(f"Outer flux entry {source!r} is missing lat/lon coordinates.")
        mask = rectangular_extent_mask(inner_grid, target_lat=flux["lat"], target_lon=flux["lon"])
        masked_dataset = flux_dataset.copy(deep=False)
        masked_dataset["flux"] = remove_domain_overlap(flux, mask)
        masked_flux_entries[source] = masked_dataset

    return _NestedDomainData(
        site_data=site_data, flux_data=masked_flux_entries,
        site_options=outer.site_options,
    )


def _inner_domain_name(outer_domain: str, inner_domain: str) -> str:
    """Return the OpenGHG nested domain name without duplicating its prefix."""
    outer = str(outer_domain).strip()
    inner = str(inner_domain).strip()
    if not inner:
        raise ValueError("`inner_domain` must be a non-empty OpenGHG domain name or suffix.")
    return inner if inner == outer or inner.startswith(f"{outer}-") else f"{outer}-{inner}"


def inner_data_config(
    outer_data: DataConfig,
    *,
    inner_domain: str,
    inner_footprint_store: str | None = None,
    inner_flux_store: str | None = None,
    inner_flux_domain: str | None = None,
    inner_basis_algorithm: str | None = None,
    inner_nbasis: int | None = None,
    inner_basis_path: str | Path | None = None,
    inner_basis_partition_path: str | Path | None = None,
    inner_basis_partition_paths: Sequence[str | Path] | None = None,
    inner_landsea_path: str | Path | None = None,
    inner_outer_regions_path: str | Path | None = None,
    inner_basis_output_path: str | Path | None = None,
) -> DataConfig:
    """Resolve the native inner acquisition and basis choices from an outer request.

    Stores and the basis target inherit outer values when omitted. The inner
    basis defaults to quadtree, has independent resources and no boundary
    component. Relative resource paths resolve from the current directory.
    An omitted ``inner_nbasis`` still means automatic total-budget allocation
    when passed to :func:`prepare_nested_rhime_inputs`.

    Returns:
        A new configuration without mutating ``outer_data`` or opening files.
        Use :func:`~openghg_inversions.rhime.preparation.resolve_basis_inputs`
        to load its selected resources.
    """
    basis_options = resolve_resource_paths(
        {
            "domain": _inner_domain_name(outer_data.domain, inner_domain),
            "basis_path": inner_basis_path,
            "basis_partition_path": inner_basis_partition_path,
            "basis_partition_paths": inner_basis_partition_paths,
            "landsea_path": inner_landsea_path,
            "outer_regions_path": inner_outer_regions_path,
            "basis_output_path": inner_basis_output_path,
        },
        annotations=get_type_hints(DataConfig), base_dir=Path.cwd(),
    )
    return replace(
        outer_data, **basis_options,
        footprint_store=inner_footprint_store or outer_data.footprint_store,
        flux_store=inner_flux_store or outer_data.flux_store,
        flux_domain=inner_flux_domain,
        use_bc=False, bc_input=None, bc_basis_paths=None, fix_basis_outer_regions=False,
        basis_algorithm=inner_basis_algorithm if inner_basis_algorithm is not None else "quadtree",
        nbasis=outer_data.nbasis if inner_nbasis is None else inner_nbasis,
    )


def _prepare_one_domain(
    merged: _NestedDomainData, *,
    domain: str, start_date: str, flux_sources: tuple[str, ...],
    basis_algorithm: str | None, nbasis: int, fix_basis_outer_regions: bool,
    use_bc: bool, bc_freq: str | None, min_error: MinErrorConfig,
    min_error_options: Mapping[str, bool], basis_inputs: Mapping[str, Any],
    bc_basis: xr.Dataset | None = None, allow_empty_inner_region: bool = False,
) -> PreparedInputs:
    """Build one native basis and its labelled sensitivities from resolved choices."""
    runtime_flux = resolve_runtime_flux(merged.flux_data, split_by_sectors=False)
    basis_inputs = dict(basis_inputs)
    if (partition := basis_inputs.get("basis_partition")) is not None:
        if domain.upper() in get_domain_info():
            try:
                partition = align_lat_lon(partition, domain)
            except ValueError:
                # Partitions on a different native inventory grid use its labels.
                pass
        basis_inputs["basis_partition"] = force_align(partition, runtime_flux, dims=("lat", "lon"))
    with timed("rhime.prepare_inputs.basis_build", basis_algorithm=basis_algorithm):
        basis_functions = prepare_basis(
            site_data=merged.site_data, runtime_flux=runtime_flux,
            weighting_flux=merged.flux_data[flux_sources[0]]["flux"],
            domain=domain, start_date=start_date, basis_algorithm=basis_algorithm,
            nbasis=nbasis, fix_outer_regions=fix_basis_outer_regions,
            **basis_inputs, allow_empty_inner_region=allow_empty_inner_region,
        )
    site_data = build_sensitivities(
        merged.site_data, basis_functions, flux_sources=flux_sources, use_bc=use_bc,
        bc_basis=bc_basis, multisector=False,
    )
    return assemble_inputs(
        site_data, basis_functions, site_options=merged.site_options,
        domain=domain, start_date=start_date, bc_freq=bc_freq, min_error=min_error,
        min_error_options=min_error_options, use_bc=use_bc,
    )


def _nested_sensitivity_reductions(
    merged: MergedData | _NestedDomainData,
    *,
    label: str,
) -> list[xr.DataArray]:
    """Return lazy per-site absolute ``fp_x_flux`` reductions."""
    reductions: list[xr.DataArray] = []
    for site in merged.sites:
        dataset = merged.site_data[site]
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"{label} nested site {site!r} must be an xarray Dataset.")
        if "fp_x_flux" in dataset:
            sensitivity = dataset["fp_x_flux"]
        elif "fp_x_flux_sectoral" in dataset:
            sensitivity = dataset["fp_x_flux_sectoral"]
        else:
            raise ValueError(
                f"{label} nested site {site!r} requires `fp_x_flux` or `fp_x_flux_sectoral` "
                "for automatic basis-budget allocation. Set `inner_nbasis` explicitly to "
                "avoid automatic allocation."
            )
        reductions.append(abs(sensitivity).fillna(0.0).sum())
    if not reductions:
        raise ValueError(f"{label} nested data contains no retained sites for basis-budget allocation.")
    return reductions


def _allocate_nested_nbasis(
    outer: MergedData | _NestedDomainData,
    inner: MergedData | _NestedDomainData,
    *,
    total_nbasis: int,
) -> tuple[int, int]:
    """Materialize and split a total nested basis budget by sensitivity.

    The square-root sensitivity ratio damps differences between grids. The
    inner share is bounded to 35--60 percent, matching the final behavior on
    the legacy inner-domain branch while keeping execution explicit in the
    modern preparation stage.
    """
    if isinstance(total_nbasis, bool) or not isinstance(total_nbasis, Integral) or total_nbasis < 2:
        raise ValueError("Nested automatic basis allocation requires integer `nbasis >= 2`.")
    total_nbasis = int(total_nbasis)

    outer_reductions = _nested_sensitivity_reductions(outer, label="Outer")
    inner_reductions = _nested_sensitivity_reductions(inner, label="Inner")
    reductions = [*outer_reductions, *inner_reductions]
    materialization_start = timer_start()
    computed = dask_compute(*(to_dense(reduction).data for reduction in reductions))
    outer_total = float(sum(np.asarray(value).item() for value in computed[: len(outer_reductions)]))
    inner_total = float(sum(np.asarray(value).item() for value in computed[len(outer_reductions) :]))
    if not np.isfinite(outer_total) or not np.isfinite(inner_total):
        raise ValueError("Nested fp_x_flux sensitivity totals must be finite.")

    damped_outer = outer_total**0.5
    damped_inner = inner_total**0.5
    damped_total = damped_outer + damped_inner
    damped_inner_share = 0.5 if damped_total == 0.0 else damped_inner / damped_total
    inner_share = min(0.60, max(0.35, damped_inner_share))
    inner_nbasis = max(1, min(total_nbasis - 1, int(round(total_nbasis * inner_share))))
    outer_nbasis = total_nbasis - inner_nbasis
    log_timing(
        "rhime.nested_basis_budget_materialize",
        timer_seconds(materialization_start),
        total_nbasis=total_nbasis,
        outer_nbasis=outer_nbasis,
        inner_nbasis=inner_nbasis,
        outer_sensitivity=outer_total,
        inner_sensitivity=inner_total,
    )
    return outer_nbasis, inner_nbasis


def prepare_nested_rhime_inputs(
    outer: MergedData, inner: MergedData, *,
    domain: str, inner_domain: str, start_date: str, flux_sources: tuple[str, ...],
    averaging_error: bool, filters: Any, basis_algorithm: str | None,
    inner_basis_algorithm: str | None, nbasis: int, inner_nbasis: int | None,
    fix_basis_outer_regions: bool, use_bc: bool, bc_freq: str | None,
    min_error: MinErrorConfig, min_error_options: Mapping[str, bool],
    outer_basis_inputs: Mapping[str, Any], inner_basis_inputs: Mapping[str, Any],
    bc_basis: xr.Dataset | None = None,
    time_tolerance: str | pd.Timedelta | None = None,
) -> NestedRhimePreparedInputs:
    """Prepare two acquired native grids using explicit scientific choices.

    Each domain is shared across retained sites. Observation errors are derived
    before filtering; filtered outer observations determine the inner selection.
    Outer response and flux are masked before basis construction. Selected
    resources are supplied by the runner, which also owns optional basis exports.

    Args:
        outer: Borrowed acquired outer-domain observations, transport and flux.
        inner: Borrowed acquired inner-domain observations, transport and flux.
        domain: Outer transport-domain label.
        inner_domain: Complete inner transport-domain label.
        start_date: Requested start anchoring basis and boundary periods.
        flux_sources: Single source represented on both grids.
        averaging_error: Include variability when deriving observation errors.
        filters: Filters evaluated once on the outer observations.
        basis_algorithm: Outer basis-generation algorithm.
        inner_basis_algorithm: Inner basis-generation algorithm.
        nbasis: Outer target, or combined budget when both bases are generated
            and inner_nbasis is omitted.
        inner_nbasis: Independent inner target; omission enables automatic
            allocation when neither grid uses a saved basis.
        fix_basis_outer_regions: Retain the supplied fixed outer partition.
        use_bc: Include boundary sensitivities from the outer domain only.
        bc_freq: Outer boundary-condition scaling period.
        min_error: Minimum-error policy interpreted during assembly.
        min_error_options: Options for the minimum-error calculation.
        outer_basis_inputs: Selected outer operator or partition and support maps.
        inner_basis_inputs: Selected inner operator or partition and support maps.
        bc_basis: Loaded outer boundary-condition basis, when selected.
        time_tolerance: Explicit nearest-time tolerance; omission requires exact
            timestamps. Each inner footprint can match at most one outer row.

    Returns:
        Native preparations and observation-aligned combined sensitivities.
        Borrowed arrays remain unchanged and may remain Dask-backed. Automatic
        basis allocation and basis algorithms have explicit eager boundaries.

    Raises:
        ValueError: If more than one source is selected, observations cannot be
            aligned, support is missing or the automatic budget is invalid.
    """
    if len(flux_sources) != 1:
        raise ValueError("Nested RHIME preparation requires exactly one emissions source.")
    preparation_start = timer_start()
    outer_data = _NestedDomainData(outer.site_data, outer.flux_data, outer.site_options)
    inner_data = _NestedDomainData(inner.site_data, inner.flux_data, inner.site_options)
    outer_data = outer_data.with_site_data(
        prepare_observation_errors(outer_data.site_data, averaging_error=averaging_error),
        context="Observation error preparation",
    )
    inner_data = inner_data.with_site_data(
        prepare_observation_errors(inner_data.site_data, averaging_error=averaging_error),
        context="Observation error preparation",
    )
    outer_data, inner_data = _retain_common_sites(outer_data, inner_data)
    outer_site_data, _ = filter_observations(
        outer_data.site_data, outer_data.site_options, filters=filters,
    )
    outer_filtered = outer_data.with_site_data(outer_site_data, context="Observation filtering")
    outer_filtered, inner_data = _retain_common_sites(outer_filtered, inner_data)
    inner_filtered = align_inner_merged_to_outer_observations(
        outer_filtered, inner_data, time_tolerance=time_tolerance,
    )
    outer_filtered = _align_domain_sites(outer_filtered, domain=domain, label="Outer")
    inner_filtered = _align_domain_sites(inner_filtered, domain=inner_domain, label="Inner")
    masked_outer = _mask_outer_for_aligned_inner_domain(outer_filtered, inner_filtered)
    # Masking established one shared footprint grid; basis flux labels may differ.
    inner_grid = inner_filtered.site_data[inner_filtered.sites[0]]
    inner_lat = inner_grid.get_index("lat")
    inner_lon = inner_grid.get_index("lon")
    mask_extent = (
        float(inner_lat.min()), float(inner_lat.max()),
        float(inner_lon.min()), float(inner_lon.max()),
    )

    outer_nbasis = nbasis
    inner_target = nbasis if inner_nbasis is None else inner_nbasis
    if inner_nbasis is None and not any(
        inputs.get(name) is not None
        for inputs in (outer_basis_inputs, inner_basis_inputs)
        for name in ("basis_path", "basis_partition")
    ):
        outer_nbasis, inner_target = _allocate_nested_nbasis(
            masked_outer, inner_filtered, total_nbasis=nbasis,
        )

    outer_prepared = _prepare_one_domain(
        masked_outer, domain=domain, start_date=start_date, flux_sources=flux_sources,
        basis_algorithm=basis_algorithm, nbasis=outer_nbasis,
        fix_basis_outer_regions=fix_basis_outer_regions, use_bc=use_bc,
        bc_freq=bc_freq, min_error=min_error, min_error_options=min_error_options,
        basis_inputs=outer_basis_inputs, bc_basis=bc_basis, allow_empty_inner_region=True,
    )
    inner_prepared = _prepare_one_domain(
        inner_filtered, domain=inner_domain, start_date=start_date, flux_sources=flux_sources,
        basis_algorithm=inner_basis_algorithm, nbasis=inner_target,
        fix_basis_outer_regions=False, use_bc=False, bc_freq=bc_freq,
        min_error=min_error, min_error_options=min_error_options, basis_inputs=inner_basis_inputs,
    )
    prepared = combine_nested_rhime_inputs(
        outer_prepared, inner_prepared, time_tolerance=time_tolerance,
        inner_domain_label=inner_domain, outer_overlap_masked=True,
        outer_overlap_mask_policy=_RETRIEVAL_OVERLAP_MASK_POLICY,
        outer_overlap_mask_extent=mask_extent,
    )
    log_timing(
        "rhime.prepare_nested_inputs", timer_seconds(preparation_start),
        sites=len(prepared.outer.sites),
        outer_regions=prepared.combined.inv_inputs.sizes.get("region"),
        inner_regions=prepared.combined.inv_inputs.sizes.get("inner_region"),
    )
    return prepared


def nested_model_input_names(
    prepared: PreparedInputs,
    model_spec: RhimeModelSpec,
) -> tuple[str, ...]:
    """Declare arrays required by the selected nested-model components."""
    if "H_inner" not in prepared.inv_inputs:
        raise ValueError("Nested inner flux component requires prepared input 'H_inner'.")
    return (*standard_model_input_names(prepared, model_spec), "H_inner")


def materialize_nested_inputs(
    prepared: PreparedInputs,
    model_spec: RhimeModelSpec,
    *,
    inner_x_prior: PriorArgs | None = None,
) -> tuple[xr.Dataset, dict[str, PriorArgs]]:
    """Compute selected nested arrays and enabled priors in one Dask operation.

    Args:
        prepared: Borrowed combined inputs containing H and H_inner.
        model_spec: Resolved choices selecting enabled inputs and priors.
        inner_x_prior: Inner prior; omission reuses the outer prior parameters.
            Labelled vectors must use the inner state dimension and coordinates.

    Returns:
        A shallow Dataset with selected inputs eager and numerical prior mappings
        keyed by x_outer, x_inner and enabled baseline/likelihood variable names.
        Borrowed preparation and unselected lazy extensions remain unchanged.

    Raises:
        ValueError: A selected component input is absent or the model is multisector.
    """
    priors = _standard_prior_parameters(model_spec)
    outer = priors.pop("x")
    priors.update(x_outer=outer, x_inner=outer if inner_x_prior is None else inner_x_prior)
    return _materialize_pymc_inputs_and_priors(
        prepared, variable_names=nested_model_input_names(prepared, model_spec), prior_parameters=priors,
    )


def build_nested_rhime_model(
    model_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    priors: Mapping[str, PriorArgs],
    inner_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Build the two native-grid flux terms, shared baseline and likelihood.

    Args:
        model_inputs: Combined labelled inputs, with selected arrays eager.
            Unselected lazy extensions remain borrowed and unused.
        model_spec: Resolved scientific choices for one source on two grids.
        priors: Eager parameters keyed by x_outer, x_inner and enabled
            baseline/likelihood variable names. Materialize these together with
            inputs using :func:`materialize_nested_inputs`.
        inner_state_activity: Optional inner-state sensitivity/pruning policy.
        likelihood_builder: Optional ordinary custom likelihood callable.
        likelihood_kwargs: Options belonging to that callable.

    Returns:
        Concrete PyMC graph and its scientific output-role contract.

    Raises:
        ValueError: Selected inputs or priors are lazy, required priors are absent,
            or state dimensions coincide. Input availability and model choices
            belong to materialization and the runner respectively.
    """
    names = ["H", "H_inner", "mf", "mf_error"]
    if model_spec.likelihood is not None:
        names.extend(model_spec.likelihood.required_prepared_inputs)
    if model_spec.use_bc:
        names.append("H_bc")
    names.extend(aggregation_error_input_names(model_inputs, model_spec.aggregation_error_mode))
    required = (_standard_prior_parameters(model_spec).keys() - {"x"}) | {"x_outer", "x_inner"}
    missing = required - priors.keys()
    if missing:
        raise ValueError(f"Missing materialized prior parameter mappings: {sorted(missing)!r}.")
    lazy_priors, _ = unpack_collections(priors)
    if any(is_dask_collection(model_inputs[name]) for name in names) or lazy_priors:
        raise ValueError("Nested model inputs and prior parameters must be materialized before building.")
    outer_flux_sensitivity = model_inputs["H"]
    inner_flux_sensitivity = model_inputs["H_inner"]
    outer_state_dim = _state_dimension(outer_flux_sensitivity, label="Outer")
    inner_state_dim = _state_dimension(inner_flux_sensitivity, label="Inner")
    if outer_state_dim == inner_state_dim:
        raise ValueError("Nested sensitivities require distinct state-dimension names.")
    outer_flux_sensitivity, inner_flux_sensitivity = xr.align(
        outer_flux_sensitivity, inner_flux_sensitivity, join="exact", copy=False,
    )
    observations = model_inputs["mf"]
    observation_error = model_inputs["mf_error"]
    aggregation_error = resolve_aggregation_error(model_inputs, model_spec.aggregation_error_mode)
    minimum_error = model_inputs.get("min_error")
    boundary_sensitivity = model_inputs.get("H_bc")
    use_bc, add_offset = model_spec.use_bc, model_spec.add_offset
    offset_args = model_spec.offset_args
    likelihood_settings = model_spec.likelihood
    sector = model_spec.sectors[0]
    outer_state_activity = sector.state_activity if sector.state_activity is not None else model_spec.state_activity
    bc_state_activity = model_spec.bc_state_activity
    prepared_outer = prepare_linear_sensitivity(outer_flux_sensitivity)
    prepared_inner = prepare_linear_sensitivity(inner_flux_sensitivity)
    prepared_boundary = (
        prepare_linear_sensitivity(boundary_sensitivity)
        if use_bc and boundary_sensitivity is not None
        else None
    )
    resolved_outer_prior = dict(priors["x_outer"])
    resolved_inner_prior = dict(priors["x_inner"])
    resolved_bc_prior = dict(priors.get("bc", DEFAULT_BC_PRIOR))
    resolved_offset_prior = dict(priors.get("offset_latent", DEFAULT_OFFSET_PRIOR))

    with registered_model() as model:
        outer_component = add_linear_component(
            prepared_outer,
            data_name="hx_outer",
            prior_args=resolved_outer_prior,
            var_name="x_outer",
            output_name="mu_outer",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=outer_state_activity,
        )
        inner_component = add_linear_component(
            prepared_inner,
            data_name="hx_inner",
            prior_args=resolved_inner_prior,
            var_name="x_inner",
            output_name="mu_inner",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=inner_state_activity,
        )
        pollution_mean = pm.Deterministic(
            "mu",
            outer_component.output + inner_component.output,
            dims=("nmeasure",),
        )

        boundary_mean = None
        if use_bc:
            if boundary_sensitivity is None:
                raise ValueError(
                    "Nested baseline component requires `boundary_sensitivity` when `use_bc` is true."
                )
            assert prepared_boundary is not None
            boundary_mean = add_linear_component(
                prepared_boundary,
                data_name="hbc",
                prior_args=resolved_bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=bc_state_activity,
                coordinate_owner="bc",
            ).output

        offset = None
        if add_offset:
            offset = add_offset_component(
                observations,
                prior_args=resolved_offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **(offset_args or {}),
            )

        baseline_mean = boundary_mean
        if offset is not None:
            baseline_mean = offset if baseline_mean is None else baseline_mean + offset
        modelled_mean = pollution_mean if baseline_mean is None else pollution_mean + baseline_mean
        add_rhime_likelihood(
            settings=likelihood_settings,
            likelihood_builder=likelihood_builder,
            likelihood_kwargs=likelihood_kwargs,
            forward=ForwardModelTerms(
                total=modelled_mean,
                pollution=pollution_mean,
                baseline=baseline_mean,
            ),
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
            minimum_error=minimum_error,
            output_dim="nmeasure",
            prior_parameters=priors,
        )
    roles = {
        "observation": "mf",
        "observation_error": "mf_error",
        "concentration": "y",
        "model_error": "epsilon",
        "flux_scale:outer": "x_outer",
        "flux_contribution:outer": "mu_outer",
        "emissions_sensitivity:outer": "hx_outer",
        "flux_scale:inner": "x_inner",
        "flux_contribution:inner": "mu_inner",
        "emissions_sensitivity:inner": "hx_inner",
    }
    if "min_error" in model_inputs:
        roles["minimum_error"] = "min_error"
    if model_spec.use_bc:
        roles.update(
            {
                "boundary": "mu_bc",
                "baseline_scale": "bc",
                "baseline_sensitivity": "hbc",
            }
        )
    if model_spec.add_offset:
        roles["offset"] = "offset"
    result = RhimeModelBuildResult(
        model=model,
        output_contract=OutputContract(
            variable_roles=roles,
            supported_output_formats=("none", "paris"),
            metadata={
                "kind": "builtin_nested",
                "outer_state_dimension": _state_dimension(
                    model_inputs["H"],
                    label="Outer",
                ),
                "inner_state_dimension": _state_dimension(
                    model_inputs["H_inner"],
                    label="Inner",
                ),
            },
        ),
    )
    return result


def _write_nested_paris_outputs(output: NestedOutput, output_spec: RhimeOutputSpec) -> NestedOutput:
    """Write and retain native flux products and one shared concentration product."""
    from openghg_inversions.postprocessing.nested_paris_outputs import make_nested_paris_products
    from openghg_inversions.postprocessing.paris_options import NestedParisOptions
    from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs

    validate_nested_output_resources(output_spec)
    inner_label = output.inner.output_metadata["nested_domain"]["inner_domain"]
    params = dict(output_spec.paris_postprocessing_kwargs or {})
    if output_spec.output_path is not None:
        params.setdefault("country_file_cache_dir", Path(output_spec.output_path) / ".country_cache")
    options = NestedParisOptions.from_params(params)
    flux_outer, flux_inner, concentration = make_nested_paris_products(
        output.outer, output.inner, inner_domain_label=inner_label,
        country_file=output_spec.country_file,
        obs_avg_period=output.outer.run_metadata["averaging_period"][0] or "0h",
        inner_country_file=options.inner_country_file,
        time_point=options.time_point, template_version=options.template_version,
        flux_frequency=options.flux_frequency, report_mode=options.report_mode,
        inversion_grid=options.inversion_grid, country_selections=options.country_selections,
        country_file_cache_dir=options.country_file_cache_dir,
    )
    Path(output_spec.output_path).mkdir(parents=True, exist_ok=True)
    products = {"paris_flux": flux_outer, "paris_flux_inner": flux_inner, "paris_concentration": concentration}
    paths = {}
    for name, domain, suffix in (
        ("paris_flux", output.outer.domain, "flux"),
        ("paris_flux_inner", inner_label, "flux"),
        ("paris_concentration", output.outer.domain, "conc"),
    ):
        path = _define_derived_output_filename(
            output_spec, species=output.outer.species, domain=domain,
            output_name=f"{output_spec.output_name}_{suffix}",
            start_date=output.outer.run_metadata["start_date"], ext=".nc",
        )
        product = products[name]
        write_netcdf_preserving_bounds_attrs(
            product, path, unlimited_dims=["index" if "index" in product.dims else "time"],
        )
        paths[f"{name}_path"] = Path(path)
    return replace(output, products=products, paths=paths)


_NESTED_PARAMETER_NAMES = frozenset(
    {
        "inner_domain",
        "inner_footprint_store",
        "inner_flux_store",
        "inner_flux_domain",
        "inner_basis_algorithm",
        "inner_nbasis",
        "inner_basis_path",
        "inner_basis_partition_path",
        "inner_basis_partition_paths",
        "inner_landsea_path",
        "inner_outer_regions_path",
        "inner_basis_output_path",
        "inner_time_tolerance",
        "inner_x_prior",
    }
)


def run_nested(
    *,
    config_file: str | Path | None = None,
    config: RhimeConfig | None = None,
    prepared: NestedRhimePreparedInputs | None = None,
    inner_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> NestedOutput:
    """Run a standard one-source RHIME inversion on nested native grids.

    Nested options may be supplied directly or in the INI file. Ordinary
    options use the same modern vocabulary as :func:`run_standard`.
    ``output_format='none'`` returns the sampled result without formatted
    products and writes no result artifacts; ``output_format='paris'`` creates
    separate native-grid flux products and one shared concentration product.

    Args:
        config_file: Optional INI file; raw keyword options override its values.
        config: Resolved outer request, mutually exclusive with config_file and
            ordinary raw options. Inner preparation options may accompany it.
        prepared: Borrowed native-grid handoff; skips acquisition and preparation
            and is validated before the shared build/sample/output sequence.
        inner_state_activity: Optional policy for the inner state block.
        likelihood_builder: Optional ordinary custom likelihood callable.
        likelihood_kwargs: Options for that callable.
        **kwargs: Outer RHIME options and inner acquisition/basis/time selectors.
            inner_domain is required for fresh preparation. inner_x_prior defaults
            to the outer prior. Explicit values, including None, override file settings.

    Returns:
        Native scientific views borrowing the prepared arrays and one trace.
        PARIS writes requested products, two output artifacts and one shared trace
        according to the configured export settings before returning.

    Raises:
        ValueError: Configuration sources conflict, preparation options accompany
            supplied prepared inputs, or the model/output selection is unsupported.
        TypeError: The likelihood builder or sampler is invalid.
        FileNotFoundError: A selected resource does not exist.
    """
    if likelihood_builder is not None and not callable(likelihood_builder):
        raise TypeError(
            f"`likelihood_builder` must be callable or None; got {type(likelihood_builder).__name__}."
        )
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    if config is not None and config_file is not None:
        raise ValueError("Pass either resolved `config` or `config_file`/run parameters, not both.")
    params = read_rhime_ini(config_file) if config_file is not None else {}
    params.update(kwargs)
    nested_options = {name: params.pop(name) for name in tuple(params) if name in _NESTED_PARAMETER_NAMES}
    if config is not None and params:
        raise ValueError("Pass either resolved `config` or `config_file`/run parameters, not both.")
    inner_domain = nested_options.pop("inner_domain", None)
    inner_x_prior = nested_options.pop("inner_x_prior", None)
    if inner_x_prior is not None and not isinstance(inner_x_prior, Mapping):
        raise ValueError("`inner_x_prior` must be a prior mapping/dict or None.")
    time_tolerance = nested_options.pop("inner_time_tolerance", None)
    inner_basis_output_path = resolve_resource_paths(
        {"basis_output_path": nested_options.pop("inner_basis_output_path", None)},
        annotations=get_type_hints(DataConfig), base_dir=Path.cwd(),
    )["basis_output_path"]

    if config is None:
        config = RhimeConfig.from_params(params, multisector=False)
    if config.data.split_by_sectors or len(config.model.sectors) != 1:
        raise ValueError("Nested RHIME requires a single-source configuration.")
    if likelihood_builder is not None and config.model.likelihood is not None:
        raise ValueError("A custom likelihood cannot be combined with a built-in mismatch model.")
    sampler = resolve_sampler(config.sampler)
    if likelihood_builder is None and config.model.likelihood is None:
        raise ValueError("A nested RHIME run requires a built-in or custom likelihood.")
    validate_nested_output_resources(config.output)
    if prepared is None:
        if inner_domain is None:
            raise ValueError("Nested preparation requires `inner_domain`.")
        outer_data = config.data
        inner_data = inner_data_config(
            outer_data, inner_domain=inner_domain, inner_basis_output_path=inner_basis_output_path,
            **nested_options,
        )
        outer_basis_inputs = resolve_basis_inputs(outer_data)
        inner_basis_inputs = resolve_basis_inputs(inner_data)
        bc_basis = (
            load_bc_basis(outer_data.bc_basis_paths)
            if outer_data.use_bc and outer_data.bc_basis_paths is not None else None
        )
        outer_merged = acquire_merged_data(outer_data)
        inner_merged = acquire_merged_data(inner_data)
        prepared = prepare_nested_rhime_inputs(
            outer_merged, inner_merged,
            **outer_data.select(
                "domain", "start_date", "flux_sources", "averaging_error", "filters",
                "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
                "min_error", "min_error_options",
            ),
            inner_domain=inner_data.domain, inner_basis_algorithm=inner_data.basis_algorithm,
            inner_nbasis=nested_options.get("inner_nbasis"), outer_basis_inputs=outer_basis_inputs,
            inner_basis_inputs=inner_basis_inputs, bc_basis=bc_basis, time_tolerance=time_tolerance,
        )
    else:
        if inner_domain is not None or nested_options or time_tolerance is not None:
            raise ValueError("Supplied prepared inputs cannot be combined with preparation options.")
        prepared = prepared.validated()

    outer_basis = export_basis(prepared.outer.basis_functions, config.data.basis_output_path)
    inner_basis = export_basis(prepared.inner.basis_functions, inner_basis_output_path)
    if outer_basis is not prepared.outer.basis_functions:
        prepared = replace(
            prepared,
            outer=replace(prepared.outer, basis_functions=outer_basis),
            combined=replace(prepared.combined, basis_functions=outer_basis),
        )
    if inner_basis is not prepared.inner.basis_functions:
        prepared = replace(prepared, inner=replace(prepared.inner, basis_functions=inner_basis))
    model_inputs, priors = materialize_nested_inputs(
        prepared.combined, config.model, inner_x_prior=inner_x_prior,
    )
    built = build_nested_rhime_model(
        model_inputs, config.model, priors=priors, inner_state_activity=inner_state_activity,
        likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
    )
    built.output_contract.validate_requested_output(config.output.output_format)
    posterior = sample_rhime_model(built, sampler)
    annotate_sample_trace(
        built, posterior, likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
    )
    output = make_nested_inversion_outputs(
        prepared=prepared, model_spec=config.model, posterior=posterior,
        output_contract=built.output_contract, start_date=config.data.start_date,
        end_date=config.data.end_date, actual_inference=sampler.describe(),
        provenance=installed_ogi_provenance(),
        inner_state_activity=inner_state_activity,
    )
    if config.output.output_format == "paris":
        output = _write_nested_paris_outputs(output, config.output)
        paths = save_inversion_output(output.outer, config.output)
        paths.update(save_inversion_output(
            output.inner, replace(config.output, save_trace=False), suffix="_inner",
        ))
        output = replace(
            output, paths={**output.paths, **paths},
        )
    return output
