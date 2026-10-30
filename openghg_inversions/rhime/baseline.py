"""Supplied deterministic backgrounds and explicit inner-domain preparation."""

from __future__ import annotations

from dataclasses import replace
import json

from dask import is_dask_collection
import numpy as np
import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.models.coords import add_model_data
from ._domain_support import remove_domain_overlap

BASELINE_INPUT_NAMES = ("supplied_baseline", "inner_fixed_contribution")


def _concentration(array: xr.DataArray, observations: xr.DataArray, *, name: str) -> xr.DataArray:
    """Select exact retained releases and convert concentration units lazily."""
    if array.dims != observations.dims or array.dims != ("nmeasure",):
        raise ValueError(f"{name} must have only the nmeasure dimension.")
    index = array.get_index("nmeasure")
    if not index.is_unique:
        raise ValueError(f"{name} has duplicate observation labels.")
    # Label selection permits a superset, but never interpolation or missing releases.
    try:
        selected = array.sel(nmeasure=observations.nmeasure)
    except KeyError as exc:
        raise ValueError(f"{name} is missing retained observations.") from exc
    selected, _ = xr.align(selected, observations, join="exact", copy=False)
    source_units = array.attrs.get("units")
    target_units = observations.attrs.get("units")
    if not source_units or not target_units:
        raise ValueError(f"{name} and observations require explicit concentration units.")
    scale = mole_fraction_unit_scale(source_units, context=name) / mole_fraction_unit_scale(
        target_units, context="observations",
    )
    return (selected * scale).rename(name).assign_attrs({**array.attrs, "units": target_units})


def prepare_supplied_baseline(
    prepared: PreparedInputs,
    baseline: xr.DataArray | xr.Dataset,
    *,
    native_response: xr.DataArray,
    inner_support: xr.DataArray,
    inner_fixed_contribution: xr.DataArray,
    baseline_site_metadata: xr.Dataset,
    species: str,
    provenance: str,
    constituents: tuple[str, ...],
    sampled_boundary_is_correction: bool = False,
) -> PreparedInputs:
    """Rebuild an inner-only operator and attach an absolute supplied background.

    This bounded single-sector preparation implements
    ``baseline + inner_fixed_contribution + H_inner @ x``. ``native_response``
    is footprint times prior flux, on the retained basis grid and observation
    MultiIndex. ``inner_support`` is an eager Boolean cell mask on that exact
    grid: whole cells are included or excluded, with no interpolation or partial
    overlap. Mask before basis reduction; states may span the domain edge.
    Native flux outputs are also zero outside this support.

    ``constituents`` must include ``boundary`` and ``exterior_flux`` and may
    name other supplied background constituents, but must exclude
    ``inner_fixed_contribution`` and ``offset``. The inner affine contribution
    is required explicitly (supply a labelled zero series when absent).
    Baseline uncertainty is omitted. Exterior prior dependence is discarded;
    callers must supply an inner prior and any aggregation covariance appropriate
    to that deterministic exterior treatment. Neither is silently conditioned.

    A Dataset may provide each declared constituent as a separate series; those
    series are converted, summed once and retained alongside the total.
    Baseline and affine series may contain additional observation labels. All
    retained releases must exist; missing/nonfinite values are rejected at the
    PyMC materialization boundary. Site/inlet and averaging metadata must agree
    with the prepared observation metadata. Species and provenance are explicit.
    An enabled sampled boundary must be declared a correction to this background.
    Inputs remain borrowed and native numerical payloads remain lazy.
    """
    if "supplied_baseline" in prepared.inv_inputs:
        raise ValueError("A supplied baseline is already installed; prepare from the original inputs.")
    if not species.strip() or not provenance.strip():
        raise ValueError("Supplied baseline species and provenance must be nonempty.")
    if not {"boundary", "exterior_flux"}.issubset(constituents) or set(constituents) & {
        "inner_fixed_contribution", "offset",
    } or len(set(constituents)) != len(constituents):
        raise ValueError("Baseline constituents must include boundary/exterior_flux and exclude inner fixed/offset.")
    if baseline.attrs.get("species", "").lower() != species.lower():
        raise ValueError("Supplied baseline species must match the declared species.")
    sites = list(prepared.sites)
    try:
        metadata = baseline_site_metadata.sel(site=sites)
    except KeyError as exc:
        raise ValueError("Baseline metadata is missing retained sites.") from exc
    if not metadata.equals(prepared.site_metadata):
        raise ValueError("Baseline site/inlet/averaging metadata must match retained observations.")
    grid_dims = prepared.basis_functions.operator.meta.grid_dims
    if set(inner_support.dims) != set(grid_dims) or inner_support.dtype.kind != "b":
        raise ValueError("Inner support must be a Boolean mask on exactly the native grid dimensions.")
    if is_dask_collection(inner_support):
        raise ValueError("Materialize the inner-domain support mask explicitly before preparation.")
    if not bool(inner_support.any()):
        raise ValueError("Inner support must contain at least one cell.")
    if set(native_response.dims) != {*grid_dims, "nmeasure"}:
        raise ValueError("Native response must have only native grid and nmeasure dimensions (single sector).")
    if prepared.basis_functions.source_labels is not None:
        raise ValueError("Supplied inner-domain preparation currently requires a single-sector basis.")
    observations = prepared.inv_inputs.mf
    native_response, _ = xr.align(native_response, observations, join="exact", copy=False)
    masked_response = remove_domain_overlap(native_response, ~inner_support)
    masked_flux = remove_domain_overlap(prepared.basis_functions.flux, ~inner_support)
    basis = replace(prepared.basis_functions, flux=masked_flux)
    matrix, masked_response = xr.align(basis.operator.basis_matrix, masked_response, join="exact", copy=False)
    sensitivity = xr.dot(masked_response, matrix, dim=list(grid_dims)).rename("H")
    sensitivity = _response_units(sensitivity, native_response, observations)
    fixed = _concentration(inner_fixed_contribution, observations, name="inner_fixed_contribution")
    components = {}
    if isinstance(baseline, xr.Dataset):
        if set(baseline.data_vars) != set(constituents):
            raise ValueError("Baseline Dataset variables must match the declared constituents.")
        components = {
            f"supplied_baseline_{name}": _concentration(value, observations, name=f"supplied_baseline_{name}")
            for name, value in baseline.data_vars.items()
        }
        supplied = sum(components.values()).assign_attrs(units=observations.attrs["units"])
    else:
        supplied = _concentration(baseline, observations, name="supplied_baseline")
    supplied = supplied.assign_attrs(
        species=species.lower(), provenance=provenance,
        constituents=json.dumps(list(constituents)), uncertainty="omitted",
        sampled_boundary_is_correction=int(sampled_boundary_is_correction),
        domain_policy="exact-grid-whole-cell-mask",
    )
    inputs = prepared.inv_inputs.assign(
        H=sensitivity, supplied_baseline=supplied, inner_fixed_contribution=fixed,
        inner_domain_support=inner_support, **components,
    )
    return PreparedInputs(inputs, basis, prepared.site_metadata)


def _response_units(sensitivity: xr.DataArray, native: xr.DataArray, observations: xr.DataArray) -> xr.DataArray:
    """Convert projected response units without reducing or materializing it."""
    source, target = native.attrs.get("units"), observations.attrs.get("units")
    if not source or not target:
        raise ValueError("Native response and observations require concentration units.")
    factor = mole_fraction_unit_scale(source, context="native response") / mole_fraction_unit_scale(
        target, context="observations",
    )
    return (sensitivity * factor).assign_attrs(units=target)


def supplied_baseline_input_names(inputs: xr.Dataset, *, species: str, use_bc: bool) -> tuple[str, ...]:
    """Validate the labelled supplied-background contract without computing data."""
    present = set(BASELINE_INPUT_NAMES) & set(inputs)
    if not present:
        return ()
    if present != set(BASELINE_INPUT_NAMES) or "inner_domain_support" not in inputs:
        raise ValueError("Use prepare_supplied_baseline to install the complete inner-domain baseline contract.")
    baseline = inputs.supplied_baseline
    try:
        constituents = json.loads(baseline.attrs.get("constituents", "null"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Supplied baseline constituents must be a JSON list.") from exc
    if (
        not isinstance(constituents, list)
        or not all(isinstance(name, str) and name.strip() for name in constituents)
        or not {"boundary", "exterior_flux"}.issubset(constituents)
        or {"offset", "inner_fixed_contribution"}.intersection(constituents)
        or len(set(constituents)) != len(constituents)
        or not baseline.attrs.get("provenance")
        or baseline.attrs.get("uncertainty") != "omitted"
        or baseline.attrs.get("domain_policy") != "exact-grid-whole-cell-mask"
    ):
        raise ValueError("Incomplete supplied baseline accounting; use prepare_supplied_baseline.")
    if baseline.attrs.get("species", "").lower() != species.lower():
        raise ValueError("Supplied baseline species does not match the model.")
    if use_bc and not baseline.attrs.get("sampled_boundary_is_correction"):
        raise ValueError("Sampled boundary would double-count the supplied background; explicitly select a correction.")
    for name in BASELINE_INPUT_NAMES:
        array = inputs[name]
        if array.dims != ("nmeasure",) or array.attrs.get("units") != inputs.mf.attrs.get("units"):
            raise ValueError(f"{name} must match observation dimensions and units.")
        xr.align(array, inputs.mf, join="exact", copy=False)
    return BASELINE_INPUT_NAMES


def add_baseline_component(
    inputs: xr.Dataset, sampled_boundary: TensorVariable | None,
) -> TensorVariable | None:
    """Compose supplied and sampled baseline once; keep the offset separate.

    Called at the eager PyMC boundary. The returned effective expression is
    shared by every likelihood and stored for prior/posterior reconstruction.
    """
    if "supplied_baseline" not in inputs:
        return sampled_boundary
    for name in BASELINE_INPUT_NAMES:
        if not np.isfinite(inputs[name].to_numpy()).all():
            raise ValueError(f"{name} must be finite at every retained observation.")
    supplied = add_model_data(inputs.supplied_baseline)
    effective = supplied if sampled_boundary is None else supplied + sampled_boundary
    return pm.Deterministic("mu_baseline", effective, dims="nmeasure")
