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

    This independent-state preparation implements
    ``baseline + inner_fixed_contribution + H_inner @ x``. Masking occurs before
    basis reduction; states may span the domain edge. Native flux outputs are
    also zero outside the support. Observation offsets remain separate from the
    supplied background and any sampled boundary correction. Existing coherent
    reductions instead use ``prepare_co2_supplied_baseline`` to select their
    effective operator verbatim.

    Baseline uncertainty is omitted. Exterior prior dependence is discarded;
    callers must supply an inner prior and any aggregation covariance appropriate
    to that deterministic exterior treatment. Neither is silently conditioned.

    Missing retained observation labels are rejected during preparation.
    Nonfinite total background or inner affine values are rejected during model
    construction at the eager PyMC boundary. Inputs remain borrowed and native
    numerical payloads remain lazy.

    Args:
        prepared: Original independent-state preparation without an installed supplied
            baseline, retaining the basis grid, prior flux and observation metadata.
        baseline: Absolute background total, or a Dataset with one variable per
            declared constituent. Its ``species`` attribute must match ``species``.
            Each series must have only ``nmeasure``, unique observation labels covering
            all retained releases, and explicit concentration units convertible to
            observation units. Additional releases are selected away; Dataset series
            are converted, summed once and retained alongside the total.
        native_response: Footprint times prior flux with exactly the native grid and
            ``nmeasure`` dimensions. Indexed grid coordinates must match the retained
            basis; the observation index must match the retained releases exactly.
            Source-aware responses also require an indexed ``source`` dimension
            matching the retained source labels; sector priors remain independent.
            Explicit concentration units must be convertible to observation units.
        inner_support: Eager Boolean whole-cell mask on exactly the indexed native
            grid, retaining at least one cell. No interpolation or partial overlap
            is supported.
        inner_fixed_contribution: Complete inner-only affine concentration with the
            same label and unit requirements as ``baseline``. Supply a labelled zero
            series when absent; exclude it from the supplied background.
        baseline_site_metadata: Site, inlet and averaging metadata equal to the
            preparation's metadata after selection of retained sites.
        species: Nonempty species name matching the background and intended model.
        provenance: Nonempty description of the background product and processing.
        constituents: Unique names including ``boundary`` and ``exterior_flux``,
            excluding ``inner_fixed_contribution`` and ``offset``. Dataset variable
            names must match these names exactly.
        sampled_boundary_is_correction: Declare any enabled sampled boundary to be
            a correction to the supplied background. Defaults to false; model input
            validation rejects boundary sampling unless this is true. Multisector
            supplied backgrounds replace all sampled boundary operands and reject
            corrections; observation offsets must also be disabled.

    Returns:
        A new preparation with rebuilt inner sensitivity, masked native prior flux
        carrying ``flux_support``, and separately retained background, inner affine
        term, support mask and any named background constituents. The input preparation
        is unchanged.

    Raises:
        ValueError: If a baseline is already installed, the preparation is coherently
            reduced, the support or response dimensions/indexes are invalid, concentration
            units are missing or incompatible, retained observations or site metadata
            do not match, or species, provenance or constituent accounting is invalid.
    """
    if {"alpha_prior_mean", "alpha_prior_covariance", "fixed_prior_contribution"}.issubset(prepared.inv_inputs):
        raise ValueError("Coherent preparations require prepare_co2_supplied_baseline; do not rebuild their H.")
    supplied, fixed, components = _prepare_baseline_series(
        prepared, baseline, inner_fixed_contribution=inner_fixed_contribution,
        baseline_site_metadata=baseline_site_metadata, species=species,
        provenance=provenance, constituents=constituents,
        sampled_boundary_is_correction=sampled_boundary_is_correction,
    )
    grid_dims = prepared.basis_functions.operator.meta.grid_dims
    if set(inner_support.dims) != set(grid_dims) or inner_support.dtype.kind != "b":
        raise ValueError("Inner support must be a Boolean mask on exactly the native grid dimensions.")
    if is_dask_collection(inner_support):
        raise ValueError("Materialize the inner-domain support mask explicitly before preparation.")
    if not bool(inner_support.any()):
        raise ValueError("Inner support must contain at least one cell.")
    sources = prepared.basis_functions.source_labels
    expected_dims = {*grid_dims, "nmeasure"} | ({"source"} if sources is not None else set())
    if set(native_response.dims) != expected_dims:
        raise ValueError("Native response must have exactly native grid, nmeasure and any retained source dimensions.")
    if sources is not None:
        index = native_response.indexes.get("source")
        if index is None or not index.is_unique or set(index) != set(sources):
            raise ValueError("Native response source labels must match the retained basis exactly.")
        native_response = native_response.sel(source=list(sources))
    for name, array in (("Inner support", inner_support), ("Native response", native_response)):
        for dim in grid_dims:
            if dim not in array.indexes:
                raise ValueError(f"{name} requires an indexed coordinate for native grid dimension {dim!r}.")
    if "nmeasure" not in native_response.indexes:
        raise ValueError("Native response requires indexed observation labels on nmeasure.")
    observations = prepared.inv_inputs.mf
    native_response, _ = xr.align(native_response, observations, join="exact", copy=False)
    masked_response = remove_domain_overlap(native_response, ~inner_support)
    masked_flux = remove_domain_overlap(prepared.basis_functions.flux, ~inner_support).assign_coords(
        flux_support=inner_support,
    )
    basis = replace(prepared.basis_functions, flux=masked_flux)
    matrix = basis.operator.basis_matrix
    if sources is not None and "source" in matrix.coords and "source" in prepared.inv_inputs.H.dims:
        per_source = []
        for source in sources:
            source_matrix = basis.operator.operator_for_source(
                source, state_dim=basis.operator.meta.state_dim,
            ).basis_matrix
            response, source_matrix = xr.align(
                masked_response.sel(source=source, drop=True), source_matrix, join="exact", copy=False,
            )
            per_source.append(xr.dot(response, source_matrix, dim=list(grid_dims)))
        sensitivity = xr.concat(per_source, dim=xr.IndexVariable("source", list(sources))).rename("H")
    else:
        if sources is not None and "source" not in matrix.dims and "source" in matrix.coords:
            # Pair each gathered state with its source before the spatial contraction.
            state_dim = basis.operator.meta.state_dim
            masked_response = masked_response.sel(source=xr.DataArray(matrix.source.to_numpy(), dims=state_dim))
            masked_response = masked_response.drop_vars("source").assign_coords(
                xr.Coordinates.from_pandas_multiindex(matrix.get_index(state_dim), state_dim),
            )
        matrix, masked_response = xr.align(matrix, masked_response, join="exact", copy=False)
        sensitivity = xr.dot(masked_response, matrix, dim=list(grid_dims)).rename("H")
    sensitivity = _response_units(sensitivity, native_response, observations)
    inputs = prepared.inv_inputs.assign(
        H=sensitivity, supplied_baseline=supplied, inner_fixed_contribution=fixed,
        inner_domain_support=inner_support, **components,
    )
    if sources is not None:
        if sampled_boundary_is_correction:
            raise ValueError("Supplied multisector baselines replace sampled boundaries; corrections are unsupported.")
        inputs = inputs.drop_vars(("H_bc", "H_bc_anomaly", "H_bc_mass", "bc_reference"), errors="ignore")
        if "bc_region" in inputs.dims:
            inputs = inputs.drop_dims("bc_region")
    return PreparedInputs(inputs, basis, prepared.site_metadata)


def _prepare_baseline_series(
    prepared: PreparedInputs,
    baseline: xr.DataArray | xr.Dataset,
    *,
    inner_fixed_contribution: xr.DataArray,
    baseline_site_metadata: xr.Dataset,
    species: str,
    provenance: str,
    constituents: tuple[str, ...],
    sampled_boundary_is_correction: bool = False,
) -> tuple[xr.DataArray, xr.DataArray, dict[str, xr.DataArray]]:
    """Align absolute-background accounting shared by the preparation recipes."""
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
    observations = prepared.inv_inputs.mf
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
    return supplied, fixed, components


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
