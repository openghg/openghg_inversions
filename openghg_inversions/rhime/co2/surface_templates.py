"""Explicit retained-coefficient recipe for maintained native surface templates."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import asdict, replace
import json
from typing import Any

from dask import compute
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.basis._helpers import _project_centered_boundaries
from openghg_inversions.inversion_data.surface import SURFACE_POLICY_REVISION
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_data import MergedData
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.observation_error import AggregationError

from .co2_preparation import requested_co2_facts
from .co2_o2_preparation import Co2O2PreparedInputs, _stack, prepare_co2_o2_inputs


def _ppm(array: xr.DataArray, *, context: str, inherited_units: str | None = None) -> xr.DataArray:
    units = array.attrs.get("units", inherited_units)
    if not isinstance(units, str) or not units.strip():
        raise ValueError(f"{context} requires explicit concentration units.")
    factor = mole_fraction_unit_scale(units, context=context) / 1e-6
    return (array if factor == 1 else array * factor).assign_attrs(array.attrs, units="ppm")


def _materialize_surface_references(
    channels: Mapping[str, MergedData],
) -> dict[str, dict[str, xr.DataArray]]:
    """Resolve only the related scalar references, before borrowing transport."""
    references = {}
    for channel, data in channels.items():
        for site in data.sites:
            reference = data.site_data[site].bc_reference
            if reference.ndim:
                raise ValueError(f"{channel} requires one agreed scalar bc_reference across its sites.")
            _ppm(reference, context=f"{channel}/{site}/bc_reference")
            references[channel, site] = reference
    # Small metadata boundary: share scalar graphs across both channels and sites.
    resolved = compute(references)[0]
    return {channel: {site: resolved[channel, site] for site in data.sites}
            for channel, data in channels.items()}


def _channel_templates(
    data: MergedData, channel: str, mapping: Mapping[str, Hashable], prior: CorrelatedLognormalPrior,
    references: Mapping[str, xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.Dataset, dict[str, Any]]:
    facts = data.acquisition
    if facts.species != channel or not all((facts.domain, facts.start_date, facts.end_date)):
        raise ValueError(f"{channel} requires explicit species/domain/window acquisition facts.")
    if not data.split_by_sectors or not data.sites or any(value is None for value in data.site_options.inlet):
        raise ValueError(f"{channel} requires source-resolved data and explicit site/inlet selectors.")
    if set(mapping) != set(data.flux_data):
        raise ValueError(f"{channel} source_state must map exactly the acquired flux sources.")
    mean = prior.mean
    state_dim = prior.state_dim
    labels = mean.indexes[state_dim]
    if any(label not in labels for label in mapping.values()):
        raise ValueError(f"{channel} source_state contains unknown retained states.")
    required = [label for label, scope in zip(labels, mean.tracer_scope.values, strict=True)
                if str(scope).lower() in {"shared", channel}]
    if set(mapping.values()) != set(required):
        raise ValueError(f"{channel} source_state must cover exactly its shared and native ocean states.")
    observations, errors, sensitivities, boundaries = [], [], [], []
    records = {}
    reference = None
    reference_value = None
    offset = 0
    dim = f"{channel}_observation"
    for site, inlet in zip(data.sites, data.site_options.inlet, strict=True):
        ds = data.site_data[site]
        if ds.attrs.get("surface_policy_revision") != SURFACE_POLICY_REVISION:
            raise ValueError(f"{channel}/{site} lacks supported maintained surface preparation provenance.")
        selected = json.loads(ds.attrs["surface_transport_provenance"])
        if set(selected.get("sources", {})) != set(mapping):
            raise ValueError(f"{channel}/{site} selected source provenance does not match source_state.")
        for name, value in (("species", channel), ("site", site), ("inlet", inlet)):
            if name in ds.attrs and ds.attrs[name] != value:
                raise ValueError(f"{channel}/{site} conflicts with acquisition {name}.")
        if ds.mf.dims != ("time",):
            raise ValueError(f"{channel}/{site} mf requires only the native time axis.")
        if "mf_error" not in ds:
            raise ValueError(f"{channel}/{site} requires independently supplied mf_error.")
        times = ds.indexes.get("time")
        if not isinstance(times, pd.DatetimeIndex) or not times.is_unique or times.empty:
            raise ValueError(f"{channel}/{site} requires unique native observation timestamps.")
        if times.min() < pd.Timestamp(facts.start_date) or times.max() >= pd.Timestamp(facts.end_date):
            raise ValueError(f"{channel}/{site} native rows fall outside the declared acquisition window.")
        row_labels = np.arange(offset, offset + len(times))
        offset += len(times)

        def native(array: xr.DataArray) -> xr.DataArray:
            return array.rename(time=dim).assign_coords(
                {dim: row_labels, "time": (dim, times.values), "site": (dim, [site] * len(times))}
            )

        obs = _ppm(ds.mf, context=f"{channel}/{site}/mf")
        observations.append(native(obs))
        if ds.mf_error.dims != ("time",):
            raise ValueError(f"{channel}/{site} mf_error requires only the native time axis.")
        errors.append(native(_ppm(ds.mf_error, context=f"{channel}/{site}/mf_error",
                                  inherited_units=ds.mf.attrs.get("units"))))
        response = _ppm(ds.fp_x_flux_sectoral, context=f"{channel}/{site}/templates")
        if set(response.indexes["source"]) != set(mapping):
            raise ValueError(f"{channel}/{site} response source labels do not match source_state.")
        response = response.sum([axis for axis in response.dims if axis not in {"source", "time"}], skipna=False)
        columns = []
        for label in labels:
            sources = [source for source, target in mapping.items() if target == label]
            column = response.sel(source=sources).sum("source", skipna=False) if sources else xr.zeros_like(obs)
            columns.append(column.drop_vars("source", errors="ignore"))
        sensitivity = xr.concat(columns, dim=mean[state_dim]).transpose("time", state_dim)
        sensitivities.append(native(sensitivity))
        if ds.attrs.get("boundary_representation") != "centered-v1":
            raise ValueError(f"{channel}/{site} requires native centered endpoint operands.")
        current_reference = _ppm(references[site], context=f"{channel}/{site}/bc_reference")
        current_value = current_reference.item()
        if not np.isfinite(current_value) or (reference_value is not None and current_value != reference_value):
            raise ValueError(f"{channel} requires one agreed scalar bc_reference across its sites.")
        reference = current_reference
        reference_value = current_value
        normalized = ds.assign(
            bc_reference=current_reference,
            **{f"bc_anomaly_{direction}": _ppm(
                ds[f"bc_anomaly_{direction}"], context=f"{channel}/{site}/bc_anomaly_{direction}",
            ) for direction in "nesw"},
        )
        projected = {name: native(value) for name, value in
                     _project_centered_boundaries(normalized, basis=None).items()}
        boundaries.append(xr.Dataset(projected))
        records[site] = {"inlet": inlet, "input_units": ds.mf.attrs.get("units"),
                         "bc_reference": references[site].item(), "bc_reference_units": ds.bc_reference.attrs.get("units"),
                         "selected_surface_inputs": selected,
                         "input_variable_units": {name: array.attrs.get("units") for name, array in ds.data_vars.items()}}
        temporal = {name: ds.attrs[name] for name in (
            "boundary_temporal_policy", "boundary_temporal_policy_revision",
            "boundary_temporal_nearest_tie", "boundary_input_semantics",
            "boundary_temporal_max_gap", "boundary_temporal_bounds", "boundary_retrieval_padding",
        ) if name in ds.attrs}
        if temporal:
            records[site]["boundary_temporal_provenance"] = temporal
    boundary = xr.concat(boundaries, dim=dim, join="exact").rename({dim: "nmeasure"})
    boundary["bc_reference"] = reference
    boundary.attrs["boundary_representation"] = "centered-v1"
    return (xr.concat(observations, dim=dim), xr.concat(errors, dim=dim),
            xr.concat(sensitivities, dim=dim), boundary,
            {"acquisition": asdict(facts), "selected_inputs": asdict(data.provenance), "sites": records})


def prepare_linked_surface_templates(
    co2: MergedData,
    o2: MergedData,
    *,
    retained_prior: CorrelatedLognormalPrior,
    source_state: Mapping[str, Mapping[str, Hashable]],
    aggregation_error: AggregationError,
    provenance: Mapping[str, Any],
) -> Co2O2PreparedInputs:
    """Map acquired source templates explicitly onto retained coefficients.

    This declared template model has zero affine intercept. It does not regenerate
    a native coherent reduction or estimate unresolved uncertainty. Signed O2
    templates are used once, without a scalar oxidative multiplier. Multiple
    sources may explicitly share a coefficient; every active coefficient must
    be declared. Concentrations are converted to physical ppm with Pint.
    This adapter projects the four native NESW curtains; it has no spatial
    boundary-basis argument for twelve-region projection.

    Native rows follow SiteOptions order then each site's original time order,
    numbered from zero separately for each tracer, retaining site/time coordinates.
    Supply dense aggregation covariance on the gathered (species,
    channel_observation) MultiIndex in CO2-then-O2 order; column level names
    have suffix ``_cov``. Both observation_units coordinates must already be ppm.
    Declared boundary temporal-policy attributes are retained per channel/site
    beside selected input identities; absent attributes are not inferred.
    Scalar boundary references across both channels/sites are resolved together
    at a small metadata boundary. The covariance validation boundary may compute
    covariance and the opposite ocean columns; other borrowed numerical payloads retain their lazy graphs.

    Args:
        co2: Maintained source-resolved CO2 surface inputs with explicit
            acquisition facts, native observation errors and centered endpoint
            operands. Sites must share one scalar boundary reference.
        o2: Independently prepared O2 inputs with the same declared domain and
            acquisition window. Native sites and observation rows may differ.
        retained_prior: Positive retained-state prior with labelled state,
            ``source`` and ``tracer_scope`` coordinates identifying sources
            and shared or native states.
        source_state: Per-channel mapping of every acquired flux source onto
            its retained coefficient. Multiple sources may share a coefficient.
        aggregation_error: Explicit dense joint aggregation covariance with
            the exact gathered native row labels and ppm axis-unit coordinates.
        provenance: Caller-supplied provenance, supplemented with selected
            native input identities and the declared template-model assumption.

    Returns:
        Linked prepared inputs in physical ppm, retaining native channel rows,
        signed source sensitivities and centered boundary operands. The affine
        intercept is zero and no additional oxidative ratio is applied.

    Raises:
        ValueError: Acquisition facts, source mappings, units, native boundary
            representation or covariance labels do not satisfy the contract.
        KeyError: A required native field, dimension or coordinate is absent.
    """
    if set(source_state) != {"co2", "o2"}:
        raise ValueError("source_state requires exactly co2 and o2 mappings.")
    references = _materialize_surface_references({"co2": co2, "o2": o2})
    channels = {name: _channel_templates(data, name, source_state[name], retained_prior, references[name])
                for name, data in (("co2", co2), ("o2", o2))}
    if (co2.acquisition.domain, co2.acquisition.start_date, co2.acquisition.end_date) != (
        o2.acquisition.domain, o2.acquisition.start_date, o2.acquisition.end_date,
    ):
        raise ValueError("Linked channels require the same declared acquisition domain and window.")
    co2_obs, co2_error, co2_h, co2_bc, co2_record = channels["co2"]
    o2_obs, o2_error, o2_h, o2_bc, o2_record = channels["o2"]
    joint = _stack(co2_obs, o2_obs, co2_units="ppm", o2_units="ppm", name="observed_concentration")
    error = _stack(co2_error, o2_error, co2_units="ppm", o2_units="ppm", name="independent_error_sd")
    covariance = aggregation_error.covariance
    if aggregation_error.mode != "dense" or covariance is None:
        raise ValueError("Explicit dense joint aggregation covariance is required.")
    expected = joint.indexes["observation"]
    if (covariance.dims != ("observation", "observation_cov")
            or covariance.indexes["observation"].names != expected.names
            or covariance.indexes["observation_cov"].names != [f"{n}_cov" for n in expected.names]
            or not covariance.indexes["observation"].equals(expected)
            or not covariance.indexes["observation_cov"].equals(expected.set_names([f"{n}_cov" for n in expected.names]))):
        raise ValueError("Aggregation covariance must match exact gathered native observation labels.")
    unit_axes = (("observation_units", "observation"), ("observation_units_cov", "observation_cov"))
    for name, axis in unit_axes:
        if name not in covariance.coords or covariance[name].dims != (axis,):
            raise ValueError("Aggregation covariance axes must explicitly use common ppm units.")
    # Covariance metadata boundary: auxiliary unit coordinates may themselves be lazy.
    axis_units = compute(*(covariance[name] for name, _ in unit_axes))
    if any(not (units.data == "ppm").all() for units in axis_units):
        raise ValueError("Aggregation covariance axes must explicitly use common ppm units.")
    nco2 = co2_obs.size

    def block(rows: slice, columns: slice, row: xr.DataArray, column: xr.DataArray) -> xr.DataArray:
        return xr.DataArray(covariance.data[rows, columns], dims=(row.dims[0], f"{column.dims[0]}_cov"),
                            coords={row.dims[0]: row[row.dims[0]], f"{column.dims[0]}_cov": column.indexes[column.dims[0]].values})

    prepared = prepare_co2_o2_inputs(
        co2_observations=co2_obs, o2_observations=o2_obs,
        co2_prior_forward_mean=xr.dot(co2_h, retained_prior.mean, dim=retained_prior.state_dim),
        o2_prior_forward_mean=xr.dot(o2_h, retained_prior.mean, dim=retained_prior.state_dim),
        co2_sensitivity=co2_h, o2_sensitivity=o2_h,
        o2_co2_flux_ratio=None,
        o2_co2_flux_ratio_unavailable_reason="Native signed source templates do not declare scalar state-resolved oxidative ratios.",
        co2_aggregation_covariance=block(slice(None, nco2), slice(None, nco2), co2_obs, co2_obs),
        co2_o2_aggregation_covariance=block(slice(None, nco2), slice(nco2, None), co2_obs, o2_obs),
        o2_aggregation_covariance=block(slice(nco2, None), slice(nco2, None), o2_obs, o2_obs),
        retained_prior=retained_prior, co2_units="ppm", o2_units="ppm",
        provenance={**provenance, "surface_template_recipe": {
            "affine_intercept_assumption": "zero: fully declared retained-coefficient template model",
            "source_state": _json_value(source_state),
            "co2": co2_record, "o2": o2_record}},
        boundary_operands={"co2": co2_bc, "o2": o2_bc}, independent_error_sd=error,
    )
    facts = requested_co2_facts(prepared, start_date=co2.acquisition.start_date,
                                end_date=co2.acquisition.end_date, domain=co2.acquisition.domain)
    return replace(prepared, provenance={**prepared.provenance, **facts})
