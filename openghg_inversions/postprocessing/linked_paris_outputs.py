"""Separate PARIS products from one joint CO2/O2 posterior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path

import arviz as az
from dask import compute as dask_compute
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.operators import BucketBasisOperator
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.rhime.co2.co2_o2_model import _gather_co2_o2_sensitivity
from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
from openghg_inversions.utils import write_netcdf_preserving_bounds_attrs

from .inversion_output import InversionOutput
from .make_paris_outputs import (
    PARIS_LATEST_COUNTRIES,
    _add_observation_time_bounds,
    _cast_data_vars_to_template_dtypes,
    _convert_time_and_bounds_to_epoch_days,
    _expand_sector_template_mapping,
    _prepare_latest_paris_netcdf_encoding,
    add_variable_attrs,
    get_data_var_attrs,
    make_global_attrs,
    paris_flux_output,
    paris_template_files,
)

__all__ = ["make_co2_o2_paris_outputs", "reconstruct_co2_o2_concentrations"]


def _role(trace: az.InferenceData, role: str, group: str) -> xr.DataArray:
    """Select an explicitly declared scientific quantity."""
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    name = roles.get(role)
    dataset = getattr(trace, group, xr.Dataset())
    if name not in dataset:
        raise ValueError(f"Linked output requires scientific role {role!r} in {group}.")
    value = dataset[name]
    if group == "prior":
        value = value.rename({dim: f"prior_{dim}" for dim in ("chain", "draw") if dim in value.dims})
    return value


def reconstruct_co2_o2_concentrations(
    trace: az.InferenceData,
    prepared: Co2O2PreparedInputs,
) -> dict[str, xr.Dataset]:
    """Reconstruct channel components while preserving joint sample axes.

    Args:
        trace: Joint inference data with scientific roles and restored indexes.
            Prior/posterior flux scaling and posterior modelled concentration are required.
            A selected boundary/offset also requires prior draws for that term.
        prepared: Labelled inputs used for this posterior.

    Returns:
        CO2 and O2 datasets with observations, posterior model means, residuals
        (observed minus modelled), affine, boundary, offset, and per-source
        contributions in channel units. Chain/draw labels remain shared. Prior
        draws are required so fixed-state activity is preserved. Their sample
        dimensions are named ``prior_chain`` and ``prior_draw``.
        Inputs remain borrowed; computation is deferred to product creation.

    Raises:
        ValueError: If roles/prior baseline draws are missing or independently
            supplied posterior and preparation labels differ.
    """
    if trace.attrs.get("rhime_recipe") not in {"co2_o2", "co2_o2_cached_sigma_fixed_ou"}:
        raise ValueError("Linked PARIS outputs require one joint co2_o2 posterior.")
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    sensitivity = _gather_co2_o2_sensitivity(prepared.co2_sensitivity, prepared.o2_sensitivity)
    state_dim = prepared.retained_prior.state_dim
    scaling = _role(trace, "flux_scale", "posterior")
    sensitivity, scaling = xr.align(sensitivity, scaling, join="exact", copy=False)
    observations, modelled = xr.align(
        prepared.observations,
        _role(trace, "modelled_concentration", "posterior"),
        join="exact",
        copy=False,
    )
    prior_scale = _role(trace, "flux_scale", "prior")
    sensitivity, prior_scale = xr.align(sensitivity, prior_scale, join="exact", copy=False)
    result = {}
    for species in ("co2", "o2"):
        rows = np.flatnonzero(observations["species"].values == species)
        observed = observations.isel(observation=rows)
        fixed = prepared.fixed_prior_contribution.isel(observation=rows)
        data = {"observed": observed, "affine": fixed, "modelled_posterior": modelled.isel(observation=rows)}
        total_prior = fixed
        for when, scales in (("posterior", scaling), ("prior", prior_scale)):
            for component in ("boundary", "offset"):
                role = f"{component}_concentration"
                value = (
                    _role(trace, role, when).isel(observation=rows)
                    if role in roles
                    else xr.zeros_like(observed)
                )
                data[f"{component}_{when}"] = value
                if when == "prior":
                    total_prior = total_prior + value
            selected = prepared.retained_prior.mean.tracer_scope.str.lower().isin(["shared", species])
            for source in np.unique(prepared.retained_prior.mean.source.values[selected.values]):
                states = np.flatnonzero((selected & (prepared.retained_prior.mean.source == source)).values)
                term = xr.dot(
                    sensitivity.isel(observation=rows, **{state_dim: states}),
                    scales.isel({state_dim: states}),
                    dim=state_dim,
                )
                data[f"{str(source).lower()}_{when}"] = term
                if when == "prior":
                    total_prior = total_prior + term
        data["modelled_prior"] = total_prior
        data["residual_posterior"] = observed - data["modelled_posterior"]
        data["total_error"] = _role(trace, "total_marginal_error", "posterior").isel(observation=rows)
        dataset = xr.Dataset(data)
        dataset.attrs.update(
            species=species,
            has_offset=f"{species}_offset_concentration" in roles,
        )
        result[species] = dataset
    return result


def _concentration_product(
    components: xr.Dataset,
    *,
    domain: str,
    obs_avg_period: str,
) -> xr.Dataset:
    """Apply existing v04 names, units, time conventions, and dtypes."""
    species = components.attrs["species"]
    raw_units = components.attrs["units"]
    # A per-meg ratio anomaly is dimensionless but not a dry-air mole fraction.
    if "meg" in raw_units.lower() or "delta" in raw_units.lower():
        raise ValueError(
            f"PARIS {species} concentration cannot represent {raw_units!r} as dry-air mole fraction."
        )
    scale = mole_fraction_unit_scale(raw_units, context=f"linked PARIS {species}")
    if "time" not in components.coords or "site" not in components.coords:
        raise ValueError("Linked PARIS concentration requires observation-aligned site and time coordinates.")
    if not np.issubdtype(components.time.dtype, np.datetime64):
        raise ValueError("Linked PARIS concentration requires datetime observation coordinates.")
    sampled_dims = [dim for dim in ("chain", "draw", "prior_chain", "prior_draw") if dim in components.dims]
    means = components.mean(sampled_dims).reset_index("observation").rename(observation="index")
    sources = [
        str(name).removesuffix("_posterior")
        for name in means.data_vars
        if str(name).endswith("_posterior")
        and str(name).removesuffix("_posterior") not in {"modelled", "residual", "boundary", "offset"}
    ]
    summed = components.affine + components.boundary_posterior + components.offset_posterior
    for source in sources:
        summed = summed + components[f"{source}_posterior"]
    summed, modelled = xr.broadcast(summed, components.modelled_posterior)
    if not np.allclose(summed, modelled.transpose(*summed.dims), rtol=1e-6, atol=1e-8):
        raise ValueError(f"Linked {species} concentration does not close against its declared components.")
    result = xr.Dataset(
        {
            "mf_observed": means.observed,
            "stdev_mf_total": means.total_error,
            "mf_coherent_prior": means.affine,
            "mf_residual_posterior": means.residual_posterior,
        }
    )
    for when in ("prior", "posterior"):
        result[f"mf_{when}"] = means[f"modelled_{when}"]
        result[f"mf_bc_{when}"] = means[f"boundary_{when}"] + means[f"offset_{when}"]
        if components.attrs["has_offset"]:
            result[f"mf_bias_{when}"] = means[f"offset_{when}"]
        dims = [d for d in sampled_dims if d in components[f"modelled_{when}"].dims]
        if dims:
            result[f"stdev_mf_{when}"] = (
                components[f"modelled_{when}"]
                .std(dims)
                .reset_index("observation")
                .rename(observation="index")
            )
        for source in sources:
            result[f"mf_{source}_{when}"] = means[f"{source}_{when}"]
    platforms, identifiers = np.unique(result.site.values.astype(str), return_inverse=True)
    result = result.assign_coords(platform=platforms, sector=sources)
    result["number_of_identifier"] = ("index", identifiers)
    result["assimilation_flag"] = xr.ones_like(result.mf_observed, dtype="int16")
    for name in ("latitude", "longitude", "altitude"):
        if name not in result:
            result[name] = xr.full_like(result.mf_observed, np.nan)
    result = result.drop_vars(["species", "site", "observation_units"], errors="ignore")
    template = paris_template_files("latest")
    attrs = _expand_sector_template_mapping(get_data_var_attrs(template.concentration, species), sources)
    for name, description in (
        ("mf_coherent_prior", "fixed coherent affine contribution"),
        ("mf_residual_posterior", "observed minus posterior modelled mole fraction"),
    ):
        attrs[name] = {"units": "mol mol-1", "long_name": description}
    result = result.pipe(_add_observation_time_bounds, obs_avg_period).pipe(
        _convert_time_and_bounds_to_epoch_days
    )
    result = add_variable_attrs(result, attrs, scale)
    result.attrs = make_global_attrs(
        "conc", species=species, domain=domain, apriori_description="See linked model provenance"
    )
    result.attrs["paris_concentration_template_version"] = template.concentration_version
    result = _cast_data_vars_to_template_dtypes(result, template.concentration, sector_names=sources)
    return _prepare_latest_paris_netcdf_encoding(result)


def make_co2_o2_paris_outputs(
    trace: az.InferenceData,
    prepared: Co2O2PreparedInputs,
    *,
    domain: str = "EUROPE",
    obs_avg_period: str = "4h",
    template_version: str = "latest",
    native_flux_bases: Mapping[str, BasisFunctions] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    flux_frequency: str = "monthly",
    country_file: str | Path | None = None,
    country_selections: tuple[str, ...] | None = PARIS_LATEST_COUNTRIES,
    output_path: str | Path | None = None,
) -> dict[str, dict[str, xr.Dataset]]:
    """Create separate CO2/O2 products using existing PARIS templates.

    Args:
        trace: Joint posterior with scientific roles and restored indexes. It
            is never split or modified; covariance remains in this artifact.
        prepared: Inputs used to build the posterior.
        domain: Domain label for both products.
        obs_avg_period: Averaging interval; observation timestamps mark starts.
        template_version: Only ``latest`` (concentration v04, flux v03).
            Delta O2/N2 per-meg concentrations are unsupported.
        native_flux_bases: Optional species-to-basis mapping for total native
            flux. Each single-source bucket basis uses the full retained state labels
            and zero flux response for the other tracer's private states.
            Prior flux already contains native tracer signs/ratios. Native
            flux is never inferred from concentration sensitivity. Overlapping
            source and coherent native conditional reconstructions are not
            supported here. Native uncertainty is conditional on retained state.
        start_date: Inversion start, required for native flux products.
        end_date: Inversion end, required for native flux products.
        flux_frequency: Flux reporting interval for the existing flux writer.
        country_file: Country masks for native flux products.
        country_selections: Country codes, or None for all supplied countries.
        output_path: Optional directory for separate species/product NetCDFs.

    Returns:
        Mapping keyed by ``co2`` and ``o2``, containing ``concentration`` and,
        when requested, ``flux`` datasets. Products are eager marginals of one
        joint posterior. Cross-channel covariance and joint likelihood remain
        in the linked artifact. Serialization does not change input objects.

    Raises:
        ValueError: If a template, unit, component, or native representation
            cannot be mapped faithfully to the supported product.
    """
    if template_version != "latest":
        raise ValueError("Linked PARIS products support only template_version='latest'.")
    native_flux_bases = native_flux_bases or {}
    if native_flux_bases.keys() - {"co2", "o2"}:
        raise ValueError("Native linked PARIS flux accepts only co2 and o2 species.")
    if native_flux_bases and (start_date is None or end_date is None):
        raise ValueError("Native linked PARIS flux requires start_date and end_date.")
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    provenance = {
        "rhime_recipe": trace.attrs["rhime_recipe"],
        "rhime_variable_roles": json.dumps(roles, sort_keys=True),
        "rhime_model_metadata": trace.attrs.get("rhime_model_metadata", "{}"),
        "linked_preparation_provenance": json.dumps(dict(prepared.provenance), sort_keys=True),
        "linked_ratio_provenance": prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"],
        "linked_state_provenance": json.dumps(
            {name: prepared.retained_prior.mean[name].values.tolist() for name in ("source", "tracer_scope")}
        ),
        "linked_posterior": "Marginals of one joint CO2/O2 posterior; cross-channel covariance remains in the linked trace.",
    }
    components = reconstruct_co2_o2_concentrations(trace, prepared)
    # Materialize both channels together at the eager PARIS product boundary.
    components = dict(
        zip(
            components,
            dask_compute(*(value.map(to_dense, keep_attrs=True) for value in components.values())),
            strict=True,
        )
    )
    for value in components.values():
        value.attrs["units"] = str(value.observation_units.values[0])
    products = {}
    for species, values in components.items():
        products[species] = {
            "concentration": _concentration_product(
                values,
                domain=domain,
                obs_avg_period=obs_avg_period,
            )
        }
        if species in native_flux_bases:
            basis = native_flux_bases[species]
            if basis.source_labels is not None or not isinstance(basis.operator, BucketBasisOperator):
                raise ValueError("Linked native PARIS flux requires a single-source bucket total-flux basis.")
            state_dim = prepared.retained_prior.state_dim
            basis_dim = basis.operator.meta.state_dim
            matrix = basis.operator.basis_matrix.rename({basis_dim: state_dim})
            matrix, state = xr.align(matrix, prepared.retained_prior.mean, join="exact", copy=False)
            excluded = state.tracer_scope.str.lower() == ("o2" if species == "co2" else "co2")
            other_flux = matrix.isel({state_dim: np.flatnonzero(excluded.values)}) * basis.flux
            if bool((other_flux != 0).any().compute()):
                raise ValueError(f"Native {species} flux basis includes the other tracer's private states.")
            try:
                flux = basis.flux.pint.quantify().pint.to("mol / m^2 / s").pint.dequantify()
            except Exception as exc:
                raise ValueError(f"Native {species} flux units must convert to mol m-2 s-1.") from exc
            view = InversionOutput(
                trace=trace,
                inv_inputs=values.observed.assign_attrs(units=values.attrs["units"])
                .rename(roles.get("observation", "mf"))
                .to_dataset(),
                basis_functions=replace(basis, flux=flux),
                run_metadata={"start_date": start_date, "end_date": end_date},
                model_metadata={
                    "species": species,
                    "domain": domain,
                    "variable_roles": roles,
                    "state_dimension_mapping": {"trace": state_dim, "basis": basis_dim},
                },
            )
            products[species]["flux"] = paris_flux_output(
                view,
                country_file=country_file,
                country_selections=country_selections,
                flux_frequency=flux_frequency,
                template_version="latest",
                inversion_grid=False,
            )
            # The common single-state writer scales statistics by signed prior
            # flux. For a bucket cell, negative scaling reverses its quantiles
            # and standard deviations use the absolute scale. The v03 writer
            # uses the symmetric 0.159/0.841 pair; country statistics already
            # come from signed draws and require no correction.
            native = products[species]["flux"]
            negative = native.flux_total_prior < 0
            for when in ("prior", "posterior"):
                name = f"stdev_flux_total_{when}"
                native[name] = native[name].copy(data=np.abs(native[name].data))
                name = f"percentile_flux_total_{when}"
                quantiles = native[name]
                reversed_quantiles = quantiles.isel(percentile=slice(None, None, -1)).assign_coords(
                    percentile=quantiles.percentile,
                )
                native[name] = xr.where(negative, reversed_quantiles, quantiles).transpose(*quantiles.dims)
                native[name].attrs = dict(quantiles.attrs)
                native[name].encoding = dict(quantiles.encoding)
        for product, dataset in products[species].items():
            dataset.attrs.update(provenance)
            if product == "flux":
                dataset.attrs["linked_native_flux_provenance"] = json.dumps(
                    native_flux_bases[species].metadata, sort_keys=True
                )
                dataset.attrs["native_uncertainty"] = (
                    "Conditional on retained state; excludes unresolved native-state covariance."
                )
    if output_path is not None:
        destination = Path(output_path)
        destination.mkdir(parents=True, exist_ok=True)
        for species, outputs in products.items():
            for product, dataset in outputs.items():
                write_netcdf_preserving_bounds_attrs(dataset, destination / f"{species}_{product}.nc")
    return products
