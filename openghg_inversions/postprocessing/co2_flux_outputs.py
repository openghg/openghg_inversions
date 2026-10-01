"""Conditional native and country flux summaries for coherent CO2 posteriors."""

from __future__ import annotations

import json

import xarray as xr
from openghg.util import cf_ureg, molar_mass  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import sparse_xr_dot, to_dense
from openghg_inversions.basis.affine_flux_map import _in_dimensionless_units
from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap

from .countries import Countries
from .inversion_output import trace_group
from .stats import calculate_stats

__all__ = ["co2_native_flux_outputs", "co2_country_flux_outputs"]


def _output_inputs(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
) -> tuple[dict[str, xr.DataArray], str | None, dict[str, str]]:
    """Read explicit scientific roles and validate independent sample inputs."""
    metadata = json.loads(trace.attrs.get("rhime_model_metadata", "{}"))
    if not isinstance(metadata, dict):
        raise ValueError("CO2 model metadata must be a JSON object.")
    recipe = metadata.get("recipe", trace.attrs.get("rhime_recipe"))
    if not isinstance(recipe, str) or recipe not in {"co2", "co2_cached_sigma_fixed_ou"}:
        raise ValueError("Affine CO2 flux outputs require a CO2-only recipe.")
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    if not isinstance(roles, dict):
        raise ValueError("CO2 variable roles must be a JSON object.")
    name = roles.get("flux_scale")
    if not isinstance(name, str):
        raise ValueError("Affine CO2 flux outputs require an explicit flux_scale role.")
    native_dims = bound.affine_map.native_dims
    if native_dims[-2:] != ("lat", "lon") or len(native_dims) not in (2, 3):
        raise ValueError("CO2 flux outputs require lat/lon native dimensions, optionally preceded by source.")
    source_dim = native_dims[0] if len(native_dims) == 3 else None
    samples = {}
    for group in ("prior", "posterior"):
        if group not in trace.children or name not in trace_group(trace, group):
            raise ValueError(f"Affine CO2 flux outputs require {group} flux_scale draws.")
        state = trace_group(trace, group)[name]
        if set(state.dims) != {"chain", "draw", bound.affine_map.state_dim}:
            raise ValueError(
                "CO2 flux_scale draws require exactly chain, draw, and retained-state dimensions."
            )
        if not state.sizes["chain"] or not state.sizes["draw"]:
            raise ValueError("CO2 flux_scale draws must contain samples.")
        samples[group] = state
    artifact = bound.artifact
    attrs = {
        "species": "co2",
        "rhime_recipe": recipe,
        "rhime_variable_roles": json.dumps(roles, sort_keys=True),
        "uncertainty_scope": bound.affine_map.uncertainty_scope,
        "prepared_inputs_id": artifact.prepared_inputs_id,
        "projection_provenance": json.dumps(dict(artifact.projection_provenance), sort_keys=True),
        "reconstruction_provenance": json.dumps(dict(artifact.reconstruction_provenance), sort_keys=True),
        "source_provenance": json.dumps(dict(artifact.source_provenance), sort_keys=True),
    }
    return samples, source_dim, attrs


def _flux_unit_scale(bound: BoundCo2AffineFluxMap) -> float:
    """Convert the signed reference flux to the reporting unit without loading it."""
    try:
        units = cf_ureg.parse_expression(bound.affine_map.flux.attrs["units"])
        return float(cf_ureg.Quantity(units).to("mol / m^2 / s").magnitude)
    except Exception as exc:
        raise ValueError("CO2 reference flux units must convert to mol m-2 s-1.") from exc


def _summarize(
    draws: xr.DataArray,
    *,
    prefix: str,
    group: str,
    source_dim: str | None,
) -> xr.Dataset:
    """Sum signed sources before reducing samples, retaining source summaries."""
    total = draws.sum(source_dim, skipna=False) if source_dim is not None else draws
    variables = {f"{prefix}_total_{group}": total}
    if source_dim is not None:
        variables[f"{prefix}_{group}"] = draws
    return calculate_stats(xr.Dataset(variables), stats=["mean", "stdev", "quantiles"])


def _with_metadata(result: xr.Dataset, attrs: dict[str, str], *, units: str) -> xr.Dataset:
    """Attach the same reconstruction identity and uncertainty scope to every quantity."""
    result.attrs = {**attrs, "units": units}
    for value in result.data_vars.values():
        value.attrs.update(attrs, units=units)
    return result


def co2_native_flux_outputs(trace: xr.DataTree, bound: BoundCo2AffineFluxMap) -> xr.Dataset:
    """Summarize signed affine native flux draws from a CO2-only trace.

    Args:
        trace: Annotated ordinary or cached fixed-OU CO2 trace containing prior
            and posterior draws of the explicitly declared ``flux_scale``.
            Each group needs chain, draw, and the complete retained-state axis.
        bound: Reconstruction bound to the saved prepared inputs, with the
            authoritative retained reference mean. The caller must supply the
            trace generated from that preparation; trace identity is not checked.
            Native dimensions are
            ``lat, lon`` or a leading source dimension followed by ``lat, lon``.

    Returns:
        Mean, population standard deviation, and 0.159/0.841 quantiles in
        ``mol m-2 s-1``. ``flux_total_prior_*`` and ``flux_total_posterior_*``
        summarize source sums. Multisource maps also retain ``flux_prior_*``
        and ``flux_posterior_*`` on their native source axis. Prior and posterior
        sample axes are reduced independently, pooling all chains. Results
        remain lazy for Dask inputs; quantiles consolidate sample chunks through
        the existing statistics helper. Inputs remain borrowed.

    Raises:
        ValueError: If the recipe, role, samples, native layout, state labels,
            or units cannot represent these products faithfully.

    Notes:
        Uncertainty describes retained-state-conditional native means only;
        unresolved native covariance is not added. Nonlinear statistics are
        calculated after signed reconstruction and after source summation.
    """
    samples, source_dim, attrs = _output_inputs(trace, bound)
    scale = _flux_unit_scale(bound)
    outputs = [
        _summarize(
            bound.state_to_flux(state.astype("float64")) * scale,
            prefix="flux",
            group=group,
            source_dim=source_dim,
        )
        for group, state in samples.items()
    ]
    return _with_metadata(xr.merge(outputs), attrs, units="mol m-2 s-1")


def co2_country_flux_outputs(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
) -> xr.Dataset:
    """Summarize affine CO2 country flux without constructing native flux draws.

    Args:
        trace: CO2-only annotated trace, with the same sample requirements as
            :func:`co2_native_flux_outputs`.
        bound: Reconstruction bound to its authoritative prepared reference.
            The caller must supply the trace generated from that preparation;
            trace identity is not checked.
        countries: Country membership and cell areas on exactly the map's
            indexed ``lat, lon`` grid. No regridding or partial overlap is used.
            ``Countries.area_grid`` is in square metres.

    Returns:
        ``country_total_prior_*`` and ``country_total_posterior_*`` mean,
        population standard deviation, and 0.159/0.841 quantiles in grams of
        CO2 per fixed 365-day year. Multisource maps additionally retain
        ``country_prior_*`` and ``country_posterior_*`` on the source axis.
        Other flux dimensions, such as time, remain in the result. Computation
        stays lazy for Dask inputs. Compact country reference/response arrays
        use dense chunk payloads; sample chunks for quantiles are consolidated.
        Inputs are borrowed.

    Raises:
        ValueError: If input requirements or exact country-grid alignment fail.

    Notes:
        Country reference values ``A F m`` and responses ``A F U*`` are formed
        before applying any sample axes. The result carries the conditional
        uncertainty scope and reconstruction identity; unresolved native
        covariance is not included. Source totals are summed per draw.
    """
    samples, source_dim, attrs = _output_inputs(trace, bound)
    affine = bound.affine_map
    flux, matrix, area = xr.align(
        affine.flux, countries.matrix, countries.area_grid, join="exact", copy=False
    )
    if set(matrix.dims) != {"country", "lat", "lon"} or set(area.dims) != {"lat", "lon"}:
        raise ValueError("Country membership and area must use country/lat/lon and lat/lon dimensions.")
    weighted_flux = (
        flux.astype("float64")
        * matrix
        * area
        * _flux_unit_scale(bound)
        * (365 * 24 * 3600)
        * molar_mass("co2")
    )
    # These package-internal affine helpers preserve exact state validation,
    # dimensionless conversion and lazy bucket/explicit prolongation. Reusing
    # them avoids introducing a second reconstruction or quantity-map API.
    native_mean = _in_dimensionless_units(affine.native_mean, name="native_mean")
    reference = to_dense(sparse_xr_dot(weighted_flux, native_mean, dim=["lat", "lon"]))
    # Compact country arrays use dense chunks to keep mixed sparse/dense
    # additions consistent. to_dense preserves outer Dask laziness.
    outputs = []
    response = None
    for group, state in samples.items():
        centred, prolongation = affine._centred_state(state, bound.reference_state)
        if response is None:
            response = to_dense(sparse_xr_dot(weighted_flux, prolongation, dim=["lat", "lon"]))
        draws = reference + xr.dot(response, centred.astype("float64"), dim=affine.state_dim)
        outputs.append(_summarize(draws, prefix="country", group=group, source_dim=source_dim))
    attrs["annualization"] = "365-day year; grams of CO2, not grams of carbon"
    return _with_metadata(xr.merge(outputs), attrs, units="g yr-1")
