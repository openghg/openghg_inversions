"""Conditional native and country flux summaries for coherent CO2 posteriors."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from openghg.util import cf_ureg, molar_mass  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions.array_ops import same_index, sparse_xr_dot, to_dense
from openghg_inversions.basis.affine_flux_map import _in_dimensionless_units

from ._basis_products import bounded_affine_flux_map
from .countries import Countries, _bounded_projection
from .inversion_output import trace_group
from .stats import calculate_stats
from openghg_inversions.coherent_reduction import marginalise_functional_states

from .unresolved_functionals import UnresolvedFunctionalConditioning, complete_functional_draws

if TYPE_CHECKING:
    from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap


__all__ = ["co2_native_flux_outputs", "co2_country_flux_outputs", "co2_country_functionals"]

COUNTRY_FUNCTIONAL_DIM = "country_functional"
FUNCTIONAL_SOURCE = "functional_source"
COMPLETE_NATIVE = "complete_native"
ANNUALIZATION = "365-day year; grams of CO2, not grams of carbon"


def _output_inputs(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
) -> tuple[dict[str, xr.DataArray], str | None, dict[str, str]]:
    """Read explicit scientific roles and validate independent sample inputs."""
    metadata = json.loads(trace.attrs.get("rhime_model_metadata", "{}"))
    if not isinstance(metadata, dict):
        raise ValueError("CO2 model metadata must be a JSON object.")
    recipe = metadata.get("recipe", trace.attrs.get("rhime_recipe"))
    if recipe != "co2":
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
        trace (xarray.DataTree): Annotated ordinary or cached fixed-OU CO2 trace containing prior
            and posterior draws of the explicitly declared ``flux_scale``.
            Each group needs chain, draw, and the complete retained-state axis.
        bound (openghg_inversions.rhime.co2.co2_affine_output.BoundCo2AffineFluxMap):
            Reconstruction bound to the saved prepared inputs, with the
            authoritative retained reference mean. The caller must supply the
            trace generated from that preparation; trace identity is not checked.
            Native dimensions are
            ``lat, lon`` or a leading source dimension followed by ``lat, lon``.

    Returns:
        xarray.Dataset:
            Mean, population standard deviation, and 0.159/0.841 quantiles in
            ``mol m-2 s-1``. ``flux_total_prior_*`` and ``flux_total_posterior_*``
            summarize source sums. Multisource maps also retain ``flux_prior_*``
            and ``flux_posterior_*`` on their native source axis. Prior and posterior
            sample axes are reduced independently, pooling all chains. Native mean,
            reference flux and prolongation are chunked within the summary point
            budget before draws are reconstructed, so eager or lazy inputs give
            bounded tasks. Summaries are computed in bounded non-sample blocks;
            quantiles retain complete samples within each block. Inputs remain
            borrowed.

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
    bound = replace(bound, artifact=replace(bound.artifact, affine_map=bounded_affine_flux_map(bound.affine_map)))
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


def _weighted_country_flux(
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    *,
    source_dim: str | None = None,
) -> xr.DataArray:
    """Return signed country weights ``A F`` in grams of CO2 per year per unit scaling.

    Country membership may also vary by source when ``source_dim`` names the
    map's native source dimension.
    """
    flux, matrix, area = xr.align(
        bound.affine_map.flux, countries.matrix, countries.area_grid, join="exact", copy=False
    )
    membership_dims = [{"country", "lat", "lon"}]
    if source_dim is not None:
        membership_dims.append({"country", source_dim, "lat", "lon"})
    if set(matrix.dims) not in membership_dims or set(area.dims) != {"lat", "lon"}:
        raise ValueError(
            "Country membership and area must use country/lat/lon (optionally with the source "
            "dimension) and lat/lon dimensions."
        )
    return (
        flux.astype("float64")
        * matrix
        * area
        * _flux_unit_scale(bound)
        * (365 * 24 * 3600)
        * molar_mass("co2")
    )


def co2_country_functionals(
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    *,
    source_dim: str | None = None,
) -> xr.DataArray:
    """Return CO2 country totals as labelled native linear functionals ``Q``.

    Pass the result to :func:`openghg_inversions.coherent_reduction.reduce_native_gaussian`
    as ``functionals`` with ``functional_dim="country_functional"`` to prepare
    the products that :func:`co2_country_flux_outputs` needs for complete
    country uncertainty.

    Args:
        bound (openghg_inversions.rhime.co2.co2_affine_output.BoundCo2AffineFluxMap):
            Reconstruction whose signed reference flux defines the totals.
        countries: Country membership and cell areas on the map's grid.
        source_dim: The map's native source dimension, when country
            membership varies by source.

    Returns:
        ``Q`` on ``("country_functional", *native_dims)`` in ``g yr-1`` per
        unit scaling. A multisource map has one row per country and source,
        labelled by the ``country`` and ``functional_source`` levels, so the
        joint covariance of source terms is kept; a single-source map has
        one row per country. The payload stays sparse when the country
        matrix is sparse.

    Raises:
        ValueError: If the country grid does not exactly match the map, or
            the flux has dimensions other than the native ones, such as time.
    """
    weighted = _weighted_country_flux(bound, countries, source_dim=source_dim)
    native_dims = bound.affine_map.native_dims
    extra = set(weighted.dims) - {"country", *native_dims}
    if extra:
        raise ValueError(
            f"Country functionals require a flux on native dimensions only; got extra {sorted(map(str, extra))!r}."
        )
    levels = ["country"]
    if len(native_dims) == 3:
        source_dim = native_dims[0]
        sources = weighted[source_dim]
        selector = xr.DataArray(
            np.eye(sources.size),
            dims=(FUNCTIONAL_SOURCE, source_dim),
            coords={FUNCTIONAL_SOURCE: sources.values, source_dim: sources.values},
        )
        weighted = weighted * selector
        levels.append(FUNCTIONAL_SOURCE)
    functionals = weighted.stack({COUNTRY_FUNCTIONAL_DIM: levels})
    functionals = functionals.transpose(COUNTRY_FUNCTIONAL_DIM, *native_dims)
    return functionals.rename("country_functionals").assign_attrs(
        units="g yr-1", annualization=ANNUALIZATION, mathematical_name="Q"
    )


def co2_country_flux_outputs(
    trace: xr.DataTree,
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    *,
    unresolved: UnresolvedFunctionalConditioning | None = None,
) -> xr.Dataset:
    """Summarize affine CO2 country flux without constructing native flux draws.

    Args:
        trace (xarray.DataTree): CO2-only annotated trace, with the same sample requirements as
            :func:`co2_native_flux_outputs`.
        bound (openghg_inversions.rhime.co2.co2_affine_output.BoundCo2AffineFluxMap):
            Reconstruction bound to its authoritative prepared reference.
            The caller must supply the trace generated from that preparation;
            trace identity is not checked.
        countries: Country membership and cell areas on exactly the map's
            indexed ``lat, lon`` grid. No regridding or partial overlap is used.
            ``Countries.area_grid`` is in square metres. A multisource map's
            membership may also carry its source dimension.
        unresolved: Optional products and per-draw conditioning inputs for
            the unresolved native state of the functionals returned by
            :func:`co2_country_functionals` for the same ``bound`` and
            ``countries``. When given, country totals are complete native
            functionals rather than retained-state-conditional ones. The
            products must be full-restriction products whose ``D`` equals
            this map's ``A F U*``. The fit's inactive states (from the
            trace's ``<flux_scale>_is_active`` constant data) must be exactly
            the declared marginalised states; those are then moved into the
            unresolved part.

    Returns:
        xarray.Dataset:
            ``country_total_prior_*`` and ``country_total_posterior_*`` mean,
            population standard deviation, and 0.159/0.841 quantiles in grams of
            CO2 per fixed 365-day year. Multisource maps additionally retain
            ``country_prior_*`` and ``country_posterior_*`` on the source axis.
            Other flux dimensions, such as time, remain in the result. Summaries
            are computed in bounded non-sample blocks. Compact country
            reference/response arrays use dense chunk payloads and are prepared like
            ordinary country projections, chunking their output dimensions within
            the summary point budget; sample chunks for quantiles are consolidated.
            Existing producers must already have bounded task payloads. Inputs are
            borrowed. ``uncertainty_scope`` is ``"retained_state_conditional"``
            without ``unresolved`` and ``"complete_native"`` with it.

    Raises:
        ValueError: If input requirements or exact country-grid alignment
            fail, or if ``unresolved`` was prepared for a different country
            functional, native mean, partition or covariance, or does not
            account for the fit's inactive states.

    Notes:
        Country reference values ``A F m`` and responses ``A F U*`` are formed
        before applying any sample axes. Without ``unresolved``, the
        unresolved native state is not included: country totals and their
        spread describe ``Q m + D (alpha - Pi m)`` only. With it, each draw
        also gets the conditional mean of ``Q x_perp`` and one draw from its
        conditional covariance (prior draws: its prior covariance), using
        the products' ``D``; this is an eager per-draw boundary. Source
        totals are summed per draw.
    """
    samples, source_dim, attrs = _output_inputs(trace, bound)
    attrs["annualization"] = ANNUALIZATION
    if unresolved is not None:
        return _complete_country_outputs(
            trace, samples, bound, countries, unresolved, source_dim=source_dim, attrs=attrs
        )
    affine = bound.affine_map
    weighted_flux = _weighted_country_flux(bound, countries, source_dim=source_dim)
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
            response = _bounded_projection(response, [affine.state_dim])
            reference = _bounded_projection(reference, [])
        draws = reference + xr.dot(response, centred.astype("float64"), dim=affine.state_dim)
        outputs.append(_summarize(draws, prefix="country", group=group, source_dim=source_dim))
    return _with_metadata(xr.merge(outputs), attrs, units="g yr-1")


def _complete_country_outputs(
    trace: xr.DataTree,
    samples: dict[str, xr.DataArray],
    bound: BoundCo2AffineFluxMap,
    countries: Countries,
    unresolved: UnresolvedFunctionalConditioning,
    *,
    source_dim: str | None,
    attrs: dict[str, str],
) -> xr.Dataset:
    """Summarize complete native country functionals, conditioning each posterior draw."""
    affine = bound.affine_map
    products = unresolved.products
    if products.functional_dim != COUNTRY_FUNCTIONAL_DIM:
        raise ValueError(f"Country functional products must use the {COUNTRY_FUNCTIONAL_DIM!r} dimension.")
    if products.marginalised is not None:
        raise ValueError(
            "Pass full-restriction functional products and declare marginalised states on the conditioning."
        )
    functionals = co2_country_functionals(bound, countries, source_dim=source_dim)
    if not same_index(
        products.functional_mean.indexes[COUNTRY_FUNCTIONAL_DIM], functionals.indexes[COUNTRY_FUNCTIONAL_DIM]
    ):
        raise ValueError("Unresolved functional products were prepared for different countries or sources.")
    # Q m identifies the functional and native mean; D = Q U* also identifies
    # the retained partition and covariance, which Q m does not depend on.
    native_mean = _in_dimensionless_units(affine.native_mean, name="native_mean")
    expected_mean = to_dense(sparse_xr_dot(functionals, native_mean, dim=list(affine.native_dims)))
    if not np.allclose(products.functional_mean.values, expected_mean.values, rtol=1e-9, atol=0.0):
        raise ValueError(
            "Unresolved functional products were prepared for a different country functional or native mean."
        )
    centred = {
        group: affine._centred_state(state, bound.reference_state)[0] for group, state in samples.items()
    }
    _, prolongation = affine._centred_state(samples["posterior"], bound.reference_state)
    expected_response = to_dense(sparse_xr_dot(functionals, prolongation, dim=list(affine.native_dims)))
    expected_response = expected_response.transpose(COUNTRY_FUNCTIONAL_DIM, affine.state_dim)
    response = products.retained_response
    if response.dims != expected_response.dims or not same_index(
        response.indexes[affine.state_dim], expected_response.indexes[affine.state_dim]
    ):
        raise ValueError("Unresolved functional products use different retained-state labels from the map.")
    scale = float(np.max(np.abs(expected_response.values), initial=0.0))
    if not np.allclose(response.values, expected_response.values, rtol=1e-8, atol=1e-10 * scale):
        raise ValueError(
            "Unresolved functional products were prepared for a different retained partition or covariance."
        )

    marginalised = _marginalised_states(trace, response, unresolved)
    if marginalised.any():
        for group, value in centred.items():
            held = value.sel({affine.state_dim: marginalised.values})
            if not np.allclose(held.values, 0.0, atol=1e-12):
                raise ValueError(
                    f"Marginalised states must be held at their retained prior mean in {group} draws."
                )
        products = marginalise_functional_states(products, marginalised)

    rng = np.random.default_rng(unresolved.seed)
    outputs = []
    for group, value in centred.items():
        posterior = group == "posterior"
        draws = complete_functional_draws(
            value,
            products,
            residual=unresolved.residual if posterior else None,
            solve=unresolved.solve if posterior else None,
            rng=rng,
        )
        if posterior:
            eigenvalue = draws.attrs["conditional_covariance_min_eigenvalue"]
            attrs["conditional_covariance_min_eigenvalue"] = f"{eigenvalue:.3e}"
        draws = draws.unstack(COUNTRY_FUNCTIONAL_DIM)
        if source_dim is not None:
            draws = draws.rename({FUNCTIONAL_SOURCE: source_dim})
        outputs.append(_summarize(draws, prefix="country", group=group, source_dim=source_dim))
    attrs["uncertainty_scope"] = COMPLETE_NATIVE
    attrs["unresolved_conditioning"] = unresolved.solve.description
    attrs["marginalised_state_count"] = str(int(marginalised.sum()))
    return _with_metadata(xr.merge(outputs), attrs, units="g yr-1")


def _marginalised_states(
    trace: xr.DataTree,
    response: xr.DataArray,
    unresolved: UnresolvedFunctionalConditioning,
) -> xr.DataArray:
    """Return the declared marginalised states after checking them against the fit's inactive states.

    A fixed state contributes nothing to the retained draws. Its functional
    contribution is complete only if the fit moved its variance into ``A``,
    which the conditioning must declare.
    """
    state_dim = str(response.dims[1])
    index = response.indexes[state_dim]
    inactive = xr.DataArray(np.zeros(len(index), dtype=bool), dims=state_dim, coords={state_dim: index})
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    name = f"{roles['flux_scale']}_is_active"
    if "constant_data" in trace.children and name in trace_group(trace, "constant_data"):
        active = trace_group(trace, "constant_data")[name]
        if active.dims != (state_dim,) or not same_index(active.indexes[state_dim], index):
            raise ValueError(f"{name!r} must use the retained-state labels of the functional products.")
        inactive = inactive.copy(data=~np.asarray(active.values, dtype=bool))
    declared = inactive.copy(data=np.zeros(len(index), dtype=bool))
    if unresolved.marginalised is not None:
        if unresolved.marginalised.dims != (state_dim,) or not same_index(
            unresolved.marginalised.indexes[state_dim], index
        ):
            raise ValueError("The declared marginalised states must use the retained-state labels.")
        declared = declared.copy(data=np.asarray(unresolved.marginalised.values, dtype=bool))
    if not np.array_equal(declared.values, inactive.values):
        raise ValueError(
            "The fit's inactive states must be exactly the states marginalised into A; "
            "complete country uncertainty is undefined for states fixed without marginalisation."
        )
    return declared
