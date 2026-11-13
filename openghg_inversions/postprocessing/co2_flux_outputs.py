"""Conditional native and country flux summaries for coherent CO2 posteriors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from openghg.util import cf_ureg, molar_mass  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions._labelled_matrices import coordinates_on_dims
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


__all__ = [
    "co2_native_flux_outputs",
    "co2_country_flux_outputs",
    "co2_country_functionals",
    "co2_flux_country_functionals",
]

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


def _flux_unit_scale(flux: xr.DataArray) -> float:
    """Convert the signed reference flux to the reporting unit without loading it."""
    try:
        units = cf_ureg.parse_expression(flux.attrs["units"])
        return float(cf_ureg.Quantity(units).to("mol / m^2 / s").magnitude)
    except Exception as exc:
        raise ValueError("CO2 reference flux units must convert to mol m-2 s-1.") from exc


def _summarize(
    draws: xr.DataArray,
    *,
    prefix: str,
    group: str,
    source_dim: str | None,
    source_to_sector: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Sum signed sources before reducing samples, retaining source and sector summaries.

    Each sector sums its sources from the same draws as the total, so sector
    and total summaries describe one joint sample.
    """
    total = draws.sum(source_dim, skipna=False) if source_dim is not None else draws
    variables = {f"{prefix}_total_{group}": total}
    if source_dim is not None:
        variables[f"{prefix}_{group}"] = draws
        for sector, sources in _sector_sources(source_to_sector).items():
            selected = draws.sel({source_dim: draws[source_dim].isin(sources)})
            variables[f"{prefix}_{sector}_{group}"] = selected.sum(source_dim, skipna=False)
    return calculate_stats(xr.Dataset(variables), stats=["mean", "stdev", "quantiles"])


def _sector_sources(source_to_sector: Mapping[str, str] | None) -> dict[str, Sequence[str]]:
    """Group native sources by reporting sector, in sorted sector order."""
    if not source_to_sector:
        return {}
    return {
        sector: [source for source, mapped in source_to_sector.items() if mapped == sector]
        for sector in sorted(set(source_to_sector.values()))
    }


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
    scale = _flux_unit_scale(bound.affine_map.flux)
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
    reference_flux: xr.DataArray,
    countries: Countries,
    *,
    source_dim: str | None = None,
) -> xr.DataArray:
    """Return signed country weights ``A F`` in grams of CO2 per year per unit scaling.

    Country membership may also vary by source when ``source_dim`` names the
    flux's native source dimension.
    """
    flux, matrix, area = xr.align(
        reference_flux, countries.matrix, countries.area_grid, join="exact", copy=False
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
        * _flux_unit_scale(reference_flux)
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
    return _country_functionals(
        bound.affine_map.flux, countries, native_dims=bound.affine_map.native_dims, source_dim=source_dim
    )


def co2_flux_country_functionals(
    flux: xr.DataArray,
    countries: Countries,
    *,
    source_dim: str | None = None,
) -> xr.DataArray:
    """Return CO2 country totals ``Q`` from the signed reference flux alone.

    This is :func:`co2_country_functionals` before any affine map exists. A
    coherent preparation needs ``Q`` when it reduces the native problem, which
    happens before the prepared artifact and therefore before the affine
    reconstruction bound to it. ``Q`` depends only on the signed reference
    flux ``F`` and the country membership ``A``, so the same flux the affine
    map will carry gives the same ``Q``.

    Args:
        flux: Signed reference flux on ``(lat, lon)`` or
            ``(source, lat, lon)``, with units convertible to
            ``mol m-2 s-1``. Its dimension order is the native order.
        countries: Country membership and cell areas on the flux's grid.
        source_dim: The flux's source dimension, when country membership
            varies by source.

    Returns:
        ``Q`` on ``("country_functional", *flux.dims)`` as described by
        :func:`co2_country_functionals`.

    Raises:
        ValueError: If the flux is not on ``lat, lon`` native dimensions,
            optionally preceded by one source dimension, or if the country
            grid does not exactly match it.
    """
    native_dims = tuple(str(dim) for dim in flux.dims)
    if native_dims[-2:] != ("lat", "lon") or len(native_dims) not in (2, 3):
        raise ValueError("CO2 country functionals require lat/lon flux dimensions, optionally preceded by source.")
    return _country_functionals(flux, countries, native_dims=native_dims, source_dim=source_dim)


def _country_functionals(
    flux: xr.DataArray,
    countries: Countries,
    *,
    native_dims: tuple[str, ...],
    source_dim: str | None,
) -> xr.DataArray:
    """Return labelled country functionals for one signed flux on ``native_dims``."""
    weighted = _weighted_country_flux(flux, countries, source_dim=source_dim)
    extra = set(weighted.dims) - {"country", *native_dims}
    if extra:
        raise ValueError(
            "Country functionals require a flux on native dimensions only; "
            f"got extra {sorted(map(str, extra))!r}."
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
    source_to_sector: Mapping[str, str] | None = None,
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
            products must be full-restriction products whose ``Q m`` and
            ``D`` equal this map's ``A F m`` and ``A F U*`` within a
            tolerance set by the input precision. The fit's inactive states
            must be exactly the declared marginalised states; those are then
            moved into the unresolved part. Inactive states come from the
            trace's ``<flux_scale>_is_active`` constant data or, when it is
            absent, from posterior draws that are all identical.
        source_to_sector: Optional mapping from every native source of a
            multisource map to a reporting sector. Each sector's country
            totals are summed from the same draws as the source and total
            values, so complete sector totals keep the unresolved
            contribution and the cross-source covariance.

    Returns:
        xarray.Dataset:
            ``country_total_prior_*`` and ``country_total_posterior_*`` mean,
            population standard deviation, and 0.159/0.841 quantiles in grams of
            CO2 per fixed 365-day year. Multisource maps additionally retain
            ``country_prior_*`` and ``country_posterior_*`` on the source axis,
            and, with ``source_to_sector``, ``country_<sector>_prior_*`` and
            ``country_<sector>_posterior_*``. Other flux dimensions, such as time, remain in the result. Summaries
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
    if source_to_sector:
        if source_dim is None:
            raise ValueError("CO2 source_to_sector requires a multisource affine map.")
        labels = set(map(str, bound.affine_map.flux[source_dim].values))
        if set(source_to_sector) != labels:
            raise ValueError("CO2 source_to_sector must map every native source exactly once.")
        attrs["source_to_sector"] = json.dumps(dict(source_to_sector), sort_keys=True)
    if unresolved is not None:
        return _complete_country_outputs(
            trace,
            samples,
            bound,
            countries,
            unresolved,
            source_dim=source_dim,
            attrs=attrs,
            source_to_sector=source_to_sector,
        )
    affine = bound.affine_map
    weighted_flux = _weighted_country_flux(affine.flux, countries, source_dim=source_dim)
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
        outputs.append(
            _summarize(
                draws, prefix="country", group=group, source_dim=source_dim, source_to_sector=source_to_sector
            )
        )
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
    source_to_sector: Mapping[str, str] | None = None,
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
    # Both are compared relative to the contraction magnitude |Q| |x|, with a
    # tolerance set by the least precise input, so float32 storage of U*,
    # the flux or the mean is accepted while another partition is not.
    native_mean = _in_dimensionless_units(affine.native_mean, name="native_mean")
    centred = {
        group: affine._centred_state(state, bound.reference_state)[0] for group, state in samples.items()
    }
    _, prolongation = affine._centred_state(samples["posterior"], bound.reference_state)
    tolerance = _identity_tolerance(
        affine.flux, native_mean, prolongation, products.functional_mean, products.retained_response
    )
    native_dims = affine.native_dims
    same_mean = _same_contraction(products.functional_mean, functionals, native_mean, native_dims, tolerance)
    if not same_mean:
        raise ValueError(
            "Unresolved functional products were prepared for a different country functional or native mean "
            f"(beyond a relative tolerance of {tolerance:.1e} for the input precision)."
        )
    response = products.retained_response
    if response.dims != (COUNTRY_FUNCTIONAL_DIM, affine.state_dim) or not same_index(
        response.indexes[affine.state_dim], prolongation.indexes[affine.state_dim]
    ):
        raise ValueError("Unresolved functional products use different retained-state labels from the map.")
    if not _same_contraction(response, functionals, prolongation, affine.native_dims, tolerance):
        raise ValueError(
            "Unresolved functional products were prepared for a different retained partition or covariance "
            f"(beyond a relative tolerance of {tolerance:.1e} for the input precision)."
        )

    marginalised = _marginalised_states(trace, samples["posterior"], response, unresolved)
    if marginalised.any():
        reference_scale = max(1.0, float(np.max(np.abs(bound.reference_state.values))))
        state_tolerance = _identity_tolerance(bound.reference_state, *samples.values()) * reference_scale
        for group, value in centred.items():
            held = value.sel({affine.state_dim: marginalised.values})
            if np.max(np.abs(held.values), initial=0.0) > state_tolerance:
                raise ValueError(
                    f"Marginalised states must be held at their retained prior mean in {group} draws "
                    f"(beyond {state_tolerance:.1e} for the input precision)."
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
        outputs.append(
            _summarize(
                draws, prefix="country", group=group, source_dim=source_dim, source_to_sector=source_to_sector
            )
        )
    attrs["uncertainty_scope"] = COMPLETE_NATIVE
    attrs["unresolved_conditioning"] = unresolved.solve.description
    attrs["marginalised_state_count"] = str(int(marginalised.sum()))
    return _with_metadata(xr.merge(outputs), attrs, units="g yr-1")


def _identity_tolerance(*arrays: xr.DataArray) -> float:
    """Return a relative identity tolerance for the least precise floating input.

    It is 100 machine epsilons of that dtype, plus a 1e-10 floor for float64
    accumulation in differently ordered sums.
    """
    eps = max(
        (float(np.finfo(array.dtype).eps) for array in arrays if np.issubdtype(array.dtype, np.floating)),
        default=float(np.finfo(np.float64).eps),
    )
    return 100.0 * eps + 1e-10


def _same_contraction(
    actual: xr.DataArray,
    functionals: xr.DataArray,
    operand: xr.DataArray,
    native_dims: tuple[str, ...],
    tolerance: float,
) -> bool:
    """Compare ``actual`` with ``Q operand`` relative to the magnitude ``|Q| |operand|``."""
    dims = list(native_dims)
    expected = to_dense(sparse_xr_dot(functionals, operand.astype("float64"), dim=dims))
    magnitude = to_dense(sparse_xr_dot(abs(functionals), abs(operand.astype("float64")), dim=dims))
    expected = expected.transpose(*actual.dims)
    scale = float(np.max(magnitude.values, initial=0.0))
    return bool(np.max(np.abs(actual.values - expected.values), initial=0.0) <= tolerance * scale)


def _marginalised_states(
    trace: xr.DataTree,
    posterior_state: xr.DataArray,
    response: xr.DataArray,
    unresolved: UnresolvedFunctionalConditioning,
) -> xr.DataArray:
    """Return the declared marginalised states after checking them against the fit's inactive states.

    A fixed state contributes nothing to the retained draws. Its functional
    contribution is complete only if the fit moved its variance into ``A``,
    which the conditioning must declare. Inactive states come from the
    trace's ``<flux_scale>_is_active`` constant data, which models record
    only when some state is fixed. Without it, states whose posterior draws
    are all identical are treated as fixed.
    """
    state_dim = str(response.dims[1])
    index = response.indexes[state_dim]
    roles = json.loads(trace.attrs.get("rhime_variable_roles", "{}"))
    name = f"{roles['flux_scale']}_is_active"
    if "constant_data" in trace.children and name in trace_group(trace, "constant_data"):
        active = trace_group(trace, "constant_data")[name]
        if active.dims != (state_dim,) or not same_index(active.indexes[state_dim], index):
            raise ValueError(f"{name!r} must use the retained-state labels of the functional products.")
        inactive = ~np.asarray(active.values, dtype=bool)
    else:
        draws = np.asarray(posterior_state.transpose("chain", "draw", state_dim).values)
        draws = draws.reshape(-1, draws.shape[-1])
        if draws.shape[0] < 2:
            raise ValueError(
                f"Fixed states cannot be detected from one posterior draw; record {name!r} in constant_data."
            )
        inactive = np.all(draws == draws[:1], axis=0)
    declared: np.ndarray = np.zeros(len(index), dtype=bool)
    if unresolved.marginalised is not None:
        if unresolved.marginalised.dims != (state_dim,) or not same_index(
            unresolved.marginalised.indexes[state_dim], index
        ):
            raise ValueError("The declared marginalised states must use the retained-state labels.")
        declared = np.asarray(unresolved.marginalised.values, dtype=bool)
    if not np.array_equal(declared, inactive):
        raise ValueError(
            "The fit's inactive states must be exactly the states marginalised into A; "
            "complete country uncertainty is undefined for states fixed without marginalisation."
        )
    return xr.DataArray(declared, dims=state_dim, coords=coordinates_on_dims(response, {state_dim}))
