"""Explicit continuous-surface CO2/O2 preparation beside ModelScenario.

This pilot uses OpenGHG's public integrated convolution and private alignment
and resolved convolution helpers. Missing named operations fail explicitly;
private imports can migrate when equivalent public operations become available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
import json
from typing import Literal, Protocol, cast

from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData, ObsData

import pandas as pd
import xarray as xr

from ._surface_support import _time_index, validate_flux_coverage as _coverage
from ._provenance import footprint_metadata, selected_provenance
from ._units import mole_fraction_unit_scale
from ._site_options import is_surface_platform


TransportMode = Literal["integrated", "time_resolved"]
FootprintInput = xr.Dataset | FootprintData
SurfaceInput = xr.Dataset | ObsData | FootprintData | FluxData | BoundaryConditionsData


class _AlignObservations(Protocol):
    """Provider callable for paired observation/transport resampling."""

    def __call__(
        self, obs: xr.Dataset, other: xr.Dataset, *, resample_to: str
    ) -> tuple[xr.Dataset, xr.Dataset]: ...


def _surface_operations() -> tuple[
    _AlignObservations,
    Callable[[xr.Dataset, xr.Dataset], xr.DataArray],
    Callable[[xr.DataArray | xr.Dataset, xr.Dataset], xr.DataArray],
]:
    """Small dependency seam for provider-owned alignment and convolution."""
    try:
        from openghg.analyse._alignment import resample_obs_and_other
        from openghg.analyse import fp_x_flux_integrated
        from openghg.analyse._modelled_obs import fp_x_flux_time_resolved
    except ImportError as exc:
        raise RuntimeError(
            "Surface preparation requires OpenGHG alignment and integrated/time-resolved "
            "convolution helpers. Install a compatible OpenGHG development revision."
        ) from exc
    return resample_obs_and_other, partial(fp_x_flux_integrated, allow_climatology=False), fp_x_flux_time_resolved


def _regular_footprint_times(footprint: xr.Dataset) -> pd.DatetimeIndex:
    """Require indexed, regular native footprint releases without gaps."""
    times = _time_index(footprint, "Footprint")
    if len(times) < 2 or (len(times) >= 3 and pd.infer_freq(times) is None):
        raise ValueError(
            "Surface footprints require a regular cadence with at least two timestamps and no gaps."
        )
    return times


def _align_footprint(
    obs: xr.Dataset, fp: xr.Dataset, align: _AlignObservations
) -> tuple[xr.Dataset, xr.Dataset]:
    """Choose coarsest observation bins and aligned site transport.

    The provider averages observations/transport to their coarsest cadence.
    Repair its retained cadence metadata, check observed coverage, then fill
    site transport onto those bins. Native source releases remain separate for
    forming products before observation averaging. Inputs borrow array storage.
    """
    times = _regular_footprint_times(fp)
    aligned_obs, aligned_fp = align(obs.copy(deep=False), fp, resample_to="coarsest")
    obs_times = _time_index(aligned_obs, "Aligned observations")
    fp_times = _time_index(aligned_fp, "Aligned footprint")
    if obs_times[0] < fp_times[0] or obs_times[-1] > fp_times[-1]:
        raise ValueError("Inadequate footprint coverage at observed timestamps.")
    # Provider resampling keeps the original sampling metadata. Correct it before
    # checking other contributions against this site's chosen observation cadence.
    if len(fp_times) > 1:
        period = (fp_times[1] - fp_times[0]).total_seconds()
    else:
        inferred_period = aligned_obs.attrs.get(
            "averaged_period", aligned_obs.attrs.get(
                "sampling_period_estimate", aligned_obs.attrs.get("sampling_period")
            )
        )
        period = max(
            (times[1] - times[0]).total_seconds(),
            1.0 if inferred_period is None or inferred_period == "multiple" else float(inferred_period),
        )
    aligned_obs.attrs["sampling_period"] = period
    if "averaged_period" in aligned_obs.attrs:
        aligned_obs.attrs["averaged_period"] = period
    return aligned_obs, aligned_fp.reindex(time=obs_times, method="ffill")


def _transport_provenance(value: FootprintInput) -> dict[str, object]:
    """Keep selected input identity and transport metadata for supplied inputs too."""
    return {**asdict(selected_provenance(value)), **footprint_metadata(value)}


def _dataset(value: SurfaceInput) -> xr.Dataset:
    """Own shallow metadata containers, borrow arrays, and import wrapper validity metadata."""
    dataset = value if isinstance(value, xr.Dataset) else value.data
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Surface inputs must be xarray Datasets or OpenGHG data wrappers.")
    # Provider alignment may write descriptive attrs. Own containers, borrow arrays.
    result = dataset.copy(deep=False)
    period = getattr(value, "metadata", {}).get("time_period")
    if "time_period" not in result.attrs and period is not None:
        result.attrs["time_period"] = period
    return result


def _validate_platform(value: xr.Dataset | ObsData) -> None:
    platform = (
        value.data.attrs.get("platform") if not isinstance(value, xr.Dataset) else value.attrs.get("platform")
    )
    if platform is None:
        platform = getattr(value, "metadata", {}).get("platform")
    if platform is not None and not is_surface_platform(platform):
        raise ValueError(
            f"Surface preparation supports continuous surface observations, got platform {platform!r}."
        )


def _observation_roles(
    observations: xr.Dataset, *, species: str, observation_variable: str | None
) -> dict[str, str]:
    """Resolve each supported observation role independently before renaming."""
    roles = {}
    for suffix in ("", "_error", "_repeatability", "_variability", "_number_of_observations"):
        candidates = {"mf" + suffix, species + suffix}
        if not suffix and observation_variable is not None:
            if observation_variable not in observations:
                raise ValueError(f"Observation primary variable {observation_variable!r} is missing.")
            candidates.add(observation_variable)
        present = sorted(candidates.intersection(observations.data_vars))
        role = "primary mole fraction" if not suffix else suffix[1:]
        if len(present) > 1:
            raise ValueError(f"Observation role {role!r} has duplicate aliases: {', '.join(present)}.")
        if present:
            if present[0] in roles:
                raise ValueError(f"Observation variable {present[0]!r} is selected for both the primary "
                                 f"mole fraction and {role!r} roles.")
            roles[present[0]] = "mf" + suffix
        elif not suffix:
            raise ValueError("Observation role 'primary mole fraction' requires 'mf', the species name, "
                             "or observation_variable.")
    return roles


def _native_releases(footprint: xr.Dataset, observations: xr.Dataset) -> xr.Dataset:
    """Retain native releases in the selected, aligned observation intervals."""
    times = _time_index(observations, "Aligned observations")
    period = pd.Timedelta(seconds=float(
        observations.attrs.get("averaged_period", observations.attrs["sampling_period"])
    ))
    return footprint.sel(time=slice(times[0], times[-1] + period - pd.Timedelta("1ns")))


def _convert(array: xr.DataArray, units: str) -> xr.DataArray:
    return array.pint.quantify().pint.to(units).pint.dequantify()


def _resolve_surface_observations(
    obs: xr.Dataset, site_footprint_view: xr.Dataset, *, roles: Mapping[str, str],
    output_units: str | None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Convert validated observation roles before averaging, then retain native releases.

    Returns aligned observations, site transport on those observation bins,
    and native site releases within the bins. These borrowed payloads can be
    reused for retrieval support and the subsequent scientific calculation.
    """
    obs = obs.rename({name: role for name, role in roles.items() if name != role})
    observation_units = obs.mf.attrs.get("units")
    units = output_units or observation_units
    if not isinstance(units, str) or not isinstance(observation_units, str):
        raise ValueError("Surface observations require explicit mole-fraction units.")
    mole_fraction_unit_scale(units, context="surface observations")
    for name in obs.data_vars:
        if str(name) in {"mf", "mf_repeatability", "mf_variability", "mf_error"}:
            if "units" not in obs[name].attrs:
                obs[name].attrs["units"] = observation_units
            obs[name] = _convert(obs[name], units)
    align, _, _ = _surface_operations()
    aligned_obs, site_fp = _align_footprint(obs, site_footprint_view, align)
    return aligned_obs, site_fp, _native_releases(site_footprint_view, aligned_obs)


def _select_native_footprint(footprint: xr.Dataset, aligned_obs: xr.Dataset, *, source: str) -> xr.Dataset:
    """Validate a distinct source cadence and retain releases in observation bins."""
    times = _regular_footprint_times(footprint)
    period = pd.Timedelta(seconds=float(aligned_obs.attrs.get("averaged_period", aligned_obs.attrs["sampling_period"])))
    if times[1] - times[0] > period:
        raise ValueError(
            f"Footprint source {source!r} requires matching spatial and observed timestamps; "
            "its cadence must not be coarser than the explicit site observation axis."
        )
    return _native_releases(footprint, aligned_obs)


def prepare_surface_site(
    obs_data: xr.Dataset | ObsData,
    footprint_data: FootprintInput | Mapping[str, FootprintInput],
    flux_data: Mapping[str, xr.Dataset | FluxData],
    *,
    species: str,
    transport: Mapping[str, TransportMode],
    site_footprint_data: FootprintInput | None = None,
    boundary_data: xr.Dataset | BoundaryConditionsData | None = None,
    output_units: str | None = None,
    observation_variable: str | None = None,
) -> xr.Dataset:
    """Prepare supplied CO2/O2 observations and per-source responses for one site.

    Args:
        obs_data: Borrowed observations with indexed ``time``, ``mf`` or species-
            named mole fractions and optional repeatability/variability/error
            fields. Mole fractions require Pint-compatible units (e.g. ``ppm``);
            errors without units inherit them. ``sampling_period`` gives cadence.
        footprint_data: Shared footprint or source-to-footprint mapping with
            indexed ``lat``, ``lon``, regular ``time`` and sensitivity units
            compatible with ``s m2 / mol``. Integrated transport uses ``fp``;
            resolved transport uses ``fp_time_resolved`` plus ``fp_residual``
            or legacy ``fp_HiTRes``, with indexed backward hours ``H_back``.
        flux_data: Source-to-flux mapping; each ``flux(lat, lon, time)`` has units
            convertible to ``mol / m2 / s``. Native source time axes may differ.
            Validity requires regular cadence or ``time_period`` metadata.
        species: ``co2`` or ``o2`` mole fractions; per-meg/APO is unsupported.
        transport: Exactly one ``integrated`` or ``time_resolved`` per source.
            Resolved transport includes its monthly residual once; requires full
            residual months/backward history, whole-hour coordinate intervals,
            compatible rolling windows and flux cadence within the history window.
        site_footprint_data: Required for mapped footprints; otherwise defaults
            to the shared footprint. Owns coarsest-cadence observation alignment,
            filter fields and boundary transport. Source responses must preserve
            the resulting observation axis and spatial grid.
        boundary_data: Optional ``vmr_n/e/s/w`` curtains on ``height``, ``lon``
            (north/south) or ``lat`` (east/west), interval-start ``time``, with
            ``time_period`` metadata and mole-fraction units. Requires matching
            ``particle_locations_n/e/s/w`` in site transport.
        output_units: Concentration output units; defaults to observation units.
        observation_variable: Explicit primary mole-fraction variable for supplied
            observations. Error/count aliases still use ``mf`` or the species prefix.
            Duplicate aliases for any role are rejected; unrelated fields are preserved.

    Returns:
        Dataset with aligned ``mf``, available errors/filter fields,
        ``fp_x_flux_sectoral`` with dimensions ``source``, ``lat``, ``lon``,
        ``time`` and source sum ``fp_x_flux``; spatial sums ``mf_mod_sectoral(source, time)`` and
        ``mf_mod(time)``; transport/provenance attributes. With boundaries,
        ordinary output contains ``bc_n/e/s/w`` on native curtain axes.
        ``bc_mod(time)`` is the absolute unit-factor contribution. Concentrations
        use output units. Non-time spatial axis order follows labelled inputs;
        ``time`` is last. Payloads stay lazy where supported; inputs are never mutated.

    Raises:
        ValueError: Unsupported platform/species/transport, incompatible labelled
            coordinates, missing coverage, unsupported resolved cadence, or
            invalid boundary units. Pint reports incompatible units.
        RuntimeError: Required OpenGHG alignment or convolution capability is missing.

    """
    if species.lower() not in {"co2", "o2"}:
        raise ValueError("Surface preparation currently supports only CO2 and O2.")
    _validate_platform(obs_data)
    if not flux_data or set(flux_data) != set(transport):
        raise ValueError("transport must specify exactly one mode per flux source.")
    if any(not isinstance(source, str) for source in flux_data):
        raise ValueError("Surface flux source labels must be strings.")
    if any(mode not in {"integrated", "time_resolved"} for mode in transport.values()):
        raise ValueError("Surface transport modes must be 'integrated' or 'time_resolved'.")
    mapped = isinstance(footprint_data, Mapping) and not isinstance(footprint_data, xr.Dataset)
    if not mapped:
        shared_footprint = cast(FootprintInput, footprint_data)
        source_footprints: Mapping[str, FootprintInput] = {
            source: shared_footprint for source in flux_data
        }
    else:
        source_footprints = cast(Mapping[str, FootprintInput], footprint_data)
        if set(source_footprints) != set(flux_data):
            raise ValueError("Per-source footprints must match the flux source labels.")
    if site_footprint_data is not None:
        site_footprint = site_footprint_data
    elif not mapped:
        site_footprint = shared_footprint
    else:
        raise ValueError("Mapped source footprints require explicit site_footprint_data.")
    obs = _dataset(obs_data)
    site_footprint_view = _dataset(site_footprint)
    source_footprint_views = {
        source: site_footprint_view if value is site_footprint else _dataset(value)
        for source, value in source_footprints.items()
    }
    flux_views = {source: _dataset(value) for source, value in flux_data.items()}
    boundary_view = _dataset(boundary_data) if boundary_data is not None else None
    _time_index(obs, "Observations")
    species = species.lower()
    roles = _observation_roles(obs, species=species, observation_variable=observation_variable)
    aligned_obs, site_fp, native_site_fp = _resolve_surface_observations(
        obs, site_footprint_view, roles=roles, output_units=output_units,
    )
    native_footprints = {
        source: native_site_fp if value is site_footprint_view else _select_native_footprint(value, aligned_obs, source=source)
        for source, value in source_footprint_views.items()
    }
    site = _surface_site_responses(
        aligned_obs, site_fp, native_site_fp, native_footprints, flux_views,
        species=species, transport=transport, boundary_view=boundary_view,
    )
    return _attach_surface_provenance(
        site, site_footprint=site_footprint, source_footprints=source_footprints,
        flux_data=flux_data, boundary_data=boundary_data, transport=transport,
        site_footprint_selection="explicit" if site_footprint_data is not None else "shared",
    )


def _surface_site_responses(
    aligned_obs: xr.Dataset, site_fp: xr.Dataset, native_site_fp: xr.Dataset,
    native_footprints: Mapping[str, xr.Dataset], flux_views: Mapping[str, xr.Dataset],
    *, species: str, transport: Mapping[str, TransportMode], boundary_view: xr.Dataset | None,
) -> xr.Dataset:
    """Form native source products and boundary operands before observation averaging."""
    align, integrated, resolved = _surface_operations()
    units = aligned_obs.mf.attrs["units"]
    obs_times = _time_index(aligned_obs, "Aligned observations")
    site = xr.merge([site_fp, aligned_obs], join="exact", compat="no_conflicts")
    site.attrs = {**site_fp.attrs, **aligned_obs.attrs, "resample_to": "coarsest"}
    def observation_means(values: xr.Dataset) -> xr.Dataset:
        # Form products on native releases before provider observation averaging.
        if "time" not in values.dims or values.indexes["time"].equals(obs_times):
            return values
        contribution_obs, averaged = align(aligned_obs.copy(deep=False), values, resample_to="obs")
        if not contribution_obs.indexes["time"].equals(obs_times):
            raise ValueError("Surface contributions require matching spatial and observed timestamps; "
                             "contributions must preserve the explicit site footprint observation axis.")
        return averaged.reindex(time=obs_times, method="ffill")

    # Each source product is formed on native releases, then averaged to the bins.
    responses = []
    for source in sorted(flux_views):
        aligned_fp = native_footprints[source]
        flux = flux_views[source]
        _coverage(flux, aligned_fp, source=source, mode=transport[source])
        if transport[source] == "integrated":
            if "fp" not in aligned_fp:
                raise ValueError(f"Integrated source {source!r} requires an integrated 'fp'.")
            response = integrated(aligned_fp, flux)
        else:
            if "fp_HiTRes" in aligned_fp:
                resolved_fp = aligned_fp.fp_HiTRes
                if not resolved_fp.attrs.get("units") and "fp" in aligned_fp:
                    inherited_units = aligned_fp.fp.attrs.get("units")
                    if inherited_units:
                        resolved_fp = resolved_fp.assign_attrs(units=inherited_units)
            else:
                if not {"fp_time_resolved", "fp_residual"} <= set(aligned_fp):
                    raise ValueError("Time-resolved transport requires resolved and residual footprint fields.")
                resolved_fp = aligned_fp[["fp_time_resolved", "fp_residual"]]
            response = resolved(resolved_fp, flux)
        response = observation_means(_convert(response, units).rename("response").to_dataset()).response
        if set(response.dims) != {"lat", "lon", "time"} or not response.indexes["time"].equals(obs_times):
            raise ValueError(
                f"Provider returned incompatible observation-space coordinates for source {source!r}."
            )
        try:
            site, response = xr.align(site, response, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError(
                "Surface source responses require matching spatial and observed timestamps."
            ) from exc
        responses.append(response)
    sectoral = xr.concat(responses, dim=xr.IndexVariable("source", sorted(flux_views)), join="exact")
    site["fp_x_flux_sectoral"] = sectoral
    site["fp_x_flux"] = sectoral.sum("source", skipna=False, keep_attrs=True)
    site["mf_mod_sectoral"] = sectoral.sum(["lat", "lon"], skipna=False, keep_attrs=True)
    site["mf_mod"] = site.fp_x_flux.sum(["lat", "lon"], skipna=False, keep_attrs=True)
    if boundary_view is not None:
        # Boundary operands use the same retained native releases as source products.
        from openghg.analyse._modelled_baseline import baseline_sensitivities

        absolute_operands = baseline_sensitivities(
            boundary_view, native_site_fp, species=species, allow_climatology=False
        )
        absolute = {
            str(name): value for name, value in observation_means(
                xr.Dataset({name: _convert(value, units) for name, value in absolute_operands.data_vars.items()})
            ).data_vars.items()
        }
        site = site.assign(absolute)
        directional_totals = [
            value.sum([dim for dim in value.dims if dim != "time"], skipna=False)
            for value in absolute.values()
        ]
        site["bc_mod"] = sum(directional_totals[1:], start=directional_totals[0]).assign_attrs(units=units)
    return site.transpose(..., "time")


def _attach_surface_provenance(
    site: xr.Dataset, *, site_footprint: FootprintInput,
    source_footprints: Mapping[str, FootprintInput], flux_data: Mapping[str, xr.Dataset | FluxData],
    boundary_data: xr.Dataset | BoundaryConditionsData | None, transport: Mapping[str, TransportMode],
    site_footprint_selection: str,
) -> xr.Dataset:
    """Attach selected input identities and scientific preparation policy once."""
    for name, value in footprint_metadata(site_footprint).items():
        site.attrs[f"footprint_{name}"] = value
    import openghg

    site.attrs["surface_transport"] = json.dumps(dict(transport), sort_keys=True)
    # Scientific policy: independent source products on native releases, then observation means.
    # This provenance marker is separate from persisted artifact/manifest format versions.
    site.attrs["surface_policy_revision"] = "957-source-wise-v1"
    site.attrs["surface_provider_version"] = str(getattr(openghg, "__version__", None) or "unknown")
    site.attrs["surface_provider_revision"] = str(getattr(openghg, "__revisionid__", None) or "unknown")
    site.attrs["surface_transport_provenance"] = json.dumps(
        {
            "site_footprint": _transport_provenance(site_footprint),
            "site_footprint_selection": site_footprint_selection,
            "boundary": asdict(selected_provenance(boundary_data)) if boundary_data is not None else None,
            "sources": {
                source: {
                    "mode": transport[source],
                    "footprint": _transport_provenance(source_footprints[source]),
                    "flux": asdict(selected_provenance(flux_data[source])),
                }
                for source in sorted(flux_data)
            },
        },
        sort_keys=True,
    )
    if "time_resolved" in transport.values():
        site.attrs["surface_provider_policy_revision"] = "openghg-time-resolved-v1"

    return site
