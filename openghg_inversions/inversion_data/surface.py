"""Explicit continuous-surface CO2/O2 preparation beside ModelScenario.

This pilot temporarily consumes OpenGHG private scientific helpers. A provider
without the named operations fails explicitly; no legacy scientific fallback is
used. Promote these imports when OpenGHG supplies equivalent public operations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from inspect import signature
import json
from math import gcd
from typing import Literal, Protocol, cast

from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData, ObsData

import numpy as np
import pandas as pd
import xarray as xr

from ._provenance import selected_provenance
from ._units import mole_fraction_unit_scale
from ._boundary import baseline_reference_sensitivities as _reference_sensitivities


TransportMode = Literal["integrated", "time_resolved"]
FootprintInput = xr.Dataset | FootprintData
SurfaceInput = xr.Dataset | ObsData | FootprintData | FluxData | BoundaryConditionsData


class _AlignObservations(Protocol):
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
        from openghg.analyse._modelled_obs import (
            fp_x_flux_integrated,
            fp_x_flux_time_resolved,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Surface preparation requires OpenGHG alignment and integrated/time-resolved "
            "convolution helpers. Install a compatible OpenGHG development revision."
        ) from exc
    # Released OpenGHG uses timestamp-only filling. Newer providers also offer
    # climatological year reuse; the explicit surface validity contract forbids it.
    integrated: Callable[..., xr.DataArray] = fp_x_flux_integrated
    if "allow_climatology" in signature(integrated).parameters:
        integrated = partial(integrated, allow_climatology=False)
    return resample_obs_and_other, integrated, fp_x_flux_time_resolved


def _require_residual_month_support() -> None:
    """Check monthly slicing on owned toy data before resolved preparation."""
    message = (
        "Time-resolved surface transport requires the OpenGHG midmonth residual-flux fix. "
        "Install the provider revision documented in the surface preparation guide."
    )
    try:
        from openghg.analyse._modelled_obs import _make_low_freq_flux
    except ImportError as exc:
        raise RuntimeError(message) from exc

    # Probe the behavior itself so editable installs and compatible backports work.
    flux = xr.DataArray(
        [[[2.0, 6.0]]], dims=("lat", "lon", "time"),
        coords={"lat": [0.0], "lon": [0.0], "time": pd.to_datetime(["2020-01-01", "2020-02-01"])},
    )
    footprint = xr.Dataset(coords={
        "lat": [0.0], "lon": [0.0], "H_back": [0, 1],
        "time": pd.to_datetime(["2020-01-16", "2020-02-02"]),
    })
    try:
        monthly = _make_low_freq_flux(flux, footprint)
    except (ValueError, KeyError, IndexError) as exc:
        raise RuntimeError(message) from exc
    if not np.array_equal(monthly.values, flux.values):
        raise RuntimeError(message)


def _align_footprint(
    obs: xr.Dataset, value: FootprintInput, align: _AlignObservations
) -> tuple[xr.Dataset, xr.Dataset]:
    fp = _dataset(value)
    times = _time_index(fp, "Footprint")
    if len(times) < 2 or (len(times) >= 3 and pd.infer_freq(times) is None):
        raise ValueError(
            "Surface footprints require a regular cadence with at least two timestamps and no gaps."
        )
    aligned_obs, aligned_fp = align(obs.copy(deep=False), fp, resample_to="coarsest")
    obs_times = _time_index(aligned_obs, "Aligned observations")
    fp_times = _time_index(aligned_fp, "Aligned footprint")
    if obs_times[0] < fp_times[0] or obs_times[-1] > fp_times[-1]:
        raise ValueError("Inadequate footprint coverage at observed timestamps.")
    # Provider resampling keeps the original sampling metadata. Correct it before
    # checking other contributions against this site's chosen observation cadence.
    if len(fp_times) > 1:
        period = (fp_times[1] - fp_times[0]).total_seconds()
        aligned_obs.attrs["sampling_period"] = period
        if "averaged_period" in aligned_obs.attrs:
            aligned_obs.attrs["averaged_period"] = period
    return aligned_obs, aligned_fp.reindex(time=obs_times, method="ffill")


def _transport_provenance(value: FootprintInput) -> dict[str, object]:
    """Keep selected input identity and transport metadata for supplied inputs too."""
    fp = _dataset(value)
    metadata = {**getattr(value, "metadata", {}), **fp.attrs}
    return {
        **asdict(selected_provenance(value)),
        **{
            key: metadata[key]
            for key in (
                "model",
                "transport_model",
                "transport_model_version",
                "model_version",
                "met_model",
                "metmodel",
                "max_level",
            )
            if key in metadata
        },
    }


def _dataset(value: SurfaceInput) -> xr.Dataset:
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
    if platform is not None and str(platform).lower() not in {
        "surface",
        "insitu",
        "in-situ",
        "continuous",
        "tower",
    }:
        raise ValueError(
            f"Surface preparation supports continuous surface observations, got platform {platform!r}."
        )


def _time_index(dataset: xr.Dataset, name: str) -> pd.DatetimeIndex:
    times = dataset.indexes.get("time")
    if (
        not isinstance(times, pd.DatetimeIndex)
        or times.empty
        or not times.is_monotonic_increasing
        or times.has_duplicates
    ):
        raise ValueError(f"{name} requires a nonempty, increasing, unique datetime time coordinate.")
    return times


def _coverage(flux: xr.Dataset, fp: xr.Dataset, *, source: str, mode: TransportMode) -> None:
    """Validate declared intervals and history before provider forward filling."""
    times = _time_index(flux, f"Flux source {source!r}")
    releases = _time_index(fp, "Aligned footprint")
    period = flux.flux.attrs.get("time_period", flux.attrs.get("time_period"))
    offsets = {
        "yearly": pd.DateOffset(years=1),
        "monthly": pd.DateOffset(months=1),
        "daily": pd.Timedelta(days=1),
        "hourly": pd.Timedelta(hours=1),
    }
    offset = offsets.get(str(period).lower())
    if offset is None and period is not None:
        from openghg.util import relative_time_offset

        try:
            offset = relative_time_offset(period=str(period))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Flux source {source!r} has an invalid time_period {period!r}.") from exc
    if offset is None and len(times) > 1:
        frequency = pd.infer_freq(times) if len(times) >= 3 else None
        if frequency:
            offset = pd.tseries.frequencies.to_offset(frequency)
        elif len(times) == 2:
            offset = times[1] - times[0]
    if offset is None:
        raise ValueError(
            f"Flux source {source!r} has no regular cadence or declared time_period; coverage is unknown."
        )
    if len(times) > 1 and any(a + offset != b for a, b in zip(times[:-1], times[1:], strict=True)):
        raise ValueError(
            f"Flux source {source!r} has gaps; surface transport cannot extrapolate missing support."
        )
    start, end = releases[0], releases[-1]
    if mode == "time_resolved":
        if "H_back" not in fp.coords or fp.sizes["H_back"] < 2 or len(releases) < 2 or len(times) < 2:
            raise ValueError("Time-resolved transport requires at least two flux/release/H_back coordinates.")
        h_back = fp.H_back
        if "fp_HiTRes" in fp:
            # The legacy final H_back entry is residual mass, not resolved history.
            h_back = h_back.where(h_back != h_back.max(), drop=True)
        if h_back.size < 2:
            raise ValueError("Time-resolved transport requires at least two resolved H_back coordinates.")
        hours = np.asarray(h_back.values)
        hour = pd.Timedelta(hours=1)
        release_steps = (releases[1:] - releases[:-1]) / hour
        flux_steps = (times[1:] - times[:-1]) / hour
        if (
            not np.isfinite(hours).all()
            or (hours < 0).any()
            or (hours != np.floor(hours)).any()
            or not (np.diff(hours) > 0).all()
            or not (np.diff(hours) == np.diff(hours)[0]).all()
            or not (release_steps == release_steps[0]).all()
            or (release_steps != np.floor(release_steps)).any()
            or (flux_steps != np.floor(flux_steps)).any()
            or release_steps[0] < 1
            or flux_steps[0] < 1
        ):
            raise ValueError("Time-resolved transport requires regular whole-hour flux, releases and H_back histories.")
        spacing = int(hours[1] - hours[0])
        if spacing > 24 or 24 % spacing:
            raise ValueError("Time-resolved H_back spacing must divide 24 hours exactly.")
        grid_hours = gcd(int(release_steps[0]), spacing)
        cadence_hours = int(flux_steps[0])
        # The provider slices before upsampling and can lose the preceding flux
        # interval. Bound this pilot to exact downsampling on its transport grid.
        if cadence_hours > grid_hours:
            raise ValueError(
                f"Time-resolved source {source!r} has flux cadence coarser than the "
                f"resolved transport grid GCD ({grid_hours}h). Supply finer flux or "
                "explicitly select integrated transport."
            )
        history_hours = int(hours[-1]) + 1
        padded_start = start - pd.Timedelta(hours=history_hours)
        # Provider rolling windows use max(H_back)+1 hours and exact grid labels.
        if (
            grid_hours % cadence_hours
            or history_hours % grid_hours
            or (hours % grid_hours != 0).any()
            or (releases != releases.floor("h")).any()
            or (times != times.floor("h")).any()
            or (padded_start - times[0]) % pd.Timedelta(hours=cadence_hours)
        ):
            raise ValueError("Time-resolved flux/history phases must align exactly with the provider transport grid.")
        start = padded_start.to_period("M").start_time
        end = end.to_period("M").end_time
    if times[0] > start or times[-1] + offset <= end:
        raise ValueError(
            f"Inadequate flux coverage for source {source!r} ({mode}): required "
            f"{start} through {end}, including backward history and full residual months."
        )


def _convert(array: xr.DataArray, units: str) -> xr.DataArray:
    return array.pint.quantify().pint.to(units).pint.dequantify()


def prepare_surface_site(
    obs_data: xr.Dataset | ObsData,
    footprint_data: FootprintInput | Mapping[str, FootprintInput],
    flux_data: Mapping[str, xr.Dataset | FluxData],
    *,
    species: str,
    transport: Mapping[str, TransportMode],
    site_footprint_data: FootprintInput | None = None,
    boundary_data: xr.Dataset | BoundaryConditionsData | None = None,
    boundary_reference: float | None = None,
    boundary_reference_units: str | None = None,
    boundary_mass_tolerance: float | None = None,
    output_units: str | None = None,
) -> xr.Dataset:
    """Prepare continuous CO2/O2 observations and per-source surface responses.

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
            residual months/backward history, regular whole-hour coordinates,
            exact grid alignment and flux no coarser than the transport grid.
        site_footprint_data: Required for mapped footprints; otherwise defaults
            to the shared footprint. Owns coarsest-cadence observation alignment,
            filter fields and boundary transport. Source responses must preserve
            the resulting observation axis and spatial grid.
        boundary_data: Optional ``vmr_n/e/s/w`` curtains on ``height``, ``lon``
            (north/south) or ``lat`` (east/west), interval-start ``time``, with
            ``time_period`` metadata and mole-fraction units. Requires matching
            ``particle_locations_n/e/s/w`` in site transport.
        boundary_reference: Scalar reference, required with boundary data.
        boundary_reference_units: Reference mole-fraction units, required with
            boundary data; converted before subtraction.
        boundary_mass_tolerance: Required with boundary data; allowed absolute
            deviation of total nonnegative curtain mass from one before joint
            normalization.
        output_units: Concentration output units; defaults to observation units.

    Returns:
        Dataset with aligned ``mf``, available errors/filter fields,
        ``fp_x_flux_sectoral(source, lat, lon, time)`` and source sum
        ``fp_x_flux``; spatial sums ``mf_mod_sectoral(source, time)`` and
        ``mf_mod(time)``; transport/provenance attributes. With boundaries,
        ``bc_anomaly_n/e/s/w`` and dimensionless ``bc_mass_n/e/s/w`` retain
        native curtain axes, ``bc_reference`` is scalar and ``bc_mod(time)``
        is the absolute unit-factor contribution. Concentrations use output
        units. Payloads stay lazy where supported; inputs are never mutated.

    Raises:
        ValueError: Unsupported platform/species/transport, incompatible labelled
            coordinates, missing coverage, unsupported resolved cadence, or
            invalid boundary units/mass. Pint reports incompatible units.
        RuntimeError: Required OpenGHG alignment/convolution or monthly-residual
            capability is missing.

    Derive observation errors before filtering, then project native boundary
    operands through a chosen basis before gathering.
    Standard/multisector builders and cached/custom likelihoods reject centered
    inputs; ordinary CO2 and linked CO2/O2 recipes accept them. See
    :ref:`surface-centered-boundary-contract` for projection, coherent reduction,
    prepared save/reopen and absolute predictive replay.
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
    if "time_resolved" in transport.values():
        _require_residual_month_support()
    obs = _dataset(obs_data)
    _time_index(obs, "Observations")
    species = species.lower()
    obs = obs.rename(
        {
            name: "mf" + str(name)[len(species) :]
            for name in obs.data_vars
            if name == species or str(name).startswith(species + "_")
        }
    )
    if "mf" not in obs:
        raise ValueError(f"Surface observations require mf or {species!r} mole fractions.")
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
    align, integrated, resolved = _surface_operations()
    aligned_obs, site_fp = _align_footprint(obs, site_footprint, align)
    obs_times = _time_index(aligned_obs, "Aligned observations")
    site = xr.merge([site_fp, aligned_obs], join="exact", compat="no_conflicts")
    site.attrs = {**site_fp.attrs, **aligned_obs.attrs, "resample_to": "coarsest"}
    site_metadata = {**getattr(site_footprint, "metadata", {}), **_dataset(site_footprint).attrs}
    for name, aliases in {
        "transport_model": ("model", "transport_model"),
        "transport_model_version": ("transport_model_version", "model_version"),
        "met_model": ("met_model", "metmodel"),
    }.items():
        for alias in aliases:
            if alias in site_metadata:
                site.attrs[f"footprint_{name}"] = site_metadata[alias]
                break
    if "max_level" in site_metadata:
        site.attrs["footprint_max_level"] = site_metadata["max_level"]
    responses = []
    for source in sorted(flux_data):
        if not mapped and site_footprint_data is None:
            aligned_fp = site_fp
        else:
            contribution_obs, aligned_fp = _align_footprint(aligned_obs, source_footprints[source], align)
            if not contribution_obs.indexes["time"].equals(obs_times):
                raise ValueError(
                    "Surface source responses require matching spatial and observed timestamps; "
                    "contribution alignment must preserve the explicit site footprint observation axis."
                )
        flux = _dataset(flux_data[source])
        _coverage(flux, aligned_fp, source=source, mode=transport[source])
        if transport[source] == "integrated":
            if "fp" not in aligned_fp:
                raise ValueError(f"Integrated source {source!r} requires an integrated 'fp'.")
            response = integrated(aligned_fp, flux)
        else:
            if "fp_HiTRes" in aligned_fp:
                resolved_fp = aligned_fp.fp_HiTRes
            else:
                if not {"fp_time_resolved", "fp_residual"} <= set(aligned_fp):
                    raise ValueError(
                        "Time-resolved transport requires resolved and residual footprint fields."
                    )
                resolved_fp = aligned_fp[["fp_time_resolved", "fp_residual"]]
            response = resolved(resolved_fp, flux)
        response = _convert(response, units)
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
    sectoral = xr.concat(responses, dim=xr.IndexVariable("source", sorted(flux_data)), join="exact")
    site["fp_x_flux_sectoral"] = sectoral
    site["fp_x_flux"] = sectoral.sum("source", skipna=False, keep_attrs=True)
    site["mf_mod_sectoral"] = sectoral.sum(["lat", "lon"], skipna=False, keep_attrs=True)
    site["mf_mod"] = site.fp_x_flux.sum(["lat", "lon"], skipna=False, keep_attrs=True)
    import openghg

    site.attrs["surface_transport"] = json.dumps(dict(transport), sort_keys=True)
    site.attrs["surface_policy_revision"] = "957-source-wise-v1"
    site.attrs["surface_provider_version"] = str(getattr(openghg, "__version__", None) or "unknown")
    site.attrs["surface_provider_revision"] = str(getattr(openghg, "__revisionid__", None) or "unknown")
    site.attrs["surface_transport_provenance"] = json.dumps(
        {
            "site_footprint": _transport_provenance(site_footprint),
            "site_footprint_selection": "explicit" if site_footprint_data is not None else "shared",
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
        site.attrs["surface_provider_policy_revision"] = "1790-midmonth-residual-v1"

    if boundary_data is not None:
        if boundary_reference is None or boundary_reference_units is None or boundary_mass_tolerance is None:
            raise ValueError("Surface boundaries require reference, reference_units and mass_tolerance.")
        site.attrs["boundary_representation"] = "centered-v1"
        site.attrs["boundary_mass_tolerance"] = boundary_mass_tolerance
        site.attrs["boundary_normalization_policy"] = "joint-curtain-unit-mass-v1"
        site.attrs["boundary_provider_policy_revision"] = "957-reference-sensitivities-v1"
        site.attrs["boundary_operator_owner"] = "openghg_inversions"

        operands = _reference_sensitivities(
            _dataset(boundary_data),
            site,
            species=species,
            reference=boundary_reference,
            reference_units=boundary_reference_units,
            mass_tolerance=boundary_mass_tolerance,
        )
        for name in operands.data_vars:
            site[name] = (
                operands[name] if str(name).startswith("bc_mass_") else _convert(operands[name], units)
            )
        baseline = []
        for direction in "nesw":
            anomaly, mass = site[f"bc_anomaly_{direction}"], site[f"bc_mass_{direction}"]
            spatial = [dim for dim in anomaly.dims if dim != "time"]
            baseline.append(
                anomaly.sum(spatial, skipna=False) + site.bc_reference * mass.sum(spatial, skipna=False)
            )
        site["bc_mod"] = sum(baseline)
        site.bc_mod.attrs["units"] = units
    return site.transpose(..., "time")
