"""Coordinate-only support checks shared by surface retrieval and transport."""

from __future__ import annotations

from math import gcd
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


TransportMode = Literal["integrated", "time_resolved"]


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


def _resolved_hours(footprint: xr.Dataset) -> np.ndarray:
    index = footprint.indexes.get("H_back")
    if index is None:
        raise ValueError("Time-resolved transport requires indexed H_back coordinates.")
    # The last legacy entry stores residual mass, not a resolved lag.
    hours = np.asarray(index[:-1] if "fp_HiTRes" in footprint else index)
    if hours.size < 2:
        raise ValueError("Time-resolved transport requires at least two resolved H_back coordinates.")
    if (
        not np.isfinite(hours).all()
        or (hours < 0).any()
        or (hours != np.floor(hours)).any()
        or not (np.diff(hours) > 0).all()
        or not (np.diff(hours) == np.diff(hours)[0]).all()
    ):
        raise ValueError("Time-resolved transport requires regular whole-hour H_back histories.")
    spacing = int(hours[1] - hours[0])
    if spacing > 24 or 24 % spacing:
        raise ValueError("Time-resolved H_back spacing must divide 24 hours exactly.")
    return hours


def required_flux_support(
    footprint: xr.Dataset, *, mode: TransportMode
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive start and exclusive end of flux support for retained releases."""
    releases = _time_index(footprint, "Aligned footprint")
    if mode == "integrated":
        return releases[0], releases[-1] + pd.Timedelta(1, "ns")
    hours = _resolved_hours(footprint)
    history_start = releases[0] - pd.Timedelta(hours=int(hours[-1]) + 1)
    return (
        min(history_start, releases[0].to_period("M").start_time),
        (releases[-1].to_period("M") + 1).start_time,
    )


def validate_flux_coverage(
    flux: xr.Dataset, footprint: xr.Dataset, *, source: str, mode: TransportMode
) -> None:
    """Validate declared intervals and provider grids without computing payloads."""
    times = _time_index(flux, f"Flux source {source!r}")
    releases = _time_index(footprint, "Aligned footprint")
    period = flux.flux.attrs.get("time_period", flux.attrs.get("time_period"))
    offsets: dict[str, pd.offsets.BaseOffset | pd.Timedelta] = {
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
        raise ValueError(f"Flux source {source!r} has gaps; surface transport cannot extrapolate missing support.")
    start, end = required_flux_support(footprint, mode=mode)
    if mode == "time_resolved":
        if len(releases) < 2 or len(times) < 2:
            raise ValueError("Time-resolved transport requires at least two flux/release coordinates.")
        hours = _resolved_hours(footprint)
        hour = pd.Timedelta(hours=1)
        release_steps = (releases[1:] - releases[:-1]) / hour
        flux_steps = (times[1:] - times[:-1]) / hour
        if (
            not (release_steps == release_steps[0]).all()
            or (release_steps != np.floor(release_steps)).any()
            or (flux_steps != np.floor(flux_steps)).any()
            or release_steps[0] < 1
            or flux_steps[0] < 1
        ):
            raise ValueError("Time-resolved transport requires regular whole-hour flux and release intervals.")
        grid_hours = gcd(int(release_steps[0]), int(hours[1] - hours[0]))
        history_hours = int(hours[-1]) + 1
        if history_hours % grid_hours or (hours % grid_hours != 0).any():
            raise ValueError("Time-resolved history must align with the provider rolling window and transport grid.")
        if (flux_steps > history_hours).any():
            raise ValueError(
                f"Time-resolved source {source!r} has flux cadence longer than its resolved history; "
                "the provider would substitute monthly flux. Select integrated transport explicitly."
            )
    if times[0] > start or times[-1] + offset < end:
        raise ValueError(
            f"Inadequate flux coverage for source {source!r} ({mode}): required "
            f"{start} through {end} (exclusive), including backward history and full residual months."
        )
