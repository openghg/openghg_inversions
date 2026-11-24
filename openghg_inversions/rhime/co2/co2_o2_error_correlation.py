"""Fixed correlation between the independent errors of two linked channels at the same site and time.

The pairing is keyed by channel (``species``) labels, so it is not specific to
O2: any primary/secondary tracer pair, in any units, uses the same rule.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.models.paired_error import PairedErrorCorrelation

CrossSpeciesCorrelation = float | Mapping[str, float]


def _correlation_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a number strictly between -1 and 1.")
    result = float(value)
    if not math.isfinite(result) or abs(result) >= 1.0:
        raise ValueError(f"{name} must be a number strictly between -1 and 1.")
    return result


def validate_cross_species_error_correlation(value: object) -> CrossSpeciesCorrelation:
    """Check one correlation, or a site-keyed mapping of correlations, without observations."""
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("cross_species_error_correlation mapping must name at least one site.")
        return {
            str(site): _correlation_value(rho, f"cross_species_error_correlation[{site!r}]")
            for site, rho in value.items()
        }
    return _correlation_value(value, "cross_species_error_correlation")


def linked_cross_species_pairs(
    observations: xr.DataArray,
    correlation: CrossSpeciesCorrelation,
    *,
    channels: tuple[str, str] = ("co2", "o2"),
) -> PairedErrorCorrelation:
    """Pair each row of the first channel with the second channel's row at the same site and time.

    The correlation applies to the rows' independent errors, so its cross
    covariance ``rho * sd_first * sd_second`` is in the product of the two
    channels' units (for example ppm times per meg, or ppm times per mil);
    ``rho`` itself is dimensionless. Times must match exactly. Rows with a
    missing site or time (``None``, NaN or NaT) and rows without a partner
    stay independent.

    Args:
        observations: Joint observations with observation-aligned ``species``,
            ``site`` and ``time`` coordinates.
        correlation: One correlation for every site, or a mapping from site to
            correlation. A mapping must cover every site that has pairs and
            name only observed sites.
        channels: The two ``species`` labels to pair, first then second.

    Returns:
        Row-position pairs (first channel first) and their correlations.

    Raises:
        ValueError: If coordinates are missing, a site/time has more than one
            row of a species, no pairs exist, or correlations are invalid or
            do not match the observed sites.
    """
    correlation = validate_cross_species_error_correlation(correlation)
    first, second = (str(channel) for channel in channels)
    if first == second:
        raise ValueError("Cross-species error correlation needs two different channels.")
    for name in ("species", "site", "time"):
        if name not in observations.coords or observations[name].dims != observations.dims:
            raise ValueError(
                f"Cross-species error correlation requires observation-aligned {name!r} coordinates."
            )
    # A missing site or timestamp is not a known common site or time, so those
    # rows stay unpaired. Drop them before labels are stringified, which would
    # make two missing values equal.
    sites = np.asarray(observations.site.values, dtype=object)
    times = pd.to_datetime(np.asarray(observations.time.values))
    known = ~pd.isna(sites) & ~pd.isna(times)
    frame = pd.DataFrame(
        {
            "species": [str(value) for value in observations.species.values[known]],
            "site": [str(value) for value in sites[known]],
            "time": times[known],
            "row": np.flatnonzero(known),
        }
    )
    if frame[["species", "site", "time"]].duplicated().any():
        raise ValueError("Cross-species error correlation requires one row per species, site and time.")
    pairs = frame[frame.species == first].merge(
        frame[frame.species == second], on=["site", "time"], suffixes=("_first", "_second")
    )
    if pairs.empty:
        raise ValueError(
            f"Cross-species error correlation found no same-site, same-time {first}/{second} pairs."
        )
    if isinstance(correlation, Mapping):
        unknown = set(correlation) - set(frame.site)
        if unknown:
            raise ValueError(f"cross_species_error_correlation names unobserved sites: {sorted(unknown)}.")
        missing = set(pairs.site) - set(correlation)
        if missing:
            raise ValueError(f"cross_species_error_correlation lacks paired sites: {sorted(missing)}.")
        values = pairs.site.map(correlation).to_numpy(dtype=np.float64)
    else:
        values = np.full(len(pairs), correlation, dtype=np.float64)
    return PairedErrorCorrelation(
        first=pairs.row_first.to_numpy(dtype=np.int64),
        second=pairs.row_second.to_numpy(dtype=np.int64),
        correlation=values,
    )


__all__ = [
    "CrossSpeciesCorrelation",
    "linked_cross_species_pairs",
    "validate_cross_species_error_correlation",
]
