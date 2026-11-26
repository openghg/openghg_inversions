"""Fixed model-error correlation between two linked channels at the same site and time.

``rho`` correlates only the fixed model error, ``model_error_sd``: the
linked residual covariance is ``A + D_obs + Sigma_model(rho) [+ OU]``, with
aggregation covariance ``A`` and measurement variance ``D_obs`` unchanged (see
:mod:`openghg_inversions.models.paired_error`). The pairing is keyed by channel
(``species``) labels, so it is not specific to O2: any primary/secondary tracer
pair, in any units, uses the same rule.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.models.paired_error import (
    FIXED_MODEL_MISMATCH,
    PairedErrorCorrelation,
    pair_correlation_diagnostic,
)
from openghg_inversions.observation_error import AggregationError

CrossSpeciesCorrelation = float | Mapping[str, float]
ModelErrorSd = Mapping[str, float]


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


def validate_model_error_sd(value: object) -> dict[str, float]:
    """Check a model-error standard deviation mapping without observations.

    Keys are channels (``"co2"``, ``"o2"``) or ``"species:SITE"`` groups; each
    value is a finite, non-negative standard deviation in that channel's units.
    A scalar is rejected because the channels may use different units.
    """
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "model_error_sd must be a non-empty mapping from channel (or 'species:SITE') "
            "to a standard deviation in that channel's units."
        )
    result: dict[str, float] = {}
    for key, sd in value.items():
        name = f"model_error_sd[{key!r}]"
        if isinstance(sd, bool) or not isinstance(sd, int | float | np.integer | np.floating):
            raise ValueError(f"{name} must be a finite, non-negative number.")
        number = float(sd)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be a finite, non-negative number.")
        result[str(key)] = number
    return result


def linked_model_error_sd(observations: xr.DataArray, model_error_sd: ModelErrorSd) -> xr.DataArray:
    """Expand channel or ``species:SITE`` model-error standard deviations over observation rows.

    A ``species:SITE`` key overrides its channel's key. Every row must be
    covered, and every key must name an observed channel or species/site group.

    Returns:
        Observation-aligned ``fixed_model_mismatch`` in each row's channel units,
        with the observations' index and ``observation_units`` coordinates.
    """
    values = validate_model_error_sd(model_error_sd)
    for name in ("species", "site"):
        if name not in observations.coords or observations[name].dims != observations.dims:
            raise ValueError(f"model_error_sd requires observation-aligned {name!r} coordinates.")
    species = np.asarray([str(item) for item in observations.species.values], dtype=object)
    sites = np.asarray([str(item) for item in observations.site.values], dtype=object)
    groups = species + ":" + sites
    unknown = set(values) - set(species) - set(groups)
    if unknown:
        raise ValueError(f"model_error_sd names unobserved channels or groups: {sorted(unknown)}.")
    result = np.full(observations.size, np.nan, dtype=np.float64)
    for index, (channel, group) in enumerate(zip(species, groups, strict=True)):
        result[index] = values.get(group, values.get(channel, np.nan))
    missing = sorted(set(groups[np.isnan(result)]))
    if missing:
        raise ValueError(f"model_error_sd does not cover observed groups: {missing}.")
    # Keep the observation index and units, like independent_error_sd; site and time stay on the observations.
    auxiliary = [
        name for name in observations.coords if name not in observations.indexes and name != "observation_units"
    ]
    output = observations.copy(data=result).drop_vars(auxiliary).rename(FIXED_MODEL_MISMATCH)
    output.attrs = {"units": "see observation_units coordinate", "long_name": "fixed model-error standard deviation"}
    return output


def linked_cross_species_pairs(
    observations: xr.DataArray,
    correlation: CrossSpeciesCorrelation,
    *,
    channels: tuple[str, str] = ("co2", "o2"),
) -> PairedErrorCorrelation:
    """Pair each row of the first channel with the second channel's row at the same site and time.

    The correlation applies to the rows' model errors, so its cross
    covariance ``rho * m_first * m_second`` is in the product of the two
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


def linked_model_error_terms(
    observations: xr.DataArray,
    *,
    model_error_sd: ModelErrorSd | None,
    cross_species_error_correlation: CrossSpeciesCorrelation | None,
) -> tuple[xr.DataArray | None, PairedErrorCorrelation | None]:
    """Resolve the linked fixed model error and its CO2/O2 pairs.

    Returns:
        Observation-aligned ``fixed_model_mismatch`` (or ``None``) and the
        same-site, same-time pairs (or ``None``).

    Raises:
        ValueError: If a correlation is given without ``model_error_sd``, which
            it correlates, or either option is invalid for the observations.
    """
    if cross_species_error_correlation is not None and model_error_sd is None:
        raise ValueError(
            "cross_species_error_correlation acts only on the model error, so it requires "
            "model_error_sd; measurement (independent_error_sd) and aggregation errors stay "
            "uncorrelated."
        )
    model_error = None if model_error_sd is None else linked_model_error_sd(observations, model_error_sd)
    pairs = (
        None
        if cross_species_error_correlation is None
        else linked_cross_species_pairs(observations, cross_species_error_correlation)
    )
    return model_error, pairs


def _aggregation_cross_and_variance(
    aggregation_error: AggregationError | None, pairs: PairedErrorCorrelation, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A_pq`` per pair and the diagonal of ``A`` without forming a dense low-rank ``A``."""
    if aggregation_error is None:
        return np.zeros(pairs.first.size), np.zeros(size)
    variance = np.asarray(aggregation_error.marginal_variance, dtype=np.float64).reshape(size)
    if aggregation_error.mode == "dense":
        assert aggregation_error.covariance is not None
        covariance = np.asarray(aggregation_error.covariance.values, dtype=np.float64)
        return covariance[pairs.first, pairs.second], variance
    if aggregation_error.mode == "low_rank":
        assert aggregation_error.factor is not None
        factor = np.asarray(aggregation_error.factor.values, dtype=np.float64)
        return np.einsum("ij,ij->i", factor[pairs.first], factor[pairs.second]), variance
    return np.zeros(pairs.first.size), variance


def linked_pair_correlation_diagnostic(
    observations: xr.DataArray,
    *,
    pairs: PairedErrorCorrelation,
    aggregation_error: AggregationError | None,
    independent_error_sd: xr.DataArray,
    model_error_sd: xr.DataArray,
    ou_variance: xr.DataArray | np.ndarray | None = None,
) -> xr.Dataset:
    """Tabulate, pair by pair, the CO2/O2 correlation implied by each covariance part.

    ``aggregation_correlation`` is ``A_co / sqrt(A_cc A_oo)``, what the
    aggregation covariance alone implies; ``model_correlation`` is ``rho``; and
    ``total_correlation`` is the correlation of the whole covariance at zero
    lag, ``(A_co + rho m_c m_o) / sqrt((A_cc + e_c**2 + m_c**2 + a_c**2)
    (A_oo + e_o**2 + m_o**2 + a_o**2))``, with OU marginal variance ``a**2``
    from ``ou_variance`` when given. A ``rho`` estimated from total residuals
    should be compared with ``total_correlation``, not ``model_correlation``.

    Returns:
        A dataset on a ``pair`` dimension with the pairs' ``site``, ``time``
        and first/second row positions as coordinates.
    """
    size = observations.size
    cross, variance = _aggregation_cross_and_variance(aggregation_error, pairs, size)
    other = None if ou_variance is None else np.asarray(ou_variance, dtype=np.float64).reshape(size)
    values = pair_correlation_diagnostic(
        pairs,
        aggregation_cross=cross,
        aggregation_variance=variance,
        observation_sd=np.asarray(independent_error_sd.values, dtype=np.float64),
        model_error_sd=np.asarray(model_error_sd.values, dtype=np.float64),
        other_variance=other,
    )
    coords = {
        "site": ("pair", np.asarray(observations.site.values)[pairs.first]),
        "time": ("pair", np.asarray(observations.time.values)[pairs.first]),
        "first_row": ("pair", pairs.first),
        "second_row": ("pair", pairs.second),
    }
    return xr.Dataset(
        {
            "aggregation_correlation": ("pair", values["aggregation"]),
            "model_correlation": ("pair", values["model"]),
            "total_correlation": ("pair", values["total"]),
        },
        coords=coords,
    )


def summarise_pair_correlation(diagnostic: xr.Dataset) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarise a pair diagnostic by site as JSON-ready median, minimum, maximum and pair count."""
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for site in sorted({str(item) for item in diagnostic.site.values}):
        selected = diagnostic.where(diagnostic.site.astype(str) == site, drop=True)
        summary[site] = {}
        for name in ("aggregation_correlation", "total_correlation"):
            data = np.asarray(selected[name].values, dtype=np.float64)
            finite = data[np.isfinite(data)]
            summary[site][name.removesuffix("_correlation")] = (
                {"median": float(np.median(finite)), "min": float(finite.min()),
                 "max": float(finite.max()), "pairs": int(finite.size)}
                if finite.size else {"pairs": 0}
            )
    return summary


__all__ = [
    "CrossSpeciesCorrelation",
    "ModelErrorSd",
    "linked_cross_species_pairs",
    "linked_model_error_sd",
    "linked_model_error_terms",
    "linked_pair_correlation_diagnostic",
    "summarise_pair_correlation",
    "validate_cross_species_error_correlation",
    "validate_model_error_sd",
]
