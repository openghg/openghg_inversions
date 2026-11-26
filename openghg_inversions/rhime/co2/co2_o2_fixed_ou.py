"""Labelled species/site grouping for the linked fixed-OU likelihood."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.sigma import SigmaAlignment


def linked_fixed_ou_alignment(observations: xr.DataArray) -> SigmaAlignment:
    """Group joint rows by ``species:site``, preserving their original order.

    Observations must have a labelled ``observation`` axis and aligned
    ``species``, ``site``, ``time`` and ``observation_units`` coordinates. Each
    group has one unit, so CO2 and O2 may use different units (for example
    ppm and per meg); OU amplitudes are then in each group's channel units.
    Scalar amplitudes and tau apply independently to each group; mappings use
    the explicit ``co2:SITE`` and ``o2:SITE`` labels. See
    :func:`validate_linked_ou_amplitude_units` for the amplitude rules with
    mixed units.
    """
    for name in ("species", "site", "time", "observation_units"):
        if name not in observations.coords or observations[name].dims != observations.dims:
            raise ValueError(f"Linked fixed-OU requires observation-aligned {name!r} coordinates.")
    if np.any(pd.isna(observations.site.values)):
        raise ValueError("Linked fixed-OU requires complete site labels.")
    species = observations.species.values.astype(str)
    if not set(species) <= {"co2", "o2"}:
        raise ValueError("Linked fixed-OU species must be 'co2' or 'o2'.")
    groups = np.char.add(np.char.add(species, ":"), observations.site.values.astype(str))
    positions, labels = pd.factorize(groups, sort=False)
    units = np.asarray([str(unit) for unit in observations.observation_units.values])
    for index, label in enumerate(labels):
        if np.unique(units[positions == index]).size != 1:
            raise ValueError(f"Linked fixed-OU group {label!r} mixes observation units.")
    # Keep native MultiIndex levels intact while grouping by species and site.
    site_index = observations.copy(data=positions, deep=False)
    return SigmaAlignment.from_indices(
        site_index,
        xr.zeros_like(site_index, dtype=int),
        site_labels=labels,
    )


def linked_channel_units(observations: xr.DataArray) -> dict[str, str]:
    """Return the unit label of each channel present on the joint observation axis."""
    species = np.asarray([str(value) for value in observations.species.values])
    units = np.asarray([str(value) for value in observations.observation_units.values])
    result = {}
    for channel in dict.fromkeys(species):
        labels = np.unique(units[species == channel])
        if labels.size != 1:
            raise ValueError(f"Linked {channel} observations require one unit label.")
        result[channel] = str(labels[0])
    return result


def validate_linked_ou_amplitude_units(
    observations: xr.DataArray,
    *,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> None:
    """Require OU amplitudes that state their units when CO2 and O2 units differ.

    With one common unit nothing changes. With mixed units (for example ppm
    CO2 and per-meg O2), a nonzero scalar amplitude would mean different
    things on each channel, so fixed amplitudes must be a ``co2:SITE``/
    ``o2:SITE`` mapping, each value in its own channel's units. One inferred
    amplitude prior cannot hold two units, so inferred amplitudes are refused.

    Raises:
        ValueError: If channel units differ and amplitudes are a nonzero scalar
            or inferred.
    """
    if len(set(linked_channel_units(observations).values())) <= 1:
        return
    if site_amplitude_prior is not None:
        raise ValueError(
            "Inferred linked OU amplitudes share one prior, which cannot hold mixed CO2/O2 units; "
            "use fixed_site_amplitudes keyed by 'co2:SITE'/'o2:SITE'."
        )
    if fixed_site_amplitudes is not None and not isinstance(fixed_site_amplitudes, Mapping):
        if float(fixed_site_amplitudes) != 0.0:
            raise ValueError(
                "With mixed CO2/O2 units, fixed_site_amplitudes must map 'co2:SITE'/'o2:SITE' "
                "to amplitudes in each channel's units."
            )


__all__ = ["linked_fixed_ou_alignment", "linked_channel_units", "validate_linked_ou_amplitude_units"]
