"""Labelled species/site grouping for the linked fixed-OU likelihood."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.sigma import SigmaAlignment


def linked_fixed_ou_alignment(observations: xr.DataArray) -> SigmaAlignment:
    """Group joint rows by ``species:site``, preserving their original order.

    Observations must have a labelled ``observation`` axis, aligned ``species``,
    ``site`` and ``time`` coordinates, and one common ``observation_units``
    value. This first linked OU target excludes heterogeneous channel units.
    Scalar amplitudes and tau apply independently to each group; mappings use
    the explicit ``co2:SITE`` and ``o2:SITE`` labels.
    """
    for name in ("species", "site", "time", "observation_units"):
        if name not in observations.coords or observations[name].dims != observations.dims:
            raise ValueError(f"Linked fixed-OU requires observation-aligned {name!r} coordinates.")
    if np.unique(observations.observation_units.values.astype(str)).size != 1:
        raise ValueError("Linked fixed-OU requires the same units for CO2 and O2 (OPE-86).")
    if np.any(pd.isna(observations.site.values)):
        raise ValueError("Linked fixed-OU requires complete site labels.")
    species = observations.species.values.astype(str)
    if not set(species) <= {"co2", "o2"}:
        raise ValueError("Linked fixed-OU species must be 'co2' or 'o2'.")
    groups = np.char.add(np.char.add(species, ":"), observations.site.values.astype(str))
    positions, labels = pd.factorize(groups, sort=False)
    # Keep native MultiIndex levels intact while grouping by species and site.
    site_index = observations.copy(data=positions, deep=False)
    return SigmaAlignment.from_indices(
        site_index,
        xr.zeros_like(site_index, dtype=int),
        site_labels=labels,
    )
