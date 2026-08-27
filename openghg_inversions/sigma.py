"""Represent observation-to-sigma alignment independently of model backends.

The canonical site and period indexes are eager, non-negative integer vectors
on ``nmeasure``. They can be prepared once and consumed by PyMC or another
inversion backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.inversion_inputs import DatetimeLike, make_sigma_freq, make_site_indicator

_OBS_DIM = "nmeasure"
_SITE_INDEX_NAME = "sigma_site_index"
_PERIOD_INDEX_NAME = "sigma_period_index"


def _normalise_index(index: xr.DataArray, name: str) -> xr.DataArray:
    """Return an eager, non-negative integer index with a canonical name.

    Args:
        index: Observation-aligned index values.
        name: Canonical output name and diagnostic label.

    Returns:
        Copied eager integer index with ``name``.

    Raises:
        TypeError: If ``index`` is not an xarray DataArray.
        ValueError: If it is empty, misaligned, non-integral, or outside the
            supported non-negative integer range.
    """
    if not isinstance(index, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray.")
    if index.dims != (_OBS_DIM,) or index.size == 0:
        raise ValueError(f"{name} must be a non-empty vector on {_OBS_DIM!r}.")

    values = np.asarray(index.values)
    is_numeric = values.dtype.kind in "iuf"
    if values.dtype.kind == "b" or not is_numeric:
        raise ValueError(f"{name} must contain integer values.")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{name} must contain finite integer values.")
    if np.any(values < 0) or np.any(values > np.iinfo(np.int_).max):
        raise ValueError(f"{name} must contain non-negative integers in the supported range.")
    return index.copy(data=values.astype(int, copy=True)).rename(name)


def _time_coord(index: xr.DataArray) -> xr.DataArray | None:
    """Return an observation-aligned time coordinate, when present."""
    time = index.coords.get("time")
    if time is not None and time.dims == (_OBS_DIM,):
        return time

    obs_index = index.indexes.get(_OBS_DIM)
    if isinstance(obs_index, pd.MultiIndex) and "time" in obs_index.names:
        return xr.DataArray(
            obs_index.get_level_values("time").to_numpy(),
            dims=(_OBS_DIM,),
            coords={_OBS_DIM: index.coords[_OBS_DIM]},
            name="time",
        )
    return None


@dataclass(frozen=True, eq=False, slots=True)
class SigmaAlignment:
    """Backend-neutral indexes mapping latent sigma values to observations.

    Args:
        site_index: Observation-aligned site positions.
        period_index: Observation-aligned sigma-period positions.

    Raises:
        TypeError: If an index is not an xarray DataArray.
        ValueError: If indexes are invalid or have incompatible coordinates.
    """

    site_index: xr.DataArray
    period_index: xr.DataArray

    def __post_init__(self) -> None:
        """Normalise indexes and enforce their shared observation alignment."""
        site = _normalise_index(self.site_index, _SITE_INDEX_NAME)
        period = _normalise_index(self.period_index, _PERIOD_INDEX_NAME)
        if site.sizes[_OBS_DIM] != period.sizes[_OBS_DIM]:
            raise ValueError("Sigma site and period indexes must have the same observation length.")

        site_coords = {name: coord for name, coord in site.coords.items() if coord.dims == (_OBS_DIM,)}
        period_coords = {
            name: coord for name, coord in period.coords.items() if coord.dims == (_OBS_DIM,)
        }
        for name in site_coords.keys() & period_coords.keys():
            if not site_coords[name].identical(period_coords[name]):
                raise ValueError(f"Sigma indexes have incompatible coordinate {name!r}.")

        coords = {**period_coords, **site_coords}
        object.__setattr__(self, "site_index", site.assign_coords(coords))
        object.__setattr__(self, "period_index", period.assign_coords(coords))

    @classmethod
    def from_frequency(
        cls,
        site_indicator: xr.DataArray,
        frequency: str | None = None,
        *,
        per_site: bool = True,
        anchor_time: DatetimeLike | None = None,
    ) -> SigmaAlignment:
        """Derive sigma alignment from site positions and observation times.

        Args:
            site_indicator: Observation-aligned site positions.
            frequency: Sigma period frequency. ``None`` creates one period.
            per_site: Whether sigma varies by site.
            anchor_time: Optional fixed-duration period anchor.

        Returns:
            Canonical sigma alignment.

        Raises:
            ValueError: If indexes or observation timestamps are invalid.
        """
        site = _normalise_index(site_indicator, _SITE_INDEX_NAME)
        if frequency is None:
            period = xr.zeros_like(site, dtype=int)
        else:
            time = _time_coord(site)
            if time is None or np.any(pd.isna(np.asarray(time.values))):
                raise ValueError("Sigma frequencies require complete observation timestamps.")
            period = make_sigma_freq(time, freq=frequency, anchor_time=anchor_time)
        return cls.from_indices(site, period, per_site=per_site)

    @classmethod
    def from_observations(
        cls,
        observations: xr.DataArray,
        frequency: str | None = None,
        *,
        per_site: bool = True,
        anchor_time: DatetimeLike | None = None,
    ) -> SigmaAlignment:
        """Derive sigma alignment from observation site and time coordinates.

        Args:
            observations: Observation vector with an aligned ``site``
                coordinate and, when ``frequency`` is set, observation times.
            frequency: Sigma period frequency. ``None`` creates one period.
            per_site: Whether sigma varies by site.
            anchor_time: Optional fixed-duration period anchor.

        Returns:
            Canonical sigma alignment.

        Raises:
            ValueError: If the required observation coordinates are absent or
                invalid.
        """
        site = observations.coords.get("site")
        if site is None or site.dims != (_OBS_DIM,):
            raise ValueError(
                "Sigma alignment requires an observation-aligned 'site' coordinate."
            )
        return cls.from_frequency(
            make_site_indicator(site),
            frequency=frequency,
            per_site=per_site,
            anchor_time=anchor_time,
        )

    @classmethod
    def from_indices(
        cls,
        site_index: xr.DataArray,
        period_index: xr.DataArray,
        *,
        per_site: bool = True,
    ) -> SigmaAlignment:
        """Build sigma alignment from explicit observation indexes.

        Args:
            site_index: Observation-aligned site positions.
            period_index: Observation-aligned sigma-period positions.
            per_site: Whether sigma varies by site.

        Returns:
            Canonical sigma alignment.

        Raises:
            TypeError: If an index is not an xarray DataArray.
            ValueError: If indexes are invalid or incompatible.
        """
        alignment = cls(site_index, period_index)
        if per_site:
            return alignment
        return cls(xr.zeros_like(alignment.site_index, dtype=int), alignment.period_index)

    @classmethod
    def from_model_data(cls, model_data: xr.Dataset) -> SigmaAlignment:
        """Restore alignment from canonical registered model data.

        Args:
            model_data: Dataset containing ``sigma_site_index`` and
                ``sigma_period_index``.

        Returns:
            Canonical sigma alignment.

        Raises:
            KeyError: If a required index variable is absent.
            ValueError: If stored indexes are invalid or incompatible.
        """
        return cls(
            model_data[_SITE_INDEX_NAME],
            model_data[_PERIOD_INDEX_NAME],
        )

    @property
    def nsite(self) -> int:
        """Number of latent site positions."""
        return int(self.site_index.max().item()) + 1

    @property
    def nperiod(self) -> int:
        """Number of latent period positions."""
        return int(self.period_index.max().item()) + 1

    def align(self, sigma: xr.DataArray) -> xr.DataArray:
        """Index latent sigma values onto observations.

        Args:
            sigma: Values with ``nsigma_site`` and ``nsigma_time`` dimensions.

        Returns:
            ``sigma_aligned`` with latent dimensions replaced by ``nmeasure``.

        Raises:
            ValueError: If a required latent dimension is absent.
            IndexError: If an alignment position is outside the latent array.
        """
        return sigma.isel(
            nsigma_site=self.site_index,
            nsigma_time=self.period_index,
        ).rename("sigma_aligned")
