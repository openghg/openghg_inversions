"""Prepare labelled, backend-neutral boundary-condition sensitivities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.inversion_inputs import DatetimeLike, make_freq_indicator

__all__ = [
    "BoundaryAlignment",
    "scale_satellite_boundary_sensitivity",
]


@dataclass(frozen=True, eq=False, slots=True)
class BoundaryAlignment:
    """Resolved alignment of sampled boundary states to fitted periods.

    Args:
        data: Prepared ``H_bc`` with a ``bc_region`` MultiIndex whose levels
            are ``bc_curtain`` and ``bc_period``, plus one observation axis.

    Raises:
        ValueError: If dimensions or state labels are invalid.
    """

    data: xr.DataArray

    def __post_init__(self) -> None:
        """Enforce the invariants carried by the typed value."""
        data = self.data
        observation_dim = next((dim for dim in ("nmeasure", "time") if dim in data.dims), None)
        if observation_dim is None or set(data.dims) != {"bc_region", observation_dim}:
            raise ValueError(
                "Resolved boundary sensitivity must have exactly 'bc_region' and one observation dimension."
            )
        if "bc_region" not in data.coords or not data.indexes["bc_region"].is_unique:
            raise ValueError("Resolved boundary sensitivity requires unique labelled boundary states.")
        index = data.indexes["bc_region"]
        if list(index.names) != ["bc_curtain", "bc_period"]:
            raise ValueError(
                "Resolved boundary sensitivity states must be labelled by 'bc_curtain' and 'bc_period'."
            )
        object.__setattr__(self, "data", data.rename("H_bc"))

    @classmethod
    def prepare(
        cls,
        sensitivity: xr.DataArray,
        *,
        frequency: str | None = None,
        anchor_time: DatetimeLike | None = None,
        observation_labels: xr.DataArray | None = None,
    ) -> BoundaryAlignment:
        """Align and expand raw boundary sensitivity over fitted periods.

        Args:
            sensitivity: Boundary sensitivity with exactly ``bc_region`` and
                an observation dimension named ``time`` or ``nmeasure``.
            frequency: Optional fitted boundary-period frequency. ``None``
                creates one period; ``"monthly"`` follows calendar months.
            anchor_time: Optional anchor for fixed-duration periods.
            observation_labels: Optional authoritative observation coordinate;
                values are selected and reordered to these exact labels.

        Returns:
            Prepared boundary sensitivity carrying resolved state invariants.

        Raises:
            ValueError: If dimensions or observation labels are invalid.
        """
        observation_dim = next((dim for dim in ("time", "nmeasure") if dim in sensitivity.dims), None)
        if observation_dim is None or set(sensitivity.dims) != {"bc_region", observation_dim}:
            raise ValueError(
                "Boundary sensitivity must have exactly 'bc_region' and one observation "
                "dimension named 'time' or 'nmeasure'."
            )
        for dim in ("bc_region", observation_dim):
            if dim not in sensitivity.coords or sensitivity.coords[dim].dims != (dim,):
                raise ValueError(
                    f"Boundary sensitivity requires an explicit one-dimensional {dim!r} coordinate."
                )
            if not sensitivity.indexes[dim].is_unique:
                raise ValueError(f"Boundary sensitivity {dim!r} labels must be unique.")
        if observation_labels is not None:
            if observation_labels.dims != (observation_dim,):
                raise ValueError(
                    f"Authoritative observation labels must have exactly dimension {observation_dim!r}."
                )
            requested = observation_labels.values
            available = sensitivity.coords[observation_dim].values
            missing = requested[~np.isin(requested, available)]
            if missing.size:
                raise ValueError(f"Boundary sensitivity is missing observation label(s): {missing.tolist()!r}.")
            sensitivity = sensitivity.sel({observation_dim: requested})

        time = sensitivity.coords.get("time")
        if time is None or time.dims != (observation_dim,):
            raise ValueError("Boundary sensitivity requires an observation-aligned 'time' coordinate.")
        period = (
            make_freq_indicator(time, frequency, anchor_time=anchor_time)
            if frequency is not None
            else xr.zeros_like(time, dtype=int)
        ).rename("bc_period")
        mask = get_xr_dummies(period, return_sparse=False, cat_dim="bc_period")
        expanded = (sensitivity.rename(bc_region="bc_curtain") * mask).stack(
            bc_region=("bc_curtain", "bc_period")
        )
        data = expanded.transpose("bc_region", observation_dim).rename("H_bc")
        data.attrs = dict(sensitivity.attrs)
        data.attrs.update(
            {
                "boundary_sensitivity_preparation": "sampled-period-expansion",
                "bc_frequency": frequency or "none",
                "bc_anchor_time": "first-observation" if anchor_time is None else str(anchor_time),
            }
        )
        return cls(data)

def scale_satellite_boundary_sensitivity(
    inputs: xr.Dataset,
    *,
    sites: Sequence[str],
    platform: Sequence[str | None],
) -> xr.Dataset:
    """Scale satellite boundary sensitivity into corrected-column space.

    Args:
        inputs: Gathered inversion inputs containing ``H_bc`` and observation
            arrays. The dataset is borrowed and never mutated.
        sites: Site labels in the same order as ``platform``.
        platform: Platform values aligned to ``sites``.

    Returns:
        The unchanged borrowed dataset when scaling is inapplicable, otherwise
        a shallow copy whose satellite ``H_bc`` rows are explicitly scaled and
        carry transform provenance.
    """
    required = {"H_bc", "mf", "mf_prior_factor", "mf_prior_upper_level_factor", "site"}
    if not required <= set(inputs.variables):
        return inputs
    satellite_sites = [
        site
        for site, value in zip(sites, platform, strict=True)
        if value is not None and "satellite" in str(value).lower()
    ]
    if not satellite_sites:
        return inputs
    satellite_mask = inputs["site"].astype(str).isin(satellite_sites)
    if not bool(satellite_mask.any()):
        return inputs

    raw_column = inputs["mf"] + inputs["mf_prior_factor"] + inputs["mf_prior_upper_level_factor"]
    # Retain the released workaround until retrieval exposes the information
    # needed for an exact corrected-column transform.
    scale = xr.where(raw_column > 0, inputs["mf"] / raw_column, 1.0).clip(min=0.0, max=1.0)
    result = inputs.copy(deep=False)
    result["H_bc"] = inputs["H_bc"] * scale.where(satellite_mask, 1.0)
    result["H_bc"].attrs = dict(inputs["H_bc"].attrs)
    result["H_bc"].attrs["satellite_column_bc_scale"] = (
        "Applied to satellite rows using mf / (mf + mf_prior_factor + mf_prior_upper_level_factor)."
    )
    return result
