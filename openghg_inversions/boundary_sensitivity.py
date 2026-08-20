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
    "scale_satellite_boundary_sensitivity_to_column_signal",
]


@dataclass(frozen=True, eq=False, slots=True)
class BoundaryAlignment:
    """Resolved alignment of sampled boundary states to fitted periods.

    Args:
        data: Prepared ``H_bc`` with a ``bc_region`` MultiIndex whose levels
            are ``bc_curtain`` and ``bc_period``, plus one observation axis.

    Raises:
        ValueError: If the resolved boundary-state labels are invalid.
    """

    data: xr.DataArray

    def __post_init__(self) -> None:
        """Enforce the invariants carried by the typed value."""
        index = self.data.indexes.get("bc_region")
        if index is None:
            raise ValueError("Resolved boundary sensitivity requires labelled boundary states.")
        if list(index.names) != ["bc_curtain", "bc_period"]:
            raise ValueError(
                "Resolved boundary sensitivity states must be labelled by 'bc_curtain' and 'bc_period'."
            )
        object.__setattr__(self, "data", self.data.rename("H_bc"))

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

        """
        observation_dim = "time" if "time" in sensitivity.dims else "nmeasure"
        sensitivity = sensitivity.transpose("bc_region", observation_dim)
        if observation_labels is not None:
            labels = observation_labels.transpose(observation_dim)
            sensitivity = sensitivity.sel({observation_dim: labels})

        time = sensitivity["time"].transpose(observation_dim)
        # Dummy construction below is eager, so materialize a lazy auxiliary
        # time coordinate once rather than executing its graph for each lookup.
        if hasattr(time.data, "__dask_graph__"):
            time = time.compute()
            sensitivity = sensitivity.assign_coords(time=(observation_dim, time.data))
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
        effective_anchor = (
            str(np.datetime64(anchor_time) if anchor_time is not None else time.values.min())
            if frequency not in (None, "monthly")
            else "not-applicable"
        )
        data.attrs.update(
            {
                "boundary_sensitivity_preparation": "sampled-period-expansion",
                "bc_frequency": frequency or "none",
                "bc_anchor_time": effective_anchor,
            }
        )
        return cls(data)


def scale_satellite_boundary_sensitivity_to_column_signal(
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
