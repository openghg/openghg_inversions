"""Prepare labelled, backend-neutral boundary-condition sensitivities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.inversion_inputs import DatetimeLike, make_freq_indicator

__all__ = [
    "BoundaryAlignment",
    "validate_boundary_operands",
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
        if frequency is not None:
            # Dummy construction below is eager, so materialize a lazy auxiliary
            # time coordinate once rather than executing its graph for each lookup.
            if hasattr(time.data, "__dask_graph__"):
                time = time.compute()
                sensitivity = sensitivity.assign_coords(time=(observation_dim, time.data))
            period = make_freq_indicator(time, frequency, anchor_time=anchor_time)
        else:
            period = xr.zeros_like(time, dtype=int)
        period = period.rename("bc_period")
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


def validate_boundary_operands(operands: xr.Dataset) -> bool:
    """Validate raw spatial centered operands before model-period expansion.

    Args:
        operands: Borrowed dataset with ``boundary_representation='centered-v1'``,
            ``H_bc_anomaly`` and ``H_bc_mass`` on unique indexed ``bc_region``
            and ``time`` or ``nmeasure`` dimensions, and scalar ``bc_reference``.
            Anomaly and reference share the exact concentration unit string;
            mass has units ``'1'``. Optional ``mf`` has matching units. An
            absolute ``H_bc`` diagnostic must be marked ``unexpanded``.

    Returns:
        Whether centered operands are present. Accepted operands remain raw:
        :func:`align_boundary_operands` expands them once at model binding.
        Timestamps are not required for a single period. Finite payload values
        are checked at the shared materialization boundary; this validator does
        not compute or mutate borrowed numerical arrays.

    Raises:
        ValueError: If centered operands are incomplete, already expanded, or
            have incompatible representation, dimensions, labels or units.
    """
    names = {"H_bc_anomaly", "H_bc_mass", "bc_reference"}
    representation = operands.attrs.get("boundary_representation")
    if representation is not None and representation != "centered-v1":
        raise ValueError(f"Unsupported boundary_representation {representation!r}.")
    centered = bool(names.intersection(operands.variables)) or operands.attrs.get(
        "boundary_representation"
    ) == "centered-v1"
    if not centered:
        return False
    if not names <= set(operands.variables):
        raise ValueError("Centered boundaries require H_bc_anomaly, H_bc_mass and bc_reference.")
    if operands.attrs.get("boundary_representation") != "centered-v1":
        raise ValueError("Centered boundaries require boundary_representation='centered-v1'.")
    if "H_bc" in operands and operands.H_bc.attrs.get("boundary_sensitivity_preparation") != "unexpanded":
        raise ValueError("Centered boundaries cannot include expanded or unmarked absolute H_bc.")
    anomaly, mass, reference = (operands[name] for name in ("H_bc_anomaly", "H_bc_mass", "bc_reference"))
    if any(
        value.attrs.get("boundary_sensitivity_preparation") not in (None, "unexpanded")
        or isinstance(value.indexes.get("bc_region"), pd.MultiIndex)
        for value in (anomaly, mass)
    ):
        raise ValueError("Centered preparation requires unexpanded spatial operands; align only at model binding.")
    observation_dim = "time" if "time" in anomaly.dims else "nmeasure"
    anomaly = anomaly.transpose("bc_region", observation_dim)
    mass = mass.transpose("bc_region", observation_dim)
    xr.align(anomaly, mass, join="exact", copy=False)
    for dim in ("bc_region", observation_dim):
        if dim not in anomaly.indexes or not anomaly.indexes[dim].is_unique:
            raise ValueError(f"Centered boundaries require unique indexed {dim} labels.")
    if reference.ndim != 0:
        raise ValueError("Centered bc_reference must be scalar.")
    units = anomaly.attrs.get("units")
    if not units or reference.attrs.get("units") != units or mass.attrs.get("units") != "1":
        raise ValueError("Centered anomaly/reference must share concentration units; mass units must be '1'.")
    if "mf" in operands and operands.mf.attrs.get("units") != units:
        raise ValueError("Centered boundary concentration units must match mf units.")
    return True


def scale_satellite_boundary_sensitivity_to_column_signal(
    inputs: xr.Dataset,
    *,
    sites: Sequence[str],
    platform: Sequence[str | None],
    observation_max_level: Sequence[int | None] | None = None,
    footprint_max_level: Sequence[int | None] | None = None,
) -> xr.Dataset:
    """Scale satellite boundary sensitivity for unmatched vertical extents.

    Args:
        inputs: Gathered inversion inputs containing ``H_bc`` and observation
            arrays. The dataset is borrowed and never mutated.
        sites: Site labels in the same order as ``platform``.
        platform: Platform values aligned to ``sites``.
        observation_max_level: Optional column-processing levels aligned to
            ``sites``.
        footprint_max_level: Optional verified footprint extents aligned to
            ``sites``.

    Returns:
        The unchanged borrowed dataset when scaling is inapplicable or its
        verified footprint and observation levels agree. Otherwise, a shallow
        copy whose satellite ``H_bc`` rows are explicitly scaled and carry
        transform provenance.
    """
    required = {"H_bc", "mf", "mf_prior_factor", "mf_prior_upper_level_factor", "site"}
    if not required <= set(inputs.variables):
        return inputs
    if observation_max_level is None or footprint_max_level is None:
        observation_max_level = footprint_max_level = (None,) * len(sites)
    inconsistent_satellite_sites = [
        site
        for site, value, observation_level, footprint_level in zip(
            sites,
            platform,
            observation_max_level,
            footprint_max_level,
            strict=True,
        )
        if value is not None
        and "satellite" in str(value).lower()
        and not (
            observation_level is not None
            and footprint_level is not None
            and observation_level == footprint_level
        )
    ]
    if not inconsistent_satellite_sites:
        return inputs
    scale_mask = inputs["site"].astype(str).isin(inconsistent_satellite_sites)
    if not bool(scale_mask.any()):
        return inputs

    raw_column = inputs["mf"] + inputs["mf_prior_factor"] + inputs["mf_prior_upper_level_factor"]
    # Retain the released workaround when provenance is absent or shows that
    # the footprint and observation use different vertical extents.
    scale = xr.where(raw_column > 0, inputs["mf"] / raw_column, 1.0).clip(min=0.0, max=1.0)
    result = inputs.copy(deep=False)
    result["H_bc"] = inputs["H_bc"] * scale.where(scale_mask, 1.0)
    result["H_bc"].attrs = dict(inputs["H_bc"].attrs)
    result["H_bc"].attrs["satellite_column_bc_scale"] = (
        "Applied to satellite rows with missing or inconsistent footprint max_level provenance using "
        "mf / (mf + mf_prior_factor + mf_prior_upper_level_factor)."
    )
    return result
