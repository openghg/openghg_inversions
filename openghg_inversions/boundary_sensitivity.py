"""Prepare labelled, backend-neutral boundary-condition sensitivities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.inversion_inputs import DatetimeLike, make_period_calendar, normalise_frequency

BoundaryRepresentation = Literal["ordinary", "centered"]
"""Prepared boundary sensitivity: absolute ``H_bc`` or centered anomaly, mass and reference."""

__all__ = [
    "BoundaryAlignment",
    "BoundaryRepresentation",
    "bind_period_states",
    "validate_boundary_operands",
    "scale_satellite_boundary_sensitivity_to_column_signal",
]


@dataclass(frozen=True, eq=False, slots=True)
class BoundaryAlignment:
    """Resolved alignment of sampled boundary states to fitted periods.

    Args:
        data: Prepared ``H_bc`` with a ``bc_region`` MultiIndex whose levels
            are ``bc_curtain`` and interval-start ``bc_period``, plus one
            observation axis.

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
                an observation dimension named ``time`` or ``nmeasure``, with a
                datetime ``time`` coordinate along that dimension.
            frequency: Optional fitted boundary-period frequency. ``None``
                creates one period; ``"monthly"`` follows calendar months.
            anchor_time: Optional anchor for fixed-duration periods.
            observation_labels: Optional authoritative observation coordinate;
                values are selected and reordered to these exact labels.

        Returns:
            Prepared boundary sensitivity carrying resolved state invariants.

        """
        observation_dim = "time" if "time" in sensitivity.dims else "nmeasure"
        raw = sensitivity.rename("H_bc").to_dataset()
        return cls(_expand_boundary_periods(
            raw, observation_dim=observation_dim, frequency=frequency,
            anchor_time=anchor_time, observation_labels=observation_labels,
        ).H_bc)


def _expand_boundary_periods(
    raw: xr.Dataset,
    *,
    observation_dim: str,
    frequency: str | None,
    anchor_time: DatetimeLike | None,
    observation_labels: xr.DataArray | None,
) -> xr.Dataset:
    """Select scientific rows and apply one period mask to all boundary operands.

    The required datetime ``time`` coordinate along the observations is
    computed once if lazy, because period labels are interval-start timestamps
    (the first time when ``frequency`` is None).
    """
    normalise_frequency(frequency)
    if observation_labels is not None:
        raw = raw.sel({observation_dim: observation_labels.transpose(observation_dim)})
    if "time" not in raw.coords or raw.time.dims != (observation_dim,) or raw.time.dtype.kind != "M":
        raise ValueError("Boundary period expansion requires a datetime time coordinate along observations.")
    time = raw.time
    if time.chunks is not None:
        time = time.compute()
        raw = raw.assign_coords(time=(observation_dim, time.data))
    _, period = make_period_calendar(time, frequency, anchor_time=anchor_time)
    mask = get_xr_dummies(
        period, categories=np.unique(period.data), return_sparse=False, cat_dim="bc_period"
    )
    result = (raw.rename(bc_region="bc_curtain") * mask).stack(
        bc_region=("bc_curtain", "bc_period")
    ).transpose("bc_region", observation_dim)
    result.attrs = dict(raw.attrs)
    for name in raw.data_vars:
        result[name].attrs = dict(raw[name].attrs)
        result[name].attrs.update(
            boundary_sensitivity_preparation="sampled-period-expansion",
            bc_frequency=frequency or "none",
            bc_anchor_time=(
                str(np.datetime64(anchor_time) if anchor_time is not None else time.values.min())
                if frequency not in (None, "monthly")
                else "not-applicable"
            ),
        )
    return result


def _select_native_boundary(
    inputs: xr.Dataset,
    observations: xr.DataArray,
    observation_dim: str,
) -> xr.Dataset:
    """Select rows, borrow native timestamps, and check indexed-time identity.

    Indexed boundary times cannot be replaced independently of observation
    labels. This named model-binding check computes only native timestamp
    coordinates if necessary, leaving numerical boundary payloads borrowed.
    """
    if observation_dim not in inputs.dims:
        return inputs  # The operand validator owns malformed layouts.
    native_dim = str(observations.dims[0])
    labels = xr.DataArray(observations[native_dim].data, dims=observation_dim)
    selected = inputs.sel({observation_dim: labels})
    time = observations.coords.get("time")
    if time is not None and time.dims == (native_dim,):
        if "time" in selected.indexes:
            if time.chunks is not None:
                time = time.compute()
            if not np.array_equal(selected.time.data, time.data):
                raise ValueError("Indexed boundary time labels conflict with native observation timestamps.")
        else:
            selected = selected.assign_coords(time=(observation_dim, time.data))
    return selected


def bind_period_states(
    raw: xr.Dataset,
    observations: xr.DataArray,
    *,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
) -> xr.Dataset:
    """Select one fit's rows of raw spatial boundary data and expand its states by period.

    Observation labels own the row order and their auxiliary timestamps own
    period membership; indexed boundary times must match those timestamps.
    This check may compute a lazy timestamp coordinate, while numerical
    payloads stay borrowed.

    Args:
        raw: Raw spatial boundary data on ``bc_region`` and the ``time``,
            ``nmeasure`` or native observation axis.
        observations: One fit's observations with the observation axis first.
        frequency: ``None`` for one period, ``"monthly"``, or a positive fixed
            duration.
        anchor_time: Optional start of the fixed-duration period grid.

    Returns:
        ``raw``'s variables on ``(observation, bc_region)``, where
        ``bc_region`` is the ``(bc_curtain, bc_period)`` MultiIndex.

    Raises:
        ValueError: If the period settings are invalid, a frequency needs a
            datetime ``time`` coordinate that is missing, or indexed boundary
            times conflict with the observation timestamps.
        KeyError: If an observation label is absent from ``raw``.
    """
    native_dim = str(observations.dims[0])
    raw_dim = native_dim if native_dim in raw.dims else ("time" if "time" in raw.dims else "nmeasure")
    expanded = _expand_boundary_periods(
        _select_native_boundary(raw, observations, raw_dim), observation_dim=raw_dim,
        frequency=frequency, anchor_time=anchor_time, observation_labels=None,
    )
    return _restore_native_dim(expanded, raw_dim, native_dim).transpose(native_dim, "bc_region")


def _restore_native_dim(data: xr.Dataset, raw_dim: str, native_dim: str) -> xr.Dataset:
    """Rename selected rows to the native observation dimension."""
    if raw_dim == native_dim:
        return data
    # Native time has already determined periods; the restored dimension
    # index owns those same labels, so its auxiliary copy is redundant.
    if native_dim == "time" and "time" in data.coords:
        data = data.drop_vars("time")
    return data.rename({raw_dim: native_dim})


def bind_boundary_sensitivity(
    sensitivity: xr.DataArray,
    observations: xr.DataArray,
    *,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
) -> xr.DataArray:
    """Select an ordinary boundary and expand raw spatial states at model binding.

    Rows and periods follow :func:`bind_period_states`. Raw sensitivities can
    be reused with different frequencies; already expanded sensitivities may
    be consumed only without a period override. Direct callers may supply an
    independently named state axis. Unmarked direct inputs retain their state
    labels when no temporal expansion is requested; marked ``raw-spatial``
    preparations and explicit frequency choices produce the canonical
    ``bc_region`` curtain/period index. Payloads stay borrowed and lazy until
    the model's explicit materialization boundary.
    """
    validate_centered_boundary_periods(frequency, anchor_time)
    native_dim = str(observations.dims[0])
    raw_dim = native_dim if native_dim in sensitivity.dims else next(
        str(dim) for dim in sensitivity.dims if dim in ("nmeasure", "time")
    )
    state_dim = next(str(dim) for dim in sensitivity.dims if dim != raw_dim)
    data = sensitivity.rename("H_bc").to_dataset()
    expanded = (
        sensitivity.attrs.get("boundary_sensitivity_preparation") == "sampled-period-expansion"
        or "bc_period" in sensitivity.coords
        or isinstance(sensitivity.indexes.get(state_dim), pd.MultiIndex)
    )
    if expanded and (frequency is not None or anchor_time is not None):
        raise ValueError("Boundary period overrides require raw spatial sensitivity; regenerate baked inputs.")
    if not expanded and (frequency is not None or anchor_time is not None
                         or sensitivity.attrs.get("boundary_sensitivity_preparation") == "raw-spatial"):
        # Prepared raw-spatial arrays explicitly opt into the reusable
        # curtain/period contract, including one period.
        data = data.rename({state_dim: "bc_region"}) if state_dim != "bc_region" else data
        return bind_period_states(data, observations, frequency=frequency, anchor_time=anchor_time).H_bc
    # Expanded inputs, and direct scientific callers requesting no expansion,
    # retain state labels shared with their priors.
    selected = _select_native_boundary(data, observations, raw_dim)
    return _restore_native_dim(selected, raw_dim, native_dim).H_bc.transpose(native_dim, state_dim)


def validate_boundary_operands(operands: xr.Dataset) -> bool:
    """Detect raw centered operands and check their structure before period expansion.

    Centered sensitivity is present when any of ``H_bc_anomaly``, ``H_bc_mass``
    or ``bc_reference`` exists or ``boundary_representation`` is
    ``'centered-v1'``. Present operands must then satisfy
    :class:`~openghg_inversions.models.boundary.contracts.RawCenteredSensitivity`,
    whose docstring lists the structural guarantees.

    Args:
        operands: Borrowed dataset, possibly Dask-backed.

    Returns:
        Whether centered operands are present. Accepted operands remain raw
        for period expansion at model binding. Finite payload values are
        checked at the shared materialization boundary; this does not compute
        or mutate borrowed numerical arrays.

    Raises:
        ValueError: If ``boundary_representation`` is unsupported, or centered
            operands violate the raw centered sensitivity contract.
    """
    from openghg_inversions.models.boundary.contracts import CENTERED_NAMES, RawCenteredSensitivity

    representation = operands.attrs.get("boundary_representation")
    if representation is not None and representation != "centered-v1":
        raise ValueError(f"Unsupported boundary_representation {representation!r}.")
    if not set(CENTERED_NAMES).intersection(operands.variables) and representation is None:
        return False
    RawCenteredSensitivity.validate(operands)
    return True


def validate_centered_boundary_periods(frequency: str | None, anchor_time: DatetimeLike | None) -> None:
    """Reject invalid or ignored period settings at each external binding boundary."""
    normalise_frequency(frequency)
    if anchor_time is not None:
        if frequency in (None, "monthly"):
            raise ValueError("Boundary anchor_time requires a fixed-duration frequency.")
        try:
            if np.isnat(np.datetime64(anchor_time)):
                raise ValueError("NaT is not an anchor")
        except (TypeError, ValueError) as exc:
            raise ValueError("Boundary anchor_time must be a valid datetime.") from exc


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
