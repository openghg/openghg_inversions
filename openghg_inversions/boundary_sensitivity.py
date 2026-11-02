"""Prepare labelled, backend-neutral boundary-condition sensitivities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.inversion_inputs import DatetimeLike, make_freq_indicator, normalise_frequency

__all__ = [
    "BoundaryAlignment",
    "align_boundary_operands",
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
    """Select scientific rows and apply one period mask to all boundary operands."""
    normalise_frequency(frequency)
    if observation_labels is not None:
        raw = raw.sel({observation_dim: observation_labels.transpose(observation_dim)})
    if frequency is None:
        period = xr.zeros_like(raw[observation_dim], dtype=int)
    else:
        if "time" not in raw.coords or raw.time.dims != (observation_dim,) or raw.time.dtype.kind != "M":
            raise ValueError("Frequency-based boundary expansion requires a datetime time coordinate along observations.")
        time = raw.time
        if hasattr(time.data, "__dask_graph__"):
            time = time.compute()
            raw = raw.assign_coords(time=(observation_dim, time.data))
        period = make_freq_indicator(time, frequency, anchor_time=anchor_time)
    mask = get_xr_dummies(period.rename("bc_period"), return_sparse=False, cat_dim="bc_period")
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
                if frequency not in (None, "monthly") else "not-applicable"
            ),
        )
    return result


def _select_native_boundary(
    inputs: xr.Dataset, observations: xr.DataArray, observation_dim: str,
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
            if hasattr(time.data, "__dask_graph__"):
                time = time.compute()
            if not np.array_equal(selected.time.data, time.data):
                raise ValueError("Indexed boundary time labels conflict with native observation timestamps.")
        else:
            selected = selected.assign_coords(time=(observation_dim, time.data))
    return selected


def bind_boundary_sensitivity(
    sensitivity: xr.DataArray,
    observations: xr.DataArray,
    *,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
) -> xr.DataArray:
    """Select an ordinary boundary and expand raw spatial states at model binding.

    Observation labels own the row order and their auxiliary timestamps own
    period membership when supplied. Indexed boundary times must match those
    timestamps; this binding check may materialize a lazy timestamp coordinate,
    while boundary payloads remain borrowed. Raw sensitivities can be reused with
    different frequencies; already expanded sensitivities may be consumed only
    without a period override. Direct callers may supply an independently named
    state axis. Unmarked direct inputs retain their state labels when no temporal
    expansion is requested; marked ``raw-spatial`` preparations and explicit
    frequency choices produce the canonical ``bc_region`` curtain/period index.
    Payloads stay borrowed and lazy until the model's explicit materialization
    boundary.
    """
    validate_centered_boundary_periods(frequency, anchor_time)
    native_dim = str(observations.dims[0])
    raw_dim = native_dim if native_dim in sensitivity.dims else next(
        str(dim) for dim in sensitivity.dims if dim in ("nmeasure", "time")
    )
    state_dim = next(str(dim) for dim in sensitivity.dims if dim != raw_dim)
    sensitivity = _select_native_boundary(
        sensitivity.rename("H_bc").to_dataset(), observations, raw_dim,
    ).H_bc
    expanded = (
        sensitivity.attrs.get("boundary_sensitivity_preparation") == "sampled-period-expansion"
        or "bc_period" in sensitivity.coords
        or isinstance(sensitivity.indexes.get(state_dim), pd.MultiIndex)
    )
    if expanded:
        if frequency is not None or anchor_time is not None:
            raise ValueError("Boundary period overrides require raw spatial sensitivity; regenerate baked inputs.")
        aligned = sensitivity
    elif (frequency is None and anchor_time is None
          and sensitivity.attrs.get("boundary_sensitivity_preparation") != "raw-spatial"):
        # Direct scientific callers retain state labels shared with their priors
        # when requesting no expansion. Prepared raw-spatial arrays explicitly
        # opt into the reusable curtain/period contract, including one period.
        aligned = sensitivity
    else:
        if state_dim != "bc_region":
            sensitivity = sensitivity.rename({state_dim: "bc_region"})
            state_dim = "bc_region"
        aligned = _expand_boundary_periods(
            sensitivity.rename("H_bc").to_dataset(), observation_dim=raw_dim,
            frequency=frequency, anchor_time=anchor_time, observation_labels=None,
        ).H_bc
    if raw_dim != native_dim:
        # Native time has already determined periods; the restored dimension
        # index owns those same labels, so its auxiliary copy is redundant.
        if native_dim == "time" and "time" in aligned.coords:
            aligned = aligned.drop_vars("time")
        aligned = aligned.rename({raw_dim: native_dim})
    return aligned.transpose(native_dim, state_dim)


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
        Whether centered operands are present. Accepted operands remain raw
        for period expansion at model binding.
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


def align_boundary_operands(
    operands: xr.Dataset,
    *,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
    observation_labels: xr.DataArray | None = None,
) -> xr.Dataset:
    """Expand raw anomaly and mass with a shared period mask at model binding.

    Args:
        operands: Raw dataset accepted by :func:`validate_boundary_operands`.
        frequency: ``None`` creates one period without requiring timestamps.
            A frequency (including ``'monthly'``) requires a one-dimensional
            datetime ``time`` coordinate along the observation dimension.
        anchor_time: Optional anchor for fixed-duration periods; not allowed
            for a single period or calendar-month periods.
        observation_labels: Optional authoritative labels used to select and
            reorder the indexed observation dimension before expansion.

    Returns:
        New dataset with ``H_bc_anomaly`` and ``H_bc_mass`` on ``bc_region``
        and the original observation dimension. The ``bc_region`` MultiIndex
        contains ``bc_curtain`` and ``bc_period``. Units, representation and
        scalar ``bc_reference`` are preserved. Outputs are marked expanded and
        must not be passed through raw validation or expansion again. Borrowed
        payloads stay lazy and unchanged; frequency-based expansion materializes
        a lazy auxiliary time coordinate once to construct the period mask.

    Raises:
        ValueError: If raw operands or period settings are invalid, or required
            datetime coordinates are missing or incompatible.
        KeyError: If an authoritative observation label is absent.
    """
    validate_centered_boundary_periods(frequency, anchor_time)
    if not validate_boundary_operands(operands):
        raise ValueError("Expected centered boundary operands.")
    observation_dim = "time" if "time" in operands.H_bc_anomaly.dims else "nmeasure"
    result = _expand_boundary_periods(
        operands[["H_bc_anomaly", "H_bc_mass"]], observation_dim=observation_dim,
        frequency=frequency, anchor_time=anchor_time, observation_labels=observation_labels,
    )
    result["bc_reference"] = operands.bc_reference
    return result


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
