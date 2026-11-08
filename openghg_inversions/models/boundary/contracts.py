"""Static contracts for centered boundary transport before and after binding.

Centered transport represents the absolute boundary design as ``A + c M``:
the concentration anomaly ``A`` (``H_bc_anomaly``), the dimensionless mass
``M`` (``H_bc_mass``) and a scalar reference ``c`` (``bc_reference``). Values
annotated with these contracts are ordinary :class:`xarray.Dataset` objects;
see :doc:`/development/validation_and_xarray`.

Stages and owners:

* :class:`RawCenteredTransport` is prepared or loaded transport on its raw
  spatial states. Preparation and artifact loading establish it with
  :meth:`RawCenteredTransport.validate`.
* :class:`BoundCenteredTransport` is one fit's transport after
  :meth:`BoundCenteredTransport.bind`
  has selected observation rows and expanded spatial states by period. Every
  centered family consumes it; binding establishes it by construction.
* :class:`BoundAdditiveTransport` extends it with centering weights aligned to
  the bound states. Only weighted additive anomalies require it.

Absolute ``H_bc`` transport, used only by the multiplicative family, keeps its
own supported state layouts and has no contract in this pilot.

Validators inspect names, dimensions, eager indexes and attributes only. They
never compute numerical payloads or auxiliary coordinates, so Dask-backed
values stay lazy and the identical object is returned.
"""

from __future__ import annotations

from typing import cast

from dask import is_dask_collection

import numpy as np
import pandas as pd
import xarray as xr
from xarray_validate import AttrSchema, DataArraySchema, DatasetSchema, DimsSchema, SchemaError, ValidationContext

from openghg_inversions.xr_contracts import xr_contract
from openghg_inversions.boundary_sensitivity import bind_period_states
from openghg_inversions.inversion_inputs import DatetimeLike
from .operands import require_centered_precision


CENTERED_NAMES = ("H_bc_anomaly", "H_bc_mass", "bc_reference")
STATE_LEVELS = ("bc_curtain", "bc_period")
WHOLE_WINDOW = "whole_window"


def _validate_schema(schema: DatasetSchema | DataArraySchema, data: xr.Dataset | xr.DataArray, *, label: str) -> None:
    """Report schema paths and relationship errors with the caller's label."""
    context = ValidationContext(mode="lazy")  # Collect metadata errors, never Dask execution.
    try:
        schema.validate(data, context=context)
    except (SchemaError, ValueError) as error:
        summary = f"{context.result.get_error_summary()}\n" if context.has_errors else ""
        raise ValueError(f"{label}: {summary}{error}") from error
    if context.has_errors:
        raise ValueError(f"{label}: {context.result.get_error_summary()}")


def _centered_variables(observation_dim: str, *, ordered: bool) -> dict[str, DataArraySchema]:
    """Declare the common real-valued operands and unit metadata."""
    dims = DimsSchema((observation_dim, "bc_region"), ordered=ordered)
    return {
        "H_bc_anomaly": DataArraySchema(dtype=(np.integer, np.floating), dims=dims, attrs={"units": AttrSchema(type=str)}),
        "H_bc_mass": DataArraySchema(dtype=(np.integer, np.floating), dims=dims, attrs={"units": AttrSchema(value="1")}),
        "bc_reference": DataArraySchema(dtype=(np.integer, np.floating), dims=(), attrs={"units": AttrSchema(type=str)}),
    }


def _check_units(variables: xr.Dataset, *, label: str) -> None:
    """Check the shared concentration units of anomaly and reference."""
    units = variables.H_bc_anomaly.attrs.get("units")
    if not isinstance(units, str) or not units.strip() or variables.bc_reference.attrs.get("units") != units:
        raise ValueError(
            f"{label} H_bc_anomaly and bc_reference must share concentration units."
        )


def _require_unique_index(data: xr.Dataset | xr.DataArray, dim: str, *, label: str) -> pd.Index:
    """Return an indexed dimension's eager labels, requiring them to be unique."""
    index = data.indexes.get(dim)
    if index is None or not index.is_unique:
        raise ValueError(f"{label} requires unique indexed {dim} labels.")
    return index


def _require_dimensionless(array: xr.DataArray, *, label: str) -> None:
    """Require units with the same dimension and scale as ``'1'``."""
    from openghg_inversions.inversion_data._units import cf_ureg

    units = array.attrs.get("units")
    try:
        scale = float(cf_ureg.Quantity(1, units).to("1").magnitude) if isinstance(units, str) and units.strip() else None
    except Exception:  # noqa: BLE001 - any parse failure is an invalid unit
        scale = None
    if scale is None or not np.isfinite(scale) or abs(scale - 1.0) > 1e-12:
        raise ValueError(f"{label} requires dimensionless units equivalent to '1'; got {units!r}.")


def _weights_schema() -> DataArraySchema:
    """Describe raw centering weights on the curtain-period grid."""
    def relationships(weights: xr.DataArray) -> None:
        """Check indexed curtain and calendar labels and the unit scale."""
        for dim in STATE_LEVELS:
            _require_unique_index(weights, dim, label="bc_centering_weights")
        periods = weights.indexes["bc_period"]
        if not isinstance(periods, pd.DatetimeIndex) and list(periods) != [WHOLE_WINDOW]:
            raise ValueError(f"bc_period labels must be interval start times, or {WHOLE_WINDOW!r} without timestamps.")
        _require_dimensionless(weights, label="bc_centering_weights")

    return DataArraySchema(dtype=(np.integer, np.floating), dims=STATE_LEVELS, checks=[relationships])


@xr_contract
class RawCenteredTransport(xr.Dataset):
    """Prepared centered boundary transport on raw spatial boundary states.

    Structural guarantees, checked by :meth:`validate`:

    * ``attrs["boundary_representation"] == "centered-v1"``.
    * ``H_bc_anomaly`` and ``H_bc_mass`` share the dimensions ``bc_region``
      and one raw observation dimension, ``time`` when the anomaly has it and
      otherwise ``nmeasure``, in either order. Both dimensions have unique
      eager indexes, and ``bc_region`` is a plain spatial index, not the
      bound ``(bc_curtain, bc_period)`` MultiIndex. Neither operand is marked
      as expanded by ``boundary_sensitivity_preparation``.
    * The operands have real integer or floating dtypes; ``bc_reference`` is scalar.
    * ``H_bc_anomaly`` and ``bc_reference`` carry the same nonempty
      concentration unit string; ``H_bc_mass`` has units ``'1'``. An ``mf``
      variable, when present, has the same concentration units.
    * Optional ``bc_centering_weights`` lie on ``(bc_curtain, bc_period)``
      with unique curtain labels, interval-start ``DatetimeIndex`` periods (or
      the single label ``'whole_window'`` without timestamps) and units
      equivalent to ``'1'``. Only weighted additive
      anomalies consume them.
    * Other variables are permitted. An absolute ``H_bc`` diagnostic must be
      marked ``boundary_sensitivity_preparation='unexpanded'``.

    Numerical requirements are not established by this lazy-stage contract.
    After the recipe jointly materializes the operands, the explicit
    numerical operation checks finite ``A`` and ``c`` and finite, non-negative, nonzero ``M``;
    weighted additive operations also check finite, strictly positive weights.
    """

    __slots__ = ()

    @staticmethod
    def validate_weights(weights: xr.DataArray, *, label: str = "bc_centering_weights") -> None:
        """Check raw centering weights against the layout above, without reading values.

        Args:
            weights: Borrowed weights, possibly Dask-backed.
            label: Name used in error messages.

        Raises:
            ValueError: If the weights violate that layout.
        """
        _validate_schema(_weights_schema(), weights, label=label)

    @staticmethod
    def schema(*, observation_dim: str = "time") -> DatasetSchema:
        """Describe raw centered metadata, allowing either operand axis order.

        Args:
            observation_dim: Raw observation axis, normally time or nmeasure.

        Returns:
            Inspectable metadata schema; callbacks check units and eager indexes.
        """
        def relationships(data: xr.Dataset) -> None:
            """Check the representation marker and spatial state, expansion and unit relationships."""
            if data.attrs.get("boundary_representation") != "centered-v1":
                raise ValueError("require boundary_representation='centered-v1'.")
            if not set(CENTERED_NAMES).issubset(data.data_vars):
                return
            if "H_bc" in data and data.H_bc.attrs.get("boundary_sensitivity_preparation") != "unexpanded":
                raise ValueError("cannot include expanded or unmarked absolute H_bc.")
            if isinstance(data.indexes.get("bc_region"), pd.MultiIndex) or any(
                data[name].attrs.get("boundary_sensitivity_preparation") not in (None, "unexpanded")
                for name in ("H_bc_anomaly", "H_bc_mass")
            ):
                raise ValueError("require unexpanded spatial operands; align only at model binding.")
            for dim in ("bc_region", observation_dim):
                _require_unique_index(data, dim, label="Raw centered transport")
            _check_units(data, label="Raw centered transport")
            if "mf" in data and data.mf.attrs.get("units") != data.H_bc_anomaly.attrs.get("units"):
                raise ValueError("concentration units must match mf units.")

        return DatasetSchema(
            data_vars={
                **_centered_variables(observation_dim, ordered=False),
                "{bc_centering_weights}": _weights_schema(),
            },
            checks=[relationships],
        )

    @staticmethod
    def validate(data: xr.Dataset, *, label: str = "Centered boundaries") -> RawCenteredTransport:
        """Check raw centered transport structure and return the same dataset.

        Args:
            data: Borrowed dataset, possibly Dask-backed.
            label: Prefix for error messages, such as a channel name.

        Returns:
            ``data`` itself, annotated as raw centered transport.

        Raises:
            ValueError: If a required variable, attribute, dimension, index or
                unit or real numerical dtype is missing or inconsistent.
        """
        anomaly = data.get("H_bc_anomaly")
        observation_dim = "time" if anomaly is None or "time" in anomaly.dims else "nmeasure"
        _validate_schema(RawCenteredTransport.schema(observation_dim=observation_dim), data, label=label)
        return cast(RawCenteredTransport, data)

    @staticmethod
    def with_centering_weights(data: RawCenteredTransport, weights: xr.DataArray) -> RawCenteredTransport:
        """Replace raw weights without aligning them to an obsolete weight calendar.

        Only axes owned exclusively by the old weights are removed, so the
        supplied labels survive until binding checks them exactly.

        Args:
            data: Borrowed raw transport, optionally with existing weights.
            weights: Structurally valid replacement on curtain and period axes.

        Returns:
            New dataset with the replacement's exact labels; nothing is computed.

        Raises:
            ValueError: If another variable or auxiliary coordinate shares an
                old weight axis, so replacing its labels would change that data.
        """
        for dim in STATE_LEVELS:
            consumers = [
                name for name, array in data.variables.items()
                if dim in array.dims and name != "bc_centering_weights"
                and (name != dim or array.dims != (dim,))
            ]
            if consumers:
                raise ValueError(f"Boundary weight dimension {dim!r} is shared by {consumers!r}.")
        transport = data.drop_dims(STATE_LEVELS, errors="ignore")
        return cast(RawCenteredTransport, transport.assign(bc_centering_weights=weights))


@xr_contract
class BoundCenteredTransport(xr.Dataset):
    """Centered boundary transport bound to one fit's rows and periods.

    Structural guarantees, checked by :meth:`validate`:

    * ``H_bc_anomaly`` and ``H_bc_mass`` share dimensions
      ``(observation, "bc_region")`` in that order. The observation dimension
      is, by convention, ``H_bc_anomaly.dims[0]``
      (:meth:`observation_dim`); it has a unique eager index whose labels
      order the fit's rows.
    * ``bc_region`` is a unique MultiIndex with levels
      ``("bc_curtain", "bc_period")``: every bound state is one curtain in
      one boundary period, including curtains unobserved in that period.
    * The operands have real integer or floating dtypes; ``bc_reference`` is scalar.
    * ``H_bc_anomaly`` and ``bc_reference`` carry the same nonempty
      concentration unit string; ``H_bc_mass`` has units ``'1'``.
    * Other variables are permitted, including the additive extension's
      ``bc_centering_weights``.

    This contract establishes structure only. Numerical requirements remain
    with the explicit later numerical operation, as for raw transport. Binding
    selects and masks rows, so even nonzero raw mass may become all zero.
    """

    __slots__ = ()

    @staticmethod
    def schema(*, observation_dim: str = "nmeasure") -> DatasetSchema:
        """Describe bound centered metadata and its complete curtain-period grid.

        Args:
            observation_dim: First axis of both bound operands.

        Returns:
            Inspectable schema, with eager index and unit relationship callbacks.
        """
        def relationships(data: xr.Dataset) -> None:
            """Require complete, unique states without reading numerical payloads."""
            if not set(CENTERED_NAMES).issubset(data.data_vars):
                return
            _require_unique_index(data, observation_dim, label="Bound centered transport")
            states = _require_unique_index(data, "bc_region", label="Bound centered transport")
            if not isinstance(states, pd.MultiIndex) or tuple(states.names) != STATE_LEVELS:
                raise ValueError(f"requires a bc_region MultiIndex with levels {STATE_LEVELS}.")
            # MultiIndex.hasnans is unsupported; missing level codes are -1.
            if any((codes < 0).any() for codes in states.codes) or len(states) != (
                len(states.get_level_values(0).unique()) * len(states.get_level_values(1).unique())
            ):
                raise ValueError("bc_region must contain every curtain-period Cartesian combination without missing labels.")
            _check_units(data, label="Bound centered transport")

        return DatasetSchema(data_vars=_centered_variables(observation_dim, ordered=True), checks=[relationships])

    @staticmethod
    def validate(data: xr.Dataset) -> BoundCenteredTransport:
        """Check bound centered transport structure and return the same dataset.

        Args:
            data: Borrowed dataset, possibly Dask-backed.

        Returns:
            ``data`` itself, annotated as bound centered transport.

        Raises:
            ValueError: If a variable is missing, the operands do not share
                the ``(observation, bc_region)`` layout, an index is missing or
                not unique or complete, ``bc_reference`` is not scalar, units
                differ, or an operand has a non-real dtype.
        """
        anomaly = data.get("H_bc_anomaly")
        observation_dim = str(anomaly.dims[0]) if anomaly is not None and anomaly.ndim else "nmeasure"
        _validate_schema(
            BoundCenteredTransport.schema(observation_dim=observation_dim), data, label="Bound centered boundary transport",
        )
        return cast(BoundCenteredTransport, data)

    @staticmethod
    def observation_dim(data: BoundCenteredTransport) -> str:
        """Return the observation dimension, the first dimension of ``H_bc_anomaly``.

        Args:
            data: Bound centered transport.

        Returns:
            The observation dimension name.
        """
        return str(data.H_bc_anomaly.dims[0])

    @staticmethod
    def bind(
        transport: RawCenteredTransport,
        observations: xr.DataArray,
        *,
        frequency: str | None = None,
        anchor_time: DatetimeLike | None = None,
    ) -> BoundCenteredTransport:
        """Select one fit's observation rows and expand spatial states by period.

        Rows and periods follow
        :func:`~openghg_inversions.boundary_sensitivity.bind_period_states`.
        The contract holds by construction. Weights are not included; see
        :meth:`BoundAdditiveTransport.bind_weights`.

        Args:
            transport: Raw centered transport on its ``time`` or ``nmeasure`` axis.
            observations: One native channel with its uniquely indexed observation
                axis first, as established by observation preparation.
            frequency: Optional frequency expanding spatial states by period.
            anchor_time: Optional origin for fixed-duration periods.

        Returns:
            Bound transport.

        Raises:
            ValueError: If PyTensor is not configured for float64, or periods or
                indexed timestamps conflict with the observations.
        """
        require_centered_precision()
        bound = bind_period_states(
            transport[["H_bc_anomaly", "H_bc_mass"]], observations, frequency=frequency, anchor_time=anchor_time,
        )
        return cast(BoundCenteredTransport, bound.assign(bc_reference=transport.bc_reference))

    @staticmethod
    def reference_shift(data: BoundCenteredTransport) -> xr.DataArray:
        """Check eager centered values once and calculate the fixed shift ``c M 1``.

        Every centered family calls this once at the start of ``add``, after the
        recipe's joint materialization, and reuses the shift.

        Args:
            data: Bound transport materialized jointly with the model inputs.

        Returns:
            ``c M 1`` on the observation axis, with the reference's attributes.

        Raises:
            ValueError: If the operands are lazy, ``A`` or ``c`` is non-finite, ``M`` is non-finite,
                negative or all zero, or the shift is non-finite.
        """
        if any(is_dask_collection(data[name]) for name in CENTERED_NAMES):
            raise ValueError("Materialize boundary data jointly with model inputs before computing the reference shift.")
        anomaly, mass, reference = (data[name] for name in CENTERED_NAMES)
        if not np.isfinite(reference.values).all() or not np.isfinite(anomaly.values).all():
            raise ValueError("Centered bc_reference and boundary anomaly must be finite.")
        if not np.isfinite(mass.values).all() or (mass.values < 0).any() or not (mass.values > 0).any():
            raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
        shift = (reference * mass.sum("bc_region", skipna=False)).assign_attrs(reference.attrs)
        if not np.isfinite(shift.values).all():
            raise ValueError("Boundary reference shift must be finite.")
        return shift


@xr_contract
class BoundAdditiveTransport(BoundCenteredTransport):
    """Bound centered transport with additive centering weights.

    In addition to every :class:`BoundCenteredTransport` guarantee,
    ``bc_centering_weights`` lies on ``("bc_region",)``, sharing the bound
    state index, and has units equivalent to ``'1'``. Finite, strictly positive
    weights remain a requirement of the explicit later numerical operation.
    Their normalization is part of the additive graph, not the contract.
    """

    __slots__ = ()

    @staticmethod
    def schema(*, observation_dim: str = "nmeasure") -> DatasetSchema:
        """Extend the bound centered schema with additive weight metadata.

        Args:
            observation_dim: First axis of both bound operands.

        Returns:
            Shared schema extended with real, dimensionless state weights.
        """
        schema = BoundCenteredTransport.schema(observation_dim=observation_dim)
        schema.data_vars["bc_centering_weights"] = DataArraySchema(
            dtype=(np.integer, np.floating), dims=("bc_region",),
            checks=[lambda weights: _require_dimensionless(weights, label="Bound bc_centering_weights")],
        )
        return schema

    @staticmethod
    def validate(data: xr.Dataset) -> BoundAdditiveTransport:
        """Check bound additive transport structure and return the same dataset.

        Args:
            data: Borrowed dataset, possibly Dask-backed.

        Returns:
            ``data`` itself, annotated as bound additive transport.

        Raises:
            ValueError: If the shared bound structure is violated, or the
                weights are missing or not on the bound state axis.
        """
        anomaly = data.get("H_bc_anomaly")
        observation_dim = str(anomaly.dims[0]) if anomaly is not None and anomaly.ndim else "nmeasure"
        _validate_schema(
            BoundAdditiveTransport.schema(observation_dim=observation_dim), data, label="Bound additive transport",
        )
        return cast(BoundAdditiveTransport, data)

    @staticmethod
    def bind_weights(bound: BoundCenteredTransport, weights: xr.DataArray) -> BoundAdditiveTransport:
        """Align raw centering weights exactly to bound states, giving the additive extension.

        Args:
            bound: Bound centered transport.
            weights: Structurally valid raw weights on ``(bc_curtain, bc_period)``.

        Returns:
            ``bound`` with ``bc_centering_weights`` on ``bc_region``; values stay lazy.

        Raises:
            ValueError: If the weights' curtain and period labels differ from the
                bound states.
        """
        periods = weights.indexes["bc_period"]
        if isinstance(periods, pd.DatetimeIndex):
            # Period starts from the fit calendar are nanosecond timestamps.
            weights = weights.assign_coords(bc_period=periods.as_unit("ns"))
        stacked = weights.stack(bc_region=STATE_LEVELS)
        observation_dim = BoundCenteredTransport.observation_dim(bound)
        xr.align(bound.H_bc_anomaly, stacked, join="exact", copy=False, exclude={observation_dim})
        # Labels match exactly, so the bound state index is reused for the weights.
        return cast(BoundAdditiveTransport, bound.assign(bc_centering_weights=stacked.variable))

    @staticmethod
    def normalized_weights(weights: xr.DataArray) -> xr.DataArray:
        """Check materialized weights once and normalize their sum to one.

        Args:
            weights: Weights on the bound ``bc_region`` states.

        Returns:
            Dimensionless normalized weights; scaling by the maximum first avoids overflow.

        Raises:
            ValueError: If weights are lazy, empty, non-finite or not positive.
        """
        if is_dask_collection(weights.data):
            raise ValueError("Materialize bc_centering_weights before forming a weighted anomaly design.")
        if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
            raise ValueError("bc_centering_weights must be finite and strictly positive.")
        scaled = weights / weights.max()
        return (scaled / scaled.sum()).assign_attrs(units="1")


__all__ = [
    "BoundAdditiveTransport",
    "BoundCenteredTransport",
    "RawCenteredTransport",
]
