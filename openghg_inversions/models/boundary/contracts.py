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
  :func:`~openghg_inversions.models.boundary.bind_centered_transport`
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

import pandas as pd
import xarray as xr

from openghg_inversions.xr_contracts import xr_contract


CENTERED_NAMES = ("H_bc_anomaly", "H_bc_mass", "bc_reference")
STATE_LEVELS = ("bc_curtain", "bc_period")
WHOLE_WINDOW = "whole_window"


def _check_units(variables: xr.Dataset, *, label: str) -> None:
    """Check the shared concentration units of anomaly and reference, and dimensionless mass."""
    units = variables.H_bc_anomaly.attrs.get("units")
    if not units or variables.bc_reference.attrs.get("units") != units or variables.H_bc_mass.attrs.get("units") != "1":
        raise ValueError(
            f"{label} H_bc_anomaly and bc_reference must share concentration units; H_bc_mass units must be '1'."
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
    if scale is None or abs(scale - 1.0) > 1e-12:
        raise ValueError(f"{label} requires dimensionless units equivalent to '1'; got {units!r}.")


def check_centering_weights(weights: xr.DataArray, *, label: str = "bc_centering_weights") -> None:
    """Check the structure of raw additive centering weights.

    Raw weights lie on ``("bc_curtain", "bc_period")`` with unique indexed
    curtain labels and unique interval-start ``DatetimeIndex`` period labels
    (or the single label ``'whole_window'`` for timestamp-free transport),
    and have units equivalent to ``'1'``. Payload values are not inspected;
    their positivity is established by the eager boundary accepting them.

    Args:
        weights: Borrowed weights, possibly Dask-backed.
        label: Name used in error messages.

    Raises:
        ValueError: If the weights violate that structure.
    """
    if weights.dims != STATE_LEVELS:
        raise ValueError(f"{label} must use dimensions {STATE_LEVELS}; got {weights.dims}.")
    for dim in STATE_LEVELS:
        _require_unique_index(weights, dim, label=label)
    periods = weights.indexes["bc_period"]
    if not isinstance(periods, pd.DatetimeIndex) and list(periods) != [WHOLE_WINDOW]:
        raise ValueError(f"{label} bc_period labels must be interval start times, or {WHOLE_WINDOW!r} without timestamps.")
    _require_dimensionless(weights, label=label)


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
    * ``bc_reference`` is scalar.
    * ``H_bc_anomaly`` and ``bc_reference`` carry the same nonempty
      concentration unit string; ``H_bc_mass`` has units ``'1'``. An ``mf``
      variable, when present, has the same concentration units.
    * Optional ``bc_centering_weights`` lie on ``(bc_curtain, bc_period)``
      as described by :func:`check_centering_weights`. Only weighted additive
      anomalies consume them.
    * Other variables are permitted. An absolute ``H_bc`` diagnostic must be
      marked ``boundary_sensitivity_preparation='unexpanded'``.

    Numerical guarantees, established by the producer rather than the
    validator: ``A`` and ``c`` are finite and ``M`` is finite, non-negative and
    not all zero; optional weights are finite and strictly positive. The
    eager boundary that first materializes these payloads owns their checks.
    """

    __slots__ = ()

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
                unit is missing or inconsistent.
        """
        missing = [name for name in CENTERED_NAMES if name not in data.variables]
        if missing:
            raise ValueError(f"{label} require H_bc_anomaly, H_bc_mass and bc_reference; missing {', '.join(missing)}.")
        if data.attrs.get("boundary_representation") != "centered-v1":
            raise ValueError(f"{label} require boundary_representation='centered-v1'.")
        if "H_bc" in data and data.H_bc.attrs.get("boundary_sensitivity_preparation") != "unexpanded":
            raise ValueError(f"{label} cannot include expanded or unmarked absolute H_bc.")
        anomaly, mass = data.H_bc_anomaly, data.H_bc_mass
        if isinstance(data.indexes.get("bc_region"), pd.MultiIndex) or any(
            value.attrs.get("boundary_sensitivity_preparation") not in (None, "unexpanded") for value in (anomaly, mass)
        ):
            raise ValueError(f"{label} require unexpanded spatial operands; align only at model binding.")
        observation_dim = "time" if "time" in anomaly.dims else "nmeasure"
        expected = {"bc_region", observation_dim}
        if set(anomaly.dims) != expected or set(mass.dims) != expected:
            raise ValueError(
                f"{label} require H_bc_anomaly and H_bc_mass on bc_region and {observation_dim}; "
                f"got {anomaly.dims} and {mass.dims}."
            )
        for dim in ("bc_region", observation_dim):
            _require_unique_index(data, dim, label=label)
        if data.bc_reference.ndim != 0:
            raise ValueError(f"{label} bc_reference must be scalar.")
        _check_units(data, label=label)
        if "mf" in data and data.mf.attrs.get("units") != anomaly.attrs["units"]:
            raise ValueError(f"{label} concentration units must match mf units.")
        if "bc_centering_weights" in data:
            check_centering_weights(data.bc_centering_weights, label=f"{label} bc_centering_weights")
        return cast(RawCenteredTransport, data)


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
    * ``bc_reference`` is scalar.
    * ``H_bc_anomaly`` and ``bc_reference`` carry the same nonempty
      concentration unit string; ``H_bc_mass`` has units ``'1'``.
    * Other variables are permitted, including the additive extension's
      ``bc_centering_weights``.

    Numerical guarantees are those of the raw transport it was bound from;
    binding selects and masks values and introduces no new ones beyond zeros
    for states outside a row's period.
    """

    __slots__ = ()

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
                not unique, ``bc_reference`` is not scalar, or units differ.
        """
        label = "Bound centered boundary transport"
        missing = [name for name in CENTERED_NAMES if name not in data.variables]
        if missing:
            raise ValueError(f"{label} lacks {', '.join(missing)}; bind the choice again.")
        layout = data.H_bc_anomaly.dims
        if len(layout) != 2 or layout[1] != "bc_region" or data.H_bc_mass.dims != layout:
            raise ValueError(
                f"{label} requires H_bc_anomaly and H_bc_mass on (observation, 'bc_region'); "
                f"got {data.H_bc_anomaly.dims} and {data.H_bc_mass.dims}."
            )
        _require_unique_index(data, str(layout[0]), label=label)
        states = _require_unique_index(data, "bc_region", label=label)
        if not isinstance(states, pd.MultiIndex) or tuple(states.names) != STATE_LEVELS:
            raise ValueError(f"{label} requires a bc_region MultiIndex with levels {STATE_LEVELS}.")
        if data.bc_reference.ndim != 0:
            raise ValueError(f"{label} bc_reference must be scalar.")
        _check_units(data, label=label)
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


@xr_contract
class BoundAdditiveTransport(BoundCenteredTransport):
    """Bound centered transport with additive centering weights.

    In addition to every :class:`BoundCenteredTransport` guarantee,
    ``bc_centering_weights`` lies on ``("bc_region",)``, sharing the bound
    state index, and has units equivalent to ``'1'``. Weights are finite and
    strictly positive; the boundary accepting raw weights establishes that.
    Their normalization is part of the additive graph, not the contract.
    """

    __slots__ = ()

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
        BoundCenteredTransport.validate(data)
        weights = data.get("bc_centering_weights")
        if weights is None or weights.dims != ("bc_region",):
            raise ValueError("Bound additive transport requires bc_centering_weights on bc_region; bind the choice again.")
        _require_dimensionless(weights, label="Bound bc_centering_weights")
        return cast(BoundAdditiveTransport, data)


__all__ = [
    "BoundAdditiveTransport",
    "BoundCenteredTransport",
    "RawCenteredTransport",
    "check_centering_weights",
]
