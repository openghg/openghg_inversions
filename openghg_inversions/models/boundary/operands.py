"""Centered boundary transport and its single binding step.

Centered preparation stores the boundary contribution as three operands:
the concentration anomaly ``A`` (``H_bc_anomaly``), the dimensionless transport
mass ``M`` (``H_bc_mass``) and a scalar concentration reference ``c``
(``bc_reference``). Absolute concentrations are ``A + c M``. Preparation and
artifact loading validate their units, labels and layout; this module trusts
those checks and only binds the raw spatial operands to one fit's observation
rows and boundary periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytensor
import xarray as xr

from openghg_inversions.boundary_sensitivity import _expand_boundary_periods, _select_native_boundary
from openghg_inversions.inversion_inputs import DatetimeLike


def require_centered_precision() -> None:
    """Require a process-wide PyTensor ``floatX`` of float64 for centered graphs.

    ``pm.Data`` casts registered arrays to the global ``floatX``. Centered
    operands rely on float64 to keep small anomalies after cancellation, so a
    float32 process would silently round them. This only reads the global
    configuration; set ``PYTENSOR_FLAGS=floatX=float64`` before starting Python.
    """
    if pytensor.config.floatX != "float64":
        raise ValueError("Centered boundaries require explicitly configured PyTensor floatX=float64.")


def centered_boundary_expression(anomaly, mass, reference, state):
    """Evaluate ``A b + c M (b - 1)`` without subtracting a large reference.

    Args:
        anomaly: Concentration anomaly design A, on observations by states.
        mass: Dimensionless transport M on the same axes.
        reference: Scalar concentration reference c.
        state: Dimensionless physical boundary scaling vector b.

    Returns:
        Centered concentration, using NumPy or PyTensor operands supplied by
        the caller. Alignment and materialization belong to the caller.
    """
    return anomaly @ state + reference * (mass @ (state - 1.0))


@dataclass(frozen=True)
class CenteredOperands:
    """Prepared centered transport, borrowed from a validated preparation.

    Attributes:
        anomaly: ``A`` on raw spatial ``bc_region`` and the ``time`` or
            ``nmeasure`` observation axis, in observation concentration units.
        mass: ``M`` on the same axes, dimensionless.
        reference: Scalar ``c`` in the anomaly's concentration units.
        weights: Optional positive dimensionless centering weights on
            ``(bc_curtain, bc_period)``, with interval-start period labels.
            Only the additive boundary consumes them.
    """

    anomaly: xr.DataArray
    mass: xr.DataArray
    reference: xr.DataArray
    weights: xr.DataArray | None = None

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> CenteredOperands:
        """Select operands (and any centering weights) from a validated preparation."""
        return cls(
            dataset["H_bc_anomaly"], dataset["H_bc_mass"], dataset["bc_reference"],
            dataset.get("bc_centering_weights"),
        )

    def bind(
        self,
        observations: xr.DataArray,
        *,
        frequency: str | None = None,
        anchor_time: DatetimeLike | None = None,
    ) -> xr.Dataset:
        """Select one fit's observation rows and expand spatial states by period.

        Observation dimension labels own the row order; their auxiliary
        ``time`` values own period membership when supplied. Indexed boundary
        times must match those timestamps; this check may compute a lazy
        timestamp coordinate, while numerical operands remain borrowed and lazy.

        Args:
            observations: One native channel with its observation axis first.
            frequency: Optional frequency expanding spatial states by period.
            anchor_time: Optional origin for fixed-duration periods.

        Returns:
            ``H_bc_anomaly`` and ``H_bc_mass`` on ``(observation, bc_region)``
            with a ``(bc_curtain, bc_period)`` state index, plus scalar
            ``bc_reference``. Centering weights are not included.

        Raises:
            ValueError: If PyTensor is not configured for float64, or periods
                or indexed timestamps conflict with the observations.
        """
        require_centered_precision()
        native_dim = str(observations.dims[0])
        raw_dim = "time" if "time" in self.anomaly.dims else "nmeasure"
        raw = _select_native_boundary(
            xr.Dataset({"H_bc_anomaly": self.anomaly, "H_bc_mass": self.mass}), observations, raw_dim,
        )
        bound = _expand_boundary_periods(
            raw, observation_dim=raw_dim, frequency=frequency, anchor_time=anchor_time,
            observation_labels=None,
        )
        if raw_dim != native_dim:
            # Native time has already determined periods; the restored dimension
            # index owns those same labels, so its auxiliary copy is redundant.
            if native_dim == "time" and "time" in bound.coords:
                bound = bound.drop_vars("time")
            bound = bound.rename({raw_dim: native_dim})
        return bound.transpose(native_dim, "bc_region").assign(bc_reference=self.reference)


__all__ = ["CenteredOperands", "centered_boundary_expression", "require_centered_precision"]
