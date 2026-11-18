"""Configured biosphere calibration of the CO2 retained flux prior.

Configuration contains choices; labelled operators, selectors and reporting
functionals remain scientific inputs. A biosphere calibration is resolved once
and saved, because it is a scientific choice reviewed before inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openghg.util import cf_ureg
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.serialization import load_trace, save_trace

from .configuration import _number, _reject_unknown, _string, _table, _take


def _quantity(value: object, path: str) -> tuple[float, str]:
    """Read a ``{value, units}`` table at ``path`` as a number and its units."""
    options = _table(value, path)
    number = _number(_take(options, "value", path), f"{path}.value", positive=True)
    units = _string(_take(options, "units", path), f"{path}.units")
    _reject_unknown(options, path)
    cf_ureg.Unit(units)
    return number, units


def _save(data: xr.Dataset, path: str | Path, group: str) -> None:
    """Write one calibration dataset under ``group`` to a new artifact file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Use a new artifact destination: {path}")
    save_trace(xr.DataTree.from_dict({group: data}), path)


def load_biosphere_prior(path: str | Path) -> BiospherePrior:
    """Validate a saved calibration against its saved baseline, then retain it.

    Returns:
        The resolved calibration, without fitting its policy again.
    """
    return BiospherePrior.from_dataset(load_trace(path)["biosphere_prior"].to_dataset())


@dataclass(frozen=True)
class BiospherePriorConfig:
    """Exactly one retained-biosphere calibration policy, with no shrink default."""

    family: Literal["lognormal", "gaussian"]
    policy: str
    value: float
    units: str | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> BiospherePriorConfig:
        """Parse family and one multiplier, residual-SD or marginal-gross-SD target.

        Args:
            params: The ``biosphere`` table. ``family`` is ``"lognormal"``
                (default) or ``"gaussian"``. Exactly one policy is required,
                with no default: a positive ``residual_sd_multiplier``, or a
                ``target_residual_sd`` or marginal ``target_gross_sd`` given as
                a ``{value, units}`` table with a positive value and Pint units.
                No other keys are accepted.

        Returns:
            Calibration choices, without resolving numerical inputs. Unit
            compatibility and target feasibility are checked by :meth:`prepare`.

        Raises:
            TypeError: If the table, a key, ``family``, a value or ``units``
                has the wrong type.
            ValueError: If ``family`` is unknown, zero or several policies are
                given, a policy value is not finite and positive, a target
                lacks ``value`` or ``units``, or an unknown key is present.
                Units that Pint does not recognise raise Pint's
                ``UndefinedUnitError`` instead.
        """
        options = _table(params, "biosphere")
        family_name = _string(options.pop("family", "lognormal"), "biosphere.family")
        family: Literal["lognormal", "gaussian"]
        if family_name == "lognormal":
            family = "lognormal"
        elif family_name == "gaussian":
            family = "gaussian"
        else:
            raise ValueError("biosphere.family must be lognormal or gaussian.")
        policies = {"residual_sd_multiplier", "target_residual_sd", "target_gross_sd"} & options.keys()
        if len(policies) != 1:
            raise ValueError("Configure exactly one biosphere calibration policy.")
        policy = policies.pop()
        if policy == "residual_sd_multiplier":
            value, units = _number(options.pop(policy), f"biosphere.{policy}", positive=True), None
        else:
            value, units = _quantity(options.pop(policy), f"biosphere.{policy}")
        _reject_unknown(options, "biosphere")
        return cls(family, policy, value, units)

    def prepare(
        self,
        baseline: CorrelatedLognormalPrior,
        *,
        biosphere: xr.DataArray,
        weights: xr.DataArray,
        reference: xr.DataArray,
    ) -> BiospherePrior:
        """Calibrate labelled selector/functionals at the existing eager boundary.

        Forwards the configured family and policy to
        :func:`~openghg_inversions.biosphere_prior.calibrate_biosphere_prior`,
        which owns the detailed label, selector, unit and feasibility contract.
        Absolute targets carry their configured units and are converted to the
        units of ``weights``.

        Args:
            baseline: The original retained prior to calibrate.
            biosphere: Boolean selector on the baseline state dimension.
            weights: ``GPP`` and ``TER`` reporting weights on ``process`` and
                the state dimension, with declared flux units.
            reference: ``GPP`` and ``TER`` totals at the baseline mean on
                ``process``, with declared flux units.

        Returns:
            The resolved prior and audit for the exact original retained prior.

        Raises:
            ValueError: If calibration rejects the labels, selector, missing
                units or moments, or the configured target is infeasible.
        """
        residual_multiplier: float | None = None
        absolute_target: xr.DataArray | None = None
        if self.policy == "residual_sd_multiplier":
            residual_multiplier = self.value
        else:
            absolute_target = xr.DataArray(self.value, attrs={"units": self.units})
        return calibrate_biosphere_prior(
            baseline,
            biosphere=biosphere,
            weights=weights,
            reference=reference,
            family=self.family,
            residual_sd_multiplier=residual_multiplier,
            target_residual_sd=absolute_target if self.policy == "target_residual_sd" else None,
            target_gross_sd=absolute_target if self.policy == "target_gross_sd" else None,
        )

    @staticmethod
    def save(prepared: BiospherePrior, path: str | Path) -> None:
        """Save the complete original prior, selector, functionals and resolved audit.

        Args:
            prepared: A calibration returned by :meth:`prepare`.
            path: New artifact destination, reopened by ``load_biosphere_prior``.

        Raises:
            FileExistsError: If ``path`` already exists; files are never
                overwritten.
        """
        _save(prepared.to_dataset(), path, "biosphere_prior")
