"""Explicit preparation choices and frozen artifacts for CO2 prior components.

Configuration contains choices; labelled operators, selectors and reporting
functionals remain scientific inputs. Preparation is explicit and precedes row
selection. Inference reads the resulting artifact without preparing it again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openghg.util import cf_ureg
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction, SCHEMA as BC_SCHEMA
from openghg_inversions.models.shared_baseline import prepare_shared_baseline, SCHEMA as SHARED_SCHEMA
from openghg_inversions.serialization import load_trace, save_trace

from .configuration import _bool, _number, _reject_unknown, _string, _table, _take


def _quantity(value: object, path: str) -> tuple[float, str]:
    options = _table(value, path)
    number = _number(_take(options, "value", path), f"{path}.value", positive=True)
    units = _string(_take(options, "units", path), f"{path}.units")
    _reject_unknown(options, path)
    cf_ureg.Unit(units)
    return number, units


def _convert(quantity: tuple[float, str], units: str) -> float:
    value, source = quantity
    return float(cf_ureg.Quantity(value, source).to(units).magnitude)


def _save(data: xr.Dataset, path: str | Path, group: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Use a new artifact destination: {path}")
    save_trace(xr.DataTree.from_dict({group: data}), path)


def load_boundary_correction(path: str | Path) -> xr.Dataset:
    """Read frozen operators/reference/widths; never recenter or expand periods."""
    data = load_trace(path)["boundary_correction"].to_dataset()
    if data.attrs.get("boundary_correction_schema") != BC_SCHEMA:
        raise ValueError("Unsupported boundary-correction artifact.")
    return data


def load_biosphere_prior(path: str | Path) -> BiospherePrior:
    """Validate a saved calibration against its saved baseline, then retain it."""
    return BiospherePrior.from_dataset(load_trace(path)["biosphere_prior"].to_dataset())


def load_shared_baseline(path: str | Path) -> xr.Dataset:
    """Read frozen whole-support means, selectors and prior identity."""
    data = load_trace(path)["shared_baseline"].to_dataset()
    if data.attrs.get("shared_baseline_schema") != SHARED_SCHEMA:
        raise ValueError("Unsupported shared-baseline artifact.")
    return data


@dataclass(frozen=True)
class BoundaryCorrectionConfig:
    """Boundary family, reference, periods and explicit physical width choices.

    ``from_params`` accepts level_sd/pattern_sd as value/units tables. A-pattern
    widths require dimensionless units; all other widths use concentration.
    """

    family: str
    level_sd: tuple[float, str] | None = None
    pattern_sd: tuple[float, str] | None = None
    reference: str | tuple[float, str] = "stored_scalar"
    frequency: str | None = None
    anchor_time: str | None = None
    level_by_period: bool = False

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> BoundaryCorrectionConfig:
        """Parse one boundary preparation table without reading numerical inputs."""
        options = _table(params, "boundary")
        family = _string(_take(options, "family", "boundary"), "boundary.family")
        if family not in {"level", "A", "curtain_contrasts", "independent_curtains"}:
            raise ValueError("Unknown boundary.family.")
        widths = {
            name: _quantity(options.pop(name), f"boundary.{name}")
            for name in ("level_sd", "pattern_sd")
            if name in options
        }
        for name, needed in (
            ("level_sd", family != "independent_curtains"),
            ("pattern_sd", family != "level"),
        ):
            if (name in widths) != needed:
                raise ValueError(
                    f"boundary.{name} is {'required' if needed else 'inapplicable'} for {family}."
                )
        if "pattern_sd" in widths and family == "A":
            _convert(widths["pattern_sd"], "1")
        reference = options.pop("reference", "stored_scalar")
        if isinstance(reference, Mapping):
            # References may be zero or negative; widths must remain positive.
            value = _table(reference, "boundary.reference")
            number = _number(_take(value, "value", "boundary.reference"), "boundary.reference.value")
            units = _string(_take(value, "units", "boundary.reference"), "boundary.reference.units")
            _reject_unknown(value, "boundary.reference")
            reference = (number, units)
        elif reference not in {"stored_scalar", "whole_window_transport_mean"}:
            raise ValueError("boundary.reference requires a scalar value/units table or a supported policy.")
        frequency = (
            _string(options.pop("frequency"), "boundary.frequency") if "frequency" in options else None
        )
        anchor = (
            _string(options.pop("anchor_time"), "boundary.anchor_time") if "anchor_time" in options else None
        )
        period_level = _bool(options.pop("level_by_period", False), "boundary.level_by_period")
        _reject_unknown(options, "boundary")
        from .configuration import _validate_boundary_periods

        _validate_boundary_periods(frequency, anchor)
        if period_level and (frequency is None or family == "independent_curtains"):
            raise ValueError("Period levels require frequency and a family with a level.")
        return cls(
            family,
            reference=reference,
            frequency=frequency,
            anchor_time=anchor,
            level_by_period=period_level,
            **widths,
        )

    def prepare(self, operands: xr.Dataset, observations: xr.DataArray) -> xr.Dataset:
        """Resolve once on complete retained support, preserving lazy operators."""
        units = operands.bc_reference.attrs["units"]
        reference = None if self.reference == "stored_scalar" else self.reference
        if isinstance(reference, tuple):
            reference = _convert(reference, units)
        return prepare_boundary_correction(
            operands,
            observations,
            method=self.family,
            level_sd=None if self.level_sd is None else _convert(self.level_sd, units),
            pattern_sd=None
            if self.pattern_sd is None
            else _convert(self.pattern_sd, "1" if self.family == "A" else units),
            reference=reference,
            frequency=self.frequency,
            anchor_time=self.anchor_time,
            level_by_period=self.level_by_period,
        )

    def save(self, prepared: xr.Dataset, path: str | Path) -> None:
        """Materialize a resolved correction at the serialization boundary."""
        if prepared.attrs.get("boundary_correction_schema") != BC_SCHEMA:
            raise ValueError("Expected a prepared boundary correction.")
        _save(prepared, path, "boundary_correction")


@dataclass(frozen=True)
class BiospherePriorConfig:
    """Exactly one retained-biosphere calibration policy, with no shrink default."""

    family: str
    policy: str
    value: float
    units: str | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> BiospherePriorConfig:
        """Parse family and one multiplier, residual-SD or marginal-gross-SD target."""
        options = _table(params, "biosphere")
        family = _string(options.pop("family", "lognormal"), "biosphere.family")
        if family not in {"lognormal", "gaussian"}:
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
        """Calibrate labelled selector/functionals at the existing eager boundary."""
        value = self.value if self.units is None else xr.DataArray(self.value, attrs={"units": self.units})
        return calibrate_biosphere_prior(
            baseline,
            biosphere=biosphere,
            weights=weights,
            reference=reference,
            family=self.family,
            **{self.policy: value},
        )

    def save(self, prepared: BiospherePrior, path: str | Path) -> None:
        """Save the complete original prior, selector, functionals and resolved audit."""
        _save(prepared.to_dataset(), path, "biosphere_prior")


@dataclass(frozen=True)
class SharedBaselineConfig:
    """CO2-only outer mode and explicit atmospheric level width."""

    mode: str
    delta_sd: tuple[float, str]

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> SharedBaselineConfig:
        """Parse one shared-baseline preparation table."""
        options = _table(params, "shared_baseline")
        mode = _string(_take(options, "mode", "shared_baseline"), "shared_baseline.mode")
        if mode not in {"fixed", "additive", "multiplicative", "multiplicative_shrunk"}:
            raise ValueError("Unknown shared_baseline.mode.")
        width = _quantity(_take(options, "delta_sd", "shared_baseline"), "shared_baseline.delta_sd")
        _reject_unknown(options, "shared_baseline")
        return cls(mode, width)

    def prepare(
        self,
        flux_sensitivity: xr.DataArray,
        retained_prior: CorrelatedLognormalPrior,
        fixed_prior_contribution: xr.DataArray,
        boundary_correction: xr.Dataset,
        *,
        outer: xr.DataArray,
        sampled_outer: xr.DataArray,
        additive_design: xr.DataArray | None = None,
        calibrated_prior: BiospherePrior | None = None,
        additive_density_sd: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """Freeze whole-support means and selectors using the reviewed preparation."""
        return prepare_shared_baseline(
            flux_sensitivity,
            retained_prior,
            fixed_prior_contribution,
            boundary_correction,
            outer=outer,
            sampled_outer=sampled_outer,
            mode=self.mode,
            delta_sd=_convert(self.delta_sd, boundary_correction.bc_reference.attrs["units"]),
            additive_design=additive_design,
            calibrated_prior=calibrated_prior,
            additive_density_sd=additive_density_sd,
        )

    def save(self, prepared: xr.Dataset, path: str | Path) -> None:
        """Save the resolved reference, full-support means and complete prior payload."""
        if prepared.attrs.get("shared_baseline_schema") != SHARED_SCHEMA:
            raise ValueError("Expected a prepared shared baseline.")
        _save(prepared, path, "shared_baseline")


def build_shared_co2_from_config(
    config: Mapping[str, Any],
    *,
    prepared: xr.Dataset,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: Any,
):
    """Build the ordinary/cached shared recipe from existing CO2 likelihood syntax.

    ``config`` contains only a ``likelihood`` table. Preparation is already frozen;
    this explicit builder does not claim generic CO2 output/affine handoff support.
    Restore coordinates and absolute predictives before annotating/saving traces
    with ``annotate_shared_baseline_trace`` as for the Python recipe.
    """
    from .configuration import _co2_likelihood
    from .inference import uses_cached_ou
    from .shared_baseline_model import build_shared_co2_model, build_shared_co2_cached_sigma_model

    options = _table(config, "shared_co2")
    model = _co2_likelihood(_take(options, "likelihood", "shared_co2"), matched_ou=True)
    _reject_unknown(options, "shared_co2")
    inputs = dict(
        observations=observations, observation_error=observation_error, aggregation_error=aggregation_error
    )
    if uses_cached_ou(model):
        return build_shared_co2_cached_sigma_model(prepared, **inputs, **model["likelihood_kwargs"])
    no_model_error = model.pop("no_model_error", False)
    if "likelihood_builder" not in model and not no_model_error:
        from openghg_inversions.sigma import SigmaAlignment

        model["sigma_alignment"] = SigmaAlignment.from_observations(observations)
    if "likelihood_kwargs" in model:
        from .co2_runner import _resolve_co2_likelihood_kwargs

        model["likelihood_kwargs"] = _resolve_co2_likelihood_kwargs(
            model["likelihood_builder"],
            model["likelihood_kwargs"],
            **inputs,
        )
    return build_shared_co2_model(prepared, **inputs, **model)
