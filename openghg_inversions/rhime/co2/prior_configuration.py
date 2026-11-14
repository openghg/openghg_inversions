"""Configured CO2 prior components: biosphere calibration and shared baselines.

Configuration contains choices; labelled operators, selectors and reporting
functionals remain scientific inputs. A biosphere calibration is resolved once
and saved, because it is a scientific choice reviewed before inference. Shared
baselines are data-free choices bound to prepared transport at construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from openghg.util import cf_ureg
import pymc as pm
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice
from openghg_inversions.models.shared_baseline import OuterMode, SharedBaseline, SharedBaselineChoice
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.serialization import load_trace, save_trace

from .configuration import _boundary_choice, _number, _reject_unknown, _string, _table, _take
from .co2_cached_sigma_model import Co2CachedSigmaModel


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

        Returns:
            Calibration choices, without resolving numerical inputs.
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

        Returns:
            The resolved prior and audit for the exact original retained prior.
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

    def save(self, prepared: BiospherePrior, path: str | Path) -> None:
        """Save the complete original prior, selector, functionals and resolved audit."""
        _save(prepared.to_dataset(), path, "biosphere_prior")


def shared_baseline_choice(params: Mapping[str, object]) -> SharedBaselineChoice:
    """Parse one ``[shared_baseline]`` table into a data-free choice.

    ``mode`` selects the outer prior and ``delta_sd`` is the atmospheric level
    width in observation concentration units. An optional ``boundary`` table
    uses the ``[model.boundary]`` keys for its anomaly structure, with
    ``type = "additive"`` or ``"anomaly_scaling"`` and without a mean shift
    prior: the atmospheric level replaces it.

    Returns:
        The choice, to be bound to one fit's prepared transport.
    """
    options = _table(params, "shared_baseline")
    mode = _string(_take(options, "mode", "shared_baseline"), "shared_baseline.mode")
    delta_sd = _number(_take(options, "delta_sd", "shared_baseline"), "shared_baseline.delta_sd", positive=True)
    boundary = (
        _boundary_choice(dict(_table(options.pop("boundary"), "shared_baseline.boundary")), "shared_baseline.boundary")
        if "boundary" in options else None
    )
    _reject_unknown(options, "shared_baseline")
    if boundary is not None and not isinstance(boundary, AdditiveChoice | AnomalyScalingChoice):
        raise ValueError('shared_baseline.boundary requires type = "additive" or "anomaly_scaling".')
    return SharedBaselineChoice(boundary, delta_sd, cast(OuterMode, mode))


def build_shared_co2_from_config(
    config: Mapping[str, Any],
    *,
    baseline: SharedBaseline,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
) -> pm.Model | Co2CachedSigmaModel:
    """Build the ordinary/cached shared recipe from existing CO2 likelihood syntax.

    ``config`` contains only a ``likelihood`` table. ``baseline`` is already bound;
    this explicit builder does not claim generic CO2 output/affine handoff support.
    Restore coordinates and absolute predictives before annotating/saving traces
    with ``annotate_shared_baseline_trace`` as for the Python recipe.

    Returns:
        The ordinary PyMC model or the cached fixed-OU model and sampler state.
    """
    from .configuration import _co2_likelihood
    from .inference import uses_cached_ou
    from .shared_baseline_model import build_shared_co2_model, build_shared_co2_cached_sigma_model

    options = _table(config, "shared_co2")
    model = _co2_likelihood(_take(options, "likelihood", "shared_co2"), matched_ou=True)
    _reject_unknown(options, "shared_co2")
    if uses_cached_ou(model):
        return build_shared_co2_cached_sigma_model(
            baseline, observations=observations, observation_error=observation_error,
            aggregation_error=aggregation_error, **model["likelihood_kwargs"],
        )
    no_model_error = model.pop("no_model_error", False)
    if "likelihood_builder" not in model and not no_model_error:
        from openghg_inversions.sigma import SigmaAlignment

        model["sigma_alignment"] = SigmaAlignment.from_observations(observations)
    if "likelihood_kwargs" in model:
        from .co2_runner import _resolve_co2_likelihood_kwargs

        model["likelihood_kwargs"] = _resolve_co2_likelihood_kwargs(
            model["likelihood_builder"],
            model["likelihood_kwargs"],
            observations=observations, observation_error=observation_error, aggregation_error=aggregation_error,
        )
    return build_shared_co2_model(
        baseline, observations=observations, observation_error=observation_error,
        aggregation_error=aggregation_error, **model,
    )
