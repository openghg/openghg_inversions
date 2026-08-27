"""Small helpers shared by concrete RHIME model recipes."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike, make_site_names
from openghg_inversions.models.additive_sigma import (
    DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_likelihood,
)
from openghg_inversions.models.pollution_event import add_pollution_event_likelihood
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.sigma import SigmaAlignment

from .builders import (
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
    RhimeLikelihoodBuilder,
)
from .specs import (
    DEFAULT_SIGMA_PRIOR,
    AdditiveSigmaSettings,
    LikelihoodSettings,
    PollutionEventSettings,
    RhimeModelSpec,
)


@dataclass(frozen=True)
class ForwardModelTerms:
    """Named concentration terms produced by a concrete model recipe."""

    total: TensorVariable
    pollution: TensorVariable
    baseline: TensorVariable | None


def _resolve_site_additive_sigma_prior(
    prior: PriorArgs,
    site: xr.DataArray,
    *,
    per_site: bool,
) -> PriorArgs:
    """Translate site-keyed additive-sigma priors to retained-site order."""
    resolved = dict(prior)
    scale = resolved.get("sigma")
    if not isinstance(scale, Mapping):
        return resolved
    if not per_site:
        raise ValueError("A site-keyed additive-sigma prior requires `sigma_per_site=True`.")

    site_names = [str(value) for value in make_site_names(site).values]
    missing = [name for name in site_names if name not in scale]
    if missing:
        raise ValueError(f"Site-keyed additive-sigma prior is missing retained site(s): {missing!r}.")
    values = [scale[name] for name in site_names]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("Site-keyed additive-sigma prior values must be finite positive numbers.")
    scales = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Site-keyed additive-sigma prior values must be finite positive numbers.")
    resolved["sigma"] = scales[:, None]
    return resolved


def prepare_additive_sigma_inputs(
    observations: xr.DataArray,
    *,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    sigma_freq_anchor: DatetimeLike | None = None,
    no_model_error: bool = False,
) -> tuple[SigmaAlignment | None, PriorArgs | None]:
    """Resolve additive-sigma settings into direct component inputs."""
    if no_model_error:
        return None, None
    site = observations.coords.get("site")
    if site is None or site.dims != ("nmeasure",):
        raise ValueError(
            "Additive-sigma mismatch requires an observation-aligned "
            "'site' coordinate when model error is enabled."
        )
    alignment = (
        SigmaAlignment.from_observations(
            observations,
            frequency=sigma_freq,
            per_site=sigma_per_site,
            anchor_time=sigma_freq_anchor,
        )
        if sigma_alignment is None
        else sigma_alignment
    )
    prior = _resolve_site_additive_sigma_prior(
        DEFAULT_ADDITIVE_SIGMA_PRIOR if sigma_prior is None else sigma_prior,
        site,
        per_site=sigma_per_site,
    )
    return alignment, prior


def _call_custom_likelihood(
    model: pm.Model,
    likelihood_builder: RhimeLikelihoodBuilder,
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    output_dim: str,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> TensorVariable:
    """Call a custom likelihood with the stable mean-only contract."""
    likelihood = likelihood_builder(
        observations=observations,
        observation_error=observation_error,
        aggregation_error=aggregation_error,
        mean=mean,
        output_dim=output_dim,
        **(likelihood_kwargs or {}),
    )
    if not isinstance(likelihood, TensorVariable):
        raise TypeError(
            f"A RHIME likelihood builder must return a PyTensor variable; got {type(likelihood).__name__}."
        )
    if likelihood.name != "y":
        raise ValueError(
            "A RHIME likelihood builder must name its observed concentration variable `y`; "
            f"got {likelihood.name!r}."
        )
    missing_names = sorted({"y", "epsilon"} - set(model.named_vars))
    if missing_names:
        raise ValueError(
            "A RHIME likelihood builder did not create the canonical variables required by "
            f"sampling and outputs: {missing_names!r}."
        )
    return likelihood


def add_builtin_likelihood(
    settings: LikelihoodSettings,
    *,
    forward: ForwardModelTerms,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    model_inputs: xr.Dataset,
    output_dim: str,
    sigma_alignment: SigmaAlignment | None = None,
    legacy_pollution_event_baseline: TensorVariable | None = None,
    preserve_legacy_pollution_event: bool = False,
) -> TensorVariable:
    """Dispatch one resolved built-in likelihood with explicit scientific inputs."""
    if isinstance(settings, PollutionEventSettings):
        needs_sigma = not settings.no_model_error or preserve_legacy_pollution_event
        resolved_alignment = sigma_alignment
        if needs_sigma and resolved_alignment is None:
            resolved_alignment = SigmaAlignment.from_observations(
                observations,
                frequency=settings.sigma_freq,
                per_site=settings.sigma_per_site,
                anchor_time=settings.sigma_freq_anchor,
            )
        baseline = (
            legacy_pollution_event_baseline
            if preserve_legacy_pollution_event
            else forward.baseline
        )
        return add_pollution_event_likelihood(
            observations=observations,
            observation_error=observation_error,
            minimum_error=model_inputs["min_error"],
            aggregation_error=aggregation_error,
            mean=forward.total,
            pollution_mean=forward.pollution,
            pollution_event_baseline=baseline,
            sigma_alignment=resolved_alignment,
            sigma_prior=dict(
                DEFAULT_SIGMA_PRIOR if settings.sigma_prior is None else settings.sigma_prior
            ),
            power=settings.power,
            pollution_events_from_obs=settings.pollution_events_from_obs,
            no_model_error=settings.no_model_error,
            retain_unused_sigma=preserve_legacy_pollution_event,
            output_dim=output_dim,
        )

    if isinstance(settings, AdditiveSigmaSettings):
        alignment, prior = prepare_additive_sigma_inputs(
            observations,
            sigma_alignment=sigma_alignment,
            sigma_prior=settings.sigma_prior,
            sigma_freq=settings.sigma_freq,
            sigma_per_site=settings.sigma_per_site,
            sigma_freq_anchor=settings.sigma_freq_anchor,
            no_model_error=settings.no_model_error,
        )
        return add_additive_sigma_likelihood(
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
            mean=forward.total,
            minimum_error_floor=(
                model_inputs["min_error"] if settings.use_minimum_error_floor else None
            ),
            additive_sigma_alignment=alignment,
            additive_sigma_prior=prior,
            output_dim=output_dim,
        )

    raise TypeError(f"Unsupported built-in likelihood settings: {type(settings).__name__}.")


def add_rhime_likelihood(
    model: pm.Model,
    *,
    settings: LikelihoodSettings | None,
    likelihood_builder: RhimeLikelihoodBuilder | None,
    likelihood_kwargs: Mapping[str, Any] | None,
    forward: ForwardModelTerms,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    model_inputs: xr.Dataset,
    output_dim: str,
    sigma_alignment: SigmaAlignment | None = None,
    legacy_pollution_event_baseline: TensorVariable | None = None,
    preserve_legacy_pollution_event: bool = False,
) -> TensorVariable:
    """Add exactly one built-in or custom likelihood to a concrete recipe."""
    if settings is None:
        if likelihood_builder is None:
            raise ValueError("A RHIME model requires built-in settings or a custom likelihood.")
        return _call_custom_likelihood(
            model,
            likelihood_builder,
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
            mean=forward.total,
            output_dim=output_dim,
            likelihood_kwargs=likelihood_kwargs,
        )
    if likelihood_builder is not None:
        raise ValueError("A custom likelihood cannot be combined with built-in likelihood settings.")
    return add_builtin_likelihood(
        settings,
        forward=forward,
        observations=observations,
        observation_error=observation_error,
        aggregation_error=aggregation_error,
        model_inputs=model_inputs,
        output_dim=output_dim,
        sigma_alignment=sigma_alignment,
        legacy_pollution_event_baseline=legacy_pollution_event_baseline,
        preserve_legacy_pollution_event=preserve_legacy_pollution_event,
    )


def builtin_model_build_result(
    model: pm.Model,
    *,
    model_spec: RhimeModelSpec,
    multisector: bool,
    input_names: Collection[str],
    preserve_legacy_baseline: bool = False,
) -> RhimeModelBuildResult:
    """Describe a built-in graph through its whole-model output roles.

    Args:
        model: Concrete built-in PyMC graph.
        model_spec: Scientific options used to build the graph.
        multisector: Whether sector-specific flux roles are required.
        input_names: Names available from retained canonical inputs.
        preserve_legacy_baseline: Whether the compatibility graph exposes the
            historical boundary-only baseline role.

    Returns:
        Concrete graph plus semantic roles and supported output formats.
    """
    roles = {
        "observation": "mf",
        "observation_error": "mf_error",
        "concentration": "y",
        "model_error": "epsilon",
    }
    if "min_error" in input_names:
        roles["minimum_error"] = "min_error"
    if "mf_repeatability" in input_names:
        roles["observation_repeatability"] = "mf_repeatability"
    if "mf_variability" in input_names:
        roles["observation_variability"] = "mf_variability"
    if multisector:
        for sector in model_spec.sectors:
            roles[f"flux_scale:{sector.name}"] = f"x_{sector.variable_suffix}"
            roles[f"flux_contribution:{sector.name}"] = f"mu_{sector.variable_suffix}"
            roles[f"emissions_sensitivity:{sector.name}"] = f"hx_{sector.variable_suffix}"
    else:
        roles.update({"flux_scale": "x", "flux_contribution": "mu", "emissions_sensitivity": "hx"})
    if model_spec.use_bc:
        roles.update(
            {
                "boundary": "mu_bc",
                "baseline_scale": "bc",
                "baseline_sensitivity": "hbc",
            }
        )
    if model_spec.add_offset:
        roles["offset"] = "offset"
    if preserve_legacy_baseline and model_spec.use_bc:
        roles["baseline"] = "mu_bc"
    elif model_spec.use_bc and not model_spec.add_offset:
        roles["baseline"] = "mu_bc"
    elif model_spec.add_offset and not model_spec.use_bc:
        roles["baseline"] = "offset"

    supported_output_formats = (
        ("none", "inv_out", "paris") if multisector else ("none", "inv_out", "basic", "paris", "legacy")
    )
    return RhimeModelBuildResult(
        model=model,
        variable_roles=roles,
        supported_output_formats=supported_output_formats,
        metadata={"kind": "builtin"},
    )


def validated_custom_model_build(
    model_builder: RhimeModelBuilder,
    *,
    context: RhimeModelBuilderContext,
) -> RhimeModelBuildResult:
    """Call an advanced complete-model builder and validate its return type.

    Args:
        model_builder: User-owned complete-model callable.
        context: Validated prepared inputs and run settings.

    Returns:
        Concrete custom model build result.

    Raises:
        TypeError: If the callable returns the wrong result type.
    """
    result = model_builder(context)
    if not isinstance(result, RhimeModelBuildResult):
        raise TypeError(
            f"A RHIME model builder must return `RhimeModelBuildResult`; got {type(result).__name__}."
        )
    return result
