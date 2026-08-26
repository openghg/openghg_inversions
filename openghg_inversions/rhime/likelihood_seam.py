"""RHIME likelihood wiring, custom validation, and frozen compatibility translation."""

from __future__ import annotations

from collections.abc import Mapping
import json
from numbers import Real
from typing import Any, Protocol

import numpy as np
import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike, make_site_indicator, make_site_names
from openghg_inversions.models.additive_sigma import (
    DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_likelihood,
)
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.sigma import SigmaAlignment


class RhimeLikelihoodBuilder(Protocol):
    """Callable contract for a project-owned custom observation distribution."""

    def __call__(
        self,
        *,
        observations: xr.DataArray,
        observation_error: xr.DataArray,
        aggregation_error: AggregationError,
        mean: TensorVariable,
        output_dim: str,
    ) -> TensorVariable:
        """Add canonical ``y`` and ``epsilon`` variables to the active model."""
        ...


def validate_custom_likelihood_result(
    model: pm.Model,
    likelihood: object,
) -> TensorVariable:
    """Validate the canonical result of a caller-supplied likelihood."""
    if not isinstance(likelihood, TensorVariable):
        raise TypeError(
            "A RHIME likelihood builder must return a PyTensor variable; "
            f"got {type(likelihood).__name__}."
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


def validate_likelihood_kwargs(
    likelihood_builder: object | None,
    likelihood_kwargs: object | None,
) -> dict[str, Any] | None:
    """Copy and validate options owned by a custom or compatibility likelihood."""
    if likelihood_builder is not None and not callable(likelihood_builder):
        raise TypeError(
            f"`likelihood_builder` must be callable or None; got {type(likelihood_builder).__name__}."
        )
    if likelihood_kwargs is None:
        return None
    if not isinstance(likelihood_kwargs, Mapping):
        raise TypeError("`likelihood_kwargs` must be a mapping or None.")
    if any(not isinstance(key, str) for key in likelihood_kwargs):
        raise TypeError("`likelihood_kwargs` keys must be strings.")
    try:
        encoded = json.dumps(dict(likelihood_kwargs), allow_nan=False)
        options = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError("`likelihood_kwargs` must contain only JSON-compatible values.") from exc
    if options and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    return options


def _resolve_site_additive_scale_prior(
    prior: Mapping[str, Any],
    site: xr.DataArray,
    *,
    per_site: bool,
) -> dict[str, Any]:
    """Translate legacy site-keyed prior scales to retained-site order."""
    resolved = dict(prior)
    scale = resolved.get("sigma")
    if not isinstance(scale, Mapping):
        return resolved
    if not per_site:
        raise ValueError("A site-keyed additive scale prior requires `sigma_per_site=True`.")

    site_names = [str(value) for value in make_site_names(site).values]
    missing = [name for name in site_names if name not in scale]
    if missing:
        raise ValueError(f"Site-keyed additive scale prior is missing retained site(s): {missing!r}.")
    values = [scale[name] for name in site_names]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("Site-keyed additive scale prior values must be finite positive numbers.")
    scales = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Site-keyed additive scale prior values must be finite positive numbers.")
    resolved["sigma"] = scales[:, None]
    return resolved


def prepare_additive_sigma_inputs(
    observations: xr.DataArray,
    *,
    output_dim: str,
    site_indicator: xr.DataArray | None = None,
    sigma_prior: Mapping[str, Any] | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    sigma_freq_anchor: DatetimeLike | None = None,
    no_model_error: bool = False,
) -> tuple[SigmaAlignment | None, dict[str, Any] | None]:
    """Resolve RHIME additive-scale settings into direct component inputs."""
    if no_model_error:
        return None, None
    site = observations.coords.get("site")
    if site is None or site.dims != (output_dim,):
        raise ValueError(
            "Additive-sigma mismatch requires an observation-aligned "
            "'site' coordinate when model error is enabled."
        )
    indicator = make_site_indicator(site) if site_indicator is None else site_indicator
    alignment = SigmaAlignment.from_frequency(
        indicator,
        frequency=sigma_freq,
        per_site=sigma_per_site,
        anchor_time=sigma_freq_anchor,
    )
    prior = _resolve_site_additive_scale_prior(
        DEFAULT_ADDITIVE_SIGMA_PRIOR if sigma_prior is None else sigma_prior,
        site,
        per_site=sigma_per_site,
    )
    return alignment, prior


def additive_sigma_likelihood_builder(
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray | None = None,
    aggregation_error: AggregationError,
    mean: TensorVariable,
    output_dim: str,
    sigma_prior: Mapping[str, Any] | None = None,
    sigma_freq: str | None = None,
    sigma_per_site: bool = True,
    sigma_freq_anchor: DatetimeLike | None = None,
    no_model_error: bool = False,
) -> TensorVariable:
    """Preserve the legacy additive callback spelling outside model ownership."""
    alignment, prior = prepare_additive_sigma_inputs(
        observations,
        output_dim=output_dim,
        sigma_prior=sigma_prior,
        sigma_freq=sigma_freq,
        sigma_per_site=sigma_per_site,
        sigma_freq_anchor=sigma_freq_anchor,
        no_model_error=no_model_error,
    )
    return add_additive_sigma_likelihood(
        observations=observations,
        observation_error=observation_error,
        minimum_error_floor=minimum_error,
        aggregation_error=aggregation_error,
        mean=mean,
        additive_scale_alignment=alignment,
        additive_scale_prior=prior,
        output_dim=output_dim,
    )


_LEGACY_ADDITIVE_OPTION_NAMES = {
    "sigma_prior",
    "sigma_freq",
    "sigma_per_site",
    "sigma_freq_anchor",
    "no_model_error",
}


def _legacy_additive_sigma_options(
    likelihood_builder: object | None,
    likelihood_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return legacy additive options when the frozen adapter was selected."""
    if likelihood_builder is not additive_sigma_likelihood_builder:
        return None
    options = dict(likelihood_kwargs or {})
    unknown = sorted(set(options) - _LEGACY_ADDITIVE_OPTION_NAMES)
    if unknown:
        raise TypeError(f"Unsupported legacy additive-sigma option(s): {unknown!r}.")
    return options


def translate_legacy_likelihood_selection(
    likelihood_builder: RhimeLikelihoodBuilder | None,
    likelihood_kwargs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], RhimeLikelihoodBuilder | None, dict[str, Any] | None]:
    """Translate the frozen additive callback into explicit model settings.

    Public runner boundaries call this once before resolving a model spec. The
    concrete recipes therefore select built-in mismatch components only from
    :class:`RhimeModelSpec`, never from a callback's object identity.
    """
    options = _legacy_additive_sigma_options(likelihood_builder, likelihood_kwargs)
    if options is None:
        return (
            {},
            likelihood_builder,
            dict(likelihood_kwargs) if likelihood_kwargs is not None else None,
        )

    model_options: dict[str, Any] = {
        "mismatch_model": "additive_sigma",
        "use_minimum_error_floor": True,
    }
    for name in _LEGACY_ADDITIVE_OPTION_NAMES:
        if name in options:
            model_options[name] = options[name]
    return model_options, None, None


__all__ = ["RhimeLikelihoodBuilder"]
