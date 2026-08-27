"""Private validation seam for caller-supplied RHIME likelihoods."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, Protocol

import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.observation_error import AggregationError


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


def validate_likelihood_kwargs(
    likelihood_builder: object | None,
    likelihood_kwargs: object | None,
) -> dict[str, Any] | None:
    """Copy and validate options owned by a custom likelihood."""
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


__all__ = ["RhimeLikelihoodBuilder"]
