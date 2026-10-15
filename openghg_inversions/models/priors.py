"""Create reusable scalar or array-valued PyMC prior variables.

``parse_prior`` registers continuous distributions on the active model;
``lognormal_mu_sigma`` converts requested moments, including arrays. Optional
lognormal reparameterization exposes ``<name>_latent`` and keeps ``<name>`` as
the user-facing deterministic variable.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, TypeAlias

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import compute
from pymc.distributions import continuous
from pytensor.graph.basic import Variable
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.array_ops import to_dense
from openghg_inversions.models.coords import add_model_data, get_coord_registry

PriorArgs: TypeAlias = dict[str, Any]

_POSITIVE_PRIOR_FAMILIES = {
    "exponential",
    "gamma",
    "halfnormal",
    "halfstudentt",
    "lognormal",
    "uniform",
}


def lognormal_mu_sigma(
    mean: float | np.ndarray,
    stdev: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert lognormal mean and stdev into PyMC's ``mu`` and ``sigma``.

    Args:
        mean: Requested scalar or array-valued mean of the lognormal
            distribution.
        stdev: Requested scalar or array-valued standard deviation of the
            lognormal distribution.

    Returns:
        A ``(mu, sigma)`` tuple suitable for ``pm.Lognormal``.
    """
    mean_array = np.asarray(mean)
    stdev_array = np.asarray(stdev)
    var = np.log(1 + (stdev_array / mean_array) ** 2)
    mu = np.log(mean_array) - 0.5 * var
    sigma = np.sqrt(var)
    if mu.ndim == 0 and sigma.ndim == 0:
        return float(mu), float(sigma)
    return mu, sigma


def _update_log_normal_prior(prior_params: PriorArgs) -> None:
    """Rewrite requested lognormal moments to PyMC parameters in place.

    Args:
        prior_params: Mutable mapping. When ``stdev`` is present, it and
            optional ``mean`` are replaced by ``mu`` and ``sigma``.
    """
    if "stdev" not in prior_params:
        return

    stdev = prior_params["stdev"]
    mean = prior_params.get("mean", 1.0)
    variance_log = pt.log1p(pt.square(pt.as_tensor_variable(stdev) / mean))
    mu, sigma = pt.log(mean) - 0.5 * variance_log, pt.sqrt(variance_log)
    prior_params["mu"] = mu
    prior_params["sigma"] = sigma
    del prior_params["stdev"]
    if "mean" in prior_params:
        del prior_params["mean"]


def positive_prior_args(prior_params: Mapping[str, Any]) -> PriorArgs:
    """Return prior arguments whose distribution has non-negative support."""
    params = dict(prior_params)
    family = str(params.get("pdf", "")).casefold().replace("-", "")
    if family not in _POSITIVE_PRIOR_FAMILIES:
        raise ValueError(
            "A positive prior must use HalfNormal, HalfStudentT, Exponential, "
            "Gamma, LogNormal, or Uniform."
        )
    params["pdf"] = family
    if family == "uniform":
        try:
            lower = np.asarray(params.get("lower", 0.0), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("A positive Uniform prior requires a numeric lower bound.") from error
        if not np.isfinite(lower).all() or (lower < 0.0).any():
            raise ValueError("A positive Uniform prior requires a finite lower bound >= 0.")
    return params


def parse_prior(
    name: str, prior_params: PriorArgs, *,
    active_indices: np.ndarray | None = None,
    record_name: str | None = None,
    **kwargs,
) -> TensorVariable:
    """Create a continuous PyMC prior from a prior-parameter dictionary.

    Args:
        name: Name of the user-facing PyMC variable to create.
        prior_params: Prior specification including ``pdf`` and any distribution
            parameters accepted by the chosen PyMC distribution.
        active_indices: Optional graph selection from canonical registered arrays.
        record_name: Scientific variable name owning an active-only prior record.
        **kwargs: Additional keyword arguments forwarded to the created PyMC
            variable, such as ``dims``.

    Returns:
        The created PyMC random variable or deterministic transform.

    Raises:
        ValueError: If ``prior_params["pdf"]`` does not name a supported PyMC
            continuous distribution, or parameter labels, dimensions or fixed
            array values are incompatible.

    This helper must be called inside an active ``pm.Model`` context because it
    registers the created variable with the current model.

    Borrowed numerical arrays and their auxiliary coordinates are materialized
    together here. Labelled parameters align to registered scientific indexes
    and broadcast by dimension; positional parameters use trailing model axes.
    Arrays become named model data, while resolved numerical scalars stay inline
    in construction metadata. Symbolic parameters retain their dependencies.
    ``registered_model`` supplies scientific-coordinate restoration; ordinary
    ``pm.Model`` contexts retain their existing numerical construction support.
    RHIME-resolved defaults are captured; implicit PyMC defaults remain implicit.
    """
    pdf_dict = {cd.lower(): cd for cd in continuous.__all__}

    params = prior_params.copy()
    pdf = str(params.pop("pdf")).lower()

    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        ) from exc

    # Interpolated retains its existing eager construction; its persistence is
    # outside the registered-prior contract.
    if pdf == "interpolated":
        params.pop("reparameterise", None)
        return dist(name, **params, **kwargs)

    if pdf == "lognormal" and "stdev" in params:
        params.setdefault("mean", 1.0)
    model = pm.modelcontext(None)
    dims = kwargs.get("dims", ())
    dims = (dims,) if isinstance(dims, str) else tuple(dims or ())
    registry = get_coord_registry(model)
    for key, value in params.items():
        if isinstance(value, Variable) or key == "reparameterise":
            continue
        if (isinstance(value, xr.DataArray) and dims and active_indices is None
                and not set(value.dims).issubset(dims)):
            raise ValueError(f"Prior parameter {key!r} dimensions must be a subset of {dims!r}.")
        if not isinstance(value, xr.DataArray):
            value = value if hasattr(value, "ndim") else np.asarray(value)
            if value.ndim:
                if value.ndim > len(dims) and dims:
                    raise ValueError(f"Prior parameter {key!r} needs explicit labelled dimensions.")
                parameter_axes = dims[-value.ndim:] if dims else tuple(
                    f"{record_name or name}_prior_{key}_dim_{axis}" for axis in range(value.ndim)
                )
                parameter_dims = []
                coords = {}
                for dim, size in zip(parameter_axes, value.shape, strict=True):
                    canonical = registry.original_coords.get(dim) if registry is not None else model.coords.get(dim)
                    if canonical is not None and len(canonical) != size:
                        if size != 1:
                            raise ValueError(f"Prior parameter {key!r} has incompatible {dim!r} length.")
                        dim = f"{record_name or name}_prior_{key}_{dim}"
                    elif canonical is not None:
                        coords[dim] = canonical if registry is not None else list(canonical)
                    parameter_dims.append(dim)
                value = xr.DataArray(value, dims=parameter_dims, coords=coords)
        if isinstance(value, xr.DataArray):
            for dim in value.dims:
                canonical = registry.original_coords.get(dim) if registry is not None else model.coords.get(dim)
                if canonical is not None and value.sizes[dim] != len(canonical):
                    raise ValueError(f"Prior parameter {key!r} has incompatible {dim!r} length.")
                if registry is not None and canonical is not None and dim in value.indexes:
                    index = value.indexes[dim]
                    if not index.is_unique or not index.isin(canonical).all():
                        raise ValueError(f"Prior parameter {key!r} labels must match {dim!r} exactly.")
                    value = value.sel({dim: canonical})
            value = to_dense(value)
        params[key] = value

    # Compute whole labelled arrays together, including lazy auxiliaries.
    values = compute(*params.values())
    params = dict(zip(params, values, strict=True))
    record: PriorArgs = {"pdf": pdf}
    for key, value in params.items():
        if key == "reparameterise":
            record[key] = bool(value)
        elif isinstance(value, Variable):
            record[key] = {"variable": value.name} if value.name else {"symbolic": True}
            if active_indices is not None and value.ndim:
                params[key] = value[active_indices]
        elif np.ndim(value) == 0:
            value = value.item() if hasattr(value, "item") else value
            record[key] = params[key] = value
        else:
            if not np.isfinite(value.to_numpy()).all():
                raise ValueError(f"Prior parameter {key!r} must contain only finite values.")
            data_name = f"{record_name or name}_prior_{key}"
            tensor = (add_model_data(value, data_name) if registry is not None else
                      pm.Data(data_name, value.values, dims=value.dims,
                              coords={dim: value.coords[dim].values for dim in value.dims if dim in value.coords}))
            record[key] = {"data": tensor.name}
            tensor = pt.specify_broadcastable(tensor, *(axis for axis, size in enumerate(value.shape) if size == 1))
            if active_indices is not None:
                tensor = tensor[active_indices]
            elif dims and set(value.dims).issubset(dims):
                tensor = tensor.dimshuffle(*(value.dims.index(dim) if dim in value.dims else "x" for dim in dims))
            params[key] = tensor

    if pdf == "lognormal":
        _update_log_normal_prior(params)

        if params.get("reparameterise", False):
            latent = pm.Normal(f"{name}_latent", 0, 1, **kwargs)
            variable = pm.Deterministic(name, pt.exp(params["mu"] + params["sigma"] * latent), **kwargs)
            records = getattr(model, "_openghg_prior_records", {})
            records[record_name or name] = record
            model._openghg_prior_records = records
            return variable

    params.pop("reparameterise", None)

    variable = dist(name, **params, **kwargs)
    records = getattr(model, "_openghg_prior_records", {})
    records[record_name or name] = record
    model._openghg_prior_records = records
    return variable


def prior_records(model: pm.Model) -> dict[str, PriorArgs]:
    """Return construction-owned, JSON-compatible descriptions of used priors."""
    return getattr(model, "_openghg_prior_records", {})


def trace_prior_records(trace: xr.DataTree) -> dict[str, PriorArgs]:
    """Read constructed prior metadata and validate its required array references."""
    try:
        records = json.loads(trace.attrs.get("rhime_priors", "{}"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Trace `rhime_priors` must contain a JSON mapping.") from exc
    if not isinstance(records, dict) or any(not isinstance(prior, dict) for prior in records.values()):
        raise ValueError("Trace `rhime_priors` must map prior names to parameter mappings.")
    for prior in records.values():
        for value in prior.values():
            if isinstance(value, dict) and "data" in value:
                if "constant_data" not in trace or value["data"] not in trace["constant_data"]:
                    raise ValueError(f"Missing registered prior data {value['data']!r}.")
    return records
