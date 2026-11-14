"""Exact posterior draws for PyMC graphs that are linear and Gaussian.

A graph qualifies when every free variable has a Normal prior with fixed
parameters and no transform, and the likelihood mean is affine in those
variables. The affine map is read from the compiled graph itself, by
evaluating it at zero and at each unit vector, and is checked at an
independent random point, so the estimator solves exactly the model that NUTS
would sample. :func:`~openghg_inversions.inference.gls.paige_gls` then gives
the exact posterior, and independent draws pass through the unchanged graph
to recover every deterministic, as the compact NUTS path does.
"""

from __future__ import annotations

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pymc as pm
from pymc.backends.arviz import DataTreeConverter, coords_and_dims_for_inferencedata, dict_to_dataset
import pytensor
import pytensor.tensor as pt
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.variable import TensorVariable
import xarray as xr

from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords

from .gls import GaussianPosterior, paige_gls
from .sampling import RhimeSampler, _restore_deterministics


@dataclass(frozen=True)
class NormalVariables:
    """Free variables of a graph with independent fixed Normal priors.

    Attributes:
        names: Free variable names in ``model.free_RVs`` order.
        shapes: Shape of each variable.
        prior_mean: Concatenated flattened prior means.
        prior_sd: Concatenated flattened prior standard deviations.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    prior_mean: NDArray[np.float64]
    prior_sd: NDArray[np.float64]

    def split(self, flat: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Split ``(..., size)`` concatenated values into named arrays of their shapes."""
        result = {}
        start = 0
        for name, shape in zip(self.names, self.shapes):
            stop = start + int(np.prod(shape, dtype=int))
            result[name] = flat[..., start:stop].reshape(flat.shape[:-1] + shape)
            start = stop
        return result


@dataclass(frozen=True)
class AffineMap:
    """``output = offset + matrix @ theta`` for one flattened graph output.

    Attributes:
        offset: Output at zero, shape ``(k,)``.
        matrix: Columns for each concatenated free-variable element, ``(k, q)``.
    """

    offset: NDArray[np.float64]
    matrix: NDArray[np.float64]


def normal_variables(model: pm.Model) -> NormalVariables:
    """Return the fixed Normal priors of every free variable.

    Args:
        model: Graph to inspect.

    Returns:
        Names, shapes and flattened prior moments.

    Raises:
        ValueError: Naming each free variable whose prior is not a Normal
            distribution with parameters free of other random variables.
    """
    free = list(model.free_RVs)
    refused = []
    parameters: list[TensorVariable] = []
    shapes = {name: tuple(int(size) for size in shape) for name, shape in model.eval_rv_shapes().items()}
    for rv in free:
        op = rv.owner.op
        if not isinstance(op, NormalRV) or model.rvs_to_transforms.get(rv) is not None:
            refused.append(f"{rv.name!r} ({getattr(op, 'name', None) or type(op).__name__})")
            continue
        mu, sigma = op.dist_params(rv.owner)
        if any(variable in free for variable in ancestors([mu, sigma])):
            refused.append(f"{rv.name!r} (hierarchical Normal)")
            continue
        shape = shapes[rv.name]
        parameters.extend([pt.broadcast_to(mu, shape), pt.broadcast_to(sigma, shape)])
    if refused:
        raise ValueError(
            "The Gaussian estimator requires an independent Normal prior with fixed parameters for every "
            f"sampled variable; these are not: {', '.join(refused)}."
        )
    values = pytensor.function([], parameters)() if parameters else []
    means = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values[0::2]]
    sds = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values[1::2]]
    sd = np.concatenate(sds) if sds else np.zeros(0)
    if not (np.isfinite(sd).all() and (sd > 0).all()):
        raise ValueError("Normal prior standard deviations must be finite and positive.")
    return NormalVariables(
        names=tuple(str(rv.name) for rv in free),
        shapes=tuple(shapes[str(rv.name)] for rv in free),
        prior_mean=np.concatenate(means) if means else np.zeros(0),
        prior_sd=sd,
    )


def affine_maps(
    model: pm.Model,
    outputs: Sequence[TensorVariable],
    variables: NormalVariables,
    *,
    rng: np.random.Generator,
    rtol: float = 1e-9,
) -> list[AffineMap]:
    """Read the affine maps from free variables to flattened graph outputs.

    Args:
        model: Graph containing the outputs.
        outputs: Expressions in terms of the model's random variables.
        variables: Free variables from :func:`normal_variables`.
        rng: Generator for the independent affinity check point.
        rtol: Relative tolerance of the affinity check.

    Returns:
        One affine map per output.

    Raises:
        ValueError: If an output is not affine in the free variables.
    """
    values = [model.rvs_to_values[model[name]] for name in variables.names]
    expressions = model.replace_rvs_by_values([pt.as_tensor_variable(output) for output in outputs])
    function = pytensor.function(values, [expression.flatten() for expression in expressions],
                                 on_unused_input="ignore")

    def evaluate(theta: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        split = variables.split(theta)
        return [np.asarray(value, dtype=np.float64) for value in function(*(split[name] for name in variables.names))]

    size = variables.prior_mean.size
    offsets = evaluate(np.zeros(size))
    matrices = [np.empty((offset.size, size)) for offset in offsets]
    for column in range(size):
        unit = np.zeros(size)
        unit[column] = 1.0
        for matrix, value, offset in zip(matrices, evaluate(unit), offsets):
            matrix[:, column] = value - offset
    theta = variables.prior_mean + variables.prior_sd * rng.standard_normal(size)
    for output, matrix, offset, value in zip(outputs, matrices, offsets, evaluate(theta)):
        predicted = offset + matrix @ theta
        scale = np.abs(offset) + np.abs(matrix) @ np.abs(theta) + np.finfo(np.float64).tiny
        deviation = float(np.max(np.abs(value - predicted) / scale)) if value.size else 0.0
        if deviation > rtol:
            raise ValueError(
                f"Graph output {getattr(output, 'name', None)!r} is not affine in its Normal variables "
                f"(relative deviation {deviation:.2e}); the Gaussian estimator requires a linear forward model."
            )
    return [AffineMap(offset, matrix) for offset, matrix in zip(offsets, matrices)]


def solve_linear_gaussian_model(
    model: pm.Model,
    *,
    likelihood_mean: TensorVariable,
    observations: NDArray[np.float64],
    noise_factor: NDArray[np.float64],
    rng: np.random.Generator,
    rank_rtol: float | None = None,
) -> tuple[NormalVariables, GaussianPosterior]:
    """Solve the exact posterior of a linear-Gaussian graph by Paige's method.

    Args:
        model: Graph whose free variables have fixed Normal priors, with one
            observed variable and no potentials.
        likelihood_mean: Mean expression of the single observed variable,
            aligned with ``observations``.
        observations: Observed values used by the likelihood.
        noise_factor: Factor of the fixed likelihood covariance.
        rng: Generator for the affinity check.
        rank_rtol: Optional rank tolerance forwarded to :func:`paige_gls`.

    Returns:
        The free variables and their exact joint posterior.

    Raises:
        ValueError: If the graph has potentials or other than one observed
            variable, or fails :func:`normal_variables` or :func:`affine_maps`.
    """
    if model.potentials:
        raise ValueError(
            "The Gaussian estimator cannot include pm.Potential terms: "
            f"{[str(potential.name) for potential in model.potentials]!r}."
        )
    if len(model.observed_RVs) != 1:
        raise ValueError(
            "The Gaussian estimator requires exactly one observed variable, whose mean it reads; found "
            f"{[str(rv.name) for rv in model.observed_RVs]!r}."
        )
    variables = normal_variables(model)
    (mean_map,) = affine_maps(model, [likelihood_mean], variables, rng=rng)
    posterior = paige_gls(
        np.asarray(observations, dtype=np.float64) - mean_map.offset,
        mean_map.matrix,
        noise_factor,
        variables.prior_mean,
        np.diag(variables.prior_sd),
        rank_rtol=rank_rtol,
    )
    return variables, posterior


def exact_gaussian_trace(
    model: pm.Model,
    variables: NormalVariables,
    posterior: GaussianPosterior,
    *,
    sampler: RhimeSampler,
    rng: np.random.Generator,
    variable_roles: Mapping[str, str] | None = None,
) -> xr.DataTree:
    """Draw exact posterior samples and complete the trace through the graph.

    The posterior group has one chain of ``sampler.draws`` independent draws
    of every free variable, with deterministics evaluated by the unchanged
    graph. Prior and predictive groups follow the sampler's settings and use
    PyMC's forward sampling, which is exact for these priors.

    Args:
        model: Linear-Gaussian graph.
        variables: Its free variables.
        posterior: Their exact joint posterior.
        sampler: Supplies draw counts and predictive choices; its NUTS
            settings are unused.
        rng: Parent generator; independent children seed the posterior,
            prior and posterior-predictive streams.
        variable_roles: Roles resolving posterior-predictive names.

    Returns:
        Trace with posterior, observed and constant data, and requested
        predictive groups, with registered coordinates restored.
    """
    posterior_rng, prior_rng, predictive_rng = rng.spawn(3)
    draws = posterior.draw(sampler.draws, posterior_rng)
    arrays = {name: value[None] for name, value in variables.split(draws).items()}
    coords, dims = coords_and_dims_for_inferencedata(model)
    samples = dict_to_dataset(arrays, inference_library=pm, coords=coords, dims=dims)
    converter = DataTreeConverter(model=model, prior=arrays)
    groups: dict[str, Any] = {"posterior": _restore_deterministics(samples, model)}
    for name, group in (("observed_data", converter.observed_data_to_xarray()),
                        ("constant_data", converter.constant_data_to_xarray())):
        if group is not None:
            groups[name] = group
    trace = xr.DataTree.from_dict(groups)
    predictive = sampler._posterior_predictive_options(model, variable_roles)
    predictive.setdefault("random_seed", predictive_rng)
    trace = sampler._extend_predictive(
        trace, model=model, prior_random_seed=prior_rng, posterior_predictive_kwargs=predictive,
    )
    registry = get_coord_registry(model)
    if registry is not None:
        trace = restore_inferencedata_coords(trace, registry)
    return trace


__all__ = [
    "AffineMap", "NormalVariables", "affine_maps", "exact_gaussian_trace", "normal_variables",
    "solve_linear_gaussian_model",
]
